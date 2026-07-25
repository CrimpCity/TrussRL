"""The adapter between the pure verifier and the training framework.

The verifier stays framework-agnostic; this shim is the only place their
interfaces meet. It packages instances as a JSON dataset column, scores
whole generation batches — serially by default, or on an optional
persistent worker pool — and folds worker crashes and hangs into graded
penalties so a bad completion can never take down the trainer.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import multiprocessing
import multiprocessing.queues
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from queue import Empty
from types import TracebackType

from trussRL.calibration.artifacts import instance_from_payload, instance_payload
from trussRL.instance import TrussInstance
from trussRL.reward import RewardBreakdown, rung2
from trussRL.verifier import score

INSTANCE_COLUMN = "instance_json"
POOL_TASK_TIMEOUT_S = 30.0
DEFAULT_POOL_WORKERS = 2
DEFAULT_RETRY_BUDGET = 1
POLL_INTERVAL_S = 0.05
REAP_JOIN_TIMEOUT_S = 5.0
SHUTDOWN_JOIN_TIMEOUT_S = 5.0
WORKER_STARTUP_TIMEOUT_S = 60.0
WORKER_READY_MESSAGE = "ready"
WORKER_FAULT_REASON = "verifier worker crashed or timed out"

RequestItem = tuple[int, TrussInstance, str] | None
ResultItem = tuple[int, RewardBreakdown] | str


def score_batch(
    instances: Iterable[TrussInstance], completions: Iterable[str]
) -> tuple[RewardBreakdown, ...]:
    """Score a batch of completions serially, in order.

    The canonical serial path and the numerical reference every other
    scoring mode is measured against.

    Args:
        instances: one problem instance per completion
        completions: raw completion texts, aligned with instances

    Returns:
        tuple: one RewardBreakdown per completion, in input order

    Raises:
        ValueError: if instances and completions differ in length.
    """
    instance_tuple = tuple(instances)
    completion_tuple = tuple(completions)
    if len(instance_tuple) != len(completion_tuple):
        raise ValueError(
            f"instances and completions differ in length: "
            f"{len(instance_tuple)} != {len(completion_tuple)}"
        )
    return tuple(
        score(instance, completion)
        for instance, completion in zip(instance_tuple, completion_tuple, strict=True)
    )


def score_task(instance: TrussInstance, completion_text: str) -> RewardBreakdown:
    """Score one completion: the pool's default, spawn-picklable task target.

    Assumptions:
        1. Exists as a stable module-level seam so failure-injection
           harnesses can substitute an alternative task without touching
           production scoring.

    Args:
        instance: the trusted problem instance
        completion_text: the policy's raw completion text

    Returns:
        RewardBreakdown: the final reward with per-rung detail
    """
    return score(instance, completion_text)


def worker_loop(
    request_queue: multiprocessing.queues.Queue[RequestItem],
    result_queue: multiprocessing.queues.Queue[ResultItem],
    task: Callable[[TrussInstance, str], RewardBreakdown],
) -> None:
    """Run one pool worker: pull tasks, score them, push results.

    Assumptions:
        1. Runs as the spawned process's main thread, so the solver's
           SIGALRM wall stays live for interpreter-level hangs; the pool's
           per-task deadline is the backstop for native-code hangs the
           alarm cannot reach.
        2. The verifier import happens at worker startup, before the
           readiness message, so openseespy load time is never charged
           against a task's deadline.

    Args:
        request_queue: queue of (task_index, instance, completion) work
            items; a None item requests a clean exit
        result_queue: queue the readiness message and the
            (task_index, breakdown) results go to
        task: the scoring callable applied to each work item

    Returns:
        None
    """
    import trussRL.verifier  # noqa: F401

    result_queue.put(WORKER_READY_MESSAGE)
    while True:
        request = request_queue.get()
        if request is None:
            return
        task_index, instance, completion = request
        result_queue.put((task_index, task(instance, completion)))


def encode_instance_column(instance: TrussInstance) -> str:
    """Serialize an instance into its dataset-column JSON string.

    Args:
        instance: the problem instance to serialize

    Returns:
        str: canonical sorted-key JSON that decode_instance_column
            reconstructs numerically identically

    Raises:
        ValueError: if the instance contains NaN or infinite floats.
    """
    return json.dumps(instance_payload(instance), sort_keys=True, allow_nan=False)


def decode_instance_column(text: str) -> TrussInstance:
    """Rebuild an instance from its dataset-column JSON string.

    Args:
        text: the JSON string produced by encode_instance_column

    Returns:
        TrussInstance: the reconstructed instance

    Raises:
        ValueError: if the text is not a JSON object or its keys do not
            match the instance fields.
    """
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"instance column must decode to a JSON object, "
            f"got {type(payload).__name__}"
        )
    return instance_from_payload(payload)


def completion_text(completion: object) -> str:
    """Normalize one completion into raw text for the verifier.

    Plain strings pass through; the conversational form (a list of
    role/content messages) yields the last message's content.

    Args:
        completion: a raw completion string, or a sequence of message
            mappings ending with the assistant turn

    Returns:
        str: the completion text to score

    Raises:
        ValueError: if the completion is neither a string nor a message
            list whose last entry carries string content.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion:
        last = completion[-1]
        if isinstance(last, Mapping):
            content = last.get("content")
            if isinstance(content, str):
                return content
    raise ValueError(
        f"unsupported completion shape: {type(completion).__name__}; expected "
        "a string or a list of role/content messages"
    )


@dataclasses.dataclass(eq=False)
class WorkerSlot:
    """The pool's bookkeeping for one live worker process.

    Assumptions:
        1. Identity equality (eq=False) so list membership tracks the
           specific worker, never two workers with coincidentally equal
           state.

    Attributes:
        process: the spawned worker process handle
        request_queue: this worker's private work-item queue
        result_queue: this worker's private result queue
        current_task_index: batch index of the task in flight, or None
            when idle
        deadline: monotonic time the in-flight task must finish by, or
            None when idle
    """

    process: multiprocessing.process.BaseProcess
    request_queue: multiprocessing.queues.Queue[RequestItem]
    result_queue: multiprocessing.queues.Queue[ResultItem]
    current_task_index: int | None = None
    deadline: float | None = None


class VerifierPool:
    """A persistent pool of spawn workers scoring completions in parallel.

    Manages worker processes directly — each with its own queues and an
    explicit process handle — so a crashed or wedged worker can be
    terminated, reaped, and respawned without disturbing the rest of the
    batch. Pool faults never escape as exceptions: a task that repeatedly
    kills its worker lands on a graded penalty instead.
    """

    def __init__(
        self,
        max_workers: int = DEFAULT_POOL_WORKERS,
        task_timeout_s: float = POOL_TASK_TIMEOUT_S,
        retry_budget: int = DEFAULT_RETRY_BUDGET,
        task: Callable[[TrussInstance, str], RewardBreakdown] = score_task,
    ) -> None:
        """Configure the pool; workers start lazily on the first batch.

        Args:
            max_workers: number of worker processes; must be at least 1,
                since zero workers could never make progress on a batch
            task_timeout_s: per-task wall-clock deadline in seconds; must
                be finite and positive
            retry_budget: how many times a task whose worker died or timed
                out is retried on a fresh worker before it scores the
                graded worker-fault penalty; must be nonnegative
            task: the scoring callable each worker applies; must be
                spawn-picklable, i.e. module-level

        Returns:
            None

        Raises:
            ValueError: if max_workers, task_timeout_s, or retry_budget is
                out of range.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers!r}")
        if not (task_timeout_s > 0.0 and task_timeout_s != float("inf")):
            raise ValueError(
                f"task_timeout_s must be finite and positive, got {task_timeout_s!r}"
            )
        if retry_budget < 0:
            raise ValueError(f"retry_budget must be nonnegative, got {retry_budget!r}")
        self.max_workers = max_workers
        self.task_timeout_s = task_timeout_s
        self.retry_budget = retry_budget
        self.task = task
        self.restart_count = 0
        self.fallback_count = 0
        self.context = multiprocessing.get_context("spawn")
        self.slots: list[WorkerSlot] = []

    def spawn_worker(self) -> WorkerSlot:
        """Start one worker process and wait for its readiness message.

        Assumptions:
            1. Blocks until the worker has finished its imports, so a
               task deadline armed after this returns measures the task
               alone, never worker startup.
            2. Cleans up after every failure mode of its own: queues
               created before a spawn failure are closed here, and a
               worker that started but never signalled ready is reaped
               here, so callers never see a half-constructed worker.

        Args: None

        Returns:
            WorkerSlot: the started, ready worker's bookkeeping record

        Raises:
            OSError: if the process cannot be spawned.
            RuntimeError: if the worker dies or stays silent through the
                startup timeout.
        """
        created_queues: list[
            multiprocessing.queues.Queue[RequestItem]
            | multiprocessing.queues.Queue[ResultItem]
        ] = []
        try:
            request_queue: multiprocessing.queues.Queue[RequestItem] = (
                self.context.Queue()
            )
            created_queues.append(request_queue)
            result_queue: multiprocessing.queues.Queue[ResultItem] = (
                self.context.Queue()
            )
            created_queues.append(result_queue)
            process = self.context.Process(
                target=worker_loop,
                args=(request_queue, result_queue, self.task),
                daemon=True,
            )
            process.start()
        except BaseException:
            for created in created_queues:
                created.close()
                created.cancel_join_thread()
            raise
        slot = WorkerSlot(
            process=process, request_queue=request_queue, result_queue=result_queue
        )
        startup_deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_S
        while True:
            try:
                ready = slot.result_queue.get(timeout=POLL_INTERVAL_S)
            except Empty:
                if process.is_alive() and time.monotonic() <= startup_deadline:
                    continue
                self.reap_worker(slot)
                raise RuntimeError(
                    "verifier worker died or stayed silent during startup"
                ) from None
            if ready == WORKER_READY_MESSAGE:
                return slot
            self.reap_worker(slot)
            raise RuntimeError(
                f"verifier worker sent {ready!r} instead of the readiness message"
            )

    def ensure_workers(self) -> None:
        """Start workers until the pool holds max_workers of them.

        Assumptions:
            1. On partial failure the already-started workers are left in
               the pool for the caller to reap; this method does not clean
               up after itself so the caller can decide between fallback
               and propagation.

        Args: None

        Returns:
            None
        """
        while len(self.slots) < self.max_workers:
            self.slots.append(self.spawn_worker())

    def reap_worker(self, slot: WorkerSlot) -> None:
        """Tear down one worker with bounded escalation and close its queues.

        Assumptions:
            1. Each queue owns a feeder thread in the parent that can hang
               interpreter exit; close plus cancel_join_thread releases
               them even when the worker died mid-transfer.

        Args:
            slot: the worker record to tear down

        Returns:
            None
        """
        process = slot.process
        process.terminate()
        process.join(REAP_JOIN_TIMEOUT_S)
        if process.is_alive():
            process.kill()
        process.join(REAP_JOIN_TIMEOUT_S)
        if not process.is_alive():
            process.close()
        for queue in (slot.request_queue, slot.result_queue):
            queue.close()
            queue.cancel_join_thread()

    def dispatch(
        self, slot: WorkerSlot, task_index: int, instance: TrussInstance, text: str
    ) -> None:
        """Hand one task to an idle worker and arm its deadline.

        Args:
            slot: the idle worker record receiving the task
            task_index: the task's index within the batch
            instance: the task's problem instance
            text: the task's completion text

        Returns:
            None
        """
        slot.request_queue.put((task_index, instance, text))
        slot.current_task_index = task_index
        slot.deadline = time.monotonic() + self.task_timeout_s

    def recover_from_fault(
        self,
        slot: WorkerSlot,
        task_index: int,
        budgets: list[int],
        pending: collections.deque[int],
        results: dict[int, RewardBreakdown],
    ) -> None:
        """Reap a crashed or timed-out worker and route its task.

        The in-flight task is a crash suspect: it retries on a respawned
        worker while budget remains, and otherwise scores the graded
        worker-fault penalty — never a serial rerun, which would move a
        worker-contained native crash into the calling process. When the
        respawn itself fails, the suspect scores the penalty immediately
        and the dead slot leaves the pool.

        Args:
            slot: the faulted worker record
            task_index: batch index of the task that was in flight
            budgets: remaining retry budget per batch index, mutated
            pending: undispatched batch indexes, mutated
            results: completed breakdowns by batch index, mutated

        Returns:
            None
        """
        self.reap_worker(slot)
        self.restart_count += 1
        try:
            replacement = self.spawn_worker()
        except Exception:  # noqa: BLE001 - a failed respawn is a handled fault
            results[task_index] = rung2(WORKER_FAULT_REASON)
            self.slots.remove(slot)
            return
        self.slots[self.slots.index(slot)] = replacement
        if budgets[task_index] > 0:
            budgets[task_index] -= 1
            pending.appendleft(task_index)
        else:
            results[task_index] = rung2(WORKER_FAULT_REASON)

    def score_batch(
        self, instances: Iterable[TrussInstance], completions: Iterable[str]
    ) -> tuple[RewardBreakdown, ...]:
        """Score a batch on the worker pool, preserving order and scores.

        Assumptions:
            1. On a clean batch the results are numerically identical to
               serial score(); tasks are only ever routed away from that
               contract by worker faults, which score the graded
               worker-fault penalty.
            2. Initial pool startup failure falls back to scoring the
               whole batch serially — no task has been dispatched, so no
               crash suspects exist yet.

        Args:
            instances: one problem instance per completion
            completions: raw completion texts, aligned with instances

        Returns:
            tuple: one RewardBreakdown per completion, in input order

        Raises:
            ValueError: if instances and completions differ in length.
        """
        instance_tuple = tuple(instances)
        completion_tuple = tuple(completions)
        if len(instance_tuple) != len(completion_tuple):
            raise ValueError(
                f"instances and completions differ in length: "
                f"{len(instance_tuple)} != {len(completion_tuple)}"
            )
        if not instance_tuple:
            return ()
        if not self.slots:
            try:
                self.ensure_workers()
            except Exception:  # noqa: BLE001 - startup failure falls back to serial
                for slot in self.slots:
                    self.reap_worker(slot)
                self.slots = []
                self.fallback_count += 1
                return score_batch(instance_tuple, completion_tuple)
        n_tasks = len(instance_tuple)
        results: dict[int, RewardBreakdown] = {}
        budgets = [self.retry_budget] * n_tasks
        pending: collections.deque[int] = collections.deque(range(n_tasks))
        while len(results) < n_tasks:
            if not self.slots:
                while pending:
                    index = pending.popleft()
                    if budgets[index] < self.retry_budget:
                        results[index] = rung2(WORKER_FAULT_REASON)
                    else:
                        results[index] = score(
                            instance_tuple[index], completion_tuple[index]
                        )
                break
            for slot in list(self.slots):
                if not pending:
                    break
                if slot.current_task_index is not None:
                    continue
                if not slot.process.is_alive():
                    self.reap_worker(slot)
                    self.restart_count += 1
                    try:
                        replacement = self.spawn_worker()
                    except Exception:  # noqa: BLE001 - handled by slot removal
                        self.slots.remove(slot)
                        continue
                    self.slots[self.slots.index(slot)] = replacement
                    slot = replacement
                index = pending.popleft()
                self.dispatch(
                    slot, index, instance_tuple[index], completion_tuple[index]
                )
            for slot in list(self.slots):
                if len(results) >= n_tasks:
                    break
                task_index = slot.current_task_index
                if task_index is None:
                    continue
                try:
                    item = slot.result_queue.get(timeout=POLL_INTERVAL_S)
                except Empty:
                    timed_out = (
                        slot.deadline is not None and time.monotonic() > slot.deadline
                    )
                    if slot.process.is_alive() and not timed_out:
                        continue
                    self.recover_from_fault(slot, task_index, budgets, pending, results)
                    continue
                if isinstance(item, str):
                    continue
                received_index, breakdown = item
                results[received_index] = breakdown
                slot.current_task_index = None
                slot.deadline = None
        return tuple(results[index] for index in range(n_tasks))

    def close(self) -> None:
        """Shut the pool down, reaping every worker and closing every queue.

        Workers are asked to exit cleanly first; stragglers are reaped
        with the same bounded escalation used for fault recovery.

        Args: None

        Returns:
            None
        """
        for slot in self.slots:
            slot.request_queue.put(None)
        for slot in self.slots:
            slot.process.join(SHUTDOWN_JOIN_TIMEOUT_S)
        for slot in self.slots:
            self.reap_worker(slot)
        self.slots = []

    def __enter__(self) -> VerifierPool:
        """Enter the pool's context; workers still start lazily.

        Args: None

        Returns:
            VerifierPool: this pool
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the pool on context exit.

        Args:
            exc_type: exception type leaving the context, or None
            exc_value: exception leaving the context, or None
            traceback: traceback leaving the context, or None

        Returns:
            None
        """
        self.close()


class TrussRewardFunction:
    """The reward callable the GRPO trainer invokes per generation batch.

    Carries instance data through the trainer as the instance_json dataset
    column, so the trusted problem definition rides alongside each prompt
    instead of being re-derived from text. Pure Python — importable and
    testable without any RL framework installed.
    """

    __name__ = "truss_reward"

    def __init__(self, pool: VerifierPool | None = None) -> None:
        """Choose the scoring mode: serial by default, pooled when given.

        Args:
            pool: an optional worker pool to score batches on; None scores
                serially in the calling process

        Returns:
            None
        """
        self.pool = pool

    def __call__(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        **kwargs: object,
    ) -> list[float]:
        """Score one generation batch into per-completion rewards.

        Assumptions:
            1. The trainer forwards extra dataset columns as keyword
               arguments; only the instance column is consumed and every
               other column is ignored.

        Args:
            prompts: the batch's prompts, unused — the instance column is
                the trusted problem source
            completions: the batch's completions, as raw strings or
                conversational message lists
            **kwargs: forwarded dataset columns, which must include the
                instance_json column aligned with completions

        Returns:
            list: one final score per completion, in batch order

        Raises:
            ValueError: if the instance column is missing, misshapen, or
                misaligned with the completions.
        """
        if INSTANCE_COLUMN not in kwargs:
            raise ValueError(
                f"reward kwargs are missing the required dataset column "
                f"{INSTANCE_COLUMN!r}"
            )
        column = kwargs[INSTANCE_COLUMN]
        if not isinstance(column, Sequence) or isinstance(column, (str, bytes)):
            raise ValueError(
                f"dataset column {INSTANCE_COLUMN!r} must be a sequence of "
                "JSON strings, one per completion"
            )
        if len(column) != len(completions):
            raise ValueError(
                f"dataset column {INSTANCE_COLUMN!r} length {len(column)} does "
                f"not match completions length {len(completions)}"
            )
        instances: list[TrussInstance] = []
        for entry in column:
            if not isinstance(entry, str):
                raise ValueError(
                    f"dataset column {INSTANCE_COLUMN!r} entries must be "
                    f"strings, got {type(entry).__name__}"
                )
            instances.append(decode_instance_column(entry))
        texts = tuple(completion_text(completion) for completion in completions)
        if self.pool is not None:
            breakdowns = self.pool.score_batch(instances, texts)
        else:
            breakdowns = score_batch(instances, texts)
        return [breakdown.score for breakdown in breakdowns]
