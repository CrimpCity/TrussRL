"""Batch reward adapter tests: serial identity, pooling, and fault recovery."""

import dataclasses
import json
import multiprocessing.process
import multiprocessing.queues
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import trussRL.cli.reward_bench as reward_bench
from trussRL.baselines.random_design import sample_design
from trussRL.calibration.artifacts import instance_from_payload, instance_payload
from trussRL.cli.reward_bench import (
    CRASH_SENTINEL,
    HANG_SENTINEL,
    BatchBenchmark,
    InjectionResult,
    TimingSummary,
    crash_on_sentinel_task,
    hang_on_sentinel_task,
)
from trussRL.generator import generate_instances
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.schema import TrussDesign
from trussRL.training.rewards import (
    INSTANCE_COLUMN,
    TrussRewardFunction,
    VerifierPool,
    WorkerSlot,
    completion_text,
    decode_instance_column,
    encode_instance_column,
    score_batch,
)
from trussRL.verifier import score

runner = CliRunner()

KNOWN_INSTANCE = TrussInstance(
    span_ft=96.0,
    load_cases=(LoadCase(w_kip_per_ft=-2.0, level="bottom"),),
)

KNOWN_DESIGN = TrussDesign(
    truss_type="warren",
    n_bays=8,
    depth_ft=8.0,
    top_chord="HSS6X6X3/8",
    bottom_chord="HSS6X6X3/8",
    diagonals="HSS4X4X1/4",
)

KNOWN_COMPLETION = f"```json\n{KNOWN_DESIGN.model_dump_json()}\n```"
GENEROUS_COST_REF_USD = 1_000_000.0
CALIBRATED_INSTANCE = dataclasses.replace(
    KNOWN_INSTANCE, cost_ref_usd=GENEROUS_COST_REF_USD
)
MULTI_CASE_INSTANCE = TrussInstance(
    span_ft=120.0,
    load_cases=(
        LoadCase(w_kip_per_ft=-3.0, level="bottom"),
        LoadCase(w_kip_per_ft=1.2, level="top", longitudinal_kip=15.0),
    ),
    depth_limit_ft=20.0,
    defl_denom=240,
    cost_ref_usd=50_000.0,
)
GENERATED_INSTANCES = generate_instances(7, 2)
MALFORMED_COMPLETION = "the model rambled and emitted no design"
MALFORMED_INDICES = (1, 4)


def build_mixed_batch() -> tuple[tuple[TrussInstance, ...], tuple[str, ...]]:
    """Build an 8-completion batch spanning calibrated and failure paths.

    Assumptions:
        1. Positions in MALFORMED_INDICES carry malformed text so order
           preservation can be asserted against known rung-0 landings.

    Args: None

    Returns:
        tuple: (instances, completions) of length 8, aligned per
            completion, including the known rung-3 pair and the
            calibrated-cost instance
    """
    rng = random.Random(11)
    roster = (KNOWN_INSTANCE, CALIBRATED_INSTANCE, *GENERATED_INSTANCES)
    instances: list[TrussInstance] = []
    completions: list[str] = []
    position = 0
    for instance in roster:
        for completion_index in range(2):
            instances.append(instance)
            if position == 0:
                completions.append(KNOWN_COMPLETION)
            elif position in MALFORMED_INDICES:
                completions.append(MALFORMED_COMPLETION)
            else:
                design = sample_design(rng, instance)
                completions.append(f"```json\n{design.model_dump_json()}\n```")
            position += 1
    return tuple(instances), tuple(completions)


def build_sentinel_batch(
    sentinel: str,
) -> tuple[tuple[TrussInstance, ...], tuple[str, ...], int]:
    """Build a small batch with one worker-killing sentinel completion.

    Args:
        sentinel: the sentinel completion the injected task reacts to

    Returns:
        tuple: (instances, completions, sentinel_index) with four
            completions on the known instance
    """
    instances = (KNOWN_INSTANCE,) * 4
    completions = (
        KNOWN_COMPLETION,
        MALFORMED_COMPLETION,
        sentinel,
        KNOWN_COMPLETION,
    )
    return instances, completions, 2


def process_is_dead(process: multiprocessing.process.BaseProcess) -> bool:
    """Report whether a worker process is dead or already reaped.

    Assumptions:
        1. A closed process handle proves death: the pool only closes a
           handle after observing the process is no longer alive.

    Args:
        process: the worker process handle to check

    Returns:
        bool: True when the process is not alive or its handle is closed
    """
    try:
        return not process.is_alive()
    except ValueError:
        return True


def raise_spawn_failure() -> WorkerSlot:
    """Stand-in for VerifierPool.spawn_worker that always fails.

    Args: None

    Returns:
        WorkerSlot: never returns

    Raises:
        RuntimeError: always.
    """
    raise RuntimeError("injected spawn failure")


def test_score_batch_matches_serial_score_exactly() -> None:
    instances, completions = build_mixed_batch()
    batch = score_batch(instances, completions)
    for breakdown, instance, completion in zip(
        batch, instances, completions, strict=True
    ):
        assert breakdown == score(instance, completion)


def test_score_batch_preserves_order() -> None:
    instances, completions = build_mixed_batch()
    batch = score_batch(instances, completions)
    assert batch[0].rung == 3
    for index in MALFORMED_INDICES:
        assert batch[index].rung == 0
    for index, breakdown in enumerate(batch):
        if index not in MALFORMED_INDICES:
            assert breakdown.rung > 0


def test_score_batch_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        score_batch((KNOWN_INSTANCE,), (KNOWN_COMPLETION, KNOWN_COMPLETION))


def test_instance_column_round_trip() -> None:
    for instance in (
        KNOWN_INSTANCE,
        CALIBRATED_INSTANCE,
        MULTI_CASE_INSTANCE,
        *GENERATED_INSTANCES,
    ):
        assert decode_instance_column(encode_instance_column(instance)) == instance


def test_instance_from_payload_rejects_bad_keys() -> None:
    payload = instance_payload(MULTI_CASE_INSTANCE)

    extra = dict(payload)
    extra["surprise"] = 1.0
    with pytest.raises(ValueError, match="extra=\\['surprise'\\]"):
        instance_from_payload(extra)

    missing = {key: value for key, value in payload.items() if key != "span_ft"}
    with pytest.raises(ValueError, match="missing=\\['span_ft'\\]"):
        instance_from_payload(missing)

    case_extra = instance_payload(MULTI_CASE_INSTANCE)
    case_extra_load_cases = cast(list[dict[str, object]], case_extra["load_cases"])
    case_extra["load_cases"] = [
        {**case, "surprise": 1.0} for case in case_extra_load_cases
    ]
    with pytest.raises(ValueError, match="load-case keys"):
        instance_from_payload(case_extra)

    case_missing = instance_payload(MULTI_CASE_INSTANCE)
    case_missing_load_cases = cast(list[dict[str, object]], case_missing["load_cases"])
    case_missing["load_cases"] = [
        {key: value for key, value in case.items() if key != "level"}
        for case in case_missing_load_cases
    ]
    with pytest.raises(ValueError, match="load-case keys"):
        instance_from_payload(case_missing)


def test_cost_term_changes_score_of_feasible_design() -> None:
    uncalibrated = score(KNOWN_INSTANCE, KNOWN_COMPLETION)
    calibrated = score(CALIBRATED_INSTANCE, KNOWN_COMPLETION)
    assert uncalibrated.rung == 3
    assert calibrated.rung == 3
    assert calibrated.feasible is not None and calibrated.feasible > 0.0
    assert calibrated.score > uncalibrated.score


def test_reward_function_scores_from_dataset_columns() -> None:
    instances, completions = build_mixed_batch()
    serial = score_batch(instances, completions)
    reward_function = TrussRewardFunction()
    scores = reward_function(
        prompts=[""] * len(completions),
        completions=list(completions),
        **{INSTANCE_COLUMN: [encode_instance_column(i) for i in instances]},
        irrelevant_column=list(range(len(completions))),
    )
    assert scores == [breakdown.score for breakdown in serial]


def test_reward_function_handles_conversational_completions() -> None:
    conversational = [
        [{"role": "assistant", "content": KNOWN_COMPLETION}],
    ]
    reward_function = TrussRewardFunction()
    scores = reward_function(
        prompts=[""],
        completions=conversational,
        **{INSTANCE_COLUMN: [encode_instance_column(KNOWN_INSTANCE)]},
    )
    assert scores == [score(KNOWN_INSTANCE, KNOWN_COMPLETION).score]
    assert completion_text(conversational[0]) == KNOWN_COMPLETION
    with pytest.raises(ValueError, match="unsupported completion shape"):
        completion_text(1.5)


def test_reward_function_missing_instance_column_raises() -> None:
    reward_function = TrussRewardFunction()
    with pytest.raises(ValueError, match=INSTANCE_COLUMN):
        reward_function(prompts=[""], completions=[KNOWN_COMPLETION])


def test_reward_function_name_is_stable() -> None:
    assert TrussRewardFunction().__name__ == "truss_reward"


def test_rewards_import_pulls_no_training_frameworks() -> None:
    import_guard = """
import importlib.abc
import sys

blocked_roots = {
    "accelerate",
    "bitsandbytes",
    "datasets",
    "deepspeed",
    "peft",
    "torch",
    "transformers",
    "trl",
    "unsloth",
    "vllm",
    "wandb",
}

class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in blocked_roots:
            raise AssertionError(f"forbidden rewards dependency imported: {fullname}")
        return None

sys.meta_path.insert(0, RejectTrainingImports())
import trussRL.training.rewards
"""
    completed = subprocess.run(
        [sys.executable, "-c", import_guard],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_pool_matches_serial_exactly() -> None:
    instances, completions = build_mixed_batch()
    serial = score_batch(instances, completions)
    with VerifierPool(max_workers=2) as pool:
        pooled = pool.score_batch(instances, completions)
        assert pool.restart_count == 0
    assert pooled == serial


def test_pool_survives_worker_crash() -> None:
    instances, completions, sentinel_index = build_sentinel_batch(CRASH_SENTINEL)
    serial = score_batch(instances, completions)
    with VerifierPool(max_workers=2, task=crash_on_sentinel_task) as pool:
        results = pool.score_batch(instances, completions)
        assert results[sentinel_index].rung == 2
        assert "worker crashed or timed out" in results[sentinel_index].reason
        for index, breakdown in enumerate(results):
            if index != sentinel_index:
                assert breakdown == serial[index]
        assert pool.restart_count == 2
        clean_completions = tuple(
            KNOWN_COMPLETION if index == sentinel_index else completion
            for index, completion in enumerate(completions)
        )
        clean = pool.score_batch(instances, clean_completions)
    assert clean == score_batch(instances, clean_completions)


def test_pool_survives_worker_hang() -> None:
    instances, completions, sentinel_index = build_sentinel_batch(HANG_SENTINEL)
    serial = score_batch(instances, completions)
    with VerifierPool(
        max_workers=2, task=hang_on_sentinel_task, task_timeout_s=0.5
    ) as pool:
        pool.ensure_workers()
        original_processes = [slot.process for slot in pool.slots]
        results = pool.score_batch(instances, completions)
        assert results[sentinel_index].rung == 2
        for index, breakdown in enumerate(results):
            if index != sentinel_index:
                assert breakdown == serial[index]
        assert pool.restart_count == 2
        surviving = {id(slot.process) for slot in pool.slots}
        replaced = [
            process for process in original_processes if id(process) not in surviving
        ]
        assert replaced
        assert all(process_is_dead(process) for process in replaced)
        clean_completions = tuple(
            KNOWN_COMPLETION if index == sentinel_index else completion
            for index, completion in enumerate(completions)
        )
        clean = pool.score_batch(instances, clean_completions)
    assert clean == score_batch(instances, clean_completions)


def test_replacement_spawn_failure_scores_suspect_rung2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances, completions, sentinel_index = build_sentinel_batch(CRASH_SENTINEL)
    serial = score_batch(instances, completions)
    with VerifierPool(max_workers=2, task=crash_on_sentinel_task) as pool:
        pool.ensure_workers()
        monkeypatch.setattr(pool, "spawn_worker", raise_spawn_failure)
        results = pool.score_batch(instances, completions)
        assert results[sentinel_index].rung == 2
        for index, breakdown in enumerate(results):
            if index != sentinel_index:
                assert breakdown == serial[index]
        assert pool.restart_count == 1
        assert len(pool.slots) == 1


CREATED_QUEUES: list[multiprocessing.queues.Queue[object]] = []


def recording_queue_factory() -> multiprocessing.queues.Queue[object]:
    """Create a real spawn queue while recording it for leak checks.

    Assumptions:
        1. Goes through the context class rather than the instance so it
           still works when the instance's Queue attribute is patched
           with this very function.

    Args: None

    Returns:
        Queue: a spawn-context queue, also appended to CREATED_QUEUES
    """
    context = multiprocessing.get_context("spawn")
    queue = type(context).Queue(context)
    CREATED_QUEUES.append(queue)
    return queue


def raise_process_construction_failure(*args: object, **kwargs: object) -> object:
    """Stand-in process factory that fails before any worker exists.

    Args:
        *args: ignored
        **kwargs: ignored

    Returns:
        object: never returns

    Raises:
        RuntimeError: always.
    """
    raise RuntimeError("injected process construction failure")


def test_initial_startup_failure_falls_back_serially_and_closes_queues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    CREATED_QUEUES.clear()
    instances, completions = build_mixed_batch()
    serial = score_batch(instances, completions)
    with VerifierPool(max_workers=2) as pool:
        monkeypatch.setattr(pool.context, "Queue", recording_queue_factory)
        monkeypatch.setattr(pool.context, "Process", raise_process_construction_failure)
        results = pool.score_batch(instances, completions)
        assert pool.fallback_count == 1
        assert pool.slots == []
    assert results == serial
    assert len(CREATED_QUEUES) == 2
    for queue in CREATED_QUEUES:
        with pytest.raises(ValueError, match="closed"):
            queue.put(None)


def test_pool_constructor_validation() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        VerifierPool(max_workers=0)
    with pytest.raises(ValueError, match="task_timeout_s"):
        VerifierPool(task_timeout_s=0.0)
    with pytest.raises(ValueError, match="task_timeout_s"):
        VerifierPool(task_timeout_s=float("inf"))
    with pytest.raises(ValueError, match="task_timeout_s"):
        VerifierPool(task_timeout_s=float("nan"))
    with pytest.raises(ValueError, match="retry_budget"):
        VerifierPool(retry_budget=-1)


def make_timing_summary(median_s: float, n_completions: int) -> TimingSummary:
    """Build a fixed timing summary for CLI fakes.

    Args:
        median_s: the median and p95 to report, in seconds
        n_completions: completions per batch for the per-completion rate

    Returns:
        TimingSummary: a two-repeat summary at the requested median
    """
    return reward_bench.TimingSummary(
        timings_s=(median_s, median_s),
        median_s=median_s,
        p95_s=median_s,
        per_completion_ms=median_s / n_completions * 1000.0,
    )


def make_batch_benchmark(
    n_completions: int,
    serial_median_s: float,
    pool_median_s: float,
    scores_equal: bool,
) -> BatchBenchmark:
    """Build a fixed batch benchmark for CLI fakes.

    Args:
        n_completions: completions per batch
        serial_median_s: serial median seconds to report
        pool_median_s: pool median seconds to report
        scores_equal: the equality verdict to report

    Returns:
        BatchBenchmark: the assembled benchmark record
    """
    return reward_bench.BatchBenchmark(
        n_completions=n_completions,
        completions_per_instance=8,
        serial=make_timing_summary(serial_median_s, n_completions),
        pool=make_timing_summary(pool_median_s, n_completions),
        pool_startup_s=1.0,
        scores_equal=scores_equal,
    )


def fake_benchmark_serial_wins(
    seed: int,
    n_completions: int,
    completions_per_instance: int,
    workers: int,
    repeats: int,
) -> BatchBenchmark:
    """Benchmark fake where serial is faster than the pool.

    Args:
        seed: ignored
        n_completions: completions per batch, echoed into the record
        completions_per_instance: ignored
        workers: ignored
        repeats: ignored

    Returns:
        BatchBenchmark: serial median 5 ms, pool median 50 ms, equal
    """
    return make_batch_benchmark(n_completions, 0.005, 0.050, True)


def fake_benchmark_pool_wins(
    seed: int,
    n_completions: int,
    completions_per_instance: int,
    workers: int,
    repeats: int,
) -> BatchBenchmark:
    """Benchmark fake where the pool beats serial and passes the gate.

    Args:
        seed: ignored
        n_completions: completions per batch, echoed into the record
        completions_per_instance: ignored
        workers: ignored
        repeats: ignored

    Returns:
        BatchBenchmark: serial median 50 ms, pool median 5 ms, equal
    """
    return make_batch_benchmark(n_completions, 0.050, 0.005, True)


def fake_benchmark_scores_diverge(
    seed: int,
    n_completions: int,
    completions_per_instance: int,
    workers: int,
    repeats: int,
) -> BatchBenchmark:
    """Benchmark fake where pooled scores diverge from serial.

    Args:
        seed: ignored
        n_completions: completions per batch, echoed into the record
        completions_per_instance: ignored
        workers: ignored
        repeats: ignored

    Returns:
        BatchBenchmark: scores_equal False
    """
    return make_batch_benchmark(n_completions, 0.005, 0.050, False)


def fake_injection_survived(kind: str, seed: int, workers: int) -> InjectionResult:
    """Failure-injection fake reporting survival.

    Args:
        kind: the failure kind, echoed into the record
        seed: ignored
        workers: ignored

    Returns:
        InjectionResult: survived True with two restarts
    """
    return reward_bench.InjectionResult(injected=kind, survived=True, restart_count=2)


def read_bench_artifact(tmp_path: Path) -> dict[str, Any]:
    """Read the single benchmark artifact written under a temp dir.

    Args:
        tmp_path: the --output-dir used in the invocation

    Returns:
        dict: the parsed artifact payload
    """
    artifacts = sorted((tmp_path / "reward_bench").glob("reward_bench_*.json"))
    assert len(artifacts) == 1
    return cast(dict[str, Any], json.loads(artifacts[0].read_text()))


def test_cli_serial_adoption_and_artifact_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reward_bench, "benchmark_batch", fake_benchmark_serial_wins)
    monkeypatch.setattr(reward_bench, "run_failure_injection", fake_injection_survived)
    result = runner.invoke(
        reward_bench.app, ["--seed", "0", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    payload = read_bench_artifact(tmp_path)
    assert set(payload) == {
        "stamp",
        "config",
        "batches",
        "worker_failure",
        "gate",
        "provisional_adopted_mode",
        "passed",
    }
    assert payload["gate"]["gate_basis"] == "estimated_step_time"
    assert payload["gate"]["provisional"] is True
    assert payload["provisional_adopted_mode"] == "serial"
    assert payload["passed"] is True
    assert set(payload["batches"]) == {"8", "24"}
    assert payload["batches"]["24"]["scores_equal"] is True
    assert payload["worker_failure"]["crash"]["survived"] is True
    assert payload["worker_failure"]["hang"]["survived"] is True
    assert payload["config"]["seed"] == 0


def test_cli_adopts_pool_when_it_wins_and_passes_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reward_bench, "benchmark_batch", fake_benchmark_pool_wins)
    monkeypatch.setattr(reward_bench, "run_failure_injection", fake_injection_survived)
    result = runner.invoke(
        reward_bench.app, ["--seed", "0", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    payload = read_bench_artifact(tmp_path)
    assert payload["provisional_adopted_mode"] == "pool"
    assert payload["passed"] is True


def test_cli_exits_1_when_scores_diverge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reward_bench, "benchmark_batch", fake_benchmark_scores_diverge)
    monkeypatch.setattr(reward_bench, "run_failure_injection", fake_injection_survived)
    result = runner.invoke(
        reward_bench.app, ["--seed", "0", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    payload = read_bench_artifact(tmp_path)
    assert payload["passed"] is False
    assert payload["provisional_adopted_mode"] == "serial"
