"""Benchmark the batch reward adapter: pool vs. serial, faults included.

Times both scoring modes on training-shaped batches, checks that pooled
scores are numerically identical to serial, injects worker crashes and
hangs to prove recovery, and freezes it all into a stamped artifact. The
adoption verdict is provisional: it is judged against an estimated step
time and must be re-evaluated once a measured step time exists.
"""

import dataclasses
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from trussRL.baselines.random_design import sample_design
from trussRL.calibration.artifacts import (
    instance_from_payload,
    provenance_stamp,
    write_json,
)
from trussRL.generator import generate_instances
from trussRL.instance import TrussInstance
from trussRL.reward import RewardBreakdown
from trussRL.schema import TrussDesign
from trussRL.training.rewards import (
    POOL_TASK_TIMEOUT_S,
    VerifierPool,
    score_batch,
    score_task,
)

app = typer.Typer(add_completion=False)
console = Console()

BENCH_DIR = "reward_bench"
CRASH_SENTINEL = "!!crash-the-verifier-worker!!"
HANG_SENTINEL = "!!hang-the-verifier-worker!!"
HANG_SLEEP_S = 60.0
HANG_TASK_TIMEOUT_S = 0.5
GATE_STEP_FRACTION = 0.05
GATE_P95_MS = 250.0
DEFAULT_STEP_TIME_ESTIMATE_S = 10.0
COMPLETIONS_PER_INSTANCE = 8
MALFORMED_EVERY = 5
INJECTION_BATCH_SIZE = 8
SWEEP_BEST_PATH = Path("artifacts") / "sweep_best.json"


@dataclass(frozen=True)
class TimingSummary:
    """Timing statistics for one scoring mode over repeated batch runs.

    Attributes:
        timings_s: raw per-repeat wall times in seconds
        median_s: median batch time in seconds
        p95_s: 95th-percentile batch time in seconds
        per_completion_ms: median time per completion in milliseconds
    """

    timings_s: tuple[float, ...]
    median_s: float
    p95_s: float
    per_completion_ms: float


@dataclass(frozen=True)
class BatchBenchmark:
    """Serial-vs-pool comparison for one batch size.

    Attributes:
        n_completions: completions per batch
        completions_per_instance: completions sharing each instance
        serial: serial-mode timing statistics
        pool: pool-mode timing statistics, startup excluded
        pool_startup_s: one-time pool startup wall time in seconds
        scores_equal: whether pooled breakdowns exactly equal serial ones
    """

    n_completions: int
    completions_per_instance: int
    serial: TimingSummary
    pool: TimingSummary
    pool_startup_s: float
    scores_equal: bool


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of one worker failure injection run.

    Attributes:
        injected: the failure kind injected, "crash" or "hang"
        survived: whether non-sentinel scores stayed exactly serial, the
            sentinel landed on the graded worker-fault penalty, and a
            follow-up clean batch through the same pool matched serial
        restart_count: worker respawns the pool performed
    """

    injected: str
    survived: bool
    restart_count: int


@dataclass(frozen=True)
class GateVerdict:
    """One scoring mode's verdict against the step-time budget gate.

    Attributes:
        median_fraction_of_step: median batch time over the step time
        fraction_limit: maximum allowed fraction of the step time
        p95_ms: 95th-percentile batch time in milliseconds
        p95_limit_ms: maximum allowed p95 in milliseconds
        passed: whether both limits hold
    """

    median_fraction_of_step: float
    fraction_limit: float
    p95_ms: float
    p95_limit_ms: float
    passed: bool


def crash_on_sentinel_task(
    instance: TrussInstance, completion_text: str
) -> RewardBreakdown:
    """Score normally, but kill the worker process on the crash sentinel.

    Assumptions:
        1. os._exit skips all cleanup, imitating a native crash that never
           unwinds the Python stack; the injection is deterministic so a
           retried sentinel fails again.

    Args:
        instance: the trusted problem instance
        completion_text: the completion to score, or the crash sentinel

    Returns:
        RewardBreakdown: the normal serial score for non-sentinel input
    """
    if completion_text == CRASH_SENTINEL:
        os._exit(1)
    return score_task(instance, completion_text)


def hang_on_sentinel_task(
    instance: TrussInstance, completion_text: str
) -> RewardBreakdown:
    """Score normally, but wedge the worker past its deadline on the sentinel.

    Args:
        instance: the trusted problem instance
        completion_text: the completion to score, or the hang sentinel

    Returns:
        RewardBreakdown: the normal serial score for non-sentinel input
    """
    if completion_text == HANG_SENTINEL:
        time.sleep(HANG_SLEEP_S)
    return score_task(instance, completion_text)


def render_completion(design: TrussDesign) -> str:
    """Render a design as the fenced-JSON completion a policy would emit.

    Args:
        design: the design to render

    Returns:
        str: a fenced json block containing the design's JSON
    """
    return f"```json\n{design.model_dump_json()}\n```"


def load_calibrated_triple(
    sweep_best_path: Path,
) -> tuple[TrussInstance, str]:
    """Load a matched instance, cost reference, and best-design completion.

    A matched triple — not a frozen value grafted onto an unrelated
    instance — is what exercises the graded cost term: cost credit is
    gated by feasibility and vanishes when the instance has no cost
    reference, so only a feasible design paired with its own instance and
    cost reference proves the term executed.

    Args:
        sweep_best_path: path to the frozen sweep-best artifact

    Returns:
        tuple: the calibrated instance (cost reference set) and its
            paired best-design completion

    Raises:
        ValueError: if the rebuilt pair no longer scores feasible with a
            cost total under twice the cost reference, meaning the frozen
            artifact has drifted.
    """
    payload = json.loads(sweep_best_path.read_text())
    entry = payload["instances"][0]
    cost_ref_usd = float(entry["cost_ref_usd"])
    instance = dataclasses.replace(
        instance_from_payload(entry["instance"]), cost_ref_usd=cost_ref_usd
    )
    design = TrussDesign.model_validate(entry["best"]["design"])
    completion = render_completion(design)
    breakdown = score_task(instance, completion)
    if breakdown.feasible is None or breakdown.feasible <= 0.0:
        raise ValueError(
            f"calibrated triple no longer scores feasible: {breakdown.reason}"
        )
    if breakdown.cost_total_usd is None or (
        breakdown.cost_total_usd >= 2.0 * cost_ref_usd
    ):
        raise ValueError(
            f"calibrated triple cost {breakdown.cost_total_usd!r} is no longer "
            f"under twice the cost reference {cost_ref_usd!r}"
        )
    return instance, completion


def build_batch(
    seed: int,
    n_completions: int,
    completions_per_instance: int,
    malformed_every: int,
    sweep_best_path: Path = SWEEP_BEST_PATH,
) -> tuple[tuple[TrussInstance, ...], tuple[str, ...]]:
    """Build a training-shaped batch of instances and completions.

    The first instance is the calibrated triple's, with its paired
    best design as the first completion; remaining instances come from
    the canonical generator with random-design completions, and every
    Nth completion is replaced with malformed text so the parse-failure
    path is exercised.

    Args:
        seed: seed for instance generation and design sampling
        n_completions: total completions in the batch
        completions_per_instance: completions sharing each instance; must
            divide n_completions
        malformed_every: replace completions at positions where
            position % malformed_every == malformed_every - 1; 0 disables
        sweep_best_path: path to the frozen sweep-best artifact

    Returns:
        tuple: (instances, completions), each of length n_completions,
            aligned per completion

    Raises:
        ValueError: if completions_per_instance does not divide
            n_completions, or the calibrated triple fails its guard.
    """
    if n_completions % completions_per_instance != 0:
        raise ValueError(
            f"completions_per_instance={completions_per_instance} does not "
            f"divide n_completions={n_completions}"
        )
    n_instances = n_completions // completions_per_instance
    calibrated_instance, best_completion = load_calibrated_triple(sweep_best_path)
    generated = generate_instances(seed, n_instances - 1)
    roster = (calibrated_instance, *generated)
    rng = random.Random(seed)
    instances: list[TrussInstance] = []
    completions: list[str] = []
    position = 0
    for instance_index, instance in enumerate(roster):
        for completion_index in range(completions_per_instance):
            instances.append(instance)
            if instance_index == 0 and completion_index == 0:
                completions.append(best_completion)
            elif (
                malformed_every > 0
                and position % malformed_every == malformed_every - 1
            ):
                completions.append("the model rambled and emitted no design")
            else:
                completions.append(render_completion(sample_design(rng, instance)))
            position += 1
    return tuple(instances), tuple(completions)


def summarize_timings(timings_s: list[float], n_completions: int) -> TimingSummary:
    """Reduce raw per-repeat timings into the reported statistics.

    Args:
        timings_s: raw per-repeat wall times in seconds
        n_completions: completions per batch, for the per-completion rate

    Returns:
        TimingSummary: raw timings plus median, p95, and per-completion
            milliseconds
    """
    ordered = sorted(timings_s)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    median_s = statistics.median(ordered)
    return TimingSummary(
        timings_s=tuple(timings_s),
        median_s=median_s,
        p95_s=ordered[p95_index],
        per_completion_ms=median_s / n_completions * 1000.0,
    )


def time_serial(
    instances: tuple[TrussInstance, ...],
    completions: tuple[str, ...],
    repeats: int,
) -> tuple[tuple[RewardBreakdown, ...], list[float]]:
    """Time repeated serial scorings of one batch after a warmup pass.

    Args:
        instances: the batch's instances, aligned per completion
        completions: the batch's completion texts
        repeats: timed repetitions after the untimed warmup

    Returns:
        tuple: the final repeat's breakdowns and the raw per-repeat
            timings in seconds
    """
    score_batch(instances, completions)
    timings_s: list[float] = []
    breakdowns: tuple[RewardBreakdown, ...] = ()
    for _ in range(repeats):
        start = time.perf_counter()
        breakdowns = score_batch(instances, completions)
        timings_s.append(time.perf_counter() - start)
    return breakdowns, timings_s


def time_pool(
    pool: VerifierPool,
    instances: tuple[TrussInstance, ...],
    completions: tuple[str, ...],
    repeats: int,
) -> tuple[tuple[RewardBreakdown, ...], list[float], float]:
    """Time repeated pooled scorings, with startup timed separately.

    Args:
        pool: the worker pool to score on
        instances: the batch's instances, aligned per completion
        completions: the batch's completion texts
        repeats: timed repetitions after the untimed warmup

    Returns:
        tuple: the final repeat's breakdowns, the raw per-repeat timings
            in seconds, and the one-time startup seconds
    """
    start = time.perf_counter()
    pool.ensure_workers()
    startup_s = time.perf_counter() - start
    pool.score_batch(instances, completions)
    timings_s: list[float] = []
    breakdowns: tuple[RewardBreakdown, ...] = ()
    for _ in range(repeats):
        start = time.perf_counter()
        breakdowns = pool.score_batch(instances, completions)
        timings_s.append(time.perf_counter() - start)
    return breakdowns, timings_s, startup_s


def breakdowns_equal(
    first: tuple[RewardBreakdown, ...], second: tuple[RewardBreakdown, ...]
) -> bool:
    """Compare two breakdown batches for exact equality.

    Args:
        first: one batch of breakdowns
        second: the other batch of breakdowns

    Returns:
        bool: True when every breakdown is exactly equal, field for field
    """
    return first == second


def evaluate_gate(
    median_s: float, p95_s: float, step_time_estimate_s: float
) -> GateVerdict:
    """Judge one scoring mode against the step-time budget gate.

    Args:
        median_s: the mode's median batch time in seconds
        p95_s: the mode's p95 batch time in seconds
        step_time_estimate_s: the step time the fraction is judged against

    Returns:
        GateVerdict: the fraction and p95 checks with their limits
    """
    fraction = median_s / step_time_estimate_s
    p95_ms = p95_s * 1000.0
    return GateVerdict(
        median_fraction_of_step=fraction,
        fraction_limit=GATE_STEP_FRACTION,
        p95_ms=p95_ms,
        p95_limit_ms=GATE_P95_MS,
        passed=fraction <= GATE_STEP_FRACTION and p95_ms <= GATE_P95_MS,
    )


def benchmark_batch(
    seed: int,
    n_completions: int,
    completions_per_instance: int,
    workers: int,
    repeats: int,
) -> BatchBenchmark:
    """Benchmark serial vs. pool on one batch size, checking score equality.

    Args:
        seed: seed for batch construction
        n_completions: completions per batch
        completions_per_instance: completions sharing each instance
        workers: pool worker count
        repeats: timed repetitions per mode

    Returns:
        BatchBenchmark: both modes' timings and the equality verdict
    """
    instances, completions = build_batch(
        seed, n_completions, completions_per_instance, MALFORMED_EVERY
    )
    serial_breakdowns, serial_timings = time_serial(instances, completions, repeats)
    with VerifierPool(max_workers=workers) as pool:
        pool_breakdowns, pool_timings, startup_s = time_pool(
            pool, instances, completions, repeats
        )
    return BatchBenchmark(
        n_completions=n_completions,
        completions_per_instance=completions_per_instance,
        serial=summarize_timings(serial_timings, n_completions),
        pool=summarize_timings(pool_timings, n_completions),
        pool_startup_s=startup_s,
        scores_equal=breakdowns_equal(serial_breakdowns, pool_breakdowns),
    )


def run_failure_injection(kind: str, seed: int, workers: int) -> InjectionResult:
    """Inject one worker failure kind and verify the pool survives it.

    Survival means: every non-sentinel result exactly equals its serial
    score, the sentinel lands on the graded worker-fault penalty at rung
    2, and a follow-up clean batch through the same pool object matches
    serial — proving the respawned workers are healthy.

    Args:
        kind: the failure to inject, "crash" or "hang"
        seed: seed for batch construction
        workers: pool worker count

    Returns:
        InjectionResult: the injection kind, survival verdict, and the
            pool's respawn count

    Raises:
        ValueError: if kind is neither "crash" nor "hang".
    """
    if kind == "crash":
        task = crash_on_sentinel_task
        sentinel = CRASH_SENTINEL
        task_timeout_s = POOL_TASK_TIMEOUT_S
    elif kind == "hang":
        task = hang_on_sentinel_task
        sentinel = HANG_SENTINEL
        task_timeout_s = HANG_TASK_TIMEOUT_S
    else:
        raise ValueError(f"unknown failure kind {kind!r}")
    instances, clean_completions = build_batch(
        seed, INJECTION_BATCH_SIZE, INJECTION_BATCH_SIZE, MALFORMED_EVERY
    )
    sentinel_index = INJECTION_BATCH_SIZE - 1
    injected_completions = (
        *clean_completions[:sentinel_index],
        sentinel,
        *clean_completions[sentinel_index + 1 :],
    )
    serial_injected = score_batch(instances, injected_completions)
    serial_clean = score_batch(instances, clean_completions)
    with VerifierPool(
        max_workers=workers, task_timeout_s=task_timeout_s, task=task
    ) as pool:
        injected_results = pool.score_batch(instances, injected_completions)
        clean_results = pool.score_batch(instances, clean_completions)
        restart_count = pool.restart_count
    non_sentinel_equal = all(
        injected_results[index] == serial_injected[index]
        for index in range(INJECTION_BATCH_SIZE)
        if index != sentinel_index
    )
    survived = (
        non_sentinel_equal
        and injected_results[sentinel_index].rung == 2
        and breakdowns_equal(clean_results, serial_clean)
    )
    return InjectionResult(
        injected=kind, survived=survived, restart_count=restart_count
    )


def print_benchmark_table(batches: dict[str, BatchBenchmark]) -> None:
    """Print the serial-vs-pool timing summary table.

    Args:
        batches: batch benchmarks keyed by batch size label

    Returns:
        None
    """
    table = Table(title="Reward benchmark", title_justify="left")
    table.add_column("batch")
    table.add_column("serial median ms")
    table.add_column("pool median ms")
    table.add_column("pool startup s")
    table.add_column("scores equal")
    for label, batch in batches.items():
        equal = "[green]yes[/green]" if batch.scores_equal else "[red]no[/red]"
        table.add_row(
            label,
            f"{batch.serial.median_s * 1000.0:.2f}",
            f"{batch.pool.median_s * 1000.0:.2f}",
            f"{batch.pool_startup_s:.2f}",
            equal,
        )
    console.print(table)


def print_failure_table(failures: dict[str, InjectionResult]) -> None:
    """Print the worker failure injection summary table.

    Args:
        failures: injection results keyed by failure kind

    Returns:
        None
    """
    table = Table(title="Worker failure injections", title_justify="left")
    table.add_column("injected")
    table.add_column("survived")
    table.add_column("restarts")
    for result in failures.values():
        survived = "[green]yes[/green]" if result.survived else "[red]no[/red]"
        table.add_row(result.injected, survived, str(result.restart_count))
    console.print(table)


@app.command()
def run(
    seed: int = typer.Option(0, help="Seed for batch construction."),
    workers: int = typer.Option(2, min=1, help="Pool worker count."),
    repeats: int = typer.Option(
        50, min=2, help="Timed repetitions per mode and batch size."
    ),
    step_time_estimate_s: float = typer.Option(
        DEFAULT_STEP_TIME_ESTIMATE_S,
        min=1e-6,
        help="Estimated training step time the budget gate is judged against.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts"), help="Directory the benchmark artifact is written into."
    ),
) -> None:
    """Benchmark pool vs. serial reward scoring and write the artifact.

    Times both modes on 8- and 24-completion batches, verifies pooled
    scores exactly equal serial ones, injects worker crash and hang
    failures, judges the provisional step-time gate, and writes
    artifacts/reward_bench/reward_bench_<run_id>.json.

    Args:
        seed: seed for batch construction
        workers: pool worker count
        repeats: timed repetitions per mode and batch size
        step_time_estimate_s: estimated step time for the budget gate
        output_dir: directory the artifact is written into

    Returns:
        None

    Raises:
        typer.Exit: with code 1 when scores diverge, an injection is not
            survived, or the adopted mode fails the gate.
    """
    stamp = provenance_stamp()
    console.print(f"[bold]Reward bench[/bold]  run_id {stamp['run_id']}")
    batches: dict[str, BatchBenchmark] = {}
    for n_completions in (8, 24):
        console.print(f"Benchmarking {n_completions}-completion batch...")
        batches[str(n_completions)] = benchmark_batch(
            seed, n_completions, COMPLETIONS_PER_INSTANCE, workers, repeats
        )
    console.print("Injecting worker failures...")
    failures = {
        kind: run_failure_injection(kind, seed, workers) for kind in ("crash", "hang")
    }
    reference = batches["24"]
    serial_gate = evaluate_gate(
        reference.serial.median_s, reference.serial.p95_s, step_time_estimate_s
    )
    pool_gate = evaluate_gate(
        reference.pool.median_s, reference.pool.p95_s, step_time_estimate_s
    )
    pool_wins = reference.pool.median_s < reference.serial.median_s and pool_gate.passed
    provisional_adopted_mode = "pool" if pool_wins else "serial"
    adopted_gate = pool_gate if pool_wins else serial_gate
    scores_equal = all(batch.scores_equal for batch in batches.values())
    survived = all(result.survived for result in failures.values())
    passed = scores_equal and survived and adopted_gate.passed

    payload = {
        "stamp": stamp,
        "config": {
            "seed": seed,
            "workers": workers,
            "repeats": repeats,
            "step_time_estimate_s": step_time_estimate_s,
            "completions_per_instance": COMPLETIONS_PER_INSTANCE,
            "malformed_every": MALFORMED_EVERY,
        },
        "batches": {
            label: dataclasses.asdict(batch) for label, batch in batches.items()
        },
        "worker_failure": {
            kind: dataclasses.asdict(result) for kind, result in failures.items()
        },
        "gate": {
            "gate_basis": "estimated_step_time",
            "provisional": True,
            "step_time_estimate_s": step_time_estimate_s,
            "serial": dataclasses.asdict(serial_gate),
            "pool": dataclasses.asdict(pool_gate),
        },
        "provisional_adopted_mode": provisional_adopted_mode,
        "passed": passed,
    }
    artifact_path = output_dir / BENCH_DIR / f"reward_bench_{stamp['run_id']}.json"
    write_json(artifact_path, payload)

    print_benchmark_table(batches)
    print_failure_table(failures)
    console.print(
        f"Provisionally adopted mode: [bold]{provisional_adopted_mode}[/bold] "
        "(gate judged against an estimated step time; re-evaluate against a "
        "measured one)"
    )
    console.print(f"Benchmark written to {artifact_path}")
    if passed:
        console.print("[green]Reward bench passed[/green]")
        return
    console.print("[red]Reward bench failed[/red] — see the artifact for details.")
    raise typer.Exit(1)


def main() -> None:
    """Run the reward benchmark command-line interface.

    Args: None

    Returns:
        None
    """
    app()


if __name__ == "__main__":
    main()
