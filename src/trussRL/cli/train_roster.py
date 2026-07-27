"""Build the frozen training roster — 512 calibrated instances for training.

Draws a deterministic candidate stream, filters collisions against the 32
calibrated instances, computes each survivor's cost_ref by the exact
calibration protocol until 512 succeed, checks the difficulty spread, and
writes a provenance-stamped artifacts/train_roster.json that pins its
calibration sources by content hash.
"""

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress
from rich.table import Table

from trussRL.calibration.artifacts import (
    instance_from_payload,
    provenance_stamp,
    write_json,
)
from trussRL.calibration.roster import (
    COST_REF_FILENAME,
    SWEEP_BEST_FILENAME,
    TRAIN_ROSTER_FILENAME,
    DroppedCandidate,
    RosterCandidate,
    RosterConfig,
    SpreadCheck,
    artifact_sha256,
    check_difficulty_spread,
    derive_generator_child_seeds,
    instance_key,
    load_eval_split,
    quantile,
    select_candidates,
    train_roster_payload,
)
from trussRL.calibration.sweep import (
    InstanceCalibration,
    calibrate_instance,
    derive_calibration_seeds,
    validate_positive_count,
)
from trussRL.generator import generate_instances

app = typer.Typer(add_completion=False)
console = Console()


def sources_payload(artifacts_dir: Path) -> dict[str, object]:
    """Pin the calibration source artifacts by run_id and content hash.

    Args:
        artifacts_dir: directory holding the two calibration artifacts

    Returns:
        dict: filename, run_id, and sha256 for cost_ref and sweep_best
    """
    sources: dict[str, object] = {}
    for key, filename in (
        ("cost_ref", COST_REF_FILENAME),
        ("sweep_best", SWEEP_BEST_FILENAME),
    ):
        path = artifacts_dir / filename
        payload = json.loads(path.read_text())
        sources[key] = {
            "filename": filename,
            "run_id": payload["stamp"]["run_id"],
            "sha256": artifact_sha256(path),
        }
    return sources


def load_calibration_reference(
    artifacts_dir: Path,
) -> tuple[set[tuple[object, ...]], tuple[float, ...]]:
    """Read the calibration roster's identity keys and cost_ref values.

    Args:
        artifacts_dir: directory holding cost_ref.json

    Returns:
        tuple: the identity keys of all calibrated instances, and their
            cost_ref values in roster order
    """
    payload = json.loads((artifacts_dir / COST_REF_FILENAME).read_text())
    keys: set[tuple[object, ...]] = set()
    cost_refs: list[float] = []
    for entry in payload["instances"]:
        keys.add(instance_key(instance_from_payload(entry["instance"])))
        cost_refs.append(float(entry["cost_ref_usd"]))
    return keys, tuple(cost_refs)


def print_spread_table(checks: tuple[SpreadCheck, ...]) -> None:
    """Print the difficulty-spread verdicts with their measurements.

    Args:
        checks: the spread checks to print

    Returns: None
    """
    table = Table(title="Difficulty spread", title_justify="left")
    table.add_column("gate")
    table.add_column("state")
    table.add_column("measurements")
    table.add_column("criterion")
    for check in checks:
        state = (
            "[green]pass[/green]"
            if check.status == "pass"
            else f"[red]fail[/red]: {escape(check.detail)}"
        )
        measurements = ", ".join(
            f"{name}={value:g}" for name, value in check.measurements.items()
        )
        table.add_row(check.gate, state, measurements, escape(check.criterion))
    console.print(table)


def print_drop_summary(dropped: list[DroppedCandidate]) -> None:
    """Print the dropped-candidate counts grouped by reason.

    Args:
        dropped: every dropped candidate

    Returns: None
    """
    if not dropped:
        console.print("Dropped candidates: none")
        return
    counts: dict[str, int] = {}
    for item in dropped:
        reason = item.reason.split(":", 1)[0]
        counts[reason] = counts.get(reason, 0) + 1
    summary = ", ".join(f"{reason} {count}" for reason, count in sorted(counts.items()))
    console.print(f"Dropped candidates: {len(dropped)} ({summary})")


def print_cost_ref_summary(
    train_cost_refs: tuple[float, ...], calibration_cost_refs: tuple[float, ...]
) -> None:
    """Print train vs calibration cost_ref quantiles.

    Args:
        train_cost_refs: the frozen training cost_ref values
        calibration_cost_refs: the calibration roster's cost_ref values

    Returns: None
    """
    levels = (0.10, 0.50, 0.90)
    train = ", ".join(
        f"q{int(level * 100)}={quantile(train_cost_refs, level):.0f}"
        for level in levels
    )
    calibration = ", ".join(
        f"q{int(level * 100)}={quantile(calibration_cost_refs, level):.0f}"
        for level in levels
    )
    console.print(f"cost_ref quantiles: train {train}; calibration {calibration}")


@app.command()
def run(
    seed: int = typer.Option(1, help="Master seed every roster stream derives from."),
    n_target: int = typer.Option(
        512, help="Exact number of training instances to freeze."
    ),
    n_candidates: int = typer.Option(
        576, help="Candidate instances drawn before filtering."
    ),
    designs_per_instance: int = typer.Option(
        5000, help="Random designs per instance for cost_ref."
    ),
    artifacts_dir: Path = typer.Option(
        Path("artifacts"),
        help=(
            "Directory holding the calibration artifacts and receiving "
            "train_roster.json."
        ),
    ),
) -> None:
    """Build and freeze the training roster artifact.

    Loads the calibration artifacts, draws the candidate stream, filters
    collisions and duplicates, calibrates survivors in draw order until
    n_target succeed, checks the difficulty spread, and writes
    train_roster.json. Any failure exits 1 and writes nothing, so an
    existing frozen roster survives a failed run untouched.

    Args:
        seed: master seed every roster stream derives from
        n_target: exact number of training instances to freeze
        n_candidates: candidate instances drawn before filtering
        designs_per_instance: random designs per instance for cost_ref
        artifacts_dir: directory holding the calibration artifacts and
            receiving train_roster.json

    Returns: None

    Raises:
        typer.Exit: with code 1 on invalid counts, unusable calibration
            artifacts, candidate-stream exhaustion, or a failed spread
            check.
    """
    start = time.monotonic()
    try:
        validate_positive_count("n_target", n_target)
        validate_positive_count("n_candidates", n_candidates)
        validate_positive_count("designs_per_instance", designs_per_instance)
        if n_candidates < n_target:
            raise ValueError(
                f"n_candidates must be >= n_target, got {n_candidates} < {n_target}"
            )
    except ValueError as error:
        console.print(f"[red]Invalid counts:[/red] {error}")
        raise typer.Exit(1) from error

    config = RosterConfig(
        seed=seed,
        n_target=n_target,
        n_candidates=n_candidates,
        designs_per_instance=designs_per_instance,
    )
    stamp = provenance_stamp()
    console.print(
        f"[bold]Training roster[/bold]  run_id {stamp['run_id']}, seed {seed}, "
        f"{n_candidates} candidates -> {n_target} instances x "
        f"{designs_per_instance} designs"
    )
    if stamp["git_sha"] == "unknown":
        console.print(
            "[yellow]Warning:[/yellow] git SHA could not be read; provenance "
            "is stamped as 'unknown'."
        )
    if stamp["git_dirty"]:
        console.print(
            "[yellow]Warning:[/yellow] working tree is dirty; the stamped git "
            "SHA does not fully describe the code that ran."
        )

    try:
        eval_split = load_eval_split(artifacts_dir)
        sources = sources_payload(artifacts_dir)
        excluded_keys, calibration_cost_refs = load_calibration_reference(artifacts_dir)
    except (OSError, ValueError, KeyError) as error:
        console.print(f"[red]Calibration artifacts unusable:[/red] {error}")
        raise typer.Exit(1) from error
    held_out_indices = tuple(item.index for item in eval_split)

    seeds = derive_calibration_seeds(seed, n_candidates)
    child_seeds = derive_generator_child_seeds(seeds.roster_seed, n_candidates)
    roster = generate_instances(seeds.roster_seed, n_candidates, config.generator)
    candidates = tuple(
        RosterCandidate(
            generator_index=index,
            instance_seed=child_seeds[index],
            instance=instance,
            sweep_seed=seeds.instance_seeds[index].calibration,
        )
        for index, instance in enumerate(roster)
    )

    try:
        survivors, dropped_selection = select_candidates(
            candidates, excluded_keys, n_target
        )
    except ValueError as error:
        console.print(
            f"[red]Candidate stream exhausted:[/red] {error}. Nothing written; "
            "rerun with a larger --n-candidates or a different --seed."
        )
        raise typer.Exit(1) from error
    dropped: list[DroppedCandidate] = list(dropped_selection)

    accepted: list[RosterCandidate] = []
    calibrations: list[InstanceCalibration] = []
    with Progress(console=console) as progress:
        task = progress.add_task("Calibrating training instances", total=n_target)
        for candidate in survivors:
            if len(calibrations) == n_target:
                break
            try:
                calibration = calibrate_instance(
                    candidate.instance, candidate.sweep_seed, designs_per_instance
                )
            except ValueError as error:
                dropped.append(
                    DroppedCandidate(
                        generator_index=candidate.generator_index,
                        instance_seed=candidate.instance_seed,
                        instance=candidate.instance,
                        reason=f"calibration_error: {error}",
                    )
                )
                continue
            accepted.append(candidate)
            calibrations.append(calibration)
            progress.advance(task)

    if len(calibrations) < n_target:
        console.print(
            f"[red]Candidate stream exhausted:[/red] only {len(calibrations)} "
            f"instance(s) calibrated of the {n_target} needed. Nothing written; "
            "rerun with a larger --n-candidates or a different --seed."
        )
        raise typer.Exit(1)

    train_instances = tuple(candidate.instance for candidate in accepted)
    train_cost_refs = tuple(calibration.cost_ref_usd for calibration in calibrations)
    spread_checks = check_difficulty_spread(
        train_instances, train_cost_refs, calibration_cost_refs
    )
    print_spread_table(spread_checks)
    if any(check.status == "fail" for check in spread_checks):
        console.print(
            "[red]Difficulty-spread checks failed; nothing written.[/red] "
            "Rebuild with a different --seed; do not edit a threshold after "
            "seeing the data."
        )
        raise typer.Exit(1)

    payload = train_roster_payload(
        stamp=stamp,
        config=config,
        roster_seed=seeds.roster_seed,
        sources=sources,
        calibrations=calibrations,
        candidates=tuple(accepted),
        dropped=tuple(dropped),
        spread_checks=spread_checks,
        held_out_indices=held_out_indices,
    )
    write_json(artifacts_dir / TRAIN_ROSTER_FILENAME, payload)
    print_drop_summary(dropped)
    print_cost_ref_summary(train_cost_refs, calibration_cost_refs)
    duration = time.monotonic() - start
    console.print(
        f"Frozen training roster written: {artifacts_dir / TRAIN_ROSTER_FILENAME} "
        f"(run_id {stamp['run_id']}, {len(calibrations)} instances, "
        f"{duration:.1f}s build)."
    )


def main() -> None:
    """Run the training-roster command-line interface.

    Args: None

    Returns: None
    """
    app()


if __name__ == "__main__":
    main()
