"""Produce the frozen grading constants — the ground truth for training.

Runs thousands of random designs through the verifier, applies the sanity
gates, and writes cost_ref and sweep_best to artifacts/, stamped so every
result traces to an exact code and catalog state.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from trussRL.calibration.artifacts import (
    breakdown_payload,
    config_payload,
    design_payload,
    instance_payload,
    provenance_stamp,
    write_json,
)
from trussRL.calibration.gates import (
    INTERIOR_MARGIN_FRACTION,
    SATURATION_MEAN_MAX,
    TOP_FRACTION,
    VARIANCE_STD_MIN,
    GateReport,
    evaluate_gates,
)
from trussRL.calibration.sweep import (
    CalibrationConfig,
    CalibrationSeeds,
    InstanceCalibration,
    SweepBest,
    calibrate_instance,
    derive_calibration_seeds,
    find_sweep_best,
)
from trussRL.generator import GENERATOR_VERSION, generate_instances
from trussRL.instance import TrussInstance

app = typer.Typer(add_completion=False)
console = Console()

GATE_REPORT_DIR = "gate_reports"
COST_REF_FILENAME = "cost_ref.json"
SWEEP_BEST_FILENAME = "sweep_best.json"


def check_value(report: GateReport, gate: str) -> float | None:
    """Read one gate check's measured value out of a report.

    Args:
        report: the instance's gate report
        gate: the gate identifier to look up

    Returns:
        float: the check's measured value, or None when the check is
            absent or skipped
    """
    for check in report.checks:
        if check.gate == gate:
            return check.value
    return None


def thresholds_payload() -> dict[str, float]:
    """Echo the gate thresholds into the gate report artifact.

    Args: None

    Returns:
        dict: every gate threshold constant by name
    """
    return {
        "saturation_mean_max": SATURATION_MEAN_MAX,
        "variance_std_min": VARIANCE_STD_MIN,
        "top_fraction": TOP_FRACTION,
        "interior_margin_fraction": INTERIOR_MARGIN_FRACTION,
    }


def config_echo_payload(config: CalibrationConfig) -> dict[str, object]:
    """Build the config echo every artifact carries, with provenance metadata.

    Args:
        config: the run's calibration config

    Returns:
        dict: every config field plus a generator_version key
    """
    payload = config_payload(config)
    payload["generator_version"] = GENERATOR_VERSION
    return payload


def gate_report_payload(
    stamp: dict[str, object],
    config: CalibrationConfig,
    seeds: CalibrationSeeds,
    reports: list[GateReport],
    calibrations: list[InstanceCalibration | None],
) -> dict[str, object]:
    """Assemble the gate report artifact payload.

    Args:
        stamp: the run's provenance stamp
        config: the run's calibration config
        seeds: the run's derived seeds
        reports: one gate report per roster instance
        calibrations: index-aligned calibrations, None where errored

    Returns:
        dict: stamp, config echo, thresholds, all_passed, and per-instance
            stats plus checks
    """
    instances_payload: list[dict[str, object]] = []
    for report, calibration in zip(reports, calibrations, strict=True):
        entry: dict[str, object] = {
            "instance_index": report.instance_index,
            "passed": report.passed,
            "error": report.error,
            "checks": [
                {
                    "gate": check.gate,
                    "status": check.status,
                    "value": check.value,
                    "criterion": check.criterion,
                    "detail": check.detail,
                }
                for check in report.checks
            ],
        }
        if calibration is not None:
            entry["instance"] = instance_payload(calibration.instance)
            entry["sweep_seed"] = calibration.seed
            entry["n_samples"] = calibration.n_samples
            entry["n_rung3"] = calibration.n_rung3
            entry["n_strictly_feasible"] = calibration.n_strictly_feasible
            entry["cost_ref_usd"] = calibration.cost_ref_usd
        instances_payload.append(entry)
    return {
        "stamp": stamp,
        "config": config_echo_payload(config),
        "roster_seed": seeds.roster_seed,
        "thresholds": thresholds_payload(),
        "all_passed": all(report.passed for report in reports),
        "instances": instances_payload,
    }


def cost_ref_artifact_payload(
    stamp: dict[str, object],
    config: CalibrationConfig,
    seeds: CalibrationSeeds,
    calibrations: list[InstanceCalibration],
) -> dict[str, object]:
    """Assemble the frozen cost_ref artifact payload.

    Args:
        stamp: the run's provenance stamp
        config: the run's calibration config
        seeds: the run's derived seeds
        calibrations: one successful calibration per roster instance

    Returns:
        dict: stamp, config echo, roster seed, and one entry per instance
            with its inline parameters and derived cost_ref
    """
    return {
        "stamp": stamp,
        "config": config_echo_payload(config),
        "roster_seed": seeds.roster_seed,
        "instances": [
            {
                "instance_index": index,
                "instance": instance_payload(calibration.instance),
                "sweep_seed": calibration.seed,
                "n_samples": calibration.n_samples,
                "n_rung3": calibration.n_rung3,
                "n_strictly_feasible": calibration.n_strictly_feasible,
                "cost_ref_usd": calibration.cost_ref_usd,
            }
            for index, calibration in enumerate(calibrations)
        ],
    }


def sweep_best_artifact_payload(
    stamp: dict[str, object],
    config: CalibrationConfig,
    seeds: CalibrationSeeds,
    roster: tuple[TrussInstance, ...],
    bests: list[SweepBest],
) -> dict[str, object]:
    """Assemble the frozen sweep_best artifact payload.

    best.cost_total_usd inside each entry is what Phase 2's
    near-optimality metric consumes.

    Args:
        stamp: the run's provenance stamp
        config: the run's calibration config
        seeds: the run's derived seeds
        roster: the roster instances, index-aligned with bests
        bests: one sweep_best result per roster instance

    Returns:
        dict: stamp, config echo, roster seed, and one entry per instance
            with the best design and its regraded breakdown
    """
    return {
        "stamp": stamp,
        "config": config_echo_payload(config),
        "roster_seed": seeds.roster_seed,
        "instances": [
            {
                "instance_index": index,
                "instance": instance_payload(instance),
                "sweep_best_seed": best.seed,
                "n_samples": best.n_samples,
                "cost_ref_usd": best.cost_ref_usd,
                "best": {
                    "design": design_payload(best.design),
                    **breakdown_payload(best.breakdown),
                },
            }
            for index, (instance, best) in enumerate(zip(roster, bests, strict=True))
        ],
    }


def print_summary_table(
    roster: tuple[TrussInstance, ...],
    reports: list[GateReport],
    calibrations: list[InstanceCalibration | None],
) -> None:
    """Print the per-instance calibration summary table.

    Args:
        roster: the roster instances
        reports: one gate report per instance
        calibrations: index-aligned calibrations, None where errored

    Returns: None
    """
    table = Table(title="Calibration summary", title_justify="left")
    table.add_column("idx", justify="right")
    table.add_column("span ft", justify="right")
    table.add_column("depth lim", justify="right")
    table.add_column("L/n", justify="right")
    table.add_column("w kip/ft", justify="right")
    table.add_column("rung3", justify="right")
    table.add_column("feas", justify="right")
    table.add_column("cost_ref", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("pstdev", justify="right")
    table.add_column("med bays", justify="right")
    table.add_column("med depth", justify="right")
    table.add_column("state")
    for instance, report, calibration in zip(
        roster, reports, calibrations, strict=True
    ):
        depth_limit = (
            "none"
            if instance.depth_limit_ft is None
            else f"{instance.depth_limit_ft:g}"
        )
        w_kip_per_ft = instance.load_cases[0].w_kip_per_ft
        if calibration is None:
            table.add_row(
                str(report.instance_index),
                f"{instance.span_ft:g}",
                depth_limit,
                str(instance.defl_denom),
                f"{w_kip_per_ft:g}",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                f"[red]error[/red]: {report.error}",
            )
            continue
        mean = check_value(report, "saturation")
        pstdev = check_value(report, "variance")
        median_bays = check_value(report, "interior_n_bays")
        median_depth = check_value(report, "interior_depth_lower")
        upper_skipped = any(
            check.gate == "interior_depth_upper" and check.status == "skipped"
            for check in report.checks
        )
        if report.passed:
            state = "[green]pass[/green]"
            if upper_skipped:
                state += " [dim](upper depth skipped)[/dim]"
        else:
            failed = ", ".join(
                check.gate for check in report.checks if check.status == "fail"
            )
            state = f"[red]fail[/red]: {failed}"
        table.add_row(
            str(report.instance_index),
            f"{instance.span_ft:g}",
            depth_limit,
            str(instance.defl_denom),
            f"{w_kip_per_ft:g}",
            str(calibration.n_rung3),
            str(calibration.n_strictly_feasible),
            f"{calibration.cost_ref_usd:.0f}",
            "-" if mean is None else f"{mean:.3f}",
            "-" if pstdev is None else f"{pstdev:.3f}",
            "-" if median_bays is None else f"{median_bays:g}",
            "-" if median_depth is None else f"{median_depth:.2f}",
            state,
        )
    console.print(table)


def print_best_table(roster: tuple[TrussInstance, ...], bests: list[SweepBest]) -> None:
    """Print the per-instance sweep_best summary table.

    Args:
        roster: the roster instances
        bests: one sweep_best result per instance

    Returns: None
    """
    table = Table(title="Sweep best per instance", title_justify="left")
    table.add_column("idx", justify="right")
    table.add_column("span ft", justify="right")
    table.add_column("n_bays", justify="right")
    table.add_column("depth ft", justify="right")
    table.add_column("score", justify="right")
    table.add_column("cost $", justify="right")
    table.add_column("feasible", justify="right")
    for index, (instance, best) in enumerate(zip(roster, bests, strict=True)):
        cost = best.breakdown.cost_total_usd
        table.add_row(
            str(index),
            f"{instance.span_ft:g}",
            str(best.design.n_bays),
            f"{best.design.depth_ft:.2f}",
            f"{best.breakdown.score:.3f}",
            "-" if cost is None else f"{cost:.0f}",
            "-" if best.breakdown.feasible is None else f"{best.breakdown.feasible:g}",
        )
    console.print(table)


def print_remediation(reports: list[GateReport]) -> None:
    """Print gate-specific remediation guidance for a failed run.

    Args:
        reports: every instance's gate report

    Returns: None
    """
    failed_gates = {
        check.gate
        for report in reports
        for check in report.checks
        if check.status == "fail"
    }
    errors = [report for report in reports if report.error is not None]
    console.print("[red]Calibration gates failed.[/red]")
    if errors:
        console.print(
            f"  {len(errors)} instance(s) errored during calibration "
            "(see the gate report for details); a common cause is zero "
            "strictly feasible designs in the sweep."
        )
    if "interior_n_bays" in failed_gates:
        console.print(
            "  interior_n_bays: a median pegged high means COST_PER_NODE_USD "
            "(reward.py) is too low to matter — raise it; pegged low means "
            "it dominates — lower it. Iterate with --gates-only."
        )
    if "interior_depth_lower" in failed_gates or "interior_depth_upper" in failed_gates:
        console.print(
            "  interior_depth: investigate the DRC/reward coupling first — "
            "do not assume the per-node cost is the cause."
        )
    if "saturation" in failed_gates or "variance" in failed_gates:
        console.print(
            "  saturation/variance: inspect the gate report's distribution "
            "stats; the thresholds encode a settled decision — do not tune "
            "them."
        )
    console.print("  No frozen constants were written.")


@app.command()
def run(
    seed: int = typer.Option(0, help="Master seed every stream derives from."),
    n_instances: int = typer.Option(32, help="Number of roster instances."),
    designs_per_instance: int = typer.Option(
        5000, help="Random designs per instance for gates and cost_ref."
    ),
    sweep_samples: int = typer.Option(
        100000, help="Random designs per instance for the sweep_best search."
    ),
    output_dir: Path = typer.Option(
        Path("artifacts"), help="Directory the artifacts are written into."
    ),
    gates_only: bool = typer.Option(
        False,
        "--gates-only",
        help="Run gates and write only the gate report; skip sweep_best.",
    ),
) -> None:
    """Calibrate the reward landscape and freeze the grading constants.

    Sweeps random designs per roster instance, derives cost_ref, runs the
    sanity gates on the regraded distributions, and always writes a
    timestamped gate report. Only a fully successful full run writes the
    frozen cost_ref.json and sweep_best.json — both together, or neither.

    Args:
        seed: master seed every stream derives from
        n_instances: number of roster instances
        designs_per_instance: random designs per instance for gates and
            cost_ref
        sweep_samples: random designs per instance for sweep_best
        output_dir: directory the artifacts are written into
        gates_only: skip sweep_best and write only the gate report

    Returns: None

    Raises:
        typer.Exit: with code 1 when any gate fails or any instance
            errors.
    """
    config = CalibrationConfig(
        seed=seed,
        n_instances=n_instances,
        designs_per_instance=designs_per_instance,
        sweep_best_samples=sweep_samples,
    )
    stamp = provenance_stamp()
    console.print(
        f"[bold]Calibration[/bold]  run_id {stamp['run_id']}, seed {seed}, "
        f"{n_instances} instances x {designs_per_instance} designs"
        + ("" if gates_only else f", sweep_best {sweep_samples} samples")
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

    seeds = derive_calibration_seeds(seed, n_instances)
    roster = generate_instances(seeds.roster_seed, n_instances, config.generator)

    calibrations: list[InstanceCalibration | None] = []
    reports: list[GateReport] = []
    with Progress(console=console) as progress:
        task = progress.add_task("Calibrating instances", total=n_instances)
        for index, instance in enumerate(roster):
            try:
                calibration = calibrate_instance(
                    instance,
                    seeds.instance_seeds[index].calibration,
                    designs_per_instance,
                )
            except ValueError as error:
                calibrations.append(None)
                reports.append(
                    GateReport(instance_index=index, checks=(), error=str(error))
                )
            else:
                calibrations.append(calibration)
                reports.append(evaluate_gates(calibration, index))
            progress.advance(task)

    print_summary_table(roster, reports, calibrations)
    n_upper_checked = sum(
        1
        for report in reports
        for check in report.checks
        if check.gate == "interior_depth_upper" and check.status != "skipped"
    )
    console.print(
        f"Depth upper-bound check exercised on {n_upper_checked} of "
        f"{n_instances} instance(s); skipped where a tighter stated limit binds."
    )

    report_path = output_dir / GATE_REPORT_DIR / f"calibration_{stamp['run_id']}.json"
    write_json(
        report_path, gate_report_payload(stamp, config, seeds, reports, calibrations)
    )
    console.print(f"Gate report written to {report_path}")

    if not all(report.passed for report in reports):
        print_remediation(reports)
        raise typer.Exit(1)

    console.print("[green]All calibration gates passed.[/green]")
    if gates_only:
        console.print(
            "--gates-only: frozen constants not written; run without it to "
            "freeze cost_ref and sweep_best."
        )
        return

    successful = [
        calibration for calibration in calibrations if calibration is not None
    ]
    bests: list[SweepBest] = []
    with Progress(console=console) as progress:
        task = progress.add_task("Searching sweep_best", total=n_instances)
        for index, (instance, calibration) in enumerate(
            zip(roster, successful, strict=True)
        ):
            bests.append(
                find_sweep_best(
                    instance,
                    calibration.cost_ref_usd,
                    seeds.instance_seeds[index].sweep_best,
                    sweep_samples,
                )
            )
            progress.advance(task)

    write_json(
        output_dir / COST_REF_FILENAME,
        cost_ref_artifact_payload(stamp, config, seeds, successful),
    )
    write_json(
        output_dir / SWEEP_BEST_FILENAME,
        sweep_best_artifact_payload(stamp, config, seeds, roster, bests),
    )
    print_best_table(roster, bests)
    console.print(
        f"Frozen constants written: {output_dir / COST_REF_FILENAME} and "
        f"{output_dir / SWEEP_BEST_FILENAME} (run_id {stamp['run_id']})."
    )


def main() -> None:
    """Run the calibration command-line interface.

    Args: None

    Returns: None
    """
    app()


if __name__ == "__main__":
    main()
