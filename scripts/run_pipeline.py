"""Watch one design flow through the scoring pipeline, stage by stage.

Dev-only driver that calls verifier.run_pipeline and pretty-prints the
trace. It composes nothing itself, so it can never drift from score();
unbuilt stages are reported, not fatal.
"""

import json
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from trussRL.capacity import MemberCapacity
from trussRL.expander import TrussGeometry
from trussRL.instance import TrussInstance
from trussRL.loads import CaseLoads, LoadCase
from trussRL.reward import RewardBreakdown, rung0
from trussRL.schema import ParseFailure, TrussDesign, parse_design
from trussRL.solver import SolveOutcome
from trussRL.verifier import PipelineTrace, StageStatus, run_pipeline

app = typer.Typer(add_completion=False)
console = Console()

STATE_STYLES = {
    "ok": "green",
    "failed": "red",
    "not_built": "yellow",
    "skipped": "dim",
}

DEMO_DESIGN = TrussDesign(
    truss_type="warren",
    n_bays=8,
    depth_ft=8.0,
    top_chord="HSS6X6X3/8",
    bottom_chord="HSS6X6X3/8",
    diagonals="HSS4X4X1/4",
)


def load_design(path: Path) -> TrussDesign | ParseFailure:
    """Load a design dict from a JSON file and validate it against the schema.

    Args:
        path: path to a JSON file holding one design dict

    Returns:
        TrussDesign: the validated design, or ParseFailure with the reason
            when the file is not valid JSON or fails schema validation
    """
    try:
        return TrussDesign(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as error:
        return ParseFailure(reason=f"invalid JSON in {path}: {error}")
    except ValidationError as error:
        return ParseFailure(reason=f"schema validation failed for {path}: {error}")


def print_instance(instance: TrussInstance) -> None:
    """Print the problem instance the design is graded against.

    Args:
        instance: the trusted problem instance

    Returns: None
    """
    depth_limit = (
        "none stated"
        if instance.depth_limit_ft is None
        else f"{instance.depth_limit_ft:g} ft"
    )
    console.print(
        f"[bold]Instance[/bold]  span {instance.span_ft:g} ft, "
        f"depth limit {depth_limit}, deflection limit L/{instance.defl_denom}"
    )
    for index, case in enumerate(instance.load_cases):
        console.print(
            f"  case {index}: w = {case.w_kip_per_ft:g} kip/ft at {case.level} "
            f"chord, longitudinal = {case.longitudinal_kip:g} kip"
        )


def print_design(design: TrussDesign) -> None:
    """Print the design entering the pipeline.

    Args:
        design: the schema-validated design

    Returns: None
    """
    sections = ", ".join(
        f"{group}={designation}"
        for group, designation in design.sections_by_group().items()
    )
    console.print(
        f"[bold]Design[/bold]    {design.truss_type}, {design.n_bays} bays, "
        f"depth {design.depth_ft:g} ft\n  {sections}"
    )


def print_stage_table(stages: tuple[StageStatus, ...]) -> None:
    """Print the per-stage status table for one pipeline run.

    Args:
        stages: stage statuses in pipeline order, including the parse row
            the driver prepends

    Returns: None
    """
    table = Table(title="Pipeline stages", title_justify="left")
    table.add_column("stage")
    table.add_column("state")
    table.add_column("detail", overflow="fold")
    for status in stages:
        style = STATE_STYLES.get(status.state, "")
        label = "NOT BUILT" if status.state == "not_built" else status.state
        table.add_row(status.stage, f"[{style}]{label}[/{style}]", status.note)
    console.print(table)


def print_geometry(geometry: TrussGeometry) -> None:
    """Print a summary of the expanded geometry.

    Args:
        geometry: expanded truss geometry

    Returns: None
    """
    group_counts: dict[str, int] = {}
    for member in geometry.members:
        group_counts[member.group] = group_counts.get(member.group, 0) + 1
    groups = ", ".join(f"{count} {group}" for group, count in group_counts.items())
    console.print(
        f"[bold]Geometry[/bold]  {geometry.n_nodes} nodes, "
        f"{len(geometry.members)} members ({groups}), "
        f"bay width {geometry.bay_width_ft:g} ft, "
        f"pin node {geometry.pin_node}, roller node {geometry.roller_node}"
    )


def print_case_loads(case_loads: tuple[CaseLoads, ...]) -> None:
    """Print the resolved nodal loads and conservation totals per case.

    Args:
        case_loads: one CaseLoads per load case, self-weight included

    Returns: None
    """
    for index, case in enumerate(case_loads):
        table = Table(
            title=f"Case {index} nodal loads (self-weight included)",
            title_justify="left",
        )
        table.add_column("node", justify="right")
        table.add_column("fx (kip)", justify="right")
        table.add_column("fy (kip)", justify="right")
        for load in case.nodal_loads:
            table.add_row(str(load.node_id), f"{load.fx_kip:.3f}", f"{load.fy_kip:.3f}")
        table.add_row(
            "[bold]sum[/bold]",
            f"[bold]{case.total_fx_kip:.3f}[/bold]",
            f"[bold]{case.total_fy_kip:.3f}[/bold]",
        )
        console.print(table)


def print_solve(solve_outcome: SolveOutcome) -> None:
    """Print member axial forces and worst displacement per solved case.

    Args:
        solve_outcome: successful or failed solver output

    Returns: None
    """
    if not solve_outcome.ok:
        console.print(f"[red]Solve failed:[/red] {solve_outcome.failure}")
        return
    for index, solution in enumerate(solve_outcome.solutions):
        worst_dy = min(dy for _, dy in solution.node_displacements_in.values())
        table = Table(
            title=f"Case {index} member axials (worst dy {worst_dy:.4f} in)",
            title_justify="left",
        )
        table.add_column("member", justify="right")
        table.add_column("axial (kip, +tension)", justify="right")
        for member_id in sorted(solution.member_axials_kip):
            table.add_row(
                str(member_id), f"{solution.member_axials_kip[member_id]:.2f}"
            )
        console.print(table)


def print_capacities(capacities: tuple[MemberCapacity, ...]) -> None:
    """Print per-member AISC design capacities.

    Args:
        capacities: per-member capacities ordered by member id

    Returns: None
    """
    table = Table(title="Member capacities", title_justify="left")
    table.add_column("member", justify="right")
    table.add_column("tension φPn (kip)", justify="right")
    table.add_column("compression φPn (kip)", justify="right")
    for capacity in capacities:
        table.add_row(
            str(capacity.member_id),
            f"{capacity.tension_kip:.1f}",
            f"{capacity.compression_kip:.1f}",
        )
    console.print(table)


def print_breakdown(breakdown: RewardBreakdown) -> None:
    """Print the final reward breakdown.

    Args:
        breakdown: the verifier's final output

    Returns: None
    """
    console.print(
        f"[bold]Reward[/bold]    rung {breakdown.rung}, "
        f"score {breakdown.score:.3f} — {breakdown.reason}"
    )
    if breakdown.rung == 3:
        console.print(
            f"  U_strength {breakdown.u_strength}, U_buckling {breakdown.u_buckling}, "
            f"U_defl {breakdown.u_defl}, cost ${breakdown.cost_total_usd}, "
            f"feasible {breakdown.feasible}"
        )


def print_trace(parse_status: StageStatus, trace: PipelineTrace) -> None:
    """Print every artifact of one pipeline run, stage table first.

    Args:
        parse_status: how the design was obtained, shown as the parse row
        trace: the pipeline trace to render

    Returns: None
    """
    print_stage_table((parse_status, *trace.stages))
    if trace.geometry is not None:
        print_geometry(trace.geometry)
    if trace.case_loads is not None:
        print_case_loads(trace.case_loads)
    if trace.solve_outcome is not None:
        print_solve(trace.solve_outcome)
    if trace.capacities is not None:
        print_capacities(trace.capacities)
    if trace.breakdown is not None:
        print_breakdown(trace.breakdown)
    else:
        pending = ", ".join(
            status.stage
            for status in trace.stages
            if status.state in ("not_built", "failed")
        )
        console.print(f"[yellow]No score yet[/yellow] — incomplete stages: {pending}")


@app.command()
def run(
    design_json: Path | None = typer.Option(
        None, help="JSON file with one design dict; the demo design if omitted."
    ),
    completion: Path | None = typer.Option(
        None, help="Raw completion text file; exercises the parse stage instead."
    ),
    span: float = typer.Option(96.0, help="Span in feet."),
    w: float = typer.Option(
        -2.0, help="Deck line load in kip/ft; negative acts downward."
    ),
    level: str = typer.Option("bottom", help="Deck chord level, bottom or top."),
    longitudinal: float = typer.Option(
        0.0, help="Concentrated longitudinal force in kip, +X pin toward roller."
    ),
    defl_denom: int = typer.Option(360, help="Deflection-limit denominator (L/n)."),
    depth_limit: float | None = typer.Option(
        None, help="Stated depth limit in feet; omit for none stated."
    ),
) -> None:
    """Run one design through the scoring pipeline and print every stage.

    Args:
        design_json: JSON file with one design dict, or None for the demo
        completion: raw completion text file, or None to skip the parse stage
        span: span in feet
        w: deck line load in kip/ft, negative downward
        level: deck chord level carrying the loads
        longitudinal: concentrated longitudinal force in kip
        defl_denom: deflection-limit denominator
        depth_limit: stated depth limit in feet, or None

    Returns: None
    """
    instance = TrussInstance(
        span_ft=span,
        load_cases=(
            LoadCase(w_kip_per_ft=w, level=level, longitudinal_kip=longitudinal),
        ),
        depth_limit_ft=depth_limit,
        defl_denom=defl_denom,
    )
    print_instance(instance)

    if completion is not None:
        try:
            parsed = parse_design(completion.read_text())
        except NotImplementedError as error:
            print_stage_table((StageStatus("parse", "not_built", str(error)),))
            return
        if isinstance(parsed, ParseFailure):
            print_stage_table((StageStatus("parse", "failed", parsed.reason),))
            print_breakdown(rung0(parsed.reason))
            return
        parse_status = StageStatus("parse", "ok", "valid TrussDesign extracted")
        design = parsed
    elif design_json is not None:
        loaded = load_design(design_json)
        if isinstance(loaded, ParseFailure):
            print_stage_table((StageStatus("parse", "failed", loaded.reason),))
            print_breakdown(rung0(loaded.reason))
            return
        parse_status = StageStatus("parse", "ok", f"loaded from {design_json}")
        design = loaded
    else:
        parse_status = StageStatus("parse", "skipped", "demo design supplied directly")
        design = DEMO_DESIGN

    print_design(design)
    print_trace(parse_status, run_pipeline(instance, design))


if __name__ == "__main__":
    app()
