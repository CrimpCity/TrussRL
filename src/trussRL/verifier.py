"""The product: one pure function from completion text to a score.

score() composes parse, checks, solve, and reward into the single entry
point that training, calibration, baselines, and evaluation share: fast,
deterministic, milliseconds per design, importable with no RL dependencies.
run_pipeline() exposes the same chain stage by stage so partially built
pipelines can be exercised while units land.
"""

from dataclasses import dataclass

from trussRL.capacity import MemberCapacity, member_capacities
from trussRL.drc import DRCResult, run_drc
from trussRL.expander import TrussGeometry, expand
from trussRL.instance import TrussInstance
from trussRL.loads import CaseLoads, build_load_cases
from trussRL.reward import RewardBreakdown, grade, rung0, rung1, rung2
from trussRL.schema import ParseFailure, TrussDesign, parse_design
from trussRL.solver import SolveOutcome, solve_cases

STAGE_ORDER = ("drc", "expand", "loads", "solve", "capacity", "reward")


@dataclass(frozen=True)
class StageStatus:
    """How one pipeline stage ended for one design.

    Attributes:
        stage: stage name, one of "drc", "expand", "loads", "solve",
            "capacity", or "reward"
        state: outcome — "ok" (ran and succeeded), "failed" (ran and the
            design or solve failed), "not_built" (stage body raises
            NotImplementedError), or "skipped" (an input was unavailable
            or an earlier stage already decided the score)
        note: human-readable detail, e.g. a failure reason or a one-line
            output summary
    """

    stage: str
    state: str
    note: str = ""


@dataclass(frozen=True)
class PipelineTrace:
    """Every intermediate artifact from one design's trip through the chain.

    Fields after ``stages`` are populated as far as the pipeline reached;
    a None field means its stage did not run or did not succeed.

    Attributes:
        instance: the problem instance being graded against
        stages: one StageStatus per stage, in fixed STAGE_ORDER
        design: the validated design that entered the pipeline
        drc_result: design-rule check outcome, or None
        geometry: expanded truss geometry, or None
        case_loads: resolved nodal loads per case, or None
        solve_outcome: solver output, or None
        capacities: per-member AISC capacities, or None
        breakdown: the final reward, or None when unbuilt stages prevented
            grading
    """

    instance: TrussInstance
    stages: tuple[StageStatus, ...]
    design: TrussDesign | None = None
    drc_result: DRCResult | None = None
    geometry: TrussGeometry | None = None
    case_loads: tuple[CaseLoads, ...] | None = None
    solve_outcome: SolveOutcome | None = None
    capacities: tuple[MemberCapacity, ...] | None = None
    breakdown: RewardBreakdown | None = None


def run_pipeline(instance: TrussInstance, design: TrussDesign) -> PipelineTrace:
    """Run one validated design through every post-parse pipeline stage.

    Linear stage runner in fixed order: drc -> expand -> loads -> solve ->
    capacity -> reward. Each stage body is wrapped so an unbuilt stage
    (NotImplementedError) is recorded and later stages still run when their
    inputs exist; a real DRC or solver failure decides the score (rung 1 or
    2) and skips the rest.

    Assumptions:
        1. DRC runs first and expands geometry internally per
           docs/plans/drc-layer.md, so a passing DRC guarantees expand()
           succeeds; while DRC is unbuilt, expand and loads are guarded by
           try/except ValueError and record "failed" instead of raising.
        2. Never raises on model-chosen design content; NotImplementedError
           from unbuilt stages is recorded, not propagated, so driver code
           can print partial traces during construction.

    Args:
        instance: the trusted problem instance
        design: the schema-validated design to run

    Returns:
        PipelineTrace: per-stage statuses plus every intermediate artifact
            the pipeline produced; breakdown is set only when the chain
            reached a scoring exit
    """
    stages: list[StageStatus] = []
    sections = design.sections_by_group()
    breakdown: RewardBreakdown | None = None

    drc_result: DRCResult | None = None
    try:
        drc_result = run_drc(
            design.truss_type,
            instance.span_ft,
            design.n_bays,
            design.depth_ft,
            sections,
            instance.depth_limit_ft,
        )
        if drc_result.passed:
            stages.append(StageStatus("drc", "ok", "pass"))
        else:
            stages.append(StageStatus("drc", "failed", drc_result.summary))
            breakdown = rung1(drc_result)
    except NotImplementedError as error:
        stages.append(StageStatus("drc", "not_built", str(error)))

    geometry: TrussGeometry | None = None
    if breakdown is not None:
        stages.append(StageStatus("expand", "skipped", "drc failed"))
    else:
        try:
            geometry = expand(
                design.truss_type, instance.span_ft, design.n_bays, design.depth_ft
            )
            stages.append(
                StageStatus(
                    "expand",
                    "ok",
                    f"{geometry.n_nodes} nodes, {len(geometry.members)} members, "
                    f"bay {geometry.bay_width_ft:g} ft",
                )
            )
        except ValueError as error:
            stages.append(StageStatus("expand", "failed", str(error)))

    case_loads: tuple[CaseLoads, ...] | None = None
    if breakdown is not None or geometry is None:
        stages.append(StageStatus("loads", "skipped", "needs expanded geometry"))
    else:
        try:
            case_loads = build_load_cases(geometry, sections, instance.load_cases)
            total_fy = sum(case.total_fy_kip for case in case_loads)
            stages.append(
                StageStatus(
                    "loads",
                    "ok",
                    f"{len(case_loads)} case(s), sum Fy over cases {total_fy:.1f} kip",
                )
            )
        except ValueError as error:
            stages.append(StageStatus("loads", "failed", str(error)))

    solve_outcome: SolveOutcome | None = None
    if breakdown is not None or geometry is None or case_loads is None:
        stages.append(StageStatus("solve", "skipped", "needs geometry and loads"))
    else:
        try:
            solve_outcome = solve_cases(geometry, sections, case_loads)
            if solve_outcome.ok:
                stages.append(
                    StageStatus(
                        "solve", "ok", f"{len(solve_outcome.solutions)} case(s) solved"
                    )
                )
            else:
                stages.append(
                    StageStatus("solve", "failed", str(solve_outcome.failure))
                )
                breakdown = rung2(str(solve_outcome.failure))
        except NotImplementedError as error:
            stages.append(StageStatus("solve", "not_built", str(error)))

    capacities: tuple[MemberCapacity, ...] | None = None
    if breakdown is not None or geometry is None:
        stages.append(StageStatus("capacity", "skipped", "needs geometry"))
    else:
        try:
            capacities = member_capacities(geometry, sections)
            stages.append(
                StageStatus("capacity", "ok", f"{len(capacities)} member capacities")
            )
        except NotImplementedError as error:
            stages.append(StageStatus("capacity", "not_built", str(error)))
        except ValueError as error:
            stages.append(StageStatus("capacity", "failed", str(error)))

    if breakdown is not None:
        stages.append(StageStatus("reward", "skipped", "score already decided"))
    elif (
        geometry is None
        or solve_outcome is None
        or not solve_outcome.ok
        or capacities is None
    ):
        stages.append(StageStatus("reward", "skipped", "needs solve and capacity"))
    else:
        try:
            breakdown = grade(instance, geometry, solve_outcome, capacities, sections)
            stages.append(
                StageStatus(
                    "reward",
                    "ok",
                    f"rung {breakdown.rung}, score {breakdown.score:.3f}",
                )
            )
        except NotImplementedError as error:
            stages.append(StageStatus("reward", "not_built", str(error)))

    return PipelineTrace(
        instance=instance,
        stages=tuple(stages),
        design=design,
        drc_result=drc_result,
        geometry=geometry,
        case_loads=case_loads,
        solve_outcome=solve_outcome,
        capacities=capacities,
        breakdown=breakdown,
    )


def score(instance: TrussInstance, completion_text: str) -> RewardBreakdown:
    """Score one completion against one instance: the product entry point.

    parse -> run_pipeline -> breakdown. A ParseFailure exits at rung 0; a
    DRC failure at rung 1; a solver failure at rung 2; a solved design is
    graded on rung 3.

    Assumptions:
        1. A stub on the scoring path is a harness bug, never a scoreable
           outcome: if unbuilt stages prevented grading, this raises
           NotImplementedError naming them instead of inventing a score.

    Args:
        instance: the trusted problem instance
        completion_text: the policy's raw completion text

    Returns:
        RewardBreakdown: the final reward with per-rung detail

    Raises:
        NotImplementedError: if any stage needed for the score is not
            built yet.
    """
    parsed = parse_design(completion_text)
    if isinstance(parsed, ParseFailure):
        return rung0(parsed.reason)
    trace = run_pipeline(instance, parsed)
    if trace.breakdown is None:
        pending = ", ".join(
            f"{status.stage} ({status.state})"
            for status in trace.stages
            if status.state in ("not_built", "failed")
        )
        raise NotImplementedError(f"cannot score yet; incomplete stages: {pending}")
    return trace.breakdown
