"""Score a design: the reward ladder and its graded top rung.

All scoring policy lives here — the fixed floors for parse, DRC, and solver
failures, and the graded strength, buckling, deflection, and cost terms —
so the rest of the pipeline reports outcomes without deciding what they are
worth. Cost credit is gated by feasibility: cheap-and-broken can never beat
expensive-and-safe.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trussRL.capacity import MemberCapacity
from trussRL.catalog import get_section
from trussRL.drc import DRCResult
from trussRL.expander import TrussGeometry
from trussRL.instance import TrussInstance
from trussRL.solver import SolveOutcome

RUNG0_SCORE = 0.0
RUNG1_SCORE = 0.10
RUNG2_SCORE = 0.15
RUNG3_FLOOR = 0.20
RUNG3_RANGE = 0.80
COST_PER_NODE_USD = 150.0
INCHES_PER_FOOT = 12.0
OVERAGE_SPAN = 0.5
W_STRENGTH = 0.25
W_BUCKLING = 0.25
W_DEFL = 0.20
W_COST = 0.30


@dataclass(frozen=True)
class RewardBreakdown:
    """The verifier's full output: final score plus per-rung detail.

    Attributes:
        score: final reward in [0.0, 1.0]
        rung: reward ladder rung reached, 0 through 3
        reason: how the design exited the ladder, e.g. "parse failure: ...",
            "drc: ...", "solver: timeout", or "solved"
        u_strength: worst tension-side utilization over members and cases,
            or None below rung 3
        u_buckling: worst compression-side utilization over members and
            cases, or None below rung 3
        u_defl: worst deflection utilization over gravity cases, or None
            below rung 3
        cost_total_usd: member cost plus per-node connection cost, or None
            below rung 3
        feasible: min of the three sat() terms in [0, 1], gating the cost
            credit, or None below rung 3
    """

    score: float
    rung: int
    reason: str
    u_strength: float | None = None
    u_buckling: float | None = None
    u_defl: float | None = None
    cost_total_usd: float | None = None
    feasible: float | None = None


def rung0(reason: str) -> RewardBreakdown:
    """Build the breakdown for a completion that failed to parse.

    Args:
        reason: the parse failure reason

    Returns:
        RewardBreakdown: score 0.0 at rung 0
    """
    return RewardBreakdown(score=RUNG0_SCORE, rung=0, reason=f"parse failure: {reason}")


def rung1(drc: DRCResult) -> RewardBreakdown:
    """Build the breakdown for a design that failed a design-rule check.

    Args:
        drc: the failing DRC result

    Returns:
        RewardBreakdown: score 0.10 at rung 1, with the DRC summary as
            the reason
    """
    return RewardBreakdown(score=RUNG1_SCORE, rung=1, reason=f"drc: {drc.summary}")


def rung2(reason: str) -> RewardBreakdown:
    """Build the breakdown for a design the solver could not score.

    Args:
        reason: the solver failure flag (crash, NaN, timeout, ...)

    Returns:
        RewardBreakdown: score 0.15 at rung 2
    """
    return RewardBreakdown(score=RUNG2_SCORE, rung=2, reason=f"solver: {reason}")


def sat(u: float) -> float:
    """Clamp a utilization into linear graded credit.

    Full credit at or below 1.0, then a linear ramp to zero credit at
    1.0 + OVERAGE_SPAN (50% overage), so 5% overstressed outranks 40%
    overstressed instead of both falling off a cliff.

    Args:
        u: utilization, demand over capacity, nonnegative

    Returns:
        float: credit in [0.0, 1.0]
    """
    if u <= 1.0:
        return 1.0
    return max(0.0, 1.0 - (u - 1.0) / OVERAGE_SPAN)


def envelope_utilizations(
    solve_outcome: SolveOutcome, capacities: Sequence[MemberCapacity]
) -> tuple[float, float]:
    """Worst tension and worst compression utilization over members and cases.

    Assumptions:
        1. Envelope semantics: the max runs over every (member, case) pair,
           so the same member gets a tension check and a compression check,
           possibly governed by different cases.
        2. U_strength is the tension side only; compression capacity per
           Chapter E3 already caps at yield, so the compression side of
           strength is covered by the buckling term.

    Args:
        solve_outcome: successful solver output with per-case member axial
            forces, positive in tension
        capacities: per-member tension and compression capacities

    Returns:
        tuple: (u_strength, u_buckling), each the worst utilization over
            all (member, case) pairs, or 0.0 when no demand of that sign
            exists anywhere
    """
    capacity_by_id = {capacity.member_id: capacity for capacity in capacities}
    u_strength = 0.0
    u_buckling = 0.0
    for solution in solve_outcome.solutions:
        for member_id, axial_kip in solution.member_axials_kip.items():
            capacity = capacity_by_id[member_id]
            if axial_kip > 0.0:
                u_strength = max(u_strength, axial_kip / capacity.tension_kip)
            elif axial_kip < 0.0:
                u_buckling = max(u_buckling, -axial_kip / capacity.compression_kip)
    return u_strength, u_buckling


def deflection_utilization(
    instance: TrussInstance, solve_outcome: SolveOutcome
) -> float:
    """Worst deflection utilization over the instance's gravity cases.

    Assumptions:
        1. Load cases pair with solve_outcome.solutions by index, in the
           order the solver ran them.
        2. Gravity cases are those with a downward line load
           (w_kip_per_ft < 0.0); with no gravity case, no deflection
           constraint applies and the utilization is 0.0.

    Args:
        instance: the problem instance, providing span_ft and defl_denom
        solve_outcome: successful solver output with per-case nodal
            displacements in inches

    Returns:
        float: max over gravity cases of max |dy| over nodes divided by
            the span / defl_denom limit, or 0.0 with no gravity case
    """
    limit_in = instance.span_ft * INCHES_PER_FOOT / instance.defl_denom
    u_defl = 0.0
    for load_case, solution in zip(
        instance.load_cases, solve_outcome.solutions, strict=True
    ):
        if load_case.w_kip_per_ft >= 0.0:
            continue
        delta_max_in = max(abs(dy) for _, dy in solution.node_displacements_in.values())
        u_defl = max(u_defl, delta_max_in / limit_in)
    return u_defl


def design_cost_usd(
    geometry: TrussGeometry, sections_by_group: Mapping[str, str]
) -> float:
    """Total design cost: member material plus per-node connection cost.

    Args:
        geometry: expanded truss geometry, providing member lengths,
            member groups, and the node count
        sections_by_group: member group name mapped to its section
            designation

    Returns:
        float: sum of length x cost_per_ft over members plus
            COST_PER_NODE_USD x n_nodes, in dollars
    """
    member_cost = sum(
        member.length_ft * get_section(sections_by_group[member.group]).cost_per_ft_usd
        for member in geometry.members
    )
    return member_cost + COST_PER_NODE_USD * geometry.n_nodes


def validate_cost_ref(cost_ref_usd: float) -> None:
    """Reject a cost reference that cannot anchor the graded cost term.

    Assumptions:
        1. cost_ref_usd comes from the trusted calibration side, never the
           model, so raising here flags a harness bug rather than scoring
           model output.

    Args:
        cost_ref_usd: calibration cost reference in dollars

    Returns:
        None

    Raises:
        ValueError: if cost_ref_usd is not finite or not strictly positive.
    """
    if not math.isfinite(cost_ref_usd) or cost_ref_usd <= 0.0:
        raise ValueError(
            f"cost_ref_usd must be finite and strictly positive, got {cost_ref_usd!r}"
        )


def cost_term(cost_total_usd: float, cost_ref_usd: float | None) -> float:
    """Graded cost credit relative to the calibration cost reference.

    Full credit at zero cost, linearly decaying to zero credit at twice
    the reference cost.

    Assumptions:
        1. A None cost_ref_usd contributes 0.0 rather than raising: Unit
           9's calibration sweep scores designs before cost_ref exists and
           derives it from the recorded costs, so raising would deadlock
           calibration.

    Args:
        cost_total_usd: total design cost in dollars
        cost_ref_usd: calibration cost reference in dollars, or None
            before calibration has set it

    Returns:
        float: cost credit in [0.0, 1.0]

    Raises:
        ValueError: if cost_ref_usd is neither None nor finite and
            strictly positive.
    """
    if cost_ref_usd is None:
        return 0.0
    validate_cost_ref(cost_ref_usd)
    return max(0.0, 1.0 - cost_total_usd / (2.0 * cost_ref_usd))


def rung3_score(
    u_strength: float,
    u_buckling: float,
    u_defl: float,
    cost_total_usd: float,
    cost_ref_usd: float | None,
) -> tuple[float, float]:
    """Compute the rung-3 score and feasibility from the graded inputs.

    The single home of the rung-3 formula: 0.20 plus 0.80 times the
    weighted sat() terms and the feasibility-gated cost term, capped at
    1.0. grade() and regrade() both call this, so pass-1 scoring and
    calibration regrading can never disagree.

    Args:
        u_strength: worst tension-side utilization over members and cases
        u_buckling: worst compression-side utilization over members and
            cases
        u_defl: worst deflection utilization over gravity cases
        cost_total_usd: total design cost in dollars
        cost_ref_usd: calibration cost reference in dollars, or None
            before calibration has set it

    Returns:
        tuple: (score, feasible) — the final rung-3 score in [0.20, 1.0]
            and the min of the three sat() terms gating the cost credit

    Raises:
        ValueError: if cost_ref_usd is neither None nor finite and
            strictly positive.
    """
    feasible = min(sat(u_strength), sat(u_buckling), sat(u_defl))
    graded = (
        W_STRENGTH * sat(u_strength)
        + W_BUCKLING * sat(u_buckling)
        + W_DEFL * sat(u_defl)
        + W_COST * cost_term(cost_total_usd, cost_ref_usd) * feasible
    )
    return min(1.0, RUNG3_FLOOR + RUNG3_RANGE * graded), feasible


def regrade(breakdown: RewardBreakdown, cost_ref_usd: float) -> RewardBreakdown:
    """Rescore a recorded breakdown against a now-known cost reference.

    Unit 9 scores designs before cost_ref exists (cost term 0.0), derives
    cost_ref from the recorded costs, then regrades the same breakdowns so
    the gate distributions match what training will actually see. Only the
    score changes; every recorded utilization and cost is preserved.

    Assumptions:
        1. Rungs 0-2 carry no cost term, so their breakdowns are returned
           unchanged; only rung 3 is recomputed.

    Args:
        breakdown: a breakdown recorded by a pass-1 grade
        cost_ref_usd: calibration cost reference in dollars

    Returns:
        RewardBreakdown: the breakdown with its score recomputed under
            cost_ref_usd; unchanged below rung 3

    Raises:
        ValueError: if cost_ref_usd is not finite and strictly positive,
            or a rung-3 breakdown is missing utilization or cost fields.
    """
    validate_cost_ref(cost_ref_usd)
    if breakdown.rung != 3:
        return breakdown
    if (
        breakdown.u_strength is None
        or breakdown.u_buckling is None
        or breakdown.u_defl is None
        or breakdown.cost_total_usd is None
    ):
        raise ValueError(
            "cannot regrade rung-3 breakdown with missing fields: "
            f"u_strength={breakdown.u_strength!r}, "
            f"u_buckling={breakdown.u_buckling!r}, "
            f"u_defl={breakdown.u_defl!r}, "
            f"cost_total_usd={breakdown.cost_total_usd!r}"
        )
    score, _ = rung3_score(
        breakdown.u_strength,
        breakdown.u_buckling,
        breakdown.u_defl,
        breakdown.cost_total_usd,
        cost_ref_usd,
    )
    return dataclasses.replace(breakdown, score=score)


def grade(
    instance: TrussInstance,
    geometry: TrussGeometry,
    solve_outcome: SolveOutcome,
    capacities: Sequence[MemberCapacity],
    sections_by_group: Mapping[str, str],
) -> RewardBreakdown:
    """Grade a solved design: rung 3, 0.20 plus the graded terms.

    Assumptions:
        1. Envelope semantics: worst tension and worst compression are
           taken per member across all load cases — the same member gets
           two checks, possibly governed by different cases.
        2. Cost credit is gated by feasibility so "cheap and broken" can
           never tie with "expensive and safe".
        3. When instance.cost_ref_usd is None the cost term contributes
           0.0 rather than raising: Unit 9's calibration sweep scores
           designs before cost_ref exists and derives it from the
           cost_total_usd recorded in these breakdowns, so raising would
           deadlock calibration.

    Args:
        instance: the problem instance, providing defl_denom and
            cost_ref_usd
        geometry: expanded truss geometry, providing member lengths and
            the node count for the per-connection cost
        solve_outcome: successful solver output with per-case member
            axials and nodal displacements
        capacities: per-member tension and compression capacities, ordered
            by member id
        sections_by_group: member group name mapped to its section
            designation, used for member cost

    Returns:
        RewardBreakdown: score 0.20 + 0.80 * (weighted sat() terms and
            gated cost term), capped at 1.0, with all rung-3 detail fields
            populated
    """
    u_strength, u_buckling = envelope_utilizations(solve_outcome, capacities)
    u_defl = deflection_utilization(instance, solve_outcome)
    cost_total = design_cost_usd(geometry, sections_by_group)
    score, feasible = rung3_score(
        u_strength, u_buckling, u_defl, cost_total, instance.cost_ref_usd
    )
    return RewardBreakdown(
        score=score,
        rung=3,
        reason="solved",
        u_strength=u_strength,
        u_buckling=u_buckling,
        u_defl=u_defl,
        cost_total_usd=cost_total,
        feasible=feasible,
    )
