"""Score a design: the reward ladder and its graded top rung.

All scoring policy lives here — the fixed floors for parse, DRC, and solver
failures, and the graded strength, buckling, deflection, and cost terms —
so the rest of the pipeline reports outcomes without deciding what they are
worth. Cost credit is gated by feasibility: cheap-and-broken can never beat
expensive-and-safe.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from trussRL.capacity import MemberCapacity
from trussRL.drc import DRCResult
from trussRL.expander import TrussGeometry
from trussRL.instance import TrussInstance
from trussRL.solver import SolveOutcome

RUNG0_SCORE = 0.0
RUNG1_SCORE = 0.10
RUNG2_SCORE = 0.15
RUNG3_FLOOR = 0.20
COST_PER_NODE_USD = 150.0


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


def grade(
    instance: TrussInstance,
    geometry: TrussGeometry,
    solve_outcome: SolveOutcome,
    capacities: Sequence[MemberCapacity],
) -> RewardBreakdown:
    """Grade a solved design: rung 3, 0.20 plus the graded terms.

    Assumptions:
        1. Envelope semantics: worst tension and worst compression are
           taken per member across all load cases — the same member gets
           two checks, possibly governed by different cases.
        2. Cost credit is gated by feasibility so "cheap and broken" can
           never tie with "expensive and safe".

    Args:
        instance: the problem instance, providing defl_denom and
            cost_ref_usd
        geometry: expanded truss geometry, providing member lengths and
            the node count for the per-connection cost
        solve_outcome: successful solver output with per-case member
            axials and nodal displacements
        capacities: per-member tension and compression capacities, ordered
            by member id

    Returns:
        RewardBreakdown: score 0.20 + 0.80 * (weighted sat() terms and
            gated cost term), capped at 1.0, with all rung-3 detail fields
            populated
    """
    raise NotImplementedError(
        "Unit 5 (outline.md): envelope, sat() clamps, deflection, cost"
    )
