"""Unit tests for rung-3 grading: sat clamps, envelopes, deflection, cost.

Fixtures are hand-built tiny geometries, solutions, and capacities — no
OpenSees run — so every graded term can be checked against arithmetic done
by hand. One integration test walks the known-good design through the real
pipeline to confirm the reward stage now lands on rung 3.
"""

import pytest

from trussRL.capacity import MemberCapacity
from trussRL.catalog import get_section
from trussRL.expander import Member, Node, TrussGeometry
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.reward import (
    COST_PER_NODE_USD,
    RUNG3_FLOOR,
    deflection_utilization,
    design_cost_usd,
    envelope_utilizations,
    grade,
    sat,
)
from trussRL.schema import TrussDesign
from trussRL.solver import CaseSolution, SolveOutcome
from trussRL.verifier import run_pipeline

SECTIONS = {"bottom_chord": "HSS6X6X3/8", "diagonals": "HSS4X4X1/4"}
SPAN_FT = 60.0  # L/360 limit = 60 * 12 / 360 = 2.0 in


def make_geometry(scale: float = 1.0) -> TrussGeometry:
    """Build a three-member, three-node triangle geometry.

    Args:
        scale: length multiplier, so cost grows linearly with scale

    Returns:
        TrussGeometry: one bottom chord and two diagonals
    """
    nodes = (
        Node(id=0, x_ft=0.0, y_ft=0.0),
        Node(id=1, x_ft=10.0 * scale, y_ft=0.0),
        Node(id=2, x_ft=5.0 * scale, y_ft=5.0 * scale),
    )
    members = (
        Member(id=0, node_i=0, node_j=1, group="bottom_chord", length_ft=10.0 * scale),
        Member(id=1, node_i=0, node_j=2, group="diagonals", length_ft=7.5 * scale),
        Member(id=2, node_i=1, node_j=2, group="diagonals", length_ft=7.5 * scale),
    )
    return TrussGeometry(
        nodes=nodes,
        members=members,
        pin_node=0,
        roller_node=1,
        bay_width_ft=10.0 * scale,
    )


def make_capacities(
    tension_kip: float = 100.0, compression_kip: float = 50.0
) -> tuple[MemberCapacity, ...]:
    """Build identical capacities for the three fixture members.

    Args:
        tension_kip: design tension capacity for every member
        compression_kip: design compression capacity for every member

    Returns:
        tuple: one MemberCapacity per fixture member id
    """
    return tuple(
        MemberCapacity(
            member_id=member_id,
            tension_kip=tension_kip,
            compression_kip=compression_kip,
        )
        for member_id in range(3)
    )


def make_solution(axials: dict[int, float], dy_in: float = 0.0) -> CaseSolution:
    """Build a case solution with given axials and one displaced node.

    Args:
        axials: member id mapped to axial force in kips
        dy_in: vertical displacement of node 2 in inches, signed

    Returns:
        CaseSolution: fixture solution for one case
    """
    displacements = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, dy_in)}
    return CaseSolution(member_axials_kip=axials, node_displacements_in=displacements)


def make_outcome(*solutions: CaseSolution) -> SolveOutcome:
    """Wrap case solutions in a successful solve outcome.

    Args:
        solutions: one CaseSolution per load case, in case order

    Returns:
        SolveOutcome: successful outcome with the given solutions
    """
    return SolveOutcome(solutions=solutions, failure=None)


def make_instance(
    n_cases: int = 1,
    defl_denom: int = 360,
    cost_ref_usd: float | None = None,
    w_kip_per_ft: float = -2.0,
) -> TrussInstance:
    """Build a fixture instance with identical load cases.

    Args:
        n_cases: number of load cases, matching the solutions fed to grade
        defl_denom: deflection-limit denominator
        cost_ref_usd: calibration cost reference, or None
        w_kip_per_ft: line load for every case; negative is gravity

    Returns:
        TrussInstance: fixture instance spanning SPAN_FT
    """
    return TrussInstance(
        span_ft=SPAN_FT,
        load_cases=tuple(LoadCase(w_kip_per_ft=w_kip_per_ft) for _ in range(n_cases)),
        defl_denom=defl_denom,
        cost_ref_usd=cost_ref_usd,
    )


def test_sat_full_credit_at_or_below_one() -> None:
    assert sat(0.0) == 1.0
    assert sat(0.5) == 1.0
    assert sat(1.0) == 1.0


def test_sat_linear_ramp_above_one() -> None:
    assert sat(1.25) == pytest.approx(0.5)
    assert sat(1.1) == pytest.approx(0.8)


def test_sat_zero_credit_at_fifty_percent_overage() -> None:
    assert sat(1.5) == 0.0
    assert sat(3.0) == 0.0


def test_sat_small_overage_outranks_large_overage() -> None:
    assert sat(1.05) > sat(1.40)


def test_envelope_worst_case_governs_across_cases() -> None:
    case1 = make_solution({0: 80.0, 1: -10.0, 2: 0.0})
    case2 = make_solution({0: 20.0, 1: -60.0, 2: 0.0})
    u_strength, u_buckling = envelope_utilizations(
        make_outcome(case1, case2), make_capacities()
    )
    assert u_strength == pytest.approx(0.8)
    assert u_buckling == pytest.approx(1.2)


def test_same_member_checked_both_ways_from_different_cases() -> None:
    case1 = make_solution({0: 80.0, 1: 0.0, 2: 0.0})
    case2 = make_solution({0: -60.0, 1: 0.0, 2: 0.0})
    u_strength, u_buckling = envelope_utilizations(
        make_outcome(case1, case2), make_capacities()
    )
    assert u_strength == pytest.approx(0.8)
    assert u_buckling == pytest.approx(1.2)


def test_envelope_zero_when_no_demand_of_that_sign() -> None:
    tension_only = make_solution({0: 30.0, 1: 10.0, 2: 0.0})
    u_strength, u_buckling = envelope_utilizations(
        make_outcome(tension_only), make_capacities()
    )
    assert u_strength > 0.0
    assert u_buckling == 0.0

    compression_only = make_solution({0: -30.0, 1: -10.0, 2: 0.0})
    u_strength, u_buckling = envelope_utilizations(
        make_outcome(compression_only), make_capacities()
    )
    assert u_strength == 0.0
    assert u_buckling > 0.0


def test_score_drops_monotonically_with_overstress() -> None:
    instance = make_instance()
    geometry = make_geometry()
    capacities = make_capacities()
    scores = []
    for tension_kip in (100.0, 110.0, 120.0, 130.0):
        outcome = make_outcome(
            make_solution({0: tension_kip, 1: -10.0, 2: -10.0}, dy_in=-1.0)
        )
        scores.append(grade(instance, geometry, outcome, capacities, SECTIONS).score)
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_over_slender_compression_member_buckling_governs() -> None:
    outcome = make_outcome(make_solution({0: 20.0, 1: -10.0, 2: -10.0}, dy_in=-1.0))
    breakdown = grade(
        make_instance(),
        make_geometry(),
        outcome,
        make_capacities(compression_kip=5.0),
        SECTIONS,
    )
    assert breakdown.u_buckling == pytest.approx(2.0)
    assert breakdown.u_strength is not None and breakdown.u_strength < 1.0
    assert breakdown.feasible == 0.0


def test_over_limit_gravity_deflection() -> None:
    outcome = make_outcome(make_solution({0: 0.0, 1: 0.0, 2: 0.0}, dy_in=-3.0))
    u_defl = deflection_utilization(make_instance(), outcome)
    assert u_defl == pytest.approx(1.5)


def test_uplift_only_instance_has_no_deflection_constraint() -> None:
    instance = make_instance(w_kip_per_ft=1.2)
    outcome = make_outcome(make_solution({0: 0.0, 1: 0.0, 2: 0.0}, dy_in=3.0))
    assert deflection_utilization(instance, outcome) == 0.0
    breakdown = grade(instance, make_geometry(), outcome, make_capacities(), SECTIONS)
    assert breakdown.u_defl == 0.0


def test_tighter_defl_denom_lowers_score() -> None:
    outcome = make_outcome(make_solution({0: 20.0, 1: -10.0, 2: -10.0}, dy_in=-2.5))
    geometry = make_geometry()
    capacities = make_capacities()
    score_360 = grade(
        make_instance(defl_denom=360), geometry, outcome, capacities, SECTIONS
    ).score
    score_500 = grade(
        make_instance(defl_denom=500), geometry, outcome, capacities, SECTIONS
    ).score
    assert score_500 < score_360


def test_design_cost_hand_check() -> None:
    chord_cost = get_section("HSS6X6X3/8").cost_per_ft_usd
    diagonal_cost = get_section("HSS4X4X1/4").cost_per_ft_usd
    expected = 10.0 * chord_cost + 2 * 7.5 * diagonal_cost + COST_PER_NODE_USD * 3
    assert design_cost_usd(make_geometry(), SECTIONS) == pytest.approx(expected)


def test_cheap_infeasible_scores_below_expensive_feasible() -> None:
    geometry = make_geometry()
    capacities = make_capacities()
    broken = make_outcome(make_solution({0: 200.0, 1: -100.0, 2: -100.0}, dy_in=-3.5))
    cheap_broken = grade(
        make_instance(cost_ref_usd=1_000_000.0),
        geometry,
        broken,
        capacities,
        SECTIONS,
    )
    safe = make_outcome(make_solution({0: 50.0, 1: -20.0, 2: -20.0}, dy_in=-1.0))
    expensive_safe = grade(
        make_instance(cost_ref_usd=1.0), geometry, safe, capacities, SECTIONS
    )
    assert cheap_broken.feasible == 0.0
    assert cheap_broken.score < expensive_safe.score


def test_higher_cost_lowers_score_monotone() -> None:
    instance = make_instance(cost_ref_usd=design_cost_usd(make_geometry(3.0), SECTIONS))
    outcome = make_outcome(make_solution({0: 50.0, 1: -20.0, 2: -20.0}, dy_in=-1.0))
    capacities = make_capacities()
    scores = [
        grade(instance, make_geometry(scale), outcome, capacities, SECTIONS).score
        for scale in (1.0, 2.0, 3.0)
    ]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_cost_ref_none_contributes_zero_but_records_cost() -> None:
    outcome = make_outcome(make_solution({0: 50.0, 1: -20.0, 2: -20.0}, dy_in=-1.0))
    breakdown = grade(
        make_instance(), make_geometry(), outcome, make_capacities(), SECTIONS
    )
    assert breakdown.cost_total_usd == pytest.approx(
        design_cost_usd(make_geometry(), SECTIONS)
    )
    # feasible design, zero cost credit: 0.20 + 0.80 * (0.25 + 0.25 + 0.20)
    assert breakdown.score == pytest.approx(0.76)


def test_score_floor_is_rung3_floor() -> None:
    broken = make_outcome(make_solution({0: 200.0, 1: -100.0, 2: -100.0}, dy_in=-3.5))
    breakdown = grade(
        make_instance(), make_geometry(), broken, make_capacities(), SECTIONS
    )
    assert breakdown.score == pytest.approx(RUNG3_FLOOR)


def test_perfect_design_caps_at_one() -> None:
    outcome = make_outcome(make_solution({0: 1.0, 1: -1.0, 2: -1.0}, dy_in=-0.01))
    breakdown = grade(
        make_instance(cost_ref_usd=1e12),
        make_geometry(),
        outcome,
        make_capacities(),
        SECTIONS,
    )
    assert breakdown.score <= 1.0
    assert breakdown.score == pytest.approx(1.0)


def test_pipeline_reaches_rung3_on_known_good_design() -> None:
    instance = TrussInstance(
        span_ft=96.0,
        load_cases=(LoadCase(w_kip_per_ft=-2.0, level="bottom"),),
    )
    design = TrussDesign(
        truss_type="warren",
        n_bays=8,
        depth_ft=8.0,
        top_chord="HSS6X6X3/8",
        bottom_chord="HSS6X6X3/8",
        diagonals="HSS4X4X1/4",
    )
    trace = run_pipeline(instance, design)
    breakdown = trace.breakdown
    assert breakdown is not None
    assert breakdown.rung == 3
    assert breakdown.reason == "solved"
    assert breakdown.u_strength is not None
    assert breakdown.u_buckling is not None
    assert breakdown.u_defl is not None
    assert breakdown.cost_total_usd is not None
    assert breakdown.feasible is not None
    assert RUNG3_FLOOR <= breakdown.score <= 1.0
    reward_status = next(s for s in trace.stages if s.stage == "reward")
    assert reward_status.state == "ok"
