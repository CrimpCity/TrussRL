"""OpenSees wall unit tests (outline.md Unit 2).

Asserts hand-calculable member forces on a statically determinate 4-bay
Warren truss (forces are independent of member areas, so the method-of-
joints values are exact), linearity, repeatability, multi-case ordering,
and that every failure mode — exception, bad return code, NaN, timeout,
unknown section, singular geometry — returns a failure flag instead of
raising. Sign conventions: axial positive in tension, displacements in
inches with +Y up.
"""

import math
import signal
import time

import openseespy.opensees as ops
import pytest

from trussRL.expander import Node, TrussGeometry, expand
from trussRL.loads import CaseLoads, NodalLoad
from trussRL.solver import solve_cases, wipe_safely

SPAN_FT = 48.0
N_BAYS = 4
DEPTH_FT = 6.0  # bay 12 ft, half-bay 6 ft -> 45 degree diagonals
P_KIP = 10.0
SQRT2 = math.sqrt(2.0)
SECTIONS_BY_GROUP = {
    "bottom_chord": "HSS6X6X3/8",
    "top_chord": "HSS6X6X3/8",
    "diagonals": "HSS4X4X1/4",
}

# Method of joints for the 4-bay Warren above with 10 kip down at interior
# bottom nodes 1, 2, 3 (reactions 15 kip at nodes 0 and 4). Bottom nodes
# are 0-4, top nodes 5-8; forces keyed by member end-node pair, in kips,
# positive tension.
EXPECTED_AXIALS_KIP = {
    frozenset({0, 1}): 15.0,
    frozenset({1, 2}): 35.0,
    frozenset({2, 3}): 35.0,
    frozenset({3, 4}): 15.0,
    frozenset({5, 6}): -30.0,
    frozenset({6, 7}): -40.0,
    frozenset({7, 8}): -30.0,
    frozenset({0, 5}): -15.0 * SQRT2,
    frozenset({5, 1}): 15.0 * SQRT2,
    frozenset({1, 6}): -5.0 * SQRT2,
    frozenset({6, 2}): 5.0 * SQRT2,
    frozenset({2, 7}): 5.0 * SQRT2,
    frozenset({7, 3}): -5.0 * SQRT2,
    frozenset({3, 8}): 15.0 * SQRT2,
    frozenset({8, 4}): -15.0 * SQRT2,
}


def hand_calc_geometry() -> TrussGeometry:
    """Expand the canonical hand-calculable 4-bay Warren truss.

    Returns:
        TrussGeometry: 48 ft span, 4 bays, 6 ft depth (45 degree diagonals)
    """
    return expand("warren", span_ft=SPAN_FT, n_bays=N_BAYS, depth_ft=DEPTH_FT)


def point_loads(geometry: TrussGeometry, fy_by_node: dict[int, float]) -> CaseLoads:
    """Build a CaseLoads of pure vertical point loads, no self-weight.

    Args:
        geometry: expanded truss geometry
        fy_by_node: node id mapped to its vertical force in kips; nodes
            not listed get explicit zeros

    Returns:
        CaseLoads: one NodalLoad per geometry node, ordered by node id
    """
    return CaseLoads(
        nodal_loads=tuple(
            NodalLoad(node_id=node.id, fx_kip=0.0, fy_kip=fy_by_node.get(node.id, 0.0))
            for node in sorted(geometry.nodes, key=lambda n: n.id)
        )
    )


def interior_gravity_loads(geometry: TrussGeometry, p_kip: float) -> CaseLoads:
    """Point loads of p_kip downward at the interior bottom-chord nodes.

    Args:
        geometry: expanded truss geometry
        p_kip: load magnitude in kips, applied downward

    Returns:
        CaseLoads: loads at bottom nodes 1..n_bays-1, zeros elsewhere
    """
    return point_loads(geometry, {i: -p_kip for i in range(1, N_BAYS)})


def test_hand_calc_member_forces_match_method_of_joints() -> None:
    geometry = hand_calc_geometry()
    outcome = solve_cases(
        geometry, SECTIONS_BY_GROUP, [interior_gravity_loads(geometry, P_KIP)]
    )
    assert outcome.ok, outcome.failure
    (solution,) = outcome.solutions
    assert len(solution.member_axials_kip) == len(EXPECTED_AXIALS_KIP)
    for member in geometry.members:
        expected = EXPECTED_AXIALS_KIP[frozenset({member.node_i, member.node_j})]
        assert solution.member_axials_kip[member.id] == pytest.approx(
            expected, abs=1e-9
        ), f"member {member.id} ({member.group})"


def test_hand_calc_group_signs() -> None:
    geometry = hand_calc_geometry()
    outcome = solve_cases(
        geometry, SECTIONS_BY_GROUP, [interior_gravity_loads(geometry, P_KIP)]
    )
    assert outcome.ok, outcome.failure
    (solution,) = outcome.solutions
    for member in geometry.members:
        axial = solution.member_axials_kip[member.id]
        if member.group == "bottom_chord":
            assert axial > 0.0, "bottom chord must be in tension under gravity"
        if member.group == "top_chord":
            assert axial < 0.0, "top chord must be in compression under gravity"


def test_hand_calc_displacements() -> None:
    geometry = hand_calc_geometry()
    outcome = solve_cases(
        geometry, SECTIONS_BY_GROUP, [interior_gravity_loads(geometry, P_KIP)]
    )
    assert outcome.ok, outcome.failure
    (solution,) = outcome.solutions
    pin_dx, pin_dy = solution.node_displacements_in[geometry.pin_node]
    assert pin_dx == pytest.approx(0.0, abs=1e-12)
    assert pin_dy == pytest.approx(0.0, abs=1e-12)
    roller_dx, roller_dy = solution.node_displacements_in[geometry.roller_node]
    assert roller_dy == pytest.approx(0.0, abs=1e-12)
    assert roller_dx > 0.0, "roller must slide toward +X as the truss stretches"
    midspan_dy = solution.node_displacements_in[N_BAYS // 2][1]
    assert midspan_dy < 0.0, "midspan must deflect downward under gravity"


def test_linearity_doubling_loads_doubles_response() -> None:
    geometry = hand_calc_geometry()
    outcome = solve_cases(
        geometry,
        SECTIONS_BY_GROUP,
        [
            interior_gravity_loads(geometry, P_KIP),
            interior_gravity_loads(geometry, 2.0 * P_KIP),
        ],
    )
    assert outcome.ok, outcome.failure
    single, doubled = outcome.solutions
    for member_id, axial in single.member_axials_kip.items():
        assert doubled.member_axials_kip[member_id] == pytest.approx(
            2.0 * axial, rel=1e-12, abs=1e-12
        )
    for node_id, (dx, dy) in single.node_displacements_in.items():
        doubled_dx, doubled_dy = doubled.node_displacements_in[node_id]
        assert doubled_dx == pytest.approx(2.0 * dx, rel=1e-12, abs=1e-12)
        assert doubled_dy == pytest.approx(2.0 * dy, rel=1e-12, abs=1e-12)


def test_repeat_solves_are_identical() -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]
    first = solve_cases(geometry, SECTIONS_BY_GROUP, loads)
    second = solve_cases(geometry, SECTIONS_BY_GROUP, loads)
    assert first.ok and second.ok
    assert first.solutions[0].member_axials_kip == second.solutions[0].member_axials_kip
    assert (
        first.solutions[0].node_displacements_in
        == second.solutions[0].node_displacements_in
    )


def test_multi_case_solutions_keep_input_order() -> None:
    geometry = hand_calc_geometry()
    left_only = point_loads(geometry, {1: -P_KIP})
    right_only = point_loads(geometry, {3: -P_KIP})
    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, [left_only, right_only])
    assert outcome.ok, outcome.failure
    left_solution, right_solution = outcome.solutions
    # The load sits over node 1 in case 0 and node 3 in case 1, so each
    # case deflects its own loaded node the most.
    left_dy = left_solution.node_displacements_in[1][1]
    right_dy = right_solution.node_displacements_in[1][1]
    assert left_dy < right_dy < 0.0


def test_empty_case_list_is_ok_with_no_solutions() -> None:
    geometry = hand_calc_geometry()
    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, [])
    assert outcome.ok
    assert outcome.solutions == ()


def test_analysis_exception_returns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]

    def exploding_analyze(steps: int) -> int:
        raise RuntimeError("boom from openseespy")

    monkeypatch.setattr(ops, "analyze", exploding_analyze)
    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, loads)
    assert outcome.ok is False
    assert outcome.solutions == ()
    assert "boom from openseespy" in str(outcome.failure)


def test_nonzero_return_code_returns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]
    monkeypatch.setattr(ops, "analyze", lambda steps: -3)
    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, loads)
    assert outcome.ok is False
    assert "analysis failed" in str(outcome.failure)


def test_nan_results_return_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]
    monkeypatch.setattr(ops, "nodeDisp", lambda tag: [math.nan, math.nan])
    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, loads)
    assert outcome.ok is False
    assert outcome.failure == "nan in results"


def test_timeout_returns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]

    def hanging_analyze(steps: int) -> int:
        time.sleep(5.0)
        return 0

    monkeypatch.setattr(ops, "analyze", hanging_analyze)
    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, loads, timeout_s=0.05)
    assert outcome.ok is False
    assert outcome.failure == "timeout"


def test_timer_and_handler_restored_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]
    handler_before = signal.getsignal(signal.SIGALRM)

    def hanging_analyze(steps: int) -> int:
        time.sleep(5.0)
        return 0

    with monkeypatch.context() as patched:
        patched.setattr(ops, "analyze", hanging_analyze)
        assert solve_cases(geometry, SECTIONS_BY_GROUP, loads, timeout_s=0.05).ok is False
    assert signal.getsignal(signal.SIGALRM) == handler_before
    # A normal solve right after the timeout must succeed untouched.
    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, loads)
    assert outcome.ok, outcome.failure


@pytest.mark.parametrize("bad_timeout", [0.0, -1.0, math.inf, math.nan])
def test_invalid_timeout_raises(bad_timeout: float) -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]
    with pytest.raises(ValueError, match="timeout_s"):
        solve_cases(geometry, SECTIONS_BY_GROUP, loads, timeout_s=bad_timeout)


def test_caller_timer_and_handler_survive_a_solve() -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]

    def caller_handler(signum: int, frame: object) -> None:
        raise AssertionError("caller alarm must not fire during this test")

    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, caller_handler)
    signal.setitimer(signal.ITIMER_REAL, 60.0)
    try:
        outcome = solve_cases(geometry, SECTIONS_BY_GROUP, loads)
        assert outcome.ok, outcome.failure
        assert signal.getsignal(signal.SIGALRM) is caller_handler
        remaining, _interval = signal.getitimer(signal.ITIMER_REAL)
        assert remaining > 0.0, "caller's armed timer must be re-armed after a solve"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, original_handler)


def test_wipe_safely_swallows_wipe_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def exploding_wipe() -> None:
        raise RuntimeError("wipe blew up")

    monkeypatch.setattr(ops, "wipe", exploding_wipe)
    wipe_safely()  # must not raise


def test_unknown_section_returns_failure() -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]
    sections = dict(SECTIONS_BY_GROUP, diagonals="HSS999X999X9/9")
    outcome = solve_cases(geometry, sections, loads)
    assert outcome.ok is False
    assert outcome.failure


def test_missing_group_section_returns_failure() -> None:
    geometry = hand_calc_geometry()
    loads = [interior_gravity_loads(geometry, P_KIP)]
    sections = {"bottom_chord": "HSS6X6X3/8", "top_chord": "HSS6X6X3/8"}
    outcome = solve_cases(geometry, sections, loads)
    assert outcome.ok is False
    assert "diagonals" in str(outcome.failure)


def test_singular_geometry_returns_failure() -> None:
    # A node with no members attached leaves zero-stiffness DOFs, so the
    # stiffness matrix is singular; the wall must flag it, not raise.
    base = hand_calc_geometry()
    floating = Node(id=base.n_nodes, x_ft=10.0, y_ft=10.0)
    geometry = TrussGeometry(
        nodes=base.nodes + (floating,),
        members=base.members,
        pin_node=base.pin_node,
        roller_node=base.roller_node,
        bay_width_ft=base.bay_width_ft,
    )
    outcome = solve_cases(
        geometry, SECTIONS_BY_GROUP, [interior_gravity_loads(geometry, P_KIP)]
    )
    assert outcome.ok is False
    assert outcome.solutions == ()
