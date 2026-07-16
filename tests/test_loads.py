"""Load conversion unit tests (starting.md Module 3 "output to verify").

Asserts the conservation identity — total applied vertical load equals
w x span plus the summed member self-weights exactly — plus tributary
end-node halves, uplift sign flips, longitudinal splits, self-weight
lumping, output structure, and failure modes. Sign convention: downward
is negative, so "2.0 kip/ft downward" is w_kip_per_ft = -2.0.
"""

import pytest

from trussRL.catalog import get_section
from trussRL.expander import TrussGeometry, expand_warren
from trussRL.loads import (
    LoadCase,
    build_case_loads,
    build_load_cases,
    level_node_ids,
    line_load_forces,
    longitudinal_forces,
    self_weight_forces,
    tributary_widths_ft,
)

BAY_SWEEP = list(range(4, 25))
SECTIONS_BY_GROUP = {
    "bottom_chord": "HSS6X6X3/8",
    "top_chord": "HSS6X6X3/8",
    "diagonals": "HSS4X4X1/4",
}


def total_member_weight_kip(geometry: TrussGeometry) -> float:
    """Sum every member's self-weight in kips, computed independently.

    Args:
        geometry: expanded truss geometry

    Returns:
        float: total of W(lb/ft) x length_ft over all members, in kips
    """
    total_lb = sum(
        get_section(SECTIONS_BY_GROUP[member.group]).weight_lb_per_ft * member.length_ft
        for member in geometry.members
    )
    return total_lb / 1000.0


@pytest.mark.parametrize("n_bays", BAY_SWEEP)
@pytest.mark.parametrize("level", ["bottom", "top"])
@pytest.mark.parametrize("w_kip_per_ft", [-2.0, 1.2])
def test_vertical_load_conservation(
    n_bays: int, level: str, w_kip_per_ft: float
) -> None:
    span_ft = 96.0
    geometry = expand_warren(span_ft=span_ft, n_bays=n_bays, depth_ft=8.0)
    case = LoadCase(w_kip_per_ft=w_kip_per_ft, level=level)
    (loads,) = build_load_cases(geometry, SECTIONS_BY_GROUP, [case])
    expected = w_kip_per_ft * span_ft - total_member_weight_kip(geometry)
    assert loads.total_fy_kip == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("n_bays", BAY_SWEEP)
@pytest.mark.parametrize("level", ["bottom", "top"])
def test_tributary_widths_sum_to_span(n_bays: int, level: str) -> None:
    span_ft = 96.0
    geometry = expand_warren(span_ft=span_ft, n_bays=n_bays, depth_ft=8.0)
    widths = tributary_widths_ft(geometry, level)
    assert sum(widths.values()) == pytest.approx(span_ft, abs=1e-9)
    expected_count = n_bays + 1 if level == "bottom" else n_bays
    assert len(widths) == expected_count


def test_bottom_chord_end_nodes_carry_half() -> None:
    # 96 ft / 8 bays -> bay width 12 ft; w = -2.0 kip/ft downward gives
    # -12.0 kip at the aligned end nodes and -24.0 kip at interior nodes.
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    forces = line_load_forces(geometry, w_kip_per_ft=-2.0, level="bottom")
    assert forces[0] == pytest.approx(-12.0, abs=1e-9)
    assert forces[8] == pytest.approx(-12.0, abs=1e-9)
    for node_id in range(1, 8):
        assert forces[node_id] == pytest.approx(-24.0, abs=1e-9)


def test_offset_top_chord_nodes_carry_full_bay() -> None:
    # The Warren top chord is offset half a bay, so every top node absorbs
    # a full bay width including the overhang out to the span ends.
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    forces = line_load_forces(geometry, w_kip_per_ft=-2.0, level="top")
    top_ids = level_node_ids(geometry, "top")
    assert len(forces) == 8
    for node_id in top_ids:
        assert forces[node_id] == pytest.approx(-24.0, abs=1e-9)


@pytest.mark.parametrize("level", ["bottom", "top"])
def test_uplift_is_exact_negation_of_gravity(level: str) -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    gravity = line_load_forces(geometry, w_kip_per_ft=-1.2, level=level)
    uplift = line_load_forces(geometry, w_kip_per_ft=1.2, level=level)
    assert set(gravity) == set(uplift)
    for node_id, force in gravity.items():
        assert uplift[node_id] == -force


def test_longitudinal_force_split_equally() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    forces = longitudinal_forces(geometry, force_kip=15.0, level="bottom")
    assert len(forces) == 9
    for force in forces.values():
        assert force == pytest.approx(15.0 / 9.0, abs=1e-12)
    assert sum(forces.values()) == pytest.approx(15.0, abs=1e-9)


def test_longitudinal_sign_flip_is_exact_negation() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    forward = longitudinal_forces(geometry, force_kip=15.0, level="bottom")
    backward = longitudinal_forces(geometry, force_kip=-15.0, level="bottom")
    for node_id, force in forward.items():
        assert backward[node_id] == -force


def test_longitudinal_other_level_nodes_unloaded() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    case = LoadCase(longitudinal_kip=15.0, level="bottom")
    loads = build_case_loads(geometry, case, self_weight={})
    top_ids = set(level_node_ids(geometry, "top"))
    for nodal in loads.nodal_loads:
        if nodal.node_id in top_ids:
            assert nodal.fx_kip == 0.0
    assert loads.total_fx_kip == pytest.approx(15.0, abs=1e-9)


@pytest.mark.parametrize("n_bays", BAY_SWEEP)
def test_self_weight_all_downward_and_conserved(n_bays: int) -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=n_bays, depth_ft=8.0)
    forces = self_weight_forces(geometry, SECTIONS_BY_GROUP)
    assert len(forces) == geometry.n_nodes
    for force in forces.values():
        assert force <= 0.0
    expected_total = -total_member_weight_kip(geometry)
    assert sum(forces.values()) == pytest.approx(expected_total, abs=1e-9)


def test_self_weight_hand_check_support_node() -> None:
    # 96 ft / 8 bays / 8 ft deep: node 0 touches one 12 ft bottom-chord
    # member (HSS6X6X3/8, 27.48 lb/ft) and one 6-8-10 diagonal of exactly
    # 10 ft (HSS4X4X1/4, 12.21 lb/ft); it takes half of each weight.
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    forces = self_weight_forces(geometry, SECTIONS_BY_GROUP)
    expected = -(27.48 * 12.0 + 12.21 * 10.0) / 2.0 / 1000.0
    assert forces[0] == pytest.approx(expected, abs=1e-9)


def test_self_weight_identical_across_gravity_and_uplift() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    cases = [LoadCase(w_kip_per_ft=-2.0), LoadCase(w_kip_per_ft=1.2)]
    gravity, uplift = build_load_cases(geometry, SECTIONS_BY_GROUP, cases)
    self_weight = self_weight_forces(geometry, SECTIONS_BY_GROUP)
    line_gravity = line_load_forces(geometry, -2.0, "bottom")
    line_uplift = line_load_forces(geometry, 1.2, "bottom")
    for nodal_gravity, nodal_uplift in zip(gravity.nodal_loads, uplift.nodal_loads):
        node_id = nodal_gravity.node_id
        extracted_gravity = nodal_gravity.fy_kip - line_gravity.get(node_id, 0.0)
        extracted_uplift = nodal_uplift.fy_kip - line_uplift.get(node_id, 0.0)
        assert extracted_gravity == pytest.approx(self_weight[node_id], abs=1e-12)
        assert extracted_uplift == pytest.approx(self_weight[node_id], abs=1e-12)


def test_case_loads_structure() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    cases = [
        LoadCase(w_kip_per_ft=-2.0),
        LoadCase(w_kip_per_ft=1.2),
        LoadCase(longitudinal_kip=15.0),
    ]
    results = build_load_cases(geometry, SECTIONS_BY_GROUP, cases)
    assert len(results) == len(cases)
    self_weight = self_weight_forces(geometry, SECTIONS_BY_GROUP)
    for case, loads in zip(cases, results):
        node_ids = [nodal.node_id for nodal in loads.nodal_loads]
        assert node_ids == list(range(geometry.n_nodes))
        assert loads == build_case_loads(geometry, case, self_weight)


def test_empty_cases_return_empty_tuple() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    assert build_load_cases(geometry, SECTIONS_BY_GROUP, []) == ()


def test_build_load_cases_deterministic() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    cases = [LoadCase(w_kip_per_ft=-2.0), LoadCase(w_kip_per_ft=1.2)]
    first = build_load_cases(geometry, SECTIONS_BY_GROUP, cases)
    second = build_load_cases(geometry, SECTIONS_BY_GROUP, cases)
    assert first == second


def test_bad_level_raises() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    with pytest.raises(ValueError, match="unsupported level"):
        level_node_ids(geometry, "deck")
    case = LoadCase(w_kip_per_ft=-2.0, level="middle")
    with pytest.raises(ValueError, match="unsupported level"):
        build_case_loads(geometry, case, self_weight={})


def test_missing_group_raises() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    incomplete = {"bottom_chord": "HSS6X6X3/8", "top_chord": "HSS6X6X3/8"}
    with pytest.raises(ValueError, match="no section assigned"):
        self_weight_forces(geometry, incomplete)


def test_unknown_designation_raises() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    bad = dict(SECTIONS_BY_GROUP, diagonals="HSS99X99X9")
    with pytest.raises(ValueError, match="unknown section designation"):
        self_weight_forces(geometry, bad)


def test_unused_group_key_ignored() -> None:
    # Warren has no verticals; a "verticals" entry in the mapping is ignored rather
    # than rejected.
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    with_verticals = dict(SECTIONS_BY_GROUP, verticals="HSS4X4X1/4")
    assert self_weight_forces(geometry, with_verticals) == self_weight_forces(
        geometry, SECTIONS_BY_GROUP
    )
