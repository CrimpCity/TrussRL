"""Expander unit tests (starting.md Module 2 "output to verify").

Asserts node counts, member counts, and member lengths analytically for
known Warren cases, plus support placement, determinacy, symmetry, and
failure modes.
"""

import math

import pytest

from trussRL.expander import EXPANDERS, TrussGeometry, expand, expand_warren

BAY_SWEEP = list(range(4, 25))


def group_counts(geometry: TrussGeometry) -> dict[str, int]:
    """Count members per group tag.

    Args:
        geometry: expanded truss geometry

    Returns:
        dict: group name mapped to number of members carrying that tag
    """
    counts: dict[str, int] = {}
    for member in geometry.members:
        counts[member.group] = counts.get(member.group, 0) + 1
    return counts


@pytest.mark.parametrize("n_bays", BAY_SWEEP)
def test_node_and_member_count_formulas(n_bays: int) -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=n_bays, depth_ft=8.0)
    assert len(geometry.nodes) == 2 * n_bays + 1
    assert geometry.n_nodes == 2 * n_bays + 1
    assert len(geometry.members) == 4 * n_bays - 1


@pytest.mark.parametrize("n_bays", BAY_SWEEP)
def test_group_counts(n_bays: int) -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=n_bays, depth_ft=8.0)
    counts = group_counts(geometry)
    assert counts["bottom_chord"] == n_bays
    assert counts["top_chord"] == n_bays - 1
    assert counts["diagonals"] == 2 * n_bays
    assert "verticals" not in counts


@pytest.mark.parametrize("n_bays", BAY_SWEEP)
def test_static_determinacy(n_bays: int) -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=n_bays, depth_ft=8.0)
    assert len(geometry.members) + 3 == 2 * len(geometry.nodes)


def test_member_lengths_exact_case() -> None:
    # span 96 ft / 8 bays -> bay width 12 ft; depth 8 ft -> diagonals are
    # 6-8-10 right triangles with length exactly 10 ft.
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    assert geometry.bay_width_ft == 12.0
    for member in geometry.members:
        if member.group in ("bottom_chord", "top_chord"):
            assert member.length_ft == 12.0
        else:
            assert member.length_ft == 10.0


@pytest.mark.parametrize("n_bays", [4, 7, 12])
def test_member_lengths_match_node_coordinates(n_bays: int) -> None:
    geometry = expand_warren(span_ft=90.0, n_bays=n_bays, depth_ft=7.5)
    nodes = {node.id: node for node in geometry.nodes}
    for member in geometry.members:
        node_i = nodes[member.node_i]
        node_j = nodes[member.node_j]
        length = math.hypot(node_j.x_ft - node_i.x_ft, node_j.y_ft - node_i.y_ft)
        assert member.length_ft == pytest.approx(length, abs=1e-12)


def test_supports_at_bottom_chord_ends() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    nodes = {node.id: node for node in geometry.nodes}
    pin = nodes[geometry.pin_node]
    roller = nodes[geometry.roller_node]
    assert (pin.x_ft, pin.y_ft) == (0.0, 0.0)
    assert (roller.x_ft, roller.y_ft) == (96.0, 0.0)


@pytest.mark.parametrize("n_bays", [4, 5, 8, 13])
def test_geometry_sanity(n_bays: int) -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=n_bays, depth_ft=8.0)
    node_ids = [node.id for node in geometry.nodes]
    assert node_ids == list(range(len(geometry.nodes)))
    member_ids = [member.id for member in geometry.members]
    assert member_ids == list(range(len(geometry.members)))
    valid_ids = set(node_ids)
    for member in geometry.members:
        assert member.node_i in valid_ids
        assert member.node_j in valid_ids
        assert member.node_i != member.node_j
    # No duplicate connectivity regardless of end ordering.
    edges = {frozenset((m.node_i, m.node_j)) for m in geometry.members}
    assert len(edges) == len(geometry.members)


@pytest.mark.parametrize("n_bays", [4, 5, 8, 13])
def test_node_set_symmetric_about_midspan(n_bays: int) -> None:
    span_ft = 96.0
    geometry = expand_warren(span_ft=span_ft, n_bays=n_bays, depth_ft=8.0)
    coordinates = {
        (round(node.x_ft, 9), round(node.y_ft, 9)) for node in geometry.nodes
    }
    mirrored = {(round(span_ft - x, 9), y) for x, y in coordinates}
    assert coordinates == mirrored


def test_expand_is_pure_and_deterministic() -> None:
    first = expand("warren", span_ft=96.0, n_bays=8, depth_ft=8.0)
    second = expand("warren", span_ft=96.0, n_bays=8, depth_ft=8.0)
    assert first == second


def test_expand_dispatches_to_warren() -> None:
    assert EXPANDERS["warren"] is expand_warren
    via_dispatch = expand("warren", span_ft=60.0, n_bays=4, depth_ft=5.0)
    direct = expand_warren(span_ft=60.0, n_bays=4, depth_ft=5.0)
    assert via_dispatch == direct


def test_unknown_truss_type_raises() -> None:
    with pytest.raises(ValueError, match="unsupported truss_type"):
        expand("pratt", span_ft=96.0, n_bays=8, depth_ft=8.0)


@pytest.mark.parametrize(
    ("span_ft", "n_bays", "depth_ft"),
    [
        (96.0, 0, 8.0),
        (96.0, -3, 8.0),
        (0.0, 8, 8.0),
        (-96.0, 8, 8.0),
        (96.0, 8, 0.0),
        (96.0, 8, -8.0),
    ],
)
def test_invalid_inputs_raise(span_ft: float, n_bays: int, depth_ft: float) -> None:
    with pytest.raises(ValueError):
        expand_warren(span_ft=span_ft, n_bays=n_bays, depth_ft=depth_ft)
