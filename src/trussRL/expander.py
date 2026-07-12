"""Geometry expander.

Expands a validated design dict into nodes, members, supports, and
unbraced lengths, dispatching on typology. The single geometry source of truth shared
by the reward pipeline, calibration, heuristic baseline, and eval harness.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    """A truss joint.

    Attributes:
        id: zero-based node identifier, unique within a geometry
        x_ft: horizontal position in feet, measured from the left support
        y_ft: vertical position in feet, measured from the bottom chord level
    """

    id: int
    x_ft: float
    y_ft: float


@dataclass(frozen=True)
class Member:
    """A truss member connecting two nodes, tagged with its geometric group.

    Assumptions:
        1. The unbraced length equals the geometric length with K = 1.0 in
           both planes — conservative and consistent across all designs
           (vision.md section 5).

    Attributes:
        id: zero-based member identifier, unique within a geometry
        node_i: id of the first end node
        node_j: id of the second end node
        group: geometric group name, one of "top_chord", "bottom_chord",
            "verticals", or "diagonals" — never labeled by force sign
        length_ft: geometric length in feet, which doubles as the unbraced
            length for buckling checks
    """

    id: int
    node_i: int
    node_j: int
    group: str
    length_ft: float


@dataclass(frozen=True)
class TrussGeometry:
    """Complete expanded geometry for one truss design.

    Attributes:
        nodes: all joints, ordered by id
        members: all members, ordered by id, each tagged with its group
        pin_node: id of the pinned support node (left end of bottom chord)
        roller_node: id of the roller support node (right end of bottom chord)
        bay_width_ft: derived panel width, span / n_bays
    """

    nodes: tuple[Node, ...]
    members: tuple[Member, ...]
    pin_node: int
    roller_node: int
    bay_width_ft: float

    @property
    def n_nodes(self) -> int:
        """Number of joints, used for the per-connection cost term.

        Args: None

        Returns:
            int: count of nodes in the geometry
        """
        return len(self.nodes)


def validate_expander_inputs(span_ft: float, n_bays: int, depth_ft: float) -> None:
    """Reject inputs for which no geometry can be constructed.

    Assumptions:
        1. Only guards constructibility (positive dimensions, at least one
           bay); range policing such as the 4-24 bay window and depth ratio
           limits is the DRC layer's job (vision.md section 6).

    Args:
        span_ft: truss span in feet
        n_bays: number of bottom-chord bays
        depth_ft: truss depth in feet

    Returns:
        None

    Raises:
        ValueError: if span or depth is not positive, or n_bays is not a
            positive integer.
    """
    if not isinstance(n_bays, int) or n_bays < 1:
        raise ValueError(f"n_bays must be a positive integer, got {n_bays!r}")
    if span_ft <= 0:
        raise ValueError(f"span_ft must be positive, got {span_ft!r}")
    if depth_ft <= 0:
        raise ValueError(f"depth_ft must be positive, got {depth_ft!r}")


def expand_warren(span_ft: float, n_bays: int, depth_ft: float) -> TrussGeometry:
    """Expand a Warren truss design into its full geometry.

    Bottom chord nodes sit at i * bay_width on y = 0; top chord nodes are
    offset half a bay at (i + 0.5) * bay_width on y = depth. Each bay is
    triangulated by an up- and a down-diagonal; Warren has no verticals.
    Yields 2n + 1 nodes and 4n - 1 members for n bays, which is statically
    determinate (members + 3 reactions = 2 * nodes).

    Assumptions:
        1. Supports are a pin at the left bottom node and a roller at the
           right bottom node; all horizontal load exits through the pin
           (vision.md section 1).

    Args:
        span_ft: truss span in feet
        n_bays: number of bottom-chord bays; any positive integer
        depth_ft: truss depth in feet

    Returns:
        TrussGeometry: nodes, group-tagged members, support node ids, and
            derived bay width

    Raises:
        ValueError: if span or depth is not positive, or n_bays is not a
            positive integer.
    """
    validate_expander_inputs(span_ft, n_bays, depth_ft)
    bay_width_ft = span_ft / n_bays

    nodes: list[Node] = []
    for i in range(n_bays + 1):
        nodes.append(Node(id=i, x_ft=i * bay_width_ft, y_ft=0.0))
    for i in range(n_bays):
        nodes.append(
            Node(id=n_bays + 1 + i, x_ft=(i + 0.5) * bay_width_ft, y_ft=depth_ft)
        )

    diagonal_length_ft = math.hypot(bay_width_ft / 2.0, depth_ft)
    members: list[Member] = []
    for i in range(n_bays):
        members.append(
            Member(
                id=len(members),
                node_i=i,
                node_j=i + 1,
                group="bottom_chord",
                length_ft=bay_width_ft,
            )
        )
    for i in range(n_bays - 1):
        members.append(
            Member(
                id=len(members),
                node_i=n_bays + 1 + i,
                node_j=n_bays + 2 + i,
                group="top_chord",
                length_ft=bay_width_ft,
            )
        )
    for i in range(n_bays):
        members.append(
            Member(
                id=len(members),
                node_i=i,
                node_j=n_bays + 1 + i,
                group="diagonals",
                length_ft=diagonal_length_ft,
            )
        )
        members.append(
            Member(
                id=len(members),
                node_i=n_bays + 1 + i,
                node_j=i + 1,
                group="diagonals",
                length_ft=diagonal_length_ft,
            )
        )

    return TrussGeometry(
        nodes=tuple(nodes),
        members=tuple(members),
        pin_node=0,
        roller_node=n_bays,
        bay_width_ft=bay_width_ft,
    )


EXPANDERS: dict[str, Callable[[float, int, float], TrussGeometry]] = {
    "warren": expand_warren,
}


def expand(
    truss_type: str, span_ft: float, n_bays: int, depth_ft: float
) -> TrussGeometry:
    """Expand a design into geometry, dispatching on typology.

    Assumptions:
        1. Only typologies registered in EXPANDERS are supported; the DRC
           layer pre-screens the truss_type enum, but this function fails
           loudly on its own so it never silently returns wrong geometry.

    Args:
        truss_type: typology name; must be a key of EXPANDERS
            (currently only "warren")
        span_ft: truss span in feet
        n_bays: number of bottom-chord bays
        depth_ft: truss depth in feet

    Returns:
        TrussGeometry: the expanded geometry for the requested typology

    Raises:
        ValueError: if truss_type is not a registered typology, or the
            dimensional inputs are invalid.
    """
    if truss_type not in EXPANDERS:
        supported = ", ".join(sorted(EXPANDERS))
        raise ValueError(
            f"unsupported truss_type {truss_type!r}; supported: {supported}"
        )
    return EXPANDERS[truss_type](span_ft, n_bays, depth_ft)
