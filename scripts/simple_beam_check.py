"""Exercise the OpenSees wall on a textbook simply-supported beam.

Models the simplest possible "beam" the truss solver can represent: a 2-bay
Warren truss with a pin at one end, a roller at the other, and a single point
load P at midspan. Recovers the two hand-calculable results and checks them:

- support reactions equal P/2 each, and
- the midspan bending moment equals P*L/4.

Because solver.py is a pin-jointed truss solver (Truss elements carry axial
force only, and solve_cases returns neither reactions nor moments), both
quantities are reconstructed from the member axials: reactions from vertical
joint equilibrium at the supports, and the midspan moment from the top-chord
couple, |top_chord_axial| * depth. Both are exact for this statically
determinate truss, independent of member areas.

Run standalone:

    uv run python scripts/simple_beam_check.py
"""

import math
import sys

from trussRL.expander import TrussGeometry, expand
from trussRL.loads import CaseLoads, NodalLoad
from trussRL.solver import CaseSolution, solve_cases

SPAN_FT = 48.0
DEPTH_FT = 6.0
N_BAYS = 2
P_KIP = 10.0
SECTIONS_BY_GROUP = {
    "bottom_chord": "HSS6X6X3/8",
    "top_chord": "HSS6X6X3/8",
    "diagonals": "HSS4X4X1/4",
}


def midspan_node(geometry: TrussGeometry) -> int:
    """Find the bottom-chord node at midspan between the two supports.

    Assumptions:
        1. The load line is the bottom chord (y_ft == 0.0), and the geometry
           has a node exactly at the pin-roller midpoint, as a 2-bay Warren
           truss does.

    Args:
        geometry: expanded truss geometry

    Returns:
        int: id of the bottom-chord node at midspan

    Raises:
        ValueError: if no bottom-chord node sits at the midspan x-coordinate.
    """
    nodes = {node.id: node for node in geometry.nodes}
    mid_x = (nodes[geometry.pin_node].x_ft + nodes[geometry.roller_node].x_ft) / 2.0
    for node in geometry.nodes:
        if node.y_ft == 0.0 and math.isclose(node.x_ft, mid_x):
            return node.id
    raise ValueError(f"no bottom-chord node at midspan x={mid_x}")


def midspan_point_load(
    geometry: TrussGeometry, node_id: int, p_kip: float
) -> CaseLoads:
    """Build a single downward point load at one node, no self-weight.

    Assumptions:
        1. Self-weight is deliberately excluded so the reactions and moment
           land on the clean P/2 and P*L/4 values; every other node carries
           an explicit zero.

    Args:
        geometry: expanded truss geometry
        node_id: id of the loaded node
        p_kip: load magnitude in kips, applied downward

    Returns:
        CaseLoads: one NodalLoad per geometry node, ordered by node id
    """
    return CaseLoads(
        nodal_loads=tuple(
            NodalLoad(
                node_id=node.id,
                fx_kip=0.0,
                fy_kip=-p_kip if node.id == node_id else 0.0,
            )
            for node in sorted(geometry.nodes, key=lambda n: n.id)
        )
    )


def vertical_reaction(
    geometry: TrussGeometry, solution: CaseSolution, support_node: int
) -> float:
    """Recover a support's vertical reaction from vertical joint equilibrium.

    Assumptions:
        1. No external load is applied at the support node, so the reaction
           balances only the member forces meeting there. A member axial N
           (tension positive) pulls the node toward its far end, contributing
           a vertical force N * (y_far - y_node) / length to the node.

    Args:
        geometry: expanded truss geometry
        solution: solved response carrying member axials in kips
        support_node: id of the pin or roller node

    Returns:
        float: vertical reaction in kips, positive upward
    """
    nodes = {node.id: node for node in geometry.nodes}
    node = nodes[support_node]
    member_vertical_force = 0.0
    for member in geometry.members:
        if support_node not in (member.node_i, member.node_j):
            continue
        far_id = member.node_j if member.node_i == support_node else member.node_i
        far = nodes[far_id]
        length_ft = math.hypot(far.x_ft - node.x_ft, far.y_ft - node.y_ft)
        axial_kip = solution.member_axials_kip[member.id]
        member_vertical_force += axial_kip * (far.y_ft - node.y_ft) / length_ft
    return -member_vertical_force


def midspan_moment(geometry: TrussGeometry, solution: CaseSolution) -> float:
    """Recover the midspan bending moment from the top-chord couple.

    Assumptions:
        1. A vertical cut at midspan leaves only the top chord contributing a
           moment about the loaded bottom node, so M = |top_chord_axial| times
           the truss depth. Uses the single top-chord member of a 2-bay Warren.

    Args:
        geometry: expanded truss geometry
        solution: solved response carrying member axials in kips

    Returns:
        float: midspan bending moment in kip-feet

    Raises:
        ValueError: if the geometry has no top-chord member.
    """
    depth_ft = max(node.y_ft for node in geometry.nodes)
    for member in geometry.members:
        if member.group == "top_chord":
            return abs(solution.member_axials_kip[member.id]) * depth_ft
    raise ValueError("geometry has no top_chord member")


def main() -> int:
    """Solve the simple beam and check reactions and midspan moment.

    Args: None

    Returns:
        int: 0 when both checks pass, 1 otherwise
    """
    geometry = expand("warren", span_ft=SPAN_FT, n_bays=N_BAYS, depth_ft=DEPTH_FT)
    loaded_node = midspan_node(geometry)
    case = midspan_point_load(geometry, loaded_node, P_KIP)

    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, [case])
    if not outcome.ok:
        print(f"solve failed: {outcome.failure}")
        return 1
    (solution,) = outcome.solutions

    pin_reaction = vertical_reaction(geometry, solution, geometry.pin_node)
    roller_reaction = vertical_reaction(geometry, solution, geometry.roller_node)
    moment = midspan_moment(geometry, solution)

    expected_reaction = P_KIP / 2.0
    expected_moment = P_KIP * SPAN_FT / 4.0

    print(
        f"span L         = {SPAN_FT:.1f} ft, depth = {DEPTH_FT:.1f} ft, P = {P_KIP:.1f} kip"
    )
    print(
        f"pin reaction    = {pin_reaction:.3f} kip     (expected P/2  = {expected_reaction:.3f})"
    )
    print(
        f"roller reaction = {roller_reaction:.3f} kip     (expected P/2  = {expected_reaction:.3f})"
    )
    print(
        f"midspan moment  = {moment:.3f} kip*ft (expected P*L/4 = {expected_moment:.3f})"
    )

    checks_ok = (
        math.isclose(pin_reaction, expected_reaction, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(roller_reaction, expected_reaction, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(moment, expected_moment, rel_tol=1e-9, abs_tol=1e-9)
    )
    print("OK" if checks_ok else "MISMATCH")
    return 0 if checks_ok else 1


if __name__ == "__main__":
    sys.exit(main())
