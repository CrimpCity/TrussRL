"""Exercise the OpenSees wall on the simplest possible truss.

Models a single-bay Warren truss (one triangle): a pin, a roller, and an apex
node, with a point load P pushing down at the apex. This is the simplest
statically determinate truss, so every member force is closed-form (method of
joints) and independent of member areas. The script checks the solver's axial
force in each member -- with the right tension/compression sign -- against those
hand values, and cross-checks the P/2 support reactions.

Companion to ``scripts.validation.simple_beam_check``, which checks the global
beam results (reactions and midspan moment); this one focuses on the member
axial forces that a truss solver actually produces.

Run standalone:

    uv run python -m scripts.validation.simple_truss_check
"""

import math
import sys

from trussRL.expander import TrussGeometry, expand
from trussRL.loads import CaseLoads, NodalLoad
from trussRL.solver import CaseSolution, solve_cases

SPAN_FT = 48.0
DEPTH_FT = 6.0
N_BAYS = 1
P_KIP = 10.0
SECTIONS_BY_GROUP = {
    "bottom_chord": "HSS6X6X3/8",
    "diagonals": "HSS4X4X1/4",
}


def apex_node(geometry: TrussGeometry) -> int:
    """Find the apex node of the single-triangle truss.

    Assumptions:
        1. The truss has exactly one node above the bottom chord (the apex of
           a 1-bay Warren truss), sitting at the maximum y-coordinate.

    Args:
        geometry: expanded truss geometry

    Returns:
        int: id of the apex node
    """
    return max(geometry.nodes, key=lambda node: node.y_ft).id


def apex_point_load(geometry: TrussGeometry, node_id: int, p_kip: float) -> CaseLoads:
    """Build a single downward point load at one node, no self-weight.

    Assumptions:
        1. Self-weight is deliberately excluded so the member forces land on
           the clean method-of-joints values; every other node carries an
           explicit zero.

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


def expected_member_axials(geometry: TrussGeometry, p_kip: float) -> dict[int, float]:
    """Closed-form method-of-joints axial force for each member, in kips.

    Assumptions:
        1. The geometry is a single-bay Warren truss (one bottom chord and two
           diagonals) loaded by p_kip downward at the apex, with P/2 reactions.
           The bottom chord carries tension P*L/(4*d); each diagonal carries
           compression -P*Ld/(2*d), where L is the span, d the depth, and Ld
           the diagonal length.

    Args:
        geometry: expanded truss geometry
        p_kip: load magnitude in kips, applied downward at the apex

    Returns:
        dict: member id mapped to its expected axial force in kips, positive
            in tension

    Raises:
        ValueError: if a member is not a bottom_chord or diagonals member.
    """
    nodes = {node.id: node for node in geometry.nodes}
    span_ft = nodes[geometry.roller_node].x_ft - nodes[geometry.pin_node].x_ft
    depth_ft = max(node.y_ft for node in geometry.nodes)
    expected: dict[int, float] = {}
    for member in geometry.members:
        if member.group == "bottom_chord":
            expected[member.id] = p_kip * span_ft / (4.0 * depth_ft)
        elif member.group == "diagonals":
            expected[member.id] = -p_kip * member.length_ft / (2.0 * depth_ft)
        else:
            raise ValueError(f"unexpected member group {member.group!r}")
    return expected


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


def main() -> int:
    """Solve the single-triangle truss and check every member axial force.

    Args: None

    Returns:
        int: 0 when all members and reactions match, 1 otherwise
    """
    geometry = expand("warren", span_ft=SPAN_FT, n_bays=N_BAYS, depth_ft=DEPTH_FT)
    loaded_node = apex_node(geometry)
    case = apex_point_load(geometry, loaded_node, P_KIP)

    outcome = solve_cases(geometry, SECTIONS_BY_GROUP, [case])
    if not outcome.ok:
        print(f"solve failed: {outcome.failure}")
        return 1
    (solution,) = outcome.solutions

    expected = expected_member_axials(geometry, P_KIP)
    print(f"span L = {SPAN_FT:.1f} ft, depth = {DEPTH_FT:.1f} ft, P = {P_KIP:.1f} kip")
    members_ok = True
    for member in geometry.members:
        solved = solution.member_axials_kip[member.id]
        want = expected[member.id]
        state = "TENSION" if solved > 0.0 else "COMPRESSION"
        print(
            f"member {member.id} {member.group:<12s} = {solved:+8.3f} kip  "
            f"(expected {want:+8.3f})  {state}"
        )
        members_ok = members_ok and math.isclose(
            solved, want, rel_tol=1e-9, abs_tol=1e-9
        )

    pin_reaction = vertical_reaction(geometry, solution, geometry.pin_node)
    roller_reaction = vertical_reaction(geometry, solution, geometry.roller_node)
    print(
        f"pin reaction    = {pin_reaction:.3f} kip   "
        f"roller reaction = {roller_reaction:.3f} kip"
    )

    expected_reaction = P_KIP / 2.0
    reactions_ok = math.isclose(
        pin_reaction, expected_reaction, rel_tol=1e-9, abs_tol=1e-9
    ) and math.isclose(roller_reaction, expected_reaction, rel_tol=1e-9, abs_tol=1e-9)

    checks_ok = members_ok and reactions_ok
    print("OK" if checks_ok else "MISMATCH")
    return 0 if checks_ok else 1


if __name__ == "__main__":
    sys.exit(main())
