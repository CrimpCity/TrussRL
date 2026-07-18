"""Test-only reference truss solver (vision.md section 7).

A small numpy direct-stiffness solver, independent of OpenSees, used by the
differential test to cross-check member forces and deflections. It shares no
code with ``trussRL.solver`` and imports neither that module nor openseespy,
so agreement between the two is genuine independent-implementation evidence
that the OpenSees adapter is correct. Never imported by ``src/``.

Both solvers consume the same resolved ``CaseLoads`` (self-weight and
tributary distribution already applied by ``loads.py``): this checks the
solve, not load resolution, which has its own tests.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trussRL.catalog import get_section
from trussRL.expander import TrussGeometry
from trussRL.loads import CaseLoads

E_KSI = 29_000.0
INCHES_PER_FOOT = 12.0
DOF_PER_NODE = 2


@dataclass(frozen=True)
class ReferenceCaseSolution:
    """Solved response for one load case, mirroring solver.CaseSolution.

    Attributes:
        member_axials_kip: member id mapped to its axial force in kips,
            positive in tension, negative in compression
        node_displacements_in: node id mapped to its (dx, dy) displacement
            in inches, global axes (+X pin toward roller, +Y up)
    """

    member_axials_kip: Mapping[int, float]
    node_displacements_in: Mapping[int, tuple[float, float]]


def node_coordinates_in(geometry: TrussGeometry) -> np.ndarray:
    """Collect node coordinates in inches, indexed by zero-based node id.

    Assumptions:
        1. Node ids are contiguous and zero-based, as the expander builds
           them, so row i of the result is node id i.

    Args:
        geometry: expanded truss geometry

    Returns:
        np.ndarray: shape (n_nodes, 2) of (x, y) in inches
    """
    coords = np.zeros((geometry.n_nodes, 2), dtype=float)
    for node in geometry.nodes:
        coords[node.id] = (node.x_ft * INCHES_PER_FOOT, node.y_ft * INCHES_PER_FOOT)
    return coords


def member_geometry(
    member_node_i: int, member_node_j: int, coords: np.ndarray
) -> tuple[float, float, float]:
    """Compute a member's length and direction cosines from node coordinates.

    Args:
        member_node_i: id of the first end node
        member_node_j: id of the second end node
        coords: node coordinates in inches, indexed by node id

    Returns:
        tuple: (length_in, cos, sin) where cos and sin are the direction
            cosines of the axis from node i to node j
    """
    dx, dy = coords[member_node_j] - coords[member_node_i]
    length_in = float(np.hypot(dx, dy))
    return length_in, dx / length_in, dy / length_in


def member_area_in2(group: str, sections_by_group: Mapping[str, str]) -> float:
    """Look up the catalog area for a member group's assigned section.

    Args:
        group: geometric member group name
        sections_by_group: member group name mapped to its section
            designation

    Returns:
        float: cross-sectional area in square inches

    Raises:
        ValueError: if the group has no section assignment.
    """
    if group not in sections_by_group:
        raise ValueError(f"no section assigned to member group {group!r}")
    return get_section(sections_by_group[group]).area_in2


def element_dofs(node_i: int, node_j: int) -> list[int]:
    """Return the four global DOF indices for a two-node truss element.

    Args:
        node_i: id of the first end node
        node_j: id of the second end node

    Returns:
        list: [ix, iy, jx, jy] global DOF indices, x then y per node
    """
    return [
        DOF_PER_NODE * node_i,
        DOF_PER_NODE * node_i + 1,
        DOF_PER_NODE * node_j,
        DOF_PER_NODE * node_j + 1,
    ]


def assemble_stiffness(
    geometry: TrussGeometry, sections_by_group: Mapping[str, str], coords: np.ndarray
) -> np.ndarray:
    """Assemble the global stiffness matrix in kip/inch.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation, used for member areas
        coords: node coordinates in inches, indexed by node id

    Returns:
        np.ndarray: the (2*n_nodes, 2*n_nodes) global stiffness matrix
    """
    size = DOF_PER_NODE * geometry.n_nodes
    stiffness = np.zeros((size, size), dtype=float)
    for member in geometry.members:
        length_in, cos, sin = member_geometry(member.node_i, member.node_j, coords)
        axial_stiffness = E_KSI * member_area_in2(member.group, sections_by_group) / length_in
        block = axial_stiffness * np.array(
            [
                [cos * cos, cos * sin, -cos * cos, -cos * sin],
                [cos * sin, sin * sin, -cos * sin, -sin * sin],
                [-cos * cos, -cos * sin, cos * cos, cos * sin],
                [-cos * sin, -sin * sin, cos * sin, sin * sin],
            ]
        )
        dofs = element_dofs(member.node_i, member.node_j)
        stiffness[np.ix_(dofs, dofs)] += block
    return stiffness


def constrained_dofs(geometry: TrussGeometry) -> list[int]:
    """List the fixed DOF indices: pin in x and y, roller in y only.

    Assumptions:
        1. Matches solver.build_model exactly — ops.fix(pin, 1, 1) and
           ops.fix(roller, 0, 1) — so both solvers share support conditions.

    Args:
        geometry: expanded truss geometry

    Returns:
        list: sorted fixed global DOF indices
    """
    return sorted(
        {
            DOF_PER_NODE * geometry.pin_node,
            DOF_PER_NODE * geometry.pin_node + 1,
            DOF_PER_NODE * geometry.roller_node + 1,
        }
    )


def load_vector(geometry: TrussGeometry, case: CaseLoads) -> np.ndarray:
    """Build the global load vector in kips from resolved nodal loads.

    Args:
        geometry: expanded truss geometry
        case: resolved nodal loads for one case, self-weight included

    Returns:
        np.ndarray: length 2*n_nodes force vector, x then y per node
    """
    forces = np.zeros(DOF_PER_NODE * geometry.n_nodes, dtype=float)
    for load in case.nodal_loads:
        forces[DOF_PER_NODE * load.node_id] = load.fx_kip
        forces[DOF_PER_NODE * load.node_id + 1] = load.fy_kip
    return forces


def solve_displacements(
    stiffness: np.ndarray, forces: np.ndarray, fixed: Sequence[int]
) -> np.ndarray:
    """Solve K u = f over the free DOFs, holding fixed DOFs at zero.

    Args:
        stiffness: global stiffness matrix in kip/inch
        forces: global load vector in kips
        fixed: fixed global DOF indices

    Returns:
        np.ndarray: full displacement vector in inches, zero at fixed DOFs
    """
    free = [dof for dof in range(forces.size) if dof not in set(fixed)]
    displacements = np.zeros_like(forces)
    displacements[free] = np.linalg.solve(
        stiffness[np.ix_(free, free)], forces[free]
    )
    return displacements


def recover_axials(
    geometry: TrussGeometry,
    sections_by_group: Mapping[str, str],
    coords: np.ndarray,
    displacements: np.ndarray,
) -> dict[int, float]:
    """Recover member axial forces from nodal displacements, tension positive.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation, used for member areas
        coords: node coordinates in inches, indexed by node id
        displacements: full displacement vector in inches

    Returns:
        dict: member id mapped to its axial force in kips, positive tension
    """
    axials: dict[int, float] = {}
    for member in geometry.members:
        length_in, cos, sin = member_geometry(member.node_i, member.node_j, coords)
        area_in2 = member_area_in2(member.group, sections_by_group)
        dofs = element_dofs(member.node_i, member.node_j)
        ux_i, uy_i, ux_j, uy_j = displacements[dofs]
        elongation_in = (ux_j - ux_i) * cos + (uy_j - uy_i) * sin
        axials[member.id] = float(E_KSI * area_in2 / length_in * elongation_in)
    return axials


def solve_reference(
    geometry: TrussGeometry,
    sections_by_group: Mapping[str, str],
    case_loads: Sequence[CaseLoads],
) -> tuple[ReferenceCaseSolution, ...]:
    """Solve one linear static case per load case with numpy direct stiffness.

    Assumptions:
        1. All members are linear elastic pin-jointed truss elements in a 2D
           model, E = 29,000 ksi, catalog area per member group; the global
           stiffness is reused across cases since geometry and sections are
           fixed within a design.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation, used for member areas
        case_loads: resolved nodal loads, one CaseLoads per load case, as
            produced by loads.build_load_cases

    Returns:
        tuple: one ReferenceCaseSolution per load case, in case order

    Raises:
        ValueError: if a geometry group has no section assignment, or a
            designation is not a catalog entry.
    """
    coords = node_coordinates_in(geometry)
    stiffness = assemble_stiffness(geometry, sections_by_group, coords)
    fixed = constrained_dofs(geometry)
    solutions: list[ReferenceCaseSolution] = []
    for case in case_loads:
        displacements = solve_displacements(stiffness, load_vector(geometry, case), fixed)
        node_displacements_in = {
            node.id: (
                float(displacements[DOF_PER_NODE * node.id]),
                float(displacements[DOF_PER_NODE * node.id + 1]),
            )
            for node in geometry.nodes
        }
        solutions.append(
            ReferenceCaseSolution(
                member_axials_kip=recover_axials(
                    geometry, sections_by_group, coords, displacements
                ),
                node_displacements_in=node_displacements_in,
            )
        )
    return tuple(solutions)
