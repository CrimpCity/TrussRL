"""Differential test: OpenSees vs the numpy reference solver.

Two independent solvers must agree on member forces and nodal
displacements to 1e-6, which is the strongest available evidence that the
OpenSees adapter in ``trussRL.solver`` is correct (vision.md section 7,
OPENSEES_SCOPE.md). Coverage is a fixed-seed random sweep over the design
space plus three named anchor designs.

For a statically determinate Warren truss the member forces are area
independent, so varying the section per group chiefly exercises the
displacement comparison, while varying span, bay count, and depth exercises
the adapter's DOF-mapping and direction-cosine paths -- where an adapter bug
would actually hide.

``reference_solver`` is imported as a top-level module: pytest's default
prepend import mode puts ``tests/`` on sys.path, and the helper is not
collected because it has no ``test_`` prefix.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from tests.reference_solver import ReferenceCaseSolution, solve_reference
else:
    from reference_solver import ReferenceCaseSolution, solve_reference

from trussRL.catalog import list_sections
from trussRL.expander import expand
from trussRL.loads import LoadCase, build_load_cases
from trussRL.solver import CaseSolution, solve_cases

RTOL = 1e-6
ATOL = 1e-6
MEMBER_GROUPS = ("bottom_chord", "top_chord", "diagonals")
DESIGNATIONS = tuple(section.designation for section in list_sections())
SWEEP_SEED = 20260717
SWEEP_SIZE = 300

# span_ft, n_bays, depth_ft, sections_by_group, cases
FIXED_ANCHORS = (
    (
        10.0,
        4,
        1.25,
        {
            "bottom_chord": "HSS4X4X1/4",
            "top_chord": "HSS4X4X1/8",
            "diagonals": "HSS2-1/2X2-1/2X5/16",
        },
        (LoadCase(w_kip_per_ft=-1.0),),
    ),
    (
        30.0,
        8,
        3.75,
        {
            "bottom_chord": "HSS6X6X3/8",
            "top_chord": "HSS5-1/2X5-1/2X5/16",
            "diagonals": "HSS4X4X1/4",
        },
        (LoadCase(w_kip_per_ft=-3.0),),
    ),
    (
        60.0,
        12,
        7.5,
        {
            "bottom_chord": "HSS8X8X3/8",
            "top_chord": "HSS7X7X5/16",
            "diagonals": "HSS4X4X1/2",
        },
        (LoadCase(w_kip_per_ft=-10.0),),
    ),
)


def assert_case_agrees(
    wall: CaseSolution,
    reference: ReferenceCaseSolution,
    member_ids: list[int],
    node_ids: list[int],
    label: str,
) -> None:
    """Assert one case's forces and displacements agree between solvers.

    Args:
        wall: the OpenSees CaseSolution
        reference: the numpy ReferenceCaseSolution for the same case
        member_ids: member ids in stable sorted order
        node_ids: node ids in stable sorted order
        label: design description included in failure messages

    Returns: None
    """
    assert set(wall.member_axials_kip) == set(member_ids), label
    assert set(reference.member_axials_kip) == set(member_ids), label
    wall_forces = np.array([wall.member_axials_kip[i] for i in member_ids])
    ref_forces = np.array([reference.member_axials_kip[i] for i in member_ids])
    np.testing.assert_allclose(
        wall_forces, ref_forces, rtol=RTOL, atol=ATOL, err_msg=f"forces: {label}"
    )
    wall_disp = np.array([wall.node_displacements_in[i] for i in node_ids])
    ref_disp = np.array([reference.node_displacements_in[i] for i in node_ids])
    np.testing.assert_allclose(
        wall_disp, ref_disp, rtol=RTOL, atol=ATOL, err_msg=f"displacements: {label}"
    )


def assert_solvers_agree(
    span_ft: float,
    n_bays: int,
    depth_ft: float,
    sections_by_group: dict[str, str],
    cases: tuple[LoadCase, ...],
) -> None:
    """Expand one design, solve it both ways, and assert full agreement.

    Args:
        span_ft: truss span in feet
        n_bays: number of bottom-chord bays
        depth_ft: truss depth in feet
        sections_by_group: section designation per Warren member group
        cases: load cases fed identically to both solvers

    Returns: None
    """
    label = (
        f"span={span_ft} n_bays={n_bays} depth={depth_ft} sections={sections_by_group}"
    )
    geometry = expand("warren", span_ft=span_ft, n_bays=n_bays, depth_ft=depth_ft)
    case_loads = build_load_cases(geometry, sections_by_group, cases)
    outcome = solve_cases(geometry, sections_by_group, case_loads)
    assert outcome.ok, f"{label}: {outcome.failure}"
    reference = solve_reference(geometry, sections_by_group, case_loads)
    assert len(outcome.solutions) == len(reference) == len(cases), label
    member_ids = sorted(member.id for member in geometry.members)
    node_ids = sorted(node.id for node in geometry.nodes)
    for wall, ref in zip(outcome.solutions, reference):
        assert_case_agrees(wall, ref, member_ids, node_ids, label)


def sample_design(
    rng: np.random.Generator,
) -> tuple[float, int, float, dict[str, str], tuple[LoadCase, ...]]:
    """Draw one random Warren gravity design inside the physical envelope.

    Assumptions:
        1. The differential test bypasses the DRC layer, so only
           constructibility matters; keeping the depth/bay-width aspect in
           [0.3, 3.5] also keeps the stiffness matrix well conditioned.

    Args:
        rng: seeded numpy generator driving every draw

    Returns:
        tuple: (span_ft, n_bays, depth_ft, sections_by_group, cases)
    """
    while True:
        span_ft = float(rng.uniform(60.0, 120.0))
        n_bays = int(rng.integers(4, 25))
        depth_ft = float(rng.uniform(span_ft / 25.0, span_ft / 4.0))
        if 0.3 <= depth_ft / (span_ft / n_bays) <= 3.5:
            break
    sections_by_group = {
        group: str(rng.choice(DESIGNATIONS)) for group in MEMBER_GROUPS
    }
    cases = tuple(
        LoadCase(w_kip_per_ft=float(rng.uniform(-10.0, -0.5)))
        for _ in range(int(rng.integers(1, 3)))
    )
    return span_ft, n_bays, depth_ft, sections_by_group, cases


@pytest.mark.parametrize(
    "span_ft,n_bays,depth_ft,sections_by_group,cases", FIXED_ANCHORS
)
def test_fixed_anchor_designs_agree(
    span_ft: float,
    n_bays: int,
    depth_ft: float,
    sections_by_group: dict[str, str],
    cases: tuple[LoadCase, ...],
) -> None:
    assert_solvers_agree(span_ft, n_bays, depth_ft, sections_by_group, cases)


def test_random_sweep_designs_agree() -> None:
    rng = np.random.default_rng(SWEEP_SEED)
    for _ in range(SWEEP_SIZE):
        assert_solvers_agree(*sample_design(rng))
