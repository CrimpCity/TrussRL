"""Determinism test (CI).

Solving a fixed design 100 times must produce byte-identical results, which
proves the ops.wipe() isolation contract holds: no state from one solve can
perturb the next (vision.md section 7). Serializing with struct.pack makes
the comparison exact down to the last bit, catching signed zero and any
low-bit drift that an approximate compare would miss.

This is a stronger superset of test_solver.test_repeat_solves_are_identical
(two solves, dict equality); both are kept.

Byte-identity is asserted within one environment. It is guaranteed for the
currently locked openseespy; the project still permits openseespy>=3.8.0.0,
so a future minor bump could in principle reorder LAPACK operations. Pinning
the solver exactly is tracked separately.
"""

import struct

from trussRL.expander import expand
from trussRL.loads import LoadCase, build_load_cases
from trussRL.solver import SolveOutcome, solve_cases

SPAN_FT = 30.0
N_BAYS = 8
DEPTH_FT = 3.75
W_KIP_PER_FT = -3.0
SECTIONS_BY_GROUP = {
    "bottom_chord": "HSS6X6X3/8",
    "top_chord": "HSS5-1/2X5-1/2X5/16",
    "diagonals": "HSS4X4X1/4",
}
N_RUNS = 100


def serialize_outcome(outcome: SolveOutcome) -> bytes:
    """Serialize a solve outcome to bytes in a stable, exact order.

    Cases are traversed in solve order, members by ascending id, then nodes
    by ascending id, and every float is packed as a little-endian double so
    two byte strings are equal only when every value matches bit for bit.

    Args:
        outcome: a successful SolveOutcome to serialize

    Returns:
        bytes: the concatenated packed doubles for every case
    """
    buffer = bytearray()
    for solution in outcome.solutions:
        for member_id in sorted(solution.member_axials_kip):
            buffer += struct.pack("<d", solution.member_axials_kip[member_id])
        for node_id in sorted(solution.node_displacements_in):
            dx, dy = solution.node_displacements_in[node_id]
            buffer += struct.pack("<dd", dx, dy)
    return bytes(buffer)


def test_fixed_design_solves_bit_identically() -> None:
    geometry = expand("warren", span_ft=SPAN_FT, n_bays=N_BAYS, depth_ft=DEPTH_FT)
    case_loads = build_load_cases(
        geometry, SECTIONS_BY_GROUP, [LoadCase(w_kip_per_ft=W_KIP_PER_FT)]
    )
    first: bytes | None = None
    for run in range(N_RUNS):
        outcome = solve_cases(geometry, SECTIONS_BY_GROUP, case_loads)
        assert outcome.ok, outcome.failure
        serialized = serialize_outcome(outcome)
        if first is None:
            first = serialized
        else:
            assert serialized == first, f"run {run} diverged from run 0"
