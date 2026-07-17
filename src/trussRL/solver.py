"""The wall around OpenSees: solve, and never let a failure escape.

All openseespy interaction lives here, wrapped in try/except, a timeout,
and a model wipe, so a crash, NaN, or hang becomes a scored outcome (reward
ladder rung 2) instead of taking down a training run. Nothing outside this
module touches OpenSees, in either direction.
"""

import math
import signal
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import FrameType
from typing import Any

import openseespy.opensees as ops

from trussRL.catalog import get_section
from trussRL.expander import TrussGeometry
from trussRL.loads import CaseLoads

SOLVE_TIMEOUT_S = 5.0
E_KSI = 29_000.0
INCHES_PER_FOOT = 12.0
TAG_OFFSET = 1
MATERIAL_TAG = 1
TIME_SERIES_TAG = 1
PATTERN_TAG = 1


@dataclass(frozen=True)
class CaseSolution:
    """Solved response for one load case.

    Attributes:
        member_axials_kip: member id mapped to its axial force in kips,
            positive in tension, negative in compression
        node_displacements_in: node id mapped to its (dx, dy) displacement
            in inches, global axes (+X pin toward roller, +Y up), so
            downward deflections are negative dy
    """

    member_axials_kip: Mapping[int, float]
    node_displacements_in: Mapping[int, tuple[float, float]]


@dataclass(frozen=True)
class SolveOutcome:
    """Result of solving every load case, or a clean failure flag.

    Any exception, non-convergence, NaN, or timeout maps to a failure
    outcome — a graded rung-2 exit, never a training-loop crash.

    Attributes:
        solutions: one CaseSolution per load case, in case order; empty
            when the solve failed
        failure: reason the solve failed ("timeout", "nan in results",
            exception text, ...), or None on success
    """

    solutions: tuple[CaseSolution, ...]
    failure: str | None

    @property
    def ok(self) -> bool:
        """Whether every case solved cleanly.

        Args: None

        Returns:
            bool: True when failure is None
        """
        return self.failure is None


def raise_timeout(signum: int, frame: FrameType | None) -> None:
    """Convert a SIGALRM delivery into a TimeoutError for the wall to catch.

    Args:
        signum: signal number delivered by the OS
        frame: interrupted stack frame, unused

    Returns:
        None

    Raises:
        TimeoutError: always, so the armed timer aborts the solve.
    """
    raise TimeoutError("timeout")


def wipe_safely() -> None:
    """Clear all OpenSees state without letting cleanup mask the outcome.

    Used only for the final cleanup wipe: a failure while wiping must never
    replace the solve's real result or exception. The wipe at model-build
    time stays unguarded on purpose — there a failure genuinely is a failed
    solve and belongs in the outcome.

    Args: None

    Returns: None
    """
    try:
        ops.wipe()
    except Exception:
        pass


def build_model(geometry: TrussGeometry, sections_by_group: Mapping[str, str]) -> None:
    """Build a fresh 2D OpenSees truss model from the expanded geometry.

    The model is built in kip-inch units: coordinates converted from feet,
    areas in square inches from the catalog, E in ksi, so solved forces come
    out in kips and displacements in inches with no output conversion.

    Assumptions:
        1. Geometry ids are zero-based, so every OpenSees tag is offset by
           TAG_OFFSET to stay strictly positive.
        2. Starts from ops.wipe(), so no state from a previous solve can
           leak into this model.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation, used for member areas

    Returns:
        None

    Raises:
        ValueError: if a geometry group has no entry in sections_by_group
            or a designation is not a catalog entry.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for node in geometry.nodes:
        ops.node(
            node.id + TAG_OFFSET,
            node.x_ft * INCHES_PER_FOOT,
            node.y_ft * INCHES_PER_FOOT,
        )
    ops.fix(geometry.pin_node + TAG_OFFSET, 1, 1)
    ops.fix(geometry.roller_node + TAG_OFFSET, 0, 1)
    ops.uniaxialMaterial("Elastic", MATERIAL_TAG, E_KSI)
    for member in geometry.members:
        if member.group not in sections_by_group:
            raise ValueError(f"no section assigned to member group {member.group!r}")
        area_in2 = get_section(sections_by_group[member.group]).area_in2
        ops.element(
            "Truss",
            member.id + TAG_OFFSET,
            member.node_i + TAG_OFFSET,
            member.node_j + TAG_OFFSET,
            area_in2,
            MATERIAL_TAG,
        )


def apply_loads(case: CaseLoads) -> None:
    """Apply one case's nodal loads to the current model as a plain pattern.

    Assumptions:
        1. Loads arrive in kips with the global sign convention (+X pin
           toward roller, +Y up), matching the kip-inch model exactly.

    Args:
        case: resolved nodal loads for one load case, self-weight included

    Returns: None
    """
    ops.timeSeries("Linear", TIME_SERIES_TAG)
    ops.pattern("Plain", PATTERN_TAG, TIME_SERIES_TAG)
    for load in case.nodal_loads:
        ops.load(load.node_id + TAG_OFFSET, load.fx_kip, load.fy_kip)


def run_static_analysis() -> int:
    """Run one linear static analysis step on the current model.

    Assumptions:
        1. Linear algorithm with a single full LoadControl step — no
           iteration, no convergence loop, per the linear elastic contract.
        2. Uses the ProfileSPD system because it propagates a singular
           stiffness matrix as a nonzero analyze return code; the LAPACK
           band solvers return 0 with garbage results in that case.

    Args: None

    Returns:
        int: the ops.analyze return code, 0 on success
    """
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("ProfileSPD")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    return int(ops.analyze(1))


def extract_solution(geometry: TrussGeometry) -> CaseSolution:
    """Read member forces and nodal displacements from the solved model.

    Assumptions:
        1. ops.basicForce on a Truss element returns the axial force with
           positive meaning tension, matching the CaseSolution convention.

    Args:
        geometry: expanded truss geometry, used to enumerate ids

    Returns:
        CaseSolution: axial force in kips per member id and (dx, dy)
            displacement in inches per node id
    """
    member_axials_kip = {
        member.id: float(ops.basicForce(member.id + TAG_OFFSET)[0])
        for member in geometry.members
    }
    node_displacements_in = {}
    for node in geometry.nodes:
        displacement = ops.nodeDisp(node.id + TAG_OFFSET)
        node_displacements_in[node.id] = (
            float(displacement[0]),
            float(displacement[1]),
        )
    return CaseSolution(
        member_axials_kip=member_axials_kip,
        node_displacements_in=node_displacements_in,
    )


def solution_is_finite(solution: CaseSolution) -> bool:
    """Check every extracted number for NaN or infinity.

    Args:
        solution: solved response for one load case

    Returns:
        bool: True when every force and displacement component is finite
    """
    forces_finite = all(
        math.isfinite(force) for force in solution.member_axials_kip.values()
    )
    displacements_finite = all(
        math.isfinite(dx) and math.isfinite(dy)
        for dx, dy in solution.node_displacements_in.values()
    )
    return forces_finite and displacements_finite


def solve_cases(
    geometry: TrussGeometry,
    sections_by_group: Mapping[str, str],
    case_loads: Sequence[CaseLoads],
    timeout_s: float = SOLVE_TIMEOUT_S,
) -> SolveOutcome:
    """Solve one linear static analysis per load case behind the wall.

    Assumptions:
        1. All members are linear elastic Truss elements in a 2D model
           (ndm=2, ndf=2); member axial stiffness uses the catalog area of
           the member's group section.
        2. Every solve is wrapped in try/except plus a hard timeout with
           ops.wipe() between runs; any exception, non-convergence, NaN,
           or timeout returns a failure outcome instead of raising.
        3. The timeout is enforced with SIGALRM, which requires the main
           thread; called off the main thread, the solve still runs with
           every other protection but no timer.
        4. Any SIGALRM handler and ITIMER_REAL timer the caller had armed
           are restored on exit, so a solve never clobbers an outer alarm.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation, used for member areas
        case_loads: resolved nodal loads, one CaseLoads per load case,
            as produced by loads.build_load_cases
        timeout_s: hard wall-clock limit per case solve in seconds

    Returns:
        SolveOutcome: per-case member axial forces and nodal displacements,
            or a clean failure flag mapping to reward ladder rung 2

    Raises:
        ValueError: if timeout_s is not finite and positive — a caller bug,
            not a solve failure, so it fails loudly instead of scoring.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(f"timeout_s must be finite and positive, got {timeout_s!r}")
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous_handler: Any = None
    previous_timer = (0.0, 0.0)
    if on_main_thread:
        previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        solutions: list[CaseSolution] = []
        for case in case_loads:
            if on_main_thread:
                signal.setitimer(signal.ITIMER_REAL, timeout_s)
            build_model(geometry, sections_by_group)
            apply_loads(case)
            return_code = run_static_analysis()
            if return_code != 0:
                return SolveOutcome(
                    solutions=(), failure=f"analysis failed (rc={return_code})"
                )
            solution = extract_solution(geometry)
            if not solution_is_finite(solution):
                return SolveOutcome(solutions=(), failure="nan in results")
            solutions.append(solution)
        return SolveOutcome(solutions=tuple(solutions), failure=None)
    except TimeoutError:
        return SolveOutcome(solutions=(), failure="timeout")
    except Exception as exc:
        return SolveOutcome(solutions=(), failure=str(exc) or type(exc).__name__)
    finally:
        if on_main_thread:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            restored = (
                previous_handler if previous_handler is not None else signal.SIG_DFL
            )
            signal.signal(signal.SIGALRM, restored)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        wipe_safely()
