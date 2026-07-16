"""Load conversion and self-weight (starting.md Module 3).

Converts deck line loads and concentrated longitudinal forces into
panel-point nodal loads and adds a single-pass self-weight from the chosen
sections. Sign convention throughout: global +X points from the pin toward
the roller, global +Y points up, so downward forces are negative. Outputs
are in kips, directly consumable by the OpenSees wall (ndm=2, ndf=2).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import attrgetter

from trussRL.catalog import get_section
from trussRL.expander import TrussGeometry

LEVELS = ("bottom", "top")


@dataclass(frozen=True)
class LoadCase:
    """One applied load case, before self-weight is added.

    Zero magnitudes mean the component is absent, so a longitudinal-only
    case is ``LoadCase(longitudinal_kip=15.0)``.

    Attributes:
        w_kip_per_ft: uniform line load on the deck chord in kips per foot;
            negative acts downward (gravity), positive acts upward, so
            "2.0 kip/ft downward" is -2.0 and "1.2 kip/ft net uplift"
            is 1.2
        level: deck chord carrying the line and longitudinal loads,
            "bottom" or "top"
        longitudinal_kip: concentrated longitudinal force in kips, signed
            in global +X (positive points from the pin toward the roller)
    """

    w_kip_per_ft: float = 0.0
    level: str = "bottom"
    longitudinal_kip: float = 0.0


@dataclass(frozen=True)
class NodalLoad:
    """Resolved force on one node for one load case.

    Attributes:
        node_id: id of the loaded node
        fx_kip: horizontal force in kips, global +X (pin toward roller)
        fy_kip: vertical force in kips, global +Y (up), so downward loads
            are negative
    """

    node_id: int
    fx_kip: float
    fy_kip: float


@dataclass(frozen=True)
class CaseLoads:
    """All nodal loads for one load case, self-weight included.

    Attributes:
        nodal_loads: one entry per geometry node, ordered by node id,
            with zero-force nodes included
    """

    nodal_loads: tuple[NodalLoad, ...]

    @property
    def total_fx_kip(self) -> float:
        """Sum of horizontal nodal forces, for conservation checks.

        Args: None

        Returns:
            float: total fx in kips, which must equal the case's
                longitudinal force
        """
        return sum(load.fx_kip for load in self.nodal_loads)

    @property
    def total_fy_kip(self) -> float:
        """Sum of vertical nodal forces, for conservation checks.

        Args: None

        Returns:
            float: total fy in kips, which must equal w × span minus the
                summed member self-weights
        """
        return sum(load.fy_kip for load in self.nodal_loads)


def level_node_ids(geometry: TrussGeometry, level: str) -> tuple[int, ...]:
    """Return the node ids on a chord level, ordered left to right.

    Assumptions:
        1. Levels are selected by exact y-coordinate — the bottom chord at
           y_ft == 0.0 and the top chord at the maximum y_ft — never by
           node-id arithmetic, so the rule is typology-agnostic. Exact
           float comparison is safe because the expander constructs these
           coordinates as literals.

    Args:
        geometry: expanded truss geometry
        level: chord level, "bottom" or "top"

    Returns:
        tuple: node ids on the level, sorted by x-coordinate

    Raises:
        ValueError: if level is not "bottom" or "top".
    """
    if level not in LEVELS:
        supported = ", ".join(LEVELS)
        raise ValueError(f"unsupported level {level!r}; supported: {supported}")
    target_y = 0.0 if level == "bottom" else max(node.y_ft for node in geometry.nodes)
    selected = [node for node in geometry.nodes if node.y_ft == target_y]
    return tuple(node.id for node in sorted(selected, key=attrgetter("x_ft")))


def tributary_widths_ft(geometry: TrussGeometry, level: str) -> dict[int, float]:
    """Compute the tributary deck width carried by each node on a level.

    Node boundaries are the span ends plus the midpoints between adjacent
    loaded-chord nodes, so the widths always telescope to the full span:
    an aligned chord gets half-widths at its end nodes, while an offset
    chord (the Warren top chord) gets a full bay width at every node.

    Assumptions:
        1. The deck line load extends over the full pin-to-roller span even
           when the loaded chord's end nodes are offset from the supports,
           so end nodes absorb the overhang out to the span ends.

    Args:
        geometry: expanded truss geometry
        level: chord level carrying the line load, "bottom" or "top"

    Returns:
        dict: node id mapped to its tributary width in feet; the widths
            sum to the span exactly

    Raises:
        ValueError: if level is not "bottom" or "top".
    """
    node_ids = level_node_ids(geometry, level)
    nodes = {node.id: node for node in geometry.nodes}
    xs = [nodes[node_id].x_ft for node_id in node_ids]
    span_start = nodes[geometry.pin_node].x_ft
    span_end = nodes[geometry.roller_node].x_ft
    boundaries = (
        [span_start]
        + [(left + right) / 2.0 for left, right in zip(xs, xs[1:])]
        + [span_end]
    )
    return {
        node_id: boundaries[index + 1] - boundaries[index]
        for index, node_id in enumerate(node_ids)
    }


def line_load_forces(
    geometry: TrussGeometry, w_kip_per_ft: float, level: str
) -> dict[int, float]:
    """Convert a uniform deck line load into vertical panel-point forces.

    Assumptions:
        1. Each node carries fy = w × tributary width, so a negative
           (downward) w yields negative forces and the total equals
           w × span exactly.

    Args:
        geometry: expanded truss geometry
        w_kip_per_ft: uniform line load in kips per foot; negative acts
            downward
        level: chord level carrying the line load, "bottom" or "top"

    Returns:
        dict: node id mapped to its vertical force in kips (global +Y)

    Raises:
        ValueError: if level is not "bottom" or "top".
    """
    widths = tributary_widths_ft(geometry, level)
    return {node_id: w_kip_per_ft * width for node_id, width in widths.items()}


def longitudinal_forces(
    geometry: TrussGeometry, force_kip: float, level: str
) -> dict[int, float]:
    """Distribute a concentrated longitudinal force over the deck nodes.

    Assumptions:
        1. The force is split equally across all nodes at the stated deck
           level; no tributary weighting is applied.

    Args:
        geometry: expanded truss geometry
        force_kip: concentrated longitudinal force in kips, signed in
            global +X (positive points from the pin toward the roller)
        level: deck level receiving the force, "bottom" or "top"

    Returns:
        dict: node id mapped to its horizontal force in kips (global +X);
            the forces sum to force_kip

    Raises:
        ValueError: if level is not "bottom" or "top".
    """
    node_ids = level_node_ids(geometry, level)
    share = force_kip / len(node_ids)
    return {node_id: share for node_id in node_ids}


def self_weight_forces(
    geometry: TrussGeometry, sections_by_group: Mapping[str, str]
) -> dict[int, float]:
    """Compute the downward nodal forces from member self-weight.

    Assumptions:
        1. Self-weight always acts downward (negative fy) and is applied in
           every load case, including net uplift — no per-case flag.
        2. Each member weighs W (lb/ft) × length_ft / 1000 kips, lumped
           half to each end node.
        3. Group keys in sections_by_group with no members in the geometry for example
           the "verticals" for a Warren truss are ignored.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation

    Returns:
        dict: node id mapped to its vertical self-weight force in kips
            (global +Y, all values negative)

    Raises:
        ValueError: if a geometry group has no entry in sections_by_group,
            or a designation is not a catalog entry.
    """
    forces: dict[int, float] = {}
    for member in geometry.members:
        if member.group not in sections_by_group:
            raise ValueError(f"no section assigned to member group {member.group!r}")
        section = get_section(sections_by_group[member.group])
        weight_kip = section.weight_lb_per_ft * member.length_ft / 1000.0
        for node_id in (member.node_i, member.node_j):
            forces[node_id] = forces.get(node_id, 0.0) - weight_kip / 2.0
    return forces


def build_case_loads(
    geometry: TrussGeometry, case: LoadCase, self_weight: Mapping[int, float]
) -> CaseLoads:
    """Resolve one load case plus self-weight into per-node forces.

    Assumptions:
        1. Every geometry node appears in the output, ordered by node id,
           with explicit zeros for unloaded nodes.

    Args:
        geometry: expanded truss geometry
        case: the applied load case (line load and longitudinal force)
        self_weight: node id mapped to its self-weight fy in kips, as
            produced by self_weight_forces

    Returns:
        CaseLoads: the resolved nodal loads for the case

    Raises:
        ValueError: if the case's level is not "bottom" or "top".
    """
    fy_line = line_load_forces(geometry, case.w_kip_per_ft, case.level)
    fx_longitudinal = longitudinal_forces(geometry, case.longitudinal_kip, case.level)
    nodal_loads = tuple(
        NodalLoad(
            node_id=node.id,
            fx_kip=fx_longitudinal.get(node.id, 0.0),
            fy_kip=fy_line.get(node.id, 0.0) + self_weight.get(node.id, 0.0),
        )
        for node in sorted(geometry.nodes, key=attrgetter("id"))
    )
    return CaseLoads(nodal_loads=nodal_loads)


def build_load_cases(
    geometry: TrussGeometry,
    sections_by_group: Mapping[str, str],
    cases: Sequence[LoadCase],
) -> tuple[CaseLoads, ...]:
    """Resolve every load case into nodal loads, self-weight included.

    The Module 3 entry point: computes the self-weight vector once and
    adds it to every case in a single deterministic pass.

    Assumptions:
        1. Self-weight is applied downward in every case, including net uplift.
        2. An empty cases sequence returns an empty tuple; policing the
           1-3 case window is the DRC/instance layer's job.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation
        cases: the applied load cases

    Returns:
        tuple: one CaseLoads per case, in case order

    Raises:
        ValueError: if a geometry group has no section assignment, a
            designation is unknown, or a case level is invalid.
    """
    self_weight = self_weight_forces(geometry, sections_by_group)
    return tuple(build_case_loads(geometry, case, self_weight) for case in cases)
