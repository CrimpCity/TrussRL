"""Design rule checks: reject invalid designs before they reach the solver.

Cheap sanity checks that fail fast with a human-readable reason, so obvious
junk never costs an OpenSees run — most valuable early in training, when the
policy is still poor and most designs are junk.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from trussRL.catalog import get_section, has_section
from trussRL.expander import EXPANDERS, TrussGeometry, expand

INCHES_PER_FOOT = 12.0
N_BAYS_MIN = 4
N_BAYS_MAX = 24
DEPTH_SPAN_FRACTION_MIN = 1 / 25
DEPTH_SPAN_FRACTION_MAX = 1 / 4
SLENDERNESS_LIMIT = 200.0
BAY_ASPECT_MIN = 0.3
BAY_ASPECT_MAX = 3.5


@dataclass(frozen=True)
class DRCFailure:
    """One failed design-rule check.

    Attributes:
        check: check identifier, one of "truss_type", "sections", "n_bays",
            "depth", "slenderness", or "bay_aspect"
        reason: human-readable reason including the offending values
    """

    check: str
    reason: str


@dataclass(frozen=True)
class DRCResult:
    """Outcome of all design-rule checks for one design.

    Attributes:
        failures: every failed check, in fixed check order; empty on pass
    """

    failures: tuple[DRCFailure, ...]

    @property
    def passed(self) -> bool:
        """Whether the design cleared every check.

        Args: None

        Returns:
            bool: True when no check failed
        """
        return not self.failures

    @property
    def summary(self) -> str:
        """One-line result for logging and the reward reason string.

        Args: None

        Returns:
            str: "pass", or the failure reasons joined by "; "
        """
        if self.passed:
            return "pass"
        return "; ".join(failure.reason for failure in self.failures)


def can_expand(truss_type: str, span_ft: float, n_bays: int, depth_ft: float) -> bool:
    """Report whether the expander can construct geometry for these fields.

    Assumptions:
        1. Boolean mirror of expander.validate_expander_inputs plus
           truss_type membership in EXPANDERS — kept in sync by hand so DRC
           never needs try/except around expand().

    Args:
        truss_type: typology name from the design
        span_ft: truss span in feet
        n_bays: number of bottom-chord bays
        depth_ft: truss depth in feet

    Returns:
        bool: True when expand() would succeed on these inputs
    """
    return (
        truss_type in EXPANDERS
        and isinstance(n_bays, int)
        and n_bays >= 1
        and span_ft > 0
        and depth_ft > 0
    )


def geometry_groups(geometry: TrussGeometry) -> tuple[str, ...]:
    """List the member groups a geometry actually uses.

    Args:
        geometry: expanded truss geometry

    Returns:
        tuple: unique group names in first-appearance member order
    """
    return tuple(dict.fromkeys(member.group for member in geometry.members))


def groups_resolvable(
    geometry: TrussGeometry, sections_by_group: Mapping[str, str]
) -> bool:
    """Report whether every geometry group resolves to a catalog section.

    Gates the slenderness check so an unknown or missing section produces
    only a "sections" failure, never a slenderness cascade on top.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation

    Returns:
        bool: True when every used group is mapped to a known designation
    """
    return all(
        group in sections_by_group and has_section(sections_by_group[group])
        for group in geometry_groups(geometry)
    )


def check_truss_type(truss_type: str) -> tuple[DRCFailure, ...]:
    """Check that the typology is a registered expander.

    Args:
        truss_type: typology name from the design

    Returns:
        tuple: one "truss_type" failure, or empty on pass
    """
    if truss_type in EXPANDERS:
        return ()
    supported = ", ".join(sorted(EXPANDERS))
    return (
        DRCFailure(
            check="truss_type",
            reason=f"unsupported truss_type {truss_type!r}; supported: {supported}",
        ),
    )


def check_n_bays(n_bays: int) -> tuple[DRCFailure, ...]:
    """Check that the bay count is an integer within the allowed window.

    Assumptions:
        1. Strict integer typing — 8.0 fails; coercing numeric strings or
           whole floats is the schema layer's job, not DRC's.

    Args:
        n_bays: number of bottom-chord bays from the design

    Returns:
        tuple: one "n_bays" failure, or empty on pass
    """
    if not isinstance(n_bays, int):
        return (
            DRCFailure(
                check="n_bays",
                reason=f"n_bays must be an integer, got {n_bays!r}",
            ),
        )
    if not (N_BAYS_MIN <= n_bays <= N_BAYS_MAX):
        return (
            DRCFailure(
                check="n_bays",
                reason=(
                    f"n_bays must be within [{N_BAYS_MIN}, {N_BAYS_MAX}], got {n_bays}"
                ),
            ),
        )
    return ()


def check_depth(
    span_ft: float, depth_ft: float, depth_limit_ft: float | None
) -> tuple[DRCFailure, ...]:
    """Check the depth ratio window and any stated depth limit.

    Assumptions:
        1. The window comparison is written as not (lo <= d <= hi) so a NaN
           depth fails rather than slipping through inverted comparisons.
        2. The ratio window and the stated limit are independent rules and
           may both fail on the same design.

    Args:
        span_ft: truss span in feet, from the trusted instance
        depth_ft: truss depth in feet from the design
        depth_limit_ft: stated depth limit in feet, or None when unstated

    Returns:
        tuple: up to two "depth" failures, or empty on pass
    """
    failures: list[DRCFailure] = []
    depth_min_ft = span_ft * DEPTH_SPAN_FRACTION_MIN
    depth_max_ft = span_ft * DEPTH_SPAN_FRACTION_MAX
    if not (depth_min_ft <= depth_ft <= depth_max_ft):
        failures.append(
            DRCFailure(
                check="depth",
                reason=(
                    f"depth_ft must be within [span/25, span/4] = "
                    f"[{depth_min_ft:.3g}, {depth_max_ft:.3g}] ft, "
                    f"got {depth_ft!r}"
                ),
            )
        )
    if depth_limit_ft is not None and not (depth_ft <= depth_limit_ft):
        failures.append(
            DRCFailure(
                check="depth",
                reason=(
                    f"depth_ft {depth_ft!r} exceeds stated depth limit "
                    f"{depth_limit_ft!r} ft"
                ),
            )
        )
    return tuple(failures)


def check_sections(
    sections_by_group: Mapping[str, str], required_groups: tuple[str, ...]
) -> tuple[DRCFailure, ...]:
    """Check that provided designations are valid and required groups covered.

    Assumptions:
        1. Every provided designation is validated even for groups the
           geometry never uses, per vision section 3: any provided value
           must be a valid catalog value.

    Args:
        sections_by_group: member group name mapped to its section
            designation
        required_groups: groups the expanded geometry actually uses; empty
            when geometry could not be expanded

    Returns:
        tuple: one "sections" failure per unknown designation or uncovered
            group, or empty on pass
    """
    failures: list[DRCFailure] = []
    for group, name in sections_by_group.items():
        if not has_section(name):
            failures.append(
                DRCFailure(
                    check="sections",
                    reason=(
                        f"unknown section designation {name!r} "
                        f"for member group {group!r}"
                    ),
                )
            )
    for group in required_groups:
        if group not in sections_by_group:
            failures.append(
                DRCFailure(
                    check="sections",
                    reason=f"no section assigned to member group {group!r}",
                )
            )
    return tuple(failures)


def check_slenderness(
    geometry: TrussGeometry, sections_by_group: Mapping[str, str]
) -> tuple[DRCFailure, ...]:
    """Check every member's L/r against the slenderness limit.

    Assumptions:
        1. Emits a single failure citing the worst member plus the offender
           count rather than one failure per member, keeping calibration
           histograms clean.
        2. Uses Section.r_min_in (governing radius) with the geometric
           length as unbraced length (K = 1.0), consistent with Member.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation; every used group must resolve (see
            groups_resolvable)

    Returns:
        tuple: one "slenderness" failure citing the worst member, or empty
            on pass
    """
    worst_ratio = 0.0
    worst_member_id = -1
    worst_group = ""
    offenders = 0
    for member in geometry.members:
        section = get_section(sections_by_group[member.group])
        ratio = member.length_ft * INCHES_PER_FOOT / section.r_min_in
        if ratio > SLENDERNESS_LIMIT:
            offenders += 1
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_member_id = member.id
            worst_group = member.group
    if offenders == 0:
        return ()
    return (
        DRCFailure(
            check="slenderness",
            reason=(
                f"slenderness L/r {worst_ratio:.1f} exceeds "
                f"{SLENDERNESS_LIMIT:.0f} on member {worst_member_id} "
                f"({worst_group}); {offenders} member(s) over the limit"
            ),
        ),
    )


def check_bay_aspect(
    span_ft: float, n_bays: int, depth_ft: float
) -> tuple[DRCFailure, ...]:
    """Check the depth-to-bay-width aspect ratio window.

    Args:
        span_ft: truss span in feet, from the trusted instance
        n_bays: number of bottom-chord bays; must be a positive integer
        depth_ft: truss depth in feet from the design; must be positive

    Returns:
        tuple: one "bay_aspect" failure, or empty on pass
    """
    bay_width_ft = span_ft / n_bays
    aspect = depth_ft / bay_width_ft
    if BAY_ASPECT_MIN <= aspect <= BAY_ASPECT_MAX:
        return ()
    return (
        DRCFailure(
            check="bay_aspect",
            reason=(
                f"bay aspect depth/bay_width = {aspect:.3g} must be within "
                f"[{BAY_ASPECT_MIN}, {BAY_ASPECT_MAX}]"
            ),
        ),
    )


def run_drc(
    truss_type: str,
    span_ft: float,
    n_bays: int,
    depth_ft: float,
    sections_by_group: Mapping[str, str],
    depth_limit_ft: float | None = None,
) -> DRCResult:
    """Run the five vision section 6 design-rule checks on a design.

    Takes raw design fields and expands geometry internally (gated on
    constructibility) per the contract resolved in docs/plans/drc-layer.md.
    Collects all failed checks with anti-cascade gating rather than
    stopping at the first failure.

    Assumptions:
        1. Never raises on model-chosen fields (truss_type, n_bays, depth,
           sections) — those become DRCFailures; raises only on a
           nonpositive span, which comes from the trusted instance and so
           indicates a harness bug.

    Args:
        truss_type: typology name from the design
        span_ft: truss span in feet, from the trusted instance
        n_bays: number of bottom-chord bays from the design
        depth_ft: truss depth in feet from the design
        sections_by_group: member group name mapped to its section
            designation, same shape as loads.build_load_cases consumes
        depth_limit_ft: stated depth limit in feet from the instance, or
            None when no limit is stated

    Returns:
        DRCResult: all failed checks, empty on a fully passing design

    Raises:
        ValueError: if span_ft is not positive.
    """
    if span_ft <= 0:
        raise ValueError(f"span_ft must be positive, got {span_ft!r}")

    failures: list[DRCFailure] = []
    failures.extend(check_truss_type(truss_type))
    failures.extend(check_n_bays(n_bays))
    failures.extend(check_depth(span_ft, depth_ft, depth_limit_ft))

    geometry = (
        expand(truss_type, span_ft, n_bays, depth_ft)
        if can_expand(truss_type, span_ft, n_bays, depth_ft)
        else None
    )
    required_groups = geometry_groups(geometry) if geometry is not None else ()
    failures.extend(check_sections(sections_by_group, required_groups))
    if geometry is not None and groups_resolvable(geometry, sections_by_group):
        failures.extend(check_slenderness(geometry, sections_by_group))
    if isinstance(n_bays, int) and n_bays >= 1 and depth_ft > 0:
        failures.extend(check_bay_aspect(span_ft, n_bays, depth_ft))
    return DRCResult(failures=tuple(failures))
