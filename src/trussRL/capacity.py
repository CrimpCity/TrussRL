"""How much each member can carry, per AISC.

Tension yielding and compression buckling from section and length alone —
pure arithmetic, no solver state, no load cases. These capacities are the
denominators of the reward layer's utilization checks.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from trussRL.catalog import Section
from trussRL.expander import TrussGeometry

FY_KSI = 50.0
PHI = 0.9


@dataclass(frozen=True)
class MemberCapacity:
    """Design capacities for one member, from its section and length alone.

    Attributes:
        member_id: id of the member within its geometry
        tension_kip: design tension capacity phi * Pn in kips,
            gross-section yielding
        compression_kip: design compression capacity phi * Pn in kips,
            Chapter E3 flexural buckling from the member's L/r; stored
            positive even though compression demands are negative
    """

    member_id: int
    tension_kip: float
    compression_kip: float


def tension_capacity_kip(section: Section) -> float:
    """Design tension capacity of a section by gross-section yielding.

    Args:
        section: catalog section

    Returns:
        float: phi * Fy * A in kips
    """
    raise NotImplementedError("Unit 4 (outline.md): AISC tension capacity")


def compression_capacity_kip(section: Section, length_ft: float) -> float:
    """Design compression capacity by Chapter E3 flexural buckling.

    Assumptions:
        1. K = 1.0 and the unbraced length equals the member length in both
           planes; the governing radius is the section's r_min.
        2. No Chapter E7 effective-area logic: the frozen catalog is
           pre-filtered to nonslender walls, so slenderness is handled by
           catalog curation, not code.

    Args:
        section: catalog section
        length_ft: member unbraced length in feet

    Returns:
        float: phi * Pn in kips, elastic or inelastic buckling per E3
    """
    raise NotImplementedError("Unit 4 (outline.md): Chapter E3 compression capacity")


def member_capacities(
    geometry: TrussGeometry, sections_by_group: Mapping[str, str]
) -> tuple[MemberCapacity, ...]:
    """Compute tension and compression capacities for every member.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section
            designation

    Returns:
        tuple: one MemberCapacity per member, ordered by member id

    Raises:
        ValueError: if a geometry group has no section assignment or a
            designation is not a catalog entry.
    """
    raise NotImplementedError("Unit 4 (outline.md): per-member AISC capacities")
