"""Member tension yielding and compression buckling design capacity per AISC."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from trussRL.catalog import Section, get_section
from trussRL.expander import TrussGeometry

FY_KSI = 50.0
PHI = 0.9
E_KSI = 29_000.0
INCHES_PER_FOOT = 12.0
SLENDERNESS_BOUNDARY = 4.71 * math.sqrt(E_KSI / FY_KSI)
WALL_LAMBDA_R = 1.40 * math.sqrt(E_KSI / FY_KSI)
C1_STIFFENED = 0.20
C2_STIFFENED = 1.38


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
    return PHI * FY_KSI * section.area_in2


def has_slender_walls(section: Section) -> bool:
    """Report whether either wall of a section exceeds the nonslender limit.

    Args:
        section: catalog section carrying the width-to-thickness ratios
            ``b/tdes`` and ``h/tdes`` in its properties mapping

    Returns:
        bool: True if the governing wall width-to-thickness ratio exceeds
            the AISC Table B4.1a Case 6 limit 1.40 * sqrt(E / Fy)
    """
    governing = max(section.properties["b/tdes"], section.properties["h/tdes"])
    return governing > WALL_LAMBDA_R


def effective_wall_width_in(
    width_in: float, width_to_thickness: float, fcr_ksi: float
) -> float:
    """Effective width of one stiffened HSS wall per AISC Section E7.

    Assumptions:
        1. Uses the Table E7.1 case (b) imperfection constants for walls of
           square and rectangular HSS (c1 = 0.20, c2 = 1.38).
        2. Fcr has already been computed on gross section properties; when
           the wall satisfies lambda <= lambda_r * sqrt(Fy / Fcr) the full
           width is effective (Equation E7-2), otherwise the reduced width
           of Equation E7-3 applies with Fel from Equation E7-5.

    Args:
        width_in: flat wall width b in inches
        width_to_thickness: wall slenderness lambda = b / t
        fcr_ksi: critical stress Fcr in ksi from the gross-section
            buckling check

    Returns:
        float: effective width be in inches
    """
    if width_to_thickness <= WALL_LAMBDA_R * math.sqrt(FY_KSI / fcr_ksi):
        return width_in
    fel_ksi = (C2_STIFFENED * WALL_LAMBDA_R / width_to_thickness) ** 2 * FY_KSI
    stress_ratio = math.sqrt(fel_ksi / fcr_ksi)
    return width_in * (1.0 - C1_STIFFENED * stress_ratio) * stress_ratio


def slender_compression_capacity_kip(section: Section, fcr_ksi: float) -> float:
    """Design compression capacity of a slender-walled HSS per AISC Section E7.

    Assumptions:
        1. Fcr has already been computed per Chapter E3 on gross section
           properties; Section E7 only reduces the area, per Equation E7-1
           Pn = Fcr * Ae.
        2. The section is a rectangular HSS with two walls of each flat
           width, so Ae = Ag - 2 * tdes * ((b - be) + (h - he)).

    Args:
        section: catalog section carrying flat widths ``b`` and ``h``, the
            design wall thickness ``tdes``, and the ratios ``b/tdes`` and
            ``h/tdes`` in its properties mapping
        fcr_ksi: critical stress Fcr in ksi from the gross-section
            buckling check

    Returns:
        float: phi * Fcr * Ae in kips
    """
    b_in = section.properties["b"]
    h_in = section.properties["h"]
    t_in = section.properties["tdes"]
    be_in = effective_wall_width_in(b_in, section.properties["b/tdes"], fcr_ksi)
    he_in = effective_wall_width_in(h_in, section.properties["h/tdes"], fcr_ksi)
    ae_in2 = section.area_in2 - 2.0 * t_in * ((b_in - be_in) + (h_in - he_in))
    return PHI * fcr_ksi * ae_in2


def compression_capacity_kip(section: Section, length_ft: float) -> float:
    """Design compression capacity by Chapter E3 flexural buckling.

    Assumptions:
        1. K = 1.0 and the unbraced length equals the member length in both
           planes; the governing radius is the section's r_min.
        2. Fcr is computed per E3 on gross section properties. Sections with
           slender walls (Table B4.1a Case 6) are then reduced per Section
           E7 via slender_compression_capacity_kip; nonslender sections use
           the full gross area.

    Args:
        section: catalog section
        length_ft: member unbraced length in feet

    Returns:
        float: phi * Pn in kips, elastic or inelastic buckling per E3, with
            the E7 effective-area reduction applied when walls are slender
    """
    kl_over_r = length_ft * INCHES_PER_FOOT / section.r_min_in
    if kl_over_r == 0.0:
        fcr_ksi = FY_KSI
    elif kl_over_r <= SLENDERNESS_BOUNDARY:
        fe_ksi = math.pi**2 * E_KSI / kl_over_r**2
        fcr_ksi = 0.658 ** (FY_KSI / fe_ksi) * FY_KSI
    else:
        fe_ksi = math.pi**2 * E_KSI / kl_over_r**2
        fcr_ksi = 0.877 * fe_ksi
    if has_slender_walls(section):
        return slender_compression_capacity_kip(section, fcr_ksi)
    return PHI * fcr_ksi * section.area_in2


def member_capacities(
    geometry: TrussGeometry, sections_by_group: Mapping[str, str]
) -> tuple[MemberCapacity, ...]:
    """Compute tension and compression capacities for every member.

    Args:
        geometry: expanded truss geometry
        sections_by_group: member group name mapped to its section designation

    Returns:
        tuple: one MemberCapacity per member, ordered by member id

    Raises:
        ValueError: if a geometry group has no section assignment or a
            designation is not a catalog entry.
    """
    capacities: list[MemberCapacity] = []
    for member in geometry.members:
        if member.group not in sections_by_group:
            raise ValueError(f"no section assigned for group {member.group!r}")
        section = get_section(sections_by_group[member.group])
        capacities.append(
            MemberCapacity(
                member_id=member.id,
                tension_kip=tension_capacity_kip(section),
                compression_kip=compression_capacity_kip(section, member.length_ft),
            )
        )
    return tuple(capacities)
