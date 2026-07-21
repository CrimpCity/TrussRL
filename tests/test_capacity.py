"""AISC capacity unit tests (outline.md Unit 4).

Anchors the compression curve to Table 4-22-style critical stresses at
Fy = 50 ksi, checks the tension and compression formulas at short and
slender L/r including continuity at the E3 branch boundary, and asserts
the per-member assembly matches direct scalar calls on real Warren
geometry. The phi*Fcr anchor values are derived from the E3 formulas
(Table 4-22 is E3 tabulated and rounded); verify them against the
printed AISC Manual Table 4-22 when a copy is at hand.
"""

import itertools
import math

import pytest

from trussRL.capacity import (
    E_KSI,
    FY_KSI,
    PHI,
    SLENDERNESS_BOUNDARY,
    compression_capacity_kip,
    effective_wall_width_in,
    has_slender_walls,
    member_capacities,
    tension_capacity_kip,
)
from trussRL.catalog import get_section
from trussRL.expander import expand_warren

SECTION = get_section("HSS6X6X3/8")
SLENDER_SECTION = get_section("HSS12X12X1/4")

SECTIONS_BY_GROUP = {
    "bottom_chord": "HSS6X6X3/8",
    "top_chord": "HSS6X6X3/8",
    "diagonals": "HSS4X4X1/4",
}


def length_ft_for_slenderness(kl_over_r: float) -> float:
    """Back-solve the member length that yields a target KL/r for SECTION.

    Args:
        kl_over_r: target slenderness ratio

    Returns:
        float: length in feet giving that KL/r with SECTION's r_min
    """
    return kl_over_r * SECTION.r_min_in / 12.0


@pytest.mark.parametrize(
    ("kl_over_r", "phi_fcr_ksi"),
    [
        (40.0, 40.0),
        (80.0, 28.2),
        (120.0, 15.7),
        (160.0, 8.82),
        (200.0, 5.65),
    ],
)
def test_critical_stress_matches_table_4_22(
    kl_over_r: float, phi_fcr_ksi: float
) -> None:
    length_ft = length_ft_for_slenderness(kl_over_r)
    capacity = compression_capacity_kip(SECTION, length_ft)
    assert capacity / SECTION.area_in2 == pytest.approx(phi_fcr_ksi, rel=5e-3)


def test_tension_is_phi_fy_a() -> None:
    # HSS6X6X3/8: A = 7.58 in^2, so phi*Pn = 0.9 * 50 * 7.58 = 341.1 kips.
    assert tension_capacity_kip(SECTION) == pytest.approx(341.1)


def test_short_column_approaches_squash_load() -> None:
    squash_kip = PHI * FY_KSI * SECTION.area_in2
    assert compression_capacity_kip(SECTION, 0.0) == pytest.approx(squash_kip)
    short_kip = compression_capacity_kip(SECTION, 0.01)
    assert short_kip < squash_kip
    assert short_kip == pytest.approx(squash_kip, rel=1e-3)


def test_slender_column_matches_elastic_buckling() -> None:
    kl_over_r = 180.0
    fe_ksi = math.pi**2 * E_KSI / kl_over_r**2
    expected_kip = PHI * 0.877 * fe_ksi * SECTION.area_in2
    length_ft = length_ft_for_slenderness(kl_over_r)
    assert compression_capacity_kip(SECTION, length_ft) == pytest.approx(expected_kip)


def test_curve_is_continuous_at_branch_boundary() -> None:
    just_inelastic = compression_capacity_kip(
        SECTION, length_ft_for_slenderness(SLENDERNESS_BOUNDARY - 1e-6)
    )
    just_elastic = compression_capacity_kip(
        SECTION, length_ft_for_slenderness(SLENDERNESS_BOUNDARY + 1e-6)
    )
    # E3's branches are only approximately continuous: the 4.71 boundary
    # constant and the 0.658/0.877 coefficients are rounded, leaving a
    # deliberate sub-0.1% seam where Fcr ~ 0.39 * Fy.
    assert just_inelastic == pytest.approx(just_elastic, rel=1e-3)
    fe_boundary_ksi = math.pi**2 * E_KSI / SLENDERNESS_BOUNDARY**2
    inelastic_fcr_ksi = just_inelastic / (PHI * SECTION.area_in2)
    elastic_fcr_ksi = just_elastic / (PHI * SECTION.area_in2)
    expected_inelastic = 0.658 ** (FY_KSI / fe_boundary_ksi) * FY_KSI
    assert inelastic_fcr_ksi == pytest.approx(expected_inelastic, rel=1e-5)
    assert elastic_fcr_ksi == pytest.approx(0.877 * fe_boundary_ksi, rel=1e-5)


def test_compression_strictly_decreases_with_length() -> None:
    lengths_ft = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    capacities = [compression_capacity_kip(SECTION, length) for length in lengths_ft]
    assert all(a > b for a, b in itertools.pairwise(capacities))


def test_has_slender_walls_flags_thin_walled_sections() -> None:
    assert has_slender_walls(SLENDER_SECTION)
    assert not has_slender_walls(SECTION)


def test_slender_section_matches_hand_calc_e7_anchor() -> None:
    # HSS12X12X1/4 at KL = 15 ft, hand-worked per E3 then E7:
    # KL/r = 180/4.79 = 37.6, Fe = 203 ksi, Fcr = 45.1 ksi (inelastic).
    # lambda = 48.5 > lambda_r * sqrt(Fy/Fcr) = 35.5, so both wall pairs
    # reduce: Fel = (1.38 * 33.72 / 48.5)^2 * 50 = 46.0 ksi, be = 9.11 in,
    # Ae = 10.8 - 4(0.233)(11.3 - 9.11) = 8.76 in^2,
    # phi * Pn = 0.9(45.1)(8.76) = 355 kips.
    assert compression_capacity_kip(SLENDER_SECTION, 15.0) == pytest.approx(
        355.4, rel=1e-3
    )


def test_slender_section_reduced_below_gross_at_short_length() -> None:
    squash_kip = PHI * FY_KSI * SLENDER_SECTION.area_in2
    assert compression_capacity_kip(SLENDER_SECTION, 0.0) < squash_kip


def test_slender_reduction_vanishes_at_long_length() -> None:
    # At KL/r = 150 the elastic Fcr = 0.877 * Fe = 11.2 ksi is low enough
    # that lambda = 48.5 <= lambda_r * sqrt(Fy/Fcr) = 71.4, so E7-2 leaves
    # the full width effective and the capacity equals the gross E3 value.
    kl_over_r = 150.0
    length_ft = kl_over_r * SLENDER_SECTION.r_min_in / 12.0
    fe_ksi = math.pi**2 * E_KSI / kl_over_r**2
    gross_kip = PHI * 0.877 * fe_ksi * SLENDER_SECTION.area_in2
    assert compression_capacity_kip(SLENDER_SECTION, length_ft) == pytest.approx(
        gross_kip
    )


def test_slender_compression_strictly_decreases_with_length() -> None:
    lengths_ft = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0]
    capacities = [
        compression_capacity_kip(SLENDER_SECTION, length) for length in lengths_ft
    ]
    assert all(a > b for a, b in itertools.pairwise(capacities))


def test_effective_wall_width_bounded_by_full_width() -> None:
    width_in = SLENDER_SECTION.properties["b"]
    ratio = SLENDER_SECTION.properties["b/tdes"]
    for fcr_ksi in [1.0, 5.0, 11.2, 20.0, 30.0, 45.1, 50.0]:
        be_in = effective_wall_width_in(width_in, ratio, fcr_ksi)
        assert 0.0 < be_in <= width_in


def test_member_capacities_match_scalar_calls_in_id_order() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    capacities = member_capacities(geometry, SECTIONS_BY_GROUP)
    assert [capacity.member_id for capacity in capacities] == [
        member.id for member in geometry.members
    ]
    for member, capacity in zip(geometry.members, capacities, strict=True):
        section = get_section(SECTIONS_BY_GROUP[member.group])
        assert capacity.tension_kip == tension_capacity_kip(section)
        assert capacity.compression_kip == compression_capacity_kip(
            section, member.length_ft
        )


def test_group_mates_share_capacities_but_groups_differ() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    capacities = member_capacities(geometry, SECTIONS_BY_GROUP)
    by_group: dict[str, set[tuple[float, float]]] = {}
    for member, capacity in zip(geometry.members, capacities, strict=True):
        by_group.setdefault(member.group, set()).add(
            (capacity.tension_kip, capacity.compression_kip)
        )
    assert all(len(values) == 1 for values in by_group.values())
    assert by_group["diagonals"] != by_group["top_chord"]


def test_missing_group_assignment_raises() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    incomplete = {k: v for k, v in SECTIONS_BY_GROUP.items() if k != "diagonals"}
    with pytest.raises(ValueError, match="diagonals"):
        member_capacities(geometry, incomplete)


def test_unknown_designation_raises() -> None:
    geometry = expand_warren(span_ft=96.0, n_bays=8, depth_ft=8.0)
    bogus = {**SECTIONS_BY_GROUP, "diagonals": "HSS9X9X9"}
    with pytest.raises(ValueError, match="unknown section designation"):
        member_capacities(geometry, bogus)
