"""Catalog unit tests (starting.md Module 1 "output to verify").

Spot-checks section rows by hand against the AISC Shape Database v16,
asserts the cost rule and lookup semantics, and re-derives the catalog
dict from the frozen CSV independently to guard against hand-edits or
stale regeneration of the generated module.
"""

import csv
from pathlib import Path

import pytest

from trussRL.catalog import (
    SECTIONS,
    Section,
    canonical_designation,
    get_section,
    has_section,
    list_sections,
)
from trussRL.catalog_data import CATALOG_DATA, COST_PER_LB_USD

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "HSS_subset.csv"

DROP_COLUMNS = ("Type", "EDI_Std_Nomenclature", "AISC_Manual_Label", "T_F")


def test_catalog_has_forty_sections() -> None:
    assert len(CATALOG_DATA) == 40
    assert len(SECTIONS) == 40
    assert len(list_sections()) == 40


def test_spot_check_hss4x4x1_4_against_aisc_v16() -> None:
    section = get_section("HSS4X4X1/4")
    assert section.weight_lb_per_ft == 12.21
    assert section.area_in2 == 3.37
    assert section.rx_in == 1.52
    assert section.ry_in == 1.52
    assert section.r_min_in == 1.52


def test_spot_check_hss6x6x3_8_against_aisc_v16() -> None:
    section = get_section("HSS6X6X3/8")
    assert section.weight_lb_per_ft == 27.48
    assert section.area_in2 == 7.58
    assert section.rx_in == 2.28
    assert section.ry_in == 2.28
    assert section.properties["Ix"] == 39.5


def test_spot_check_hss34x10x1_against_aisc_v16() -> None:
    section = get_section("HSS34X10X1")
    assert section.weight_lb_per_ft == 277.07
    assert section.area_in2 == 76.2
    assert section.rx_in == 11.2
    assert section.ry_in == 4.19


def test_rectangular_r_min_is_weak_axis() -> None:
    section = get_section("HSS6X4X3/8")
    assert section.rx_in == 2.14
    assert section.ry_in == 1.55
    assert section.r_min_in == 1.55
    assert section.r_min_in < section.rx_in


def test_cost_rule_for_all_sections() -> None:
    assert COST_PER_LB_USD == 1.60
    for section in list_sections():
        assert section.cost_per_ft_usd == pytest.approx(
            section.weight_lb_per_ft * 1.60, abs=1e-12
        )


def test_core_properties_positive() -> None:
    for section in list_sections():
        assert section.weight_lb_per_ft > 0
        assert section.area_in2 > 0
        assert section.rx_in > 0
        assert section.ry_in > 0
        assert section.cost_per_ft_usd > 0
        assert section.r_min_in == min(section.rx_in, section.ry_in)


def test_lookup_is_case_insensitive_and_stripped() -> None:
    assert canonical_designation("  hss6x6x3/8 ") == "HSS6X6X3/8"
    assert has_section("hss6x6x3/8")
    assert has_section("  HSS6X6X3/8  ")
    section = get_section(" hss6x6x3/8 ")
    assert section.designation == "HSS6X6X3/8"
    assert section is get_section("HSS6X6X3/8")


def test_unknown_designation_raises() -> None:
    assert not has_section("HSS99X99X9")
    with pytest.raises(ValueError, match="unknown section designation"):
        get_section("HSS99X99X9")


def test_edi_designation_not_accepted() -> None:
    assert not has_section("HSS4X4X.250")
    with pytest.raises(ValueError, match="unknown section designation"):
        get_section("HSS4X4X.250")


def test_sections_expose_full_properties() -> None:
    for designation, section in SECTIONS.items():
        assert isinstance(section, Section)
        assert section.designation == designation
        assert section.properties == CATALOG_DATA[designation]


def recompute_catalog_from_csv() -> dict[str, dict[str, float]]:
    """Independently re-derive the catalog dict from the frozen CSV.

    Mirrors the scripts/make_catalog.py derivation with a separate
    implementation: imperial (first-occurrence) columns only, identity and
    flag columns dropped, float-parsable fields kept, and the derived
    cost_per_ft_usd appended.

    Args: None

    Returns:
        dict: canonical designation mapped to its property dict, in CSV
            row order
    """
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    columns: dict[str, int] = {}
    for index, name in enumerate(header):
        if name and name not in DROP_COLUMNS and name not in columns:
            columns[name] = index
    label_index = header.index("AISC_Manual_Label")
    catalog: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        properties: dict[str, float] = {}
        for name, index in columns.items():
            try:
                properties[name] = float(row[index])
            except ValueError:
                continue
        properties["cost_per_ft_usd"] = properties["W"] * COST_PER_LB_USD
        catalog[row[label_index].strip().upper()] = properties
    return catalog


def test_generated_module_in_sync_with_csv() -> None:
    recomputed = recompute_catalog_from_csv()
    assert recomputed == CATALOG_DATA
    assert list(recomputed) == list(CATALOG_DATA)
