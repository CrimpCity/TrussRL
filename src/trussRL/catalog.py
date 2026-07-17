"""The frozen section catalog: every property a design needs, in one lookup.

One typed lookup from designation to area, radii of gyration, weight, and
cost is shared by self-weight, slenderness, capacity, and cost computations.
Restricting the policy to a curated catalog means nothing needs to be
invented — the catalog is itself a design-rule check.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from trussRL.catalog_data import CATALOG_DATA


@dataclass(frozen=True)
class Section:
    """One HSS shape from the frozen catalog.

    Attributes:
        designation: canonical AISC manual label, e.g. "HSS6X6X3/8"
        weight_lb_per_ft: nominal weight W in pounds per foot
        area_in2: cross-sectional area A in square inches
        rx_in: strong-axis radius of gyration in inches
        ry_in: weak-axis radius of gyration in inches
        cost_per_ft_usd: material cost per foot, W x COST_PER_LB_USD
        properties: full property mapping from the generated catalog,
            for consumers needing values beyond the named attributes
    """

    designation: str
    weight_lb_per_ft: float
    area_in2: float
    rx_in: float
    ry_in: float
    cost_per_ft_usd: float
    properties: Mapping[str, float]

    @property
    def r_min_in(self) -> float:
        """Governing radius of gyration for buckling checks.

        The smaller of the two radii governs KL/r slenderness because the
        unbraced length is the same in both planes.

        Args: None

        Returns:
            float: min(rx_in, ry_in) in inches
        """
        return min(self.rx_in, self.ry_in)


def canonical_designation(name: str) -> str:
    """Normalize a section designation to its canonical catalog key.

    Assumptions:
        1. Canonical form is the stripped, uppercased AISC manual label
           (e.g. "HSS6X6X3/8"); the EDI form ("HSS4X4X.250") is a
           different string and is deliberately not translated.

    Args:
        name: section designation in any letter case, possibly padded
            with whitespace

    Returns:
        str: the stripped, uppercased designation
    """
    return name.strip().upper()


def build_sections() -> dict[str, Section]:
    """Build Section records for every catalog entry.

    Args: None

    Returns:
        dict: canonical designation mapped to its Section, in catalog
            (ascending strong-axis Ix) order
    """
    sections: dict[str, Section] = {}
    for designation, properties in CATALOG_DATA.items():
        sections[designation] = Section(
            designation=designation,
            weight_lb_per_ft=properties["W"],
            area_in2=properties["A"],
            rx_in=properties["rx"],
            ry_in=properties["ry"],
            cost_per_ft_usd=properties["cost_per_ft_usd"],
            properties=properties,
        )
    return sections


SECTIONS: dict[str, Section] = build_sections()


def has_section(name: str) -> bool:
    """Report whether a designation exists in the catalog.

    Args:
        name: section designation; matched case-insensitively after
            stripping whitespace

    Returns:
        bool: True if the designation is a catalog entry
    """
    return canonical_designation(name) in SECTIONS


def get_section(name: str) -> Section:
    """Look up a Section by designation.

    Args:
        name: section designation; matched case-insensitively after
            stripping whitespace

    Returns:
        Section: the catalog record for the designation

    Raises:
        ValueError: if the designation is not a catalog entry.
    """
    key = canonical_designation(name)
    if key not in SECTIONS:
        raise ValueError(f"unknown section designation {name!r}")
    return SECTIONS[key]


def list_sections() -> tuple[Section, ...]:
    """Return every catalog Section in catalog order.

    Args: None

    Returns:
        tuple: all Section records, in catalog (ascending strong-axis Ix)
            order
    """
    return tuple(SECTIONS.values())
