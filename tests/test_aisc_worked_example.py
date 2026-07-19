"""AISC Design Example E.7 (WT7x34 compression member) as a published anchor.

Reproduces AISC V16.0 Companion Design Example E.7: a WT7x34 in ASTM
A992/A992M steel with pinned ends and KL = 20 ft has an available compressive
strength of phi_c * Pn = 128 kips, governed by flexural buckling about the
x-x axis (Fe = 16.2 ksi, Fcr = 0.877 * Fe = 14.2 ksi). Because the governing
mode is r_min flexural buckling, our E3-only check reproduces the published
result; the example's torsional and flexural-torsional E4 modes (29.5 ksi and
30.0 ksi) are not modeled and do not control here.

The WT shape is outside the frozen HSS catalog, so the Section is built
manually. Its stem (d/tw = 16.9) and flange (bf/2tf = 6.97) are nonslender
per the example, so the capacity check must take the full-gross-area path;
both ratios also sit below the HSS wall limit used by has_slender_walls, so
the gate agrees with the example's conclusion. This example exercises the
nonslender E3 curve and the slenderness gate; the E7 reduction branch is
anchored by the HSS hand calculation in test_capacity.py.
"""

import pytest

from trussRL.capacity import compression_capacity_kip, has_slender_walls
from trussRL.catalog import Section

WT7X34 = Section(
    designation="WT7X34",
    weight_lb_per_ft=34.0,
    area_in2=10.0,
    rx_in=1.81,
    ry_in=2.46,
    cost_per_ft_usd=0.0,
    properties={"b/tdes": 6.97, "h/tdes": 16.9},
)


def test_wt7x34_has_no_slender_elements() -> None:
    assert not has_slender_walls(WT7X34)


def test_wt7x34_capacity_matches_published_example() -> None:
    # Example E.7: KL/r = 240/1.81 = 133, Fe = 16.2 ksi, Fy/Fe > 2.25,
    # Fcr = 0.877 * 16.2 = 14.2 ksi, Pn = 142 kips, phi_c * Pn = 128 kips.
    assert compression_capacity_kip(WT7X34, 20.0) == pytest.approx(128.0, rel=1e-2)
