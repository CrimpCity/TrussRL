"""Generator unit tests (Unit 7).

Asserts seed reproducibility, config-range compliance, depth-limit variant
resolution, and the exit criterion: generated instances round-trip through
expand() + run_drc() without error across the full knob space.
"""

import math
import random

import pytest

from trussRL.drc import run_drc
from trussRL.expander import expand
from trussRL.generator import (
    DEPTH_LIMIT_VARIANTS,
    GeneratorConfig,
    depth_limit_ft,
    generate_instance,
    generate_instances,
    sample_grid,
)
from trussRL.instance import TrussInstance

SEEDS = range(50)


def reference_design(instance: TrussInstance) -> tuple[int, float, dict[str, str]]:
    """Build a deterministic feasible recipe for a generated instance.

    Assumptions:
        1. Provably passes DRC over the whole knob space: depth sits in
           [~span/14, span/12], a subset of the DRC window [span/25,
           span/4], and respects any stated limit since generous/tight
           limits are always >= span/14; 8 bays keeps bay aspect
           8*depth/span in [~0.57, 0.67], inside [0.3, 3.5]; HSS6X6X3/8
           keeps worst L/r comfortably under 200 at span 120.

    Args:
        instance: the generated problem instance

    Returns:
        tuple: (n_bays, depth_ft, sections_by_group) for a design that
            passes DRC for this instance
    """
    depth_cap_ft = (
        instance.depth_limit_ft if instance.depth_limit_ft is not None else math.inf
    )
    depth_ft = min(instance.span_ft / 12.0, depth_cap_ft)
    sections_by_group = {
        "top_chord": "HSS6X6X3/8",
        "bottom_chord": "HSS6X6X3/8",
        "diagonals": "HSS6X6X3/8",
    }
    return 8, depth_ft, sections_by_group


def test_same_seed_reproduces_instance() -> None:
    for seed in (0, 1, 42, 12345):
        assert generate_instance(seed) == generate_instance(seed)


def test_different_seeds_produce_variety() -> None:
    instances = [generate_instance(seed) for seed in SEEDS]
    assert len({instance.span_ft for instance in instances}) > 1
    assert len({instance.depth_limit_ft for instance in instances}) > 1
    assert len({instance.defl_denom for instance in instances}) > 1


def test_instance_fields_within_config_ranges() -> None:
    for seed in range(100):
        instance = generate_instance(seed)
        assert 60.0 <= instance.span_ft <= 120.0
        assert instance.span_ft == round(instance.span_ft)
        assert len(instance.load_cases) == 1
        case = instance.load_cases[0]
        assert -4.0 <= case.w_kip_per_ft <= -0.5
        assert round(case.w_kip_per_ft * 10) == pytest.approx(case.w_kip_per_ft * 10)
        assert case.level == "bottom"
        assert case.longitudinal_kip == 0.0
        assert instance.defl_denom in (240, 360, 500)
        assert (
            instance.depth_limit_ft is None
            or instance.depth_limit_ft > instance.span_ft / 25.0
        )
        assert instance.cost_ref_usd is None


def test_depth_limit_variants() -> None:
    assert depth_limit_ft("none", 96.0) is None
    assert depth_limit_ft("generous", 96.0) == 16.0
    assert depth_limit_ft("tight", 96.0) == 7.0
    with pytest.raises(ValueError, match="unknown depth_limit variant"):
        depth_limit_ft("bogus", 96.0)


@pytest.mark.parametrize("span_ft", range(60, 121))
def test_tight_limit_always_above_drc_floor(span_ft: int) -> None:
    assert depth_limit_ft("tight", float(span_ft)) > span_ft / 25.0


class FixedRandint(random.Random):
    """A Random stand-in whose randint() always returns a fixed value."""

    def __init__(self, fixed_value: int) -> None:
        super().__init__(0)
        self.fixed_value = fixed_value

    def randint(self, a: int, b: int) -> int:
        return self.fixed_value


def test_sample_grid_endpoints_and_rounding() -> None:
    assert sample_grid(FixedRandint(0), 0.5, 4.0, 0.1) == 0.5
    n_steps = round((4.0 - 0.5) / 0.1)
    assert sample_grid(FixedRandint(n_steps), 0.5, 4.0, 0.1) == 4.0
    value = sample_grid(FixedRandint(3), 0.5, 4.0, 0.1)
    assert value == round(value, 6)
    with pytest.raises(ValueError, match="step must be positive"):
        sample_grid(FixedRandint(0), 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="max_value must be"):
        sample_grid(FixedRandint(0), 1.0, 0.0, 0.1)


def test_generate_instances_reproducible() -> None:
    assert generate_instances(7, 10) == generate_instances(7, 10)
    assert generate_instances(7, 10) != generate_instances(8, 10)
    assert len(generate_instances(7, 10)) == 10
    assert generate_instances(7, 0) == ()


def test_generated_instances_roundtrip_expand_and_drc() -> None:
    for seed in SEEDS:
        instance = generate_instance(seed)
        n_bays, depth_ft, sections_by_group = reference_design(instance)
        geometry = expand("warren", instance.span_ft, n_bays, depth_ft)
        assert geometry is not None
        result = run_drc(
            truss_type="warren",
            span_ft=instance.span_ft,
            n_bays=n_bays,
            depth_ft=depth_ft,
            sections_by_group=sections_by_group,
            depth_limit_ft=instance.depth_limit_ft,
        )
        assert result.passed, result.summary

    restricted_config = GeneratorConfig(defl_denoms=(360,))
    for seed in SEEDS:
        instance = generate_instance(seed, restricted_config)
        n_bays, depth_ft, sections_by_group = reference_design(instance)
        expand("warren", instance.span_ft, n_bays, depth_ft)
        result = run_drc(
            truss_type="warren",
            span_ft=instance.span_ft,
            n_bays=n_bays,
            depth_ft=depth_ft,
            sections_by_group=sections_by_group,
            depth_limit_ft=instance.depth_limit_ft,
        )
        assert result.passed, result.summary


def test_config_restriction_is_respected() -> None:
    config = GeneratorConfig(depth_limit_variants=("tight",), defl_denoms=(360,))
    for seed in SEEDS:
        instance = generate_instance(seed, config)
        assert instance.depth_limit_ft is not None
        assert instance.defl_denom == 360


def test_depth_limit_variants_constant_matches_default_order() -> None:
    assert DEPTH_LIMIT_VARIANTS == ("none", "generous", "tight")
