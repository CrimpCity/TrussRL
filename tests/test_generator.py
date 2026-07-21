"""Unit tests for the canonical heavy-family generator.

Asserts the pre-promotion seed contract, config-range compliance, helper
validation, depth-limit behavior, and round-tripping through expand() and
run_drc() across the full knob space.
"""

import math
import random

import pytest

from trussRL.drc import run_drc
from trussRL.expander import expand
from trussRL.generator import (
    DEPTH_LIMIT_VARIANTS,
    GENERATOR_VERSION,
    GeneratorConfig,
    depth_limit_ft,
    generate_instance,
    generate_instances,
    sample_grid,
)
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase

SEEDS = range(200)


def generous_limit_ft(span_ft: float) -> float:
    """Independently calculate span/6 rounded up to the next half foot."""
    return math.ceil((span_ft / 6.0) / 0.5) * 0.5


def reference_design(instance: TrussInstance) -> tuple[int, float, dict[str, str]]:
    """Build a deterministic feasible recipe for a generated instance.

    Assumptions:
        1. Provably passes DRC over the whole knob space: depth sits in
           [~span/14, span/12], a subset of the DRC window [span/25,
           span/4], and respects any stated limit since generous/tight
           limits are always >= span/14; 8 bays keeps bay aspect
           8*depth/span in [~0.57, 0.67], inside [0.3, 3.5]; HSS6X6X3/8
           keeps worst L/r under 200 at span 160.

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


def test_representative_seeds_match_pre_promotion_outputs() -> None:
    expected = {
        0: TrussInstance(
            span_ft=144.0,
            load_cases=(LoadCase(w_kip_per_ft=-4.3, level="bottom"),),
            depth_limit_ft=None,
            defl_denom=360,
            cost_ref_usd=None,
        ),
        1: TrussInstance(
            span_ft=128.0,
            load_cases=(LoadCase(w_kip_per_ft=-4.8, level="bottom"),),
            depth_limit_ft=None,
            defl_denom=360,
            cost_ref_usd=None,
        ),
        42: TrussInstance(
            span_ft=160.0,
            load_cases=(LoadCase(w_kip_per_ft=-3.3, level="bottom"),),
            depth_limit_ft=None,
            defl_denom=500,
            cost_ref_usd=None,
        ),
        12345: TrussInstance(
            span_ft=146.0,
            load_cases=(LoadCase(w_kip_per_ft=-3.0, level="bottom"),),
            depth_limit_ft=24.5,
            defl_denom=360,
            cost_ref_usd=None,
        ),
    }
    assert {seed: generate_instance(seed) for seed in expected} == expected


def test_different_seeds_produce_variety() -> None:
    instances = [generate_instance(seed) for seed in SEEDS]
    assert instances[0] != instances[1]
    assert len({instance.span_ft for instance in instances}) > 1
    assert len({instance.depth_limit_ft for instance in instances}) > 1
    assert len({instance.defl_denom for instance in instances}) > 1


def test_instance_fields_within_config_ranges() -> None:
    config = GeneratorConfig()
    for seed in SEEDS:
        instance = generate_instance(seed)
        assert config.span_min_ft <= instance.span_ft <= config.span_max_ft
        span_step = (instance.span_ft - config.span_min_ft) / config.span_step_ft
        assert span_step == round(span_step)
        assert len(instance.load_cases) == 1
        case = instance.load_cases[0]
        load_magnitude = -case.w_kip_per_ft
        assert config.w_min_kip_per_ft <= load_magnitude <= config.w_max_kip_per_ft
        load_step = (
            load_magnitude - config.w_min_kip_per_ft
        ) / config.w_step_kip_per_ft
        assert load_step == pytest.approx(round(load_step))
        assert case.level == config.load_level
        assert case.longitudinal_kip == 0.0
        assert instance.defl_denom in config.defl_denoms
        if instance.depth_limit_ft is not None:
            assert instance.depth_limit_ft == generous_limit_ft(instance.span_ft)
        assert instance.cost_ref_usd is None


def test_default_depth_limit_variants_across_representative_seeds() -> None:
    observed_variants = set()
    for seed in SEEDS:
        instance = generate_instance(seed)
        if instance.depth_limit_ft is None:
            variant = "none"
        elif instance.depth_limit_ft == generous_limit_ft(instance.span_ft):
            variant = "generous"
        elif instance.depth_limit_ft == depth_limit_ft("tight", instance.span_ft):
            pytest.fail(f"seed {seed} generated forbidden default variant 'tight'")
        else:
            pytest.fail(
                f"seed {seed} generated unrecognized depth limit "
                f"{instance.depth_limit_ft!r} for span {instance.span_ft!r}"
            )
        observed_variants.add(variant)

    assert observed_variants == {"none", "generous"}


@pytest.mark.parametrize(
    ("variant", "span_ft", "expected_limit_ft"),
    (
        ("none", 96.0, None),
        ("generous", 96.0, 16.0),
        ("generous", 100.0, 17.0),
        ("tight", 96.0, 7.0),
        ("tight", 99.0, 7.5),
    ),
)
def test_depth_limit_variants(
    variant: str, span_ft: float, expected_limit_ft: float | None
) -> None:
    assert depth_limit_ft(variant, span_ft) == expected_limit_ft


def test_depth_limit_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown depth_limit variant"):
        depth_limit_ft("bogus", 96.0)


@pytest.mark.parametrize("span_ft", range(120, 161))
def test_tight_limit_always_above_drc_floor(span_ft: int) -> None:
    span_limit = depth_limit_ft("tight", float(span_ft))
    assert span_limit is not None
    assert span_limit > span_ft / 25.0


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
    assert sample_grid(FixedRandint(1), 0.0, 1.0, 1.0 / 3.0) == 0.333333


@pytest.mark.parametrize("step", (0.0, -0.1))
def test_sample_grid_rejects_nonpositive_step(step: float) -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        sample_grid(FixedRandint(0), 0.0, 1.0, step)


def test_sample_grid_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="max_value must be"):
        sample_grid(FixedRandint(0), 1.0, 0.0, 0.1)


def test_generate_instances_reproducible() -> None:
    assert generate_instances(7, 10) == generate_instances(7, 10)
    assert generate_instances(7, 10) != generate_instances(8, 10)
    assert len(generate_instances(7, 10)) == 10
    assert generate_instances(7, 0) == ()
    assert len(set(generate_instances(0, 20))) > 1
    with pytest.raises(ValueError, match="count must be nonnegative"):
        generate_instances(7, -1)


@pytest.mark.parametrize(
    "config",
    (
        pytest.param(GeneratorConfig(), id="default"),
        pytest.param(
            GeneratorConfig(depth_limit_variants=("tight",), defl_denoms=(360,)),
            id="tight-only",
        ),
    ),
)
def test_generated_instances_roundtrip_expand_and_drc(
    config: GeneratorConfig,
) -> None:
    for seed in SEEDS:
        instance = generate_instance(seed, config)
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


def test_fully_restricted_config_is_respected() -> None:
    config = GeneratorConfig(
        span_min_ft=137.0,
        span_max_ft=137.0,
        span_step_ft=1.0,
        w_min_kip_per_ft=4.4,
        w_max_kip_per_ft=4.4,
        w_step_kip_per_ft=0.1,
        depth_limit_variants=("generous",),
        defl_denoms=(500,),
        load_level="top",
    )
    for seed in SEEDS:
        instance = generate_instance(seed, config)
        assert instance.span_ft == 137.0
        assert instance.load_cases == (LoadCase(w_kip_per_ft=-4.4, level="top"),)
        assert instance.depth_limit_ft == generous_limit_ft(137.0)
        assert instance.defl_denom == 500
        assert instance.cost_ref_usd is None


def test_depth_limit_variants_constant_matches_default_order() -> None:
    assert DEPTH_LIMIT_VARIANTS == ("none", "generous")
    assert GeneratorConfig().depth_limit_variants == ("none", "generous")
    assert GENERATOR_VERSION == "v2"
