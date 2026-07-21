"""Unit tests for the seeded random-design sampler.

Determinism, bound respect across all three depth-limit variants, window
clamping, and endpoint coverage — all pure sampling, no solver runs.
"""

import random

from trussRL.baselines.random_design import (
    SECTION_NAMES,
    TRUSS_TYPES,
    depth_window_ft,
    sample_design,
)
from trussRL.drc import (
    DEPTH_SPAN_FRACTION_MAX,
    DEPTH_SPAN_FRACTION_MIN,
    N_BAYS_MAX,
    N_BAYS_MIN,
)
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase

GRAVITY_CASE = LoadCase(w_kip_per_ft=-2.0, level="bottom")


def make_instance(depth_limit_ft: float | None) -> TrussInstance:
    """Build a 100 ft gravity instance with the given stated depth limit.

    Args:
        depth_limit_ft: stated depth limit in feet, or None for unstated

    Returns:
        TrussInstance: a single-case gravity instance
    """
    return TrussInstance(
        span_ft=100.0,
        load_cases=(GRAVITY_CASE,),
        depth_limit_ft=depth_limit_ft,
        defl_denom=360,
        cost_ref_usd=None,
    )


def test_sample_design_is_seed_deterministic() -> None:
    instance = make_instance(None)
    first = [sample_design(random.Random(7), instance) for _ in range(1)]
    rng_a = random.Random(123)
    rng_b = random.Random(123)
    designs_a = [sample_design(rng_a, instance) for _ in range(50)]
    designs_b = [sample_design(rng_b, instance) for _ in range(50)]
    assert designs_a == designs_b
    assert first[0] == sample_design(random.Random(7), instance)


def test_different_seeds_differ() -> None:
    instance = make_instance(None)
    designs_a = [sample_design(random.Random(1), instance) for _ in range(10)]
    designs_b = [sample_design(random.Random(2), instance) for _ in range(10)]
    assert designs_a != designs_b


def test_depth_window_no_stated_limit() -> None:
    instance = make_instance(None)
    low, high = depth_window_ft(instance)
    assert low == 100.0 * DEPTH_SPAN_FRACTION_MIN
    assert high == 100.0 * DEPTH_SPAN_FRACTION_MAX


def test_depth_window_clamped_by_tighter_stated_limit() -> None:
    instance = make_instance(8.0)
    low, high = depth_window_ft(instance)
    assert low == 100.0 * DEPTH_SPAN_FRACTION_MIN
    assert high == 8.0


def test_depth_window_ignores_looser_stated_limit() -> None:
    instance = make_instance(30.0)
    _, high = depth_window_ft(instance)
    assert high == 100.0 * DEPTH_SPAN_FRACTION_MAX


def test_samples_respect_bounds_across_depth_limit_variants() -> None:
    for depth_limit_ft in (None, 100.0 / 6, 8.0):
        instance = make_instance(depth_limit_ft)
        low, high = depth_window_ft(instance)
        rng = random.Random(42)
        for _ in range(300):
            design = sample_design(rng, instance)
            assert design.truss_type in TRUSS_TYPES
            assert N_BAYS_MIN <= design.n_bays <= N_BAYS_MAX
            assert low <= design.depth_ft <= high
            assert design.top_chord in SECTION_NAMES
            assert design.bottom_chord in SECTION_NAMES
            assert design.diagonals in SECTION_NAMES
            assert design.verticals in SECTION_NAMES


def test_n_bays_endpoints_are_reachable() -> None:
    instance = make_instance(None)
    rng = random.Random(0)
    seen = {sample_design(rng, instance).n_bays for _ in range(500)}
    assert N_BAYS_MIN in seen
    assert N_BAYS_MAX in seen
