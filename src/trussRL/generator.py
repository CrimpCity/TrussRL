"""The generator is the dataset: seeded procedural instance creation.

Spans, loads, and limits are difficulty knobs rather than fixed examples —
infinite fresh instances, reproducible from a seed, with difficulty
schedulable during training.
"""

import math
import random
from dataclasses import dataclass

from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase

GENEROUS_DEPTH_SPAN_FRACTION = 1 / 6
TIGHT_DEPTH_SPAN_FRACTION = 1 / 14
DEPTH_LIMIT_STEP_FT = 0.5
DEPTH_LIMIT_VARIANTS = ("none", "generous", "tight")


@dataclass(frozen=True)
class GeneratorConfig:
    """Difficulty knobs for procedural instance generation (vision section 11).

    Restricting a tuple field (e.g. ``defl_denoms=(360,)``) is how training
    schedules difficulty: the generator itself always serves the full
    family, and a narrower config is how a training script asks for less
    of it.

    Attributes:
        span_min_ft: lower bound of the span grid, in feet
        span_max_ft: upper bound of the span grid, in feet
        span_step_ft: span grid step, in feet; the default whole-foot step
            yields 61 possible spans
        w_min_kip_per_ft: lower bound of the gravity load magnitude grid,
            in kips per foot; sign is applied at build time
        w_max_kip_per_ft: upper bound of the gravity load magnitude grid,
            in kips per foot
        w_step_kip_per_ft: gravity load magnitude grid step, in kips per
            foot
        depth_limit_variants: depth-limit variants eligible for sampling
        defl_denoms: deflection-limit denominators eligible for sampling
        load_level: deck chord level carrying the gravity load, "bottom"
            or "top"
    """

    span_min_ft: float = 60.0
    span_max_ft: float = 120.0
    span_step_ft: float = 1.0
    w_min_kip_per_ft: float = 0.5
    w_max_kip_per_ft: float = 4.0
    w_step_kip_per_ft: float = 0.1
    depth_limit_variants: tuple[str, ...] = DEPTH_LIMIT_VARIANTS
    defl_denoms: tuple[int, ...] = (240, 360, 500)
    load_level: str = "bottom"


def sample_grid(
    rng: random.Random, min_value: float, max_value: float, step: float
) -> float:
    """Draw a uniform sample from the inclusive grid [min_value, max_value].

    Assumptions:
        1. The range is an integer multiple of step, so every grid point
           from min_value to max_value is reachable; the generator's
           default configs are built to guarantee this.

    Args:
        rng: seeded random source
        min_value: lower bound of the grid, inclusive
        max_value: upper bound of the grid, inclusive
        step: grid spacing; must be positive

    Returns:
        float: min_value + k * step for a uniformly drawn integer k,
            rounded to 6 decimal places to remove floating-point dust

    Raises:
        ValueError: if step is not positive, or max_value < min_value.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step!r}")
    if max_value < min_value:
        raise ValueError(
            f"max_value must be >= min_value, got {max_value!r} < {min_value!r}"
        )
    n_steps = round((max_value - min_value) / step)
    k = rng.randint(0, n_steps)
    return round(min_value + k * step, 6)


def depth_limit_ft(variant: str, span_ft: float) -> float | None:
    """Resolve a depth-limit variant into a stated limit for a span.

    The generous and tight fractions are rounded up to the nearest half
    foot rather than down or to nearest, so the stated limit never rounds
    below the DRC depth floor of span/25 — span/14 rounded up always clears
    it, since span/14 already exceeds span/25.

    Args:
        variant: depth-limit variant, one of "none", "generous", "tight"
        span_ft: truss span in feet

    Returns:
        float: None for "none"; otherwise the fraction of span rounded up
            to the nearest DEPTH_LIMIT_STEP_FT

    Raises:
        ValueError: if variant is not a known depth-limit variant.
    """
    if variant == "none":
        return None
    if variant == "generous":
        fraction = GENEROUS_DEPTH_SPAN_FRACTION
    elif variant == "tight":
        fraction = TIGHT_DEPTH_SPAN_FRACTION
    else:
        supported = ", ".join(DEPTH_LIMIT_VARIANTS)
        raise ValueError(
            f"unknown depth_limit variant {variant!r}; supported: {supported}"
        )
    raw_limit_ft = span_ft * fraction
    return math.ceil(raw_limit_ft / DEPTH_LIMIT_STEP_FT) * DEPTH_LIMIT_STEP_FT


def generate_instance(
    seed: int, config: GeneratorConfig = GeneratorConfig()
) -> TrussInstance:
    """Procedurally generate one Warren-only, gravity-only problem instance.

    Assumptions:
        1. Draw order is fixed and part of the reproducibility contract:
           (1) span, (2) load magnitude, (3) depth-limit variant, (4)
           deflection denominator. Inserting, removing, or reordering a
           draw is a seed-reproducibility break and must be treated as
           one, since it silently changes what every downstream seed
           produces.

    Args:
        seed: seed for the instance's private random source
        config: difficulty knobs to sample from

    Returns:
        TrussInstance: one gravity-only, single-load-case instance with
            cost_ref_usd unset (None) pending Unit 9 calibration
    """
    rng = random.Random(seed)
    span_ft = sample_grid(
        rng, config.span_min_ft, config.span_max_ft, config.span_step_ft
    )
    w_magnitude_kip_per_ft = sample_grid(
        rng, config.w_min_kip_per_ft, config.w_max_kip_per_ft, config.w_step_kip_per_ft
    )
    variant = rng.choice(config.depth_limit_variants)
    defl_denom = rng.choice(config.defl_denoms)
    load_case = LoadCase(w_kip_per_ft=-w_magnitude_kip_per_ft, level=config.load_level)
    return TrussInstance(
        span_ft=span_ft,
        load_cases=(load_case,),
        depth_limit_ft=depth_limit_ft(variant, span_ft),
        defl_denom=defl_denom,
        cost_ref_usd=None,
    )


def generate_instances(
    seed: int, count: int, config: GeneratorConfig = GeneratorConfig()
) -> tuple[TrussInstance, ...]:
    """Procedurally generate a batch of independent problem instances.

    Assumptions:
        1. Child seeds are derived from the master RNG's getrandbits(64)
           rather than seed + i, so batches drawn from nearby seeds never
           share overlapping random streams.

    Args:
        seed: seed for the batch's master random source
        count: number of instances to generate; must be nonnegative
        config: difficulty knobs to sample from, shared across the batch

    Returns:
        tuple: count instances, each from an independently derived seed

    Raises:
        ValueError: if count is negative.
    """
    if count < 0:
        raise ValueError(f"count must be nonnegative, got {count!r}")
    master_rng = random.Random(seed)
    child_seeds = [master_rng.getrandbits(64) for _ in range(count)]
    return tuple(generate_instance(child_seed, config) for child_seed in child_seeds)
