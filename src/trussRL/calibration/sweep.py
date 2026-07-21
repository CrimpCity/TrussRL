"""Score thousands of random designs to see what the reward actually does.

The collected distributions are the evidence behind the sanity gates and the
basis for the frozen grading constants — measured before training, so
environment bugs and training dynamics are never debugged at the same time.
"""

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from trussRL.baselines.random_design import sample_design
from trussRL.generator import GeneratorConfig
from trussRL.instance import TrussInstance
from trussRL.reward import RewardBreakdown, regrade, validate_cost_ref
from trussRL.schema import TrussDesign
from trussRL.verifier import run_pipeline

COST_REF_DECILE = 0.10


@dataclass(frozen=True)
class CalibrationConfig:
    """The full knob set for one calibration run, echoed into every artifact.

    Attributes:
        seed: master seed every per-instance stream is derived from
        n_instances: number of roster instances to calibrate
        designs_per_instance: random designs per instance for gates and
            cost_ref
        sweep_best_samples: random designs per instance for the independent
            sweep_best search
        generator: difficulty knobs for the roster instances
    """

    seed: int = 0
    n_instances: int = 32
    designs_per_instance: int = 5000
    sweep_best_samples: int = 100000
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)


@dataclass(frozen=True)
class SweepSeeds:
    """The two independent random streams one roster instance consumes.

    Attributes:
        calibration: seed for the gates + cost_ref sweep
        sweep_best: seed for the independent sweep_best search
    """

    calibration: int
    sweep_best: int


@dataclass(frozen=True)
class CalibrationSeeds:
    """Every derived seed of one calibration run, fixed by the master seed.

    Attributes:
        roster_seed: seed handed to generate_instances for the roster
        instance_seeds: per-instance sweep seeds, index-aligned with the
            roster
    """

    roster_seed: int
    instance_seeds: tuple[SweepSeeds, ...]


@dataclass(frozen=True)
class DesignSample:
    """One sampled design and its pass-1 (cost_ref-free) grade.

    Geometry and solver solutions are deliberately dropped — only the
    design and its breakdown are retained, keeping 5k-sample sweeps small.

    Attributes:
        design: the sampled design
        breakdown: the verifier's breakdown with cost_ref unset, so the
            rung-3 cost term contributed 0.0
    """

    design: TrussDesign
    breakdown: RewardBreakdown


@dataclass(frozen=True)
class InstanceCalibration:
    """Everything one instance's calibration sweep produced.

    Attributes:
        instance: the roster instance, with cost_ref_usd still None
        seed: the calibration sweep seed
        n_samples: number of designs sampled
        samples: every sampled design with its pass-1 breakdown
        cost_ref_usd: the derived cost reference
        regraded_scores: per-sample scores regraded under cost_ref_usd,
            index-aligned with samples — the distribution the gates see
    """

    instance: TrussInstance
    seed: int
    n_samples: int
    samples: tuple[DesignSample, ...]
    cost_ref_usd: float
    regraded_scores: tuple[float, ...]

    @property
    def n_rung3(self) -> int:
        """Number of samples that reached rung 3.

        Args: None

        Returns:
            int: count of samples with a rung-3 breakdown
        """
        return sum(1 for sample in self.samples if sample.breakdown.rung == 3)

    @property
    def n_strictly_feasible(self) -> int:
        """Number of samples that were strictly feasible.

        Args: None

        Returns:
            int: count of samples with feasible == 1.0
        """
        return sum(1 for sample in self.samples if sample.breakdown.feasible == 1.0)


@dataclass(frozen=True)
class SweepBest:
    """The best regraded design found by one instance's sweep_best search.

    Attributes:
        seed: the sweep_best search seed
        n_samples: number of designs searched
        cost_ref_usd: the cost reference every rung-3 design was regraded
            under
        design: the argmax design; ties broken first-seen
        breakdown: the argmax design's regraded breakdown
    """

    seed: int
    n_samples: int
    cost_ref_usd: float
    design: TrussDesign
    breakdown: RewardBreakdown


def validate_positive_count(name: str, value: int) -> None:
    """Reject a count parameter that is not strictly positive.

    Args:
        name: parameter name for the error message
        value: the count to validate

    Returns:
        None

    Raises:
        ValueError: if value is not strictly positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def derive_calibration_seeds(seed: int, n_instances: int) -> CalibrationSeeds:
    """Derive every seed one calibration run consumes from the master seed.

    Assumptions:
        1. Draw order is fixed and part of the reproducibility contract:
           (1) roster_seed, then per instance (2i) calibration and (2i+1)
           sweep_best, all via getrandbits(64) — the repo's child-seed
           convention, so nearby master seeds never share streams.

    Args:
        seed: master seed for the run
        n_instances: number of roster instances; must be positive

    Returns:
        CalibrationSeeds: the roster seed plus one SweepSeeds per instance

    Raises:
        ValueError: if n_instances is not positive.
    """
    validate_positive_count("n_instances", n_instances)
    master_rng = random.Random(seed)
    roster_seed = master_rng.getrandbits(64)
    instance_seeds = tuple(
        SweepSeeds(
            calibration=master_rng.getrandbits(64),
            sweep_best=master_rng.getrandbits(64),
        )
        for _ in range(n_instances)
    )
    return CalibrationSeeds(roster_seed=roster_seed, instance_seeds=instance_seeds)


def run_sweep(
    instance: TrussInstance, seed: int, n_samples: int
) -> tuple[DesignSample, ...]:
    """Sample and score n_samples random designs against one instance.

    Assumptions:
        1. The instance must not carry a preset cost_ref_usd: pass-1
           breakdowns are defined as cost_ref-free so cost_ref can be
           derived from them, and a preset value would silently poison
           that derivation.
        2. Typed designs from the sampler can never exit at rung 0, and
           every later stage is built, so a None breakdown indicates a
           harness bug and raises rather than being skipped.

    Args:
        instance: the problem instance to grade against; cost_ref_usd must
            be None
        seed: seed for the sweep's private random stream
        n_samples: number of designs to sample; must be positive

    Returns:
        tuple: one DesignSample per draw, in draw order

    Raises:
        ValueError: if the instance has a preset cost_ref_usd, or
            n_samples is not positive.
        RuntimeError: if the pipeline returned no breakdown for a design.
    """
    if instance.cost_ref_usd is not None:
        raise ValueError(
            "run_sweep requires instance.cost_ref_usd to be None so pass-1 "
            f"breakdowns are cost_ref-free, got {instance.cost_ref_usd!r}"
        )
    validate_positive_count("n_samples", n_samples)
    rng = random.Random(seed)
    samples: list[DesignSample] = []
    for index in range(n_samples):
        design = sample_design(rng, instance)
        trace = run_pipeline(instance, design)
        if trace.breakdown is None:
            raise RuntimeError(
                f"pipeline produced no breakdown for sample {index}: "
                f"{[f'{s.stage}={s.state}' for s in trace.stages]}"
            )
        samples.append(DesignSample(design=design, breakdown=trace.breakdown))
    return tuple(samples)


def compute_cost_ref(samples: Sequence[DesignSample]) -> float:
    """Derive the cost reference from a sweep's strictly feasible designs.

    Among designs with feasible == 1.0, takes the cheapest decile by total
    cost (at least one design) and returns its median cost — a reference
    that rewards being cheap among safe designs, not cheap among broken
    ones.

    Assumptions:
        1. Strict feasibility means feasible exactly 1.0; near-feasible
           designs (e.g. 0.999) are excluded so overstressed cheapness
           never drags the reference down.
        2. Raises loudly when no strictly feasible sample exists — a
           silent default here would freeze a meaningless constant.

    Args:
        samples: sweep output to derive the reference from

    Returns:
        float: the median total cost of the cheapest feasible decile

    Raises:
        ValueError: if no sample is strictly feasible.
    """
    feasible_costs = sorted(
        sample.breakdown.cost_total_usd
        for sample in samples
        if sample.breakdown.feasible == 1.0
        and sample.breakdown.cost_total_usd is not None
    )
    if not feasible_costs:
        n_rung3 = sum(1 for sample in samples if sample.breakdown.rung == 3)
        raise ValueError(
            "cannot compute cost_ref: no strictly feasible sample among "
            f"{len(samples)} samples ({n_rung3} reached rung 3)"
        )
    decile_count = max(1, math.ceil(COST_REF_DECILE * len(feasible_costs)))
    return float(statistics.median(feasible_costs[:decile_count]))


def calibrate_instance(
    instance: TrussInstance, seed: int, n_samples: int
) -> InstanceCalibration:
    """Run one instance's full calibration: sweep, cost_ref, regrade.

    The reusable per-instance unit: Phase 3 calls this for fresh training
    instances to compute their cost_ref deterministically, exactly as the
    frozen roster's was computed.

    Args:
        instance: the problem instance to calibrate; cost_ref_usd must be
            None
        seed: seed for the calibration sweep's private random stream
        n_samples: number of designs to sample; must be positive

    Returns:
        InstanceCalibration: the samples, the derived cost_ref, and the
            regraded score distribution the gates evaluate

    Raises:
        ValueError: if the instance has a preset cost_ref_usd, n_samples
            is not positive, or no sample is strictly feasible.
        RuntimeError: if the pipeline returned no breakdown for a design.
    """
    samples = run_sweep(instance, seed, n_samples)
    cost_ref_usd = compute_cost_ref(samples)
    regraded_scores = tuple(
        regrade(sample.breakdown, cost_ref_usd).score for sample in samples
    )
    return InstanceCalibration(
        instance=instance,
        seed=seed,
        n_samples=n_samples,
        samples=samples,
        cost_ref_usd=cost_ref_usd,
        regraded_scores=regraded_scores,
    )


def find_sweep_best(
    instance: TrussInstance, cost_ref_usd: float, seed: int, n_samples: int
) -> SweepBest:
    """Search an independent random stream for the best regraded design.

    Streaming strict argmax: each design is sampled, scored, regraded
    under cost_ref_usd when it reached rung 3, and compared against the
    incumbent; only the incumbent is retained, so 100k-sample searches
    stay flat in memory. Ties keep the first-seen design, making the
    result deterministic.

    Args:
        instance: the problem instance to grade against
        cost_ref_usd: the frozen cost reference to regrade rung-3 designs
            under; must be finite and strictly positive
        seed: seed for the search's private random stream, independent of
            the calibration sweep's
        n_samples: number of designs to search; must be positive

    Returns:
        SweepBest: the argmax design and its regraded breakdown

    Raises:
        ValueError: if cost_ref_usd is invalid or n_samples is not
            positive.
        RuntimeError: if the pipeline returned no breakdown for a design.
    """
    validate_cost_ref(cost_ref_usd)
    validate_positive_count("n_samples", n_samples)
    rng = random.Random(seed)
    best_design: TrussDesign | None = None
    best_breakdown: RewardBreakdown | None = None
    for index in range(n_samples):
        design = sample_design(rng, instance)
        trace = run_pipeline(instance, design)
        if trace.breakdown is None:
            raise RuntimeError(
                f"pipeline produced no breakdown for sample {index}: "
                f"{[f'{s.stage}={s.state}' for s in trace.stages]}"
            )
        breakdown = trace.breakdown
        if breakdown.rung == 3:
            breakdown = regrade(breakdown, cost_ref_usd)
        if best_breakdown is None or breakdown.score > best_breakdown.score:
            best_design = design
            best_breakdown = breakdown
    if best_design is None or best_breakdown is None:
        raise RuntimeError("sweep_best search retained no incumbent")
    return SweepBest(
        seed=seed,
        n_samples=n_samples,
        cost_ref_usd=cost_ref_usd,
        design=best_design,
        breakdown=best_breakdown,
    )
