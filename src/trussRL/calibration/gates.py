"""Sanity gates that must pass before any training run.

Check that reward is not saturated, variance is healthy, and the optimum is
interior rather than pegged at a bound — then freeze cost_ref and
sweep_best, which must never depend on the policy.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from trussRL.baselines.random_design import depth_window_ft
from trussRL.calibration.sweep import DesignSample, InstanceCalibration
from trussRL.drc import DEPTH_SPAN_FRACTION_MAX, N_BAYS_MAX, N_BAYS_MIN
from trussRL.instance import TrussInstance

SATURATION_MEAN_MAX = 0.35
VARIANCE_STD_MIN = 0.05
TOP_FRACTION = 0.01
INTERIOR_MARGIN_FRACTION = 0.10

GateStatus = Literal["pass", "fail", "skipped"]


@dataclass(frozen=True)
class GateCheck:
    """One sanity-gate check on one instance's regraded distribution.

    Attributes:
        gate: check identifier, one of "saturation", "variance",
            "interior_n_bays", "interior_depth_lower", or
            "interior_depth_upper"
        status: "pass", "fail", or "skipped" (the check does not apply)
        value: the measured statistic, or None when skipped
        criterion: human-readable statement of what was required
        detail: extra context, e.g. why a check was skipped
    """

    gate: str
    status: GateStatus
    value: float | None
    criterion: str
    detail: str = ""


@dataclass(frozen=True)
class GateReport:
    """Every gate check for one roster instance, or the error that stopped it.

    Attributes:
        instance_index: position of the instance in the roster
        checks: every gate check run; empty when calibration errored
        error: why calibration failed for this instance, or None
    """

    instance_index: int
    checks: tuple[GateCheck, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Whether this instance cleared every applicable gate.

        Args: None

        Returns:
            bool: True when calibration succeeded and no check failed;
                skipped checks do not count against passing
        """
        return self.error is None and all(
            check.status != "fail" for check in self.checks
        )


def saturation_check(scores: Sequence[float]) -> GateCheck:
    """Check that the random-design mean reward is well below 0.5.

    A saturated reward (random designs already scoring high) leaves no
    headroom for training signal; vision section 9 requires the random
    mean well below 0.5, encoded here as SATURATION_MEAN_MAX.

    Args:
        scores: regraded scores of every sampled design

    Returns:
        GateCheck: "saturation" with the mean as its value

    Raises:
        ValueError: if scores is empty.
    """
    if not scores:
        raise ValueError("saturation_check requires at least one score")
    mean = statistics.fmean(scores)
    return GateCheck(
        gate="saturation",
        status="pass" if mean <= SATURATION_MEAN_MAX else "fail",
        value=mean,
        criterion=f"mean regraded score <= {SATURATION_MEAN_MAX}",
    )


def variance_check(scores: Sequence[float]) -> GateCheck:
    """Check that the regraded score distribution has healthy spread.

    A near-constant reward gives the policy gradient nothing to climb;
    the population standard deviation must clear VARIANCE_STD_MIN.

    Assumptions:
        1. Uses the population standard deviation (pstdev) rather than the
           sample estimator, so the check is deterministic and defined
           even at n = 1.

    Args:
        scores: regraded scores of every sampled design

    Returns:
        GateCheck: "variance" with the pstdev as its value

    Raises:
        ValueError: if scores is empty.
    """
    if not scores:
        raise ValueError("variance_check requires at least one score")
    std = statistics.pstdev(scores)
    return GateCheck(
        gate="variance",
        status="pass" if std >= VARIANCE_STD_MIN else "fail",
        value=std,
        criterion=f"pstdev of regraded scores >= {VARIANCE_STD_MIN}",
    )


def top_fraction_samples(
    samples: Sequence[DesignSample], scores: Sequence[float]
) -> tuple[DesignSample, ...]:
    """Select the top-scoring fraction of samples for the interior checks.

    Assumptions:
        1. Count is max(1, ceil(TOP_FRACTION * n)), so tiny sweeps still
           yield a non-empty top set.
        2. Sort key is (-score, original index) — a stable, deterministic
           order where ties keep draw order.

    Args:
        samples: sweep output, index-aligned with scores
        scores: regraded score per sample

    Returns:
        tuple: the top-fraction samples, best first

    Raises:
        ValueError: if samples is empty or lengths disagree.
    """
    if not samples:
        raise ValueError("top_fraction_samples requires at least one sample")
    if len(samples) != len(scores):
        raise ValueError(
            f"samples and scores lengths disagree: {len(samples)} != {len(scores)}"
        )
    count = max(1, math.ceil(TOP_FRACTION * len(samples)))
    order = sorted(range(len(samples)), key=lambda index: (-scores[index], index))
    return tuple(samples[index] for index in order[:count])


def interior_n_bays_check(top_samples: Sequence[DesignSample]) -> GateCheck:
    """Check that the top designs' bay counts sit interior to [4, 24].

    The top-1% median n_bays must clear each bound by 10% of the range
    (margin 2.0, so [6.0, 22.0]). A median pegged high means the per-node
    cost is too low to matter; pegged low means it dominates.

    Args:
        top_samples: the top-fraction samples from top_fraction_samples

    Returns:
        GateCheck: "interior_n_bays" with the median as its value

    Raises:
        ValueError: if top_samples is empty.
    """
    if not top_samples:
        raise ValueError("interior_n_bays_check requires at least one sample")
    margin = INTERIOR_MARGIN_FRACTION * (N_BAYS_MAX - N_BAYS_MIN)
    low = N_BAYS_MIN + margin
    high = N_BAYS_MAX - margin
    median = float(statistics.median(sample.design.n_bays for sample in top_samples))
    return GateCheck(
        gate="interior_n_bays",
        status="pass" if low <= median <= high else "fail",
        value=median,
        criterion=f"top-fraction median n_bays within [{low:g}, {high:g}]",
    )


def interior_depth_check(
    instance: TrussInstance, top_samples: Sequence[DesignSample]
) -> tuple[GateCheck, GateCheck]:
    """Check that the top designs' depths sit interior to the sampled window.

    Both checks run over the effective window the sampler drew from
    (span/25 up to the lesser of span/4 and the stated limit), with a
    margin of 10% of that window.

    Assumptions:
        1. The lower bound is always checked: a top-1% median pegged at
           the depth floor signals a reward mis-weighting regardless of
           any stated limit.
        2. The upper bound is checked only when the effective upper bound
           is span/4 (no stated limit, or a stated limit looser than
           span/4). When a tighter stated limit binds, a median pegged at
           it expresses the prompt constraint, not a mis-weighting, so
           the upper check is skipped with an explicit detail.

    Args:
        instance: the roster instance, providing span and stated limit
        top_samples: the top-fraction samples from top_fraction_samples

    Returns:
        tuple: ("interior_depth_lower", "interior_depth_upper") checks

    Raises:
        ValueError: if top_samples is empty.
    """
    if not top_samples:
        raise ValueError("interior_depth_check requires at least one sample")
    low_ft, high_ft = depth_window_ft(instance)
    margin_ft = INTERIOR_MARGIN_FRACTION * (high_ft - low_ft)
    median_ft = float(
        statistics.median(sample.design.depth_ft for sample in top_samples)
    )
    lower = GateCheck(
        gate="interior_depth_lower",
        status="pass" if median_ft >= low_ft + margin_ft else "fail",
        value=median_ft,
        criterion=f"top-fraction median depth >= {low_ft + margin_ft:g} ft",
    )
    ratio_max_ft = instance.span_ft * DEPTH_SPAN_FRACTION_MAX
    if high_ft < ratio_max_ft:
        upper = GateCheck(
            gate="interior_depth_upper",
            status="skipped",
            value=None,
            criterion=f"top-fraction median depth <= {high_ft - margin_ft:g} ft",
            detail=(
                f"stated depth limit {instance.depth_limit_ft:g} ft binds below "
                f"span/4 = {ratio_max_ft:g} ft; a median at the limit expresses "
                "the prompt constraint, not a reward mis-weighting"
            ),
        )
    else:
        upper = GateCheck(
            gate="interior_depth_upper",
            status="pass" if median_ft <= high_ft - margin_ft else "fail",
            value=median_ft,
            criterion=f"top-fraction median depth <= {high_ft - margin_ft:g} ft",
        )
    return lower, upper


def evaluate_gates(calibration: InstanceCalibration, instance_index: int) -> GateReport:
    """Run every sanity gate on one instance's regraded distribution.

    Args:
        calibration: the instance's calibration sweep output
        instance_index: position of the instance in the roster, recorded
            in the report

    Returns:
        GateReport: all five checks; passed only when none failed
    """
    scores = calibration.regraded_scores
    top = top_fraction_samples(calibration.samples, scores)
    depth_lower, depth_upper = interior_depth_check(calibration.instance, top)
    return GateReport(
        instance_index=instance_index,
        checks=(
            saturation_check(scores),
            variance_check(scores),
            interior_n_bays_check(top),
            depth_lower,
            depth_upper,
        ),
    )
