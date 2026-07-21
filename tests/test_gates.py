"""Tests for the calibration sanity gates — synthetic samples, no solver.

Threshold constants are pinned to the values decided with the user; every
check is exercised on both sides of its boundary, plus the stated-limit
skip semantics of the depth upper bound.
"""

import pytest

from trussRL.calibration.gates import (
    INTERIOR_MARGIN_FRACTION,
    SATURATION_MEAN_MAX,
    TOP_FRACTION,
    VARIANCE_STD_MIN,
    GateReport,
    evaluate_gates,
    interior_depth_check,
    interior_n_bays_check,
    saturation_check,
    top_fraction_samples,
    variance_check,
)
from trussRL.calibration.sweep import DesignSample, InstanceCalibration
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.reward import RewardBreakdown
from trussRL.schema import TrussDesign

SPAN_FT = 100.0  # window without stated limit: [4.0, 25.0] ft


def make_instance(depth_limit_ft: float | None) -> TrussInstance:
    """Build a 100 ft gravity instance with the given stated depth limit.

    Args:
        depth_limit_ft: stated depth limit in feet, or None for unstated

    Returns:
        TrussInstance: a single-case gravity instance
    """
    return TrussInstance(
        span_ft=SPAN_FT,
        load_cases=(LoadCase(w_kip_per_ft=-2.0, level="bottom"),),
        depth_limit_ft=depth_limit_ft,
    )


def make_sample(n_bays: int = 12, depth_ft: float = 12.0) -> DesignSample:
    """Build a synthetic sample with chosen n_bays and depth.

    Args:
        n_bays: bay count for the sample's design
        depth_ft: depth in feet for the sample's design

    Returns:
        DesignSample: a dummy design with a rung-3 breakdown
    """
    design = TrussDesign(
        truss_type="warren",
        n_bays=n_bays,
        depth_ft=depth_ft,
        top_chord="HSS6X6X3/8",
        bottom_chord="HSS6X6X3/8",
        diagonals="HSS4X4X1/4",
    )
    breakdown = RewardBreakdown(
        score=0.5,
        rung=3,
        reason="solved",
        u_strength=0.5,
        u_buckling=0.5,
        u_defl=0.5,
        cost_total_usd=10000.0,
        feasible=1.0,
    )
    return DesignSample(design=design, breakdown=breakdown)


def test_threshold_constants_are_frozen() -> None:
    assert SATURATION_MEAN_MAX == 0.35
    assert VARIANCE_STD_MIN == 0.05
    assert TOP_FRACTION == 0.01
    assert INTERIOR_MARGIN_FRACTION == 0.10


def test_saturation_check_both_sides() -> None:
    assert saturation_check([0.30, 0.40]).status == "pass"  # mean 0.35
    assert saturation_check([0.36]).status == "fail"
    check = saturation_check([0.2, 0.4])
    assert check.gate == "saturation"
    assert check.value == pytest.approx(0.3)


def test_variance_check_both_sides() -> None:
    assert variance_check([0.0, 0.1]).status == "pass"  # pstdev 0.05
    assert variance_check([0.2, 0.2, 0.2]).status == "fail"


def test_variance_check_pstdev_defined_at_n_1() -> None:
    check = variance_check([0.5])
    assert check.value == 0.0
    assert check.status == "fail"


def test_top_fraction_count_and_tie_break() -> None:
    samples = [make_sample(n_bays=4 + index % 21) for index in range(150)]
    scores = [0.1] * 150
    scores[7] = 0.9
    scores[3] = 0.5
    top = top_fraction_samples(samples, scores)
    # ceil(0.01 * 150) = 2: the 0.9 then the 0.5.
    assert len(top) == 2
    assert top[0] is samples[7]
    assert top[1] is samples[3]
    # All-tied scores keep draw order.
    tied = top_fraction_samples(samples, [0.1] * 150)
    assert tied == (samples[0], samples[1])


def test_top_fraction_minimum_of_one() -> None:
    samples = [make_sample(), make_sample()]
    assert len(top_fraction_samples(samples, [0.1, 0.2])) == 1


def test_interior_n_bays_boundaries() -> None:
    assert interior_n_bays_check([make_sample(n_bays=5)]).status == "fail"
    assert interior_n_bays_check([make_sample(n_bays=23)]).status == "fail"
    assert interior_n_bays_check([make_sample(n_bays=12)]).status == "pass"
    # Margin bounds 6.0 and 22.0 are inclusive passes.
    assert interior_n_bays_check([make_sample(n_bays=6)]).status == "pass"
    assert interior_n_bays_check([make_sample(n_bays=22)]).status == "pass"


def test_interior_depth_no_limit_pegged_high_fails() -> None:
    # Window [4.0, 25.0], margin 2.1: median 24.9 exceeds 22.9.
    lower, upper = interior_depth_check(
        make_instance(None), [make_sample(depth_ft=24.9)]
    )
    assert lower.status == "pass"
    assert upper.status == "fail"


def test_interior_depth_no_limit_interior_passes() -> None:
    lower, upper = interior_depth_check(
        make_instance(None), [make_sample(depth_ft=12.0)]
    )
    assert lower.status == "pass"
    assert upper.status == "pass"


def test_interior_depth_binding_limit_at_cap_passes_with_skip() -> None:
    # Stated limit 10.0 < span/4 = 25.0 binds: upper is skipped.
    lower, upper = interior_depth_check(
        make_instance(10.0), [make_sample(depth_ft=10.0)]
    )
    assert lower.status == "pass"
    assert upper.status == "skipped"
    assert upper.value is None
    assert "stated depth limit" in upper.detail


def test_interior_depth_looser_stated_limit_keeps_upper_check() -> None:
    # Stated limit 30.0 > span/4 = 25.0: effective cap is span/4, checked.
    lower, upper = interior_depth_check(
        make_instance(30.0), [make_sample(depth_ft=24.9)]
    )
    assert upper.status == "fail"


def test_interior_depth_pegged_low_always_fails() -> None:
    for depth_limit_ft in (None, 10.0, 30.0):
        lower, _ = interior_depth_check(
            make_instance(depth_limit_ft), [make_sample(depth_ft=4.05)]
        )
        assert lower.status == "fail"


def test_errored_report_fails_overall() -> None:
    report = GateReport(instance_index=3, checks=(), error="no feasible sample")
    assert not report.passed


def test_empty_report_without_error_passes() -> None:
    assert GateReport(instance_index=0, checks=()).passed


def make_calibration(
    scores: list[float], top_sample: DesignSample
) -> InstanceCalibration:
    """Build a synthetic calibration whose top-1 sample is controlled.

    The highest score is placed on top_sample so the interior checks see
    exactly that design.

    Args:
        scores: regraded scores for the filler samples; must all be below
            the 0.9 assigned to top_sample
        top_sample: the sample the top-fraction selection must return

    Returns:
        InstanceCalibration: synthetic sweep output for evaluate_gates
    """
    samples = [make_sample() for _ in scores] + [top_sample]
    all_scores = [*scores, 0.9]
    return InstanceCalibration(
        instance=make_instance(None),
        seed=0,
        n_samples=len(samples),
        samples=tuple(samples),
        cost_ref_usd=10000.0,
        regraded_scores=tuple(all_scores),
    )


def test_evaluate_gates_all_pass() -> None:
    calibration = make_calibration([0.1] * 99, make_sample(n_bays=12, depth_ft=12.0))
    report = evaluate_gates(calibration, instance_index=7)
    assert report.instance_index == 7
    assert report.error is None
    assert report.passed
    assert len(report.checks) == 5


def test_evaluate_gates_conjunction_single_failure_fails_report() -> None:
    calibration = make_calibration([0.1] * 99, make_sample(n_bays=24, depth_ft=12.0))
    report = evaluate_gates(calibration, instance_index=0)
    assert not report.passed
    failed = [check.gate for check in report.checks if check.status == "fail"]
    assert failed == ["interior_n_bays"]
