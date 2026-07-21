"""Tests for the calibration sweep: seeds, cost_ref, and sweep_best.

Real solver coverage for Unit 9 lives here: small seeded sweeps exercise
the full pipeline, while cost_ref arithmetic is pinned by hand on synthetic
samples.
"""

import json

import pytest

from trussRL.calibration.sweep import (
    DesignSample,
    calibrate_instance,
    compute_cost_ref,
    derive_calibration_seeds,
    find_sweep_best,
    run_sweep,
)
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.reward import RewardBreakdown, regrade
from trussRL.schema import TrussDesign
from trussRL.verifier import run_pipeline, score

INSTANCE = TrussInstance(
    span_ft=60.0,
    load_cases=(LoadCase(w_kip_per_ft=-0.5, level="bottom"),),
    depth_limit_ft=None,
    defl_denom=360,
    cost_ref_usd=None,
)


def make_synthetic_sample(
    cost_total_usd: float, feasible: float, rung: int = 3
) -> DesignSample:
    """Build a synthetic sample with a chosen cost and feasibility.

    Args:
        cost_total_usd: recorded total cost in dollars
        feasible: recorded feasibility gate value
        rung: reward ladder rung of the breakdown

    Returns:
        DesignSample: a dummy design with a hand-built breakdown
    """
    design = TrussDesign(
        truss_type="warren",
        n_bays=8,
        depth_ft=6.0,
        top_chord="HSS6X6X3/8",
        bottom_chord="HSS6X6X3/8",
        diagonals="HSS4X4X1/4",
    )
    breakdown = RewardBreakdown(
        score=0.5,
        rung=rung,
        reason="solved" if rung == 3 else "drc: synthetic",
        u_strength=0.5 if rung == 3 else None,
        u_buckling=0.5 if rung == 3 else None,
        u_defl=0.5 if rung == 3 else None,
        cost_total_usd=cost_total_usd if rung == 3 else None,
        feasible=feasible if rung == 3 else None,
    )
    return DesignSample(design=design, breakdown=breakdown)


def test_derive_calibration_seeds_deterministic_and_distinct() -> None:
    seeds_a = derive_calibration_seeds(0, 8)
    seeds_b = derive_calibration_seeds(0, 8)
    assert seeds_a == seeds_b
    all_seeds = [seeds_a.roster_seed]
    for pair in seeds_a.instance_seeds:
        all_seeds.extend((pair.calibration, pair.sweep_best))
    assert len(all_seeds) == len(set(all_seeds))
    assert derive_calibration_seeds(1, 8) != seeds_a


def test_derive_calibration_seeds_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="n_instances"):
        derive_calibration_seeds(0, 0)
    with pytest.raises(ValueError, match="n_instances"):
        derive_calibration_seeds(0, -3)


def test_run_sweep_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        run_sweep(INSTANCE, 0, 0)


def test_run_sweep_rejects_preset_cost_ref() -> None:
    preset = TrussInstance(
        span_ft=60.0,
        load_cases=(LoadCase(w_kip_per_ft=-0.5, level="bottom"),),
        cost_ref_usd=20000.0,
    )
    with pytest.raises(ValueError, match="cost_ref_usd"):
        run_sweep(preset, 0, 10)


def test_run_sweep_is_deterministic() -> None:
    samples_a = run_sweep(INSTANCE, 11, 30)
    samples_b = run_sweep(INSTANCE, 11, 30)
    assert samples_a == samples_b


def test_small_real_sweep_reaches_rungs_1_and_3_never_0() -> None:
    samples = run_sweep(INSTANCE, 0, 200)
    rungs = {sample.breakdown.rung for sample in samples}
    assert 0 not in rungs
    assert 1 in rungs
    assert 3 in rungs


def test_compute_cost_ref_decile_median_arithmetic() -> None:
    # 25 strictly feasible costs 100..2500 => decile ceil(2.5)=3 cheapest
    # (100, 200, 300), median 200.
    samples = [
        make_synthetic_sample(cost_total_usd=100.0 * (i + 1), feasible=1.0)
        for i in range(25)
    ]
    assert compute_cost_ref(samples) == 200.0


def test_compute_cost_ref_excludes_near_feasible() -> None:
    samples = [
        make_synthetic_sample(cost_total_usd=10.0, feasible=0.999),
        make_synthetic_sample(cost_total_usd=500.0, feasible=1.0),
    ]
    assert compute_cost_ref(samples) == 500.0


def test_compute_cost_ref_single_feasible_sample() -> None:
    samples = [make_synthetic_sample(cost_total_usd=750.0, feasible=1.0)]
    assert compute_cost_ref(samples) == 750.0


def test_compute_cost_ref_raises_with_zero_feasible() -> None:
    samples = [
        make_synthetic_sample(cost_total_usd=10.0, feasible=0.5),
        make_synthetic_sample(cost_total_usd=0.0, feasible=0.0, rung=1),
    ]
    with pytest.raises(ValueError, match="no strictly feasible"):
        compute_cost_ref(samples)


def test_calibrate_instance_regraded_scores_dominate_pass_1() -> None:
    calibration = calibrate_instance(INSTANCE, 3, 200)
    assert calibration.cost_ref_usd > 0.0
    assert len(calibration.regraded_scores) == len(calibration.samples)
    for sample, regraded_score in zip(
        calibration.samples, calibration.regraded_scores, strict=True
    ):
        assert regraded_score >= sample.breakdown.score
    assert calibration.n_rung3 >= calibration.n_strictly_feasible > 0


def test_find_sweep_best_matches_manual_argmax() -> None:
    seed = 5
    n_samples = 300
    samples = run_sweep(INSTANCE, seed, n_samples)
    cost_ref_usd = compute_cost_ref(samples)
    best_index = -1
    best_score = float("-inf")
    regraded: list[RewardBreakdown] = []
    for index, sample in enumerate(samples):
        breakdown = sample.breakdown
        if breakdown.rung == 3:
            breakdown = regrade(breakdown, cost_ref_usd)
        regraded.append(breakdown)
        if breakdown.score > best_score:
            best_score = breakdown.score
            best_index = index
    result = find_sweep_best(INSTANCE, cost_ref_usd, seed, n_samples)
    assert result.design == samples[best_index].design
    assert result.breakdown == regraded[best_index]
    assert result.seed == seed
    assert result.n_samples == n_samples
    assert result.cost_ref_usd == cost_ref_usd


def test_find_sweep_best_validates_inputs() -> None:
    with pytest.raises(ValueError, match="cost_ref_usd"):
        find_sweep_best(INSTANCE, 0.0, 0, 10)
    with pytest.raises(ValueError, match="n_samples"):
        find_sweep_best(INSTANCE, 100.0, 0, 0)


def test_sampled_design_scores_identically_via_score() -> None:
    samples = run_sweep(INSTANCE, 9, 5)
    for sample in samples:
        completion = (
            "Here is my design:\n```json\n"
            + json.dumps(sample.design.model_dump())
            + "\n```\n"
        )
        assert score(INSTANCE, completion) == sample.breakdown
        trace = run_pipeline(INSTANCE, sample.design)
        assert trace.breakdown == sample.breakdown
