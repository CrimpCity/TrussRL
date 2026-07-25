"""Unit tests for the TRL preflight verdict logic.

Pure-logic tests over synthetic GRPOConfig observations — no TRL, no
torch. The real defaults are read at trussrl-model-check time; these
tests pin how observations map to verdicts and how verdicts decide
static completeness.
"""

from trussRL.training.preflight import (
    KIND_ACCEPTANCE,
    KIND_DIVISIBILITY,
    KIND_RUNTIME,
    STATUS_DEFERRED,
    STATUS_MISMATCH,
    STATUS_MISSING,
    STATUS_RECORDED,
    STATUS_VERIFIED,
    TRL_PREFLIGHT_CHECKS,
    TrlPreflightCheck,
    TrlPreflightResult,
    evaluate_trl_preflight,
    preflight_passed,
)


def healthy_field_defaults() -> dict[str, object]:
    """Build field defaults where every preflight check is satisfied.

    Args: None

    Returns:
        dict: every field any check depends on, defaults matching the
            expected values
    """
    return {
        "loss_type": "dapo",
        "epsilon": 0.2,
        "epsilon_high": None,
        "scale_rewards": "group",
        "cast_lm_head_to_fp32": False,
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_clip_max": 3.0,
        "vllm_mode": "colocate",
        "num_generations": 8,
        "generation_batch_size": None,
        "beta": 0.0,
    }


def healthy_acceptance_errors() -> dict[str, str | None]:
    """Build acceptance results where every probe was accepted.

    Args: None

    Returns:
        dict: one None entry per acceptance/divisibility check
    """
    return {
        check.name: None
        for check in TRL_PREFLIGHT_CHECKS
        if check.kind in (KIND_ACCEPTANCE, KIND_DIVISIBILITY)
    }


def result_by_name(
    results: tuple[TrlPreflightResult, ...], name: str
) -> TrlPreflightResult:
    """Find one verdict by check name.

    Args:
        results: the evaluated verdict tuple
        name: the check name to look up

    Returns:
        TrlPreflightResult: the matching verdict
    """
    matches = [result for result in results if result.name == name]
    assert len(matches) == 1
    return matches[0]


def test_healthy_observations_pass_with_only_runtime_deferred() -> None:
    results = evaluate_trl_preflight(
        healthy_field_defaults(), healthy_acceptance_errors()
    )
    assert len(results) == len(TRL_PREFLIGHT_CHECKS)
    assert preflight_passed(results)
    statuses = {result.name: result.status for result in results}
    assert statuses["vllm_colocate_lora_sync"] == STATUS_DEFERRED
    for result in results:
        if result.kind == KIND_RUNTIME:
            continue
        assert result.status in (STATUS_VERIFIED, STATUS_RECORDED)


def test_clip_max_is_recorded_with_actual_default_and_no_expectation() -> None:
    results = evaluate_trl_preflight(
        healthy_field_defaults(), healthy_acceptance_errors()
    )
    clip_max = result_by_name(results, "vllm_importance_sampling_clip_max")
    assert clip_max.status == STATUS_RECORDED
    assert clip_max.expected is None
    assert clip_max.actual == 3.0


def test_wrong_default_is_a_mismatch_and_fails_preflight() -> None:
    defaults = healthy_field_defaults()
    defaults["vllm_importance_sampling_correction"] = False
    results = evaluate_trl_preflight(defaults, healthy_acceptance_errors())
    correction = result_by_name(results, "vllm_importance_sampling_correction")
    assert correction.status == STATUS_MISMATCH
    assert correction.actual is False
    assert not preflight_passed(results)


def test_absent_field_is_api_drift_missing() -> None:
    defaults = healthy_field_defaults()
    del defaults["vllm_importance_sampling_clip_max"]
    results = evaluate_trl_preflight(defaults, healthy_acceptance_errors())
    clip_max = result_by_name(results, "vllm_importance_sampling_clip_max")
    assert clip_max.status == STATUS_MISSING
    assert "vllm_importance_sampling_clip_max" in clip_max.detail
    assert not preflight_passed(results)


def test_rejected_probe_is_a_mismatch_carrying_the_error() -> None:
    errors = healthy_acceptance_errors()
    errors["scale_rewards_batch"] = "ValueError: unknown scale_rewards 'batch'"
    results = evaluate_trl_preflight(healthy_field_defaults(), errors)
    scale = result_by_name(results, "scale_rewards_batch")
    assert scale.status == STATUS_MISMATCH
    assert "unknown scale_rewards" in scale.detail
    assert not preflight_passed(results)


def test_unprobed_acceptance_check_is_missing() -> None:
    errors = healthy_acceptance_errors()
    del errors["beta_zero"]
    results = evaluate_trl_preflight(healthy_field_defaults(), errors)
    beta = result_by_name(results, "beta_zero")
    assert beta.status == STATUS_MISSING
    assert not preflight_passed(results)


def test_runtime_check_with_absent_field_is_missing_not_deferred() -> None:
    defaults = healthy_field_defaults()
    del defaults["vllm_mode"]
    results = evaluate_trl_preflight(defaults, healthy_acceptance_errors())
    sync = result_by_name(results, "vllm_colocate_lora_sync")
    assert sync.status == STATUS_MISSING
    assert not preflight_passed(results)


def test_divisibility_arithmetic_rejects_indivisible_pair() -> None:
    check = TrlPreflightCheck(
        name="bad_divisibility",
        description="synthetic indivisible pair",
        kind=KIND_DIVISIBILITY,
        fields=("num_generations", "generation_batch_size"),
        probe_kwargs=(("num_generations", 8), ("generation_batch_size", 25)),
    )
    results = evaluate_trl_preflight(
        healthy_field_defaults(), {"bad_divisibility": None}, checks=(check,)
    )
    assert results[0].status == STATUS_MISMATCH
    assert "not divisible" in results[0].detail
    assert not preflight_passed(results)


def test_divisibility_passes_arithmetic_then_requires_acceptance() -> None:
    check = next(
        check
        for check in TRL_PREFLIGHT_CHECKS
        if check.name == "generation_divisibility"
    )
    accepted = evaluate_trl_preflight(
        healthy_field_defaults(), {check.name: None}, checks=(check,)
    )
    assert accepted[0].status == STATUS_VERIFIED
    rejected = evaluate_trl_preflight(
        healthy_field_defaults(),
        {check.name: "ValueError: batch size mismatch"},
        checks=(check,),
    )
    assert rejected[0].status == STATUS_MISMATCH


def test_deferred_does_not_fail_preflight_passed() -> None:
    check = TrlPreflightCheck(
        name="runtime_only",
        description="synthetic runtime check",
        kind=KIND_RUNTIME,
        fields=("vllm_mode",),
    )
    results = evaluate_trl_preflight(healthy_field_defaults(), {}, checks=(check,))
    assert results[0].status == STATUS_DEFERRED
    assert preflight_passed(results)
