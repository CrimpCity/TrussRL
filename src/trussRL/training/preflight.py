"""The install-time TRL preflight checks as pure, dependency-free logic.

Defines the recipe's assumptions about installed TRL as data and turns
raw GRPOConfig observations — field defaults and acceptance-probe
results read by the heavy backend (model_check.py) — into per-check
verdicts. Never imports TRL, so the verdict logic is unit-testable with
no training extra installed.
"""

import dataclasses
from collections.abc import Mapping

KIND_DEFAULT = "default"
KIND_ACCEPTANCE = "acceptance"
KIND_RECORD = "record"
KIND_DIVISIBILITY = "divisibility"
KIND_RUNTIME = "runtime"

STATUS_VERIFIED = "verified"
STATUS_RECORDED = "recorded"
STATUS_MISMATCH = "mismatch"
STATUS_MISSING = "missing"
STATUS_DEFERRED = "deferred"


@dataclasses.dataclass(frozen=True)
class TrlPreflightCheck:
    """One recipe assumption about installed TRL and how to verify it.

    Assumptions:
        1. probe_kwargs is a tuple of pairs rather than a dict so the
           dataclass stays hashable/frozen; consumers dict() it.

    Args:
        name: unique identifier, key into the acceptance-probe results
        description: human-readable description of the requirement this
            check verifies
        kind: one of default/acceptance/record/divisibility/runtime
        fields: GRPOConfig field names the check depends on; any absent
            field is API drift and yields status missing
        probe_kwargs: kwargs the backend passes to GRPOConfig for
            acceptance and divisibility kinds
        expected_default: the required field default for kind=default
    """

    name: str
    description: str
    kind: str
    fields: tuple[str, ...]
    probe_kwargs: tuple[tuple[str, object], ...] = ()
    expected_default: object = None


@dataclasses.dataclass(frozen=True)
class TrlPreflightResult:
    """The verdict for one preflight check.

    Args:
        name: the check's identifier
        description: human-readable description of the requirement
        kind: the check's verification kind
        status: verified, recorded, mismatch, missing, or deferred
        expected: the expected value, None where the check only records
        actual: what the installed package reports, None where unknown
        detail: human-readable explanation of the verdict
    """

    name: str
    description: str
    kind: str
    status: str
    expected: object
    actual: object
    detail: str


TRL_PREFLIGHT_CHECKS: tuple[TrlPreflightCheck, ...] = (
    TrlPreflightCheck(
        name="loss_type_cispo",
        description=(
            'loss_type="cispo" accepted; epsilon/epsilon_high are the '
            "IS-weight cap under CISPO"
        ),
        kind=KIND_ACCEPTANCE,
        fields=("loss_type", "epsilon", "epsilon_high"),
        probe_kwargs=(("loss_type", "cispo"), ("epsilon_high", 5.0)),
    ),
    TrlPreflightCheck(
        name="scale_rewards_batch",
        description='scale_rewards="batch" accepted',
        kind=KIND_ACCEPTANCE,
        fields=("scale_rewards",),
        probe_kwargs=(("scale_rewards", "batch"),),
    ),
    TrlPreflightCheck(
        name="cast_lm_head_to_fp32",
        description="cast_lm_head_to_fp32 supported",
        kind=KIND_ACCEPTANCE,
        fields=("cast_lm_head_to_fp32",),
        probe_kwargs=(("cast_lm_head_to_fp32", True),),
    ),
    TrlPreflightCheck(
        name="vllm_importance_sampling_correction",
        description="vllm_importance_sampling_correction defaults to on",
        kind=KIND_DEFAULT,
        fields=("vllm_importance_sampling_correction",),
        expected_default=True,
    ),
    TrlPreflightCheck(
        name="vllm_importance_sampling_clip_max",
        description=(
            "actual default vllm_importance_sampling_clip_max read and "
            "recorded, never assumed"
        ),
        kind=KIND_RECORD,
        fields=("vllm_importance_sampling_clip_max",),
    ),
    TrlPreflightCheck(
        name="vllm_colocate_lora_sync",
        description=(
            'vllm_mode="colocate" + PEFT/LoRA adapter weight-sync works '
            "on the installed pair"
        ),
        kind=KIND_RUNTIME,
        fields=("vllm_mode",),
    ),
    TrlPreflightCheck(
        name="generation_divisibility",
        description=(
            "num_generations=8, generation_batch_size=24 divisibility "
            "constraints satisfied"
        ),
        kind=KIND_DIVISIBILITY,
        fields=("num_generations", "generation_batch_size"),
        probe_kwargs=(("num_generations", 8), ("generation_batch_size", 24)),
    ),
    TrlPreflightCheck(
        name="beta_zero",
        description="beta=0.0 accepted to disable reference-model KL",
        kind=KIND_ACCEPTANCE,
        fields=("beta",),
        probe_kwargs=(("beta", 0.0),),
    ),
    TrlPreflightCheck(
        name="fallback_ppo_clip",
        description=(
            "fallback vanilla PPO-clip loss_type + epsilon_high=0.28 accepted"
        ),
        kind=KIND_ACCEPTANCE,
        fields=("loss_type", "epsilon_high"),
        probe_kwargs=(("loss_type", "grpo"), ("epsilon_high", 0.28)),
    ),
)


def missing_fields(
    check: TrlPreflightCheck, field_defaults: Mapping[str, object]
) -> tuple[str, ...]:
    """List the check's GRPOConfig fields absent from the installed version.

    Args:
        check: the preflight check being evaluated
        field_defaults: field name to default value, as read from
            dataclasses.fields(GRPOConfig)

    Returns:
        tuple: the absent field names, empty when all are present
    """
    return tuple(field for field in check.fields if field not in field_defaults)


def evaluate_default_check(
    check: TrlPreflightCheck, field_defaults: Mapping[str, object]
) -> TrlPreflightResult:
    """Verify a check whose installed default must match an expected value.

    Args:
        check: a kind=default preflight check with all fields present
        field_defaults: field name to default value

    Returns:
        TrlPreflightResult: verified when the default matches, mismatch
            otherwise
    """
    actual = field_defaults[check.fields[0]]
    if actual == check.expected_default:
        status, detail = STATUS_VERIFIED, f"default is {actual!r} as expected"
    else:
        status = STATUS_MISMATCH
        detail = f"default is {actual!r}, expected {check.expected_default!r}"
    return TrlPreflightResult(
        name=check.name,
        description=check.description,
        kind=check.kind,
        status=status,
        expected=check.expected_default,
        actual=actual,
        detail=detail,
    )


def evaluate_acceptance_check(
    check: TrlPreflightCheck, acceptance_errors: Mapping[str, str | None]
) -> TrlPreflightResult:
    """Verify a check by whether GRPOConfig accepted the probed kwargs.

    Assumptions:
        1. A check with no entry in acceptance_errors was never probed by
           the backend, which is treated as missing data, not success.

    Args:
        check: a kind=acceptance or kind=divisibility preflight check
        acceptance_errors: probe name to error string, None when the
            probe constructed without raising

    Returns:
        TrlPreflightResult: verified on acceptance, mismatch on a
            recorded error, missing when no probe result exists
    """
    expected = dict(check.probe_kwargs)
    if check.name not in acceptance_errors:
        return TrlPreflightResult(
            name=check.name,
            description=check.description,
            kind=check.kind,
            status=STATUS_MISSING,
            expected=expected,
            actual=None,
            detail="no acceptance probe result recorded",
        )
    error = acceptance_errors[check.name]
    if error is None:
        status, detail = STATUS_VERIFIED, f"GRPOConfig accepted {expected!r}"
    else:
        status, detail = STATUS_MISMATCH, f"GRPOConfig rejected {expected!r}: {error}"
    return TrlPreflightResult(
        name=check.name,
        description=check.description,
        kind=check.kind,
        status=status,
        expected=expected,
        actual=error,
        detail=detail,
    )


def evaluate_divisibility_check(
    check: TrlPreflightCheck, acceptance_errors: Mapping[str, str | None]
) -> TrlPreflightResult:
    """Verify the generation batch arithmetic plus config acceptance.

    Args:
        check: a kind=divisibility preflight check whose probe kwargs
            carry num_generations and generation_batch_size
        acceptance_errors: probe name to error string, None on acceptance

    Returns:
        TrlPreflightResult: verified only when generation_batch_size
            divides evenly by num_generations and GRPOConfig accepted
            the pair
    """
    values = dict(check.probe_kwargs)
    num_generations = int(str(values["num_generations"]))
    generation_batch_size = int(str(values["generation_batch_size"]))
    if generation_batch_size % num_generations != 0:
        return TrlPreflightResult(
            name=check.name,
            description=check.description,
            kind=check.kind,
            status=STATUS_MISMATCH,
            expected=values,
            actual=None,
            detail=(
                f"generation_batch_size={generation_batch_size} is not "
                f"divisible by num_generations={num_generations}"
            ),
        )
    return evaluate_acceptance_check(check, acceptance_errors)


def evaluate_trl_preflight(
    field_defaults: Mapping[str, object],
    acceptance_errors: Mapping[str, str | None],
    checks: tuple[TrlPreflightCheck, ...] = TRL_PREFLIGHT_CHECKS,
) -> tuple[TrlPreflightResult, ...]:
    """Turn raw GRPOConfig observations into per-check verdicts.

    Assumptions:
        1. Any check field absent from field_defaults is API drift and
           short-circuits to missing regardless of kind, except runtime
           checks which stay deferred only when their fields exist.

    Args:
        field_defaults: field name to default value, as read from
            dataclasses.fields(GRPOConfig) by the backend
        acceptance_errors: probe name to error string, None on acceptance
        checks: the preflight checks to evaluate

    Returns:
        tuple: one TrlPreflightResult per check, in check order
    """
    results: list[TrlPreflightResult] = []
    for check in checks:
        absent = missing_fields(check, field_defaults)
        if absent:
            results.append(
                TrlPreflightResult(
                    name=check.name,
                    description=check.description,
                    kind=check.kind,
                    status=STATUS_MISSING,
                    expected=(
                        check.expected_default or dict(check.probe_kwargs) or None
                    ),
                    actual=None,
                    detail=f"GRPOConfig fields absent: {', '.join(absent)}",
                )
            )
        elif check.kind == KIND_RUNTIME:
            results.append(
                TrlPreflightResult(
                    name=check.name,
                    description=check.description,
                    kind=check.kind,
                    status=STATUS_DEFERRED,
                    expected=None,
                    actual=None,
                    detail=(
                        "needs a CUDA GPU with vLLM; deferred until a "
                        "training box is provisioned"
                    ),
                )
            )
        elif check.kind == KIND_DEFAULT:
            results.append(evaluate_default_check(check, field_defaults))
        elif check.kind == KIND_RECORD:
            actual = field_defaults[check.fields[0]]
            results.append(
                TrlPreflightResult(
                    name=check.name,
                    description=check.description,
                    kind=check.kind,
                    status=STATUS_RECORDED,
                    expected=None,
                    actual=actual,
                    detail=f"installed default recorded: {actual!r}",
                )
            )
        elif check.kind == KIND_DIVISIBILITY:
            results.append(evaluate_divisibility_check(check, acceptance_errors))
        else:
            results.append(evaluate_acceptance_check(check, acceptance_errors))
    return tuple(results)


def preflight_passed(results: tuple[TrlPreflightResult, ...]) -> bool:
    """Decide whether the static portion of the TRL preflight is complete.

    Assumptions:
        1. deferred never fails the preflight — runtime checks are
           explicitly out of scope until a GPU box exists.

    Args:
        results: the evaluated preflight verdicts

    Returns:
        bool: True iff no result is missing or mismatch
    """
    return all(
        result.status not in (STATUS_MISSING, STATUS_MISMATCH) for result in results
    )
