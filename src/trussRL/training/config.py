"""The training recipe as frozen, JSON-round-trippable data.

One TrainingConfig carries every recipe knob the GRPO trainer consumes;
run manifests embed its payload so a training run always traces to the
exact knob set that produced it. Stdlib-only plus model identity
constants — never imports torch, trl, or peft, so the config loads
without the training extra installed.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from trussRL.training.model_ids import MODEL_ID, MODEL_REVISION

LEARNING_RATE_SCREEN: tuple[float, ...] = (1e-5, 5e-5, 1e-4)

FALLBACK_LOSS_TYPE = "grpo"
FALLBACK_EPSILON = 0.2
FALLBACK_EPSILON_HIGH = 0.28

LORA_TASK_TYPE = "CAUSAL_LM"


@dataclass(frozen=True)
class LoraSettings:
    """The LoRA adapter shape handed to peft.

    Attributes:
        r: adapter rank
        alpha: LoRA scaling numerator (peft lora_alpha)
        dropout: dropout probability on adapter inputs (peft lora_dropout)
        target_modules: which modules receive adapters; "all-linear"
            attaches to every linear layer
    """

    r: int = 8
    alpha: int = 32
    dropout: float = 0.0
    target_modules: str = "all-linear"


@dataclass(frozen=True)
class TrainingConfig:
    """Every knob of one GRPO training run, serializable into manifests.

    GRPO field names are kept identical to their GRPOConfig kwargs so the
    mapping stays name-for-name; LoRA settings nest separately because
    they target peft under renamed kwargs.

    Assumptions:
        1. Under loss_type="cispo", epsilon/epsilon_high are the
           importance-sampling weight cap, not a PPO objective clip —
           which is why epsilon_high=5.0 is valid there and rejected for
           every other loss type.
        2. Installed TRL 1.9.0 defaults the vLLM importance-sampling
           clip maximum to 3.0; the recipe explicitly pins 2.0.
        3. Per-device batch size and gradient accumulation are memory
           tuning left to the trainer, not recipe knobs.

    Attributes:
        model_id: Hugging Face model identifier to train
        model_revision: pinned model revision, None for the default branch
        loss_type: GRPO loss variant
        epsilon: lower clip range
        epsilon_high: upper clip range; the IS-weight cap under CISPO
        scale_rewards: reward scaling scheme
        beta: reference-model KL coefficient; 0.0 disables the reference
            model
        cast_lm_head_to_fp32: whether the LM head runs in fp32 for logit
            stability
        learning_rate: optimizer learning rate; midpoint of
            LEARNING_RATE_SCREEN until the canary screen picks a winner
        num_generations: completions sampled per prompt
        generation_batch_size: completions per generation step; must be a
            multiple of num_generations (24 = 8 completions x 3 prompts)
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        max_completion_length: completion token budget
        use_vllm: whether generation runs through vLLM
        vllm_mode: how vLLM shares the box with the trainer
        vllm_importance_sampling_correction: whether generation/train
            policy mismatch is corrected with IS weights
        vllm_importance_sampling_clip_max: maximum value of those clipped
            IS weights
        filter_zero_variance_groups: drop prompt groups whose completions
            all earned the same reward (no advantage signal); owned by
            the trainer loop, not a TRL kwarg
        retirement_success_threshold: rolling success rate above which a
            prompt is retired from the curriculum; owned by the trainer
            loop, not a TRL kwarg
        lora: LoRA adapter settings handed to peft
    """

    model_id: str = MODEL_ID
    model_revision: str | None = MODEL_REVISION
    loss_type: str = "cispo"
    epsilon: float = 0.2
    epsilon_high: float = 5.0
    scale_rewards: str = "batch"
    beta: float = 0.0
    cast_lm_head_to_fp32: bool = True
    learning_rate: float = 5e-5
    num_generations: int = 8
    generation_batch_size: int = 24
    temperature: float = 0.7
    top_p: float = 0.95
    max_completion_length: int = 256
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_importance_sampling_correction: bool = True
    # TODO: replace the pinned 2.0 clip maximum with a value measured from
    # IS-weight statistics on an early GPU run.
    vllm_importance_sampling_clip_max: float = 2.0
    filter_zero_variance_groups: bool = True
    retirement_success_threshold: float = 0.9
    lora: LoraSettings = field(default_factory=LoraSettings)

    def __post_init__(self) -> None:
        """Reject knob combinations the recipe forbids.

        Assumptions:
            1. Runs on every construction path — direct, dataclasses.replace,
               and payload loading — so an invalid combination can never
               exist as a TrainingConfig instance.

        Args: None

        Returns:
            None

        Raises:
            ValueError: if generation_batch_size is not a multiple of
                num_generations, or if a non-CISPO loss carries an
                epsilon_high >= 1.0 (an IS-weight cap is not a PPO clip
                half-width).
        """
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size={self.generation_batch_size} is not "
                f"divisible by num_generations={self.num_generations}"
            )
        if self.loss_type != "cispo" and self.epsilon_high >= 1.0:
            raise ValueError(
                f"epsilon_high={self.epsilon_high} with "
                f"loss_type={self.loss_type!r}: an epsilon_high >= 1.0 is an "
                "IS-weight cap, only meaningful under CISPO; PPO-clip losses "
                "need a clip half-width below 1.0"
            )


def fallback_training_config(config: TrainingConfig) -> TrainingConfig:
    """Derive the vanilla PPO-clip fallback from a CISPO config.

    Assumptions:
        1. All three loss knobs are overwritten unconditionally so the
           CISPO IS-weight cap can never leak into the fallback's clip
           range.

    Args:
        config: the primary (typically CISPO) training config

    Returns:
        TrainingConfig: the same config with loss_type, epsilon, and
            epsilon_high replaced by the fallback PPO-clip values
    """
    return dataclasses.replace(
        config,
        loss_type=FALLBACK_LOSS_TYPE,
        epsilon=FALLBACK_EPSILON,
        epsilon_high=FALLBACK_EPSILON_HIGH,
    )


def grpo_config_kwargs(config: TrainingConfig) -> dict[str, object]:
    """Extract the kwargs handed verbatim to trl.GRPOConfig.

    Model identity is excluded (it is the trainer's model argument, not a
    GRPOConfig kwarg), as are the LoRA settings (peft) and the curriculum
    fields (trainer-loop logic with no TRL counterpart).

    Args:
        config: the training config to project

    Returns:
        dict: GRPOConfig kwarg name to value, names identical to the
            config's field names
    """
    return {
        "loss_type": config.loss_type,
        "epsilon": config.epsilon,
        "epsilon_high": config.epsilon_high,
        "scale_rewards": config.scale_rewards,
        "beta": config.beta,
        "cast_lm_head_to_fp32": config.cast_lm_head_to_fp32,
        "learning_rate": config.learning_rate,
        "num_generations": config.num_generations,
        "generation_batch_size": config.generation_batch_size,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_completion_length": config.max_completion_length,
        "use_vllm": config.use_vllm,
        "vllm_mode": config.vllm_mode,
        "vllm_importance_sampling_correction": (
            config.vllm_importance_sampling_correction
        ),
        "vllm_importance_sampling_clip_max": (config.vllm_importance_sampling_clip_max),
    }


def lora_config_kwargs(config: TrainingConfig) -> dict[str, object]:
    """Extract the kwargs handed to peft.LoraConfig.

    Args:
        config: the training config whose LoRA settings to project

    Returns:
        dict: peft LoraConfig kwargs, including the fixed causal-LM task
            type
    """
    return {
        "r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules": config.lora.target_modules,
        "task_type": LORA_TASK_TYPE,
    }


def training_config_payload(config: TrainingConfig) -> dict[str, object]:
    """Serialize a training config for manifest embedding.

    Args:
        config: the training config to serialize

    Returns:
        dict: every config field, the LoRA settings nested
    """
    return dataclasses.asdict(config)


def training_config_from_payload(payload: Mapping[str, object]) -> TrainingConfig:
    """Rebuild a training config from a manifest payload.

    Assumptions:
        1. Strict on shape: every field has a default, so a stale or
           renamed key would otherwise silently half-load — extra and
           missing keys both fail instead.
        2. Validation reruns on construction, so an invalid stored
           combination is rejected the same way a fresh one is.

    Args:
        payload: the deserialized manifest payload, as produced by
            training_config_payload

    Returns:
        TrainingConfig: the reconstructed config

    Raises:
        ValueError: if the payload's keys do not exactly match the config
            fields, at either nesting level.
    """
    expected = {
        config_field.name for config_field in dataclasses.fields(TrainingConfig)
    }
    actual = set(payload)
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(
            f"payload keys do not match TrainingConfig fields: "
            f"extra={extra}, missing={missing}"
        )
    lora_payload = payload["lora"]
    if not isinstance(lora_payload, Mapping):
        raise ValueError("payload key 'lora' must be a mapping of LoRA settings")
    lora_expected = {lora_field.name for lora_field in dataclasses.fields(LoraSettings)}
    lora_actual = set(lora_payload)
    if lora_actual != lora_expected:
        extra = sorted(lora_actual - lora_expected)
        missing = sorted(lora_expected - lora_actual)
        raise ValueError(
            f"payload key 'lora' does not match LoraSettings fields: "
            f"extra={extra}, missing={missing}"
        )
    flat = {key: value for key, value in payload.items() if key != "lora"}
    return TrainingConfig(
        **cast(dict[str, Any], flat),
        lora=LoraSettings(**cast(dict[str, Any], lora_payload)),
    )
