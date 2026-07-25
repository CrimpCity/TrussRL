"""Tests for the serializable training config.

Pins the recipe values, the JSON round trip, the kwarg projections, and
their reconciliation against both the preflight checks and installed
TRL/peft.
"""

import dataclasses
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import cast

import pytest

from trussRL.calibration.artifacts import write_json
from trussRL.training.config import (
    FALLBACK_EPSILON,
    FALLBACK_EPSILON_HIGH,
    FALLBACK_LOSS_TYPE,
    LEARNING_RATE_SCREEN,
    LoraSettings,
    TrainingConfig,
    fallback_training_config,
    grpo_config_kwargs,
    lora_config_kwargs,
    training_config_from_payload,
    training_config_payload,
)
from trussRL.training.model_ids import MODEL_ID, MODEL_REVISION
from trussRL.training.preflight import (
    KIND_ACCEPTANCE,
    KIND_DIVISIBILITY,
    TRL_PREFLIGHT_CHECKS,
)


def round_trip(config: TrainingConfig) -> TrainingConfig:
    """Push a config through JSON text and back.

    Args:
        config: the config to serialize and reload

    Returns:
        TrainingConfig: the config rebuilt from the JSON round trip
    """
    payload = json.loads(json.dumps(training_config_payload(config)))
    return training_config_from_payload(payload)


def test_default_config_matches_recipe_values() -> None:
    config = TrainingConfig()
    assert config.model_id == MODEL_ID
    assert config.model_revision == MODEL_REVISION
    assert config.loss_type == "cispo"
    assert config.epsilon == 0.2
    assert config.epsilon_high == 5.0
    assert config.scale_rewards == "batch"
    assert config.beta == 0.0
    assert config.cast_lm_head_to_fp32 is True
    assert config.learning_rate == 5e-5
    assert config.num_generations == 8
    assert config.generation_batch_size == 24
    assert config.temperature == 0.7
    assert config.top_p == 0.95
    assert config.max_completion_length == 256
    assert config.use_vllm is True
    assert config.vllm_mode == "colocate"
    assert config.vllm_importance_sampling_correction is True
    assert config.vllm_importance_sampling_clip_max == 2.0
    assert config.filter_zero_variance_groups is True
    assert config.retirement_success_threshold == 0.9
    assert config.lora == LoraSettings(
        r=8, alpha=32, dropout=0.0, target_modules="all-linear"
    )
    assert LEARNING_RATE_SCREEN == (1e-5, 5e-5, 1e-4)
    assert config.learning_rate == LEARNING_RATE_SCREEN[1]


def test_payload_round_trips_exactly() -> None:
    default = TrainingConfig()
    variants = (
        default,
        dataclasses.replace(default, learning_rate=LEARNING_RATE_SCREEN[0]),
        fallback_training_config(default),
        dataclasses.replace(default, lora=LoraSettings(r=16, alpha=64)),
    )
    for config in variants:
        assert round_trip(config) == config


def test_payload_uses_vllm_importance_sampling_clip_max_key() -> None:
    payload = training_config_payload(TrainingConfig())
    assert payload["vllm_importance_sampling_clip_max"] == 2.0


def test_payload_survives_write_json(tmp_path: Path) -> None:
    config = TrainingConfig()
    target = tmp_path / "training_config.json"
    write_json(target, training_config_payload(config))
    reloaded = training_config_from_payload(json.loads(target.read_text()))
    assert reloaded == config


def test_from_payload_rejects_extra_and_missing_keys() -> None:
    payload = training_config_payload(TrainingConfig())
    extra = dict(payload)
    extra["max_prompt_length"] = 800
    with pytest.raises(ValueError, match="max_prompt_length"):
        training_config_from_payload(extra)
    missing = dict(payload)
    del missing["temperature"]
    with pytest.raises(ValueError, match="temperature"):
        training_config_from_payload(missing)
    bad_lora = dict(payload)
    lora = cast(dict[str, object], payload["lora"])
    bad_lora["lora"] = {**lora, "rank": 8}
    with pytest.raises(ValueError, match="rank"):
        training_config_from_payload(bad_lora)


def test_grpo_kwargs_cover_every_preflight_check_field() -> None:
    kwargs = grpo_config_kwargs(TrainingConfig())
    for check in TRL_PREFLIGHT_CHECKS:
        for field_name in check.fields:
            assert field_name in kwargs, (check.name, field_name)


def test_grpo_kwargs_agree_with_preflight_probe_values() -> None:
    config = TrainingConfig()
    primary_kwargs = grpo_config_kwargs(config)
    fallback_kwargs = grpo_config_kwargs(fallback_training_config(config))
    for check in TRL_PREFLIGHT_CHECKS:
        if check.kind not in (KIND_ACCEPTANCE, KIND_DIVISIBILITY):
            continue
        probed = dict(check.probe_kwargs)
        if check.name == "fallback_ppo_clip":
            assert probed.items() <= fallback_kwargs.items()
        else:
            assert probed.items() <= primary_kwargs.items(), check.name


def test_fallback_changes_only_the_loss_knobs() -> None:
    config = TrainingConfig()
    fallback = fallback_training_config(config)
    assert fallback.loss_type == FALLBACK_LOSS_TYPE == "grpo"
    assert fallback.epsilon == FALLBACK_EPSILON == 0.2
    assert fallback.epsilon_high == FALLBACK_EPSILON_HIGH == 0.28
    loss_keys = {"loss_type", "epsilon", "epsilon_high"}
    primary_rest = {
        key: value
        for key, value in training_config_payload(config).items()
        if key not in loss_keys
    }
    fallback_rest = {
        key: value
        for key, value in training_config_payload(fallback).items()
        if key not in loss_keys
    }
    assert primary_rest == fallback_rest


def test_is_weight_cap_never_survives_into_ppo_clip() -> None:
    with pytest.raises(ValueError, match="epsilon_high"):
        TrainingConfig(loss_type="grpo")
    config = TrainingConfig()
    with pytest.raises(ValueError, match="epsilon_high"):
        dataclasses.replace(config, loss_type="grpo")
    fallback = fallback_training_config(config)
    assert fallback.loss_type == "grpo"


def test_indivisible_generation_batch_rejected() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        TrainingConfig(generation_batch_size=25)
    assert TrainingConfig(generation_batch_size=24).generation_batch_size == 24


def test_grpo_kwargs_resolve_against_installed_trl() -> None:
    trl = pytest.importorskip("trl")
    field_names = {
        config_field.name for config_field in dataclasses.fields(trl.GRPOConfig)
    }
    field_defaults = {
        config_field.name: config_field.default
        for config_field in dataclasses.fields(trl.GRPOConfig)
    }
    assert field_defaults["vllm_importance_sampling_clip_max"] == 3.0
    config = TrainingConfig()
    kwargs_variants = (
        grpo_config_kwargs(config),
        grpo_config_kwargs(fallback_training_config(config)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for kwargs in kwargs_variants:
            assert set(kwargs) <= field_names
            assert kwargs["vllm_importance_sampling_clip_max"] == 2.0
            built = trl.GRPOConfig(**kwargs)
            for name, value in kwargs.items():
                assert getattr(built, name) == value, name


def test_lora_kwargs_resolve_against_installed_peft() -> None:
    peft = pytest.importorskip("peft")
    kwargs = lora_config_kwargs(TrainingConfig())
    built = peft.LoraConfig(**kwargs)
    assert built.r == 8
    assert built.lora_alpha == 32
    assert built.lora_dropout == 0.0
    assert built.target_modules == "all-linear"


def test_config_module_imports_without_training_deps() -> None:
    probe = (
        "import sys\n"
        "import trussRL.training.config\n"
        "loaded = set(sys.modules)\n"
        "assert 'trl' not in loaded, 'config imported trl'\n"
        "assert 'torch' not in loaded, 'config imported torch'\n"
        "assert 'peft' not in loaded, 'config imported peft'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
