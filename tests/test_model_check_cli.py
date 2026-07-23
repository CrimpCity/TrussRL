"""Orchestration tests for the trussrl-model-check CLI.

The heavy backend is replaced by a namespace of stub functions, so these
tests pin the CLI contract — exit codes, manifest schema, the
import-lightness invariant — without torch, TRL, or any download. The
real model run is the acceptance test, executed once per environment.
"""

import functools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import trussRL.cli.model_check as model_check_cli
from trussRL.training.preflight import TRL_PREFLIGHT_CHECKS

runner = CliRunner()

FAKE_SHA = "abc123def456abc123def456abc123def456abc1"
FAKE_LOCK_SHA = "f" * 64


def stub_environment(cuda: bool, mps: bool) -> dict[str, object]:
    """Build a canned environment report.

    Args:
        cuda: reported CUDA availability
        mps: reported MPS availability

    Returns:
        dict: a minimal environment payload
    """
    return {
        "platform": "test-platform",
        "python": "3.13.0",
        "packages": {"trl": "1.9.0"},
        "cuda_available": cuda,
        "mps_available": mps,
    }


def stub_lock_hash() -> str:
    """Return a fixed uv.lock hash.

    Args: None

    Returns:
        str: a 64-hex placeholder digest
    """
    return FAKE_LOCK_SHA


def stub_resolve(revision: str | None) -> tuple[str, str]:
    """Resolve to a fixed local path and sha without downloading.

    Args:
        revision: the requested revision, ignored

    Returns:
        tuple: (fake local path, fixed sha)
    """
    return "/fake/snapshot", FAKE_SHA


def failing_resolve(revision: str | None) -> tuple[str, str]:
    """Fail the way an offline download does.

    Args:
        revision: the requested revision, ignored

    Returns:
        tuple: never returns

    Raises:
        OSError: always.
    """
    raise OSError("network unreachable")


def stub_tokenizer_check(model_path: str) -> dict[str, object]:
    """Return a passing tokenizer check payload.

    Args:
        model_path: the snapshot path handed in by the CLI

    Returns:
        dict: a passing check payload
    """
    return {"status": "pass", "detail": "tokenizer ok", "vocab_size": 151936}


def failing_tokenizer_check(model_path: str) -> dict[str, object]:
    """Fail the way a corrupted snapshot does.

    Args:
        model_path: the snapshot path handed in by the CLI

    Returns:
        dict: never returns

    Raises:
        RuntimeError: always.
    """
    raise RuntimeError("tokenizer files corrupted")


def stub_prompt_check(model_path: str) -> dict[str, object]:
    """Return a passing prompt-length check payload.

    Args:
        model_path: the snapshot path handed in by the CLI

    Returns:
        dict: a passing check payload
    """
    return {"status": "pass", "detail": "max 812 tokens", "max_tokens": 812}


def stub_mps_generation(model_path: str, max_new_tokens: int) -> dict[str, object]:
    """Return a passing MPS generation check payload.

    Args:
        model_path: the snapshot path handed in by the CLI
        max_new_tokens: the completion length handed in by the CLI

    Returns:
        dict: a passing check payload recording the dtype used
    """
    return {"status": "pass", "detail": "generated", "dtype": "bfloat16"}


def unexpected_vllm_generation(
    model_path: str, max_new_tokens: int
) -> dict[str, object]:
    """Fail loudly if the CLI routes to vLLM on a non-CUDA environment.

    Args:
        model_path: the snapshot path handed in by the CLI
        max_new_tokens: the completion length handed in by the CLI

    Returns:
        dict: never returns

    Raises:
        AssertionError: always.
    """
    raise AssertionError("vllm generation must not run without CUDA")


def healthy_config_observations() -> dict[str, object]:
    """Build inspect_trl_config output where every static check passes.

    Args: None

    Returns:
        dict: trl_version, field_defaults, and all-accepted probe results
    """
    field_defaults: dict[str, object] = {
        "loss_type": "dapo",
        "epsilon": 0.2,
        "epsilon_high": None,
        "scale_rewards": "group",
        "cast_lm_head_to_fp32": False,
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_cap": 2.0,
        "vllm_mode": "colocate",
        "num_generations": 8,
        "generation_batch_size": None,
        "beta": 0.0,
    }
    acceptance_errors: dict[str, str | None] = {
        check.name: None
        for check in TRL_PREFLIGHT_CHECKS
        if check.kind in ("acceptance", "divisibility")
    }
    return {
        "trl_version": "1.9.0",
        "field_defaults": field_defaults,
        "acceptance_errors": acceptance_errors,
    }


def inspect_with(observations: dict[str, object]) -> dict[str, object]:
    """Return canned GRPOConfig observations.

    Args:
        observations: the payload to hand back to the CLI

    Returns:
        dict: the observations unchanged
    """
    return observations


def make_backend(**overrides: object) -> SimpleNamespace:
    """Assemble a healthy stub backend, with per-test overrides.

    Args:
        **overrides: attribute replacements applied on top of the
            healthy defaults

    Returns:
        SimpleNamespace: the stubbed model-check backend
    """
    attributes: dict[str, object] = {
        "environment_report": functools.partial(stub_environment, False, True),
        "lock_file_hash": stub_lock_hash,
        "resolve_and_download": stub_resolve,
        "check_tokenizer": stub_tokenizer_check,
        "check_prompt_lengths": stub_prompt_check,
        "check_mps_generation": stub_mps_generation,
        "check_vllm_generation": unexpected_vllm_generation,
        "inspect_trl_config": functools.partial(
            inspect_with, healthy_config_observations()
        ),
    }
    attributes.update(overrides)
    return SimpleNamespace(**attributes)


def return_backend(backend: SimpleNamespace) -> SimpleNamespace:
    """Stand-in loader returning a prebuilt stub backend.

    Args:
        backend: the stub backend to return

    Returns:
        SimpleNamespace: the same backend
    """
    return backend


def raise_missing_extra() -> SimpleNamespace:
    """Fail the way load_backend does without the training extra.

    Args: None

    Returns:
        SimpleNamespace: never returns

    Raises:
        ImportError: always.
    """
    raise ImportError(
        "training dependencies are not installed; "
        "install with: uv sync --extra training"
    )


def patch_backend(monkeypatch, backend: SimpleNamespace) -> None:
    """Point the CLI's load_backend at a stub backend.

    Args:
        monkeypatch: pytest's monkeypatch fixture
        backend: the stub backend to serve

    Returns: None
    """
    monkeypatch.setattr(
        model_check_cli, "load_backend", functools.partial(return_backend, backend)
    )


def read_manifest(output_dir: Path) -> dict[str, object]:
    """Load the single manifest the CLI wrote.

    Args:
        output_dir: the --output-dir handed to the CLI

    Returns:
        dict: the parsed manifest payload
    """
    manifests = list((output_dir / "model_check").glob("model_check_*.json"))
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text())


def test_verifier_import_never_loads_torch() -> None:
    code = "import trussRL.verifier, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_module_import_never_loads_torch() -> None:
    code = "import trussRL.cli.model_check, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_happy_path_writes_passing_manifest(monkeypatch, tmp_path: Path) -> None:
    patch_backend(monkeypatch, make_backend())
    result = runner.invoke(model_check_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    manifest = read_manifest(tmp_path)
    assert manifest["passed"] is True
    assert manifest["lock_sha256"] == FAKE_LOCK_SHA
    assert set(manifest["stamp"]) == {
        "created_utc",
        "git_sha",
        "git_dirty",
        "catalog_sha256",
        "run_id",
    }
    model = manifest["model"]
    assert model["model_id"] == "Qwen/Qwen3-4B-Base"
    assert model["resolved_revision"] == FAKE_SHA
    assert model["local_path"] == "/fake/snapshot"
    assert isinstance(model["revision_match"], bool)
    statuses = {check["name"]: check["status"] for check in manifest["checks"]}
    assert statuses["download"] == "pass"
    assert statuses["tokenizer"] == "pass"
    assert statuses["prompt_lengths"] == "pass"
    assert statuses["mps_generation"] == "pass"
    assert statuses["vllm_generation"] == "deferred"
    preflight = manifest["trl_preflight"]
    assert preflight["trl_version"] == "1.9.0"
    assert preflight["static_complete"] is True
    assert preflight["deferred"] == ["vllm_colocate_lora_sync"]
    assert len(preflight["items"]) == len(TRL_PREFLIGHT_CHECKS)


def test_missing_training_extra_exits_with_remedy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(model_check_cli, "load_backend", raise_missing_extra)
    result = runner.invoke(model_check_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "uv sync --extra training" in result.output
    assert not (tmp_path / "model_check").exists()


def test_failing_check_fails_run_but_writes_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    patch_backend(monkeypatch, make_backend(check_tokenizer=failing_tokenizer_check))
    result = runner.invoke(model_check_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    manifest = read_manifest(tmp_path)
    assert manifest["passed"] is False
    tokenizer = next(
        check for check in manifest["checks"] if check["name"] == "tokenizer"
    )
    assert tokenizer["status"] == "fail"
    assert "RuntimeError" in tokenizer["detail"]


def test_mismatched_preflight_check_fails_run(monkeypatch, tmp_path: Path) -> None:
    observations = healthy_config_observations()
    errors = observations["acceptance_errors"]
    assert isinstance(errors, dict)
    errors["scale_rewards_batch"] = "ValueError: unknown scale_rewards 'batch'"
    patch_backend(
        monkeypatch,
        make_backend(inspect_trl_config=functools.partial(inspect_with, observations)),
    )
    result = runner.invoke(model_check_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    manifest = read_manifest(tmp_path)
    assert manifest["passed"] is False
    assert manifest["trl_preflight"]["static_complete"] is False


def test_download_failure_skips_model_checks(monkeypatch, tmp_path: Path) -> None:
    patch_backend(monkeypatch, make_backend(resolve_and_download=failing_resolve))
    result = runner.invoke(model_check_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    manifest = read_manifest(tmp_path)
    statuses = {check["name"]: check["status"] for check in manifest["checks"]}
    assert statuses["download"] == "fail"
    assert statuses["tokenizer"] == "skipped"
    assert statuses["prompt_lengths"] == "skipped"
    assert manifest["passed"] is False


def test_cpu_only_environment_is_a_hard_failure(monkeypatch, tmp_path: Path) -> None:
    patch_backend(
        monkeypatch,
        make_backend(
            environment_report=functools.partial(stub_environment, False, False)
        ),
    )
    result = runner.invoke(model_check_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    manifest = read_manifest(tmp_path)
    generation = next(
        check for check in manifest["checks"] if check["name"] == "generation"
    )
    assert generation["status"] == "fail"
    assert "CPU-only" in generation["detail"]
