"""Orchestration and metric-math tests for the trussrl-rollout-gate CLI.

The heavy backend is replaced by a namespace of stub functions serving
canned rollouts, while the committed calibration artifacts back the real
eval-split loader and the real verifier scores the canned completions.
These tests pin the CLI contract — exit codes, artifact schema, verdict
semantics, the import-lightness invariant — and the pure metric formulas,
without torch or any download. The real model run is the acceptance test.
"""

import functools
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import trussRL.cli.rollout_gate as rollout_gate_cli
from trussRL.calibration.roster import load_eval_split
from trussRL.cli.reward_bench import render_completion
from trussRL.cli.rollout_gate import PromptRollouts
from trussRL.reward import RewardBreakdown
from trussRL.training.model_ids import MODEL_REVISION

runner = CliRunner()

FAKE_SHA = "abc123def456abc123def456abc123def456abc1"
FAKE_LOCK_SHA = "e" * 64
GROUP_SIZE = 8
SMOKE_ARGS = ["--limit-prompts", "2", "--rollouts-per-prompt", "8"]

RAMBLE_TEXT = "A truss with a handful of bays and a modest depth seems sensible."
TRUNCATED_TEXT = '```json\n{"topolog'

EVAL_SPLIT = load_eval_split()
SWEEP_BEST_COMPLETIONS = tuple(
    "The frozen best design should score cleanly.\n"
    + render_completion(eval_instance.sweep_best_design)
    for eval_instance in EVAL_SPLIT
)


def canned_rollout(
    text: str,
    token_count: int,
    truncated: bool,
    entropy_sum: float,
    entropy_tokens: int,
) -> dict[str, object]:
    """Build one canned rollout record for the stub sampler.

    Args:
        text: the completion text the verifier will score
        token_count: valid tokens the completion claims
        truncated: whether the completion hit the cap with no EOS
        entropy_sum: the completion's canned entropy sum in nats
        entropy_tokens: valid tokens behind the entropy sum

    Returns:
        dict: the canned rollout record
    """
    return {
        "text": text,
        "token_count": token_count,
        "truncated": truncated,
        "entropy_sum": entropy_sum,
        "entropy_tokens": entropy_tokens,
    }


def mixed_group(prompt_position: int) -> list[dict[str, object]]:
    """Build a canned group of 6 sweep-best rollouts, a ramble, and a cap hit.

    Args:
        prompt_position: the eval-split position whose sweep-best design
            the parseable rollouts render

    Returns:
        list: eight canned rollout records
    """
    sweep_best = SWEEP_BEST_COMPLETIONS[prompt_position]
    return [canned_rollout(sweep_best, 120, False, 60.0, 120) for _ in range(6)] + [
        canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40),
        canned_rollout(TRUNCATED_TEXT, 64, True, 96.0, 64),
    ]


SMOKE_ROLLOUTS = mixed_group(0)


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
        "packages": {"torch": "2.11.0"},
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


def none_lock_hash() -> None:
    """Report the uv.lock file as unavailable.

    Args: None

    Returns: None
    """
    return None


def fixed_stamp(git_dirty: bool, git_sha: str = FAKE_SHA) -> dict[str, object]:
    """Build a deterministic provenance stamp for authority tests.

    Args:
        git_dirty: the working-tree state the stamp reports
        git_sha: the commit sha the stamp reports

    Returns:
        dict: a provenance stamp with fixed identifiers
    """
    return {
        "created_utc": "2026-07-28T00:00:00+00:00",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "catalog_sha256": "c" * 64,
        "run_id": "20260728T000000Z_teststamp",
    }


def pop_stamp(stamps: list[dict[str, object]]) -> dict[str, object]:
    """Serve the next queued provenance stamp on each successive call.

    Args:
        stamps: per-call stamps, consumed front to back

    Returns:
        dict: the next queued stamp
    """
    return stamps.pop(0)


def patch_stamp(monkeypatch: pytest.MonkeyPatch, git_dirty: bool = False) -> None:
    """Pin the CLI's provenance stamp to a fixed working-tree state.

    Args:
        monkeypatch: pytest's monkeypatch fixture
        git_dirty: the working-tree state the stamp reports

    Returns: None
    """
    monkeypatch.setattr(
        rollout_gate_cli,
        "provenance_stamp",
        functools.partial(fixed_stamp, git_dirty),
    )


def stub_resolve(revision: str | None) -> tuple[str, str]:
    """Resolve the requested revision to a fixed local path without downloading.

    Args:
        revision: the requested revision, echoed back as resolved

    Returns:
        tuple: (fake local path, the requested revision or a fixed sha)
    """
    return "/fake/snapshot", revision if revision is not None else FAKE_SHA


def stub_seed(seed: int) -> None:
    """Accept the seed without touching torch.

    Args:
        seed: the seed handed in by the CLI

    Returns: None
    """


def stub_load_sampler(model_path: str, device: str) -> dict[str, object]:
    """Return a sampler record without loading any model.

    Args:
        model_path: the snapshot path handed in by the CLI
        device: the resolved generation device

    Returns:
        dict: a minimal sampler payload
    """
    return {
        "tokenizer": None,
        "model": None,
        "device": device,
        "dtype": "bfloat16",
        "lm_head_dtype": "float32",
        "eos_token_id": 151643,
    }


def serve_group(
    rollouts: list[dict[str, object]],
    sampler: dict[str, object],
    prompt: str,
    group_size: int,
    group_batch_size: int,
    max_new_tokens: int,
    sampling: dict[str, object],
) -> dict[str, object]:
    """Serve one canned rollout group in the backend's sample_group shape.

    Args:
        rollouts: the canned rollout records, one per requested rollout
        sampler: the sampler record, ignored
        prompt: the rendered prompt, ignored
        group_size: total rollouts requested; must match the canned count
        group_batch_size: rollouts per generate call, ignored
        max_new_tokens: completion token cap, ignored
        sampling: the explicit sampling policy, ignored

    Returns:
        dict: the sample_group result built from the canned records
    """
    assert len(rollouts) == group_size
    token_counts = [cast(int, rollout["token_count"]) for rollout in rollouts]
    return {
        "texts": [rollout["text"] for rollout in rollouts],
        "token_counts": token_counts,
        "truncated": [rollout["truncated"] for rollout in rollouts],
        "entropy_sums": [rollout["entropy_sum"] for rollout in rollouts],
        "entropy_token_counts": [rollout["entropy_tokens"] for rollout in rollouts],
        "new_tokens": sum(token_counts),
        "elapsed_s": 2.0,
    }


def serve_group_sequence(
    groups: list[list[dict[str, object]]],
    sampler: dict[str, object],
    prompt: str,
    group_size: int,
    group_batch_size: int,
    max_new_tokens: int,
    sampling: dict[str, object],
) -> dict[str, object]:
    """Serve a different canned group on each successive sample_group call.

    Args:
        groups: per-call canned groups, consumed front to back
        sampler: the sampler record, ignored
        prompt: the rendered prompt, ignored
        group_size: total rollouts requested per call
        group_batch_size: rollouts per generate call, ignored
        max_new_tokens: completion token cap, ignored
        sampling: the explicit sampling policy, ignored

    Returns:
        dict: the sample_group result for this call's group
    """
    return serve_group(
        groups.pop(0),
        sampler,
        prompt,
        group_size,
        group_batch_size,
        max_new_tokens,
        sampling,
    )


def make_backend(
    rollouts: list[dict[str, object]], **overrides: object
) -> SimpleNamespace:
    """Assemble a healthy stub backend, with per-test overrides.

    Args:
        rollouts: canned rollouts every sample_group call serves
        **overrides: attribute replacements applied on top of the
            healthy defaults

    Returns:
        SimpleNamespace: the stubbed rollout-gate backend
    """
    attributes: dict[str, object] = {
        "environment_report": functools.partial(stub_environment, False, True),
        "lock_file_hash": stub_lock_hash,
        "resolve_and_download": stub_resolve,
        "seed_generation": stub_seed,
        "load_sampler": stub_load_sampler,
        "sample_group": functools.partial(serve_group, rollouts),
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


def patch_backend(monkeypatch: pytest.MonkeyPatch, backend: SimpleNamespace) -> None:
    """Point the CLI's load_backend at a stub backend.

    Args:
        monkeypatch: pytest's monkeypatch fixture
        backend: the stub backend to serve

    Returns: None
    """
    monkeypatch.setattr(
        rollout_gate_cli, "load_backend", functools.partial(return_backend, backend)
    )


def read_artifact(output_dir: Path) -> dict[str, Any]:
    """Load the single gate artifact the CLI wrote.

    Args:
        output_dir: the --output-dir handed to the CLI

    Returns:
        dict: the parsed artifact payload
    """
    artifacts = list((output_dir / "rollout_gate").glob("rollout_gate_*.json"))
    assert len(artifacts) == 1
    return cast(dict[str, Any], json.loads(artifacts[0].read_text()))


def make_prompt_rollouts(
    breakdowns: tuple[RewardBreakdown, ...],
    texts: tuple[str, ...],
    token_counts: tuple[int, ...],
    truncated: tuple[bool, ...],
    entropy_sums: tuple[float, ...],
    entropy_token_counts: tuple[int, ...],
    sweep_best_cost_usd: float = 100.0,
    new_tokens: int | None = None,
    elapsed_s: float = 2.0,
) -> PromptRollouts:
    """Build a PromptRollouts with sensible defaults for math tests.

    Args:
        breakdowns: the per-completion verifier breakdowns
        texts: the completion texts, aligned with breakdowns
        token_counts: valid-token counts, aligned with breakdowns
        truncated: cap-truncation flags, aligned with breakdowns
        entropy_sums: per-completion entropy sums
        entropy_token_counts: valid tokens behind the entropy sums
        sweep_best_cost_usd: the sweep-best cost anchoring cost gaps
        new_tokens: total valid tokens; defaults to sum(token_counts)
        elapsed_s: summed generate wall time

    Returns:
        PromptRollouts: the assembled record
    """
    return PromptRollouts(
        index=3,
        sweep_best_cost_usd=sweep_best_cost_usd,
        sweep_best_score=0.95,
        prompt_sha256="0" * 64,
        texts=texts,
        breakdowns=breakdowns,
        token_counts=token_counts,
        truncated=truncated,
        entropy_sums=entropy_sums,
        entropy_token_counts=entropy_token_counts,
        new_tokens=sum(token_counts) if new_tokens is None else new_tokens,
        elapsed_s=elapsed_s,
    )


def test_cli_module_imports_without_torch() -> None:
    code = "import trussRL.cli.rollout_gate, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_training_extra_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(rollout_gate_cli, "load_backend", raise_missing_extra)
    result = runner.invoke(
        rollout_gate_cli.app, [*SMOKE_ARGS, "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "uv sync --extra training" in result.output
    assert not (tmp_path / "rollout_gate").exists()


def test_cpu_only_refused_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_backend(
        monkeypatch,
        make_backend(
            SMOKE_ROLLOUTS,
            environment_report=functools.partial(stub_environment, False, False),
        ),
    )
    result = runner.invoke(
        rollout_gate_cli.app, [*SMOKE_ARGS, "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "CPU-only" in result.output
    assert not (tmp_path / "rollout_gate").exists()


def test_rollouts_not_multiple_of_group_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_backend(monkeypatch, make_backend(SMOKE_ROLLOUTS))
    result = runner.invoke(
        rollout_gate_cli.app,
        ["--rollouts-per-prompt", "12", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert not (tmp_path / "rollout_gate").exists()


def test_smoke_run_writes_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_backend(
        monkeypatch,
        make_backend(
            SMOKE_ROLLOUTS,
            sample_group=functools.partial(
                serve_group_sequence, [mixed_group(0), mixed_group(1)]
            ),
        ),
    )
    result = runner.invoke(
        rollout_gate_cli.app,
        [*SMOKE_ARGS, "--max-new-tokens", "64", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert set(artifact) == {
        "stamp",
        "stamp_at_write",
        "config",
        "environment",
        "inputs",
        "data_use",
        "per_prompt",
        "metrics",
        "entropy",
        "verdict",
    }
    assert set(artifact["stamp_at_write"]) == {"git_sha", "git_dirty"}
    assert artifact["data_use"] == {
        "prompt_source": "held_out_eval_split",
        "completions": "audit_only",
    }
    assert set(artifact["inputs"]) == {
        "cost_ref_sha256",
        "sweep_best_sha256",
        "lock_sha256",
    }
    assert artifact["inputs"]["lock_sha256"] == FAKE_LOCK_SHA
    config = artifact["config"]
    assert config["run_mode"] == "smoke"
    assert config["resolved_revision"] == MODEL_REVISION
    assert config["sampling"] == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "min_p": None,
        "repetition_penalty": 1.0,
    }
    metrics = artifact["metrics"]
    assert metrics["n_completions"] == 16
    assert metrics["parse_rate"] == pytest.approx(0.75)
    assert metrics["rung3_rate"] == pytest.approx(0.75)
    assert metrics["cap_hit_structured_parse_failure_rate"] == pytest.approx(0.125)
    assert metrics["truncation_rate"] == pytest.approx(0.125)
    assert metrics["feasibility_rate"] == pytest.approx(0.375)
    assert metrics["feasibility_given_rung3"] == pytest.approx(0.5)
    assert metrics["cost_gap_mean"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["mean_group_reward_std"] > 0.0
    assert metrics["zero_variance_group_fraction"] == 0.0
    expected_baseline = (6 * 60.0 + 80.0 + 96.0) / (6 * 120 + 40 + 64)
    assert artifact["entropy"]["baseline"] == pytest.approx(expected_baseline)
    assert artifact["entropy"]["unit"] == "nats_per_token"
    verdict = artifact["verdict"]
    assert verdict["authoritative"] is False
    assert any("smoke" in reason for reason in verdict["non_authoritative_reasons"])
    assert verdict["sft_recommended"] is False
    assert verdict["completion_cap"] == 512
    assert verdict["measured_completion_cap"] == 64
    assert verdict["rerun_at_raised_cap_required"] is True
    assert verdict["variance_gate_passed"] is True
    assert verdict["zero_variance_prompt_indices"] == []
    assert verdict["entropy_baseline"] == pytest.approx(expected_baseline)
    per_prompt = artifact["per_prompt"]
    assert len(per_prompt) == 2
    assert per_prompt[0]["index"] == EVAL_SPLIT[0].index
    assert len(per_prompt[0]["completions"]) == 8
    assert per_prompt[0]["cost_gap_mean"] is None
    assert per_prompt[1]["cost_gap_mean"] == pytest.approx(0.0, abs=1e-9)


def test_sft_recommended_when_rung3_low(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rambles = [canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40) for _ in range(8)]
    patch_backend(monkeypatch, make_backend(rambles))
    result = runner.invoke(
        rollout_gate_cli.app, [*SMOKE_ARGS, "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert artifact["metrics"]["rung3_rate"] == 0.0
    assert artifact["verdict"]["sft_recommended"] is True
    assert artifact["verdict"]["completion_cap"] == 256
    assert artifact["verdict"]["variance_gate_passed"] is False
    assert artifact["verdict"]["zero_variance_prompt_indices"] == [
        eval_instance.index for eval_instance in EVAL_SPLIT[:2]
    ]


def test_canonical_run_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rambles = [canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40) for _ in range(64)]
    patch_backend(monkeypatch, make_backend(rambles))
    patch_stamp(monkeypatch)
    result = runner.invoke(rollout_gate_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert artifact["config"]["run_mode"] == "canonical"
    assert artifact["config"]["n_prompts"] == len(EVAL_SPLIT)
    assert artifact["metrics"]["n_completions"] == 64 * len(EVAL_SPLIT)
    verdict = artifact["verdict"]
    assert verdict["authoritative"] is True
    assert verdict["non_authoritative_reasons"] == []
    assert verdict["measured_completion_cap"] == 256
    assert verdict["rerun_at_raised_cap_required"] is False
    assert verdict["variance_gate_passed"] is False
    assert verdict["zero_variance_prompt_indices"] == [
        eval_instance.index for eval_instance in EVAL_SPLIT
    ]


def test_canonical_run_at_raised_cap_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rambles = [canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40) for _ in range(64)]
    patch_backend(monkeypatch, make_backend(rambles))
    patch_stamp(monkeypatch)
    result = runner.invoke(
        rollout_gate_cli.app,
        ["--max-new-tokens", "512", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert artifact["config"]["run_mode"] == "canonical"
    verdict = artifact["verdict"]
    assert verdict["authoritative"] is True
    assert verdict["completion_cap"] == 512
    assert verdict["measured_completion_cap"] == 512
    assert verdict["rerun_at_raised_cap_required"] is False


def test_revision_override_not_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rambles = [canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40) for _ in range(64)]
    patch_backend(monkeypatch, make_backend(rambles))
    patch_stamp(monkeypatch)
    result = runner.invoke(
        rollout_gate_cli.app,
        ["--revision", "some-other-branch", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert artifact["config"]["run_mode"] == "canonical"
    assert artifact["config"]["resolved_revision"] == "some-other-branch"
    verdict = artifact["verdict"]
    assert verdict["authoritative"] is False
    reasons = verdict["non_authoritative_reasons"]
    assert len(reasons) == 1 and "pinned revision" in reasons[0]


def test_missing_lock_hash_not_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rambles = [canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40) for _ in range(64)]
    patch_backend(
        monkeypatch,
        make_backend(rambles, lock_file_hash=none_lock_hash),
    )
    patch_stamp(monkeypatch)
    result = runner.invoke(rollout_gate_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert artifact["config"]["run_mode"] == "canonical"
    verdict = artifact["verdict"]
    assert verdict["authoritative"] is False
    reasons = verdict["non_authoritative_reasons"]
    assert len(reasons) == 1 and "uv.lock" in reasons[0]


def test_dirty_tree_not_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rambles = [canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40) for _ in range(64)]
    patch_backend(monkeypatch, make_backend(rambles))
    patch_stamp(monkeypatch, git_dirty=True)
    result = runner.invoke(rollout_gate_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert artifact["config"]["run_mode"] == "canonical"
    verdict = artifact["verdict"]
    assert verdict["authoritative"] is False
    reasons = verdict["non_authoritative_reasons"]
    assert len(reasons) == 1 and "uncommitted" in reasons[0]


def test_unknown_git_sha_not_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rambles = [canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40) for _ in range(64)]
    patch_backend(monkeypatch, make_backend(rambles))
    monkeypatch.setattr(
        rollout_gate_cli,
        "provenance_stamp",
        functools.partial(fixed_stamp, False, "unknown"),
    )
    result = runner.invoke(rollout_gate_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert artifact["config"]["run_mode"] == "canonical"
    verdict = artifact["verdict"]
    assert verdict["authoritative"] is False
    reasons = verdict["non_authoritative_reasons"]
    assert len(reasons) == 1 and "git sha is unknown" in reasons[0]


def test_source_changed_during_run_not_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rambles = [canned_rollout(RAMBLE_TEXT, 40, False, 80.0, 40) for _ in range(64)]
    patch_backend(monkeypatch, make_backend(rambles))
    monkeypatch.setattr(
        rollout_gate_cli,
        "provenance_stamp",
        functools.partial(pop_stamp, [fixed_stamp(False), fixed_stamp(True)]),
    )
    result = runner.invoke(rollout_gate_cli.app, ["--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    artifact = read_artifact(tmp_path)
    assert artifact["config"]["run_mode"] == "canonical"
    assert artifact["stamp"]["git_dirty"] is False
    assert artifact["stamp_at_write"] == {"git_sha": FAKE_SHA, "git_dirty": True}
    verdict = artifact["verdict"]
    assert verdict["authoritative"] is False
    reasons = verdict["non_authoritative_reasons"]
    assert len(reasons) == 1 and "changed during the run" in reasons[0]


def test_invalid_device_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_backend(monkeypatch, make_backend(SMOKE_ROLLOUTS))
    result = runner.invoke(
        rollout_gate_cli.app,
        [*SMOKE_ARGS, "--device", "cdua", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "device must be auto, cuda, or mps" in result.output
    assert not (tmp_path / "rollout_gate").exists()


def test_group_reward_spread() -> None:
    spread = rollout_gate_cli.group_reward_spread(
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], 4
    )
    assert len(spread) == 2
    assert spread[0]["std_population"] == 0.0
    assert spread[0]["variance_population"] == 0.0
    assert spread[1]["std_population"] == pytest.approx(0.5)
    assert spread[1]["variance_population"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        rollout_gate_cli.group_reward_spread([1.0, 0.0, 1.0], 2)
    with pytest.raises(ValueError):
        rollout_gate_cli.group_reward_spread([], 2)


def test_truncation_category() -> None:
    assert rollout_gate_cli.truncation_category(False, 0, TRUNCATED_TEXT) is None
    assert (
        rollout_gate_cli.truncation_category(True, 1, SWEEP_BEST_COMPLETIONS[0])
        == "cap_hit_parseable"
    )
    assert (
        rollout_gate_cli.truncation_category(True, 0, TRUNCATED_TEXT)
        == "cap_hit_structured_parse_failure"
    )
    assert (
        rollout_gate_cli.truncation_category(True, 0, RAMBLE_TEXT)
        == "cap_hit_unstructured"
    )
    bare_json_cut = 'Reasoning first.\n{"truss_type": "warren", "n_bays"'
    assert (
        rollout_gate_cli.truncation_category(True, 0, bare_json_cut)
        == "cap_hit_structured_parse_failure"
    )
    nested_cut = 'Final answer: {"truss_type": "pratt", "extras": {"a": 1}'
    assert (
        rollout_gate_cli.truncation_category(True, 0, nested_cut)
        == "cap_hit_structured_parse_failure"
    )
    quoted_marker = "You could open a ```json fence and give the design there."
    assert (
        rollout_gate_cli.truncation_category(True, 0, quoted_marker)
        == "cap_hit_unstructured"
    )
    prose_brace = "A brace { by itself proves nothing about a design"
    assert (
        rollout_gate_cli.truncation_category(True, 0, prose_brace)
        == "cap_hit_unstructured"
    )
    malformed_complete = 'Here.\n```json\n{"truss_type": 5}\n```\nMore words'
    assert (
        rollout_gate_cli.truncation_category(True, 0, malformed_complete)
        == "cap_hit_unstructured"
    )
    closed_fence_incomplete_json = 'x ```json\n{"a": \n``` prose'
    assert (
        rollout_gate_cli.truncation_category(True, 0, closed_fence_incomplete_json)
        == "cap_hit_unstructured"
    )
    complete_then_cut_fence = (
        '```json\n{"a": 1}\n```\nRevised: ```json\n{"truss_type": "warren", "n_ba'
    )
    assert (
        rollout_gate_cli.truncation_category(True, 0, complete_then_cut_fence)
        == "cap_hit_structured_parse_failure"
    )


def test_truncation_evidence() -> None:
    assert rollout_gate_cli.truncation_evidence(TRUNCATED_TEXT)
    assert not rollout_gate_cli.truncation_evidence(
        "You could open a ```json fence and give the design there."
    )
    assert rollout_gate_cli.truncation_evidence(
        'Reasoning first.\n{"truss_type": "warren", "n_bays"'
    )
    assert not rollout_gate_cli.truncation_evidence(RAMBLE_TEXT)
    assert not rollout_gate_cli.truncation_evidence('x ```json\n{"a": \n``` prose')
    assert rollout_gate_cli.truncation_evidence('```json\n{"a": 1}\n```\n```json\n{"b"')
    assert not rollout_gate_cli.truncation_evidence(
        '```json\n{"a": 1}\n```\nthen {"cut'
    )


def test_unclosed_design_evidence() -> None:
    assert rollout_gate_cli.has_unclosed_trailing_object('{"a": {"b": 1}')
    assert not rollout_gate_cli.has_unclosed_trailing_object('{"a": 1} closed')
    assert not rollout_gate_cli.has_unclosed_trailing_object("prose with { only")
    assert rollout_gate_cli.has_unclosed_trailing_object(
        '{"note": "a } inside a string", "cut'
    )
    assert not rollout_gate_cli.has_unclosed_trailing_object(
        '{"note": "a { inside a string"} trailing prose'
    )
    assert rollout_gate_cli.has_unclosed_trailing_object(
        '{"quote": "she said \\"hi\\" {"'
    )
    assert not rollout_gate_cli.has_unclosed_trailing_object(
        'He said "opening brace incoming and then never delivered'
    )


def test_non_authoritative_reasons() -> None:
    assert (
        rollout_gate_cli.non_authoritative_reasons(
            "canonical",
            cast(str, MODEL_REVISION),
            FAKE_LOCK_SHA,
            False,
            FAKE_SHA,
            False,
        )
        == []
    )
    smoke = rollout_gate_cli.non_authoritative_reasons(
        "smoke", cast(str, MODEL_REVISION), FAKE_LOCK_SHA, False, FAKE_SHA, False
    )
    assert len(smoke) == 1 and "smoke" in smoke[0]
    wrong_revision = rollout_gate_cli.non_authoritative_reasons(
        "canonical", "deadbeef", FAKE_LOCK_SHA, False, FAKE_SHA, False
    )
    assert len(wrong_revision) == 1 and "pinned revision" in wrong_revision[0]
    no_lock = rollout_gate_cli.non_authoritative_reasons(
        "canonical", cast(str, MODEL_REVISION), None, False, FAKE_SHA, False
    )
    assert len(no_lock) == 1 and "uv.lock" in no_lock[0]
    dirty = rollout_gate_cli.non_authoritative_reasons(
        "canonical", cast(str, MODEL_REVISION), FAKE_LOCK_SHA, True, FAKE_SHA, False
    )
    assert len(dirty) == 1 and "uncommitted" in dirty[0]
    unknown_sha = rollout_gate_cli.non_authoritative_reasons(
        "canonical", cast(str, MODEL_REVISION), FAKE_LOCK_SHA, False, "unknown", False
    )
    assert len(unknown_sha) == 1 and "git sha is unknown" in unknown_sha[0]
    source_changed = rollout_gate_cli.non_authoritative_reasons(
        "canonical", cast(str, MODEL_REVISION), FAKE_LOCK_SHA, False, FAKE_SHA, True
    )
    assert len(source_changed) == 1 and "changed during the run" in source_changed[0]
    all_six = rollout_gate_cli.non_authoritative_reasons(
        "smoke", "deadbeef", None, True, "unknown", True
    )
    assert len(all_six) == 6


def test_gate_verdict_cap_and_variance() -> None:
    verdict = rollout_gate_cli.gate_verdict([], 0.5, 0.0, 1.0, 256, [])
    assert verdict["authoritative"] is True
    assert verdict["completion_cap"] == 256
    assert verdict["rerun_at_raised_cap_required"] is False
    assert verdict["variance_gate_passed"] is True
    raised = rollout_gate_cli.gate_verdict([], 0.5, 0.10, 1.0, 256, [3, 11])
    assert raised["completion_cap"] == 512
    assert raised["rerun_at_raised_cap_required"] is True
    assert raised["variance_gate_passed"] is False
    assert raised["zero_variance_prompt_indices"] == [3, 11]
    at_raised = rollout_gate_cli.gate_verdict([], 0.5, 0.0, 1.0, 512, [])
    assert at_raised["completion_cap"] == 512
    assert at_raised["rerun_at_raised_cap_required"] is False


def test_verdict_threshold_boundaries() -> None:
    at_sft_threshold = rollout_gate_cli.gate_verdict([], 0.10, 0.0, 1.0, 256, [])
    assert at_sft_threshold["sft_recommended"] is False
    below_sft_threshold = rollout_gate_cli.gate_verdict([], 0.099, 0.0, 1.0, 256, [])
    assert below_sft_threshold["sft_recommended"] is True
    at_cap_threshold = rollout_gate_cli.gate_verdict([], 0.5, 0.05, 1.0, 256, [])
    assert at_cap_threshold["completion_cap"] == 256
    above_cap_threshold = rollout_gate_cli.gate_verdict([], 0.5, 0.051, 1.0, 256, [])
    assert above_cap_threshold["completion_cap"] == 512


def test_overall_metrics_formulas() -> None:
    breakdowns = (
        RewardBreakdown(
            score=0.9, rung=3, reason="solved", cost_total_usd=110.0, feasible=1.0
        ),
        RewardBreakdown(
            score=0.5, rung=3, reason="solved", cost_total_usd=150.0, feasible=0.4
        ),
        RewardBreakdown(score=0.10, rung=1, reason="drc: too shallow"),
        RewardBreakdown(score=0.0, rung=0, reason="parse failure: no json"),
    )
    prompt = make_prompt_rollouts(
        breakdowns=breakdowns,
        texts=("a", "b", "c", TRUNCATED_TEXT),
        token_counts=(50, 60, 70, 80),
        truncated=(False, False, False, True),
        entropy_sums=(25.0, 30.0, 35.0, 40.0),
        entropy_token_counts=(50, 60, 70, 80),
    )
    metrics = rollout_gate_cli.overall_metrics([prompt], 2)
    assert metrics["n_completions"] == 4
    assert metrics["parse_rate"] == pytest.approx(0.75)
    assert metrics["drc_pass_rate"] == pytest.approx(0.5)
    assert metrics["rung3_rate"] == pytest.approx(0.5)
    assert metrics["feasibility_rate"] == pytest.approx(0.25)
    assert metrics["feasibility_given_rung3"] == pytest.approx(0.5)
    assert metrics["cost_gap_mean"] == pytest.approx(0.1)
    assert metrics["cost_gap_min"] == pytest.approx(0.1)
    assert metrics["mean_group_reward_std"] == pytest.approx((0.2 + 0.05) / 2)
    assert metrics["group_std_min"] == pytest.approx(0.05)
    assert metrics["zero_variance_group_fraction"] == 0.0
    assert metrics["truncation_rate"] == pytest.approx(0.25)
    assert metrics["cap_hit_structured_parse_failure_rate"] == pytest.approx(0.25)
    assert metrics["cap_hit_parseable_rate"] == 0.0
    assert metrics["cap_hit_unstructured_rate"] == 0.0
    assert metrics["total_new_tokens"] == 260
    assert metrics["tokens_per_second"] == pytest.approx(130.0)


def test_length_stats_percentiles() -> None:
    stats = rollout_gate_cli.length_stats([10, 20, 30, 40])
    assert stats["mean"] == pytest.approx(25.0)
    assert stats["p50"] == pytest.approx(25.0)
    assert stats["p90"] == pytest.approx(37.0)
    with pytest.raises(ValueError):
        rollout_gate_cli.length_stats([])


def test_token_weighted_entropy_vs_completion_mean() -> None:
    breakdowns = (
        RewardBreakdown(score=0.0, rung=0, reason="parse failure: x"),
        RewardBreakdown(score=0.0, rung=0, reason="parse failure: x"),
    )
    prompt = make_prompt_rollouts(
        breakdowns=breakdowns,
        texts=("a", "b"),
        token_counts=(10, 100),
        truncated=(False, False),
        entropy_sums=(20.0, 50.0),
        entropy_token_counts=(10, 100),
    )
    summary = rollout_gate_cli.entropy_summary([prompt])
    weighted = (20.0 + 50.0) / 110
    completion_mean = (2.0 + 0.5) / 2
    assert summary["baseline"] == pytest.approx(weighted)
    diagnostics = cast(dict[str, Any], summary["diagnostics"])
    assert diagnostics["per_completion_mean"] == pytest.approx(completion_mean)
    assert not math.isclose(weighted, completion_mean)
    assert diagnostics["per_prompt_mean"] == [pytest.approx(weighted)]
    assert diagnostics["p10"] <= diagnostics["p50"] <= diagnostics["p90"]
