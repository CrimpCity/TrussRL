"""Backend for trussrl-model-check: the only module that touches torch.

Downloads the pinned base model at a resolved revision, exercises the
tokenizer/generation stack on the local device, and reads the installed
TRL's GRPOConfig so the preflight checks verify real defaults instead of
hearsay. Imported lazily by the CLI so the base install never needs the
training extra.
"""

import dataclasses
import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import trl
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from trussRL.generator import generate_instances
from trussRL.prompts import render_prompt
from trussRL.training.model_ids import EXPECTED_PROMPT_TOKENS_MAX, MODEL_ID
from trussRL.training.preflight import (
    KIND_ACCEPTANCE,
    KIND_DIVISIBILITY,
    TRL_PREFLIGHT_CHECKS,
)

PROMPT_SAMPLE_SEED = 0
PROMPT_SAMPLE_COUNT = 8

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "huggingface-hub",
    "vllm",
)


def resolve_and_download(revision: str | None) -> tuple[str, str]:
    """Resolve the model revision to a commit sha and download it.

    Assumptions:
        1. The snapshot is requested at the resolved sha, never at a
           moving ref, so the downloaded weights always match the sha
           recorded in the manifest.

    Args:
        revision: branch, tag, or sha to resolve; None resolves the
            repository's default branch head

    Returns:
        tuple: (local snapshot path, resolved commit sha)
    """
    api = HfApi()
    info = api.repo_info(MODEL_ID, revision=revision)
    resolved_sha = str(info.sha)
    local_path = snapshot_download(MODEL_ID, revision=resolved_sha)
    return str(local_path), resolved_sha


def package_version(name: str) -> str | None:
    """Read one installed package version without importing the package.

    Args:
        name: the distribution name to look up

    Returns:
        str: the installed version, or None when not installed
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_driver_version() -> str | None:
    """Read the NVIDIA driver version via nvidia-smi when present.

    Args:
        None

    Returns:
        str: the driver version, or None when nvidia-smi is absent or
            fails
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def environment_report() -> dict[str, object]:
    """Snapshot the runtime environment for the manifest.

    Args:
        None

    Returns:
        dict: platform, python, tracked package versions, CUDA/driver
            info, GPU name and memory, and cuda/mps availability flags
    """
    cuda_available = torch.cuda.is_available()
    gpu_name: str | None = None
    gpu_memory_total: int | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        gpu_memory_total = int(properties.total_memory)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "packages": {name: package_version(name) for name in TRACKED_PACKAGES},
        "cuda_version": torch.version.cuda,
        "cuda_driver": nvidia_driver_version(),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_total": gpu_memory_total,
        "mps_available": torch.backends.mps.is_available(),
    }


def lock_file_hash() -> str | None:
    """Hash uv.lock so the manifest pins the exact resolved dependency set.

    Assumptions:
        1. The lock file is found by walking upward from the working
           directory, matching how the CLI is run from the repo root.

    Args:
        None

    Returns:
        str: 64-hex sha256 of uv.lock, or None when no lock file is
            found
    """
    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / "uv.lock"
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return None


def sample_prompts() -> tuple[str, ...]:
    """Render the fixed prompt sample every generation check reuses.

    Args:
        None

    Returns:
        tuple: PROMPT_SAMPLE_COUNT rendered prompts from the seed-0
            instance batch
    """
    instances = generate_instances(PROMPT_SAMPLE_SEED, PROMPT_SAMPLE_COUNT)
    return tuple(render_prompt(instance) for instance in instances)


def check_tokenizer(model_path: str) -> dict[str, object]:
    """Load the tokenizer and record its identity-critical properties.

    Args:
        model_path: local snapshot path of the downloaded model

    Returns:
        dict: status plus vocab size and eos/pad token identities
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return {
        "status": "pass",
        "detail": "tokenizer loaded",
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
    }


def check_prompt_lengths(model_path: str) -> dict[str, object]:
    """Tokenize rendered prompts and enforce the prompt token budget.

    Assumptions:
        1. The seed-0 sample stands in for the training distribution;
           the generator draws spans and loads from bounded grids, so
           prompt length varies only via number formatting and the
           depth-limit line.

    Args:
        model_path: local snapshot path of the downloaded model

    Returns:
        dict: status, per-prompt token counts, and the max/mean against
            EXPECTED_PROMPT_TOKENS_MAX
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    lengths = [len(tokenizer.encode(prompt)) for prompt in sample_prompts()]
    max_length = max(lengths)
    passed = max_length <= EXPECTED_PROMPT_TOKENS_MAX
    return {
        "status": "pass" if passed else "fail",
        "detail": (f"max {max_length} tokens vs budget {EXPECTED_PROMPT_TOKENS_MAX}"),
        "lengths": lengths,
        "max_tokens": max_length,
        "mean_tokens": sum(lengths) / len(lengths),
        "budget_tokens": EXPECTED_PROMPT_TOKENS_MAX,
    }


def run_mps_generation(
    model_path: str, dtype: torch.dtype, max_new_tokens: int
) -> dict[str, object]:
    """Load the model at one dtype on MPS and time a short generation.

    Args:
        model_path: local snapshot path of the downloaded model
        dtype: torch dtype to load the weights in
        max_new_tokens: completion length for the timing run

    Returns:
        dict: dtype used, new token count, elapsed seconds, tokens per
            second, and MPS bytes allocated after generation
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True
    )
    model = model.to("mps")
    model.eval()
    prompt = sample_prompts()[0]
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
    return {
        "status": "pass",
        "detail": f"generated {new_tokens} tokens on mps in {elapsed:.1f}s",
        "device": "mps",
        "dtype": str(dtype).removeprefix("torch."),
        "new_tokens": new_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": new_tokens / elapsed if elapsed > 0 else 0.0,
        "mps_allocated_bytes": int(torch.mps.current_allocated_memory()),
    }


def check_mps_generation(model_path: str, max_new_tokens: int) -> dict[str, object]:
    """Verify local generation on Apple silicon, bf16 first, fp16 fallback.

    Assumptions:
        1. A bf16 failure is recorded, not raised: MPS bf16 coverage has
           gaps, so the check falls back to fp16 and reports the dtype
           that actually ran.

    Args:
        model_path: local snapshot path of the downloaded model
        max_new_tokens: completion length for the timing run

    Returns:
        dict: the successful run's metrics, with any bf16 failure noted
            in the detail
    """
    try:
        return run_mps_generation(model_path, torch.bfloat16, max_new_tokens)
    except Exception as bf16_error:  # noqa: BLE001 - fallback is the contract
        fallback = run_mps_generation(model_path, torch.float16, max_new_tokens)
        fallback["detail"] = (
            f"{fallback['detail']}; bf16 failed and fp16 was used instead "
            f"({type(bf16_error).__name__}: {bf16_error})"
        )
        return fallback


def check_vllm_generation(model_path: str, max_new_tokens: int) -> dict[str, object]:
    """Verify batched vLLM generation on a CUDA device.

    Assumptions:
        1. Written for the CUDA training box; it has not run yet because
           no CUDA device exists locally — first execution happens when
           one is provisioned.

    Args:
        model_path: local snapshot path of the downloaded model
        max_new_tokens: completion length per prompt for the timing run

    Returns:
        dict: device name, batch throughput, and CUDA memory high-water
            mark
    """
    from vllm import LLM, SamplingParams

    prompts = list(sample_prompts())
    llm = LLM(model=model_path, dtype="bfloat16")
    start = time.perf_counter()
    outputs = llm.generate(
        prompts, SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    )
    elapsed = time.perf_counter() - start
    new_tokens = sum(
        len(completion.token_ids) for output in outputs for completion in output.outputs
    )
    return {
        "status": "pass",
        "detail": (
            f"generated {new_tokens} tokens over {len(prompts)} prompts "
            f"in {elapsed:.1f}s"
        ),
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "prompts": len(prompts),
        "new_tokens": new_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": new_tokens / elapsed if elapsed > 0 else 0.0,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def grpo_field_defaults() -> dict[str, object]:
    """Read every GRPOConfig field default from the installed TRL.

    Assumptions:
        1. default_factory values are materialized by calling the
           factory; a factory that raises is recorded as None rather
           than crashing the inspection.

    Args:
        None

    Returns:
        dict: field name to default value for all GRPOConfig fields
    """
    defaults: dict[str, object] = {}
    for field in dataclasses.fields(trl.GRPOConfig):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            try:
                defaults[field.name] = field.default_factory()
            except Exception:  # noqa: BLE001 - inspection must not crash
                defaults[field.name] = None
        else:
            defaults[field.name] = None
    return defaults


def inspect_trl_config() -> dict[str, object]:
    """Observe the installed GRPOConfig for the TRL preflight checks.

    Assumptions:
        1. Every acceptance probe constructs GRPOConfig against a scratch
           output_dir because __post_init__ (via TrainingArguments) has
           filesystem side effects.
        2. A probe exception is data, recorded as the error string —
           never propagated.

    Args:
        None

    Returns:
        dict: trl_version, field_defaults, and acceptance_errors keyed
            by check name (None where the probe was accepted)
    """
    acceptance_errors: dict[str, str | None] = {}
    for check in TRL_PREFLIGHT_CHECKS:
        if check.kind not in (KIND_ACCEPTANCE, KIND_DIVISIBILITY):
            continue
        probe_kwargs: dict[str, Any] = dict(check.probe_kwargs)
        with tempfile.TemporaryDirectory() as scratch:
            try:
                trl.GRPOConfig(output_dir=scratch, **probe_kwargs)
            except Exception as error:  # noqa: BLE001 - the exception is data
                acceptance_errors[check.name] = f"{type(error).__name__}: {error}"
            else:
                acceptance_errors[check.name] = None
    return {
        "trl_version": package_version("trl"),
        "field_defaults": grpo_field_defaults(),
        "acceptance_errors": acceptance_errors,
    }
