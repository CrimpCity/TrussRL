"""Linux/A100 smoke test for the locked vLLM training stack.

Verifies, in order: the locked NumPy/Numba pair executes JIT code, CUDA is
available, the pinned Qwen checkpoint is already in the Hugging Face cache,
and vLLM can load it offline and generate one token.

Usage (on a CUDA Linux host with the checkpoint already cached):
    uv run --extra training python -m scripts.models.vllm_smoke
"""

import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from numba import njit  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B-Base"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


@njit
def jit_answer() -> int:
    """Return a constant through Numba JIT compilation.

    Args: None

    Returns:
        int: the constant 42, proving the locked NumPy/Numba pair executes
            JIT-compiled code
    """
    return 42


def check_numba_jit() -> None:
    """Compile and run a trivial Numba kernel under the locked NumPy.

    Args: None

    Returns: None

    Raises:
        RuntimeError: if the JIT-compiled function does not return 42
    """
    result = int(jit_answer())
    if result != 42:
        raise RuntimeError(f"Numba JIT smoke returned {result}, expected 42")
    print("Numba JIT smoke: OK (42)")


def require_cuda() -> None:
    """Fail fast when no CUDA device is available.

    Assumptions:
        1. This script targets the Linux/A100 training host; running it
           without CUDA is a configuration error, not a soft fallback.

    Args: None

    Returns: None

    Raises:
        RuntimeError: if torch reports no available CUDA device
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this smoke test requires a CUDA GPU")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def locate_cached_snapshot(model_id: str, revision: str) -> Path:
    """Locate an already-downloaded model snapshot in the Hugging Face cache.

    Assumptions:
        1. The checkpoint was previously downloaded with
           ``scripts.models.download_qwen``; this function never downloads.

    Args:
        model_id: Hugging Face model repository identifier
        revision: full commit SHA of the pinned checkpoint

    Returns:
        Path: revision-addressed snapshot directory in the Hugging Face cache

    Raises:
        huggingface_hub.errors.LocalEntryNotFoundError: if the snapshot is
            not fully present in the local cache
    """
    snapshot_path = snapshot_download(
        repo_id=model_id, revision=revision, local_files_only=True
    )
    print(f"Cached snapshot: {snapshot_path}")
    return Path(snapshot_path)


def generate_one_token(snapshot_path: Path) -> str:
    """Initialize vLLM from a local snapshot and generate exactly one token.

    Assumptions:
        1. Loading from the local snapshot directory keeps vLLM fully offline;
           combined with ``HF_HUB_OFFLINE=1`` no network access occurs.

    Args:
        snapshot_path: local directory containing the model snapshot

    Returns:
        str: the single generated token as text
    """
    llm = LLM(model=str(snapshot_path), dtype="bfloat16")
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(["The capital of France is"], sampling_params)
    return str(outputs[0].outputs[0].text)


def main() -> None:
    """Run the full Linux/A100 vLLM smoke sequence.

    Args: None

    Returns: None
    """
    check_numba_jit()
    require_cuda()
    snapshot_path = locate_cached_snapshot(MODEL_ID, MODEL_REVISION)
    token_text = generate_one_token(snapshot_path)
    print(f"Generated token: {token_text!r}")
    print("vLLM smoke: OK")


if __name__ == "__main__":
    main()
