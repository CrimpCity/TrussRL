"""Download the pinned Qwen base checkpoint to the Hugging Face cache.

The requested revision is resolved to a full commit SHA before downloading,
so the printed model ID and revision form a reproducible checkpoint identity.
No repository-local model directory is created.

Usage:
    uv run --extra training python -m scripts.models.download_qwen
    uv run --extra training python -m scripts.models.download_qwen \
        --revision <full-commit-sha>
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL_ID = "Qwen/Qwen3-4B-Base"
DEFAULT_REVISION = "main"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the checkpoint download.

    Args: None

    Returns:
        argparse.Namespace: parsed revision requested by the user
    """
    parser = argparse.ArgumentParser(
        description=(
            f"Download {MODEL_ID} to the revision-addressed Hugging Face cache."
        )
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="branch, tag, or full commit SHA to resolve (default: main)",
    )
    return parser.parse_args()


def resolve_revision(model_id: str, revision: str) -> str:
    """Resolve a model revision to its immutable full commit SHA.

    Args:
        model_id: Hugging Face model repository identifier
        revision: branch, tag, or commit SHA requested by the user

    Returns:
        str: full commit SHA reported by the Hugging Face Hub

    Raises:
        RuntimeError: if the Hub response does not contain a commit SHA
    """
    resolved_revision: str | None = HfApi().repo_info(model_id, revision=revision).sha
    if resolved_revision is None:
        raise RuntimeError(
            f"Hugging Face did not return a commit SHA for {model_id}@{revision}"
        )
    return resolved_revision


def download_snapshot(model_id: str, resolved_revision: str) -> Path:
    """Download one exact model revision to the default Hugging Face cache.

    Assumptions:
        1. The caller has already resolved the revision to a full commit SHA.
        2. The Hugging Face default cache location is intentional; this function
           does not set ``local_dir`` or ``cache_dir``.

    Args:
        model_id: Hugging Face model repository identifier
        resolved_revision: immutable full commit SHA to download

    Returns:
        Path: revision-addressed snapshot directory in the Hugging Face cache
    """
    return Path(snapshot_download(repo_id=model_id, revision=resolved_revision))


def main() -> None:
    """Resolve and download the requested Qwen checkpoint revision.

    Args: None

    Returns: None
    """
    args = parse_args()
    requested_revision = str(args.revision)

    print(f"Resolving {MODEL_ID}@{requested_revision}...")
    resolved_revision = resolve_revision(MODEL_ID, requested_revision)
    print(f"Resolved revision: {resolved_revision}")
    print("Downloading checkpoint to the Hugging Face cache...")

    snapshot_path = download_snapshot(MODEL_ID, resolved_revision)

    print()
    print(f"Model: {MODEL_ID}")
    print(f"Requested revision: {requested_revision}")
    print(f"Resolved revision: {resolved_revision}")
    print(f"Snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
