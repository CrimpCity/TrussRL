"""Validate the model/tokenizer/generation stack and write a run manifest.

Resolves and downloads the pinned base checkpoint, exercises tokenizer,
prompt-length, and short-generation checks on the local device, runs the
TRL preflight checks against the installed GRPOConfig, and freezes it
all into a stamped manifest. This module must import without the training
extra installed — every heavy import lives behind load_backend().
"""

import dataclasses
import types
from pathlib import Path
from typing import Any, Callable

import typer
from rich.console import Console
from rich.table import Table

from trussRL.calibration.artifacts import provenance_stamp, write_json
from trussRL.training.model_ids import MODEL_ID, MODEL_REVISION
from trussRL.training.preflight import (
    STATUS_DEFERRED,
    TrlPreflightResult,
    evaluate_trl_preflight,
    preflight_passed,
)

app = typer.Typer(add_completion=False)
console = Console()

MANIFEST_DIR = "model_check"


def load_backend() -> types.ModuleType:
    """Import the heavy model-check backend, failing with a clear remedy.

    Assumptions:
        1. Kept as a module-level function so tests monkeypatch it with a
           stub namespace instead of installing torch.

    Args:
        None

    Returns:
        ModuleType: the trussRL.training.model_check module

    Raises:
        ImportError: when the training extra is not installed.
    """
    try:
        import trussRL.training.model_check as backend
    except ImportError as error:
        raise ImportError(
            "training dependencies are not installed; "
            "install with: uv sync --extra training"
        ) from error
    return backend


def run_backend_check(
    name: str, check: Callable[..., dict[str, object]], *args: object
) -> dict[str, object]:
    """Run one backend check, converting any exception into a failure.

    Args:
        name: check name recorded in the manifest
        check: backend function returning a status/detail/metrics dict
        *args: positional arguments forwarded to the check

    Returns:
        dict: the check's payload with name, status, and detail always
            present; an exception becomes status fail with the error as
            detail
    """
    try:
        result = check(*args)
    except Exception as error:  # noqa: BLE001 - a crashed check is a failed check
        return {
            "name": name,
            "status": "fail",
            "detail": f"{type(error).__name__}: {error}",
        }
    payload: dict[str, object] = {"name": name}
    payload.update(result)
    payload.setdefault("status", "pass")
    payload.setdefault("detail", "")
    return payload


def skipped_check(name: str, detail: str) -> dict[str, object]:
    """Build a manifest entry for a check that never ran.

    Args:
        name: check name recorded in the manifest
        detail: why the check was skipped

    Returns:
        dict: a status=skipped check payload
    """
    return {"name": name, "status": "skipped", "detail": detail}


def generation_checks(
    backend: Any,
    environment: dict[str, object],
    device: str,
    local_path: str,
    max_new_tokens: int,
) -> list[dict[str, object]]:
    """Run the generation check on the selected device.

    Assumptions:
        1. CPU-only is a hard failure: a box with neither CUDA nor MPS
           cannot validate the training runtime, so the check fails
           instead of silently degrading.
        2. The vLLM path is always represented: run on CUDA, deferred
           otherwise, so the manifest states explicitly what remains
           unverified.

    Args:
        backend: the loaded model-check backend module
        environment: the backend's environment report
        device: auto, cuda, or mps
        local_path: local snapshot path of the downloaded model
        max_new_tokens: completion length for the timing run

    Returns:
        list: one or two check payloads covering local generation and
            the vLLM path
    """
    cuda_available = bool(environment.get("cuda_available"))
    mps_available = bool(environment.get("mps_available"))
    use_cuda = device == "cuda" or (device == "auto" and cuda_available)
    use_mps = device == "mps" or (device == "auto" and not cuda_available)
    if use_cuda:
        return [
            run_backend_check(
                "vllm_generation",
                backend.check_vllm_generation,
                local_path,
                max_new_tokens,
            )
        ]
    deferred_vllm: dict[str, object] = {
        "name": "vllm_generation",
        "status": "deferred",
        "detail": "requires Linux + CUDA; deferred until a training box is provisioned",
    }
    if use_mps and mps_available:
        return [
            run_backend_check(
                "mps_generation",
                backend.check_mps_generation,
                local_path,
                max_new_tokens,
            ),
            deferred_vllm,
        ]
    return [
        {
            "name": "generation",
            "status": "fail",
            "detail": "no CUDA or MPS device available; CPU-only run refused",
        },
        deferred_vllm,
    ]


def trl_preflight_payload(config_info: dict[str, Any]) -> dict[str, object]:
    """Evaluate the TRL preflight checks and shape them for the manifest.

    Args:
        config_info: the backend's inspect_trl_config() result with
            trl_version, field_defaults, and acceptance_errors

    Returns:
        dict: trl_version, serialized TrlPreflightResult items,
            static_complete, and the deferred check names
    """
    results = evaluate_trl_preflight(
        config_info["field_defaults"], config_info["acceptance_errors"]
    )
    return {
        "trl_version": config_info["trl_version"],
        "items": [dataclasses.asdict(result) for result in results],
        "static_complete": preflight_passed(results),
        "deferred": [
            result.name for result in results if result.status == STATUS_DEFERRED
        ],
    }


def preflight_results_from_payload(
    payload: dict[str, object],
) -> tuple[TrlPreflightResult, ...]:
    """Rehydrate TrlPreflightResult rows from the manifest-shaped payload.

    Args:
        payload: the trl_preflight_payload() result

    Returns:
        tuple: one TrlPreflightResult per serialized item
    """
    items = payload["items"]
    assert isinstance(items, list)
    return tuple(TrlPreflightResult(**item) for item in items)


def print_check_table(checks: list[dict[str, object]]) -> None:
    """Print the runtime check summary table.

    Args:
        checks: the manifest check payloads

    Returns: None
    """
    table = Table(title="Model checks", title_justify="left")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for check in checks:
        status = str(check["status"])
        color = {"pass": "green", "fail": "red", "deferred": "yellow"}.get(
            status, "dim"
        )
        table.add_row(
            str(check["name"]), f"[{color}]{status}[/{color}]", str(check["detail"])
        )
    console.print(table)


def print_preflight_table(results: tuple[TrlPreflightResult, ...]) -> None:
    """Print the TRL preflight verdicts.

    Args:
        results: the evaluated preflight rows

    Returns: None
    """
    table = Table(title="TRL preflight checks", title_justify="left")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for result in results:
        color = {
            "verified": "green",
            "recorded": "cyan",
            "deferred": "yellow",
            "mismatch": "red",
            "missing": "red",
        }.get(result.status, "dim")
        table.add_row(result.name, f"[{color}]{result.status}[/{color}]", result.detail)
    console.print(table)


@app.command()
def run(
    revision: str | None = typer.Option(
        None,
        help="Model revision to resolve; defaults to the pinned MODEL_REVISION "
        "or the repository head when unpinned.",
    ),
    device: str = typer.Option("auto", help="Generation device: auto, cuda, or mps."),
    max_new_tokens: int = typer.Option(
        32, help="Completion length for the generation timing check."
    ),
    output_dir: Path = typer.Option(
        Path("artifacts"), help="Directory the manifest is written into."
    ),
) -> None:
    """Validate the reproducible model runtime and write the run manifest.

    Downloads the base checkpoint at a resolved commit sha, checks
    tokenizer, prompt lengths, and short generation on the local device,
    runs the TRL preflight checks against the installed GRPOConfig, and
    writes artifacts/model_check/model_check_<run_id>.json.

    Args:
        revision: model revision override; None uses the pinned revision
        device: generation device selection
        max_new_tokens: completion length for the generation check
        output_dir: directory the manifest is written into

    Returns: None

    Raises:
        typer.Exit: with code 1 when the training extra is missing, any
            check fails, or the TRL preflight has missing/mismatch
            results.
    """
    stamp = provenance_stamp()
    console.print(
        f"[bold]Model check[/bold]  run_id {stamp['run_id']}, model {MODEL_ID}"
    )
    try:
        backend = load_backend()
    except ImportError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    environment = backend.environment_report()
    lock_sha256 = backend.lock_file_hash()
    if lock_sha256 is None:
        console.print(
            "[yellow]Warning:[/yellow] uv.lock not found; the manifest "
            "cannot pin the resolved dependency set."
        )

    checks: list[dict[str, object]] = []
    target_revision = revision if revision is not None else MODEL_REVISION
    local_path: str | None = None
    resolved_revision: str | None = None
    try:
        local_path, resolved_revision = backend.resolve_and_download(target_revision)
    except Exception as error:  # noqa: BLE001 - recorded, manifest still written
        checks.append(
            {
                "name": "download",
                "status": "fail",
                "detail": f"{type(error).__name__}: {error}",
            }
        )
    else:
        checks.append(
            {
                "name": "download",
                "status": "pass",
                "detail": f"snapshot at {resolved_revision}",
            }
        )

    if local_path is None:
        checks.append(skipped_check("tokenizer", "model download failed"))
        checks.append(skipped_check("prompt_lengths", "model download failed"))
        checks.append(skipped_check("generation", "model download failed"))
    else:
        checks.append(
            run_backend_check("tokenizer", backend.check_tokenizer, local_path)
        )
        checks.append(
            run_backend_check(
                "prompt_lengths", backend.check_prompt_lengths, local_path
            )
        )
        checks.extend(
            generation_checks(backend, environment, device, local_path, max_new_tokens)
        )

    preflight = trl_preflight_payload(backend.inspect_trl_config())
    preflight_results = preflight_results_from_payload(preflight)
    static_complete = bool(preflight["static_complete"])
    passed = static_complete and all(check["status"] != "fail" for check in checks)

    manifest = {
        "stamp": stamp,
        "lock_sha256": lock_sha256,
        "model": {
            "model_id": MODEL_ID,
            "pinned_revision": MODEL_REVISION,
            "resolved_revision": resolved_revision,
            "revision_match": (
                MODEL_REVISION is not None and resolved_revision == MODEL_REVISION
            ),
            "local_path": local_path,
        },
        "environment": environment,
        "checks": checks,
        "trl_preflight": preflight,
        "passed": passed,
    }
    manifest_path = output_dir / MANIFEST_DIR / f"model_check_{stamp['run_id']}.json"
    write_json(manifest_path, manifest)

    print_check_table(checks)
    print_preflight_table(preflight_results)
    console.print(f"Manifest written to {manifest_path}")
    deferred_names = [
        result.name for result in preflight_results if result.status == STATUS_DEFERRED
    ]
    if passed:
        console.print(
            "[green]Model check passed[/green] — static preflight checks complete"
            + (f", deferred: {', '.join(deferred_names)}" if deferred_names else "")
        )
        return
    console.print("[red]Model check failed[/red] — see the manifest for details.")
    raise typer.Exit(1)


def main() -> None:
    """Run the model-check command-line interface.

    Args: None

    Returns: None
    """
    app()


if __name__ == "__main__":
    main()
