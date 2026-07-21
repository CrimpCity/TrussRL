"""Artifact shapes and provenance stamping for calibration outputs.

Every artifact is self-describing — provenance stamp, config echo, and
inline instance parameters — so a frozen constant always traces to the
exact code, catalog, and run that produced it. Kept separate from the CLI
so the shapes are testable and reusable outside it.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from trussRL.calibration.sweep import CalibrationConfig
from trussRL.catalog import list_sections
from trussRL.instance import TrussInstance
from trussRL.reward import RewardBreakdown
from trussRL.schema import TrussDesign

RUN_ID_SHA_LENGTH = 7


def catalog_hash() -> str:
    """Hash the frozen section catalog for provenance stamping.

    Assumptions:
        1. Canonical form is json.dumps with sort_keys over every field of
           every section in catalog order, so any change to a designation,
           property, or cost changes the hash.

    Args: None

    Returns:
        str: 64-hex sha256 of the canonical catalog serialization
    """
    payload = [
        {
            "designation": section.designation,
            "weight_lb_per_ft": section.weight_lb_per_ft,
            "area_in2": section.area_in2,
            "rx_in": section.rx_in,
            "ry_in": section.ry_in,
            "cost_per_ft_usd": section.cost_per_ft_usd,
            "properties": dict(section.properties),
        }
        for section in list_sections()
    ]
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_sha() -> str:
    """Read the current git commit SHA for provenance stamping.

    Assumptions:
        1. Never raises: any failure (no git, not a repository) yields
           "unknown" so calibration can still run, and the CLI warns when
           it sees that value.

    Args: None

    Returns:
        str: the full HEAD commit SHA, or "unknown" on any failure
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def git_dirty() -> bool:
    """Report whether the working tree differs from the stamped commit.

    Assumptions:
        1. Conservative on failure: when git cannot be queried the tree
           is reported dirty, so a stamp never falsely claims a clean
           tree.

    Args: None

    Returns:
        bool: True when the working tree has uncommitted changes or git
            could not be queried
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(result.stdout.strip())


def provenance_stamp() -> dict[str, object]:
    """Build the provenance stamp shared by every artifact of one run.

    The run_id (<YYYYMMDDTHHMMSSZ>_<sha7>) is generated here, once per CLI
    invocation, and stamped into every artifact that invocation writes so
    the frozen constants and their gate report are provably from the same
    run.

    Args: None

    Returns:
        dict: created_utc, git_sha, git_dirty, catalog_sha256, run_id
    """
    created = datetime.now(timezone.utc)
    sha = git_sha()
    run_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}_{sha[:RUN_ID_SHA_LENGTH]}"
    return {
        "created_utc": created.isoformat(),
        "git_sha": sha,
        "git_dirty": git_dirty(),
        "catalog_sha256": catalog_hash(),
        "run_id": run_id,
    }


def instance_payload(instance: TrussInstance) -> dict[str, object]:
    """Serialize a problem instance for inline artifact embedding.

    Args:
        instance: the problem instance to serialize

    Returns:
        dict: every instance field, load cases as a list of dicts
    """
    payload = dataclasses.asdict(instance)
    payload["load_cases"] = list(payload["load_cases"])
    return payload


def design_payload(design: TrussDesign) -> dict[str, object]:
    """Serialize a design for artifact embedding.

    Args:
        design: the schema-validated design to serialize

    Returns:
        dict: the seven schema fields
    """
    return design.model_dump()


def breakdown_payload(breakdown: RewardBreakdown) -> dict[str, object]:
    """Serialize a reward breakdown for artifact embedding.

    Args:
        breakdown: the breakdown to serialize

    Returns:
        dict: score, rung, reason, and the rung-3 detail fields
    """
    return dataclasses.asdict(breakdown)


def config_payload(config: CalibrationConfig) -> dict[str, object]:
    """Serialize a calibration config for the artifact config echo.

    Args:
        config: the run's calibration config

    Returns:
        dict: every config field, the generator config nested
    """
    return dataclasses.asdict(config)


def write_json(path: Path, payload: object) -> None:
    """Atomically write a payload as pretty-printed strict JSON.

    Assumptions:
        1. allow_nan=False so a NaN or infinity anywhere in a payload
           fails the write instead of freezing an unreadable artifact.
        2. Atomic: the payload lands in a same-directory temp file that
           os.replace moves into place, so a crash mid-write can never
           leave a truncated artifact at the target path.

    Args:
        path: destination path; parent directories are created
        payload: JSON-serializable payload

    Returns:
        None

    Raises:
        ValueError: if the payload contains NaN or infinite floats.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text + "\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
