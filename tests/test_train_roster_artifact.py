"""Acceptance tests against the frozen train_roster.json artifact.

These prove the frozen roster's contract: shape and provenance, full
disjointness from the 32 calibrated instances, disjoint seed domains, and
a preserved difficulty spread. They fail — not skip — when the artifact
is absent: the artifact and these tests enter the repository together.
"""

import hashlib
import json
import math
from pathlib import Path

from trussRL.calibration.artifacts import instance_from_payload
from trussRL.calibration.roster import (TRAIN_ROSTER_SIZE,
                                        check_difficulty_spread,
                                        derive_generator_child_seeds,
                                        instance_key, load_train_roster,
                                        spread_check_payload)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def train_roster_payload_from_disk() -> dict:
    """Read the committed train roster artifact.

    Args: None

    Returns:
        dict: the deserialized artifact payload
    """
    return json.loads((ARTIFACTS_DIR / "train_roster.json").read_text())


def calibration_payloads_from_disk() -> tuple[dict, dict]:
    """Read the committed calibration artifacts.

    Args: None

    Returns:
        tuple: the cost_ref payload and the sweep_best payload
    """
    cost_ref = json.loads((ARTIFACTS_DIR / "cost_ref.json").read_text())
    sweep_best = json.loads((ARTIFACTS_DIR / "sweep_best.json").read_text())
    return cost_ref, sweep_best


def test_train_roster_exists_with_expected_shape() -> None:
    payload = train_roster_payload_from_disk()
    assert set(payload) == {
        "stamp",
        "sources",
        "config",
        "roster_seed",
        "held_out",
        "dropped",
        "spread",
        "instances",
    }
    assert set(payload["stamp"]) == {
        "created_utc",
        "git_sha",
        "git_dirty",
        "catalog_sha256",
        "run_id",
    }
    assert payload["config"]["n_target"] == TRAIN_ROSTER_SIZE
    assert payload["config"]["generator_version"] == "v2"
    assert payload["held_out"]["indices"] == [3, 7, 11, 15, 19, 23, 27, 31]
    assert len(payload["instances"]) == TRAIN_ROSTER_SIZE
    cost_ref, sweep_best = calibration_payloads_from_disk()
    for key, filename, source_artifact in (
        ("cost_ref", "cost_ref.json", cost_ref),
        ("sweep_best", "sweep_best.json", sweep_best),
    ):
        source = payload["sources"][key]
        assert source["filename"] == filename
        raw = (ARTIFACTS_DIR / filename).read_bytes()
        assert source["sha256"] == hashlib.sha256(raw).hexdigest()
        assert source["run_id"] == source_artifact["stamp"]["run_id"]
    for position, entry in enumerate(payload["instances"]):
        assert entry["instance_index"] == position
        instance = instance_from_payload(entry["instance"])
        assert instance.cost_ref_usd is None
        assert entry["n_samples"] == payload["config"]["designs_per_instance"]


def test_no_calibrated_instance_appears_in_training() -> None:
    payload = train_roster_payload_from_disk()
    cost_ref, _ = calibration_payloads_from_disk()
    train_keys = [
        instance_key(instance_from_payload(entry["instance"]))
        for entry in payload["instances"]
    ]
    calibration_keys = {
        instance_key(instance_from_payload(entry["instance"]))
        for entry in cost_ref["instances"]
    }
    assert len(calibration_keys) == 32
    assert len(set(train_keys)) == len(train_keys)
    assert set(train_keys) & calibration_keys == set()


def test_seed_domains_are_disjoint() -> None:
    payload = train_roster_payload_from_disk()
    cost_ref, sweep_best = calibration_payloads_from_disk()
    assert payload["roster_seed"] != cost_ref["roster_seed"]
    train_instance_seeds = [entry["instance_seed"] for entry in payload["instances"]]
    train_sweep_seeds = [entry["sweep_seed"] for entry in payload["instances"]]
    assert len(set(train_instance_seeds)) == len(train_instance_seeds)
    assert len(set(train_sweep_seeds)) == len(train_sweep_seeds)
    calibration_generator_seeds = derive_generator_child_seeds(
        cost_ref["roster_seed"], len(cost_ref["instances"])
    )
    calibration_seeds = (
        set(calibration_generator_seeds)
        | {entry["sweep_seed"] for entry in cost_ref["instances"]}
        | {entry["sweep_best_seed"] for entry in sweep_best["instances"]}
    )
    train_seeds = set(train_instance_seeds) | set(train_sweep_seeds)
    assert train_seeds & calibration_seeds == set()


def test_train_roster_preserves_difficulty_spread() -> None:
    payload = train_roster_payload_from_disk()
    cost_ref, _ = calibration_payloads_from_disk()
    train_instances = [
        instance_from_payload(entry["instance"]) for entry in payload["instances"]
    ]
    train_cost_refs = [entry["cost_ref_usd"] for entry in payload["instances"]]
    calibration_cost_refs = [
        entry["cost_ref_usd"] for entry in cost_ref["instances"]
    ]
    checks = check_difficulty_spread(
        train_instances, train_cost_refs, calibration_cost_refs
    )
    for check in checks:
        assert check.status == "pass", f"{check.gate}: {check.detail}"
    assert payload["spread"]["all_passed"] is True
    assert payload["spread"]["checks"] == [
        spread_check_payload(check) for check in checks
    ]


def test_load_train_roster_applies_cost_ref() -> None:
    roster = load_train_roster(ARTIFACTS_DIR)
    assert len(roster) == TRAIN_ROSTER_SIZE
    for instance in roster:
        assert instance.cost_ref_usd is not None
        assert math.isfinite(instance.cost_ref_usd)
        assert instance.cost_ref_usd > 0.0
