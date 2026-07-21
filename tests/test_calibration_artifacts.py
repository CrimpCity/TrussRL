"""Tests for calibration artifact shapes, provenance, and JSON writing."""

import json
import re
from pathlib import Path

import pytest

from trussRL.calibration.artifacts import (
    breakdown_payload,
    catalog_hash,
    config_payload,
    design_payload,
    instance_payload,
    provenance_stamp,
    write_json,
)
from trussRL.calibration.sweep import CalibrationConfig
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.reward import RewardBreakdown
from trussRL.schema import TrussDesign


def test_catalog_hash_is_deterministic_64_hex() -> None:
    first = catalog_hash()
    assert first == catalog_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_provenance_stamp_keys_and_run_id_shape() -> None:
    stamp = provenance_stamp()
    assert set(stamp) == {
        "created_utc",
        "git_sha",
        "git_dirty",
        "catalog_sha256",
        "run_id",
    }
    assert isinstance(stamp["git_dirty"], bool)
    assert stamp["catalog_sha256"] == catalog_hash()
    run_id = stamp["run_id"]
    assert isinstance(run_id, str)
    assert re.fullmatch(r"\d{8}T\d{6}Z_.{7}", run_id)


def test_instance_payload_round_trips_fields() -> None:
    instance = TrussInstance(
        span_ft=90.0,
        load_cases=(LoadCase(w_kip_per_ft=-1.5, level="bottom"),),
        depth_limit_ft=12.5,
        defl_denom=240,
        cost_ref_usd=None,
    )
    payload = instance_payload(instance)
    assert payload["span_ft"] == 90.0
    assert payload["depth_limit_ft"] == 12.5
    assert payload["defl_denom"] == 240
    assert payload["cost_ref_usd"] is None
    assert payload["load_cases"] == [
        {"w_kip_per_ft": -1.5, "level": "bottom", "longitudinal_kip": 0.0}
    ]


def test_design_and_breakdown_and_config_payload_shapes() -> None:
    design = TrussDesign(
        truss_type="warren",
        n_bays=10,
        depth_ft=9.0,
        top_chord="HSS6X6X3/8",
        bottom_chord="HSS6X6X3/8",
        diagonals="HSS4X4X1/4",
    )
    assert design_payload(design)["n_bays"] == 10
    assert design_payload(design)["verticals"] is None
    breakdown = RewardBreakdown(score=0.42, rung=3, reason="solved", feasible=1.0)
    assert breakdown_payload(breakdown)["score"] == 0.42
    assert breakdown_payload(breakdown)["u_strength"] is None
    config = config_payload(CalibrationConfig())
    assert config["seed"] == 0
    assert config["n_instances"] == 32
    assert config["designs_per_instance"] == 5000
    assert config["sweep_best_samples"] == 100000
    assert isinstance(config["generator"], dict)


def test_write_json_round_trip_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "artifact.json"
    payload = {"a": 1, "b": [1.5, None, "x"]}
    write_json(target, payload)
    assert json.loads(target.read_text()) == payload


def test_write_json_atomic_replacement(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    write_json(target, {"version": 1})
    write_json(target, {"version": 2})
    assert json.loads(target.read_text()) == {"version": 2}
    assert list(tmp_path.iterdir()) == [target]


def test_write_json_rejects_nan_and_leaves_no_file(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    with pytest.raises(ValueError):
        write_json(target, {"bad": float("nan")})
    assert not target.exists()
    write_json(target, {"good": 1.0})
    with pytest.raises(ValueError):
        write_json(target, {"bad": float("inf")})
    assert json.loads(target.read_text()) == {"good": 1.0}
