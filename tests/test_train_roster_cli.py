"""Orchestration tests for the training-roster CLI.

calibrate_instance and check_difficulty_spread are patched with
deterministic fakes — no solver runs — so these tests pin the CLI
contract: exit codes, when the artifact is written, backfill on
calibration errors, and the source pinning. The real 512x5k build is the
statistical acceptance test.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence, cast

import pytest
from typer.testing import CliRunner, Result

import trussRL.cli.train_roster as train_roster_cli
from trussRL.calibration.roster import SpreadCheck
from trussRL.calibration.sweep import DesignSample, InstanceCalibration
from trussRL.generator import GeneratorConfig
from trussRL.instance import TrussInstance
from trussRL.reward import RewardBreakdown
from trussRL.schema import TrussDesign

runner = CliRunner()

JsonDict = dict[str, Any]

FAKE_DESIGN = TrussDesign(
    truss_type="warren",
    n_bays=12,
    depth_ft=8.0,
    top_chord="HSS6X6X3/8",
    bottom_chord="HSS6X6X3/8",
    diagonals="HSS4X4X1/4",
)
FAKE_BREAKDOWN = RewardBreakdown(
    score=0.8,
    rung=3,
    reason="solved",
    u_strength=0.5,
    u_buckling=0.5,
    u_defl=0.5,
    cost_total_usd=12000.0,
    feasible=1.0,
)
FAKE_STAMP: JsonDict = {
    "created_utc": "2026-01-01T00:00:00+00:00",
    "git_sha": "0" * 40,
    "git_dirty": False,
    "catalog_sha256": "0" * 64,
    "run_id": "20260101T000000Z_0000000",
}
PASSING_SPREAD = (
    SpreadCheck(
        gate="spread_span",
        status="pass",
        measurements={"span_min_ft": 120.0},
        criterion="test stand-in",
    ),
)
FAILING_SPREAD = (
    SpreadCheck(
        gate="spread_span",
        status="fail",
        measurements={"span_min_ft": 140.0},
        criterion="test stand-in",
        detail="min 140 > 125",
    ),
)


def instance_payload_dict(span_ft: float) -> JsonDict:
    """Build a serialized fixture instance with a distinctive span.

    Args:
        span_ft: span in feet; fixture spans stay far below the v2
            family's 120-160 ft so they never collide with candidates

    Returns:
        dict: an instance payload accepted by instance_from_payload
    """
    return {
        "span_ft": span_ft,
        "load_cases": [
            {"w_kip_per_ft": -1.0, "level": "bottom", "longitudinal_kip": 0.0}
        ],
        "depth_limit_ft": None,
        "defl_denom": 360,
        "cost_ref_usd": None,
    }


def write_calibration_fixtures(tmp_path: Path, n_instances: int = 4) -> None:
    """Write tiny consistent cost_ref.json and sweep_best.json fixtures.

    Args:
        tmp_path: directory to write into
        n_instances: calibration roster size; 4 puts exactly index 3 in
            the eval split

    Returns: None
    """
    config = {
        "seed": 0,
        "n_instances": n_instances,
        "designs_per_instance": 5,
        "sweep_best_samples": 5,
        "generator": dataclasses.asdict(GeneratorConfig()),
        "generator_version": "v2",
    }
    instances = [instance_payload_dict(10.0 + index) for index in range(n_instances)]
    cost_ref = {
        "stamp": FAKE_STAMP,
        "config": config,
        "roster_seed": 999,
        "instances": [
            {
                "instance_index": index,
                "instance": instances[index],
                "sweep_seed": 1000 + index,
                "n_samples": 5,
                "n_rung3": 5,
                "n_strictly_feasible": 3,
                "cost_ref_usd": 50000.0 + index,
            }
            for index in range(n_instances)
        ],
    }
    sweep_best = {
        "stamp": FAKE_STAMP,
        "config": config,
        "roster_seed": 999,
        "instances": [
            {
                "instance_index": index,
                "instance": instances[index],
                "sweep_best_seed": 2000 + index,
                "n_samples": 5,
                "cost_ref_usd": 50000.0 + index,
                "best": {
                    "design": FAKE_DESIGN.model_dump(),
                    "score": 0.9,
                    "rung": 3,
                    "reason": "solved",
                    "u_strength": 0.5,
                    "u_buckling": 0.5,
                    "u_defl": 0.5,
                    "cost_total_usd": 48000.0,
                    "feasible": 1.0,
                },
            }
            for index in range(n_instances)
        ],
    }
    (tmp_path / "cost_ref.json").write_text(json.dumps(cost_ref))
    (tmp_path / "sweep_best.json").write_text(json.dumps(sweep_best))


def fake_calibrate_instance(
    instance: TrussInstance, seed: int, n_samples: int
) -> InstanceCalibration:
    """Deterministic stand-in for the real per-instance calibration.

    Args:
        instance: the candidate instance handed in by the CLI
        seed: the sweep seed handed in by the CLI
        n_samples: the sample count handed in by the CLI

    Returns:
        InstanceCalibration: a fixed two-sample calibration
    """
    sample = DesignSample(design=FAKE_DESIGN, breakdown=FAKE_BREAKDOWN)
    return InstanceCalibration(
        instance=instance,
        seed=seed,
        n_samples=n_samples,
        samples=(sample, sample),
        cost_ref_usd=10000.0,
        regraded_scores=(0.3, 0.2),
    )


def erroring_calibrate_instance(
    instance: TrussInstance, seed: int, n_samples: int
) -> InstanceCalibration:
    """Stand-in that fails the way a zero-feasible sweep does.

    Args:
        instance: the candidate instance handed in by the CLI
        seed: the sweep seed handed in by the CLI
        n_samples: the sample count handed in by the CLI

    Returns:
        InstanceCalibration: never returns

    Raises:
        ValueError: always.
    """
    raise ValueError("cannot compute cost_ref: no strictly feasible sample")


class FlakyCalibrate:
    """Calibration stand-in that fails a fixed number of leading calls."""

    def __init__(self, n_failures: int) -> None:
        """Set up the failure budget.

        Args:
            n_failures: how many leading calls raise before succeeding

        Returns: None
        """
        self.remaining_failures = n_failures

    def __call__(
        self, instance: TrussInstance, seed: int, n_samples: int
    ) -> InstanceCalibration:
        """Fail while the budget lasts, then defer to the passing fake.

        Args:
            instance: the candidate instance handed in by the CLI
            seed: the sweep seed handed in by the CLI
            n_samples: the sample count handed in by the CLI

        Returns:
            InstanceCalibration: the passing fake's calibration

        Raises:
            ValueError: while the failure budget lasts.
        """
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise ValueError("cannot compute cost_ref: no strictly feasible sample")
        return fake_calibrate_instance(instance, seed, n_samples)


def fake_spread_pass(
    train_instances: Sequence[TrussInstance],
    train_cost_refs: Sequence[float],
    calibration_cost_refs: Sequence[float],
) -> tuple[SpreadCheck, ...]:
    """Spread stand-in where every gate passes.

    Args:
        train_instances: the accepted instances handed in by the CLI
        train_cost_refs: the frozen cost_refs handed in by the CLI
        calibration_cost_refs: the calibration cost_refs handed in by the CLI

    Returns:
        tuple: a single passing check
    """
    return PASSING_SPREAD


def fake_spread_fail(
    train_instances: Sequence[TrussInstance],
    train_cost_refs: Sequence[float],
    calibration_cost_refs: Sequence[float],
) -> tuple[SpreadCheck, ...]:
    """Spread stand-in where a gate fails.

    Args:
        train_instances: the accepted instances handed in by the CLI
        train_cost_refs: the frozen cost_refs handed in by the CLI
        calibration_cost_refs: the calibration cost_refs handed in by the CLI

    Returns:
        tuple: a single failing check
    """
    return FAILING_SPREAD


def invoke(
    tmp_path: Path,
    n_target: int = 3,
    n_candidates: int = 6,
    designs_per_instance: int = 5,
) -> Result:
    """Invoke the CLI against a temp artifacts dir with tiny counts.

    Args:
        tmp_path: temp directory used as --artifacts-dir
        n_target: instances to freeze
        n_candidates: candidates to draw
        designs_per_instance: sample count per instance

    Returns:
        Result: the Click test result
    """
    return runner.invoke(
        train_roster_cli.app,
        [
            "--seed",
            "1",
            "--n-target",
            str(n_target),
            "--n-candidates",
            str(n_candidates),
            "--designs-per-instance",
            str(designs_per_instance),
            "--artifacts-dir",
            str(tmp_path),
        ],
    )


def test_run_writes_artifact_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_calibration_fixtures(tmp_path)
    monkeypatch.setattr(train_roster_cli, "calibrate_instance", fake_calibrate_instance)
    monkeypatch.setattr(train_roster_cli, "check_difficulty_spread", fake_spread_pass)
    result = invoke(tmp_path)
    assert result.exit_code == 0, result.output
    payload = cast(JsonDict, json.loads((tmp_path / "train_roster.json").read_text()))
    assert set(payload["stamp"]) == {
        "created_utc",
        "git_sha",
        "git_dirty",
        "catalog_sha256",
        "run_id",
    }
    assert payload["config"]["seed"] == 1
    assert payload["config"]["n_target"] == 3
    assert payload["config"]["n_candidates"] == 6
    assert payload["config"]["designs_per_instance"] == 5
    assert payload["config"]["generator_version"] == "v2"
    assert payload["config"]["generator"]["span_min_ft"] == 120.0
    assert payload["roster_seed"] == 10499958131665514997
    assert payload["held_out"] == {"indices": [3]}
    assert payload["spread"]["all_passed"] is True
    assert payload["spread"]["checks"][0]["gate"] == "spread_span"
    for key in ("cost_ref", "sweep_best"):
        source = payload["sources"][key]
        raw = (tmp_path / source["filename"]).read_bytes()
        assert source["sha256"] == hashlib.sha256(raw).hexdigest()
        assert source["run_id"] == FAKE_STAMP["run_id"]
    assert len(payload["instances"]) == 3
    for position, entry in enumerate(payload["instances"]):
        assert entry["instance_index"] == position
        assert entry["cost_ref_usd"] == 10000.0
        assert entry["n_samples"] == 5
        assert entry["instance"]["cost_ref_usd"] is None
        assert isinstance(entry["instance_seed"], int)
        assert isinstance(entry["sweep_seed"], int)


def test_run_backfills_to_exact_target_on_calibration_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_calibration_fixtures(tmp_path)
    monkeypatch.setattr(
        train_roster_cli, "calibrate_instance", FlakyCalibrate(n_failures=2)
    )
    monkeypatch.setattr(train_roster_cli, "check_difficulty_spread", fake_spread_pass)
    result = invoke(tmp_path)
    assert result.exit_code == 0, result.output
    payload = cast(JsonDict, json.loads((tmp_path / "train_roster.json").read_text()))
    assert len(payload["instances"]) == 3
    errors = [
        item
        for item in payload["dropped"]
        if item["reason"].startswith("calibration_error:")
    ]
    assert len(errors) == 2
    for item in errors:
        assert set(item) == {"generator_index", "instance_seed", "reason", "instance"}


def test_stream_exhaustion_exits_one_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_calibration_fixtures(tmp_path)
    monkeypatch.setattr(
        train_roster_cli, "calibrate_instance", erroring_calibrate_instance
    )
    monkeypatch.setattr(train_roster_cli, "check_difficulty_spread", fake_spread_pass)
    result = invoke(tmp_path)
    assert result.exit_code == 1
    assert "exhausted" in result.output
    assert not (tmp_path / "train_roster.json").exists()


def test_spread_failure_exits_one_and_preserves_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_calibration_fixtures(tmp_path)
    preexisting = '{"frozen": "do not touch"}'
    (tmp_path / "train_roster.json").write_text(preexisting)
    monkeypatch.setattr(train_roster_cli, "calibrate_instance", fake_calibrate_instance)
    monkeypatch.setattr(train_roster_cli, "check_difficulty_spread", fake_spread_fail)
    result = invoke(tmp_path)
    assert result.exit_code == 1
    assert (tmp_path / "train_roster.json").read_text() == preexisting


def test_invalid_counts_rejected(tmp_path: Path) -> None:
    result = invoke(tmp_path, n_target=6, n_candidates=3)
    assert result.exit_code == 1
    assert "n_candidates must be >= n_target" in result.output
    result = invoke(tmp_path, n_target=0)
    assert result.exit_code == 1
    result = invoke(tmp_path, designs_per_instance=0)
    assert result.exit_code == 1
    assert not (tmp_path / "train_roster.json").exists()


def test_missing_calibration_artifacts_fail_loudly(tmp_path: Path) -> None:
    result = invoke(tmp_path)
    assert result.exit_code == 1
    assert "Calibration artifacts unusable" in result.output
    assert not (tmp_path / "train_roster.json").exists()
