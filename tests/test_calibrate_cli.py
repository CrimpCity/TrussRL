"""Orchestration tests for the calibration CLI.

calibrate_instance, evaluate_gates, and find_sweep_best are patched with
deterministic fakes — no seed-fishing, no solver runs — so these tests pin
the CLI contract: exit codes, which artifacts get written when, and the
shared run_id. The real 32x5k run is the statistical acceptance test.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

import trussRL.cli.calibrate as calibrate_cli
from trussRL.calibration.gates import GateCheck, GateReport
from trussRL.calibration.sweep import (
    DesignSample,
    InstanceCalibration,
    SweepBest,
)
from trussRL.generator import GeneratorConfig, generate_instances
from trussRL.instance import TrussInstance
from trussRL.reward import RewardBreakdown
from trussRL.schema import TrussDesign

runner = CliRunner()

GENERATOR_CALLS: list[str] = []

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


def fake_calibrate_instance(
    instance: TrussInstance, seed: int, n_samples: int
) -> InstanceCalibration:
    """Deterministic stand-in for the real per-instance calibration.

    Args:
        instance: the roster instance handed in by the CLI
        seed: the calibration sweep seed handed in by the CLI
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
        instance: the roster instance handed in by the CLI
        seed: the calibration sweep seed handed in by the CLI
        n_samples: the sample count handed in by the CLI

    Returns:
        InstanceCalibration: never returns

    Raises:
        ValueError: always.
    """
    raise ValueError("cannot compute cost_ref: no strictly feasible sample")


def make_report(instance_index: int, passed: bool) -> GateReport:
    """Build a one-check gate report with the requested outcome.

    Args:
        instance_index: roster position recorded in the report
        passed: whether the single check passes

    Returns:
        GateReport: a minimal pass or fail report
    """
    return GateReport(
        instance_index=instance_index,
        checks=(
            GateCheck(
                gate="saturation",
                status="pass" if passed else "fail",
                value=0.3,
                criterion="mean regraded score <= 0.35",
            ),
        ),
    )


def fake_evaluate_gates_pass(
    calibration: InstanceCalibration, instance_index: int
) -> GateReport:
    """Gate evaluation stand-in where every instance passes.

    Args:
        calibration: the calibration handed in by the CLI
        instance_index: roster position handed in by the CLI

    Returns:
        GateReport: a passing report
    """
    return make_report(instance_index, passed=True)


def fake_evaluate_gates_fail_index_1(
    calibration: InstanceCalibration, instance_index: int
) -> GateReport:
    """Gate evaluation stand-in where instance 1 fails.

    Args:
        calibration: the calibration handed in by the CLI
        instance_index: roster position handed in by the CLI

    Returns:
        GateReport: failing for instance 1, passing otherwise
    """
    return make_report(instance_index, passed=instance_index != 1)


def fake_find_sweep_best(
    instance: TrussInstance, cost_ref_usd: float, seed: int, n_samples: int
) -> SweepBest:
    """Deterministic stand-in for the sweep_best search.

    Args:
        instance: the roster instance handed in by the CLI
        cost_ref_usd: the cost reference handed in by the CLI
        seed: the sweep_best seed handed in by the CLI
        n_samples: the sample count handed in by the CLI

    Returns:
        SweepBest: a fixed best design
    """
    return SweepBest(
        seed=seed,
        n_samples=n_samples,
        cost_ref_usd=cost_ref_usd,
        design=FAKE_DESIGN,
        breakdown=FAKE_BREAKDOWN,
    )


def recording_generate_instances(
    seed: int, count: int, config: GeneratorConfig = GeneratorConfig()
) -> tuple[TrussInstance, ...]:
    """Canonical roster generation that records it was called.

    Args:
        seed: the roster seed handed in by the CLI
        count: the roster size handed in by the CLI
        config: the generator config handed in by the CLI

    Returns:
        tuple: the real canonical roster
    """
    GENERATOR_CALLS.append("canonical")
    return generate_instances(seed, count, config)


def invoke(tmp_path: Path, gates_only: bool = False) -> Result:
    """Invoke the CLI against a temp output dir with a tiny roster.

    Args:
        tmp_path: temp directory used as --output-dir
        gates_only: whether to pass --gates-only

    Returns:
        Result: the Click test result
    """
    args = [
        "--seed",
        "0",
        "--n-instances",
        "2",
        "--designs-per-instance",
        "10",
        "--sweep-samples",
        "10",
        "--output-dir",
        str(tmp_path),
    ]
    if gates_only:
        args.append("--gates-only")
    return runner.invoke(calibrate_cli.app, args)


def gate_report_files(tmp_path: Path) -> list[Path]:
    """List the gate report files written under the output dir.

    Args:
        tmp_path: the --output-dir used in the invocation

    Returns:
        list: gate report paths, sorted by name
    """
    report_dir = tmp_path / "gate_reports"
    if not report_dir.exists():
        return []
    return sorted(report_dir.glob("calibration_*.json"))


def test_all_pass_full_run_writes_three_artifacts_with_shared_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibrate_cli, "calibrate_instance", fake_calibrate_instance)
    monkeypatch.setattr(calibrate_cli, "evaluate_gates", fake_evaluate_gates_pass)
    monkeypatch.setattr(calibrate_cli, "find_sweep_best", fake_find_sweep_best)
    result = invoke(tmp_path)
    assert result.exit_code == 0, result.output
    reports = gate_report_files(tmp_path)
    assert len(reports) == 1
    cost_ref = json.loads((tmp_path / "cost_ref.json").read_text())
    sweep_best = json.loads((tmp_path / "sweep_best.json").read_text())
    gate_report = json.loads(reports[0].read_text())
    run_ids = {
        payload["stamp"]["run_id"] for payload in (cost_ref, sweep_best, gate_report)
    }
    assert len(run_ids) == 1
    assert reports[0].name == f"calibration_{run_ids.pop()}.json"
    for payload in (cost_ref, sweep_best, gate_report):
        assert set(payload["stamp"]) == {
            "created_utc",
            "git_sha",
            "git_dirty",
            "catalog_sha256",
            "run_id",
        }
        assert payload["config"]["n_instances"] == 2
        assert payload["config"]["generator_version"] == "v2"
        assert len(payload["instances"]) == 2
    assert gate_report["all_passed"] is True
    assert cost_ref["instances"][0]["cost_ref_usd"] == 10000.0
    assert sweep_best["instances"][0]["best"]["cost_total_usd"] == 12000.0
    assert sweep_best["instances"][0]["best"]["design"] == FAKE_DESIGN.model_dump()


def test_all_pass_gates_only_writes_only_gate_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibrate_cli, "calibrate_instance", fake_calibrate_instance)
    monkeypatch.setattr(calibrate_cli, "evaluate_gates", fake_evaluate_gates_pass)
    result = invoke(tmp_path, gates_only=True)
    assert result.exit_code == 0, result.output
    assert len(gate_report_files(tmp_path)) == 1
    assert not (tmp_path / "cost_ref.json").exists()
    assert not (tmp_path / "sweep_best.json").exists()


def test_gate_failure_exits_1_and_preserves_frozen_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preexisting = '{"frozen": "do not touch"}'
    (tmp_path / "cost_ref.json").write_text(preexisting)
    (tmp_path / "sweep_best.json").write_text(preexisting)
    monkeypatch.setattr(calibrate_cli, "calibrate_instance", fake_calibrate_instance)
    monkeypatch.setattr(
        calibrate_cli, "evaluate_gates", fake_evaluate_gates_fail_index_1
    )
    result = invoke(tmp_path)
    assert result.exit_code == 1
    reports = gate_report_files(tmp_path)
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["all_passed"] is False
    assert (tmp_path / "cost_ref.json").read_text() == preexisting
    assert (tmp_path / "sweep_best.json").read_text() == preexisting


def test_instance_error_recorded_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        calibrate_cli, "calibrate_instance", erroring_calibrate_instance
    )
    monkeypatch.setattr(calibrate_cli, "evaluate_gates", fake_evaluate_gates_pass)
    result = invoke(tmp_path)
    assert result.exit_code == 1
    reports = gate_report_files(tmp_path)
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text())
    assert payload["all_passed"] is False
    for entry in payload["instances"]:
        assert entry["passed"] is False
        assert "no strictly feasible" in entry["error"]
    assert not (tmp_path / "sweep_best.json").exists()


def test_exit_code_matches_report_all_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibrate_cli, "calibrate_instance", fake_calibrate_instance)
    monkeypatch.setattr(calibrate_cli, "evaluate_gates", fake_evaluate_gates_pass)
    result = invoke(tmp_path, gates_only=True)
    payload = json.loads(gate_report_files(tmp_path)[0].read_text())
    assert (result.exit_code == 0) == payload["all_passed"]


def test_canonical_generator_is_used_and_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    GENERATOR_CALLS.clear()
    monkeypatch.setattr(
        calibrate_cli, "generate_instances", recording_generate_instances
    )
    monkeypatch.setattr(calibrate_cli, "calibrate_instance", fake_calibrate_instance)
    monkeypatch.setattr(calibrate_cli, "evaluate_gates", fake_evaluate_gates_pass)
    result = invoke(tmp_path, gates_only=True)
    assert result.exit_code == 0, result.output
    assert GENERATOR_CALLS == ["canonical"]
    payload = json.loads(gate_report_files(tmp_path)[0].read_text())
    assert payload["config"]["generator_version"] == "v2"
    assert payload["config"]["generator"]["span_min_ft"] == 120.0
    assert payload["config"]["generator"]["depth_limit_variants"] == [
        "none",
        "generous",
    ]


def test_generator_selection_option_is_not_exposed(tmp_path: Path) -> None:
    help_result = runner.invoke(calibrate_cli.app, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "--generator" not in help_result.output
    result = runner.invoke(calibrate_cli.app, ["--generator", "v1"])
    assert result.exit_code != 0
    assert gate_report_files(tmp_path) == []
