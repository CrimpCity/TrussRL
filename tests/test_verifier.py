"""End-to-end coverage for verifier composition and reward-ladder exits."""

import subprocess
import sys
from unittest.mock import Mock

import pytest

import trussRL.verifier as verifier
from trussRL.catalog import get_section
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.reward import RUNG0_SCORE, RUNG1_SCORE, RUNG2_SCORE, RUNG3_FLOOR
from trussRL.schema import TrussDesign
from trussRL.solver import SolveOutcome

SPAN_FT = 96.0
W_KIP_PER_FT = -2.0

INSTANCE = TrussInstance(
    span_ft=SPAN_FT,
    load_cases=(LoadCase(w_kip_per_ft=W_KIP_PER_FT, level="bottom"),),
)

DESIGN = TrussDesign(
    truss_type="warren",
    n_bays=8,
    depth_ft=8.0,
    top_chord="HSS6X6X3/8",
    bottom_chord="HSS6X6X3/8",
    diagonals="HSS4X4X1/4",
)


def test_stage_order_is_fixed() -> None:
    trace = verifier.run_pipeline(INSTANCE, DESIGN)
    assert tuple(status.stage for status in trace.stages) == verifier.STAGE_ORDER
    assert all(status.state == "ok" for status in trace.stages)


def test_known_good_design_expands() -> None:
    trace = verifier.run_pipeline(INSTANCE, DESIGN)
    assert trace.geometry is not None
    assert trace.geometry.n_nodes == 2 * DESIGN.n_bays + 1
    assert len(trace.geometry.members) == 4 * DESIGN.n_bays - 1
    assert trace.geometry.bay_width_ft == SPAN_FT / DESIGN.n_bays


def test_known_good_design_loads_conserve_forces() -> None:
    trace = verifier.run_pipeline(INSTANCE, DESIGN)
    assert trace.geometry is not None
    assert trace.case_loads is not None
    assert len(trace.case_loads) == len(INSTANCE.load_cases)

    sections = DESIGN.sections_by_group()
    self_weight_kip = sum(
        get_section(sections[member.group]).weight_lb_per_ft * member.length_ft / 1000.0
        for member in trace.geometry.members
    )
    expected_fy = W_KIP_PER_FT * SPAN_FT - self_weight_kip
    assert trace.case_loads[0].total_fy_kip == pytest.approx(expected_fy)
    assert trace.case_loads[0].total_fx_kip == pytest.approx(0.0)


def test_garbage_completion_exits_at_rung0() -> None:
    result = verifier.score(INSTANCE, "no json here at all")

    assert result.rung == 0
    assert result.score == RUNG0_SCORE
    assert result.reason == "parse failure: no JSON block found"


def test_parseable_drc_invalid_completion_exits_at_rung1() -> None:
    invalid_design = DESIGN.model_copy(update={"n_bays": 30})
    result = verifier.score(INSTANCE, invalid_design.model_dump_json())

    assert result.rung == 1
    assert result.score == RUNG1_SCORE
    assert result.reason.startswith("drc: ")
    assert "n_bays must be within [4, 24], got 30" in result.reason


def test_solver_failure_exits_at_rung2_after_preceding_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_outcome = SolveOutcome(solutions=(), failure="injected solver failure")
    solve_mock = Mock(return_value=failed_outcome)
    monkeypatch.setattr(verifier, "solve_cases", solve_mock)

    result = verifier.score(INSTANCE, DESIGN.model_dump_json())

    assert result.rung == 2
    assert result.score == RUNG2_SCORE
    assert result.reason == "solver: injected solver failure"
    solve_mock.assert_called_once()
    geometry, sections, case_loads = solve_mock.call_args.args
    assert geometry.n_nodes == 2 * DESIGN.n_bays + 1
    assert sections == DESIGN.sections_by_group()
    assert len(case_loads) == len(INSTANCE.load_cases)


def test_real_solver_completion_reaches_populated_rung3() -> None:
    result = verifier.score(INSTANCE, DESIGN.model_dump_json())

    assert result.rung == 3
    assert result.score >= RUNG3_FLOOR
    assert result.reason == "solved"
    assert result.u_strength is not None
    assert result.u_buckling is not None
    assert result.u_defl is not None
    assert result.cost_total_usd is not None
    assert result.feasible is not None


def test_verifier_import_pulls_no_training_dependencies() -> None:
    import_guard = """
import importlib.abc
import sys

blocked_roots = {
    "accelerate",
    "bitsandbytes",
    "datasets",
    "deepspeed",
    "peft",
    "torch",
    "transformers",
    "trl",
    "unsloth",
    "vllm",
    "wandb",
}

class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if (
            fullname == "trussRL.training"
            or fullname.startswith("trussRL.training.")
            or root in blocked_roots
        ):
            raise AssertionError(f"forbidden verifier dependency imported: {fullname}")
        return None

sys.meta_path.insert(0, RejectTrainingImports())
import trussRL.verifier
"""
    completed = subprocess.run(
        [sys.executable, "-c", import_guard],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
