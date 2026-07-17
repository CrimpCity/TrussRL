"""Integration tests for the verifier composition.

Asserts only what stays true as pipeline units land: built stages produce
consistent artifacts, unbuilt stages are recorded without crashing, stage
order is fixed, and the ladder constructors score correctly. Nothing here
pins a stage to its current not_built state, so landing a unit never
breaks this suite.
"""

import sys

import pytest

from trussRL.catalog import get_section
from trussRL.drc import DRCFailure, DRCResult
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.reward import (
    RUNG0_SCORE,
    RUNG1_SCORE,
    RUNG2_SCORE,
    RewardBreakdown,
    rung0,
    rung1,
    rung2,
)
from trussRL.schema import TrussDesign
from trussRL.verifier import STAGE_ORDER, run_pipeline, score

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
    trace = run_pipeline(INSTANCE, DESIGN)
    assert tuple(status.stage for status in trace.stages) == STAGE_ORDER


def test_every_stage_has_a_known_state() -> None:
    trace = run_pipeline(INSTANCE, DESIGN)
    for status in trace.stages:
        assert status.state in ("ok", "failed", "not_built", "skipped")


def test_known_good_design_expands() -> None:
    trace = run_pipeline(INSTANCE, DESIGN)
    assert trace.geometry is not None
    assert trace.geometry.n_nodes == 2 * DESIGN.n_bays + 1
    assert len(trace.geometry.members) == 4 * DESIGN.n_bays - 1
    assert trace.geometry.bay_width_ft == SPAN_FT / DESIGN.n_bays


def test_known_good_design_loads_conserve_forces() -> None:
    trace = run_pipeline(INSTANCE, DESIGN)
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


def test_unconstructible_design_is_recorded_not_raised() -> None:
    bad = DESIGN.model_copy(update={"truss_type": "pratt"})
    trace = run_pipeline(INSTANCE, bad)
    assert trace.geometry is None
    assert any(status.state == "failed" for status in trace.stages)
    assert trace.breakdown is None or trace.breakdown.rung == 1


def test_score_on_gibberish_is_rung0_or_not_implemented() -> None:
    try:
        result = score(INSTANCE, "no json here at all")
    except NotImplementedError:
        return
    assert isinstance(result, RewardBreakdown)
    assert result.rung == 0
    assert result.score == RUNG0_SCORE


def test_score_never_returns_partial_result() -> None:
    completion = DESIGN.model_dump_json()
    try:
        result = score(INSTANCE, completion)
    except NotImplementedError as error:
        assert str(error)
        return
    assert isinstance(result, RewardBreakdown)
    assert 0.0 <= result.score <= 1.0


def test_ladder_constructors() -> None:
    assert rung0("bad json").score == RUNG0_SCORE
    assert rung0("bad json").rung == 0
    drc = DRCResult(failures=(DRCFailure(check="n_bays", reason="n_bays 30 out"),))
    assert rung1(drc).score == RUNG1_SCORE
    assert rung1(drc).rung == 1
    assert "n_bays 30 out" in rung1(drc).reason
    assert rung2("timeout").score == RUNG2_SCORE
    assert rung2("timeout").rung == 2


def test_verifier_import_pulls_no_rl_dependencies() -> None:
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules
    assert "trl" not in sys.modules
