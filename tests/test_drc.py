"""DRC layer unit tests (drc-layer.md Unit 1).

Asserts the known-good baseline passes, every check fails in isolation
with a stable reason substring, boundary values pass inclusively, and the
anti-cascade gating holds: one root cause produces one failure, never a
pile of downstream noise. The slenderness case with a rectangular HSS
proves both the governing-radius choice (ry, not rx) and the feet-to-
inches conversion.
"""

import math
from typing import Any

import pytest

from trussRL.drc import DRCResult, run_drc

BASELINE: dict[str, Any] = {
    "truss_type": "warren",
    "span_ft": 96.0,
    "n_bays": 8,
    "depth_ft": 8.0,
    "sections_by_group": {
        "bottom_chord": "HSS6X6X3/8",
        "top_chord": "HSS6X6X3/8",
        "diagonals": "HSS4X4X1/4",
    },
    "depth_limit_ft": None,
}

BIG_DIAGONAL_SECTIONS = {
    "bottom_chord": "HSS6X6X3/8",
    "top_chord": "HSS6X6X3/8",
    "diagonals": "HSS6X6X3/8",
}


def run_drc_with(**overrides: Any) -> DRCResult:
    """Run the DRC on the known-good baseline with fields overridden.

    Assumptions:
        1. Passes fields as **kwargs so mistyped values (n_bays=8.5)
           reach run_drc without mypy casts, matching how untrusted
           design fields arrive in production.

    Args:
        **overrides: run_drc keyword arguments to replace in the baseline

    Returns:
        DRCResult: the check outcome for the modified design
    """
    fields = {**BASELINE, **overrides}
    return run_drc(**fields)


def test_known_good_design_passes() -> None:
    result = run_drc_with()
    assert result.failures == ()
    assert result.passed
    assert result.summary == "pass"


@pytest.mark.parametrize("n_bays", [4, 24])
def test_n_bays_bounds_pass_inclusively(n_bays: int) -> None:
    assert run_drc_with(n_bays=n_bays).passed


@pytest.mark.parametrize("depth_ft", [96.0 / 25.0, 24.0])
def test_depth_ratio_bounds_pass_inclusively(depth_ft: float) -> None:
    # Big diagonals keep slenderness legal at depth 24 (diagonal ~24.7 ft).
    result = run_drc_with(depth_ft=depth_ft, sections_by_group=BIG_DIAGONAL_SECTIONS)
    assert result.passed


def test_depth_equal_to_stated_limit_passes() -> None:
    assert run_drc_with(depth_ft=8.0, depth_limit_ft=8.0).passed


def test_unsupported_truss_type_is_scored_not_raised() -> None:
    result = run_drc_with(truss_type="pratt")
    assert len(result.failures) == 1
    (failure,) = result.failures
    assert failure.check == "truss_type"
    assert "unsupported truss_type" in failure.reason


def test_unknown_section_designation_fails() -> None:
    sections = {**BASELINE["sections_by_group"], "diagonals": "HSS99X99X9"}
    result = run_drc_with(sections_by_group=sections)
    assert len(result.failures) == 1
    (failure,) = result.failures
    assert failure.check == "sections"
    assert "unknown section designation" in failure.reason


def test_missing_required_group_fails() -> None:
    sections = {
        k: v for k, v in BASELINE["sections_by_group"].items() if k != "top_chord"
    }
    result = run_drc_with(sections_by_group=sections)
    assert len(result.failures) == 1
    (failure,) = result.failures
    assert failure.check == "sections"
    assert "no section assigned" in failure.reason
    assert "top_chord" in failure.reason


def test_unused_group_with_valid_section_passes() -> None:
    # Warren has no verticals; a valid designation for them is ignored.
    sections = {**BASELINE["sections_by_group"], "verticals": "HSS4X4X1/4"}
    assert run_drc_with(sections_by_group=sections).passed


def test_unused_group_with_invalid_section_fails() -> None:
    sections = {**BASELINE["sections_by_group"], "verticals": "HSS99X99X9"}
    result = run_drc_with(sections_by_group=sections)
    assert len(result.failures) == 1
    assert result.failures[0].check == "sections"


@pytest.mark.parametrize("n_bays", [0, 3, 25, -3, 100])
def test_n_bays_out_of_range_fails(n_bays: int) -> None:
    result = run_drc_with(n_bays=n_bays)
    failures = [f for f in result.failures if f.check == "n_bays"]
    assert len(failures) == 1
    assert "must be within [4, 24]" in failures[0].reason


@pytest.mark.parametrize("n_bays", [8.5, 8.0])
def test_n_bays_float_fails_without_coercion(n_bays: float) -> None:
    result = run_drc_with(n_bays=n_bays)
    assert len(result.failures) == 1
    (failure,) = result.failures
    assert failure.check == "n_bays"
    assert "must be an integer" in failure.reason


@pytest.mark.parametrize(
    ("depth_ft", "sections"),
    [
        (3.7, BASELINE["sections_by_group"]),
        (25.0, BIG_DIAGONAL_SECTIONS),
    ],
)
def test_depth_outside_ratio_window_fails(
    depth_ft: float, sections: dict[str, str]
) -> None:
    result = run_drc_with(depth_ft=depth_ft, sections_by_group=sections)
    assert len(result.failures) == 1
    (failure,) = result.failures
    assert failure.check == "depth"
    assert "must be within [span/25, span/4]" in failure.reason


def test_depth_over_stated_limit_fails() -> None:
    result = run_drc_with(depth_ft=10.0, depth_limit_ft=9.0)
    assert len(result.failures) == 1
    (failure,) = result.failures
    assert failure.check == "depth"
    assert "stated depth limit" in failure.reason
    assert run_drc_with(depth_ft=10.0, depth_limit_ft=None).passed


def test_slender_diagonals_fail() -> None:
    # Diagonal 10 ft = 120 in; r 0.557 in -> L/r = 215 > 200.
    sections = {**BASELINE["sections_by_group"], "diagonals": "HSS1-1/2X1-1/2X1/8"}
    result = run_drc_with(sections_by_group=sections)
    assert len(result.failures) == 1
    (failure,) = result.failures
    assert failure.check == "slenderness"
    assert "slenderness L/r" in failure.reason


def test_slenderness_uses_weak_axis_radius_and_inches() -> None:
    # 4 bays -> 24 ft bottom chord = 288 in. HSS12X2X1/4: ry 0.845 gives
    # L/r = 341 (fails); rx 3.75 would give 77 and a missing x12 would
    # give 28 - both passing. Exactly one failure proves axis and units.
    sections = {**BASELINE["sections_by_group"], "bottom_chord": "HSS12X2X1/4"}
    result = run_drc_with(n_bays=4, sections_by_group=sections)
    assert len(result.failures) == 1
    (failure,) = result.failures
    assert failure.check == "slenderness"
    assert "bottom_chord" in failure.reason


@pytest.mark.parametrize(
    ("n_bays", "depth_ft"),
    [
        (24, 24.0),  # bay width 4 ft -> aspect 6.0
        (4, 96.0 / 25.0),  # bay width 24 ft -> aspect 0.16
    ],
)
def test_bay_aspect_outside_window_fails(n_bays: int, depth_ft: float) -> None:
    result = run_drc_with(
        n_bays=n_bays, depth_ft=depth_ft, sections_by_group=BIG_DIAGONAL_SECTIONS
    )
    failures = [f for f in result.failures if f.check == "bay_aspect"]
    assert len(failures) == 1
    assert "bay aspect" in failures[0].reason


@pytest.mark.parametrize("depth_ft", [0.0, -4.0, math.nan])
def test_nonpositive_or_nan_depth_fails_without_cascade(depth_ft: float) -> None:
    result = run_drc_with(depth_ft=depth_ft)
    assert {f.check for f in result.failures} == {"depth"}


def test_invalid_truss_type_does_not_cascade() -> None:
    # No geometry means no required-groups or slenderness noise on top.
    result = run_drc_with(truss_type="howe")
    assert len(result.failures) == 1
    assert result.failures[0].check == "truss_type"


def test_multi_failure_design_collects_all_checks() -> None:
    sections = {**BASELINE["sections_by_group"], "diagonals": "HSS99X99X9"}
    result = run_drc_with(n_bays=3, depth_ft=30.0, sections_by_group=sections)
    assert {f.check for f in result.failures} == {"n_bays", "depth", "sections"}


def test_out_of_range_but_constructible_n_bays_still_gets_aspect() -> None:
    # n_bays = 30 fails the range check yet expands fine; bay width 3.2 ft
    # -> aspect 2.5 stays legal, so only the range failure is reported and
    # the aspect check demonstrably ran without piling on.
    result = run_drc_with(n_bays=30, sections_by_group=BIG_DIAGONAL_SECTIONS)
    assert {f.check for f in result.failures} == {"n_bays"}


def test_results_are_deterministic() -> None:
    sections = {**BASELINE["sections_by_group"], "diagonals": "HSS99X99X9"}
    first = run_drc_with(n_bays=3, depth_ft=30.0, sections_by_group=sections)
    second = run_drc_with(n_bays=3, depth_ft=30.0, sections_by_group=sections)
    assert first == second


def test_summary_joins_failure_reasons() -> None:
    result = run_drc_with(n_bays=3, depth_ft=30.0)
    assert not result.passed
    assert result.summary == "; ".join(f.reason for f in result.failures)
    assert "; " in result.summary


def test_nonpositive_span_raises() -> None:
    with pytest.raises(ValueError, match="span_ft must be positive"):
        run_drc_with(span_ft=0.0)
