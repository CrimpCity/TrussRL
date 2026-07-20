"""Prompt rendering unit tests (Unit 7).

Asserts prompt-reproducibility for a fixed seed, that every rendered value
traces back to the instance and the reward's own constants, and that no
internal grading detail (clamp slopes, DRC bounds, rungs, cost_ref) leaks
into text the policy sees.
"""

import pytest

from trussRL.catalog import list_sections
from trussRL.generator import generate_instance
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.prompts import format_load_case, render_prompt

INTERNAL_TERMS = (
    "cost_ref",
    "sweep",
    "rung",
    "clamp",
    "penalty",
    "drc",
    "0.15",
    "0.10",
    "50%",
)


def test_same_seed_reproduces_prompt_text() -> None:
    for seed in (0, 1, 42, 12345):
        instance = generate_instance(seed)
        assert render_prompt(instance) == render_prompt(instance)
        assert render_prompt(generate_instance(seed)) == render_prompt(
            generate_instance(seed)
        )


def test_prompt_shows_instance_values() -> None:
    instance = TrussInstance(
        span_ft=96.0,
        load_cases=(LoadCase(w_kip_per_ft=-2.0, level="bottom"),),
        depth_limit_ft=None,
        defl_denom=360,
    )
    prompt = render_prompt(instance)
    assert "SPAN: 96 ft" in prompt
    assert "2 kip/ft downward" in prompt
    assert "bottom chord (deck) level" in prompt
    assert "L/360" in prompt
    assert "DEPTH LIMIT: none stated" in prompt


def test_prompt_shows_stated_depth_limit() -> None:
    instance = TrussInstance(
        span_ft=96.0,
        load_cases=(LoadCase(w_kip_per_ft=-2.0, level="bottom"),),
        depth_limit_ft=5.5,
        defl_denom=360,
    )
    prompt = render_prompt(instance)
    assert "5.5 ft" in prompt
    assert "none stated" not in prompt


def test_prompt_defl_denom_is_a_parameter() -> None:
    instance = TrussInstance(
        span_ft=96.0,
        load_cases=(LoadCase(w_kip_per_ft=-2.0, level="bottom"),),
        depth_limit_ft=None,
        defl_denom=500,
    )
    prompt = render_prompt(instance)
    assert "L/500" in prompt
    assert "L/360" not in prompt


def test_prompt_lists_full_catalog() -> None:
    instance = generate_instance(0)
    prompt = render_prompt(instance)
    for section in list_sections():
        assert section.designation in prompt


def test_prompt_states_rubric_weights() -> None:
    instance = generate_instance(0)
    prompt = render_prompt(instance)
    assert "25%" in prompt
    assert "20%" in prompt
    assert "30%" in prompt
    assert "$150" in prompt


def test_prompt_lists_schema_fields() -> None:
    instance = generate_instance(0)
    prompt = render_prompt(instance)
    for field in (
        "truss_type",
        "n_bays",
        "depth_ft",
        "top_chord",
        "bottom_chord",
        "diagonals",
        "verticals",
    ):
        assert field in prompt
    assert '"warren"' in prompt


def test_prompt_hides_internals() -> None:
    instance = generate_instance(0)
    prompt_lower = render_prompt(instance).lower()
    for term in INTERNAL_TERMS:
        assert term not in prompt_lower


def test_format_load_case_directions() -> None:
    downward = format_load_case(0, LoadCase(w_kip_per_ft=-2.0, level="bottom"))
    assert "2 kip/ft downward" in downward
    assert "bottom chord (deck) level" in downward

    uplift = format_load_case(0, LoadCase(w_kip_per_ft=1.2, level="top"))
    assert "1.2 kip/ft net uplift" in uplift

    longitudinal = format_load_case(
        0, LoadCase(w_kip_per_ft=-2.0, level="bottom", longitudinal_kip=15.0)
    )
    assert "15 kip horizontal toward the roller/pin" in longitudinal

    with pytest.raises(ValueError, match="no vertical or longitudinal load"):
        format_load_case(0, LoadCase())


def test_prompt_has_no_scaffold() -> None:
    instance = generate_instance(0)
    prompt = render_prompt(instance)
    assert "Before your final JSON" not in prompt
