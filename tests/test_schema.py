"""Schema parse unit tests (outline.md Unit 6).

Covers the "done when" matrix: clean JSON, JSON embedded in reasoning
text, malformed JSON, wrong types, and unknown section names — each
yielding either a valid TrussDesign or a typed ParseFailure, never an
exception. Also pins the lenient-extraction contract: the last fenced
block is authoritative when fences are present, and a bare JSON object
without fences still parses.
"""

import json
from typing import Any

import pytest

from trussRL.schema import ParseFailure, TrussDesign, parse_design

DESIGN_FIELDS: dict[str, Any] = {
    "truss_type": "warren",
    "n_bays": 8,
    "depth_ft": 8.0,
    "top_chord": "HSS6X6X3/8",
    "bottom_chord": "HSS6X6X3/8",
    "diagonals": "HSS4X4X1/4",
}


def fenced(payload: dict[str, Any]) -> str:
    """Wrap a payload dict in the fenced JSON block the prompt asks for.

    Args:
        payload: design fields to serialize

    Returns:
        str: the payload as a ```json fenced block
    """
    return "```json\n" + json.dumps(payload) + "\n```"


def design_with(**overrides: Any) -> dict[str, Any]:
    """Copy the known-good design fields with some overridden.

    Args:
        **overrides: field values to replace in DESIGN_FIELDS

    Returns:
        dict: the merged design payload
    """
    return {**DESIGN_FIELDS, **overrides}


def test_clean_fenced_json_parses() -> None:
    result = parse_design(fenced(DESIGN_FIELDS))
    assert isinstance(result, TrussDesign)
    assert result.truss_type == "warren"
    assert result.n_bays == 8
    assert result.depth_ft == 8.0
    assert result.top_chord == "HSS6X6X3/8"
    assert result.verticals is None


def test_fence_without_language_tag_parses() -> None:
    text = "```\n" + json.dumps(DESIGN_FIELDS) + "\n```"
    assert isinstance(parse_design(text), TrussDesign)


def test_json_embedded_in_reasoning_parses() -> None:
    text = (
        "Let me think about depth. A span of 96 ft suggests depth near "
        "span/12, so 8 ft.\n\n" + fenced(DESIGN_FIELDS) + "\n\nThat is my answer."
    )
    result = parse_design(text)
    assert isinstance(result, TrussDesign)
    assert result.depth_ft == 8.0


def test_last_fenced_block_wins() -> None:
    text = (
        "An example format:\n"
        + fenced(design_with(n_bays=4))
        + "\nMy actual design:\n"
        + fenced(design_with(n_bays=10))
    )
    result = parse_design(text)
    assert isinstance(result, TrussDesign)
    assert result.n_bays == 10


def test_bare_json_without_fence_parses() -> None:
    result = parse_design(json.dumps(DESIGN_FIELDS))
    assert isinstance(result, TrussDesign)


def test_bare_json_embedded_in_prose_parses() -> None:
    text = "Here is the design: " + json.dumps(DESIGN_FIELDS) + " — done."
    assert isinstance(parse_design(text), TrussDesign)


def test_verticals_round_trip() -> None:
    result = parse_design(fenced(design_with(verticals="HSS4X4X1/4")))
    assert isinstance(result, TrussDesign)
    assert result.verticals == "HSS4X4X1/4"
    assert result.sections_by_group()["verticals"] == "HSS4X4X1/4"


def test_omitted_verticals_left_out_of_groups() -> None:
    result = parse_design(fenced(DESIGN_FIELDS))
    assert isinstance(result, TrussDesign)
    assert "verticals" not in result.sections_by_group()


def test_null_verticals_treated_as_omitted() -> None:
    result = parse_design(fenced(design_with(verticals=None)))
    assert isinstance(result, TrussDesign)
    assert result.verticals is None


def test_pure_prose_fails() -> None:
    result = parse_design("I could not decide on a design, sorry.")
    assert isinstance(result, ParseFailure)
    assert "no JSON block found" in result.reason


def test_malformed_fenced_json_fails() -> None:
    text = '```json\n{"truss_type": "warren", n_bays: 8}\n```'
    result = parse_design(text)
    assert isinstance(result, ParseFailure)
    assert "malformed JSON" in result.reason


def test_malformed_fence_is_not_rescued_by_prose_json() -> None:
    # The fence was used, so its content is the answer; a valid object in
    # the surrounding reasoning must not be silently graded instead.
    text = json.dumps(DESIGN_FIELDS) + '\n```json\n{"truss_type": broken\n```'
    result = parse_design(text)
    assert isinstance(result, ParseFailure)
    assert "malformed JSON" in result.reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_bays", "eight"),
        ("n_bays", 8.5),
        ("depth_ft", [8.0]),
        ("top_chord", 42),
    ],
)
def test_wrong_type_fails_naming_the_field(field: str, value: Any) -> None:
    result = parse_design(fenced(design_with(**{field: value})))
    assert isinstance(result, ParseFailure)
    assert "schema validation failed" in result.reason
    assert field in result.reason


def test_missing_required_field_fails() -> None:
    payload = {k: v for k, v in DESIGN_FIELDS.items() if k != "bottom_chord"}
    result = parse_design(fenced(payload))
    assert isinstance(result, ParseFailure)
    assert "schema validation failed" in result.reason
    assert "bottom_chord" in result.reason


def test_unknown_section_fails_naming_the_designation() -> None:
    result = parse_design(fenced(design_with(diagonals="HSS99X99X9")))
    assert isinstance(result, ParseFailure)
    assert "unknown section designation" in result.reason
    assert "HSS99X99X9" in result.reason


def test_unknown_verticals_section_fails_even_for_warren() -> None:
    result = parse_design(fenced(design_with(verticals="HSS99X99X9")))
    assert isinstance(result, ParseFailure)
    assert "unknown section designation" in result.reason


def test_lowercase_section_names_accepted() -> None:
    result = parse_design(fenced(design_with(top_chord="hss6x6x3/8")))
    assert isinstance(result, TrussDesign)
    assert result.top_chord == "hss6x6x3/8"


def test_unknown_truss_type_parses_for_drc_to_reject() -> None:
    result = parse_design(fenced(design_with(truss_type="pratt")))
    assert isinstance(result, TrussDesign)
    assert result.truss_type == "pratt"


def test_non_object_json_fails() -> None:
    result = parse_design("```json\n[1, 2, 3]\n```")
    assert isinstance(result, ParseFailure)
    assert "must be an object" in result.reason


def test_extra_keys_are_ignored() -> None:
    result = parse_design(fenced(design_with(notes="chosen for low cost")))
    assert isinstance(result, TrussDesign)


def test_integer_depth_coerces_to_float() -> None:
    result = parse_design(fenced(design_with(depth_ft=8)))
    assert isinstance(result, TrussDesign)
    assert result.depth_ft == 8.0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{",
        "{}",
        "``````",
        "```json\n```",
        "null",
        "{'truss_type': 'warren'}",
        "reasoning { incomplete " + json.dumps(DESIGN_FIELDS)[:-2],
    ],
)
def test_garbage_never_raises(text: str) -> None:
    result = parse_design(text)
    assert isinstance(result, (TrussDesign, ParseFailure))


def test_parse_is_deterministic() -> None:
    text = fenced(DESIGN_FIELDS)
    assert parse_design(text) == parse_design(text)
