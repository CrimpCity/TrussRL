"""Turn raw completion text into a validated design — or a scored failure.

The policy emits free text; everything downstream needs typed fields. Parse
and validation failures land on reward ladder rung 0 instead of raising, so
malformed output is a training signal, never a crash.
"""

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from trussRL.catalog import has_section

FENCED_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL
)


class TrussDesign(BaseModel):
    """The seven-field JSON contract the policy emits (vision section 3).

    The schema is fixed across typologies: unused groups (Warren has no
    verticals) are ignored, not rejected, so the parser and the model's
    format learning stay stable.

    Attributes:
        truss_type: typology name, e.g. "warren"
        n_bays: number of bottom-chord bays; bay width is derived as
            span / n_bays, never specified
        depth_ft: truss depth in feet
        top_chord: catalog designation for the top chord group
        bottom_chord: catalog designation for the bottom chord group
        diagonals: catalog designation for the diagonals group
        verticals: catalog designation for the verticals group, or None
            when the completion omits it
    """

    model_config = ConfigDict(frozen=True)

    truss_type: str
    n_bays: int
    depth_ft: float
    top_chord: str
    bottom_chord: str
    diagonals: str
    verticals: str | None = None

    def sections_by_group(self) -> dict[str, str]:
        """Adapt the section fields to the mapping shape loads and DRC consume.

        Assumptions:
            1. Groups with no assigned section (verticals omitted) are left
               out of the mapping; downstream layers ignore mapping keys
               unused by the geometry and fail loudly on geometry groups
               missing from the mapping.

        Args: None

        Returns:
            dict: member group name mapped to its section designation
        """
        sections = {
            "top_chord": self.top_chord,
            "bottom_chord": self.bottom_chord,
            "diagonals": self.diagonals,
        }
        if self.verticals is not None:
            sections["verticals"] = self.verticals
        return sections


@dataclass(frozen=True)
class ParseFailure:
    """A completion that failed to yield a valid TrussDesign (rung 0).

    Attributes:
        reason: human-readable description of why parsing or validation
            failed, e.g. "no JSON block found" or a pydantic error summary
    """

    reason: str


def extract_last_json_object(text: str) -> object | None:
    """Decode the last top-level JSON object embedded in free text.

    Assumptions:
        1. Fallback for completions that skip the fenced-block format:
           scans every "{" and keeps the last span that decodes, so a
           correct design without fences is still scored on its merits.
        2. Objects nested inside an already-decoded object are not
           considered top-level and cannot shadow their container.

    Args:
        text: raw completion text with no usable fenced block

    Returns:
        object: the decoded value of the last balanced JSON object, or
            None when no substring decodes
    """
    decoder = json.JSONDecoder()
    found: object | None = None
    index = text.find("{")
    while index != -1:
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        found = value
        index = text.find("{", end)
    return found


def decode_design_payload(completion_text: str) -> object | ParseFailure:
    """Locate and decode the design JSON payload in completion text.

    Assumptions:
        1. Fenced blocks are authoritative when present: the last block
           that decodes wins (models restate examples early, the final
           block is the answer), and if every block is malformed that is
           the failure — no rescue from surrounding prose.
        2. Without any fenced block, extraction is lenient: the last
           balanced JSON object anywhere in the text is used.

    Args:
        completion_text: the policy's full completion, optional reasoning
            followed by the design JSON

    Returns:
        object: the decoded JSON value, or ParseFailure when no block is
            found or the block content is malformed
    """
    blocks = FENCED_BLOCK_PATTERN.findall(completion_text)
    last_error: json.JSONDecodeError | None = None
    for block in reversed(blocks):
        try:
            decoded: object = json.loads(block)
            return decoded
        except json.JSONDecodeError as error:
            if last_error is None:
                last_error = error
    if last_error is not None:
        return ParseFailure(f"malformed JSON in fenced block: {last_error}")
    payload = extract_last_json_object(completion_text)
    if payload is None:
        return ParseFailure("no JSON block found")
    return payload


def parse_design(completion_text: str) -> TrussDesign | ParseFailure:
    """Extract and validate the design JSON from raw completion text.

    Assumptions:
        1. Rung 0 exists so malformed output is scored, not crashed on:
           every failure mode returns a ParseFailure, never raises.
        2. Section names are validated against the catalog here so every
           TrussDesign leaving this stage references real sections;
           truss_type and numeric ranges are DRC's job (rung 1), giving
           an unsupported typology a richer signal than rung 0.

    Args:
        completion_text: the policy's full completion, optional reasoning
            followed by a fenced JSON block

    Returns:
        TrussDesign: the validated design, or ParseFailure with the reason
            when extraction or validation fails
    """
    payload = decode_design_payload(completion_text)
    if isinstance(payload, ParseFailure):
        return payload
    if not isinstance(payload, dict):
        return ParseFailure(
            f"design JSON must be an object, got {type(payload).__name__}"
        )
    try:
        design = TrussDesign.model_validate(payload)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'design'}: {item['msg']}"
            for item in error.errors()
        )
        return ParseFailure(f"schema validation failed: {details}")
    unknown = sorted(
        {name for name in design.sections_by_group().values() if not has_section(name)}
    )
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        return ParseFailure(f"unknown section designation(s): {names}")
    return design
