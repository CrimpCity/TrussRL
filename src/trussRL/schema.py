"""Turn raw completion text into a validated design — or a scored failure.

The policy emits free text; everything downstream needs typed fields. Parse
and validation failures land on reward ladder rung 0 instead of raising, so
malformed output is a training signal, never a crash.
"""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


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


def parse_design(completion_text: str) -> TrussDesign | ParseFailure:
    """Extract and validate the design JSON from raw completion text.

    Assumptions:
        1. Rung 0 exists so malformed output is scored, not crashed on:
           every failure mode returns a ParseFailure, never raises.

    Args:
        completion_text: the policy's full completion, optional reasoning
            followed by a fenced JSON block

    Returns:
        TrussDesign: the validated design, or ParseFailure with the reason
            when extraction or validation fails
    """
    raise NotImplementedError(
        "Unit 6 (outline.md): JSON-block extraction + schema validation"
    )
