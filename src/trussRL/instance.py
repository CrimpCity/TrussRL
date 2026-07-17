"""The problem definition: everything needed to grade a design, in one place.

A TrussInstance carries every randomized quantity — span, load cases, depth
limit, deflection denominator, grading constants — so prompt rendering and
scoring read from one trusted source and can never disagree about what
problem is being graded.
"""

from dataclasses import dataclass

from trussRL.loads import LoadCase


@dataclass(frozen=True)
class TrussInstance:
    """One design problem: the trusted, harness-generated side of a rollout.

    Assumptions:
        1. Every field is shown verbatim in the prompt (vision section 11);
           nothing here is hidden from the policy except cost_ref_usd, which
           only shapes the graded cost term.
        2. Fields come from the trusted generator, not the model, so the
           verifier may raise on invalid values here instead of scoring them.

    Attributes:
        span_ft: truss span in feet
        load_cases: the applied load cases before self-weight, one to three
            per the prompt contract; self-weight is added by the loads layer
        depth_limit_ft: stated depth limit in feet, or None when the prompt
            states no limit
        defl_denom: deflection-limit denominator, so the limit is
            span / defl_denom (e.g. 360 for L/360)
        cost_ref_usd: calibration cost reference for the graded cost term,
            or None before calibration has set it
    """

    span_ft: float
    load_cases: tuple[LoadCase, ...]
    depth_limit_ft: float | None = None
    defl_denom: int = 360
    cost_ref_usd: float | None = None
