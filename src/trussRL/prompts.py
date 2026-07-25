"""Render an instance into the prompt the policy sees.

Everything the reward grades appears verbatim in the prompt text — the
policy is only ever scored on what it was shown.
"""

from trussRL.catalog import list_sections
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase
from trussRL.reward import COST_PER_NODE_USD, W_BUCKLING, W_COST, W_DEFL, W_STRENGTH

INTRO = "You are designing a simply supported steel truss to span a gap."

TASK_DESCRIPTION = (
    'The truss type for this problem is "warren". Choose the number of bays,\n'
    "the truss depth in feet, and a section from the catalog below for each\n"
    "member group."
)

RUBRIC_TEMPLATE = (
    "Your design is scored on:\n"
    "- Strength: axial demand within capacity for every member under every\n"
    "  load case ({w_strength:.0%}).\n"
    "- Buckling: compression demand within buckling capacity, using each\n"
    "  member's actual unbraced length ({w_buckling:.0%}).\n"
    "- Deflection: maximum vertical deflection at most L/{defl_denom} under\n"
    "  each gravity case ({w_defl:.0%}).\n"
    "- Cost: total member cost plus ${cost_per_node:g} per connection\n"
    "  node; lower is better ({w_cost:.0%}).\n\n"
    "Every member is checked against the worst case across all load cases.\n"
    "Credit for each constraint degrades linearly with overage. Cost credit\n"
    "is only awarded to feasible designs."
)

CATALOG_HEADER = "CATALOG (designation | A in^2 | r_min in | weight lb/ft | cost $/ft):"

RESPONSE_INSTRUCTIONS = (
    "Respond with your reasoning, then a single JSON object in a fenced\n"
    "code block, with exactly these fields:"
)

JSON_TEMPLATE = (
    "```json\n"
    '{ "truss_type": "warren", "n_bays": <integer>, "depth_ft": <number>,\n'
    '  "top_chord": "<catalog designation>", "bottom_chord": "...",\n'
    '  "diagonals": "...", "verticals": "<optional>" }\n'
    "```"
)


def format_number(value: float) -> str:
    """Render a number as clean human-readable text.

    Args:
        value: the number to render

    Returns:
        str: value formatted with Python's general format, e.g. "96",
            "2.5", "5.5"
    """
    return f"{value:g}"


def format_depth_limit(depth_limit_ft: float | None) -> str:
    """Render a depth limit for the DEPTH LIMIT prompt line.

    Args:
        depth_limit_ft: stated depth limit in feet, or None when unstated

    Returns:
        str: "none stated", or "{value} ft"
    """
    if depth_limit_ft is None:
        return "none stated"
    return f"{format_number(depth_limit_ft)} ft"


def format_load_case(index: int, case: LoadCase) -> str:
    """Render one load case as an enumerated prompt line.

    The model never sees the internal sign convention: direction words are
    derived from the signs of the case's fields instead.

    Args:
        index: zero-based case index; rendered 1-indexed
        case: the load case to describe

    Returns:
        str: an enumerated line describing the case's vertical and
            longitudinal components in plain language

    Raises:
        ValueError: if the case has no vertical or longitudinal component
            (a trusted-generator bug — every generated case carries load).
    """
    if case.w_kip_per_ft == 0.0 and case.longitudinal_kip == 0.0:
        raise ValueError(f"load case {index + 1} has no vertical or longitudinal load")
    parts = []
    if case.w_kip_per_ft < 0.0:
        magnitude = format_number(-case.w_kip_per_ft)
        parts.append(
            f"{magnitude} kip/ft downward, applied at the {case.level} "
            "chord (deck) level"
        )
    elif case.w_kip_per_ft > 0.0:
        magnitude = format_number(case.w_kip_per_ft)
        parts.append(f"{magnitude} kip/ft net uplift")
    if case.longitudinal_kip != 0.0:
        magnitude = format_number(abs(case.longitudinal_kip))
        parts.append(f"{magnitude} kip horizontal toward the roller/pin")
    return f"{index + 1}. " + "; ".join(parts) + "."


def catalog_table() -> str:
    """Render the full section catalog as a pipe-separated table.

    Args: None

    Returns:
        str: one row per catalog section, in catalog order, newline
            separated
    """
    rows = [
        " | ".join(
            (
                section.designation,
                format_number(section.area_in2),
                format_number(section.r_min_in),
                format_number(section.weight_lb_per_ft),
                format_number(section.cost_per_ft_usd),
            )
        )
        for section in list_sections()
    ]
    return "\n".join(rows)


def render_prompt(instance: TrussInstance) -> str:
    """Render a problem instance into the exact text the policy sees.

    Assumptions:
        1. Every value the reward grades (span, load cases, depth limit,
           defl_denom, rubric weights, cost-per-node) appears verbatim;
           nothing internal to grading (clamp slopes, DRC bounds, rungs,
           cost_ref, the sweep) ever does.

    Args:
        instance: the problem instance to render

    Returns:
        str: the full prompt text, ending in the fenced JSON schema
            placeholder
    """
    load_lines = "\n".join(
        format_load_case(index, case) for index, case in enumerate(instance.load_cases)
    )
    rubric = RUBRIC_TEMPLATE.format(
        w_strength=W_STRENGTH,
        w_buckling=W_BUCKLING,
        defl_denom=instance.defl_denom,
        w_defl=W_DEFL,
        cost_per_node=COST_PER_NODE_USD,
        w_cost=W_COST,
    )
    sections = (
        INTRO,
        f"SPAN: {format_number(instance.span_ft)} ft.\n"
        f"DEPTH LIMIT: {format_depth_limit(instance.depth_limit_ft)}.\n"
        f"{load_lines}",
        TASK_DESCRIPTION,
        rubric,
        f"{CATALOG_HEADER}\n{catalog_table()}",
        RESPONSE_INSTRUCTIONS,
        JSON_TEMPLATE,
    )
    return "\n\n".join(sections)
