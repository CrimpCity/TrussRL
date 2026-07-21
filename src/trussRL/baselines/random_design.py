"""Uniform random designs: the floor everything else is measured against.

Seeded sampling over the whole design space feeds calibration — the score
distributions behind the sanity gates and the frozen grading constants.
"""

import random

from trussRL.catalog import list_sections
from trussRL.drc import (
    DEPTH_SPAN_FRACTION_MAX,
    DEPTH_SPAN_FRACTION_MIN,
    N_BAYS_MAX,
    N_BAYS_MIN,
)
from trussRL.expander import EXPANDERS
from trussRL.instance import TrussInstance
from trussRL.schema import TrussDesign

TRUSS_TYPES = tuple(sorted(EXPANDERS))
SECTION_NAMES = tuple(section.designation for section in list_sections())


def depth_window_ft(instance: TrussInstance) -> tuple[float, float]:
    """The depth interval a design can occupy without a depth DRC failure.

    Assumptions:
        1. The stated depth limit only ever tightens the upper bound; the
           generator's variants (span/6, span/14 rounded up) always sit
           above the span/25 floor, so the window is never empty.

    Args:
        instance: the problem instance, providing span_ft and any stated
            depth limit

    Returns:
        tuple: (low, high) depth bounds in feet — span/25 up to the lesser
            of span/4 and the stated depth limit
    """
    low_ft = instance.span_ft * DEPTH_SPAN_FRACTION_MIN
    high_ft = instance.span_ft * DEPTH_SPAN_FRACTION_MAX
    if instance.depth_limit_ft is not None:
        high_ft = min(high_ft, instance.depth_limit_ft)
    return low_ft, high_ft


def sample_design(rng: random.Random, instance: TrussInstance) -> TrussDesign:
    """Draw one uniform random design for an instance.

    Assumptions:
        1. Draw order is fixed and part of the reproducibility contract,
           mirroring generate_instance: (1) truss_type, (2) n_bays, (3)
           depth, (4)-(7) section picks for top chord, bottom chord,
           verticals, diagonals. Inserting, removing, or reordering a draw
           silently changes what every downstream seed produces and must
           be treated as a reproducibility break.
        2. Depth is drawn from the effective window (clamped to the stated
           limit) so samples are not wasted on guaranteed rung-1 depth
           failures; slenderness and bay-aspect DRC rejections remain and
           are intended coverage of rung 1.
        3. Verticals are always drawn even though Warren geometry ignores
           them — a fixed draw count keeps the sampler typology-generic.

    Args:
        rng: seeded random source; consumed, so repeated calls yield a
            reproducible sequence of designs
        instance: the problem instance the design will be graded against

    Returns:
        TrussDesign: one schema-valid design with every group referencing
            a real catalog section
    """
    truss_type = rng.choice(TRUSS_TYPES)
    n_bays = rng.randint(N_BAYS_MIN, N_BAYS_MAX)
    low_ft, high_ft = depth_window_ft(instance)
    depth_ft = rng.uniform(low_ft, high_ft)
    top_chord = rng.choice(SECTION_NAMES)
    bottom_chord = rng.choice(SECTION_NAMES)
    verticals = rng.choice(SECTION_NAMES)
    diagonals = rng.choice(SECTION_NAMES)
    return TrussDesign(
        truss_type=truss_type,
        n_bays=n_bays,
        depth_ft=depth_ft,
        top_chord=top_chord,
        bottom_chord=bottom_chord,
        diagonals=diagonals,
        verticals=verticals,
    )
