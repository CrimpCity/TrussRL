"""Build and audit the frozen training roster of calibrated v2 instances.

The training roster extends the frozen grading constants beyond the 32
calibrated instances: each training instance carries its own cost_ref_usd
computed by the exact calibration protocol (5k random designs, median of
the strictly feasible cheapest decile). The eval split is not a new
artifact — it is the 8 calibrated instances with index % 4 == 3, which
already carry frozen sweep_best records in the committed calibration
artifacts; this module loads them from there instead of copying them.

Difficulty-spread thresholds are anchored to the generator's design
distribution, with wide slack at the 512-instance roster size. If a real
build trips one, the remedy is rebuilding with a different master seed —
never a threshold edit after seeing the data.
"""

import dataclasses
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from trussRL.calibration.artifacts import instance_from_payload, instance_payload
from trussRL.calibration.sweep import InstanceCalibration, validate_positive_count
from trussRL.generator import GENERATOR_VERSION, GeneratorConfig
from trussRL.instance import TrussInstance
from trussRL.schema import TrussDesign

EVAL_SPLIT_MODULUS = 4
EVAL_SPLIT_REMAINDER = 3

COST_REF_FILENAME = "cost_ref.json"
SWEEP_BEST_FILENAME = "sweep_best.json"
TRAIN_ROSTER_FILENAME = "train_roster.json"

SPREAD_DEFL_DENOMS = (240, 360, 500)
SPREAD_DEFL_PROPORTION_TOLERANCE = 0.10
SPREAD_DEPTH_PROPORTION_MIN = 0.30
SPREAD_SPAN_LOW_FT = 125.0
SPREAD_SPAN_HIGH_FT = 155.0
SPREAD_SPAN_DISTINCT_MIN = 30
SPREAD_LOAD_LOW_KIP_PER_FT = 3.2
SPREAD_LOAD_HIGH_KIP_PER_FT = 4.8
SPREAD_LOAD_DISTINCT_MIN = 15
COST_REF_ENVELOPE_QUANTILE_LOW = 0.10
COST_REF_ENVELOPE_QUANTILE_HIGH = 0.90

DROP_REASON_COLLISION = "calibration_collision"
DROP_REASON_DUPLICATE = "duplicate"

SpreadStatus = Literal["pass", "fail"]


@dataclass(frozen=True)
class RosterConfig:
    """The full knob set for one training-roster build, echoed into the artifact.

    Deliberately not CalibrationConfig: the roster build runs no sweep_best
    search, so echoing a sweep_best_samples knob would misdescribe the run.

    Attributes:
        seed: master seed every roster stream is derived from
        n_target: exact number of training instances the artifact freezes
        n_candidates: candidate instances drawn before filtering and
            calibration
        designs_per_instance: random designs per instance for cost_ref
        generator: difficulty knobs for the candidate instances
    """

    seed: int = 1
    n_target: int = 512
    n_candidates: int = 576
    designs_per_instance: int = 5000
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)


@dataclass(frozen=True)
class EvalInstance:
    """One held-out calibrated instance with its frozen sweep_best record.

    Attributes:
        index: position of the instance in the calibration roster
        instance: the calibrated instance with cost_ref_usd applied
        sweep_best_design: the best design the frozen sweep_best search found
        sweep_best_score: the best design's regraded score
        sweep_best_cost_usd: the best design's total cost, the anchor for
            near-optimality metrics
    """

    index: int
    instance: TrussInstance
    sweep_best_design: TrussDesign
    sweep_best_score: float
    sweep_best_cost_usd: float


@dataclass(frozen=True)
class RosterCandidate:
    """One candidate training instance with its recorded seeds.

    Attributes:
        generator_index: position of the candidate in the generator stream
        instance_seed: the generator child seed that reproduces the instance
            via generate_instance
        instance: the candidate instance, cost_ref_usd unset
        sweep_seed: seed for the candidate's calibration sweep
    """

    generator_index: int
    instance_seed: int
    instance: TrussInstance
    sweep_seed: int


@dataclass(frozen=True)
class DroppedCandidate:
    """One candidate excluded from the roster, recorded in the artifact.

    Attributes:
        generator_index: position of the candidate in the generator stream
        instance_seed: the generator child seed of the dropped candidate
        instance: the dropped instance
        reason: why the candidate was dropped — "calibration_collision",
            "duplicate", or "calibration_error: ..."
    """

    generator_index: int
    instance_seed: int
    instance: TrussInstance
    reason: str


@dataclass(frozen=True)
class SpreadCheck:
    """One difficulty-spread check over the whole training roster.

    Unlike the per-instance reward-landscape gate checks, a spread check is
    compound — one gate can bound a minimum, a maximum, and a distinct
    count at once — so it records every statistic behind the verdict
    instead of a single scalar.

    Attributes:
        gate: check identifier, one of "spread_defl_denom",
            "spread_depth_variant", "spread_span", "spread_load", or
            "spread_cost_ref"
        status: "pass" or "fail"
        measurements: every statistic the verdict was computed from
        criterion: human-readable statement of what was required
        detail: which condition(s) failed, empty on pass
    """

    gate: str
    status: SpreadStatus
    measurements: Mapping[str, float]
    criterion: str
    detail: str = ""


def instance_key(instance: TrussInstance) -> tuple[object, ...]:
    """Build the identity key used for train/calibration disjointness.

    Assumptions:
        1. cost_ref_usd is excluded: it is a grading constant derived from
           the instance, not part of the problem identity.

    Args:
        instance: the instance to key

    Returns:
        tuple: span_ft, depth_limit_ft, defl_denom, and per-load-case
            parameters in case order
    """
    cases = tuple(
        (case.w_kip_per_ft, case.level, case.longitudinal_kip)
        for case in instance.load_cases
    )
    return (instance.span_ft, instance.depth_limit_ft, instance.defl_denom, cases)


def eval_split_indices(n_instances: int = 32) -> tuple[int, ...]:
    """List the calibration-roster indices held out as the eval split.

    Args:
        n_instances: size of the calibration roster; must be positive

    Returns:
        tuple: indices i in [0, n_instances) with
            i % EVAL_SPLIT_MODULUS == EVAL_SPLIT_REMAINDER

    Raises:
        ValueError: if n_instances is not positive.
    """
    validate_positive_count("n_instances", n_instances)
    return tuple(
        index
        for index in range(n_instances)
        if index % EVAL_SPLIT_MODULUS == EVAL_SPLIT_REMAINDER
    )


def derive_generator_child_seeds(roster_seed: int, count: int) -> tuple[int, ...]:
    """Reconstruct the per-instance child seeds a generator batch draws.

    Mirrors the internal draw order of generate_instances — one
    getrandbits(64) per instance from a master Random seeded with
    roster_seed — so each roster entry can record the seed that reproduces
    its instance via generate_instance alone.

    Assumptions:
        1. A test pins generate_instance(derive_generator_child_seeds(s,
           n)[i]) against generate_instances(s, n)[i], so any drift in the
           generator's draw order is caught immediately.

    Args:
        roster_seed: the batch master seed handed to generate_instances
        count: number of child seeds to reconstruct; must be nonnegative

    Returns:
        tuple: count child seeds in draw order

    Raises:
        ValueError: if count is negative.
    """
    if count < 0:
        raise ValueError(f"count must be nonnegative, got {count!r}")
    master_rng = random.Random(roster_seed)
    return tuple(master_rng.getrandbits(64) for _ in range(count))


def artifact_sha256(path: Path) -> str:
    """Hash an artifact file's exact bytes for source pinning.

    Args:
        path: the artifact file to hash

    Returns:
        str: 64-hex sha256 of the file contents
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json_payload(path: Path) -> dict[str, Any]:
    """Read a JSON artifact into a mapping, failing loudly when unusable.

    Args:
        path: the artifact file to read

    Returns:
        dict: the deserialized top-level JSON object

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(f"calibration artifact missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"artifact {path} is not a JSON object")
    return cast(dict[str, Any], payload)


def load_calibration_payloads(
    artifacts_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load cost_ref.json and sweep_best.json and prove they are one run.

    Assumptions:
        1. The two artifacts are written together by a single calibration
           run, so run_id, roster_seed, config, and every per-index
           instance payload must agree; any divergence means a manual edit
           or a partial rerun and is rejected.

    Args:
        artifacts_dir: directory holding the two calibration artifacts

    Returns:
        tuple: the cost_ref payload and the sweep_best payload

    Raises:
        FileNotFoundError: if either artifact is missing.
        ValueError: if the artifacts disagree on run_id, roster_seed,
            config, instance count, per-index instance payloads, or
            per-index cost_ref values.
    """
    cost_ref = read_json_payload(artifacts_dir / COST_REF_FILENAME)
    sweep_best = read_json_payload(artifacts_dir / SWEEP_BEST_FILENAME)
    if cost_ref["stamp"]["run_id"] != sweep_best["stamp"]["run_id"]:
        raise ValueError(
            "calibration artifacts are not from one run: run_id "
            f"{cost_ref['stamp']['run_id']!r} != {sweep_best['stamp']['run_id']!r}"
        )
    if cost_ref["roster_seed"] != sweep_best["roster_seed"]:
        raise ValueError(
            "calibration artifacts disagree on roster_seed: "
            f"{cost_ref['roster_seed']!r} != {sweep_best['roster_seed']!r}"
        )
    if cost_ref["config"] != sweep_best["config"]:
        raise ValueError("calibration artifacts disagree on the config echo")
    cost_entries = cost_ref["instances"]
    best_entries = sweep_best["instances"]
    if len(cost_entries) != len(best_entries):
        raise ValueError(
            "calibration artifacts disagree on instance count: "
            f"{len(cost_entries)} != {len(best_entries)}"
        )
    for index, (cost_entry, best_entry) in enumerate(
        zip(cost_entries, best_entries, strict=True)
    ):
        if (
            cost_entry["instance_index"] != index
            or best_entry["instance_index"] != index
        ):
            raise ValueError(f"calibration artifacts misordered at position {index}")
        if cost_entry["instance"] != best_entry["instance"]:
            raise ValueError(
                f"calibration artifacts disagree on instance {index}'s payload"
            )
        if cost_entry["cost_ref_usd"] != best_entry["cost_ref_usd"]:
            raise ValueError(
                f"calibration artifacts disagree on instance {index}'s cost_ref_usd"
            )
    return cost_ref, sweep_best


def load_eval_split(
    artifacts_dir: Path = Path("artifacts"),
) -> tuple[EvalInstance, ...]:
    """Load the held-out eval split from the frozen calibration artifacts.

    The eval split lives only in cost_ref.json and sweep_best.json — no
    copy is embedded anywhere — so a single loader is the one place the
    held-out instances and their sweep_best records are reconstructed.

    Args:
        artifacts_dir: directory holding the two calibration artifacts

    Returns:
        tuple: one EvalInstance per held-out index, in index order

    Raises:
        FileNotFoundError: if either calibration artifact is missing.
        ValueError: if the artifacts are inconsistent, or a held-out entry
            has a non-finite or non-positive cost_ref_usd or an unusable
            sweep_best record.
    """
    cost_ref, sweep_best = load_calibration_payloads(artifacts_dir)
    cost_entries = cost_ref["instances"]
    best_entries = sweep_best["instances"]
    split: list[EvalInstance] = []
    for index in eval_split_indices(len(cost_entries)):
        cost_entry = cost_entries[index]
        instance = instance_from_payload(cost_entry["instance"])
        cost_ref_usd = cost_entry["cost_ref_usd"]
        if (
            not isinstance(cost_ref_usd, (int, float))
            or isinstance(cost_ref_usd, bool)
            or not math.isfinite(cost_ref_usd)
            or cost_ref_usd <= 0.0
        ):
            raise ValueError(
                f"instance {index} has invalid cost_ref_usd: {cost_ref_usd!r}"
            )
        best = best_entries[index]["best"]
        best_cost = best["cost_total_usd"]
        if best_cost is None or not math.isfinite(best_cost):
            raise ValueError(
                f"instance {index}'s sweep_best has no usable cost_total_usd: "
                f"{best_cost!r}"
            )
        split.append(
            EvalInstance(
                index=index,
                instance=dataclasses.replace(
                    instance, cost_ref_usd=float(cost_ref_usd)
                ),
                sweep_best_design=TrussDesign.model_validate(best["design"]),
                sweep_best_score=float(best["score"]),
                sweep_best_cost_usd=float(best_cost),
            )
        )
    return tuple(split)


def select_candidates(
    candidates: Sequence[RosterCandidate],
    excluded_keys: Iterable[tuple[object, ...]],
    n_target: int,
) -> tuple[tuple[RosterCandidate, ...], tuple[DroppedCandidate, ...]]:
    """Filter the candidate stream against calibration keys and duplicates.

    Keeps candidates in draw order, dropping any whose identity key
    collides with a calibration instance or repeats an earlier survivor.
    Survivors may exceed n_target — calibration consumes them in order
    until n_target succeed.

    Args:
        candidates: the candidate stream in draw order
        excluded_keys: identity keys of every calibration instance
        n_target: number of training instances the build must produce;
            must be positive

    Returns:
        tuple: the surviving candidates and the dropped candidates, both
            in draw order

    Raises:
        ValueError: if n_target is not positive, or fewer than n_target
            candidates survive filtering.
    """
    validate_positive_count("n_target", n_target)
    excluded = set(excluded_keys)
    seen: set[tuple[object, ...]] = set()
    survivors: list[RosterCandidate] = []
    dropped: list[DroppedCandidate] = []
    for candidate in candidates:
        key = instance_key(candidate.instance)
        if key in excluded:
            reason = DROP_REASON_COLLISION
        elif key in seen:
            reason = DROP_REASON_DUPLICATE
        else:
            seen.add(key)
            survivors.append(candidate)
            continue
        dropped.append(
            DroppedCandidate(
                generator_index=candidate.generator_index,
                instance_seed=candidate.instance_seed,
                instance=candidate.instance,
                reason=reason,
            )
        )
    if len(survivors) < n_target:
        raise ValueError(
            f"candidate stream exhausted: {len(survivors)} survivor(s) after "
            f"filtering, need {n_target}"
        )
    return tuple(survivors), tuple(dropped)


def quantile(values: Sequence[float], q: float) -> float:
    """Compute a deterministic quantile by inclusive linear interpolation.

    The single quantile definition shared by the spread checks and their
    tests: for sorted values x_0..x_(n-1), the quantile at q is x
    interpolated linearly at fractional rank q * (n - 1).

    Args:
        values: the sample; must be non-empty
        q: quantile level in [0, 1]

    Returns:
        float: the interpolated quantile

    Raises:
        ValueError: if values is empty or q is outside [0, 1].
    """
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be within [0, 1], got {q!r}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = q * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def load_magnitude_kip_per_ft(instance: TrussInstance) -> float:
    """Read an instance's gravity line-load magnitude.

    Assumptions:
        1. The v2 family is single-load-case gravity-only, so the first
           case's line load is the instance's load level.

    Args:
        instance: the instance to read

    Returns:
        float: the absolute line-load magnitude, rounded to the
            generator's 6-decimal grid
    """
    return round(abs(instance.load_cases[0].w_kip_per_ft), 6)


def check_defl_denom_spread(instances: Sequence[TrussInstance]) -> SpreadCheck:
    """Check that every deflection denominator is drawn near uniformly.

    Args:
        instances: the training instances

    Returns:
        SpreadCheck: pass when each denominator in {240, 360, 500} has
            proportion within 1/3 +/- SPREAD_DEFL_PROPORTION_TOLERANCE
    """
    counts = Counter(instance.defl_denom for instance in instances)
    measurements: dict[str, float] = {}
    failed: list[str] = []
    for denom in SPREAD_DEFL_DENOMS:
        proportion = counts.get(denom, 0) / len(instances)
        measurements[f"proportion_{denom}"] = proportion
        if abs(proportion - 1 / 3) > SPREAD_DEFL_PROPORTION_TOLERANCE:
            failed.append(str(denom))
    return SpreadCheck(
        gate="spread_defl_denom",
        status="pass" if not failed else "fail",
        measurements=measurements,
        criterion=(
            "each defl_denom in {240, 360, 500} at proportion within "
            f"1/3 +/- {SPREAD_DEFL_PROPORTION_TOLERANCE}"
        ),
        detail=(
            "" if not failed else f"out of tolerance: defl_denom {', '.join(failed)}"
        ),
    )


def check_depth_variant_spread(instances: Sequence[TrussInstance]) -> SpreadCheck:
    """Check that both depth-limit variants are well represented.

    Assumptions:
        1. The v2 family draws only "none" and "generous", so any stated
           depth limit is counted as the "generous" variant.

    Args:
        instances: the training instances

    Returns:
        SpreadCheck: pass when both variants have proportion at least
            SPREAD_DEPTH_PROPORTION_MIN
    """
    n_generous = sum(1 for instance in instances if instance.depth_limit_ft is not None)
    proportion_generous = n_generous / len(instances)
    proportion_none = 1.0 - proportion_generous
    failed: list[str] = []
    if proportion_none < SPREAD_DEPTH_PROPORTION_MIN:
        failed.append("none")
    if proportion_generous < SPREAD_DEPTH_PROPORTION_MIN:
        failed.append("generous")
    return SpreadCheck(
        gate="spread_depth_variant",
        status="pass" if not failed else "fail",
        measurements={
            "proportion_none": proportion_none,
            "proportion_generous": proportion_generous,
        },
        criterion=(
            'both depth variants ("none", "generous") at proportion >= '
            f"{SPREAD_DEPTH_PROPORTION_MIN}"
        ),
        detail=("" if not failed else f"underrepresented: variant {', '.join(failed)}"),
    )


def check_span_spread(instances: Sequence[TrussInstance]) -> SpreadCheck:
    """Check that spans cover the generator family's range and grid.

    Args:
        instances: the training instances

    Returns:
        SpreadCheck: pass when the minimum span is at most
            SPREAD_SPAN_LOW_FT, the maximum at least SPREAD_SPAN_HIGH_FT,
            and at least SPREAD_SPAN_DISTINCT_MIN distinct spans appear
    """
    spans = [instance.span_ft for instance in instances]
    span_min = min(spans)
    span_max = max(spans)
    n_distinct = len({round(span, 6) for span in spans})
    failed: list[str] = []
    if span_min > SPREAD_SPAN_LOW_FT:
        failed.append(f"min {span_min:g} > {SPREAD_SPAN_LOW_FT:g}")
    if span_max < SPREAD_SPAN_HIGH_FT:
        failed.append(f"max {span_max:g} < {SPREAD_SPAN_HIGH_FT:g}")
    if n_distinct < SPREAD_SPAN_DISTINCT_MIN:
        failed.append(f"distinct {n_distinct} < {SPREAD_SPAN_DISTINCT_MIN}")
    return SpreadCheck(
        gate="spread_span",
        status="pass" if not failed else "fail",
        measurements={
            "span_min_ft": span_min,
            "span_max_ft": span_max,
            "n_distinct_spans": n_distinct,
        },
        criterion=(
            f"span min <= {SPREAD_SPAN_LOW_FT:g} ft, max >= "
            f"{SPREAD_SPAN_HIGH_FT:g} ft, >= {SPREAD_SPAN_DISTINCT_MIN} "
            "distinct spans"
        ),
        detail="" if not failed else "; ".join(failed),
    )


def check_load_spread(instances: Sequence[TrussInstance]) -> SpreadCheck:
    """Check that load magnitudes cover the generator family's range.

    Args:
        instances: the training instances

    Returns:
        SpreadCheck: pass when the minimum magnitude is at most
            SPREAD_LOAD_LOW_KIP_PER_FT, the maximum at least
            SPREAD_LOAD_HIGH_KIP_PER_FT, and at least
            SPREAD_LOAD_DISTINCT_MIN distinct levels appear
    """
    magnitudes = [load_magnitude_kip_per_ft(instance) for instance in instances]
    load_min = min(magnitudes)
    load_max = max(magnitudes)
    n_distinct = len(set(magnitudes))
    failed: list[str] = []
    if load_min > SPREAD_LOAD_LOW_KIP_PER_FT:
        failed.append(f"min {load_min:g} > {SPREAD_LOAD_LOW_KIP_PER_FT:g}")
    if load_max < SPREAD_LOAD_HIGH_KIP_PER_FT:
        failed.append(f"max {load_max:g} < {SPREAD_LOAD_HIGH_KIP_PER_FT:g}")
    if n_distinct < SPREAD_LOAD_DISTINCT_MIN:
        failed.append(f"distinct {n_distinct} < {SPREAD_LOAD_DISTINCT_MIN}")
    return SpreadCheck(
        gate="spread_load",
        status="pass" if not failed else "fail",
        measurements={
            "load_min_kip_per_ft": load_min,
            "load_max_kip_per_ft": load_max,
            "n_distinct_loads": n_distinct,
        },
        criterion=(
            f"load magnitude min <= {SPREAD_LOAD_LOW_KIP_PER_FT:g}, max >= "
            f"{SPREAD_LOAD_HIGH_KIP_PER_FT:g} kip/ft, >= "
            f"{SPREAD_LOAD_DISTINCT_MIN} distinct levels"
        ),
        detail="" if not failed else "; ".join(failed),
    )


def check_cost_ref_spread(
    train_cost_refs: Sequence[float], calibration_cost_refs: Sequence[float]
) -> SpreadCheck:
    """Check the training cost_ref envelope against the calibration roster's.

    Calibration enters the spread checks only here, through robust envelope
    statistics: the training range must cover the calibration roster's
    q10-q90 interval, and the training median must fall inside the
    calibration [min, max].

    Args:
        train_cost_refs: one cost_ref per training instance
        calibration_cost_refs: one cost_ref per calibration instance

    Returns:
        SpreadCheck: pass when every training value is finite and
            positive and the envelope conditions hold
    """
    n_invalid = sum(
        1 for value in train_cost_refs if not math.isfinite(value) or value <= 0.0
    )
    valid = [value for value in train_cost_refs if math.isfinite(value) and value > 0.0]
    calibration_q_low = quantile(calibration_cost_refs, COST_REF_ENVELOPE_QUANTILE_LOW)
    calibration_q_high = quantile(
        calibration_cost_refs, COST_REF_ENVELOPE_QUANTILE_HIGH
    )
    calibration_min = min(calibration_cost_refs)
    calibration_max = max(calibration_cost_refs)
    measurements: dict[str, float] = {
        "n_invalid": n_invalid,
        "calibration_q10": calibration_q_low,
        "calibration_q90": calibration_q_high,
        "calibration_min": calibration_min,
        "calibration_max": calibration_max,
    }
    failed: list[str] = []
    if n_invalid:
        failed.append(f"{n_invalid} non-finite or non-positive value(s)")
    if valid:
        train_min = min(valid)
        train_max = max(valid)
        train_median = quantile(valid, 0.5)
        measurements["train_min"] = train_min
        measurements["train_max"] = train_max
        measurements["train_median"] = train_median
        if train_min > calibration_q_low:
            failed.append(
                f"train min {train_min:.0f} > calibration q10 {calibration_q_low:.0f}"
            )
        if train_max < calibration_q_high:
            failed.append(
                f"train max {train_max:.0f} < calibration q90 {calibration_q_high:.0f}"
            )
        if not calibration_min <= train_median <= calibration_max:
            failed.append(
                f"train median {train_median:.0f} outside calibration "
                f"[{calibration_min:.0f}, {calibration_max:.0f}]"
            )
    else:
        failed.append("no valid train cost_ref values")
    return SpreadCheck(
        gate="spread_cost_ref",
        status="pass" if not failed else "fail",
        measurements=measurements,
        criterion=(
            "every train cost_ref finite and > 0; train range covers "
            "calibration [q10, q90]; train median within calibration "
            "[min, max]"
        ),
        detail="" if not failed else "; ".join(failed),
    )


def check_difficulty_spread(
    train_instances: Sequence[TrussInstance],
    train_cost_refs: Sequence[float],
    calibration_cost_refs: Sequence[float],
) -> tuple[SpreadCheck, ...]:
    """Run every difficulty-spread check over the training roster.

    Categorical and range checks are anchored to the generator's uniform
    design distribution rather than the 32-instance calibration roster's
    noisy measured proportions; calibration enters only through the
    cost_ref envelope check.

    Args:
        train_instances: the training instances
        train_cost_refs: per-instance cost_ref values, index-aligned with
            train_instances
        calibration_cost_refs: the calibration roster's cost_ref values

    Returns:
        tuple: one SpreadCheck per gate, in fixed order

    Raises:
        ValueError: if train_instances is empty, train_cost_refs is not
            index-aligned with it, or calibration_cost_refs is empty.
    """
    if not train_instances:
        raise ValueError("train_instances must be non-empty")
    if len(train_cost_refs) != len(train_instances):
        raise ValueError(
            "train_cost_refs must be index-aligned with train_instances, got "
            f"{len(train_cost_refs)} != {len(train_instances)}"
        )
    if not calibration_cost_refs:
        raise ValueError("calibration_cost_refs must be non-empty")
    return (
        check_defl_denom_spread(train_instances),
        check_depth_variant_spread(train_instances),
        check_span_spread(train_instances),
        check_load_spread(train_instances),
        check_cost_ref_spread(train_cost_refs, calibration_cost_refs),
    )


def config_echo_payload(config: RosterConfig) -> dict[str, object]:
    """Serialize the roster config for the artifact config echo.

    Args:
        config: the run's roster config

    Returns:
        dict: every config field, the generator config nested, plus a
            generator_version key
    """
    payload: dict[str, object] = dataclasses.asdict(config)
    payload["generator_version"] = GENERATOR_VERSION
    return payload


def spread_check_payload(check: SpreadCheck) -> dict[str, object]:
    """Serialize one spread check for artifact embedding.

    Args:
        check: the spread check to serialize

    Returns:
        dict: gate, status, measurements, criterion, and detail
    """
    return {
        "gate": check.gate,
        "status": check.status,
        "measurements": dict(check.measurements),
        "criterion": check.criterion,
        "detail": check.detail,
    }


def train_roster_payload(
    stamp: Mapping[str, object],
    config: RosterConfig,
    roster_seed: int,
    sources: Mapping[str, object],
    calibrations: Sequence[InstanceCalibration],
    candidates: Sequence[RosterCandidate],
    dropped: Sequence[DroppedCandidate],
    spread_checks: Sequence[SpreadCheck],
    held_out_indices: Sequence[int],
) -> dict[str, object]:
    """Assemble the frozen train_roster artifact payload.

    Per-instance entries mirror cost_ref.json's, plus generator_index and
    instance_seed, so instance_from_payload and downstream consumers work
    identically on both artifacts.

    Args:
        stamp: the run's provenance stamp
        config: the run's roster config
        roster_seed: the derived generator batch seed
        sources: pinned filename, run_id, and sha256 of both calibration
            artifacts
        calibrations: one successful calibration per accepted candidate
        candidates: the accepted candidates, index-aligned with
            calibrations
        dropped: every dropped candidate with its reason
        spread_checks: the difficulty-spread verdicts
        held_out_indices: calibration-roster indices of the eval split

    Returns:
        dict: the full artifact payload
    """
    entries: list[dict[str, object]] = []
    for index, (candidate, calibration) in enumerate(
        zip(candidates, calibrations, strict=True)
    ):
        entries.append(
            {
                "instance_index": index,
                "generator_index": candidate.generator_index,
                "instance_seed": candidate.instance_seed,
                "instance": instance_payload(calibration.instance),
                "sweep_seed": calibration.seed,
                "n_samples": calibration.n_samples,
                "n_rung3": calibration.n_rung3,
                "n_strictly_feasible": calibration.n_strictly_feasible,
                "cost_ref_usd": calibration.cost_ref_usd,
            }
        )
    return {
        "stamp": dict(stamp),
        "sources": dict(sources),
        "config": config_echo_payload(config),
        "roster_seed": roster_seed,
        "held_out": {"indices": list(held_out_indices)},
        "dropped": [
            {
                "generator_index": item.generator_index,
                "instance_seed": item.instance_seed,
                "reason": item.reason,
                "instance": instance_payload(item.instance),
            }
            for item in dropped
        ],
        "spread": {
            "all_passed": all(check.status == "pass" for check in spread_checks),
            "checks": [spread_check_payload(check) for check in spread_checks],
        },
        "instances": entries,
    }


def load_train_roster(
    artifacts_dir: Path = Path("artifacts"),
) -> tuple[TrussInstance, ...]:
    """Load the frozen training roster with cost_ref_usd applied.

    The dataset entry point for training: strict on shape and grading
    constants, but it does not recompute spread statistics — proving the
    frozen artifact's spread is the acceptance tests' job, not a dataset
    loader's.

    Args:
        artifacts_dir: directory holding train_roster.json

    Returns:
        tuple: every training instance in artifact order, each with its
            frozen cost_ref_usd applied

    Raises:
        FileNotFoundError: if the artifact is missing.
        ValueError: if the top-level shape is wrong, the instance count
            does not match the config echo, an entry is misindexed or
            malformed, a cost_ref_usd is non-finite or non-positive, or
            two entries share an identity key.
    """
    payload = read_json_payload(artifacts_dir / TRAIN_ROSTER_FILENAME)
    expected_keys = {
        "stamp",
        "sources",
        "config",
        "roster_seed",
        "held_out",
        "dropped",
        "spread",
        "instances",
    }
    if set(payload) != expected_keys:
        extra = sorted(set(payload) - expected_keys)
        missing = sorted(expected_keys - set(payload))
        raise ValueError(
            f"train roster keys do not match: extra={extra}, missing={missing}"
        )
    n_target = payload["config"]["n_target"]
    entries = payload["instances"]
    if len(entries) != n_target:
        raise ValueError(
            f"train roster has {len(entries)} instances, config echo says {n_target}"
        )
    seen: set[tuple[object, ...]] = set()
    instances: list[TrussInstance] = []
    for position, entry in enumerate(entries):
        if entry["instance_index"] != position:
            raise ValueError(
                f"train roster entry at position {position} has "
                f"instance_index {entry['instance_index']!r}"
            )
        instance = instance_from_payload(entry["instance"])
        cost_ref_usd = entry["cost_ref_usd"]
        if (
            not isinstance(cost_ref_usd, (int, float))
            or isinstance(cost_ref_usd, bool)
            or not math.isfinite(cost_ref_usd)
            or cost_ref_usd <= 0.0
        ):
            raise ValueError(
                f"train roster entry {position} has invalid cost_ref_usd: "
                f"{cost_ref_usd!r}"
            )
        key = instance_key(instance)
        if key in seen:
            raise ValueError(
                f"train roster entry {position} duplicates another entry's identity key"
            )
        seen.add(key)
        instances.append(
            dataclasses.replace(instance, cost_ref_usd=float(cost_ref_usd))
        )
    return tuple(instances)
