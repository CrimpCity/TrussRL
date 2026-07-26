"""Unit tests for the training-roster core module.

No solver runs: these pin the seed derivations, the eval-split contract
against the committed calibration artifacts, candidate filtering, and the
difficulty-spread checks at their boundaries. The real 512x5k build is
covered by the acceptance tests against the frozen artifact.
"""

import json
import math
from pathlib import Path

import pytest

from trussRL.calibration.roster import (DROP_REASON_COLLISION,
                                        DROP_REASON_DUPLICATE, RosterCandidate,
                                        check_difficulty_spread,
                                        derive_generator_child_seeds,
                                        eval_split_indices, instance_key,
                                        load_eval_split, select_candidates)
from trussRL.calibration.sweep import derive_calibration_seeds
from trussRL.generator import (depth_limit_ft, generate_instance,
                               generate_instances)
from trussRL.instance import TrussInstance
from trussRL.loads import LoadCase

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

SEED_ONE_ROSTER_SEED = 10499958131665514997
SEED_ZERO_ROSTER_SEED = 7106521602475165645


def make_instance(
    span_ft: float,
    w_kip_per_ft: float = 4.0,
    depth_limit: float | None = None,
    defl_denom: int = 360,
    cost_ref_usd: float | None = None,
) -> TrussInstance:
    """Build a minimal single-load-case instance for key and spread tests.

    Args:
        span_ft: truss span in feet
        w_kip_per_ft: gravity load magnitude in kips per foot
        depth_limit: stated depth limit in feet, or None
        defl_denom: deflection-limit denominator
        cost_ref_usd: grading constant, or None

    Returns:
        TrussInstance: the assembled instance
    """
    return TrussInstance(
        span_ft=span_ft,
        load_cases=(LoadCase(w_kip_per_ft=-abs(w_kip_per_ft)),),
        depth_limit_ft=depth_limit,
        defl_denom=defl_denom,
        cost_ref_usd=cost_ref_usd,
    )


def make_candidate(generator_index: int, instance: TrussInstance) -> RosterCandidate:
    """Wrap an instance as a candidate with synthetic seeds.

    Args:
        generator_index: position in the synthetic stream
        instance: the candidate instance

    Returns:
        RosterCandidate: the assembled candidate
    """
    return RosterCandidate(
        generator_index=generator_index,
        instance_seed=1000 + generator_index,
        instance=instance,
        sweep_seed=2000 + generator_index,
    )


def healthy_train_instances() -> list[TrussInstance]:
    """Build a synthetic roster matching the generator's design distribution.

    123 instances cycling through all 41 spans, all 21 load levels, the
    three deflection denominators in exact thirds, and alternating depth
    variants — every spread gate passes with margin.

    Args: None

    Returns:
        list: the synthetic training instances
    """
    spans = [120.0 + step for step in range(41)]
    loads = [round(3.0 + 0.1 * step, 6) for step in range(21)]
    denoms = (240, 360, 500)
    instances = []
    for index in range(123):
        span = spans[index % 41]
        depth_limit = None if index % 2 == 0 else depth_limit_ft("generous", span)
        instances.append(
            make_instance(
                span_ft=span,
                w_kip_per_ft=loads[index % 21],
                depth_limit=depth_limit,
                defl_denom=denoms[index % 3],
            )
        )
    return instances


def healthy_cost_refs(count: int) -> list[float]:
    """Build training cost_refs spanning past the calibration envelope.

    Args:
        count: number of values to produce

    Returns:
        list: values spread uniformly over [55000, 95000]
    """
    return [55000.0 + 40000.0 * index / (count - 1) for index in range(count)]


CALIBRATION_COST_REFS = [60000.0 + 1000.0 * index for index in range(32)]


def check_by_gate(checks: tuple, gate: str):
    """Pick one named check out of a spread-check tuple.

    Args:
        checks: the spread checks
        gate: the gate identifier to find

    Returns:
        SpreadCheck: the matching check
    """
    matches = [check for check in checks if check.gate == gate]
    assert len(matches) == 1, f"expected exactly one {gate} check"
    return matches[0]


def test_eval_split_indices_pins_held_out_positions() -> None:
    assert eval_split_indices() == (3, 7, 11, 15, 19, 23, 27, 31)
    assert eval_split_indices(32) == (3, 7, 11, 15, 19, 23, 27, 31)
    assert eval_split_indices(8) == (3, 7)
    with pytest.raises(ValueError):
        eval_split_indices(0)


def test_seed_one_derivation_is_pinned() -> None:
    assert derive_calibration_seeds(1, 1).roster_seed == SEED_ONE_ROSTER_SEED
    assert derive_calibration_seeds(0, 1).roster_seed == SEED_ZERO_ROSTER_SEED


def test_derive_generator_child_seeds_matches_generate_instances() -> None:
    roster_seed = 424242
    count = 8
    child_seeds = derive_generator_child_seeds(roster_seed, count)
    batch = generate_instances(roster_seed, count)
    assert len(child_seeds) == count
    for child_seed, expected in zip(child_seeds, batch, strict=True):
        assert generate_instance(child_seed) == expected


def test_load_eval_split_returns_eight_calibrated_instances() -> None:
    split = load_eval_split(ARTIFACTS_DIR)
    assert tuple(item.index for item in split) == (3, 7, 11, 15, 19, 23, 27, 31)
    for item in split:
        assert item.instance.cost_ref_usd is not None
        assert item.instance.cost_ref_usd > 0.0
        assert math.isfinite(item.sweep_best_score)
        assert item.sweep_best_cost_usd > 0.0
        assert item.sweep_best_design.n_bays >= 1


def write_doctored_artifacts(tmp_path: Path, target: str, doctor) -> Path:
    """Copy the committed calibration artifacts and doctor one of them.

    Args:
        tmp_path: destination directory
        target: which artifact to doctor, "cost_ref.json" or
            "sweep_best.json"
        doctor: mutation applied to the target payload in place

    Returns:
        Path: the directory holding the doctored pair
    """
    for name in ("cost_ref.json", "sweep_best.json"):
        payload = json.loads((ARTIFACTS_DIR / name).read_text())
        if name == target:
            doctor(payload)
        (tmp_path / name).write_text(json.dumps(payload))
    return tmp_path


def doctor_run_id(payload: dict) -> None:
    """Give the payload a mismatching run_id.

    Args:
        payload: the artifact payload to mutate

    Returns: None
    """
    payload["stamp"]["run_id"] = "19700101T000000Z_0000000"


def doctor_roster_seed(payload: dict) -> None:
    """Give the payload a mismatching roster_seed.

    Args:
        payload: the artifact payload to mutate

    Returns: None
    """
    payload["roster_seed"] += 1


def doctor_generator_config(payload: dict) -> None:
    """Give the payload a mismatching generator config.

    Args:
        payload: the artifact payload to mutate

    Returns: None
    """
    payload["config"]["generator"]["span_min_ft"] = 100.0


def doctor_instance_payload(payload: dict) -> None:
    """Give one held-out instance a mismatching span.

    Args:
        payload: the artifact payload to mutate

    Returns: None
    """
    payload["instances"][3]["instance"]["span_ft"] += 1.0


def doctor_cost_ref_value(payload: dict) -> None:
    """Give one held-out instance a mismatching cost_ref_usd.

    Args:
        payload: the artifact payload to mutate

    Returns: None
    """
    payload["instances"][3]["cost_ref_usd"] += 1.0


@pytest.mark.parametrize(
    ("target", "doctor"),
    [
        ("sweep_best.json", doctor_run_id),
        ("cost_ref.json", doctor_roster_seed),
        ("cost_ref.json", doctor_generator_config),
        ("sweep_best.json", doctor_instance_payload),
        ("sweep_best.json", doctor_cost_ref_value),
    ],
)
def test_load_eval_split_rejects_mismatched_artifacts(
    tmp_path: Path, target: str, doctor
) -> None:
    write_doctored_artifacts(tmp_path, target, doctor)
    with pytest.raises(ValueError):
        load_eval_split(tmp_path)


def test_load_eval_split_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_split(tmp_path)


def test_instance_key_ignores_cost_ref_and_separates_params() -> None:
    base = make_instance(140.0, 4.0, 22.0, 360)
    with_cost_ref = make_instance(140.0, 4.0, 22.0, 360, cost_ref_usd=75000.0)
    assert instance_key(base) == instance_key(with_cost_ref)
    assert instance_key(base) != instance_key(make_instance(141.0, 4.0, 22.0, 360))
    assert instance_key(base) != instance_key(make_instance(140.0, 4.1, 22.0, 360))
    assert instance_key(base) != instance_key(make_instance(140.0, 4.0, None, 360))
    assert instance_key(base) != instance_key(make_instance(140.0, 4.0, 22.0, 500))


def test_select_candidates_filters_collisions_and_duplicates() -> None:
    kept_a = make_candidate(0, make_instance(120.0))
    colliding = make_candidate(1, make_instance(150.0))
    kept_b = make_candidate(2, make_instance(130.0))
    duplicate = make_candidate(3, make_instance(120.0))
    kept_c = make_candidate(4, make_instance(140.0))
    excluded_keys = {instance_key(make_instance(150.0))}
    survivors, dropped = select_candidates(
        (kept_a, colliding, kept_b, duplicate, kept_c), excluded_keys, 3
    )
    assert survivors == (kept_a, kept_b, kept_c)
    assert [item.generator_index for item in dropped] == [1, 3]
    assert dropped[0].reason == DROP_REASON_COLLISION
    assert dropped[1].reason == DROP_REASON_DUPLICATE
    assert dropped[0].instance_seed == colliding.instance_seed
    with pytest.raises(ValueError, match="exhausted"):
        select_candidates(
            (kept_a, colliding, kept_b, duplicate, kept_c), excluded_keys, 4
        )


def test_check_difficulty_spread_passes_healthy_roster() -> None:
    instances = healthy_train_instances()
    checks = check_difficulty_spread(
        instances, healthy_cost_refs(len(instances)), CALIBRATION_COST_REFS
    )
    assert [check.gate for check in checks] == [
        "spread_defl_denom",
        "spread_depth_variant",
        "spread_span",
        "spread_load",
        "spread_cost_ref",
    ]
    for check in checks:
        assert check.status == "pass", f"{check.gate}: {check.detail}"
        assert check.detail == ""


def test_check_difficulty_spread_fails_collapsed_defl_denom() -> None:
    instances = [
        make_instance(inst.span_ft, abs(inst.load_cases[0].w_kip_per_ft),
                      inst.depth_limit_ft, 360)
        for inst in healthy_train_instances()
    ]
    checks = check_difficulty_spread(
        instances, healthy_cost_refs(len(instances)), CALIBRATION_COST_REFS
    )
    check = check_by_gate(checks, "spread_defl_denom")
    assert check.status == "fail"
    assert check.measurements["proportion_360"] == 1.0
    assert check.measurements["proportion_240"] == 0.0
    assert check_by_gate(checks, "spread_span").status == "pass"


def test_check_difficulty_spread_fails_missing_depth_variant() -> None:
    instances = [
        make_instance(inst.span_ft, abs(inst.load_cases[0].w_kip_per_ft),
                      None, inst.defl_denom)
        for inst in healthy_train_instances()
    ]
    checks = check_difficulty_spread(
        instances, healthy_cost_refs(len(instances)), CALIBRATION_COST_REFS
    )
    check = check_by_gate(checks, "spread_depth_variant")
    assert check.status == "fail"
    assert check.measurements["proportion_generous"] == 0.0
    assert "generous" in check.detail


def test_check_difficulty_spread_fails_narrow_span() -> None:
    instances = [
        make_instance(135.0 + (index % 6), 3.0 + 0.1 * (index % 21),
                      None if index % 2 == 0 else 23.5, (240, 360, 500)[index % 3])
        for index in range(123)
    ]
    checks = check_difficulty_spread(
        instances, healthy_cost_refs(len(instances)), CALIBRATION_COST_REFS
    )
    check = check_by_gate(checks, "spread_span")
    assert check.status == "fail"
    assert check.measurements["span_min_ft"] == 135.0
    assert check.measurements["n_distinct_spans"] == 6


def test_check_difficulty_spread_fails_too_few_load_levels() -> None:
    instances = [
        make_instance(inst.span_ft, 4.0, inst.depth_limit_ft, inst.defl_denom)
        for inst in healthy_train_instances()
    ]
    checks = check_difficulty_spread(
        instances, healthy_cost_refs(len(instances)), CALIBRATION_COST_REFS
    )
    check = check_by_gate(checks, "spread_load")
    assert check.status == "fail"
    assert check.measurements["n_distinct_loads"] == 1


def test_check_difficulty_spread_fails_non_positive_cost_ref() -> None:
    instances = healthy_train_instances()
    cost_refs = healthy_cost_refs(len(instances))
    cost_refs[0] = 0.0
    checks = check_difficulty_spread(instances, cost_refs, CALIBRATION_COST_REFS)
    check = check_by_gate(checks, "spread_cost_ref")
    assert check.status == "fail"
    assert check.measurements["n_invalid"] == 1


def test_check_difficulty_spread_fails_shifted_cost_ref_envelope() -> None:
    instances = healthy_train_instances()
    cost_refs = [75000.0 + index for index in range(len(instances))]
    checks = check_difficulty_spread(instances, cost_refs, CALIBRATION_COST_REFS)
    check = check_by_gate(checks, "spread_cost_ref")
    assert check.status == "fail"
    assert check.measurements["train_min"] == 75000.0
    assert "q10" in check.detail
