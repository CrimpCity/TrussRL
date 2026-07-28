"""Decide whether cold-start SFT is needed before GRPO.

Samples base-model rollouts over the held-out eval split, scores every
completion through the verifier, and writes a provenance-stamped artifact
whose verdict records whether cold-start SFT is recommended, whether the
completion cap must be raised, whether within-group reward variance is
present for every prompt, and the baseline policy entropy later
entropy-collapse thresholds are anchored to. The gate is informational:
its verdict routes the next step and always exits 0 once written; exit 1
is reserved for operational failures. This module must import without the
training extra installed — every heavy import lives behind load_backend().

Retained completion texts are audit material for the truncation and
parse diagnostics, never SFT data: their prompts are the held-out eval
split, and training on them would contaminate the final evaluation.
"""

import hashlib
import statistics
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from trussRL.calibration.artifacts import (
    breakdown_payload,
    instance_payload,
    provenance_stamp,
    write_json,
)
from trussRL.calibration.roster import (
    COST_REF_FILENAME,
    SWEEP_BEST_FILENAME,
    EvalInstance,
    artifact_sha256,
    load_eval_split,
    quantile,
)
from trussRL.prompts import render_prompt
from trussRL.reward import RewardBreakdown
from trussRL.schema import FENCED_BLOCK_PATTERN
from trussRL.training.config import TrainingConfig
from trussRL.training.model_ids import MODEL_ID, MODEL_REVISION
from trussRL.training.rewards import score_batch

app = typer.Typer(add_completion=False)
console = Console()

GATE_DIR = "rollout_gate"
SFT_RUNG3_THRESHOLD = 0.10
STRUCTURED_TRUNCATION_THRESHOLD = 0.05
BASE_COMPLETION_CAP = 256
RAISED_COMPLETION_CAP = 512
CANONICAL_ROLLOUTS_PER_PROMPT = 64
MPS_GROUP_BATCH_SIZE = 4
CUDA_GROUP_BATCH_SIZE = 8

RUN_MODE_CANONICAL = "canonical"
RUN_MODE_SMOKE = "smoke"

CATEGORY_PARSEABLE = "cap_hit_parseable"
CATEGORY_STRUCTURED = "cap_hit_structured_parse_failure"
CATEGORY_UNSTRUCTURED = "cap_hit_unstructured"

ENTROPY_METHOD = (
    "token-weighted mean over all valid tokens of the fp32 Shannon entropy "
    "of softmax(logits / temperature) over the full vocabulary, from "
    "pre-nucleus generation-step logits; valid tokens run up to and "
    "including each completion's first EOS"
)


@dataclass(frozen=True)
class PromptRollouts:
    """One prompt's sampled rollouts with their scores and diagnostics.

    Attributes:
        index: the instance's position in the calibration roster
        sweep_best_cost_usd: the frozen sweep-best cost anchoring cost gaps
        sweep_best_score: the frozen sweep-best regraded score
        prompt_sha256: sha256 of the exact rendered prompt text
        texts: decoded completion texts in rollout order
        breakdowns: one verifier breakdown per completion
        token_counts: valid tokens per completion, first EOS included
        truncated: per-completion cap-truncation flags
        entropy_sums: per-completion entropy sums in nats
        entropy_token_counts: per-completion valid-token counts behind the
            entropy sums
        new_tokens: total valid tokens generated for this prompt
        elapsed_s: summed generate wall time for this prompt
    """

    index: int
    sweep_best_cost_usd: float
    sweep_best_score: float
    prompt_sha256: str
    texts: tuple[str, ...]
    breakdowns: tuple[RewardBreakdown, ...]
    token_counts: tuple[int, ...]
    truncated: tuple[bool, ...]
    entropy_sums: tuple[float, ...]
    entropy_token_counts: tuple[int, ...]
    new_tokens: int
    elapsed_s: float


def load_backend() -> types.ModuleType:
    """Import the heavy rollout-gate backend, failing with a clear remedy.

    Assumptions:
        1. Kept as a module-level function so tests monkeypatch it with a
           stub namespace instead of installing torch.

    Args:
        None

    Returns:
        ModuleType: the trussRL.training.rollout_gate module

    Raises:
        ImportError: when the training extra is not installed.
    """
    try:
        import trussRL.training.rollout_gate as backend
    except ImportError as error:
        raise ImportError(
            "training dependencies are not installed; "
            "install with: uv sync --extra training"
        ) from error
    return backend


def resolve_generation_device(
    device: str, environment: dict[str, object]
) -> str | None:
    """Pick the generation device, refusing CPU-only environments.

    Args:
        device: requested device — auto, cuda, or mps
        environment: the backend's environment report

    Returns:
        str: "cuda" or "mps", or None when the request cannot be satisfied
            without falling back to CPU
    """
    cuda_available = bool(environment.get("cuda_available"))
    mps_available = bool(environment.get("mps_available"))
    if device == "cuda":
        return "cuda" if cuda_available else None
    if device == "mps":
        return "mps" if mps_available else None
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return None


def default_group_batch_size(device: str) -> int:
    """Pick the per-device default rollouts per generate call.

    Assumptions:
        1. The fp32-head logits history held by generate is the memory
           driver, so the MPS default is halved relative to CUDA.

    Args:
        device: the resolved generation device

    Returns:
        int: the default generate batch size for the device
    """
    return CUDA_GROUP_BATCH_SIZE if device == "cuda" else MPS_GROUP_BATCH_SIZE


def sampling_policy(config: TrainingConfig) -> dict[str, object]:
    """Build the complete explicit sampling policy for every generate call.

    Assumptions:
        1. Every knob is stated even at its library default so neither the
           checkpoint's generation_config nor a transformers default can
           silently shape the distribution.

    Args:
        config: the training config supplying temperature and top_p

    Returns:
        dict: do_sample, temperature, top_p, top_k, min_p, and
            repetition_penalty
    """
    return {
        "do_sample": True,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": 0,
        "min_p": None,
        "repetition_penalty": 1.0,
    }


def collect_prompt_rollouts(
    backend: Any,
    sampler: dict[str, object],
    eval_instance: EvalInstance,
    rollouts_per_prompt: int,
    group_batch_size: int,
    max_new_tokens: int,
    sampling: dict[str, object],
) -> PromptRollouts:
    """Sample and score every rollout for one held-out prompt.

    Args:
        backend: the loaded rollout-gate backend module
        sampler: the backend's loaded sampler
        eval_instance: the held-out instance with its sweep-best record
        rollouts_per_prompt: total completions to sample
        group_batch_size: completions per generate call
        max_new_tokens: completion token cap
        sampling: the explicit sampling policy

    Returns:
        PromptRollouts: the prompt's completions, scores, and diagnostics
    """
    prompt = render_prompt(eval_instance.instance)
    result = backend.sample_group(
        sampler, prompt, rollouts_per_prompt, group_batch_size, max_new_tokens, sampling
    )
    texts = tuple(cast(list[str], result["texts"]))
    breakdowns = score_batch([eval_instance.instance] * len(texts), texts)
    return PromptRollouts(
        index=eval_instance.index,
        sweep_best_cost_usd=eval_instance.sweep_best_cost_usd,
        sweep_best_score=eval_instance.sweep_best_score,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        texts=texts,
        breakdowns=breakdowns,
        token_counts=tuple(cast(list[int], result["token_counts"])),
        truncated=tuple(cast(list[bool], result["truncated"])),
        entropy_sums=tuple(cast(list[float], result["entropy_sums"])),
        entropy_token_counts=tuple(cast(list[int], result["entropy_token_counts"])),
        new_tokens=cast(int, result["new_tokens"]),
        elapsed_s=cast(float, result["elapsed_s"]),
    )


def ladder_rates(breakdowns: Sequence[RewardBreakdown]) -> dict[str, object]:
    """Compute the reward-ladder rates over a set of breakdowns.

    Assumptions:
        1. feasibility_rate is over all completions, so one feasible
           design among many failures reads as rare, not as certain; the
           rung-3-conditioned rate is kept under its own explicit name.

    Args:
        breakdowns: the verifier breakdowns to summarize; must be non-empty

    Returns:
        dict: parse_rate (rung >= 1), drc_pass_rate (rung >= 2),
            rung3_rate (rung == 3), feasibility_rate (fraction of all
            completions with rung == 3 and feasible == 1.0), and
            feasibility_given_rung3 (the same count over rung-3
            completions only, None when no completion reached rung 3)

    Raises:
        ValueError: if breakdowns is empty.
    """
    if not breakdowns:
        raise ValueError("ladder_rates requires at least one breakdown")
    n = len(breakdowns)
    rung3 = [breakdown for breakdown in breakdowns if breakdown.rung == 3]
    feasible_count = sum(1 for breakdown in rung3 if breakdown.feasible == 1.0)
    return {
        "parse_rate": sum(1 for breakdown in breakdowns if breakdown.rung >= 1) / n,
        "drc_pass_rate": sum(1 for breakdown in breakdowns if breakdown.rung >= 2) / n,
        "rung3_rate": len(rung3) / n,
        "feasibility_rate": feasible_count / n,
        "feasibility_given_rung3": (feasible_count / len(rung3) if rung3 else None),
    }


def has_unclosed_trailing_object(text: str) -> bool:
    """Check whether text opens a top-level JSON object it never closes.

    Assumptions:
        1. Inside an object span, double-quoted strings are tracked with
           escape handling, so a brace inside a string value cannot
           distort the depth; outside any object, quotes are prose and
           ignored.
        2. A completely emitted but malformed object balances its braces
           and is not flagged — a malformed design is a parsing problem,
           not evidence the token cap cut the design off.
        3. The unclosed span must contain a double quote, since object
           keys are quoted; a lone prose brace does not qualify.

    Args:
        text: the completion text

    Returns:
        bool: True when a top-level object opens, the text ends before
            it closes, and the open span looks like object content
    """
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for position, char in enumerate(text):
        if depth > 0 and in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and depth > 0:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = position
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                start = None
    return start is not None and '"' in text[start:]


def truncation_evidence(text: str) -> bool:
    """Check whether text ends in a design the token cap plausibly cut off.

    Mirrors the parser's decision order: complete fenced blocks are
    authoritative to the parser, so malformed or incomplete content
    inside a closed fence is a parse problem, never cap evidence. Only
    an unterminated final fence, or fence-free text on the parser's
    bare-object fallback path, can show a cut-off design.

    Assumptions:
        1. Fence-free prose that opens a double quote followed by "{"
           and never closes the quote is syntactically indistinguishable
           from a cut-off design start and stays flagged; the failure
           direction is a conservative cap raise, never corrupted
           metrics.

    Args:
        text: the completion text

    Returns:
        bool: True when the text's unterminated final fence, or the
            whole fence-free text, opens a top-level object it never
            closes
    """
    remainder = FENCED_BLOCK_PATTERN.sub("", text)
    if remainder.count("```") % 2 == 1:
        tail = remainder[remainder.rfind("```") + 3 :]
        return has_unclosed_trailing_object(tail)
    if "```" not in text:
        return has_unclosed_trailing_object(text)
    return False


def truncation_category(truncated: bool, rung: int, text: str) -> str | None:
    """Classify one completion's cap truncation into exclusive categories.

    Assumptions:
        1. A truncated completion that still parsed completed its design,
           so the cap is not implicated; only truncated parse failures
           count toward raising the completion cap.
        2. Cut-off-design evidence follows the parser's decision order:
           a design inside a closed fence was fully emitted, so only an
           unterminated fence or fence-free text can carry an opened
           top-level object the cap cut off. A quoted fence marker, a
           lone prose brace, or a malformed-but-complete fenced payload
           is not evidence.

    Args:
        truncated: whether the completion hit the token cap with no EOS
        rung: the completion's reward-ladder rung
        text: the completion text

    Returns:
        str: the category for a truncated completion, or None when the
            completion was not truncated
    """
    if not truncated:
        return None
    if rung >= 1:
        return CATEGORY_PARSEABLE
    if truncation_evidence(text):
        return CATEGORY_STRUCTURED
    return CATEGORY_UNSTRUCTURED


def group_reward_spread(
    scores: Sequence[float], group_size: int
) -> tuple[dict[str, float], ...]:
    """Compute population reward spread per consecutive rollout group.

    A rollout-diversity diagnostic, not the trainer's normalization: the
    trainer uses Bessel-corrected std and batch-wide scaling, while this
    records the plain population spread of each prompt-local group.

    Args:
        scores: per-completion final scores in rollout order
        group_size: rollouts per group; scores must divide evenly into it

    Returns:
        tuple: one std_population/variance_population dict per group

    Raises:
        ValueError: if scores is empty or not a multiple of group_size.
    """
    if not scores or len(scores) % group_size != 0:
        raise ValueError(
            f"scores length {len(scores)} is not a positive multiple of "
            f"group_size {group_size}"
        )
    groups: list[dict[str, float]] = []
    for start in range(0, len(scores), group_size):
        group = list(scores[start : start + group_size])
        groups.append(
            {
                "std_population": statistics.pstdev(group),
                "variance_population": statistics.pvariance(group),
            }
        )
    return tuple(groups)


def cost_gaps(
    breakdowns: Sequence[RewardBreakdown], sweep_best_cost_usd: float
) -> tuple[float, ...]:
    """Compute relative cost gaps for the feasible rung-3 completions.

    Args:
        breakdowns: the prompt's verifier breakdowns
        sweep_best_cost_usd: the frozen sweep-best cost for the prompt

    Returns:
        tuple: (cost_total_usd - sweep_best_cost_usd) / sweep_best_cost_usd
            for each rung-3 completion with feasible == 1.0 and a recorded
            cost, in rollout order
    """
    return tuple(
        (breakdown.cost_total_usd - sweep_best_cost_usd) / sweep_best_cost_usd
        for breakdown in breakdowns
        if breakdown.rung == 3
        and breakdown.feasible == 1.0
        and breakdown.cost_total_usd is not None
    )


def length_stats(token_counts: Sequence[int]) -> dict[str, float]:
    """Summarize completion lengths as mean, median, and 90th percentile.

    Args:
        token_counts: valid-token counts, first EOS included; must be
            non-empty

    Returns:
        dict: mean, p50, and p90 of the counts

    Raises:
        ValueError: if token_counts is empty.
    """
    if not token_counts:
        raise ValueError("length_stats requires at least one count")
    values = [float(count) for count in token_counts]
    return {
        "mean": sum(values) / len(values),
        "p50": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
    }


def token_weighted_entropy(
    entropy_sums: Sequence[float], entropy_token_counts: Sequence[int]
) -> float:
    """Compute a token-weighted mean entropy over completions.

    Args:
        entropy_sums: per-completion entropy sums in nats
        entropy_token_counts: per-completion valid-token counts

    Returns:
        float: total entropy over total valid tokens, or 0.0 when no
            valid tokens exist
    """
    total_tokens = sum(entropy_token_counts)
    if total_tokens == 0:
        return 0.0
    return sum(entropy_sums) / total_tokens


def prompt_payload(
    rollouts: PromptRollouts, eval_instance: EvalInstance, group_size: int
) -> dict[str, object]:
    """Assemble one prompt's per_prompt artifact entry.

    Args:
        rollouts: the prompt's sampled rollouts and scores
        eval_instance: the held-out instance backing the prompt
        group_size: rollouts per group for the spread diagnostic

    Returns:
        dict: the full per-prompt payload including every completion
    """
    scores = [breakdown.score for breakdown in rollouts.breakdowns]
    spread = group_reward_spread(scores, group_size)
    zero_variance_count = sum(
        1 for group in spread if group["variance_population"] == 0.0
    )
    gaps = cost_gaps(rollouts.breakdowns, rollouts.sweep_best_cost_usd)
    completions: list[dict[str, object]] = []
    for rollout_index, breakdown in enumerate(rollouts.breakdowns):
        token_count = rollouts.entropy_token_counts[rollout_index]
        completions.append(
            {
                "rollout_index": rollout_index,
                "group_index": rollout_index // group_size,
                "token_count": rollouts.token_counts[rollout_index],
                "truncated": rollouts.truncated[rollout_index],
                "truncation_category": truncation_category(
                    rollouts.truncated[rollout_index],
                    breakdown.rung,
                    rollouts.texts[rollout_index],
                ),
                "entropy_nats": (
                    rollouts.entropy_sums[rollout_index] / token_count
                    if token_count > 0
                    else None
                ),
                "breakdown": breakdown_payload(breakdown),
                "text": rollouts.texts[rollout_index],
            }
        )
    return {
        "index": rollouts.index,
        "instance": instance_payload(eval_instance.instance),
        "prompt_sha256": rollouts.prompt_sha256,
        "sweep_best_cost_usd": rollouts.sweep_best_cost_usd,
        "sweep_best_score": rollouts.sweep_best_score,
        "rates": ladder_rates(rollouts.breakdowns),
        "group_reward_spread": [dict(group) for group in spread],
        "zero_variance_group_count": zero_variance_count,
        "all_groups_zero_variance": zero_variance_count == len(spread),
        "cost_gap_mean": statistics.mean(gaps) if gaps else None,
        "cost_gap_min": min(gaps) if gaps else None,
        "completion_tokens": length_stats(rollouts.token_counts),
        "entropy_mean_nats": token_weighted_entropy(
            rollouts.entropy_sums, rollouts.entropy_token_counts
        ),
        "new_tokens": rollouts.new_tokens,
        "elapsed_s": rollouts.elapsed_s,
        "tokens_per_second": (
            rollouts.new_tokens / rollouts.elapsed_s if rollouts.elapsed_s > 0 else 0.0
        ),
        "completions": completions,
    }


def overall_metrics(
    prompts: Sequence[PromptRollouts], group_size: int
) -> dict[str, object]:
    """Aggregate every metric over all scored completions.

    Groups are consecutive slices of group_size rollouts within one
    prompt; they never span prompts.

    Args:
        prompts: every prompt's rollouts, in eval-split order; must be
            non-empty
        group_size: rollouts per group

    Returns:
        dict: the artifact's metrics section

    Raises:
        ValueError: if prompts is empty.
    """
    if not prompts:
        raise ValueError("overall_metrics requires at least one prompt")
    breakdowns = [breakdown for prompt in prompts for breakdown in prompt.breakdowns]
    n = len(breakdowns)
    rates = ladder_rates(breakdowns)
    gaps = [
        gap
        for prompt in prompts
        for gap in cost_gaps(prompt.breakdowns, prompt.sweep_best_cost_usd)
    ]
    group_stds = [
        group["std_population"]
        for prompt in prompts
        for group in group_reward_spread(
            [breakdown.score for breakdown in prompt.breakdowns], group_size
        )
    ]
    category_counts = {
        CATEGORY_PARSEABLE: 0,
        CATEGORY_STRUCTURED: 0,
        CATEGORY_UNSTRUCTURED: 0,
    }
    truncated_total = 0
    for prompt in prompts:
        for text, breakdown, was_truncated in zip(
            prompt.texts, prompt.breakdowns, prompt.truncated, strict=True
        ):
            category = truncation_category(was_truncated, breakdown.rung, text)
            if category is not None:
                truncated_total += 1
                category_counts[category] += 1
    token_counts = [count for prompt in prompts for count in prompt.token_counts]
    total_new_tokens = sum(prompt.new_tokens for prompt in prompts)
    total_elapsed = sum(prompt.elapsed_s for prompt in prompts)
    return {
        "n_completions": n,
        "parse_rate": rates["parse_rate"],
        "drc_pass_rate": rates["drc_pass_rate"],
        "rung3_rate": rates["rung3_rate"],
        "feasibility_rate": rates["feasibility_rate"],
        "feasibility_given_rung3": rates["feasibility_given_rung3"],
        "cost_gap_mean": statistics.mean(gaps) if gaps else None,
        "cost_gap_min": min(gaps) if gaps else None,
        "mean_group_reward_std": statistics.mean(group_stds),
        "group_std_min": min(group_stds),
        "group_std_p10": quantile(group_stds, 0.10),
        "zero_variance_group_fraction": (
            sum(1 for std in group_stds if std == 0.0) / len(group_stds)
        ),
        "truncation_rate": truncated_total / n,
        "cap_hit_parseable_rate": category_counts[CATEGORY_PARSEABLE] / n,
        "cap_hit_structured_parse_failure_rate": (
            category_counts[CATEGORY_STRUCTURED] / n
        ),
        "cap_hit_unstructured_rate": category_counts[CATEGORY_UNSTRUCTURED] / n,
        "completion_tokens": length_stats(token_counts),
        "total_new_tokens": total_new_tokens,
        "total_elapsed_s": total_elapsed,
        "tokens_per_second": (
            total_new_tokens / total_elapsed if total_elapsed > 0 else 0.0
        ),
    }


def entropy_summary(prompts: Sequence[PromptRollouts]) -> dict[str, object]:
    """Compute the entropy baseline and its diagnostics.

    The baseline is the token-weighted global mean — every valid token
    weighs equally, so longer completions weigh more — matching the
    trainer's logged entropy metric. Per-completion and per-prompt means
    are diagnostics only.

    Args:
        prompts: every prompt's rollouts; must be non-empty

    Returns:
        dict: baseline plus per_completion_mean, p10, p50, p90, and
            per_prompt_mean diagnostics

    Raises:
        ValueError: if prompts is empty.
    """
    if not prompts:
        raise ValueError("entropy_summary requires at least one prompt")
    all_sums = [value for prompt in prompts for value in prompt.entropy_sums]
    all_counts = [count for prompt in prompts for count in prompt.entropy_token_counts]
    completion_means = [
        entropy_sum / count if count > 0 else 0.0
        for entropy_sum, count in zip(all_sums, all_counts, strict=True)
    ]
    return {
        "baseline": token_weighted_entropy(all_sums, all_counts),
        "diagnostics": {
            "per_completion_mean": statistics.mean(completion_means),
            "p10": quantile(completion_means, 0.10),
            "p50": quantile(completion_means, 0.50),
            "p90": quantile(completion_means, 0.90),
            "per_prompt_mean": [
                token_weighted_entropy(prompt.entropy_sums, prompt.entropy_token_counts)
                for prompt in prompts
            ],
        },
    }


def resolve_run_mode(
    limit_prompts: int | None,
    rollouts_per_prompt: int,
    group_size: int,
    max_new_tokens: int,
) -> str:
    """Decide whether a run's verdict is canonical or a smoke provisional.

    Assumptions:
        1. Both completion caps are canonical: when the cap contingency
           fires, the authoritative entropy baseline must be remeasured
           at the raised cap the training run will actually use, so a
           full-shape raised-cap run cannot be demoted to smoke.

    Args:
        limit_prompts: the prompt-limit flag; None means the full eval
            split
        rollouts_per_prompt: total completions sampled per prompt
        group_size: rollouts per group
        max_new_tokens: completion token cap

    Returns:
        str: "canonical" only for a full-split run at the canonical
            rollout count and group size with the base or raised
            completion cap; "smoke" otherwise
    """
    canonical = (
        limit_prompts is None
        and rollouts_per_prompt == CANONICAL_ROLLOUTS_PER_PROMPT
        and group_size == TrainingConfig().num_generations
        and max_new_tokens in (BASE_COMPLETION_CAP, RAISED_COMPLETION_CAP)
    )
    return RUN_MODE_CANONICAL if canonical else RUN_MODE_SMOKE


def non_authoritative_reasons(
    run_mode: str,
    resolved_revision: str,
    lock_sha256: str | None,
    git_dirty: bool,
    git_sha: str,
    source_changed: bool,
) -> list[str]:
    """Collect every reason a run's verdict cannot be authoritative.

    Assumptions:
        1. The entropy baseline anchors training kill thresholds for the
           pinned model revision, so a run against any other revision, or
           one whose dependency set cannot be pinned, stays provisional
           regardless of its shape.
        2. A dirty working tree demotes the run: the recorded git sha no
           longer fully describes the code that produced the artifact.
        3. An unreadable git identity or a source state that changed
           between the opening stamp and the artifact write demotes the
           run for the same reason: the artifact cannot name one commit
           that produced it.

    Args:
        run_mode: "canonical" or "smoke"
        resolved_revision: the revision the checkpoint resolved to
        lock_sha256: the lock-file hash, or None when unavailable
        git_dirty: whether the working tree had uncommitted changes
        git_sha: the stamped commit sha, or "unknown" when git failed
        source_changed: whether the provenance stamp re-read at write
            time no longer matches the opening stamp

    Returns:
        list: human-readable reasons; empty when the verdict is
            authoritative
    """
    reasons: list[str] = []
    if run_mode != RUN_MODE_CANONICAL:
        reasons.append(
            "smoke run: not the full eval split at the canonical rollout "
            "count, group size, and completion cap"
        )
    if resolved_revision != MODEL_REVISION:
        reasons.append(
            f"resolved revision {resolved_revision} is not the pinned "
            f"revision {MODEL_REVISION}"
        )
    if lock_sha256 is None:
        reasons.append("uv.lock hash unavailable; the dependency set is not pinned")
    if git_dirty:
        reasons.append(
            "working tree had uncommitted changes; the recorded git sha "
            "does not fully describe the source state"
        )
    if git_sha == "unknown":
        reasons.append(
            "git sha is unknown; the artifact cannot identify the source "
            "state that produced it"
        )
    if source_changed:
        reasons.append(
            "source state changed during the run; the stamp at write time "
            "does not match the opening stamp"
        )
    return reasons


def gate_verdict(
    reasons: Sequence[str],
    rung3_rate: float,
    structured_truncation_rate: float,
    entropy_baseline: float,
    max_new_tokens: int,
    zero_variance_prompt_indices: Sequence[int],
) -> dict[str, object]:
    """Build the gate's verdict from the decisive metrics.

    Assumptions:
        1. Only the structured-parse-failure truncation rate drives the
           completion-cap raise: truncated-but-parseable completions
           finished their design, and unstructured rambles would ramble
           at any cap.
        2. The recommended cap never steps back down: a run measured at
           the raised cap keeps recommending it, since a low truncation
           rate there is the effect of the raise, not evidence against
           it.
        3. rerun_at_raised_cap_required flags that the entropy baseline
           was measured under a smaller cap than the one recommended, so
           the authoritative baseline must be remeasured at the
           recommended cap before training.
        4. The variance gate fails a prompt only when every one of its
           groups has exactly zero reward variance: identical rewards
           give identically zero centered advantages, so such a prompt
           contributes no gradient and must be filtered or re-bucketed
           before training. Near-zero spread stays a diagnostic.
        5. authoritative is False whenever any reason is present;
           consumers must key off it, since non-authoritative artifacts
           still carry provisional verdict fields.

    Args:
        reasons: the non-authoritative reasons; empty for an
            authoritative run
        rung3_rate: fraction of completions reaching rung 3
        structured_truncation_rate: fraction of completions truncated
            mid-design
        entropy_baseline: the token-weighted baseline entropy in nats
        max_new_tokens: the completion cap the run was measured under
        zero_variance_prompt_indices: roster indices of prompts whose
            groups all have zero reward variance

    Returns:
        dict: the artifact's verdict section
    """
    completion_cap = (
        RAISED_COMPLETION_CAP
        if max_new_tokens >= RAISED_COMPLETION_CAP
        or structured_truncation_rate > STRUCTURED_TRUNCATION_THRESHOLD
        else BASE_COMPLETION_CAP
    )
    return {
        "authoritative": not reasons,
        "non_authoritative_reasons": list(reasons),
        "sft_recommended": rung3_rate < SFT_RUNG3_THRESHOLD,
        "sft_rung3_threshold": SFT_RUNG3_THRESHOLD,
        "rung3_rate": rung3_rate,
        "completion_cap": completion_cap,
        "measured_completion_cap": max_new_tokens,
        "rerun_at_raised_cap_required": completion_cap > max_new_tokens,
        "structured_truncation_threshold": STRUCTURED_TRUNCATION_THRESHOLD,
        "cap_hit_structured_parse_failure_rate": structured_truncation_rate,
        "variance_gate_passed": not zero_variance_prompt_indices,
        "zero_variance_prompt_indices": list(zero_variance_prompt_indices),
        "entropy_baseline": entropy_baseline,
    }


def print_prompt_table(prompt_payloads: Sequence[dict[str, object]]) -> None:
    """Print the per-prompt summary table.

    Args:
        prompt_payloads: the per_prompt artifact entries

    Returns: None
    """
    table = Table(title="Per-prompt rollouts", title_justify="left")
    table.add_column("index")
    table.add_column("parse")
    table.add_column("drc")
    table.add_column("rung3")
    table.add_column("zero-var groups")
    table.add_column("entropy nats")
    table.add_column("tok/s")
    for payload in prompt_payloads:
        rates = cast(dict[str, object], payload["rates"])
        table.add_row(
            str(payload["index"]),
            f"{cast(float, rates['parse_rate']):.2f}",
            f"{cast(float, rates['drc_pass_rate']):.2f}",
            f"{cast(float, rates['rung3_rate']):.2f}",
            str(payload["zero_variance_group_count"]),
            f"{cast(float, payload['entropy_mean_nats']):.3f}",
            f"{cast(float, payload['tokens_per_second']):.1f}",
        )
    console.print(table)


def print_metrics_table(metrics: dict[str, object]) -> None:
    """Print the overall metrics table.

    Args:
        metrics: the artifact's metrics section

    Returns: None
    """
    table = Table(title="Overall metrics", title_justify="left")
    table.add_column("metric")
    table.add_column("value")
    displayed = (
        "n_completions",
        "parse_rate",
        "drc_pass_rate",
        "rung3_rate",
        "feasibility_rate",
        "feasibility_given_rung3",
        "cost_gap_mean",
        "mean_group_reward_std",
        "zero_variance_group_fraction",
        "truncation_rate",
        "cap_hit_structured_parse_failure_rate",
        "tokens_per_second",
    )
    for name in displayed:
        value = metrics[name]
        if isinstance(value, float):
            table.add_row(name, f"{value:.4f}")
        else:
            table.add_row(name, str(value))
    console.print(table)


def print_verdict(verdict: dict[str, object]) -> None:
    """Print the gate verdict, labeling smoke runs provisional.

    Args:
        verdict: the artifact's verdict section

    Returns: None
    """
    authoritative = bool(verdict["authoritative"])
    label = (
        "[bold]Verdict[/bold]"
        if authoritative
        else "[bold]Verdict (provisional — not authoritative)[/bold]"
    )
    sft = bool(verdict["sft_recommended"])
    sft_text = (
        "[yellow]cold-start SFT recommended[/yellow]"
        if sft
        else "[green]no cold-start SFT needed[/green]"
    )
    console.print(label)
    for reason in cast(list[str], verdict["non_authoritative_reasons"]):
        console.print(f"  [yellow]not authoritative:[/yellow] {reason}")
    console.print(
        f"  rung-3 rate {cast(float, verdict['rung3_rate']):.4f} vs threshold "
        f"{cast(float, verdict['sft_rung3_threshold']):.2f}: {sft_text}"
    )
    console.print(
        f"  completion cap: {verdict['completion_cap']} (structured truncation "
        f"rate {cast(float, verdict['cap_hit_structured_parse_failure_rate']):.4f} "
        f"vs threshold {cast(float, verdict['structured_truncation_threshold']):.2f})"
    )
    if bool(verdict["rerun_at_raised_cap_required"]):
        console.print(
            "  [yellow]entropy baseline measured at cap "
            f"{verdict['measured_completion_cap']}; rerun at cap "
            f"{verdict['completion_cap']} to anchor the training "
            "baseline[/yellow]"
        )
    if bool(verdict["variance_gate_passed"]):
        console.print(
            "  [green]within-group reward variance present for every prompt[/green]"
        )
    else:
        indices = cast(list[int], verdict["zero_variance_prompt_indices"])
        console.print(
            f"  [red]zero reward variance in every group for prompt(s) "
            f"{indices}; filter or re-bucket before training[/red]"
        )
    console.print(
        f"  entropy baseline: {cast(float, verdict['entropy_baseline']):.4f} nats/token"
    )


@app.command()
def run(
    revision: str | None = typer.Option(
        None,
        help="Model revision to resolve; defaults to the pinned MODEL_REVISION.",
    ),
    device: str = typer.Option("auto", help="Generation device: auto, cuda, or mps."),
    rollouts_per_prompt: int = typer.Option(
        CANONICAL_ROLLOUTS_PER_PROMPT,
        help="Completions sampled per prompt; must be a positive multiple of "
        "the training group size.",
    ),
    group_batch_size: int | None = typer.Option(
        None,
        help="Completions per generate call; defaults to 4 on MPS and 8 on CUDA.",
    ),
    max_new_tokens: int = typer.Option(
        BASE_COMPLETION_CAP, help="Completion token cap per rollout."
    ),
    limit_prompts: int | None = typer.Option(
        None,
        help="Only sample the first N eval prompts; any limit makes the run "
        "a smoke run with a provisional verdict.",
    ),
    seed: int = typer.Option(0, help="Sampling seed, echoed into the artifact."),
    artifacts_dir: Path = typer.Option(
        Path("artifacts"),
        help="Directory holding the frozen calibration artifacts.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts"), help="Directory the gate artifact is written into."
    ),
) -> None:
    """Sample base-model rollouts on the eval split and write the gate artifact.

    Downloads the pinned base checkpoint, samples rollouts_per_prompt
    completions per held-out prompt under the training sampling policy,
    scores them through the verifier, and writes
    artifacts/rollout_gate/rollout_gate_<run_id>.json with metrics, the
    entropy baseline, and the routing verdict.

    Args:
        revision: model revision override; None uses the pinned revision
        device: generation device selection
        rollouts_per_prompt: completions sampled per prompt
        group_batch_size: completions per generate call; None picks the
            device default
        max_new_tokens: completion token cap per rollout
        limit_prompts: optional prompt limit for smoke runs
        seed: sampling seed
        artifacts_dir: directory holding the calibration artifacts
        output_dir: directory the gate artifact is written into

    Returns: None

    Raises:
        typer.Exit: with code 1 on operational failures — bad flags, the
            missing training extra, unusable calibration artifacts, a
            CPU-only environment, or a download or generation crash.
    """
    stamp = provenance_stamp()
    console.print(
        f"[bold]Rollout gate[/bold]  run_id {stamp['run_id']}, model {MODEL_ID}"
    )
    training_config = TrainingConfig()
    group_size = training_config.num_generations
    if device not in ("auto", "cuda", "mps"):
        console.print(f"[red]device must be auto, cuda, or mps, got {device!r}[/red]")
        raise typer.Exit(1)
    if rollouts_per_prompt <= 0 or rollouts_per_prompt % group_size != 0:
        console.print(
            f"[red]rollouts-per-prompt must be a positive multiple of the "
            f"group size {group_size}, got {rollouts_per_prompt}[/red]"
        )
        raise typer.Exit(1)
    if group_batch_size is not None and group_batch_size < 1:
        console.print(
            f"[red]group-batch-size must be at least 1, got {group_batch_size}[/red]"
        )
        raise typer.Exit(1)
    if max_new_tokens < 1:
        console.print(
            f"[red]max-new-tokens must be at least 1, got {max_new_tokens}[/red]"
        )
        raise typer.Exit(1)
    if limit_prompts is not None and limit_prompts < 1:
        console.print(
            f"[red]limit-prompts must be at least 1, got {limit_prompts}[/red]"
        )
        raise typer.Exit(1)

    try:
        backend = load_backend()
    except ImportError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    try:
        eval_split = load_eval_split(artifacts_dir)
        cost_ref_sha256 = artifact_sha256(artifacts_dir / COST_REF_FILENAME)
        sweep_best_sha256 = artifact_sha256(artifacts_dir / SWEEP_BEST_FILENAME)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    selected = eval_split[:limit_prompts] if limit_prompts is not None else eval_split

    environment = backend.environment_report()
    resolved_device = resolve_generation_device(device, environment)
    if resolved_device is None:
        console.print(
            "[red]no CUDA or MPS device available for the requested device "
            f"{device!r}; CPU-only run refused[/red]"
        )
        raise typer.Exit(1)
    resolved_batch = (
        group_batch_size
        if group_batch_size is not None
        else default_group_batch_size(resolved_device)
    )

    lock_sha256 = backend.lock_file_hash()
    if lock_sha256 is None:
        console.print(
            "[yellow]Warning:[/yellow] uv.lock not found; the artifact "
            "cannot pin the resolved dependency set."
        )

    target_revision = revision if revision is not None else MODEL_REVISION
    try:
        local_path, resolved_revision = backend.resolve_and_download(target_revision)
    except Exception as error:  # noqa: BLE001 - operational failure ends the run
        console.print(f"[red]model download failed: {error}[/red]")
        raise typer.Exit(1) from error

    backend.seed_generation(seed)
    try:
        sampler = backend.load_sampler(local_path, resolved_device)
    except Exception as error:  # noqa: BLE001 - operational failure ends the run
        console.print(f"[red]model load failed: {error}[/red]")
        raise typer.Exit(1) from error

    sampling = sampling_policy(training_config)
    prompt_rollouts: list[PromptRollouts] = []
    try:
        for position, eval_instance in enumerate(selected):
            console.print(
                f"Sampling prompt {position + 1}/{len(selected)} "
                f"(instance {eval_instance.index}), {rollouts_per_prompt} rollouts..."
            )
            rollouts = collect_prompt_rollouts(
                backend,
                sampler,
                eval_instance,
                rollouts_per_prompt,
                resolved_batch,
                max_new_tokens,
                sampling,
            )
            rates = ladder_rates(rollouts.breakdowns)
            console.print(
                f"  parse {cast(float, rates['parse_rate']):.2f}, "
                f"rung3 {cast(float, rates['rung3_rate']):.2f}, "
                f"{rollouts.new_tokens} tokens in {rollouts.elapsed_s:.1f}s"
            )
            prompt_rollouts.append(rollouts)
    except Exception as error:  # noqa: BLE001 - operational failure ends the run
        console.print(f"[red]rollout collection failed: {error}[/red]")
        raise typer.Exit(1) from error

    stamp_at_write = provenance_stamp()
    source_changed = cast(str, stamp_at_write["git_sha"]) != stamp["git_sha"] or (
        cast(bool, stamp_at_write["git_dirty"]) and not cast(bool, stamp["git_dirty"])
    )
    run_mode = resolve_run_mode(
        limit_prompts, rollouts_per_prompt, group_size, max_new_tokens
    )
    metrics = overall_metrics(prompt_rollouts, group_size)
    entropy = entropy_summary(prompt_rollouts)
    prompt_payloads = [
        prompt_payload(rollouts, eval_instance, group_size)
        for rollouts, eval_instance in zip(prompt_rollouts, selected, strict=True)
    ]
    zero_variance_prompts = [
        cast(int, payload["index"])
        for payload in prompt_payloads
        if payload["all_groups_zero_variance"]
    ]
    verdict = gate_verdict(
        non_authoritative_reasons(
            run_mode,
            resolved_revision,
            lock_sha256,
            cast(bool, stamp["git_dirty"]),
            cast(str, stamp["git_sha"]),
            source_changed,
        ),
        cast(float, metrics["rung3_rate"]),
        cast(float, metrics["cap_hit_structured_parse_failure_rate"]),
        cast(float, entropy["baseline"]),
        max_new_tokens,
        zero_variance_prompts,
    )
    artifact = {
        "stamp": stamp,
        "stamp_at_write": {
            "git_sha": stamp_at_write["git_sha"],
            "git_dirty": stamp_at_write["git_dirty"],
        },
        "config": {
            "model_id": MODEL_ID,
            "pinned_revision": MODEL_REVISION,
            "resolved_revision": resolved_revision,
            "local_path": local_path,
            "device": resolved_device,
            "dtype": sampler["dtype"],
            "lm_head_dtype": sampler["lm_head_dtype"],
            "seed": seed,
            "run_mode": run_mode,
            "rollouts_per_prompt": rollouts_per_prompt,
            "group_size": group_size,
            "group_batch_size": resolved_batch,
            "max_new_tokens": max_new_tokens,
            "sampling": sampling,
            "n_prompts": len(selected),
            "limit_prompts": limit_prompts,
        },
        "environment": environment,
        "inputs": {
            "cost_ref_sha256": cost_ref_sha256,
            "sweep_best_sha256": sweep_best_sha256,
            "lock_sha256": lock_sha256,
        },
        "data_use": {
            "prompt_source": "held_out_eval_split",
            "completions": "audit_only",
        },
        "per_prompt": prompt_payloads,
        "metrics": metrics,
        "entropy": {
            "method": ENTROPY_METHOD,
            "unit": "nats_per_token",
            "temperature": training_config.temperature,
            "lm_head_dtype": sampler["lm_head_dtype"],
            "baseline": entropy["baseline"],
            "diagnostics": entropy["diagnostics"],
        },
        "verdict": verdict,
    }
    artifact_path = output_dir / GATE_DIR / f"rollout_gate_{stamp['run_id']}.json"
    write_json(artifact_path, artifact)

    print_prompt_table(prompt_payloads)
    print_metrics_table(metrics)
    print_verdict(verdict)
    console.print(f"Artifact written to {artifact_path}")


def main() -> None:
    """Run the rollout-gate command-line interface.

    Args: None

    Returns: None
    """
    app()


if __name__ == "__main__":
    main()
