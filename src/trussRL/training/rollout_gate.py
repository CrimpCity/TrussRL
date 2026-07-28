"""Backend for trussrl-rollout-gate: the only gate module that touches torch.

Loads the pinned base checkpoint with the LM head cast to fp32 — mirroring
the trainer's cast_lm_head_to_fp32 behavior, including the tied-embedding
output re-cast hook — samples completion groups under a fully explicit
sampling policy, and accumulates per-token entropy stepwise so the
generation logits are never stacked or fp32-cast as a whole.

Masking and entropy follow the installed TRL GRPO trainer exactly: a
completion's valid tokens run up to and including its first EOS token, a
completion with no EOS counts every generated token, and per-token entropy
is the full-vocabulary Shannon entropy in nats of
softmax(logits / temperature), computed in fp32 from the pre-nucleus
generation-step logits. The trainer's logged entropy metric is the
token-weighted mean of these values over all valid tokens, which is the
quantity the gate reproduces as its baseline.
"""

import functools
import gc
import time
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trussRL.training.model_check import (
    environment_report,
    lock_file_hash,
    resolve_and_download,
)

__all__ = [
    "cast_lm_head_to_fp32",
    "entropy_step_update",
    "environment_report",
    "first_eos_lengths",
    "load_and_probe",
    "load_sampler",
    "lock_file_hash",
    "release_cached_device_memory",
    "resolve_and_download",
    "sample_group",
    "seed_generation",
    "synchronize_device",
]

PROBE_MAX_NEW_TOKENS = 2


def seed_generation(seed: int) -> None:
    """Seed torch's global generator ahead of sampling.

    Assumptions:
        1. Best-effort reproducibility: on MPS the same seed is provenance,
           not a bitwise guarantee across torch or macOS versions.

    Args:
        seed: the seed recorded in the artifact

    Returns: None
    """
    torch.manual_seed(seed)


def synchronize_device(device: str) -> None:
    """Block until pending kernels on the generation device finish.

    Called immediately before starting and before stopping each generation
    clock so elapsed times measure completed work, not queued work.

    Args:
        device: cuda, mps, or any other device string (a no-op)

    Returns: None
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def fp32_lm_head_forward(lm_head: Any, hidden_states: Any) -> Any:
    """Compute LM-head logits in fp32 regardless of the model dtype.

    Args:
        lm_head: the model's output projection module, weights already fp32
        hidden_states: the final hidden states entering the head

    Returns:
        Tensor: fp32 logits
    """
    bias = lm_head.bias
    return torch.nn.functional.linear(
        hidden_states.to(torch.float32),
        lm_head.weight.to(torch.float32),
        None if bias is None else bias.to(torch.float32),
    )


def cast_embedding_output(dtype: Any, module: Any, inputs: Any, output: Any) -> Any:
    """Re-cast embedding outputs to the model's compute dtype.

    Registered as a forward hook on the input embedding when the LM head
    shares its weight: casting the head to fp32 also casts the shared
    embedding weight, and this hook keeps the rest of the network computing
    in its original dtype.

    Args:
        dtype: the model's original compute dtype
        module: the embedding module the hook fired on
        inputs: the embedding call's inputs
        output: the embedding call's output

    Returns:
        Tensor: the output cast back to the original dtype
    """
    return output.to(dtype)


def cast_lm_head_to_fp32(model: Any) -> str:
    """Run the LM head in fp32, matching the trainer's head-cast behavior.

    Assumptions:
        1. With tied word embeddings, the shared weight becomes fp32 on
           purpose (logit stability) while embedding activations are cast
           back to the original compute dtype by a forward hook, so only
           the head's numerics change.

    Args:
        model: the loaded causal LM, mutated in place

    Returns:
        str: the resulting head dtype, always "float32"
    """
    original_dtype = model.lm_head.weight.dtype
    lm_head = model.lm_head.float()
    model.lm_head = lm_head
    lm_head.forward = functools.partial(fp32_lm_head_forward, lm_head)
    if getattr(model.config, "tie_word_embeddings", False):
        model.model.embed_tokens.register_forward_hook(
            functools.partial(cast_embedding_output, original_dtype)
        )
    return "float32"


def load_model_at_dtype(model_path: str, device: str, dtype: Any) -> Any:
    """Load the causal LM at one dtype and move it to the device.

    Args:
        model_path: local snapshot path of the downloaded model
        device: target device string
        dtype: torch dtype to load the weights in

    Returns:
        model: the loaded model in eval mode on the device
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    return model


def probe_generation(model: Any, tokenizer: Any, device: str) -> None:
    """Generate a couple of greedy tokens to validate the loaded dtype.

    Args:
        model: the loaded model to probe
        tokenizer: the model's tokenizer
        device: the model's device

    Returns: None
    """
    inputs = tokenizer("check", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=PROBE_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )


def load_and_probe(model_path: str, device: str, dtype: Any, tokenizer: Any) -> Any:
    """Load the model at one dtype and validate it with a greedy probe.

    Args:
        model_path: local snapshot path of the downloaded model
        device: target device string
        dtype: torch dtype to load the weights in
        tokenizer: the model's tokenizer for the probe

    Returns:
        model: the loaded and probed model
    """
    model = load_model_at_dtype(model_path, device, dtype)
    probe_generation(model, tokenizer, device)
    return model


def release_cached_device_memory(device: str) -> None:
    """Free dereferenced tensors and return cached blocks to the device.

    Called between a failed load-and-probe attempt and its fallback load
    so the failed model's memory is actually released before a second
    copy of the weights arrives.

    Args:
        device: cuda, mps, or any other device string (a no-op)

    Returns: None
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def load_sampler(model_path: str, device: str) -> dict[str, object]:
    """Load tokenizer and model for sampling, bf16 first with fp16 fallback.

    Assumptions:
        1. MPS bf16 coverage has gaps, so a bf16 load or probe failure on
           MPS falls back to fp16; on any other device the failure
           propagates.
        2. The fallback load runs outside the except block, after the
           caught traceback stops pinning the failed model, and cached
           device memory is emptied first — the fallback never holds two
           copies of the weights.
        3. The LM head is cast to fp32 after the dtype probe, so the
           sampler's logits match the trainer's fp32-head logits.

    Args:
        model_path: local snapshot path of the downloaded model
        device: generation device, cuda or mps

    Returns:
        dict: tokenizer, model, device, dtype, lm_head_dtype, and
            eos_token_id
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.bfloat16
    model: Any = None
    try:
        model = load_and_probe(model_path, device, torch.bfloat16, tokenizer)
    except Exception:  # noqa: BLE001 - fp16 fallback is the MPS contract
        if device != "mps":
            raise
        dtype = torch.float16
    if model is None:
        release_cached_device_memory(device)
        model = load_and_probe(model_path, device, torch.float16, tokenizer)
    lm_head_dtype = cast_lm_head_to_fp32(model)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "lm_head_dtype": lm_head_dtype,
        "eos_token_id": int(tokenizer.eos_token_id),
    }


def first_eos_lengths(completion_ids: Any, eos_token_id: int) -> list[int]:
    """Count each row's valid tokens up to and including its first EOS.

    Matches the trainer's completion mask: the first EOS token is itself
    valid, later tokens are padding, and a row with no EOS counts every
    generated token. Padding with the EOS id is safe because only the
    first occurrence terminates the count.

    Args:
        completion_ids: (batch, length) tensor of generated token ids,
            prompt already sliced off
        eos_token_id: the tokenizer's EOS token id

    Returns:
        list: one valid-token count per row
    """
    lengths: list[int] = []
    for row in completion_ids:
        matches = (row == eos_token_id).nonzero()
        if matches.numel() == 0:
            lengths.append(int(row.shape[0]))
        else:
            lengths.append(int(matches[0, 0]) + 1)
    return lengths


def entropy_step_update(
    step_logits: Any,
    active_rows: list[bool],
    temperature: float,
    sums: list[float],
    counts: list[int],
) -> None:
    """Accumulate one generation step's per-row entropy into running sums.

    Computes the full-vocabulary Shannon entropy in nats of
    softmax(logits / temperature) in fp32 for every row still generating,
    touching only this step's (batch, vocab) logits so memory stays
    bounded by the batch size regardless of completion length.

    Args:
        step_logits: (batch, vocab) pre-nucleus logits for one step
        active_rows: per-row flag, True while the row's valid tokens
            continue
        temperature: the sampling temperature the logits are divided by
        sums: per-row entropy sums in nats, mutated in place
        counts: per-row valid-token counts, mutated in place

    Returns: None
    """
    indices = [row for row, active in enumerate(active_rows) if active]
    if not indices:
        return
    logits = step_logits[indices].to(torch.float32) / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
    for position, value in zip(indices, entropies.tolist(), strict=True):
        sums[position] += float(value)
        counts[position] += 1


def sample_group(
    sampler: dict[str, object],
    prompt: str,
    group_size: int,
    group_batch_size: int,
    max_new_tokens: int,
    sampling: dict[str, object],
) -> dict[str, object]:
    """Sample one prompt's rollouts in sub-batches, returning plain Python.

    Each generate call repeats the same prompt, so rows are identical and
    no padding is needed; completion ids are sliced at the prompt length.
    The device is synchronized immediately before starting and before
    stopping each generation clock. A completion is truncated when it has
    no EOS anywhere and the generated length equals max_new_tokens.

    Assumptions:
        1. The complete sampling policy is passed explicitly on every
           call, so the checkpoint's generation_config defaults (such as a
           nonzero top_k) can never silently apply.
        2. output_logits is used, never output_scores: scores are
           post-warp and would misstate the pre-nucleus distribution the
           entropy baseline is defined over.
        3. new_tokens counts valid tokens only (up to and including each
           first EOS), the same tokens entropy and length statistics are
           computed over.

    Args:
        sampler: the load_sampler result
        prompt: the rendered prompt text
        group_size: total rollouts to sample for this prompt
        group_batch_size: rollouts per generate call
        max_new_tokens: completion token cap
        sampling: explicit policy with do_sample, temperature, top_p,
            top_k, min_p, and repetition_penalty

    Returns:
        dict: texts, token_counts, truncated, entropy_sums,
            entropy_token_counts, new_tokens, elapsed_s
    """
    tokenizer = cast(Any, sampler["tokenizer"])
    model = cast(Any, sampler["model"])
    device = str(sampler["device"])
    eos_token_id = cast(int, sampler["eos_token_id"])
    temperature = cast(float, sampling["temperature"])

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = int(input_ids.shape[1])

    texts: list[str] = []
    token_counts: list[int] = []
    truncated: list[bool] = []
    entropy_sums: list[float] = []
    entropy_token_counts: list[int] = []
    total_new_tokens = 0
    total_elapsed = 0.0

    remaining = group_size
    while remaining > 0:
        batch = min(group_batch_size, remaining)
        remaining -= batch
        batch_ids = input_ids.repeat(batch, 1)
        batch_mask = attention_mask.repeat(batch, 1)
        synchronize_device(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                max_new_tokens=max_new_tokens,
                do_sample=cast(bool, sampling["do_sample"]),
                temperature=temperature,
                top_p=cast(float, sampling["top_p"]),
                top_k=cast(int, sampling["top_k"]),
                min_p=cast("float | None", sampling["min_p"]),
                repetition_penalty=cast(float, sampling["repetition_penalty"]),
                pad_token_id=eos_token_id,
                return_dict_in_generate=True,
                output_logits=True,
            )
        synchronize_device(device)
        total_elapsed += time.perf_counter() - start

        completion_ids = output.sequences[:, prompt_length:]
        lengths = first_eos_lengths(completion_ids, eos_token_id)
        generated_length = int(completion_ids.shape[1])
        sums = [0.0] * batch
        counts = [0] * batch
        for step, step_logits in enumerate(output.logits):
            active = [step < length for length in lengths]
            entropy_step_update(step_logits, active, temperature, sums, counts)
        for row in range(batch):
            row_ids = completion_ids[row, : lengths[row]]
            texts.append(tokenizer.decode(row_ids, skip_special_tokens=True))
            token_counts.append(lengths[row])
            has_eos = bool((completion_ids[row] == eos_token_id).any())
            truncated.append(not has_eos and generated_length == max_new_tokens)
            entropy_sums.append(sums[row])
            entropy_token_counts.append(counts[row])
        total_new_tokens += sum(lengths)

    return {
        "texts": texts,
        "token_counts": token_counts,
        "truncated": truncated,
        "entropy_sums": entropy_sums,
        "entropy_token_counts": entropy_token_counts,
        "new_tokens": total_new_tokens,
        "elapsed_s": total_elapsed,
    }
