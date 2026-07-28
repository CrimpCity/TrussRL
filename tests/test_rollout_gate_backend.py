"""Contract tests for the rollout-gate sampling backend.

Exercised against fake models and synthetic logits so the generation
kwargs, completion slicing, and stepwise entropy accumulation are pinned
without downloading a checkpoint. The module skips unless torch and
transformers are installed; only the TRL equivalence test additionally
requires trl, pinning the gate's entropy against the installed TRL's own
entropy function.
"""

import functools
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import trussRL.training.rollout_gate as backend  # noqa: E402 - needs the skips above

VOCAB = 11
EOS = 9
PROMPT_TOKENS = 3
TEMPERATURE = 0.7

SAMPLING: dict[str, Any] = {
    "do_sample": True,
    "temperature": TEMPERATURE,
    "top_p": 0.95,
    "top_k": 0,
    "min_p": None,
    "repetition_penalty": 1.0,
}


class FakeTokenizer:
    """A tokenizer stand-in with a fixed three-token prompt encoding."""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        """Encode any text as the fixed prompt ids.

        Args:
            text: the prompt text, ignored
            return_tensors: tensor format flag, ignored

        Returns:
            dict: input_ids and attention_mask of shape (1, 3)
        """
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, PROMPT_TOKENS), dtype=torch.long),
        }

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        """Decode ids as space-joined integers, optionally dropping EOS.

        Args:
            ids: the token ids to decode
            skip_special_tokens: whether the EOS id is dropped

        Returns:
            str: the decoded text
        """
        kept = [
            str(int(token))
            for token in ids
            if not (skip_special_tokens and int(token) == EOS)
        ]
        return " ".join(kept)


class FakeModel:
    """A model stand-in that records generate kwargs and serves fixed output."""

    def __init__(self, sequences: Any, logits: tuple[Any, ...]) -> None:
        """Store the canned generation output.

        Args:
            sequences: the (batch, prompt + new) id tensor generate returns
            logits: the per-step logits tuple generate returns
        """
        self.calls: list[dict[str, Any]] = []
        self.sequences = sequences
        self.logits = logits

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        """Record the call and return the canned output.

        Args:
            **kwargs: the generation kwargs under test

        Returns:
            SimpleNamespace: sequences and logits
        """
        self.calls.append(kwargs)
        return SimpleNamespace(sequences=self.sequences, logits=self.logits)


def mixed_row_output() -> tuple[Any, tuple[Any, ...]]:
    """Build a two-row generation where one row EOSes early.

    Row 0 emits one token then EOS and is padded with EOS; row 1 runs to
    the four-token cap with no EOS.

    Args: None

    Returns:
        tuple: (sequences, logits) for FakeModel
    """
    sequences = torch.tensor(
        [
            [1, 2, 3, 5, EOS, EOS, EOS],
            [1, 2, 3, 5, 6, 7, 8],
        ]
    )
    generator = torch.Generator().manual_seed(0)
    logits = tuple(torch.randn((2, VOCAB), generator=generator) for _ in range(4))
    return sequences, logits


def make_sampler(model: FakeModel) -> dict[str, Any]:
    """Assemble a sampler dict around a fake model.

    Args:
        model: the fake model serving generate calls

    Returns:
        dict: the sampler record sample_group expects
    """
    return {
        "tokenizer": FakeTokenizer(),
        "model": model,
        "device": "cpu",
        "dtype": "bfloat16",
        "lm_head_dtype": "float32",
        "eos_token_id": EOS,
    }


def test_sample_group_generation_kwargs() -> None:
    sequences, logits = mixed_row_output()
    model = FakeModel(sequences, logits)
    backend.sample_group(make_sampler(model), "prompt", 2, 2, 4, dict(SAMPLING))
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["do_sample"] is True
    assert call["temperature"] == TEMPERATURE
    assert call["top_p"] == 0.95
    assert call["top_k"] == 0
    assert call["min_p"] is None
    assert call["repetition_penalty"] == 1.0
    assert call["max_new_tokens"] == 4
    assert call["pad_token_id"] == EOS
    assert call["return_dict_in_generate"] is True
    assert call["output_logits"] is True
    assert call["input_ids"].shape == (2, PROMPT_TOKENS)


def test_completion_slicing_mixed_rows() -> None:
    sequences, logits = mixed_row_output()
    model = FakeModel(sequences, logits)
    result = backend.sample_group(
        make_sampler(model), "prompt", 2, 2, 4, dict(SAMPLING)
    )
    assert result["token_counts"] == [2, 4]
    assert result["truncated"] == [False, True]
    assert result["entropy_token_counts"] == [2, 4]
    assert result["texts"] == ["5", "5 6 7 8"]
    assert result["new_tokens"] == 6
    assert cast(float, result["elapsed_s"]) > 0.0


def test_sample_group_sub_batches_cover_group() -> None:
    sequences, logits = mixed_row_output()
    model = FakeModel(sequences, logits)
    result = backend.sample_group(
        make_sampler(model), "prompt", 6, 2, 4, dict(SAMPLING)
    )
    assert len(model.calls) == 3
    assert len(cast(list[str], result["texts"])) == 6
    assert result["new_tokens"] == 3 * 6


def test_entropy_uniform_and_peaked() -> None:
    sums = [0.0, 0.0]
    counts = [0, 0]
    uniform = torch.zeros((2, 50))
    backend.entropy_step_update(uniform, [True, False], TEMPERATURE, sums, counts)
    assert sums[0] == pytest.approx(math.log(50), rel=1e-5)
    assert counts == [1, 0]
    assert sums[1] == 0.0

    peaked = torch.zeros((1, 50))
    peaked[0, 7] = 100.0
    peaked_sums = [0.0]
    peaked_counts = [0]
    backend.entropy_step_update(peaked, [True], TEMPERATURE, peaked_sums, peaked_counts)
    assert peaked_sums[0] == pytest.approx(0.0, abs=1e-6)
    assert peaked_counts == [1]


def test_entropy_stepwise_matches_direct() -> None:
    steps, rows, vocab = 5, 3, 20
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn((steps, rows, vocab), generator=generator)
    lengths = [5, 3, 1]
    sums = [0.0] * rows
    counts = [0] * rows
    for step in range(steps):
        active = [step < length for length in lengths]
        backend.entropy_step_update(logits[step], active, TEMPERATURE, sums, counts)
    for row in range(rows):
        direct = 0.0
        for step in range(lengths[row]):
            log_probs = torch.log_softmax(
                logits[step, row].to(torch.float32) / TEMPERATURE, dim=-1
            )
            direct += float(-(log_probs.exp() * log_probs).sum())
        assert sums[row] == pytest.approx(direct, rel=1e-6)
        assert counts[row] == lengths[row]


def test_eos_on_final_allowed_token_not_truncated() -> None:
    sequences = torch.tensor([[1, 2, 3, 5, 6, 7, EOS]])
    generator = torch.Generator().manual_seed(3)
    logits = tuple(torch.randn((1, VOCAB), generator=generator) for _ in range(4))
    model = FakeModel(sequences, logits)
    result = backend.sample_group(
        make_sampler(model), "prompt", 1, 1, 4, dict(SAMPLING)
    )
    assert result["token_counts"] == [4]
    assert result["truncated"] == [False]
    assert result["entropy_token_counts"] == [4]


def test_first_eos_lengths() -> None:
    ids = torch.tensor(
        [
            [4, EOS, EOS, EOS],
            [4, 5, 6, 7],
            [EOS, 1, EOS, 2],
        ]
    )
    assert backend.first_eos_lengths(ids, EOS) == [2, 4, 1]


def tiny_causal_lm(tie_word_embeddings: bool) -> Any:
    """Build a tiny bf16 Qwen3 causal LM for head-cast contract tests.

    Args:
        tie_word_embeddings: whether the LM head shares the input
            embedding weight, as the pinned base checkpoint does

    Returns:
        model: the tiny model in bf16 on CPU
    """
    from transformers import AutoModelForCausalLM, Qwen3Config

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model.to(torch.bfloat16)


def test_cast_lm_head_fp32_tied_embeddings() -> None:
    model = tiny_causal_lm(tie_word_embeddings=True)
    assert model.lm_head.weight is model.model.embed_tokens.weight

    head_dtype = backend.cast_lm_head_to_fp32(model)

    assert head_dtype == "float32"
    assert model.lm_head.weight.dtype == torch.float32
    assert model.lm_head.weight is model.model.embed_tokens.weight
    input_ids = torch.tensor([[1, 2, 3]])
    embedded = model.model.embed_tokens(input_ids)
    assert embedded.dtype == torch.bfloat16
    with torch.no_grad():
        output = model(input_ids=input_ids)
    assert output.logits.dtype == torch.float32


def test_cast_lm_head_fp32_untied_embeddings() -> None:
    model = tiny_causal_lm(tie_word_embeddings=False)
    assert model.lm_head.weight is not model.model.embed_tokens.weight

    head_dtype = backend.cast_lm_head_to_fp32(model)

    assert head_dtype == "float32"
    assert model.lm_head.weight.dtype == torch.float32
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        output = model(input_ids=input_ids)
    assert output.logits.dtype == torch.float32


class FakeAutoTokenizer:
    """A tokenizer stand-in served by from_pretrained without a download."""

    eos_token_id = EOS
    pad_token_id = None

    @classmethod
    def from_pretrained(cls, model_path: str) -> "FakeAutoTokenizer":
        """Build the stand-in for any snapshot path.

        Args:
            model_path: the snapshot path, ignored

        Returns:
            FakeAutoTokenizer: a fresh stand-in instance
        """
        return cls()


def stub_model_with_head() -> SimpleNamespace:
    """Build a minimal model satisfying the fp32 head cast.

    Args: None

    Returns:
        SimpleNamespace: an object with an untied fp16 lm_head and config
    """
    model = SimpleNamespace()
    model.lm_head = torch.nn.Linear(4, 8, bias=False).to(torch.float16)
    model.config = SimpleNamespace(tie_word_embeddings=False)
    return model


def load_probe_bf16_fails(
    events: list[tuple[str, object]],
    model_path: str,
    device: str,
    dtype: object,
    tokenizer: Any,
) -> SimpleNamespace:
    """Record each load attempt and fail only the bf16 one.

    Args:
        events: the shared event log, mutated in place
        model_path: the snapshot path, ignored
        device: the target device, ignored
        dtype: the requested torch dtype
        tokenizer: the tokenizer, ignored

    Returns:
        SimpleNamespace: a stub model for non-bf16 loads

    Raises:
        RuntimeError: for the bf16 attempt.
    """
    events.append(("load", dtype))
    if dtype == torch.bfloat16:
        raise RuntimeError("bf16 probe failed")
    return stub_model_with_head()


def record_release(events: list[tuple[str, object]], device: str) -> None:
    """Record a memory-release call in the shared event log.

    Args:
        events: the shared event log, mutated in place
        device: the device being released

    Returns: None
    """
    events.append(("release", device))


def test_mps_fallback_releases_memory_between_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(backend, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        backend, "load_and_probe", functools.partial(load_probe_bf16_fails, events)
    )
    monkeypatch.setattr(
        backend,
        "release_cached_device_memory",
        functools.partial(record_release, events),
    )
    sampler = backend.load_sampler("/fake/snapshot", "mps")
    assert events == [
        ("load", torch.bfloat16),
        ("release", "mps"),
        ("load", torch.float16),
    ]
    assert sampler["dtype"] == "float16"
    assert sampler["lm_head_dtype"] == "float32"
    assert sampler["eos_token_id"] == EOS


def test_non_mps_load_failure_propagates_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(backend, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        backend, "load_and_probe", functools.partial(load_probe_bf16_fails, events)
    )
    monkeypatch.setattr(
        backend,
        "release_cached_device_memory",
        functools.partial(record_release, events),
    )
    with pytest.raises(RuntimeError):
        backend.load_sampler("/fake/snapshot", "cuda")
    assert events == [("load", torch.bfloat16)]


def test_entropy_matches_pinned_trl() -> None:
    pytest.importorskip("trl")
    from trl.trainer.utils import entropy_from_logits

    steps, rows, vocab = 6, 4, 32
    generator = torch.Generator().manual_seed(2)
    logits = torch.randn((rows, steps, vocab), generator=generator)
    completion_ids = torch.randint(0, EOS, (rows, steps), generator=generator)
    completion_ids[0, 2] = EOS
    completion_ids[2, 0] = EOS
    lengths = backend.first_eos_lengths(completion_ids, EOS)

    is_eos = completion_ids == EOS
    eos_idx = torch.full((rows,), steps, dtype=torch.long)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(steps).expand(rows, -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).float()
    trl_entropies = entropy_from_logits(logits / TEMPERATURE)
    expected = float((trl_entropies * completion_mask).sum() / completion_mask.sum())

    sums = [0.0] * rows
    counts = [0] * rows
    for step in range(steps):
        active = [step < length for length in lengths]
        backend.entropy_step_update(
            logits[:, step, :], active, TEMPERATURE, sums, counts
        )
    baseline = sum(sums) / sum(counts)
    assert counts == lengths
    assert baseline == pytest.approx(expected, rel=1e-6)
