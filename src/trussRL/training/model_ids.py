"""Pinned model identity constants for the training stack.

Stdlib-only on purpose: downstream training code imports these constants
without pulling torch or any RL framework into the process.
"""

MODEL_ID = "Qwen/Qwen3-4B-Base"
MODEL_REVISION: str | None = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
EXPECTED_PROMPT_TOKENS_MAX = 2048
