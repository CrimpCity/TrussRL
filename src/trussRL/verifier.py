"""Verifier entry point.

``score(instance, completion_text) -> RewardBreakdown``: the pure
composition of the full chain (schema -> drc -> expander/loads -> solver
-> capacity -> reward). This is the single entry point the RL adapter,
calibration sweep, baselines, and eval harness all call. Importing this
module must never pull RL dependencies (torch/vllm/trl).
"""
