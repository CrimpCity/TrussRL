"""Conditional cold-start SFT (Phase 3).

Supervised fine-tuning from verifier-passing frontier rollouts, run only
if the base-model rollout gate (starting.md Module 9) says the policy
cannot reliably emit parseable designs.
"""
