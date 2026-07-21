"""Decide whether cold-start SFT is needed before GRPO.

Samples base-model rollouts and measures parse and DRC pass rates — if the
base model rarely reaches the solver, RL would spend its first updates learning
output format instead of engineering.
"""
