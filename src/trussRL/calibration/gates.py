"""Sanity gates that must pass before any training run.

Check that reward is not saturated, variance is healthy, and the optimum is
interior rather than pegged at a bound — then freeze cost_ref and
sweep_best, which must never depend on the policy.
"""
