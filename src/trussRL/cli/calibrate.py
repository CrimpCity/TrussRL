"""Produce the frozen grading constants — the ground truth for training.

Runs thousands of random designs through the verifier, applies the sanity
gates, and writes cost_ref and sweep_best to artifacts/, stamped so every
result traces to an exact code and catalog state.
"""
