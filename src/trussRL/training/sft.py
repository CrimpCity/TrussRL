"""Cold-start SFT, run only if the base model cannot emit parseable designs.

A short supervised phase on verifier-passing frontier rollouts to bootstrap
output format — kept minimal because every SFT token risks anchoring the
policy to the teacher's heuristics.
"""
