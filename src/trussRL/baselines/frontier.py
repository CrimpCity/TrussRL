"""Frontier models as a baseline: is there headroom worth training for?

Samples designs from frontier APIs for the go/no-go check and the eval
table. Held-out completions inform the headroom verdict only — they are
never SFT material, since training on held-out prompts would contaminate
the final evaluation. If cold-start SFT is needed, candidates come from a
separate verifier-filtered collection pass over training-roster prompts.
"""
