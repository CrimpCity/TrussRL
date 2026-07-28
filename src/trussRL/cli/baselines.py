"""Run the non-RL baselines and answer: is training worth doing?

Scores the scripted heuristic engineer and frontier models with the same
verifier the policy will face. Held-out completions inform the headroom
verdict only; cold-start SFT candidates come from a separate collection
pass over training-roster prompts, never from held-out prompts.
"""
