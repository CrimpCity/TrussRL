# TrussRL
Deterministic RLVR environment for single span 2D truss design

### Setup

```bash
uv lock && uv sync && source .venv/bin/activate
```

### Project structure

```
TrussRL/
├── data/                   # frozen inputs: aisc_hss.csv section catalog (Module 1)
├── artifacts/              # committed calibration outputs (cost_ref, sweep_best, gate reports)
├── scripts/                # thin typer CLIs, one per experiment phase
├── src/trussRL/
│   ├── catalog.py          # section records from data/aisc_hss.csv
│   ├── schema.py           # TrussDesign model + JSON extraction (rung 0)
│   ├── instance.py         # TrussInstance: everything the prompt shows + grading params
│   ├── generator.py        # seeded procedural instance generator
│   ├── prompts.py          # TrussInstance -> prompt text
│   ├── expander.py         # design -> nodes/members/supports (geometry source of truth)
│   ├── loads.py            # line load -> panel points + self-weight
│   ├── drc.py              # design rule checks (rung 1)
│   ├── solver.py           # OpenSees wall (rung 2)
│   ├── capacity.py         # AISC tension/buckling capacities
│   ├── reward.py           # utilization envelope, cost, reward ladder (rung 3)
│   ├── verifier.py         # score(instance, completion) -> RewardBreakdown
│   ├── baselines/          # heuristic engineer, random sampler, frontier APIs
│   ├── calibration/        # sweep + sanity gates freezing cost_ref/sweep_best
│   ├── evaluation/         # metrics + eval harness
│   ├── training/           # RL stack, gated behind the [training] extra
│   └── utilities/
└── tests/                  # incl. numpy reference solver + differential/determinism tests
```

Design principles:

1. **Verifier is a pure function chain** (`schema → drc → expander/loads → solver → capacity → reward`), composed in `verifier.py`. Each layer is the reward ladder rung it owns and is independently unit-testable.
2. **One geometry source of truth:** `expander.py` is shared by the reward pipeline, calibration, baselines, and eval harness.
3. **Instance as data:** `TrussInstance` carries every randomized quantity; `generator.py` is the dataset, seeded for reproducibility.
4. **RL-dep quarantine:** importing `trussRL.verifier` never pulls torch/vllm — RL imports live strictly inside `trussRL/training/` behind the `[training]` extra.
5. **Frozen artifacts are committed:** `cost_ref`/`sweep_best` are fixed, never policy-dependent, and stamped with git SHA + catalog hash in `artifacts/`.