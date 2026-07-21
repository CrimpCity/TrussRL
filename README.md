# TrussRL
Deterministic RLVR environment for single span 2D truss design

### Setup

```bash
uv lock && uv sync && source .venv/bin/activate
```

### Project structure

```
TrussRL/
├── data/                       # frozen AISC HSS inputs and 40-section subset
├── artifacts/                  # committed calibration outputs and gate reports
├── scripts/                    # repository-oriented utilities (namespace package)
│   ├── data/                   # catalog filtering, subsetting, and generation
│   ├── validation/             # hand-checks for the OpenSees solver wall
│   └── inspection/             # human-readable reports and geometry plots
├── src/trussRL/
│   ├── cli/                    # supported command adapters; pipeline is published
│   │   ├── pipeline.py         # inspect one design through every verifier stage
│   │   ├── calibrate.py        # planned, intentionally unimplemented
│   │   ├── baselines.py        # planned, intentionally unimplemented
│   │   └── rollout_gate.py     # planned, intentionally unimplemented
│   ├── catalog.py              # section records from the frozen catalog
│   ├── schema.py               # TrussDesign model + JSON extraction (rung 0)
│   ├── instance.py             # prompt-visible data and grading parameters
│   ├── generator.py            # seeded procedural instance generator
│   ├── prompts.py              # TrussInstance -> prompt text
│   ├── expander.py             # design -> geometry (source of truth)
│   ├── loads.py                # line load -> panel points + self-weight
│   ├── drc.py                  # design rule checks (rung 1)
│   ├── solver.py               # OpenSees wall (rung 2)
│   ├── capacity.py             # AISC tension/buckling capacities
│   ├── reward.py               # utilization, cost, and reward ladder (rung 3)
│   ├── verifier.py             # score(instance, completion) -> RewardBreakdown
│   ├── baselines/              # heuristic, random sampler, and frontier APIs
│   ├── calibration/            # sweep + sanity gates
│   ├── evaluation/             # metrics + evaluation harness
│   ├── training/               # RL stack, gated behind the training extra
│   └── utilities/
└── tests/                      # reference solver, differential, and unit tests
```

### Pipeline CLI

Run the supported pipeline command through its installed console script:

```bash
uv run trussrl-pipeline
```

The same interface is available through Python module execution:

```bash
uv run python -m trussRL.cli.pipeline
```

Pass `--help` to either form for the complete Typer option list.

### Repository utilities

Run repository utilities as namespace-package modules from the repository root.
Representative commands are:

```bash
uv run python -m scripts.data.generate_catalog
uv run python -m scripts.validation.simple_truss_check
uv run python -m scripts.inspection.check_wall_slenderness
uv run python -m scripts.inspection.plot_geometry --save /tmp/truss.png
```

Design principles:

1. **Verifier is a pure function chain** (`schema → drc → expander/loads → solver → capacity → reward`), composed in `verifier.py`. Each layer is the reward ladder rung it owns and is independently unit-testable.
2. **One geometry source of truth:** `expander.py` is shared by the reward pipeline, calibration, baselines, and eval harness.
3. **Instance as data:** `TrussInstance` carries every randomized quantity; `generator.py` is the dataset, seeded for reproducibility.
4. **RL-dep quarantine:** importing `trussRL.verifier` never pulls torch/vllm — RL imports live strictly inside `trussRL/training/` behind the `[training]` extra.
5. **Frozen artifacts are committed:** `cost_ref`/`sweep_best` are fixed, never policy-dependent, and stamped with git SHA + catalog hash in `artifacts/`.
