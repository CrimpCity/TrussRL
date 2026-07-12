# scripts/

Thin typer CLIs, one per experiment phase (KernelBench pattern). All logic
lives in `src/trussRL/`; scripts only parse arguments, call the library,
and write outputs to `artifacts/`.

- `hss_subset.py` — extract 20 rectangular + 20 square HSS shapes uniformly spaced by strong-axis Ix into one combined CSV
- `plot_geometry.py` — Module 2 eyeball check of expanded geometry
- `calibrate.py` — Module 7 calibration sweep + section 9 gates; freezes `cost_ref` / `sweep_best`
- `run_baselines.py` — Module 8 heuristic engineer + frontier go/no-go
- `rollout_gate.py` — Module 9 base-model rollout gate → cold-start decision
