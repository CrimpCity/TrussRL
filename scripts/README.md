# scripts/

Thin typer CLIs, one per experiment phase. All logic lives in `src/trussRL/`.
Scripts only parse arguments, call the library, and write outputs to `artifacts/`.

- `hss_filter.py` (Module 1) — split `data/HSS.csv` by cross-section shape into
  `data/HSS_square.csv`, `data/HSS_rectangular.csv`, and `data/HSS_round.csv`.
  Partitions rows only: the header, column layout, and field strings are
  byte-faithful to the source.
- `hss_subset.py` (Module 1) — select 20 rectangular + 20 square shapes,
  uniformly spaced by strong-axis `Ix` and including the smallest and largest of
  each family, from the shape-family splits into `data/HSS_subset.csv` in
  ascending-`Ix` order. This is the frozen 40-shape section catalog.
- `generate_catalog.py` (Module 1) — generate `src/trussRL/catalog_data.py` from
  `data/HSS_subset.csv`: one dict entry per shape holding every property that
  parses as a float, plus a derived `cost_per_ft_usd`. Rendering is deterministic,
  so re-running on an unchanged CSV leaves an empty `git diff`. Never edit the
  generated module by hand — regenerate it.
- `plot_geometry.py` (Module 2) — expand a design (`--span`, `--n-bays`,
  `--depth`, `--truss-type`) and plot its nodes and members to visually check the
  geometry. Opens a window, or writes a PNG with `--save`.
- `calibrate.py` (Module 7) — run 5k+ random designs through the verifier, apply
  the vision.md section 9 sanity gates, and write the frozen `cost_ref` and
  `sweep_best` values (stamped with git SHA + catalog hash) to `artifacts/`.
- `run_baselines.py` (Module 8) — run the heuristic engineer and the
  frontier-model go/no-go check, retaining verifier-passing frontier rollouts for
  potential cold-start SFT.
- `rollout_gate.py` (Module 9) — sample base-model rollouts on generated
  instances and measure parse/DRC pass rates to decide whether cold-start SFT is
  needed before GRPO.

Run each as a module from the repo root, e.g. `uv run python -m scripts.hss_filter`.
