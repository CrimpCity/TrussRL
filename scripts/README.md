# scripts/

Repository-oriented utilities live here, separate from supported application
commands in `trussRL.cli`. The directories intentionally have no `__init__.py`;
run their files as namespace-package modules from the repository root.

## Data

- `scripts.data.hss_filter` splits `data/HSS.csv` into square, rectangular,
  and round catalogs while preserving the original header, field strings, and
  line endings.
- `scripts.data.hss_subset` selects the frozen 20 rectangular + 20 square
  sections, spaced uniformly by strong-axis `Ix`, into `data/HSS_subset.csv`.
- `scripts.data.generate_catalog` deterministically renders
  `src/trussRL/catalog_data.py` from the frozen subset. Never edit that generated
  module by hand.

```bash
uv run python -m scripts.data.hss_filter
uv run python -m scripts.data.hss_subset
uv run python -m scripts.data.generate_catalog
```

## Validation

- `scripts.validation.simple_beam_check` checks reconstructed reactions and
  midspan moment for a hand-calculable two-bay Warren truss.
- `scripts.validation.simple_truss_check` checks every member axial force and
  both reactions for a single-triangle truss.

```bash
uv run python -m scripts.validation.simple_beam_check
uv run python -m scripts.validation.simple_truss_check
```

## Inspection

- `scripts.inspection.check_wall_slenderness` reports the Chapter E7 path for
  all 40 frozen-catalog sections.
- `scripts.inspection.plot_geometry` draws expanded geometry; it opens a window
  unless `--save` is provided.

```bash
uv run python -m scripts.inspection.check_wall_slenderness
uv run python -m scripts.inspection.plot_geometry --save /tmp/truss.png
```

## Models

- `scripts/models/download_qwen.py` resolves `Qwen/Qwen3-4B-Base` to a full
  commit SHA and downloads that exact snapshot to the revision-addressed
  Hugging Face cache. It uses the project `training` extra and does not create
  a repository-local model directory.

Download the current `main` revision:

```bash
uv run --extra training python -m scripts.models.download_qwen
```

Record the printed full SHA. To reproduce that exact checkpoint later:

```bash
uv run --extra training python -m scripts.models.download_qwen \
    --revision <full-commit-sha>
```

- `scripts/models/vllm_smoke.py` is the Linux/A100 smoke test for the locked
  vLLM stack. It runs a trivial Numba JIT kernel, requires CUDA, locates the
  pinned Qwen snapshot in the Hugging Face cache without downloading
  (`HF_HUB_OFFLINE=1`), initializes vLLM in bfloat16, and generates one token.
  It cannot run on macOS; execute it on the CUDA training host after
  downloading the checkpoint:

```bash
uv run --extra training python -m scripts.models.vllm_smoke
```
