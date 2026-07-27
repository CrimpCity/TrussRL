# artifacts/

Committed outputs from the calibration workflows in `trussRL.calibration`
(with the planned command adapter at `trussRL.cli.calibrate`), frozen per
vision.md section 9. Each artifact is stamped with the git SHA and catalog hash
it was produced from. These values must never be policy-dependent.

- `cost_ref.json` — per-instance cost normalization reference
- `sweep_best.json` — best score found by the random sweep per instance
- `train_roster.json` — frozen 512-instance training roster with per-instance
  `cost_ref_usd`, disjoint from the 32 calibrated instances; pins both
  calibration artifacts by run_id and sha256. Built with
  `uv run trussrl-train-roster` (via `trussRL.cli.train_roster`)
- `gate_reports/` — section 9 sanity-gate reports from calibration runs
- `reward_bench/` — pool-vs-serial reward adapter benchmarks with
  failure-injection results and the provisional adoption verdict
