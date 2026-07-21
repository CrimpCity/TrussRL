# artifacts/

Committed outputs from the calibration workflows in `trussRL.calibration`
(with the planned command adapter at `trussRL.cli.calibrate`), frozen per
vision.md section 9. Each artifact is stamped with the git SHA and catalog hash
it was produced from. These values must never be policy-dependent.

- `cost_ref.json` — per-instance cost normalization reference
- `sweep_best.json` — best score found by the random sweep per instance
- `gate_reports/` — section 9 sanity-gate reports from calibration runs
