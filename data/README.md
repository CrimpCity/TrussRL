# data/

Frozen input data committed to git.

- `aisc_hss.csv` (Module 1) — the section catalog, single source of truth
  for member properties. Once frozen, changing it invalidates committed
  calibration artifacts (they are stamped with the catalog hash).
