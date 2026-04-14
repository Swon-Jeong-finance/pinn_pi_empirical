# V53 notes

Core changes:

- Rolling selection simplified: default integrated protocol grid is now a single warm-start 20-year rolling protocol (`rolling240m_annual`).
- `regime_mean` moved from stage-2 model variants into the stage-1 selection unit semantics by duplicating stage-1 candidates across mean families (`baseline_mean`, `regime_mean`).
- Stage-2 is now reserved for covariance and cross-policy selection:
  - `dcc`: `estimated`, `zero`
  - `adcc`: `estimated`, `zero`
  - `regime_dcc`: `estimated`, `zero`, `regime_gated`
- `gated_cross` is restricted to `regime_dcc` models only.
- Final bundle configs only request `regime_gated` comparisons when the chosen covariance model is `regime_dcc`.

This version is intended to be lighter and conceptually cleaner than V52.
