# V55 notes

## Goal

V55 is a benchmark-cleanup release on top of v54.

- Keep the v54 global stage2 rerank behavior.
- Clean up strategy labels in OOS outputs.
- Add standard benchmark rows so rolling-20y comparisons are easier to read.

## OOS label cleanup

In `ppgdpo` experiment outputs:

- `myopic` -> `predictive_static`
- `policy` -> `pgdpo`
- projected strategies remain under `ppgdpo`, with display labels:
  - `ppgdpo`
  - `ppgdpo_zero`
  - `ppgdpo_regime_gated`

For backward compatibility, CSVs also carry `strategy_legacy_label`.

## Benchmark suite

When `comparison.include_standard_benchmarks: true` (default), OOS outputs include:

- `equal_weight`
- `market`
- `min_variance`
- `risk_parity`

Implementation notes:

- `equal_weight`, `min_variance`, and `risk_parity` are long-only and respect the same risky-cap / cash-floor feasibility budget.
- `min_variance` and `risk_parity` use trailing training-window sample covariance, not the selected predictive mean model.
- `market` uses a detected market factor column when available (`MKT`, `Mkt-RF`, etc.).
- If no market factor column is available, `market` falls back to an equal-weight proxy and the run writes that note into `benchmark_notes.yaml`.

## New run artifacts

`ppgdpo` runs now write:

- `benchmark_notes.yaml`

and the summary / monthly CSVs now include:

- `strategy_display`
- `strategy_legacy_label`
- `comparison_role`
- `benchmark_primary`
- `benchmark_note`
- `benchmark_source`

## Native selection compatibility

Stage2 selection logic is unchanged.

To keep old downstream code working while making logs easier to read:

- legacy `myopic`-named fields are still produced where needed,
- predictive-static aliases are added alongside them, e.g.
  - `ppgdpo_lite_predictive_static_ce_mean`
  - `ppgdpo_lite_ce_delta_predictive_static_mean`
- suite manifest / selected spec now include a small v55 label map and benchmark note block.
