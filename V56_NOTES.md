# V56 Notes

This release focuses on cleaning up the FF49 selection/OOS comparison stack.

## What changed

- Removed the dead residual-covariance ranking term from stage-2 native selection.
  - `resid_cov_loglik_mean` no longer appears in the stage-2 summary.
  - `stage2_rank_resid_cov` is removed from the stage-2 scoring path.
  - The stage-2 global rerank now uses:
    - `ppgdpo_lite_score_mean` (primary)
    - `ppgdpo_lite_score_q10`
    - `ppgdpo_lite_ce_delta_zero_mean`

- Added reference-only external benchmark diagnostics to stage-2 selection outputs.
  - New summary columns:
    - `ppgdpo_lite_equal_weight_ce_mean`
    - `ppgdpo_lite_min_variance_ce_mean`
    - `ppgdpo_lite_risk_parity_ce_mean`
    - `ppgdpo_lite_ce_delta_equal_weight_mean`
    - `ppgdpo_lite_ce_delta_min_variance_mean`
    - `ppgdpo_lite_ce_delta_risk_parity_mean`
  - These are report-only and do **not** enter the selection score.

- Simplified the default OOS benchmark set.
  - Default external benchmarks are now:
    - `equal_weight`
    - `min_variance`
    - `risk_parity`
  - `market` is no longer part of the default benchmark bundle.
  - Backward compatibility is preserved if an older config explicitly requests `market`.

- Updated selection manifests / selected-spec payloads.
  - Added a generic `strategy_label_map` alongside the legacy `v55_strategy_label_map` key.
  - `comparison_benchmark_notes.external_benchmarks` now lists only:
    - `equal_weight`
    - `min_variance`
    - `risk_parity`

- Updated benchmark notes output.
  - `benchmark_notes.yaml` now reports version `v56`.
  - `market_benchmark_source` is only written if `market` is explicitly enabled.

## Notes

- This release does **not** change the default transaction cost settings.
  - If `transaction_cost_bps = 0.0`, the zero-cost and all-cost summaries will still match.
- `pgdpo` remains available as a reference-only internal comparison.
