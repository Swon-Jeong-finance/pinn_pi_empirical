# Stage36: trailing 20y validation with train+val retrain for final OOS

## What changed

- Native selection now defaults to `selection_split_mode: trailing_holdout` with `selection_val_months: 240`.
- The selection pool respects the bridge split start/end dates.
- Stage1 cheap screening and stage2 real P-PGDPO rerank both use the same trailing validation block by default.
- Final OOS rank-sweep still retrains on the full pre-test pool (`train_start` through `train_pool_end`), so the validation period is only held out for model selection, not discarded before the final backtest.

## Why

Short validation folds made CE-based hedging diagnostics too sensitive to single regimes and outliers. A single trailing 20-year validation block is closer to the final OOS horizon and makes `ce_gain_vs_myopic` and `ce_gain_vs_zero` easier to interpret apples-to-apples.

## CLI

```bash
dynalloc-v2 select-native-suite \
  --base-dir ... \
  --out-dir empirical_suites_native/ff49_stage36_debug \
  --candidate-zoo factor_zoo_v1 \
  --top-k 2 \
  --rerank-top-n 3 \
  --min-train-months 120 \
  --selection-split-mode trailing_holdout \
  --selection-val-months 240
```

To restore the older expanding-CV behavior:

```bash
dynalloc-v2 select-native-suite ... --selection-split-mode expanding_cv --cv-folds 3
```

## Metadata

`selected_spec.yaml` and `suite_manifest.yaml` now include a `selection_split` block that records:

- selection mode
- train pool start/end
- validation block dates and lengths
- final test start/end
- whether final OOS retrains on train+validation (`true`)


## High-MC follow-up patch

This patch keeps the trailing 20-year validation split from stage36 but raises the default PPGDPO Monte Carlo averaging budgets to `mc_rollouts=256` and `mc_sub_batch=256` for both stage2 selection reranking and generated final OOS experiment configs.
