# Stage41: calendar-window overrides for selection and OOS setup

This stage keeps the Stage40 split/protocol refactor, and adds a config-like window override layer so the same code can be reused for alternative validation/test calendars without editing legacy configs.

## What changed

- Added `src/dynalloc_v2/experiment_windows.py`
  - named split profiles:
    - `cv2000_final20y`
    - `cv2006_final20y`
  - explicit date overrides on top of a profile
- Bridge commands now accept split overrides
  - `legacy bridge-v1-base-lane`
  - `legacy bridge-v1-base-grid`
  - `legacy bridge-v1-lane`
  - `legacy bridge-v1-suite`
- Native selection now accepts the same split overrides
  - `select-native-suite`
- Selection manifests and selected-spec YAML now record
  - `selection_split_profile`
  - `split_source`
  - `split_description`
  - `split_overrides`

## New CLI options

All bridge/selection commands now support:

```bash
--split-profile cv2006_final20y
--split-train-start YYYY-MM-DD
--split-train-pool-end YYYY-MM-DD
--split-test-start YYYY-MM-DD
--split-end-date YYYY-MM-DD
```

## Typical use

### 1) Build a 2006-2025 base bundle

```bash
dynalloc-v2 legacy bridge-v1-base-lane \
  --v1-root ~/empricial_test/dynalloc_v1_stage12_bond_hook_matrix_cv2000_rolling_fresh \
  --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed \
  --out-dir base_bundles/ff49_cv2006_curve_core \
  --split-profile cv2006_final20y
```

### 2) Run native selection on that shifted window

```bash
dynalloc-v2 select-native-suite \
  --base-dir base_bundles/ff49_cv2006_curve_core \
  --out-dir empirical_suites_native/ff49_stage41_cv2006 \
  --candidate-zoo factor_zoo_v1 \
  --top-k 2 \
  --rerank-top-n 3 \
  --selection-split-mode trailing_holdout \
  --selection-val-months 240 \
  --selection-device cuda:5
```

### 3) Then run OOS protocols as usual

```bash
dynalloc-v2 run-rank-sweep \
  --manifest empirical_suites_native/ff49_stage41_cv2006/suite_manifest.yaml \
  --device cuda:5 \
  --ppgdpo-mc-rollouts 256 \
  --ppgdpo-mc-sub-batch 256 \
  --oos-protocols fixed expanding_annual rolling20y_annual
```

## Backward compatibility

If you do not pass any split overrides, behavior stays the same as Stage40.
