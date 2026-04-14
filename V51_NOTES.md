# V51 notes

This build extends the integrated stage1/stage2 protocol selection introduced in v50.

## What changed

- The default integrated rolling protocol grid now emphasizes medium-to-long memory lengths:
  - `rolling120m_annual`
  - `rolling180m_annual`
  - `rolling240m_annual`
  - `rolling300m_annual`
  - `rolling360m_annual`
- `fixed` and `expanding_annual` remain in the default candidate set.
- Rolling protocol semantics are now described explicitly as **warm-start rolling**:
  - early validation/OOS refits use the available history
  - once enough history accumulates, the requested full rolling window is used
- Manifest and selected-spec metadata now record the warm-start rolling semantics.

## Why

The recent experiments suggested that memory length should be treated as part of model selection, not as an afterthought applied only in final OOS. In particular, 20y/25y/30y rolling windows should be allowed to compete in both stage1 and stage2 even when the earliest validation dates do not yet have the full requested history.

## Typical command

```bash
dynalloc-v2 select-native-suite \
  --base-dir ~/empricial_test/shared/base_bundles/ff49_cv2000_curve_core \
  --out-dir ~/empricial_test/shared/experiments/v51_suite_cv2000 \
  --candidate-zoo factor_zoo_v1 \
  --top-k 2 \
  --stage1-top-k 12 \
  --selection-device cuda
```

If you still want short-memory protocols, pass them explicitly with `--rolling-oos-window-grid`.
