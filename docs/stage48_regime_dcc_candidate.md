# Stage 48 — Regime DCC covariance candidate

This stage adds a two-regime asset-level DCC covariance candidate (`asset_regime_dcc`) intended for the main residual-covariance comparison axis.

## What changed

- `covariance.py`
  - added `AssetRegimeDCCCovariance`, a two-state residual DCC variant that
    - splits in-sample residual history into low/high-volatility regimes using a fitted energy threshold,
    - maintains separate DCC states for each regime,
    - predicts a blended covariance using the current smoothed regime score.
- `schema.py`
  - `covariance_model.kind` now accepts `asset_regime_dcc`
  - added configuration fields
    - `regime_threshold_quantile`
    - `regime_smoothing`
    - `regime_sharpness`
- `experiments.py`
  - direct experiment runs now support `covariance_model.kind: asset_regime_dcc`
- `native_selection.py` / `cli.py`
  - `--rerank-covariance-models` can now include `regime_dcc`
  - the default stage-2 covariance rerank list is now `const dcc regime_dcc`
  - `diag` remains available but is no longer part of the main default comparison set.
- tests
  - added PSD/update coverage for `AssetRegimeDCCCovariance`
  - added native selection coverage for the `regime_dcc` rerank label.

## Intent

The goal is to keep the paper on the “trusted structural model” track while widening the residual covariance family to include a candidate that can adapt to repeated high/low correlation regimes without moving to a full uncertainty-aware or ensemble-control framing.
