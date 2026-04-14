# Stage 47 — ADCC covariance candidate

This stage adds an asymmetric DCC (`asset_adcc`) covariance model and exposes it through native selection rerank candidates via `adcc`.

## What changed

- `src/dynalloc_v2/covariance.py`
  - added `AssetADCCCovariance`, which augments `asset_dcc` with a negative-shock asymmetry term `adcc_gamma`
- `src/dynalloc_v2/schema.py`
  - `covariance_model.kind` now accepts `asset_adcc`
  - `covariance_model.adcc_gamma` added with default `0.005`
- `src/dynalloc_v2/native_selection.py`
  - `--rerank-covariance-models` can now include `adcc`
  - selected bundle configs persist `asset_adcc` and `adcc_gamma` into generated YAML
- `src/dynalloc_v2/experiments.py`
  - direct experiment runs now support `covariance_model.kind: asset_adcc`

## Intended use

Use `adcc` first as an additional stage-2 covariance rerank candidate. This makes it easy to test whether the asymmetric covariance model wins within the current top-ranked factor specifications before widening the stage-1 candidate funnel.
