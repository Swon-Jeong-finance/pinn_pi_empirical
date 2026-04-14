V50 changes
===========

- `top-k` now means final number of models written to the manifest.
- Stage 1 is now a screen, not the final winner list.
  - New optional CLI flag: `--stage1-top-k`
  - If omitted, stage1 automatically keeps `max(top-k, rerank-top-n, 8)` spec-protocol pairs.
- Stage 2 still picks the best covariance model within each surviving spec-protocol pair.
- Final winners are now chosen by a **global stage2 rerank across all stage1 survivors**, instead of preserving stage1 order.
- Manifest/selected YAML now record both the stage1 survivor count and the final requested `top-k`.
- Default rerank covariance grid now includes ADCC: `const dcc adcc regime_dcc`.
