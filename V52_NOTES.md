# V52 Notes

This version adds three requested changes.

1. **Regime mean variants in stage2**
   - Added `factor_apt_regime` mean model.
   - Stage2 now expands each covariance candidate into four model variants:
     - base (`factor_apt` + estimated cross)
     - `regime_mean`
     - `gated_cross`
     - `regime_mean_gated_cross`

2. **Regime-gated cross strategy**
   - Added `regime_gated` cross mode.
   - Effective cross matrix is `(1 - p_crisis) * cross_est`, where `p_crisis` comes from
     the regime DCC probability when available, otherwise from the regime-mean factor-energy split.
   - OOS comparison summaries now include `cross_mode = regime_gated`.

3. **Rolling grid narrowed to 10y / 20y / 30y by default**
   - Default integrated rolling grid is now `120 240 360` months.
   - Warm-start semantics remain: early refits use available history until the full window is reached.

Implementation notes:
- `bridge_common._build_v2_config_dict(...)` now supports `mean_model_kind` and `comparison_cross_modes`.
- Selection-generated configs now always include `estimated`, `zero`, and `regime_gated` comparison outputs.
- Final selected models carry extra metadata:
  - `stage2_model_label`
  - `selected_mean_model_kind`
  - `selected_cross_policy_label`

Smoke checks performed:
- `python3 -m py_compile src/dynalloc_v2/*.py`
- `PYTHONPATH=src python3` import checks for parser and stage2 variant expansion.
