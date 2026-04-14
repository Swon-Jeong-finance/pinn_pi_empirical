# Stage20 empirical bridge: ff49 lane from dynalloc_v1 → dynalloc_v2

This stage adds a **bridge** from a successful `dynalloc_v1` lane into the new
`dynalloc_v2` APT + conditional factor-covariance engine.

## What the bridge does

Given a legacy config stem such as:

- `ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed`

it will:

1. read the selected top spec from `outputs/<stem>/selected_spec.yaml`,
2. load the legacy ff49 risky-asset panel,
3. add the `curve_core` bond sleeve as monthly **excess returns**,
4. rebuild the legacy PLS state using the selected spec,
5. build an APT factor panel (`ff5 + curve_core` by default),
6. write a v2-ready bundle:
   - `returns_panel.csv`
   - `states_panel.csv`
   - `factors_panel.csv`
   - `config_empirical_ppgdpo_apt.yaml`

The resulting returns panel is written in **excess-return** form, so cash = 0 in
v2 corresponds to the RF benchmark from the legacy world.

## Recommended command

```bash
source .venv/bin/activate
pip install -e .

# build bundle from the stage17 ff49 cv2000 lane
 dynalloc-v2 legacy bridge-v1-lane \
  --v1-root ~/empricial_test/dynalloc_v1_stage17_curve_core_pls_suite \
  --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed \
  --out-dir empirical_bundles/ff49_cv2000_curve_core \
  --factor-mode ff5_curve_core

# inspect generated config
 dynalloc-v2 plan \
  --config empirical_bundles/ff49_cv2000_curve_core/config_empirical_ppgdpo_apt.yaml

# run v2 on the empirical bridge bundle
 dynalloc-v2 run \
  --config empirical_bundles/ff49_cv2000_curve_core/config_empirical_ppgdpo_apt.yaml
```

## Notes

- Supported split profiles in this stage:
  - `cv2000_final20y`
  - `cv2006_final20y`
- Supported factor modes:
  - `ff5_curve_core` (default)
  - `ff3_curve_core`
  - `ff5_only`
- The bridge is intentionally focused on the **successful ff49 / curve_core world**
  first. Once that lane is working end-to-end, the same pattern can be extended
  to ff38, ff25, ff100, and beyond.
