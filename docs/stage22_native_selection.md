# Stage22: Native v2 Selection Recovery

This stage restores a **native v2 selection layer** so that top-k candidates can be chosen *inside* `dynalloc_v2` rather than being read from legacy `selected_spec.yaml`.

## What is new

- `dynalloc-v2 legacy bridge-v1-base-lane`
  - exports a **raw base bundle** from any v1 lane stem
  - writes:
    - `returns_panel.csv`
    - `macro_panel.csv`
    - `ff3_panel.csv`
    - `ff5_panel.csv`
    - `bond_panel.csv`
    - `base_bundle_manifest.yaml`

- `dynalloc-v2 select-native-suite`
  - builds a **PLS-only candidate pool** natively in v2
  - current pool mirrors the stage17 family:
    - `pls_H12_k2`, `pls_H12_k3`, `pls_H24_k2`, `pls_H24_k3`
    - `pls_ret_macro7_H12_k2`, `pls_ret_macro7_H12_k3`, `pls_ret_macro7_H24_k2`, `pls_ret_macro7_H24_k3`
    - `pls_ret_ff5_macro7_H12_k2`, `pls_ret_ff5_macro7_H12_k3`, `pls_ret_ff5_macro7_H24_k2`, `pls_ret_ff5_macro7_H24_k3`
  - scores candidates with blocked CV
  - emits `selected_spec.yaml`, `spec_selection_summary.csv`, and a `suite_manifest.yaml`
  - creates `rank_001`, `rank_002`, ... bundles directly for `run-rank-sweep`

## Universe support

The base bridge is generic and can be used with the original v1 worlds as long as the lane exists in v1:

- `ff1`
- `ff6`
- `ff25`
- `ff38`
- `ff49`
- `ff100`

## Example

```bash
# 1) build raw base bundle from a legacy lane
 dynalloc-v2 legacy bridge-v1-base-lane \
   --v1-root ~/empricial_test/dynalloc_v1_stage17_curve_core_pls_suite \
   --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed \
   --out-dir base_bundles/ff49_cv2000_curve_core

# 2) native v2 selection + top-k suite build
 dynalloc-v2 select-native-suite \
   --base-dir base_bundles/ff49_cv2000_curve_core \
   --out-dir empirical_suites_native/ff49_cv2000_curve_core \
   --factor-mode ff5_curve_core \
   --top-k 2

# 3) run top-k comparison with the new engine
 dynalloc-v2 run-rank-sweep \
   --manifest empirical_suites_native/ff49_cv2000_curve_core/suite_manifest.yaml
```

## Scope note

This stage **does not** yet recover the full legacy stage orchestrator. It restores the critical workflow:

- raw bundle export
- native v2 selection
- top1/top2 rank sweep

inside the new v2 architecture.
