# Stage21: Top-2 suite bridge and rank-sweep

This stage restores a light-weight orchestration layer on top of `dynalloc_v2`.

## New commands

### `bridge-v1-suite`
Build a top-k empirical suite from an existing v1 lane. It reads
`outputs/<config_stem>/selected_spec.yaml` from v1, bridges rank 1..k into
separate `rank_001`, `rank_002`, ... bundle directories, and writes a
`suite_manifest.yaml`.

Example:

```bash
 dynalloc-v2 legacy bridge-v1-suite \
   --v1-root ~/empricial_test/dynalloc_v1_stage17_curve_core_pls_suite \
   --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed \
   --out-dir empirical_suites/ff49_cv2000_curve_core \
   --factor-mode ff5_curve_core \
   --top-k 2
```

### `run-rank-sweep`
Run all entries in a suite manifest and aggregate the results into a protocol-first
rank-sweep directory with per-rank protocol folders and top-level summary CSVs.

Example:

```bash
 dynalloc-v2 run-rank-sweep \
   --manifest empirical_suites/ff49_cv2000_curve_core/suite_manifest.yaml
```

## Output structure

```text
<out_dir>/
  suite_manifest.yaml
  rank_001/
    config_empirical_ppgdpo_apt.yaml
    bridge_metadata.yaml
  rank_002/
    ...
  rank_sweep/
    rank_001/protocols/fixed/comparison/comparison_cross_modes_zero_cost_summary.csv
    rank_002/protocols/fixed/comparison/comparison_cross_modes_zero_cost_summary.csv
    rank_sweep_cross_modes_zero_cost_summary.csv
    rank_sweep_cross_modes_all_costs_summary.csv
    rank_sweep_results.csv
    rank_sweep_results.jsonl
    rank_sweep_progress.csv
```

If an older notebook still expects `rank_sweep/rank_XXX/comparison/...`, rerun with
`--emit-legacy-fixed-layout` to recreate that fixed-only compatibility layout.
