# dynalloc v1.10 — stage10 split hooks + log-safe cross-mode outputs + residual-policy bugfixes

This version keeps the stage-7 rank-sweep / multi-worker GPU orchestration, and adds three practical fixes for long empirical runs:

- **split hooks** for CV-era experiments:
  - optional `split.train_pool_start`
  - `split.force_common_calendar`
  - `split.common_start_mode`
- **log-safe cross-mode outputs** so `estimated` and `zero` no longer overwrite each other when the same spec is run under multiple modes;
- **residual-policy bugfixes** in the vendored PG-DPO runner:
  - wire experiment constraints into `ResidualPolicyConfig`;
  - refresh residual-policy myopic baselines in expanding-window re-fit paths.

## New stage-10 profiles and configs

- split profile: `profiles/split/cv1985_final15y.yaml`
  - `train_pool_start: 1964-01-01`
  - `train_pool_end: 1984-12-31`
  - `final_test_start: 1985-01-01`
  - `end_date: 1999-12-31`
- selection profile alias: `profiles/selection/cv1985_3fold_fullgrid_core.yaml`
- configs:
  - `configs/ff25_stage10_curve_core_rank_sweep_cv1985.yaml`
  - `configs/ff49_stage10_curve_core_rank_sweep_cv1985.yaml`

## Why the new split hook matters

Previously, the train pool always started at the earliest common aligned date. That is still the default, but stage 10 lets you clip the aligned sample after calendar intersection, which makes CV-era train/test studies much easier without touching the data loader.

## Why the log fix matters

Previously, stage3b logs were keyed only by spec name, so the `zero` run could overwrite the `estimated` run for the same spec. Stage 10 scopes logs by comparison output directory, so rank and cross-mode are preserved automatically.

## What stage 7 adds

- `runtime.backend: native_stage7`
- new `rank_sweep` config block:
  - `enabled`
  - `start_rank`
  - `end_rank`
  - `max_ranks`
  - `resume`
  - `stop_on_error`
  - `output_subdir`
- append-only sweep artifacts:
  - `rank_sweep/rank_sweep_progress.csv`
  - `rank_sweep/rank_sweep_results.csv`
  - `rank_sweep/rank_sweep_results.jsonl`
  - `rank_sweep/rank_sweep_cross_modes_zero_cost_summary.csv`
  - `rank_sweep/rank_sweep_cross_modes_all_costs_summary.csv`
- per-rank output folders:
  - `rank_sweep/rank_001/...`
  - `rank_sweep/rank_002/...`
  - ...
- per-rank comparison bundles that keep `estimated` and `zero` together for the same selected rank.

## New stage-7 configs

- `configs/ff25_stage7_curve_core_rank_sweep.yaml`
- `configs/ff49_stage7_curve_core_rank_sweep.yaml`

Both configs keep the stage-6 curve-core setup, but turn on the rank sweep backend.

## Typical workflow

Validate and inspect the config:

```bash

dynalloc validate --config configs/ff25_stage7_curve_core_rank_sweep.yaml
dynalloc plan --config configs/ff25_stage7_curve_core_rank_sweep.yaml
```

Run selection once:

```bash

dynalloc run --config configs/ff25_stage7_curve_core_rank_sweep.yaml --phase selection
```

Then launch multiple comparison workers against the same output directory. For example, on GPUs 2..7:

```bash

dynalloc run --config configs/ff25_stage7_curve_core_rank_sweep.yaml --phase comparison --worker-id ff25-gpu2 --device cuda:2 &
dynalloc run --config configs/ff25_stage7_curve_core_rank_sweep.yaml --phase comparison --worker-id ff25-gpu3 --device cuda:3 &
dynalloc run --config configs/ff25_stage7_curve_core_rank_sweep.yaml --phase comparison --worker-id ff25-gpu4 --device cuda:4 &
dynalloc run --config configs/ff25_stage7_curve_core_rank_sweep.yaml --phase comparison --worker-id ff25-gpu5 --device cuda:5 &
dynalloc run --config configs/ff25_stage7_curve_core_rank_sweep.yaml --phase comparison --worker-id ff25-gpu6 --device cuda:6 &
dynalloc run --config configs/ff25_stage7_curve_core_rank_sweep.yaml --phase comparison --worker-id ff25-gpu7 --device cuda:7 &
wait
```

You can do the same for FF49 with a separate output directory.

## Helper script

A simple launcher is included at:

- `scripts/launch_stage7_rank_workers.sh`

Example:

```bash

bash scripts/launch_stage7_rank_workers.sh configs/ff25_stage7_curve_core_rank_sweep.yaml 2 3 4 5 6 7
```

## Main output layout

```text
outputs/<experiment>/
  selection/
    spec_selection_summary.csv
    stage7_selection_report.yaml
  rank_sweep/
    rank_sweep_progress.csv
    rank_sweep_results.csv
    rank_sweep_results.jsonl
    rank_sweep_cross_modes_zero_cost_summary.csv
    rank_sweep_cross_modes_all_costs_summary.csv
    stage7_rank_sweep_report.yaml
    rank_001/
      stage7_rank_report.yaml
      comparison/
        stage6_comparison_report.yaml
        comparison_cross_modes_zero_cost_summary.csv
        comparison_cross_modes_all_costs_summary.csv
        estimated/
        zero/
```

## Resume semantics

- completed ranks get a `_done.yaml` marker;
- failed ranks get a `_failed.yaml` marker;
- with `rank_sweep.resume: true`, later workers skip ranks that already have one of those markers.

That means a long run can be resumed without recomputing finished ranks.

## Backward compatibility

- `native_stage6` continues to support single-rank selected-spec comparison.
- `native_stage7` delegates back to the stage-6 path when:
  - `rank_sweep.enabled: false`, or
  - `--selected-rank` is passed explicitly.

## Earlier stages

The earlier native backends remain available:

- `native_stage3a`
- `native_stage3b`
- `native_stage4`
- `native_stage5`
- `native_stage6`

Stage 7 is the first version that treats the ranked selection table as a distributed work queue.


## Stage 11

CV-era configs now use `macro.profile: bond_curve_core_cv`, and `macro.pool: custom` is supported so the shortened 1960s-friendly macro set loads correctly.


## Stage 15

- Added `cv2006_final20y` split for 2006-2025 testing.
- Added `cv2006_3fold_fullgrid_core_top2` selection alias.
- Added 18 pre-generated fixed configs across ff1/ff6/ff25/ff38/ff49/ff100.
