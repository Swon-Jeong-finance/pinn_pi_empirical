# Stage-12 bond-hook matrix

## Hook choices

- `bond_hook: none` -> no bond asset in the risky menu
- `bond_hook: ust10y` -> add only `UST10Y`
- `bond_hook: curve_core` -> add `UST2Y`, `UST5Y`, `UST10Y`

`fama_market` is accepted as a user-facing alias and is canonicalized to `ff_mkt` internally.

## Selection / rank sweep policy

All matrix configs use:

- `selection.top_k: 2`
- `rank_sweep.start_rank: 1`
- `rank_sweep.end_rank: 2`

So the model-selection phase still ranks the full candidate grid, but follow-up comparison only inspects ranks 1 and 2.

## Selection window modes

Selection diagnostics now support two window aggregations:

- `selection.window_mode: rolling` (default)
- `selection.window_mode: expanding`

The existing top-2 selection profiles remain rolling by default:

- `cv1985_3fold_fullgrid_core_top2`
- `cv2000_3fold_fullgrid_core_top2`

Optional expanding-window profile aliases were added for convenience:

- `cv1985_3fold_fullgrid_core_top2_expanding`
- `cv2000_3fold_fullgrid_core_top2_expanding`

`selection.rolling_window` is reused as:

- the trailing window length in `rolling` mode
- the minimum prefix length before diagnostics are scored in `expanding` mode

## Walk-forward evaluation modes

Method comparison now supports three walk-forward modes through `evaluation.walk_forward_mode`:

- `fixed` (default)
- `expanding`
- `rolling`

For `rolling`, you may optionally set `evaluation.rolling_train_months`. The new stage-12 rolling configs for the 2000-2019 test split use a 432-month trailing train window (matching the original 1964-1999 train-pool length).

## Split variants

### CV-era matrix (1985-1999 test, fixed walk-forward)

- `split.profile: cv1985_final15y`
- `selection.profile: cv1985_3fold_fullgrid_core_top2`
- `evaluation.walk_forward_mode: fixed`

Configs:

- `configs/ff25_stage12_rank_sweep_cv1985_no_bond.yaml`
- `configs/ff25_stage12_rank_sweep_cv1985_curve_core.yaml`
- `configs/ff25_stage12_rank_sweep_cv1985_ust10y.yaml`
- `configs/ff49_stage12_rank_sweep_cv1985_no_bond.yaml`
- `configs/ff49_stage12_rank_sweep_cv1985_curve_core.yaml`
- `configs/ff49_stage12_rank_sweep_cv1985_ust10y.yaml`
- `configs/fama_market_stage12_rank_sweep_cv1985_no_bond.yaml`
- `configs/fama_market_stage12_rank_sweep_cv1985_curve_core.yaml`
- `configs/fama_market_stage12_rank_sweep_cv1985_ust10y.yaml`

### 2000-2019 test matrix (fixed walk-forward)

- `split.profile: cv2000_final20y`
- `selection.profile: cv2000_3fold_fullgrid_core_top2`
- `evaluation.walk_forward_mode: fixed`

Configs:

- `configs/ff25_stage12_rank_sweep_cv2000_no_bond.yaml`
- `configs/ff25_stage12_rank_sweep_cv2000_curve_core.yaml`
- `configs/ff25_stage12_rank_sweep_cv2000_ust10y.yaml`
- `configs/ff49_stage12_rank_sweep_cv2000_no_bond.yaml`
- `configs/ff49_stage12_rank_sweep_cv2000_curve_core.yaml`
- `configs/ff49_stage12_rank_sweep_cv2000_ust10y.yaml`
- `configs/fama_market_stage12_rank_sweep_cv2000_no_bond.yaml`
- `configs/fama_market_stage12_rank_sweep_cv2000_curve_core.yaml`
- `configs/fama_market_stage12_rank_sweep_cv2000_ust10y.yaml`

### 2000-2019 test matrix (rolling walk-forward)

- `split.profile: cv2000_final20y`
- `selection.profile: cv2000_3fold_fullgrid_core_top2`
- `evaluation.walk_forward_mode: rolling`
- `evaluation.rolling_train_months: 432`

Configs:

- `configs/ff25_stage12_rank_sweep_cv2000_no_bond_rolling.yaml`
- `configs/ff25_stage12_rank_sweep_cv2000_curve_core_rolling.yaml`
- `configs/ff25_stage12_rank_sweep_cv2000_ust10y_rolling.yaml`
- `configs/ff49_stage12_rank_sweep_cv2000_no_bond_rolling.yaml`
- `configs/ff49_stage12_rank_sweep_cv2000_curve_core_rolling.yaml`
- `configs/ff49_stage12_rank_sweep_cv2000_ust10y_rolling.yaml`
- `configs/fama_market_stage12_rank_sweep_cv2000_no_bond_rolling.yaml`
- `configs/fama_market_stage12_rank_sweep_cv2000_curve_core_rolling.yaml`
- `configs/fama_market_stage12_rank_sweep_cv2000_ust10y_rolling.yaml`

## Generator

Generate the original 9-config CV-era matrix:

```bash
python scripts/generate_stage12_bond_hook_matrix.py
```

Generate the original 2000-2019 fixed matrix:

```bash
python scripts/generate_stage12_bond_hook_matrix.py --matrix cv2000
```

Generate the new 2000-2019 rolling matrix:

```bash
python scripts/generate_stage12_bond_hook_matrix.py --matrix cv2000_rolling
```

Generate all 18 fixed configs:

```bash
python scripts/generate_stage12_bond_hook_matrix.py --matrix both
```

Generate all 27 configs (18 fixed + 9 rolling):

```bash
python scripts/generate_stage12_bond_hook_matrix.py --matrix all
```
