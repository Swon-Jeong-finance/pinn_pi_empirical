# Stage 15 cv2006 fixed-18 matrix

Stage 15 adds a new 20-year test split covering **2006-01 through 2025-12** and
pre-generates the fixed walk-forward matrix for six universes:

- `ff1`
- `ff6`
- `ff25`
- `ff38`
- `ff49`
- `ff100`

Each universe is paired with the three bond-hook variants:

- `no_bond`
- `ust10y`
- `curve_core`

That yields **18 fixed configs** in total. Each config keeps the existing top-2
selection profile and native stage-7 rank-sweep comparison settings.

## New split

- `profiles/split/cv2006_final20y.yaml`
- `src/dynalloc/profile_data/split/cv2006_final20y.yaml`

with

- `train_pool_start: 1964-01-01`
- `train_pool_end: 2005-12-31`
- `final_test_start: 2006-01-01`
- `end_date: 2025-12-31`

## New selection alias

- `profiles/selection/cv2006_3fold_fullgrid_core_top2.yaml`
- `src/dynalloc/profile_data/selection/cv2006_3fold_fullgrid_core_top2.yaml`

## Usage

```bash
python scripts/generate_stage15_cv2006_fixed18.py
```
