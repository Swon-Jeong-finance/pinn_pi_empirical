# Stage 14 ff38 probe

Stage 14 adds a lightweight robustness lane around the ff49 results by introducing
Ken French 38 industry portfolios as a near-neighbor universe.

The intent is not to build another full symmetric matrix. Instead, stage 14 emits
just the three cv2000/fixed configs needed for a quick scale-adjacent check:

- `ff38_stage14_rank_sweep_cv2000_no_bond_fixed.yaml`
- `ff38_stage14_rank_sweep_cv2000_ust10y_fixed.yaml`
- `ff38_stage14_rank_sweep_cv2000_curve_core_fixed.yaml`

Each config keeps the existing top-2 selection profile and rank-sweep comparison
settings, so the experiment flow stays aligned with the ff49/ff100 runs.

## Usage

```bash
python scripts/generate_stage14_ff38_probe.py
```
