# Stage 13 symmetric universe matrix

Stage 13 extends the stage-12 bond-hook matrix to the remaining universe axis requested for follow-up experiments:

- `ff1` (market-only proxy)
- `ff6` (6 size–book-to-market portfolios, 2x3)
- `ff100` (100 size–book-to-market portfolios, 10x10)

For each universe, stage 13 emits the full symmetric matrix:

- 2 split hooks: `cv1985`, `cv2000`
- 3 bond hooks: `no_bond`, `ust10y`, `curve_core`
- 3 evaluation modes: `fixed`, `expanding`, `rolling`

This yields `2 x 3 x 3 = 18` configs per universe, or `54` configs for the three new universes.

The generator also supports `--universes all` to emit the full five-universe symmetric matrix (`ff1`, `ff6`, `ff25`, `ff49`, `ff100`).

## Usage

```bash
python scripts/generate_stage13_symmetric_matrix.py
python scripts/generate_stage13_symmetric_matrix.py --universes all
```
