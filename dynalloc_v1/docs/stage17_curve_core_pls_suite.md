# Stage 17: curve-core only PLS-family suite

This stage narrows the experiment matrix to the curve-core risky menu and removes FF1 from the suite.

## Design
- Universes: FF6, FF25, FF38, FF49, FF100
- Bond hook: curve_core only
- Splits: cv2000, cv2006
- Evaluation: fixed walk-forward
- Selection family: PLS-only candidate grid

## Selection family
The stage-17 selection profiles restrict candidates to a compact PLS family:
- `pls_H{12,24}_k{2,3}`
- `pls_ret_macro7_H{12,24}_k{2,3}`
- `pls_ret_ff5_macro7_H{12,24}_k{2,3}`

This keeps the search in a representation-learning world instead of mixing raw macro, PCA, and broad static baselines.

## Rationale
The stage-12 FF49 curve-core success appeared to come from latent PLS state compression rather than from raw macro predictors alone. Stage 17 focuses on that world directly.
