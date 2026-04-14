# Stage35 real PPGDPO selection wiring

This stage fixes the remaining native-selection wiring gap from stage34.

## What changed

- Stage2 block score now uses the real PPGDPO rerank score:
  - `0.50 * CE(ppgdpo_est)`
  - `+ 0.30 * gain_vs_myopic`
  - `+ 0.20 * gain_vs_zero`
- Stage2 model ranking is now anchored on the real PPGDPO score instead of myopic CE.
- Hedging gain (`gain_vs_zero`) now enters the stage2 winner selection explicitly.
- For each stage1-selected spec, the best covariance model is chosen in stage2.
- `selected_spec.yaml` and `suite_manifest.yaml` now contain only the stage2-selected model per spec (instead of the full covariance grid).
- Backward-compatible columns such as `stage2_mean_first_score` are retained as aliases so older notebooks do not break.

## Why this matters

Stage34 already contained the real costate-based PPGDPO solver, but the orchestration layer still behaved like stage33:

- `ppgdpo_lite_score` was effectively equal to myopic CE
- final selection still reflected stage1 cheap screening
- manifest entries kept every covariance variant instead of selecting a winner

Stage35 fixes those issues so model selection now reflects the actual hedging-aware PPGDPO diagnostics.
