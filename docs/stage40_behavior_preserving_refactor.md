# Stage 40: behavior-preserving split/protocol refactor

Stage 40 is a **behavior-preserving refactor** of Stage 39.

What changed:
- OOS protocol definitions moved into `src/dynalloc_v2/oos_protocols.py`
- selection split construction moved into `src/dynalloc_v2/selection_splits.py`
- `rank_sweep.py` and `native_selection.py` now delegate to those modules instead of embedding protocol/split logic inline

What did **not** change:
- default OOS protocols remain `fixed`, `expanding_annual`, `rolling20y_annual`
- Stage 39 protocol semantics are preserved exactly
- trailing holdout and expanding-CV selection block semantics are preserved exactly

Why this refactor matters:
- future split changes (for example moving the 20-year validation/test windows) can be implemented in one layer instead of patching multiple orchestration files
- regression risk is reduced because protocol and split behavior now have direct unit tests
