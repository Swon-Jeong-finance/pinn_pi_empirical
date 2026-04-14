# Stage34 real P-PGDPO

This stage removes the stage33 `ppgdpo-lite` heuristic core and ports the real stage12-style structure into the v2 engine.

## What changed

- warm-up policy training now solves a finite-horizon PG-DPO problem under a linear-Gaussian state/return model
- costates are estimated by Monte Carlo + autograd, returning `JX`, `JXX`, and `JXY`
- the stagewise projected action now uses the Pontryagin coefficients
  - `a = X (JX mu + JXY Cross^T)`
  - `Q = X^2 (-JXX) Sigma`
- final weights come from a long-only cash barrier-Newton microproblem rather than the old anchored projected-gradient heuristic

## Backward compatibility

- the v2 experiment interface is unchanged (`experiment.kind: ppgdpo`)
- native selection still keeps some historical `ppgdpo_lite_*` output names, but the implementation underneath is now the real P-PGDPO path
- later cleanup removed legacy stage33 knobs such as `value_epochs`, `value_lr`, `discount`, and `anchor_penalty` because the real stage34 solver never consumed them
