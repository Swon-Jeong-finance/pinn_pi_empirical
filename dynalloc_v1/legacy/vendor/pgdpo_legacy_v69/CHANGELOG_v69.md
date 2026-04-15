# CHANGELOG v69

## Focus
Restore speed for the common long-only / leverage-with-borrowing experiments while
keeping the flexible constraint interface introduced in v68.

## Main changes
- Added exact vectorized simplex projection fast path for `{w >= 0, sum(w) <= L}` in
  `pgdpo_yahoo/constraints.py`.
- `project_box_sum_numpy` / `project_box_sum_torch` now automatically route to the
  simplex fast path when `allow_short=False` and `per_asset_cap=None`.
- Policy modules validate constraints once at init rather than on every forward pass.
- Added `run_v69_methods.py` and `run_v69_cross_suite.py` wrappers.

## What stays generic
- `per_asset_cap` still works, but uses the slower generic box+sum projection path
  when activated.
- short-allowing configurations still use the generic path.
