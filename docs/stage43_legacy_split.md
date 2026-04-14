# Stage43: Legacy boundary split

This cleanup stage stops treating `legacy_bridge.py` as a catch-all module.

## What moved out

- `bridge_common.py`
  - `BaseBundleArtifacts`
  - `_load_yaml`
  - `_ensure_on_syspath`
  - `_read_monthly_panel_csv`
  - `_build_v2_config_dict`
- `native_selection_legacy.py`
  - legacy stage1 module loading
  - legacy spec-name mapping
  - legacy linear diagnostics / cross-rho helpers

## Why

The earlier code mixed three different concerns:

1. actual v1 bridge logic,
2. reusable bundle/config helpers, and
3. optional legacy stage1 selection support.

That made `raw_bundle.py`, `rank_sweep.py`, and `native_selection.py` depend on the bridge layer even when they were not performing a v1 bridge.

## Result

- `raw_bundle.py` now depends only on shared bundle helpers.
- `rank_sweep.py` no longer reaches into the bridge layer for YAML loading.
- `native_selection.py` keeps the legacy stage1 path, but the legacy implementation now lives in a separate module.

This stage is intentionally behavior-preserving. It reduces coupling first; deletion can happen later once the remaining legacy callers are retired.
