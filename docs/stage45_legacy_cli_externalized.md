# Stage45 legacy CLI externalization

This cleanup stage removes `dynalloc_v1` bridge commands from the main `dynalloc-v2`
CLI surface.

## What changed

- Bridge commands now live under `dynalloc-v2 legacy ...`.
- The old top-level `bridge-v1-*` command names are rewritten to `dynalloc-v2 legacy ...` at process startup, so existing shell history still works without polluting the main help output.
- `build_v1_lane_suite()` moved out of `rank_sweep.py` into `legacy_suite.py`, so
  `rank_sweep.py` no longer depends on `legacy_bridge.py`.
- The helper script `scripts/bridge_v1_lane.py` moved to `scripts/legacy/bridge_v1_lane.py`.

## Why this helps

- The default CLI help now emphasizes the v2-native workflow.
- Legacy bridge paths remain available, but they are visually and structurally
  separated from core experiment orchestration.
- `rank_sweep.py` is now focused on rank-sweep execution only.
