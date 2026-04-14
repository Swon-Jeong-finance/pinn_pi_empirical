# Stage44: make fixed legacy rank-sweep layout opt-in

`run-rank-sweep` used to duplicate the `fixed` protocol outputs into the older
per-rank paths:

- `rank_sweep/rank_XXX/comparison/...`
- `rank_sweep/rank_XXX/resolved_config.yaml`

That duplication is now **opt-in**. The default output keeps only the protocol-first
layout under `rank_sweep/rank_XXX/protocols/<protocol>/...`.

Use this only when an older notebook or downstream script still expects the fixed-only
compatibility files:

```bash
dynalloc-v2 run-rank-sweep \
  --manifest path/to/suite_manifest.yaml \
  --emit-legacy-fixed-layout
```

This keeps the modern layout as the default while still leaving one escape hatch for
older consumers that have not been migrated yet.
