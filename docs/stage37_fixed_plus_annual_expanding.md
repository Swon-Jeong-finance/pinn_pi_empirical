# Stage37: fixed baseline + annual expanding OOS protocol

Stage37 keeps the **fixed** OOS backtest as the baseline protocol and adds a second OOS protocol:

- `fixed`: fit once on the full pre-test pool and rebalance monthly during the final test.
- `expanding_annual`: refit on an expanding window once per year and hold/rebalance annually. (Historical Stage 37 behavior; Stage 39 changes this to monthly rebalancing between annual refits.)
- `rolling20y_annual`: refit on a rolling 20-year window once per year and hold/rebalance annually. (Historical Stage 38 behavior; Stage 39 changes this to monthly rebalancing between annual refits.)

`dynalloc-v2 run-rank-sweep` now runs all three protocols by default and writes:

- combined top-level CSVs with an `oos_protocol` column,
- protocol-specific CSVs such as `rank_sweep_fixed_results.csv`, `rank_sweep_expanding_annual_results.csv`, and `rank_sweep_rolling20y_annual_results.csv`,
- per-rank protocol folders under `rank_sweep/rank_XXX/protocols/<protocol>/...`.

By default, only the protocol-first layout is written.

If an older consumer still needs the historical fixed-only paths, add
`--emit-legacy-fixed-layout` to recreate:

- `rank_sweep/rank_XXX/comparison/...`
- `rank_sweep/rank_XXX/resolved_config.yaml`

You can restrict the sweep to one protocol with:

```bash
dynalloc-v2 run-rank-sweep --manifest path/to/suite_manifest.yaml --oos-protocols fixed
```

or

```bash
dynalloc-v2 run-rank-sweep --manifest path/to/suite_manifest.yaml --oos-protocols expanding_annual
```


Use only the 20-year rolling protocol:

```bash
dynalloc-v2 run-rank-sweep --manifest path/to/suite_manifest.yaml --oos-protocols rolling20y_annual
```
