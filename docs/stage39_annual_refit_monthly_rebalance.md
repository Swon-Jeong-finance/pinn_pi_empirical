# Stage 39: annual refit, monthly rebalance

This stage keeps the three OOS protocols:

- `fixed`
- `expanding_annual`
- `rolling20y_annual`

The key change is semantic rather than combinatorial:

- `fixed` still fits once on the pre-test pool and rebalances monthly.
- `expanding_annual` now **refits once every 12 months** on an expanding window **but still rebalances monthly** between refits.
- `rolling20y_annual` now **refits once every 12 months** on a rolling 240-month window **but still rebalances monthly** between refits.

This isolates the effect of **refit frequency** from the effect of **rebalancing frequency**. In Stage 37/38 those two knobs moved together for the annual protocols, which made interpretation harder.
