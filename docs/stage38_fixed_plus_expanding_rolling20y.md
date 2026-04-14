# Stage 38 (historical): fixed + annual expanding + annual rolling 20y

This stage extends stage 37 by adding a third OOS protocol:

- `fixed`
- `expanding_annual`
- `rolling20y_annual`

`rolling20y_annual` uses a rolling 240-month training window, refits once every 12 months, and holds/rebalances annually between refits.

Legacy manifests that only list `fixed` and `expanding_annual` are auto-upgraded to include `rolling20y_annual` when `--oos-protocols` is omitted.


Note: Stage 39 supersedes this behavior by keeping annual refits but switching the annual protocols back to monthly rebalancing between refits.
