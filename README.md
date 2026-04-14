# dynalloc_v2 stage36 trailing-20y validation + real P-PGDPO

This stage replaces the old stage33 PPGDPO-lite heuristic with a real two-stage PG-DPO + Pontryagin projection path using Monte Carlo costates and a barrier-regularized stagewise solve. Legacy config keys are still accepted where possible. Historical notes from earlier stages are kept below where they still help, but the stage34 note here is the current source of truth for the PPGDPO path.

# dynalloc_v2 stage24 factor independence

Current default covariance simplification: `covariance_model.factor_correlation_mode: independent`, which treats factor innovations as conditionally uncorrelated and estimates only per-factor conditional variances. (stage19: APT conditional mean/covariance)

`dynalloc_v2` is the clean-slate engine that branches away from the stage-based `dynalloc_v1` code.

## What stage19 adds

Stage18 brought a first usable `ppgdpo` core into the new engine. Stage19 pushes the migration further by making the model more explicitly **APT-based**:

- **factor APT mean model**
  - expected asset returns are generated from **state-predicted factor means**
  - rather than directly regressing every asset mean on the state vector
- **conditional factor covariance**
  - factor variances depend on the current state
  - asset covariance is induced through the loading matrix
- **no GARCH / no DCC**
  - this stage intentionally uses a simpler APT/state-conditional world
  - full multivariate volatility models are postponed to later stages

## Core equations

Asset returns:

\[
 r_{t+1} = \alpha + \Lambda f_{t+1} + \eta_{t+1}
\]

State-conditional factor means:

\[
 \mathbb E[f_{t+1}\mid s_t] = B_0 + B s_t
\]

State-conditional factor covariance:

\[
 \Sigma_t^{(r)} = \Lambda \Omega_t(s_t) \Lambda^\top + D
\]

with a diagonal factor covariance specification

\[
 \log h_{j,t} = \omega_j + \gamma_j^\top s_t.
\]

## What is still missing

This is still **not** the final empirical engine. Missing pieces include:

- v1 empirical loaders and universe profiles
- selection/rank-sweep orchestration
- broader empirical stress-testing of the new stage34 real P-PGDPO path across very large native-selection rerank suites
- optional future covariance upgrades beyond the current APT/state-only conditional covariance

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

dynalloc-v2 plan --config configs/demo_synthetic_ppgdpo.yaml
dynalloc-v2 run --config configs/demo_synthetic_ppgdpo.yaml
```

Outputs land in `outputs/`.

## Empirical bridge helper

A helper script is included to align an empirical return/state/factor bundle into the CSV format expected by v2:

```bash
python scripts/build_empirical_bundle.py \
  --returns-csv path/to/returns.csv \
  --states-csv path/to/states.csv \
  --factors-csv path/to/factors.csv \
  --out-dir empirical_bundle \
  --state-cols slow_value fast_vol curve_slope \
  --factor-cols MKT VALUE BOND \
  --train-start 1964-01-31 \
  --test-start 2000-01-31 \
  --end-date 2019-12-31
```

This writes aligned bundle CSVs plus a ready-to-edit `config_empirical_ppgdpo_apt.yaml`.


## Stage20: empirical bridge from `dynalloc_v1`

A new bridge command is available for migrating a successful legacy lane into
`dynalloc_v2`:

```bash
dynalloc-v2 legacy bridge-v1-lane   --v1-root ~/empricial_test/dynalloc_v1_stage17_curve_core_pls_suite   --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed   --out-dir empirical_bundles/ff49_cv2000_curve_core
```

See `docs/bridge_ff49_from_v1.md` for the full workflow.


## Stage21 additions

This version adds a light top-k suite workflow:

```bash
# bridge rank 1..2 from an existing v1 lane

dynalloc-v2 legacy bridge-v1-suite   --v1-root ~/empricial_test/dynalloc_v1_stage17_curve_core_pls_suite   --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed   --out-dir empirical_suites/ff49_cv2000_curve_core   --factor-mode ff5_curve_core   --top-k 2

# run rank-sweep and aggregate summaries

dynalloc-v2 run-rank-sweep   --manifest empirical_suites/ff49_cv2000_curve_core/suite_manifest.yaml
```

If you still need the old fixed-only per-rank files under `rank_sweep/rank_XXX/comparison/...`, add `--emit-legacy-fixed-layout`.


## Native selection (stage23)

Stage22 adds two new commands:

Stage46 makes the **ported v2 stage-1 evaluator** authoritative. If you provide a legacy v1 root for selection, it is now used only to emit an **external audit comparison** (`spec_selection_stage1_external_audit.csv`) rather than to override the production stage-1 metrics.


- `dynalloc-v2 legacy bridge-v1-base-lane`
- `dynalloc-v2 select-native-suite`

These commands let v2 build a raw empirical base bundle from a v1 lane, run a **native PLS-only selection** inside v2, and then pass the resulting top-k suite to `dynalloc-v2 run-rank-sweep`.


## Stage23 factor zoo example

```bash
dynalloc-v2 legacy bridge-v1-base-lane   --v1-root ~/empricial_test/dynalloc_v1_stage17_curve_core_pls_suite   --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed   --out-dir base_bundles/ff49_cv2000_curve_core

dynalloc-v2 select-native-suite   --base-dir base_bundles/ff49_cv2000_curve_core   --out-dir empirical_suites_native/ff49_cv2000_curve_core   --candidate-zoo factor_zoo_v1   --top-k 2

dynalloc-v2 run-rank-sweep   --manifest empirical_suites_native/ff49_cv2000_curve_core/suite_manifest.yaml   --device cuda:0
```


## Market-universe sweeps (stage27)

You can now override the legacy v1 universe profile at bridge time to probe smaller or larger market universes without editing the v1 configs. Supported overrides are `ff1`, `ff6`, `ff25`, `ff38`, `ff49`, and `ff100`.

Build one base bundle:

```bash
dynalloc-v2 legacy bridge-v1-base-lane \
  --v1-root ~/empricial_test/dynalloc_v1_stage17_curve_core_pls_suite \
  --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed \
  --asset-universe ff25 \
  --out-dir base_bundles/ff25_cv2000_curve_core
```

Build the standard market-universe grid in one shot:

```bash
dynalloc-v2 legacy bridge-v1-base-grid \
  --v1-root ~/empricial_test/dynalloc_v1_stage17_curve_core_pls_suite \
  --config-stem ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed \
  --out-dir base_bundles/market_universe_grid \
  --asset-universes ff1 ff6 ff25 ff38 ff49 ff100
```

Each universe gets its own bundle directory under `base_bundles/market_universe_grid/<universe>/`, so you can run `select-native-suite` separately on each and see whether smaller asset counts stabilize the downstream ranking.


## Stage36 high-MC patch

- Selection-lite PPGDPO defaults now use `mc_rollouts=256` and `mc_sub_batch=256`.
- Generated final OOS PPGDPO configs also default to `mc_rollouts=256` and `mc_sub_batch=256`.
- CLI overrides are available via `--ppgdpo-lite-mc-rollouts`, `--ppgdpo-lite-mc-sub-batch`, `--ppgdpo-mc-rollouts`, and `--ppgdpo-mc-sub-batch`.


Stage 39 keeps the three OOS protocols (`fixed`, `expanding_annual`, `rolling20y_annual`) but updates the annual protocols so they refit once per year while still rebalancing monthly.

Stage 40 is a behavior-preserving refactor that moves OOS protocol definitions and selection split construction into dedicated modules (`oos_protocols.py`, `selection_splits.py`) so future test-window changes can be added without changing orchestration behavior.


Stage 47 adds `adcc` as an alternative asset-level covariance candidate for native rerank selection (`--rerank-covariance-models ... adcc`) and as a direct config kind (`covariance_model.kind: asset_adcc`).


Stage 48 adds `regime_dcc` as a two-regime asset-level covariance candidate for native rerank selection (`--rerank-covariance-models ... regime_dcc`) and as a direct config kind (`covariance_model.kind: asset_regime_dcc`). The default native rerank set is now `const dcc regime_dcc`, while `diag` and `adcc` remain available as optional comparison candidates.


Stage 49 adds two workflow utilities:

- `dynalloc-v2 init-workspace` creates a version-independent shared workspace (default `~/empricial_test/shared`) with `cache_french`, `cache_fred`, `base_bundles`, and `experiments` subdirectories.
- `dynalloc-v2 replay-sample` replays a manifest rank on `insample_full`, `selection_train`, or `selection_validation` windows derived from `selection_split`, which is useful for train/validation crisis diagnostics.
