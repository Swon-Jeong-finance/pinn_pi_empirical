# PG-DPO / P-PGDPO Empirical (French 49, monthly) — v59-specsplit

## v59-specsplit changes (relative to v58)

- **v58 is restored as the numerical baseline** for PG-DPO / P-PGDPO training and evaluation.
- **Spec logic is split out of the monolithic runner** into:
  - `pgdpo_yahoo/state_specs.py` for the state-spec registry + builders
  - `pgdpo_yahoo/spec_selection.py` for method-free spec diagnostics
- Added a **spec-first workflow** to `run_french49_10y_model_based_latent_varx_fred.py`:
  - `--select_specs_first` runs predictive-R² / stability / cross diagnostics before any policy training
  - `--selection_only` writes spec ranking CSVs and exits
  - `--selection_top_k K` keeps only the top-K ranked specs for the downstream PG-DPO / P-PGDPO comparison
- The new spec-selection layer is **method-independent**: it uses return/state predictive R², rolling stability, and a cross-risk guard (`max |rho(eps,u)|`) instead of portfolio outcomes.
- Default behavior is unchanged unless one of the new `--selection_*` flags is passed.

### New outputs

- `spec_selection_blocks.csv` : per-block diagnostics by spec
- `spec_selection_summary.csv` : aggregated ranking table with reject / warn flags

### Example

```bash
# 1) rank specs only (no policy training)
python run_french49_10y_model_based_latent_varx_fred.py \
  --compare_specs \
  --selection_only \
  --eval_mode v53 \
  --specs pls_H24_k3 pca_only_k3 macro3_only

# 2) spec-first, then train only the top 2 specs
python run_french49_10y_model_based_latent_varx_fred.py \
  --compare_specs \
  --select_specs_first \
  --selection_top_k 2 \
  --eval_mode v53
```

## v58 changes (relative to v57)

- **Cross-world ablation now applies to every gamma** by default in `scripts/run_french49_v58_queue_gpu8.sh`:
  - `cross_mode = estimated` keeps the fitted `Cross = Cov(eps, u)`
  - `cross_mode = zero` forces `Cross=0` **after each fit/refit**
- This makes the `cross=0` run a full **counterfactual world** in both **FIXED** and **EXPANDING** modes, instead of only a `gamma=1` launcher-side ablation.
- Default v58 queue size: **300 jobs** = 50 specs × 3 gammas × 2 cross modes.
- Evaluation split plan is unchanged from v57 (`--eval_mode v57`: rolling CV folds on DEV + `test20y`).

## v53 changes (relative to v52)

- **Default evaluation plan is now `--eval_mode v53`**:
  - `val10y`: last 10 years of the **pre-test** period (internal validation; no peeking at the final test)
  - `test20y`: full last 20 years (final out-of-sample)
- This removes the extra 10-year test split (faster) and makes spec selection less prone to test leakage.

## v51 changes (relative to v50)

- **EXPANDING mode now includes the full benchmark set**:
  - TRAIN-based benchmarks (`gmv`, `risk_parity`, `inv_vol`, `static_mvo`) are **re-fit on an expanding window** and backtested walk-forward.
  - Model-based benchmarks (`*_model`) are **recomputed each month** from the refit model parameters.
  - This makes the “3 models + 8 benchmarks = 11 strategies” consistent across **FIXED** and **EXPANDING**.
- **Turnover is drift-aware** (computed against *pre-rebalance drifted weights*), so even “constant target weights” benchmarks have **non-zero turnover**.
- **TC sweep covers all strategies** (policy / Markowitz / P‑PGDPO + both benchmark families), and costs are applied as a **multiplicative wealth haircut**.


## v50 sweep

- Specs: 50 (k ∈ {1,2,3,4,5} for PCA/PLS grids)
- Gammas: {1,2,5} (log utility at γ=1)
- Historical v50 default: cross ablation (only when γ=1): `--cross_mode {estimated,zero}`
- Total runs (default): 200

Run:
```bash
bash scripts/run_french49_v50_queue_gpu8.sh fixed
```


This repo runs a **model-based** portfolio experiment on:
- Ken French **49 Industry Portfolios** (monthly)
- Ken French **Fama–French factor datasets** (monthly)
- Optional **FRED macro predictors** (monthly)
- Default **10Y Treasury bond** (CRSP extract, monthly returns from CSV; treated as an extra risky asset)
- (Optional) **bond total return index** from FRED (alternative source)

Core script:
- `run_french49_10y_model_based_latent_varx_fred.py`

Multi-GPU launchers:
- `scripts/run_french49_v58_queue_gpu8.sh` (**recommended**: bash queue launcher; keeps 8 GPUs busy over the full sweep, now with all-gamma cross ablation by default)
- `scripts/run_french49_v42_queue_gpu8.sh` (backwards-compatible name; points to v50)

## What changed in v44

v44 keeps the v43 defaults and additionally enforces a **common training start date across specs** by default:

- **Bond asset included by default**, using a CRSP 10Y fixed-term monthly return CSV.
- **Evaluation windows reduced to 3** (faster):
  - `block1`: first 10 years within the last 20 years
  - `block2`: last 10 years within the last 20 years
  - `last20y`: full last 20 years
- **Hard end-date default**: 2024-12-31 (so the bundled bond series aligns cleanly).
- **Common training start date across specs (default)**: `--force_common_calendar --common_start_mode suite` (align on macro7 + FF5).
- **P-PGDPO enabled by default** (disable with `--no_ppgdpo`).
- **Transaction-cost sweep enabled by default** (disable with `--no_tc_sweep`).

The 22-spec state sweep itself is unchanged (still the v42 plan).

## State-spec sweep (22 specs)

v42 expands the state-spec sweep to the **22-spec plan**:

A) Fixed (no tuning): 4
- `macro3_only`
- `macro7_only`
- `ff3_only` (MKT, SMB, HML)
- `ff5_only` (MKT, SMB, HML, RMW, CMA)

B) PCA-involving specs with k in {2,3}: 6
- `pca_only_k2`, `pca_only_k3`
- `macro7_pca_k2`, `macro7_pca_k3`
- `ff5_macro7_pca_k2`, `ff5_macro7_pca_k3`

C) PLS tuning grid (H × k = 3×2): 6
- `pls_H6_k2`, `pls_H6_k3`
- `pls_H12_k2`, `pls_H12_k3`
- `pls_H24_k2`, `pls_H24_k3`

D) FF3 split into singletons + pairs: 6
- `ff_mkt`, `ff_smb`, `ff_hml`
- `ff_mkt_smb`, `ff_mkt_hml`, `ff_smb_hml`

Compatibility note:
- Legacy v41 names are still accepted: `pca_only`, `macro7_pca`, `ff5_macro7_pca`, `pls_only`.
- For legacy names, `--latent_k` and `--pls_horizon` control k/H.

## Install

```bash
pip install -r requirements.txt
```

## Run (recommended: GPU queue)

```bash
export FRED_API_KEY="YOUR_KEY"  # needed for macro specs

# Fixed-window evaluation (default)
bash scripts/run_french49_v58_queue_gpu8.sh fixed

# Expanding-window walk-forward
bash scripts/run_french49_v58_queue_gpu8.sh expanding
```

By default, logs and outputs are written under:
- `runs/<timestamp>_v58/<spec>_<mode>.log`
- `runs/<timestamp>_v58/<spec>/<mode>/...`

## Run (single spec)

```bash
export FRED_API_KEY="YOUR_KEY"

python3 run_french49_10y_model_based_latent_varx_fred.py \
  --fred_api_key "$FRED_API_KEY" \
  --state_spec pca_only_k2
```

## Notes

- **FRED API key is required** for macro specs (`macro3_only`, `macro7_*`, `ff5_macro7_pca_*`).
- Bond defaults to CRSP CSV. If you switch to `--bond_source fred_tr`, you also need a FRED key.
- Evaluation defaults to `--eval_mode v53` (val10y + test20y). For the old 3-window plan, use `--eval_mode v43`.

## Disclaimer

Research/education only; not investment advice.


## v47 (look-ahead safety hardening)
- Removed backward-fill (`bfill`) from Yahoo/price-based feature builder (unsafe for OOS).
- Added strict guards requiring explicit PCA/scaler fit windows when using `pgdpo_yahoo.features.DatasetConfig` with `strict_no_lookahead=True` (default).
- This does not change the Ken French (French49) main script behavior; it hardens the optional Yahoo/price dataset builder against silent look-ahead.
