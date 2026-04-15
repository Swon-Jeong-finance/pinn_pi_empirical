#!/usr/bin/env python3
"""
MODEL-BASED empirical experiment (paper-aligned) with:

- Latent factor y_t (low-dim) extracted via PCA on TRAIN returns (fast filter)
- APT: monthly excess returns depend ONLY on y_t:
    r^{ex}_{t+1} = a + B y_t + eps_{t+1},   eps ~ N(0, Sigma)
- Factor dynamics: VARX (optionally generalized VARX / VARX):
    y_{t+1} = c + A y_t + G z_t + u_{t+1}                       (VARX)
    y_{t+1} = Phi(y_t, z_t) @ Beta + u_{t+1}                    (VARX)
  where z_t are exogenous macro predictors (FRED macro3).

- Train policy via model-based PG-DPO warm-up on simulated episodes.
- Evaluate OOS on the last 10 years of realized returns (fixed 10-year segment).

Additions requested:
- Expanding window option for walk-forward re-estimation (and optional periodic policy re-training).
- True myopic baseline: uses current model-implied (mu(y_t), Sigma) only and ignores continuation / hedging.

Notes:
- This script is discrete-time monthly (no OU mapping).
- P-PGDPO (Pontryagin projection) is included optionally via --ppgdpo.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from pgdpo_yahoo.asset_universes import (
    ASSET_UNIVERSE_CHOICES,
    describe_asset_universe,
    load_equity_universe_monthly,
)
from pgdpo_yahoo.french_data import (
    load_ff_factors_monthly,
    load_ff5_factors_monthly,
)
from pgdpo_yahoo.fred_macro import (
    FredMacroConfig,
    build_macro3_monthly,
    build_macro7_monthly,
    build_total_return_index_returns_monthly,
)
from pgdpo_yahoo.crsp_bond import (
    load_crsp_bond_returns_csv,
    load_crsp_bond_panel_from_spec_text,
    parse_named_item_specs,
)
from pgdpo_yahoo.constraints import PortfolioConstraints, merged_per_asset_cap, project_box_sum_numpy
from pgdpo_yahoo.metrics import compute_metrics
from pgdpo_yahoo.policy import PolicyConfig, PortfolioPolicy, ResidualPolicyConfig, ResidualLongOnlyCashPolicy
from pgdpo_yahoo.discrete_latent_model import (
    LatentPCAConfig,
    DiscreteLatentMarketModel,
    build_discrete_latent_market_model,
    fit_latent_pca,
    fit_apt_mu_sigma,
    fit_varx_transition,
    estimate_cross_cov,
)
from pgdpo_yahoo.discrete_simulator import (
    TorchDiscreteLatentModel,
    DiscreteSimConfig,
    TrainConfig,
    EpisodeSampler,
    train_pgdpo_discrete,
)

from pgdpo_yahoo.ppgdpo import PPGDPOConfig, ppgdpo_action
from pgdpo_yahoo.state_specs import (
    STATE_SPECS_V41,
    STATE_SPECS_V42,
    STATE_SPECS,
    build_model_for_state_spec,
    spec_requires_bond_panel,
    spec_requires_ff3,
    spec_requires_ff5,
    spec_requires_macro3,
    spec_requires_macro7,
)
from pgdpo_yahoo.spec_selection import (
    SpecSelectionConfig,
    evaluate_spec_predictive_diagnostics,
    rank_specs_from_results,
    results_to_frame,
)

# -----------------------------
# State-spec registry (extracted)
# -----------------------------
#
# v59-specsplit keeps the v58 numerical core but moves spec definitions and
# builders to pgdpo_yahoo.state_specs so spec selection can remain method-free.

# -----------------------------
# Hedging-channel diagnostics
# -----------------------------
def cross_rho_stats(
    Sigma: np.ndarray,
    Q: np.ndarray,
    Cross: np.ndarray,
    asset_names: Optional[list[str]] = None,
    state_names: Optional[list[str]] = None,
    top_k: int = 10,
) -> tuple[dict, np.ndarray, list[tuple[str, str, float]]]:
    """Summarize instantaneous hedging channel via Cross = Cov(eps, u).

    We report correlations rho_{i,j} := Cross_{i,j}/(sqrt(Sigma_{ii})*sqrt(Q_{jj})).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Cross = np.asarray(Cross, dtype=float)

    # Robustness: if state dimension k=1, some covariance estimators may return scalars.
    # Force 2D shapes so np.diag works and broadcasting is correct.
    Sigma = np.atleast_2d(Sigma)
    Q = np.atleast_2d(Q)
    if Cross.ndim == 1:
        Cross = Cross.reshape(-1, 1)


    sig = np.sqrt(np.clip(np.diag(Sigma), 1e-16, None))
    q = np.sqrt(np.clip(np.diag(Q), 1e-16, None))
    rho = Cross / (sig[:, None] * q[None, :])
    absrho = np.abs(rho)

    stats = {
        "mean_abs": float(np.mean(absrho)),
        "median_abs": float(np.median(absrho)),
        "p90_abs": float(np.quantile(absrho, 0.90)),
        "p95_abs": float(np.quantile(absrho, 0.95)),
        "max_abs": float(np.max(absrho)),
        "fro": float(np.linalg.norm(rho, ord="fro")),
    }

    # Top |rho| pairs
    n, k = absrho.shape
    flat = absrho.reshape(-1)
    idx = np.argsort(flat)[::-1][: max(1, int(top_k))]
    top = []
    for t in idx:
        i = int(t // k)
        j = int(t % k)
        an = asset_names[i] if asset_names is not None else f"asset{i}"
        sn = state_names[j] if state_names is not None else f"y{j+1}"
        top.append((an, sn, float(rho[i, j])))

    return stats, rho, top


def r2_from_residuals(Y_true: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Compute per-dimension R^2 given residuals (Y_true - Y_pred)."""
    Y_true = np.asarray(Y_true, dtype=float)
    resid = np.asarray(resid, dtype=float)
    y_bar = np.mean(Y_true, axis=0, keepdims=True)
    sst = np.sum((Y_true - y_bar) ** 2, axis=0)
    sse = np.sum(resid ** 2, axis=0)
    r2 = 1.0 - (sse / np.clip(sst, 1e-16, None))
    return r2


def print_model_hedging_diagnostics(
    *,
    title: str,
    model: DiscreteLatentMarketModel,
    apt: Optional[Any] = None,
    varx: Optional[Any] = None,
    top_k: int = 8,
) -> None:
    """Pretty-print a small set of diagnostics to decide whether 'hedging is weak'."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

    # Cross strength
    stats, _rho, top = cross_rho_stats(
        model.Sigma,
        model.Q,
        model.Cross,
        asset_names=model.asset_names,
        state_names=model.state_names,
        top_k=top_k,
    )
    print("Hedging channel (Cross = Cov(eps, u)):")
    print(f"  mean|rho|={stats['mean_abs']:.4f}  p95|rho|={stats['p95_abs']:.4f}  max|rho|={stats['max_abs']:.4f}")
    print("  top |rho| pairs:")
    for an, sn, v in top:
        print(f"    {an:>12s} x {sn:<4s} : rho={v:+.3f}")

    # Predictability (optional)
    if apt is not None:
        # apt.resid aligns with Y_true = r_ex[1:train_end+1]
        r2_rx = r2_from_residuals(apt._Y_true, apt.resid)  # type: ignore[attr-defined]
        print("Return predictability on TRAIN (APT: r_ex_{t+1} ~ a + B y_t):")
        print(f"  mean R2={float(np.mean(r2_rx)):.4f}  median={float(np.median(r2_rx)):.4f}  max={float(np.max(r2_rx)):.4f}")

    if varx is not None:
        r2_y = r2_from_residuals(varx._Y_true, varx.resid)  # type: ignore[attr-defined]
        print("Factor transition predictability on TRAIN:")
        print(f"  mean R2={float(np.mean(r2_y)):.4f}  (per-dim: {', '.join([f'{x:.3f}' for x in r2_y])})")


# -----------------------------
# Data hygiene / alignment
# -----------------------------
def _clean_returns_decimal(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric *decimal* returns.

    IMPORTANT:
      - Our Ken French loaders already convert percent -> decimal.
      - Therefore, we must NOT divide by 100 here.
    """
    out = df.astype(float).replace([np.inf, -np.inf], np.nan)
    # Keep rows even if some assets are NaN for now; we'll do a final joint dropna after alignment.
    return out.dropna(how="all")


def _clean_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric macro predictors (already in decimal units where appropriate)."""
    out = df.astype(float).replace([np.inf, -np.inf], np.nan)
    return out.dropna(how="all")


def _align_common(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    idx = None
    for d in dfs:
        idx = d.index if idx is None else idx.intersection(d.index)
    out = []
    for d in dfs:
        out.append(d.loc[idx].copy())
    return out


def _parse_custom_risky_assets_json(raw: str | None) -> list[dict[str, Any]]:
    payload = str(raw or "").strip()
    if payload == "":
        return []
    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        raise ValueError("--custom_risky_assets_json must decode to a list of asset mappings.")
    return [dict(item) for item in decoded]


def _load_custom_risky_assets_from_args(args: argparse.Namespace, *, ff: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    asset_specs = _parse_custom_risky_assets_json(getattr(args, "custom_risky_assets_json", ""))
    if len(asset_specs) == 0:
        raise ValueError("asset_universe='custom' requires --custom_risky_assets_json.")
    frames: list[pd.Series] = []
    seen_names: set[str] = set()
    base_cache: dict[str, pd.DataFrame] = {}
    for spec in asset_specs:
        name = str(spec.get("name", "") or "").strip()
        if name == "":
            raise ValueError("Each custom risky asset needs a non-empty name.")
        if name in seen_names:
            raise ValueError(f"Duplicate custom risky asset name '{name}'.")
        seen_names.add(name)
        source = str(spec.get("source", "market_total_return") or "market_total_return").strip()
        if source == "market_total_return":
            series = mkt.rename(name)
        elif source == "equity_universe_column":
            base_universe = str(spec.get("base_universe", "") or "").strip()
            column = str(spec.get("column", "") or "").strip()
            if base_universe == "" or column == "":
                raise ValueError(
                    f"Custom risky asset '{name}' with source='equity_universe_column' requires base_universe and column."
                )
            if base_universe not in base_cache:
                base_cache[base_universe] = _clean_returns_decimal(load_equity_universe_monthly(base_universe))
            base_df = base_cache[base_universe]
            if column not in base_df.columns:
                raise ValueError(
                    f"Custom risky asset '{name}' requested column '{column}' from {base_universe}, but it was not found."
                )
            series = base_df[column].rename(name)
        elif source == "csv":
            csv_path = Path(str(spec.get("csv", "") or "")).expanduser()
            if not csv_path.is_absolute():
                csv_path = (Path.cwd() / csv_path).resolve()
            ret_col = str(spec.get("ret_col", "") or "").strip()
            date_col = str(spec.get("date_col", "date") or "date").strip()
            if ret_col == "":
                raise ValueError(f"Custom risky asset '{name}' with source='csv' requires ret_col.")
            if not csv_path.exists():
                raise ValueError(f"Custom risky asset '{name}' CSV not found: {csv_path}")
            df = pd.read_csv(csv_path)
            if date_col not in df.columns or ret_col not in df.columns:
                raise ValueError(
                    f"Custom risky asset '{name}' CSV must contain columns '{date_col}' and '{ret_col}'."
                )
            raw = df[[date_col, ret_col]].copy()
            raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce") + pd.offsets.MonthEnd(0)
            raw = raw.dropna(subset=[date_col])
            raw = raw.set_index(date_col).sort_index()
            series = _clean_returns_decimal(raw[[ret_col]]).iloc[:, 0].rename(name)
        else:
            raise ValueError(f"Unknown custom risky asset source '{source}' for asset '{name}'.")
        frames.append(series)
    risky = pd.concat([s.to_frame() for s in frames], axis=1)
    risky = risky.replace([np.inf, -np.inf], np.nan)
    return risky


def _align_and_dropna_named(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align by common index and drop rows with any NaNs across all inputs.

    We prefix columns per key when building the temporary joint table to avoid
    accidental name collisions (e.g., FF3 vs FF5 both have "Mkt-RF").
    """
    if len(dfs) == 0:
        return {}

    # 1) intersection of indices
    idx = None
    for d in dfs.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    assert idx is not None

    aligned = {k: v.loc[idx].copy() for k, v in dfs.items()}

    # 2) joint dropna (after making columns unique)
    joint = pd.concat(
        [df.add_prefix(f"{k}__") for k, df in aligned.items()],
        axis=1,
    )
    joint = joint.replace([np.inf, -np.inf], np.nan).dropna()
    idx2 = joint.index
    return {k: v.loc[idx2].copy() for k, v in aligned.items()}


def _month_end_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Month-end index helper (inclusive)."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    # Normalize to month-end anchors
    start_me = (start + pd.offsets.MonthEnd(0)).normalize() + pd.offsets.MonthEnd(0)
    end_me = (end + pd.offsets.MonthEnd(0)).normalize() + pd.offsets.MonthEnd(0)
    # NOTE: pandas deprecated "M"; we explicitly use month-end.
    return pd.date_range(start=start_me, end=end_me, freq="ME")


def _get_loc_strict(idx: pd.DatetimeIndex, ts: pd.Timestamp, *, what: str) -> int:
    """Like idx.get_loc but with a clearer error message when ts is missing."""
    ts = pd.Timestamp(ts)
    try:
        return int(idx.get_loc(ts))
    except KeyError as e:
        # Provide a compact diagnostic: show closest dates and count missing months.
        full = _month_end_index(idx[0], idx[-1])
        missing = full.difference(idx)
        nearest = idx[idx.get_indexer([ts], method="nearest")][0] if len(idx) else None
        msg = (
            f"{what}={ts.date()} is not in the dataset calendar. "
            f"Available range: {idx[0].date()} .. {idx[-1].date()} (obs={len(idx)}). "
        )
        if nearest is not None:
            msg += f"Nearest available month-end: {pd.Timestamp(nearest).date()}. "
        if len(missing):
            msg += f"Missing month-ends in index: {len(missing)} (first few: {[d.date() for d in missing[:5]]})."
        raise ValueError(msg) from e


def _warn_if_missing_months(idx: pd.DatetimeIndex, label: str = "series") -> None:
    """Print a compact warning if idx is missing any calendar month-ends."""
    if len(idx) == 0:
        return
    full = _month_end_index(idx[0], idx[-1])
    if len(full) == len(idx) and full.equals(idx):
        return
    missing = full.difference(idx)
    if len(missing) == 0:
        return
    # Only print a few missing stamps to keep logs readable.
    head = ", ".join([d.strftime("%Y-%m-%d") for d in missing[:8]])
    more = "" if len(missing) <= 8 else f" (+{len(missing)-8} more)"
    print(f"[WARN] {label} index is missing {len(missing)} month-end(s): {head}{more}")


def _load_bond_panel(args: argparse.Namespace, mac_cfg: Optional[FredMacroConfig]) -> pd.DataFrame:
    """Load one or more bond return series according to CLI arguments."""
    if not bool(getattr(args, "include_bond", False)):
        return pd.DataFrame()

    bond_source = str(getattr(args, "bond_source", "crsp_csv"))
    bond_csv_specs = str(getattr(args, "bond_csv_specs", "") or "").strip()
    bond_fred_specs = str(getattr(args, "bond_fred_specs", "") or "").strip()

    if bond_source == "crsp_csv":
        if bond_csv_specs:
            print(f"Loading bond panel from CSV specs: {bond_csv_specs} ...")
            bond_df = load_crsp_bond_panel_from_spec_text(
                bond_csv_specs,
                default_date_col="date",
                default_ret_col=str(getattr(args, "bond_ret_col", "bond10y_ret") or "bond10y_ret"),
            )
        else:
            bond_name = str(getattr(args, "bond_name", "UST10Y") or "UST10Y")
            bond_csv = Path(str(getattr(args, "bond_csv", "data/bond10y_10y_fixedterm_monthly_monthend_min.csv")))
            bond_ret_col = str(getattr(args, "bond_ret_col", "bond10y_ret") or "bond10y_ret")
            print(f"Loading bond monthly returns from CSV: {bond_csv} ...")
            bond_ret = load_crsp_bond_returns_csv(
                bond_csv,
                date_col="date",
                ret_col=bond_ret_col,
                name=bond_name,
            )
            bond_df = bond_ret.to_frame()

    elif bond_source == "fred_tr":
        if mac_cfg is None:
            raise RuntimeError("bond_source='fred_tr' requires --fred_api_key")
        if bond_fred_specs:
            print(f"Downloading bond panel from FRED series specs: {bond_fred_specs} ...")
            frames: list[pd.Series] = []
            for bond_name, series_id in parse_named_item_specs(bond_fred_specs):
                frames.append(build_total_return_index_returns_monthly(mac_cfg, series_id, name=bond_name))
            bond_df = pd.concat(frames, axis=1) if len(frames) > 0 else pd.DataFrame()
        else:
            bond_name = str(getattr(args, "bond_name", "UST10Y") or "UST10Y")
            bond_series_id = str(getattr(args, "bond_series_id", "BAMLCC0A0CMTRIV") or "BAMLCC0A0CMTRIV")
            print(f"Downloading bond total return index from FRED: {bond_series_id} ...")
            bond_ret = build_total_return_index_returns_monthly(mac_cfg, bond_series_id, name=bond_name)
            bond_df = bond_ret.to_frame()
    else:
        raise ValueError(f"Unknown --bond_source: {bond_source} (expected 'crsp_csv' or 'fred_tr')")

    if bond_df.empty:
        raise RuntimeError("Bond loading produced an empty panel.")
    if not bond_df.columns.is_unique:
        dupes = bond_df.columns[bond_df.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate bond column names detected: {dupes}")
    return bond_df




# -----------------------------
# Portfolio constraints + myopic / lightweight benchmarks
# -----------------------------
def _build_portfolio_constraints_from_args(args: argparse.Namespace, n_assets: Optional[int] = None) -> PortfolioConstraints:
    pc = PortfolioConstraints(
        risky_cap=float(getattr(args, "risky_cap", 1.0)),
        cash_floor=float(getattr(args, "cash_floor", 0.0)),
        allow_short=bool(getattr(args, "allow_short", False)),
        short_floor=float(getattr(args, "short_floor", 0.0)),
        per_asset_cap=None if getattr(args, "per_asset_cap", None) is None else float(getattr(args, "per_asset_cap", None)),
    )
    pc.validate(n_assets)
    return pc


def _resolve_ppgdpo_risky_cap(args: argparse.Namespace, constraints: PortfolioConstraints) -> float:
    override = getattr(args, "ppgdpo_L", None)
    if override is None:
        return float(constraints.effective_risky_cap())
    return min(float(override), float(constraints.risky_cap_from_cash_floor()))


def _cash_from_risky_np(w: np.ndarray) -> float:
    return float(1.0 - float(np.sum(np.asarray(w, dtype=float))))


def _project_risky_numpy(
    w_raw: np.ndarray,
    *,
    constraints: Optional[PortfolioConstraints] = None,
    w_max: float | None = None,
) -> np.ndarray:
    w_raw = np.asarray(w_raw, dtype=float).reshape(-1)
    n = w_raw.shape[0]
    pc = PortfolioConstraints() if constraints is None else constraints
    pc.validate(n)
    asset_cap = merged_per_asset_cap(pc, w_max)
    w = project_box_sum_numpy(
        w_raw,
        lower=float(pc.lower_bound()),
        upper=asset_cap,
        cap=float(pc.effective_risky_cap()),
    )
    return w.astype(float)


# -----------------------------
# True myopic (same-utility, no-continuation) benchmark
# -----------------------------
def myopic_weights_long_only_cash(
    mu_excess: np.ndarray,   # (n,)
    sigma: np.ndarray,       # (n,n)
    risk_aversion: float = 5.0,
    ridge: float = 1e-6,
    w_max: float | None = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> tuple[np.ndarray, float]:
    """Same-utility myopic one-step weights under a flexible risky/cash constraint set.

    The baseline unconstrained quadratic solution is
        w* = inv(Sigma) mu / risk_aversion,
    which is then projected onto the configured feasible set for risky weights.
    Cash is the residual ``1 - sum(w)`` and may therefore be negative when
    borrowing is allowed.
    """
    mu_excess = np.asarray(mu_excess, dtype=float).reshape(-1)
    n = mu_excess.shape[0]
    ra = max(float(risk_aversion), 1e-8)
    S = np.asarray(sigma, dtype=float) + float(ridge) * np.eye(n)

    try:
        w_unc = np.linalg.solve(S, mu_excess) / ra
    except np.linalg.LinAlgError:
        # fallback: diagonal approximation
        w_unc = mu_excess / (ra * (np.diag(S) + 1e-12))

    w = _project_risky_numpy(w_unc, constraints=constraints, w_max=w_max)
    cash = _cash_from_risky_np(w)
    return w.astype(float), float(cash)


def myopic_from_model(
    model: DiscreteLatentMarketModel,
    y_t: np.ndarray,         # (k,)
    gamma: float,
    ridge: float = 1e-6,
    w_max: float | None = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> tuple[np.ndarray, float]:
    """True myopic benchmark using the current model-implied conditional mean.

    Uses
      mu_ex(y_t) = a + B y_t
    and the fitted Sigma while ignoring continuation / Cross / transition terms in
    the policy calculation.
    """
    y_t = np.asarray(y_t, dtype=float).reshape(-1)
    mu_ex = model.a + model.B @ y_t
    return myopic_weights_long_only_cash(
        mu_ex,
        model.Sigma,
        risk_aversion=gamma,
        ridge=ridge,
        w_max=w_max,
        constraints=constraints,
    )


# Backward-compatible aliases (v61/v60 naming)
markowitz_weights_long_only_cash = myopic_weights_long_only_cash
markowitz_myopic_from_model = myopic_from_model


# -----------------------------
# Additional lightweight benchmarks (requested)
# -----------------------------
def gmv_weights_long_only_cash(
    sigma: np.ndarray,
    ridge: float = 1e-6,
    w_max: float | None = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> tuple[np.ndarray, float]:
    """Global minimum-variance weights under the configured risky/cash constraints."""
    S = np.asarray(sigma, dtype=float)
    n = S.shape[0]
    S = S + float(ridge) * np.eye(n)
    ones = np.ones(n, dtype=float)

    try:
        w = np.linalg.solve(S, ones)
    except np.linalg.LinAlgError:
        w = ones / (np.diag(S) + 1e-12)

    s = float(np.sum(w))
    if abs(s) > 1e-12:
        w = w / s
    else:
        w = np.full(n, 1.0 / n, dtype=float)

    w = _project_risky_numpy(w, constraints=constraints, w_max=w_max)
    cash = _cash_from_risky_np(w)
    return w.astype(float), float(cash)


def inv_vol_weights_long_only_cash(
    sigma: np.ndarray,
    w_max: float | None = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> tuple[np.ndarray, float]:
    """Inverse-volatility weights (diagonal risk-parity proxy) under the configured constraints."""
    S = np.asarray(sigma, dtype=float)
    vol = np.sqrt(np.clip(np.diag(S), 1e-16, None))
    w = 1.0 / vol
    s = float(np.sum(w))
    if s > 1e-16:
        w = w / s
    else:
        w = np.full_like(w, 1.0 / len(w))

    w = _project_risky_numpy(w, constraints=constraints, w_max=w_max)
    cash = _cash_from_risky_np(w)
    return w.astype(float), float(cash)


def risk_parity_weights_long_only_cash(
    sigma: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
    ridge: float = 1e-8,
    w_max: float | None = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> tuple[np.ndarray, float]:
    """Full-covariance risk parity weights via a simple fixed-point iteration."""
    S = np.asarray(sigma, dtype=float)
    n = S.shape[0]
    S = S + float(ridge) * np.eye(n)

    w = np.full(n, 1.0 / n, dtype=float)
    eps = 1e-12

    for _ in range(int(max_iter)):
        m = S @ w  # marginal contribution
        rc = w * m
        avg = float(np.mean(rc))
        if not np.isfinite(avg) or avg <= 0.0:
            break
        w_new = w * (avg / (rc + eps))
        w_new = np.maximum(w_new, eps)
        w_new = w_new / float(np.sum(w_new))

        m_new = S @ w_new
        rc_new = w_new * m_new
        rel = np.max(np.abs(rc_new / (np.mean(rc_new) + eps) - 1.0))
        w = w_new
        if rel < float(tol):
            break

    w = _project_risky_numpy(w, constraints=constraints, w_max=w_max)
    cash = _cash_from_risky_np(w)
    return w.astype(float), float(cash)

def _zero_cross_policy_proxy_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "zero_cross_policy_proxy", "full")).lower()
    return mode if mode in {"myopic", "full"} else "full"


def _policy_leg_label(cross_mode: str, zero_cross_myopic_proxy: bool) -> str:
    cm = str(cross_mode).lower()
    if zero_cross_myopic_proxy:
        return "policy proxy (= Myopic in cross=0 world)"
    if cm == "zero":
        return "Dynamic policy (cross=0 world)"
    return "Full dynamic policy"


def _use_zero_cross_myopic_proxy(args: argparse.Namespace, cross_mode: str) -> bool:
    return str(cross_mode).lower() == "zero" and _zero_cross_policy_proxy_mode(args) == "myopic"


def _scale_tag(scale: float) -> str:
    s = f"{float(scale):.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _ppgdpo_variant_specs(args: argparse.Namespace, cross_mode: str) -> list[dict[str, Any]]:
    """Describe which P-PGDPO projection variants to evaluate in the current world.

    - The main `ppgdpo` leg always uses cross_scale=1 in the current world.
    - `ppgdpo_local_zerohedge` keeps the estimated-world policy / costates but
      sets projection-stage Cross to zero (local ablation).
    - Additional `--ppgdpo_cross_scales` values add more projection diagnostics.
    """
    variants: list[dict[str, Any]] = [{
        "name": "ppgdpo",
        "label": "P-PGDPO",
        "cross_scale": 1.0,
        "kind": "main",
    }]

    def _has_scale(val: float) -> bool:
        return any(abs(float(v["cross_scale"]) - float(val)) < 1e-12 for v in variants)

    if str(cross_mode).lower() == "estimated" and bool(getattr(args, "ppgdpo_local_zero_hedge", False)) and not _has_scale(0.0):
        variants.append({
            "name": "ppgdpo_local_zerohedge",
            "label": "P-PGDPO local zero-hedge",
            "cross_scale": 0.0,
            "kind": "local_zero",
        })

    extra_scales = getattr(args, "ppgdpo_cross_scales", None)
    if extra_scales:
        for sc in extra_scales:
            scf = float(sc)
            if _has_scale(scf):
                continue
            variants.append({
                "name": f"ppgdpo_crossx{_scale_tag(scf)}",
                "label": f"P-PGDPO cross x{scf:g}",
                "cross_scale": scf,
                "kind": "scale",
            })
    return variants


def _policy_logs_from_myopic_proxy(logs_mvo: dict, proxy_name: str = "myopic_zero_cross_proxy") -> dict:
    logs = copy.deepcopy(logs_mvo)
    if isinstance(logs, dict):
        logs["proxy_name"] = str(proxy_name)
        metrics = logs.get("metrics")
        if isinstance(metrics, dict):
            metrics = dict(metrics)
            metrics["policy_proxy"] = 1.0
            logs["metrics"] = metrics
        diag = logs.get("diag")
        if isinstance(diag, dict):
            diag = dict(diag)
            diag["proxy_name"] = str(proxy_name)
            logs["diag"] = diag
    return logs


def simulate_realized_fixed_constant_weights(
    w: np.ndarray,
    r_ex: np.ndarray,
    rf: np.ndarray,
    start_idx: int,
    H: int,
    *,
    gamma: float | None = None,
) -> dict:
    """Realized backtest for a constant-weight benchmark.

    v51 change
    ----------
    We now assume the portfolio is *rebalanced back to the target weights each month*.
    Turnover is computed against the *drifted* weights right before rebalancing.
    This makes transaction-cost sweeps meaningful for constant-weight benchmarks.
    """
    w = np.asarray(w, dtype=float).reshape(-1)

    X = 1.0
    wealth = [X]
    port_rets = []
    pis = []
    cashes = []
    turnovers = []

    prev_drift_full = None
    for j in range(H):
        t = start_idx + j

        r_ex_next = r_ex[t + 1]
        rf_next = rf[t + 1]
        port_ret = float(rf_next + np.dot(w, r_ex_next))
        X = X * (1.0 + port_ret)

        wealth.append(X)
        port_rets.append(port_ret)
        pis.append(w.copy())
        cash_t = float(1.0 - np.sum(w))
        cashes.append(cash_t)

        # Turnover: compare target vs drifted previous weights (including cash).
        # For the first period, we treat turnover as undefined (NaN).
        if prev_drift_full is None:
            turnovers.append(np.nan)
        else:
            turnovers.append(float(turnover_total_variation(prev_drift_full, np.append(w, cash_t))))

        # Update drifted weights for next period.
        prev_drift_full = drift_full_weights(np.append(w, cash_t), r_ex_next=r_ex_next, rf_next=rf_next)

    wealth = np.asarray(wealth, dtype=float)
    port_rets = np.asarray(port_rets, dtype=float)
    pis = np.asarray(pis, dtype=float)
    cashes = np.asarray(cashes, dtype=float)
    turnovers = np.asarray(turnovers, dtype=float)

    rf_seg = rf[start_idx + 1 : start_idx + 1 + H]
    m = compute_metrics(port_rets, rf=rf_seg, periods_per_year=12, gamma=gamma)

    gross = np.sum(np.abs(pis), axis=1)
    net = np.sum(pis, axis=1)

    return {
        "wealth": wealth,
        "port_rets": port_rets,
        "pi": pis,
        "cash": cashes,
        "turnover": turnovers,
        "metrics": m,
        "diag": {
            "avg_gross": float(np.nanmean(gross)),
            "avg_net": float(np.nanmean(net)),
            "avg_turnover": float(np.nanmean(turnovers)),
            "avg_cash": float(np.nanmean(cashes)),
            "min_cash": float(np.nanmin(cashes)),
            "max_cash": float(np.nanmax(cashes)),
        },
    }


# -----------------------------
# Realized simulators
# -----------------------------


def _to_full_weights(pi: np.ndarray, cash: float | None = None) -> np.ndarray:
    """Return a signed (n+1,) weight vector that sums to 1, with cash last.

    Unlike the older simplex-only helper, this version preserves negative cash
    (borrowing) and, if enabled in future, negative risky weights as well.
    Small numerical sum drift is absorbed into cash.
    """
    pi = np.asarray(pi, dtype=float).reshape(-1)
    cash_val = float(1.0 - float(np.sum(pi))) if cash is None else float(cash)
    w = np.concatenate([pi, np.asarray([cash_val], dtype=float)], axis=0)

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(np.sum(w))
    if not np.isfinite(s) or abs(s) <= 1e-16:
        w = np.zeros_like(w)
        w[-1] = 1.0
        return w

    # Keep risky legs intact; absorb tiny drift into cash.
    if abs(s - 1.0) > 1e-10:
        w[-1] += (1.0 - s)
    return w


def drift_full_weights(w_full: np.ndarray, *, r_ex_next: np.ndarray, rf_next: float) -> np.ndarray:
    """Compute the drifted portfolio weights after applying realized returns.

    Parameters
    ----------
    w_full : (n+1,) array
        Portfolio weights at the *start* of the month (after rebalancing), including cash.
    r_ex_next : (n,) array
        Realized next-month *excess* returns of risky assets.
    rf_next : float
        Realized next-month risk-free return.
    """
    w_full = np.asarray(w_full, dtype=float).reshape(-1)
    n = int(w_full.shape[0] - 1)
    r_ex_next = np.asarray(r_ex_next, dtype=float).reshape(n)
    rf_next = float(rf_next)

    # Total returns for risky assets are rf + r_ex.
    g_risky = 1.0 + rf_next + r_ex_next
    g_cash = 1.0 + rf_next
    g = np.concatenate([g_risky, np.asarray([g_cash], dtype=float)], axis=0)
    g = np.clip(g, 1e-12, None)

    w_dollar = w_full * g
    total = float(np.sum(w_dollar))
    if not np.isfinite(total) or total <= 1e-16:
        # pathological (e.g., portfolio wiped out); fall back to all-cash
        out = np.zeros_like(w_full)
        out[-1] = 1.0
        return out
    return w_dollar / total


def turnover_total_variation(w_prev_full: np.ndarray, w_new_full: np.ndarray) -> float:
    """One-way turnover as total-variation distance between two full portfolios.

    We define turnover as:
      0.5 * || w_new_full - w_prev_full ||_1

    where each weight vector includes cash and sums to 1.
    This fixes the common undercounting that occurs when using 0.5*||Δw_risky||_1
    under a cash-allowing constraint (sum w_risky <= 1).
    """
    a = np.asarray(w_prev_full, dtype=float).reshape(-1)
    b = np.asarray(w_new_full, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in turnover: {a.shape} vs {b.shape}")
    return float(0.5 * np.sum(np.abs(b - a)))


def _drift_risky_weights(
    pi: np.ndarray,
    cash: float,
    r_ex_next: np.ndarray,
    rf_next: float,
) -> tuple[np.ndarray, float]:
    """Compatibility wrapper used by EXPANDING-mode code.

    Returns drifted risky weights and drifted cash weight.
    """
    full = _to_full_weights(pi, cash)
    full_d = drift_full_weights(full, r_ex_next=r_ex_next, rf_next=rf_next)
    return full_d[:-1], float(full_d[-1])


def _turnover_one_way_from_drift(pi_new: np.ndarray, prev_pi_drift: Optional[np.ndarray]) -> float:
    """Compatibility wrapper used by EXPANDING-mode code.

    We compute turnover as total-variation distance between full portfolios.
    Since EXPANDING-mode tracks only drifted risky weights, we reconstruct cash
    as the residual 1 - sum(pi).
    """
    if prev_pi_drift is None:
        return float(np.nan)
    w_prev = _to_full_weights(prev_pi_drift, cash=None)
    w_new = _to_full_weights(pi_new, cash=None)
    return turnover_total_variation(w_prev, w_new)


def apply_linear_transaction_costs(
    port_rets: np.ndarray,
    turnover: np.ndarray,
    tc: float,
) -> np.ndarray:
    """Apply linear transaction costs (multiplicative wealth impact).

    Convention (v51)
    ----------------
    - Turnover is assumed to be a *one-way traded value* fraction of wealth.
    - We apply costs at the rebalancing instant, before the next-month return.

    If gross return is r_t and turnover is TO_t, then:
      (1 + r_net_t) = (1 - tc * TO_t) * (1 + r_t)

    This is slightly more faithful than the common additive approximation
    r_net = r - tc*TO, while remaining cheap to compute.
    """
    r = np.asarray(port_rets, dtype=float)
    to = np.asarray(turnover, dtype=float)
    to = np.nan_to_num(to, nan=0.0, posinf=0.0, neginf=0.0)
    tc = float(tc)
    # Guard against negative wealth multiplier due to extreme inputs.
    mult = np.clip(1.0 - tc * to, 0.0, None)
    return mult * (1.0 + r) - 1.0


def tc_sweep_metrics(
    port_rets: np.ndarray,
    turnover: np.ndarray,
    rf_seg: np.ndarray,
    *,
    gamma: float,
    tc_bps_list: list[float],
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Compute metrics under a grid of transaction cost assumptions."""
    rows = []
    for bps in tc_bps_list:
        tc = float(bps) / 1e4
        r_net = apply_linear_transaction_costs(port_rets, turnover, tc)
        m = compute_metrics(r_net, rf=rf_seg, periods_per_year=periods_per_year, gamma=gamma)
        m["tc_bps"] = float(bps)
        m["avg_turnover"] = float(np.nanmean(turnover))
        rows.append(m)
    df = pd.DataFrame(rows).set_index("tc_bps")
    return df


def simulate_realized_expanding_benchmarks_from_returns(
    *,
    r_ex: np.ndarray,
    rf: np.ndarray,
    start_idx: int,
    H: int,
    bench_list: list[str],
    bench_w_max: float | None,
    rp_max_iter: int,
    rp_tol: float,
    refit_every: int = 1,
    gamma: float = 5.0,
    window_mode: str = "expanding",
    rolling_train_months: int | None = None,
) -> dict[str, dict]:
    """Expanding-window (walk-forward) versions of the "TRAIN-estimated" benchmarks.

    At each month t = start_idx + j:
      1) estimate (mu, Sigma) using realized returns up to month t (past only)
      2) compute benchmark target weights
      3) rebalance to target, apply realized return of month t+1

    Turnover is drift-based and computed using total-variation distance over
    (risky + cash) weights.

    Parameters
    ----------
    refit_every : int
        If >1, refresh (mu,Sigma)->weights only every k months, and keep the
        last computed target weights in between. Rebalancing to that fixed
        target still happens every month.
    """
    bench_list = list(bench_list)
    refit_every = max(int(refit_every), 1)
    window_mode = _resolve_walk_forward_mode_name(window_mode)
    rolling_train_months = _resolve_rolling_train_months(rolling_train_months)

    # tracker init
    tracks: dict[str, dict[str, Any]] = {}
    for name in bench_list:
        tracks[name] = {
            "X": 1.0,
            "wealth": [1.0],
            "port_rets": [],
            "pi": [],
            "cash": [],
            "turnover": [],
            "prev_drift_full": None,
        }

    # cached target weights (updated at refit times)
    current_w: dict[str, tuple[np.ndarray, float]] = {}

    for j in range(H):
        t = start_idx + j

        # refresh weights
        if (j % refit_every == 0) or (not current_w):
            train_end = t
            train_start, train_end = _window_bounds_for_refit(
                train_end,
                window_mode=window_mode,
                rolling_train_months=rolling_train_months,
            )
            train_r_ex = np.asarray(r_ex[train_start + 1 : train_end + 1, :], dtype=float)
            if train_r_ex.shape[0] < 2:
                train_r_ex = np.asarray(r_ex[1 : train_end + 1, :], dtype=float)
            lw = LedoitWolf().fit(train_r_ex)
            Sigma_train = lw.covariance_
            mu_train = np.nanmean(train_r_ex, axis=0)

            if "gmv" in bench_list:
                w_gmv, c_gmv = gmv_weights_long_only_cash(Sigma_train, ridge=1e-6, w_max=bench_w_max)
                current_w["gmv"] = (w_gmv, float(c_gmv))

            if "risk_parity" in bench_list:
                w_rp, c_rp = risk_parity_weights_long_only_cash(
                    Sigma_train, max_iter=rp_max_iter, tol=rp_tol, ridge=1e-8, w_max=bench_w_max
                )
                current_w["risk_parity"] = (w_rp, float(c_rp))

            if "inv_vol" in bench_list:
                w_iv, c_iv = inv_vol_weights_long_only_cash(Sigma_train, w_max=bench_w_max)
                current_w["inv_vol"] = (w_iv, float(c_iv))

            if "static_mvo" in bench_list:
                w_smvo, c_smvo = markowitz_weights_long_only_cash(
                    mu_train, Sigma_train, risk_aversion=float(gamma), ridge=1e-6, w_max=bench_w_max
                )
                current_w["static_mvo"] = (w_smvo, float(c_smvo))

        # realized next-month return
        r_ex_next = np.asarray(r_ex[t + 1], dtype=float)
        rf_next = float(rf[t + 1])

        for name in bench_list:
            if name not in current_w:
                continue
            w, c = current_w[name]
            tr = tracks[name]

            full_now = _to_full_weights(w, c)
            if tr["prev_drift_full"] is None:
                tr["turnover"].append(float(np.nan))
            else:
                tr["turnover"].append(turnover_total_variation(tr["prev_drift_full"], full_now))

            ret = rf_next + float(np.dot(w, r_ex_next))
            tr["X"] = float(tr["X"]) * (1.0 + ret)
            tr["wealth"].append(float(tr["X"]))
            tr["port_rets"].append(float(ret))
            tr["pi"].append(np.asarray(w, dtype=float))
            tr["cash"].append(float(c))

            tr["prev_drift_full"] = drift_full_weights(full_now, r_ex_next=r_ex_next, rf_next=rf_next)

    rf_seg = np.asarray(rf[start_idx + 1 : start_idx + 1 + H], dtype=float)
    out: dict[str, dict] = {}
    for name, tr in tracks.items():
        port_rets = np.asarray(tr["port_rets"], dtype=float)
        turnover = np.asarray(tr["turnover"], dtype=float)
        m = compute_metrics(port_rets, rf=rf_seg, periods_per_year=12, gamma=float(gamma))
        out[name] = {
            "wealth": np.asarray(tr["wealth"], dtype=float),
            "port_rets": port_rets,
            "pi": np.asarray(tr["pi"], dtype=float),
            "cash": np.asarray(tr["cash"], dtype=float),
            "turnover": turnover,
            "metrics": m,
        }
    return out

def simulate_realized_fixed_policy(
    policy: PortfolioPolicy,
    y_all: np.ndarray,
    r_ex: np.ndarray,
    rf: np.ndarray,
    start_idx: int,
    H: int,
    *,
    gamma: float | None = None,
) -> dict:
    """
    Realized backtest (fixed segment):
      weights chosen using state y_t, applied to realized r_{t+1}.
    """
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    X = 1.0
    wealth = [X]
    port_rets = []
    pis = []
    cashes = []

    prev_drift_full = None
    turnovers = []

    for j in range(H):
        t = start_idx + j
        y_t = torch.tensor(y_all[t], device=device, dtype=dtype).view(1, -1)
        x_t = torch.tensor([[X]], device=device, dtype=dtype)
        tau = torch.tensor([[1.0 - j / H]], device=device, dtype=dtype)

        pi, cash = policy.weights_and_cash(tau, x_t, y_t)
        pi_np = pi.detach().cpu().numpy().reshape(-1)
        cash_np = float(cash.detach().cpu().item()) if cash is not None else float(1.0 - pi_np.sum())

        full_now = _to_full_weights(pi_np, cash_np)

        # realized next-month return
        r_ex_next = r_ex[t + 1]   # (n,)
        rf_next = rf[t + 1]       # scalar
        port_ret = rf_next + float(np.dot(pi_np, r_ex_next))
        X = X * (1.0 + port_ret)

        wealth.append(X)
        port_rets.append(port_ret)
        pis.append(pi_np)
        cashes.append(cash_np)

        if prev_drift_full is None:
            turnovers.append(np.nan)
        else:
            turnovers.append(turnover_total_variation(prev_drift_full, full_now))

        # Drift weights forward to the next decision time.
        prev_drift_full = drift_full_weights(full_now, r_ex_next=r_ex_next, rf_next=rf_next)

    wealth = np.asarray(wealth)
    port_rets = np.asarray(port_rets)
    pis = np.asarray(pis)
    cashes = np.asarray(cashes)
    turnovers = np.asarray(turnovers)

    rf_seg = rf[start_idx + 1 : start_idx + 1 + H]
    m = compute_metrics(port_rets, rf=rf_seg, periods_per_year=12, gamma=gamma)

    gross = np.sum(np.abs(pis), axis=1)
    net = np.sum(pis, axis=1)

    return {
        "wealth": wealth,
        "port_rets": port_rets,
        "pi": pis,
        "cash": cashes,
        "turnover": turnovers,
        "metrics": m,
        "diag": {
            "avg_gross": float(np.nanmean(gross)),
            "avg_net": float(np.nanmean(net)),
            "avg_turnover": float(np.nanmean(turnovers)),
            "avg_cash": float(np.nanmean(cashes)),
            "min_cash": float(np.nanmin(cashes)),
            "max_cash": float(np.nanmax(cashes)),
        },
    }


def simulate_realized_fixed_myopic(
    model: DiscreteLatentMarketModel,
    y_all: np.ndarray,
    r_ex: np.ndarray,
    rf: np.ndarray,
    start_idx: int,
    H: int,
    gamma: float,
    ridge: float = 1e-6,
    w_max: float | None = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> dict:
    """
    Realized backtest (fixed segment) for the true myopic (same-utility, no-continuation) benchmark.
    """
    X = 1.0
    wealth = [X]
    port_rets = []
    pis = []
    cashes = []

    prev_drift_full = None
    turnovers = []

    for j in range(H):
        t = start_idx + j
        w, cash = myopic_from_model(model, y_all[t], gamma=gamma, ridge=ridge, w_max=w_max, constraints=constraints)
        full_now = _to_full_weights(w, cash)

        r_ex_next = r_ex[t + 1]
        rf_next = rf[t + 1]
        port_ret = rf_next + float(np.dot(w, r_ex_next))
        X = X * (1.0 + port_ret)

        wealth.append(X)
        port_rets.append(port_ret)
        pis.append(w)
        cashes.append(cash)

        if prev_drift_full is None:
            turnovers.append(np.nan)
        else:
            turnovers.append(turnover_total_variation(prev_drift_full, full_now))

        prev_drift_full = drift_full_weights(full_now, r_ex_next=r_ex_next, rf_next=rf_next)

    wealth = np.asarray(wealth)
    port_rets = np.asarray(port_rets)
    pis = np.asarray(pis)
    cashes = np.asarray(cashes)
    turnovers = np.asarray(turnovers)

    rf_seg = rf[start_idx + 1 : start_idx + 1 + H]
    m = compute_metrics(port_rets, rf=rf_seg, periods_per_year=12, gamma=gamma)

    gross = np.sum(np.abs(pis), axis=1)
    net = np.sum(pis, axis=1)

    return {
        "wealth": wealth,
        "port_rets": port_rets,
        "pi": pis,
        "cash": cashes,
        "turnover": turnovers,
        "metrics": m,
        "diag": {
            "avg_gross": float(np.nanmean(gross)),
            "avg_net": float(np.nanmean(net)),
            "avg_turnover": float(np.nanmean(turnovers)),
            "avg_cash": float(np.nanmean(cashes)),
            "min_cash": float(np.nanmin(cashes)),
            "max_cash": float(np.nanmax(cashes)),
        },
    }


# Backward-compatible alias (v61/v60 naming)
simulate_realized_fixed_markowitz = simulate_realized_fixed_myopic


def simulate_realized_fixed_ppgdpo(
    torch_model: TorchDiscreteLatentModel,
    policy: PortfolioPolicy,
    y_all: np.ndarray,
    z_all: Optional[np.ndarray],
    r_ex: np.ndarray,
    rf: np.ndarray,
    start_idx: int,
    H: int,
    rf_month: float,
    sim_cfg_full: DiscreteSimConfig,
    ppgdpo_cfg: PPGDPOConfig,
    *,
    gamma: float | None = None,
    cross_scale: float = 1.0,
) -> dict:
    """
    Realized backtest with a P-PGDPO Pontryagin projection applied *at each decision time*.

    The optional ``cross_scale`` rescales the projection-stage Cross block while
    keeping the policy/training world fixed. This lets us compare:
      - full estimated world projection (cross_scale=1)
      - local zero-hedge ablation (cross_scale=0)
      - amplified / damped hedge tests (cross_scale != 1)
    """
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    X = 1.0
    wealth = [X]
    port_rets = []
    pis = []
    cashes = []
    turnovers = []
    hedge_l1 = []
    mu_l1 = []
    a_l1 = []

    prev_drift_full = None

    tau_step = 1.0 / float(H)

    for j in range(H):
        t = start_idx + j
        y_t = torch.tensor(y_all[t], device=device, dtype=dtype).view(1, -1)
        x_t = torch.tensor([[X]], device=device, dtype=dtype)

        # remaining horizon for the costate MC (including current month decision)
        H_rem = H - j
        sim_cfg_rem = DiscreteSimConfig(
            horizon_steps=int(H_rem),
            gamma=float(sim_cfg_full.gamma),
            kappa=float(sim_cfg_full.kappa),
            clamp_wealth_min=float(sim_cfg_full.clamp_wealth_min),
            clamp_state_std_abs=sim_cfg_full.clamp_state_std_abs,
            clamp_port_ret_min=float(sim_cfg_full.clamp_port_ret_min),
            clamp_port_ret_max=float(sim_cfg_full.clamp_port_ret_max),
        )

        z_path = None
        if z_all is not None:
            z_future = z_all[t : t + H_rem, :]
            z_path = torch.tensor(z_future, device=device, dtype=dtype).unsqueeze(0)  # (1,H_rem,m)

        tau0 = 1.0 - j / float(H)

        ppgdpo_cfg_step = ppgdpo_cfg
        if ppgdpo_cfg.seed is not None:
            ppgdpo_cfg_step = PPGDPOConfig(**ppgdpo_cfg.__dict__)
            ppgdpo_cfg_step.seed = int(ppgdpo_cfg.seed) + int(t)

        out = ppgdpo_action(
            model=torch_model,
            policy=policy,
            X=x_t,
            Y=y_t,
            z_path=z_path,
            rf_month=float(rf_month),
            sim_cfg=sim_cfg_rem,
            ppgdpo_cfg=ppgdpo_cfg_step,
            tau0=float(tau0),
            tau_step=float(tau_step),
            cross_scale=float(cross_scale),
            return_debug=True,
        )
        pi = out["u_pp"]
        cash = out["cash_pp"]
        dbg = out.get("debug", {})

        mu_term = dbg.get("mu_term")
        hedge_term = dbg.get("hedge_term")
        a_vec = dbg.get("a_vec")
        if isinstance(mu_term, torch.Tensor):
            mu_l1.append(float(mu_term.abs().sum(dim=1).mean().detach().cpu().item()))
        else:
            mu_l1.append(np.nan)
        if isinstance(hedge_term, torch.Tensor):
            hedge_l1.append(float(hedge_term.abs().sum(dim=1).mean().detach().cpu().item()))
        else:
            hedge_l1.append(np.nan)
        if isinstance(a_vec, torch.Tensor):
            a_l1.append(float(a_vec.abs().sum(dim=1).mean().detach().cpu().item()))
        else:
            a_l1.append(np.nan)

        pi_np = pi.detach().cpu().numpy().reshape(-1)
        cash_np = float(cash.detach().cpu().item())
        full_now = _to_full_weights(pi_np, cash_np)

        r_ex_next = r_ex[t + 1]
        rf_next = rf[t + 1]
        port_ret = rf_next + float(np.dot(pi_np, r_ex_next))
        X = X * (1.0 + port_ret)

        wealth.append(X)
        port_rets.append(port_ret)
        pis.append(pi_np)
        cashes.append(cash_np)

        if prev_drift_full is None:
            turnovers.append(np.nan)
        else:
            turnovers.append(turnover_total_variation(prev_drift_full, full_now))

        prev_drift_full = drift_full_weights(full_now, r_ex_next=r_ex_next, rf_next=rf_next)

    wealth = np.asarray(wealth)
    port_rets = np.asarray(port_rets)
    pis = np.asarray(pis)
    cashes = np.asarray(cashes)
    turnovers = np.asarray(turnovers)
    hedge_l1 = np.asarray(hedge_l1, dtype=float)
    mu_l1 = np.asarray(mu_l1, dtype=float)
    a_l1 = np.asarray(a_l1, dtype=float)

    rf_seg = rf[start_idx + 1 : start_idx + 1 + H]
    m = compute_metrics(port_rets, rf=rf_seg, periods_per_year=12, gamma=gamma)

    gross = np.sum(np.abs(pis), axis=1)
    net = np.sum(pis, axis=1)

    return {
        "wealth": wealth,
        "port_rets": port_rets,
        "pi": pis,
        "cash": cashes,
        "turnover": turnovers,
        "metrics": m,
        "cross_scale": float(cross_scale),
        "projection_diag": {
            "hedge_term_l1": hedge_l1,
            "mu_term_l1": mu_l1,
            "a_vec_l1": a_l1,
            "avg_hedge_term_l1": float(np.nanmean(hedge_l1)),
            "avg_mu_term_l1": float(np.nanmean(mu_l1)),
            "avg_a_vec_l1": float(np.nanmean(a_l1)),
            "hedge_to_mu_ratio": float(np.nanmean(hedge_l1 / np.maximum(mu_l1, 1e-12))),
        },
        "diag": {
            "avg_gross": float(np.nanmean(gross)),
            "avg_net": float(np.nanmean(net)),
            "avg_turnover": float(np.nanmean(turnovers)),
            "avg_cash": float(np.nanmean(cashes)),
            "min_cash": float(np.nanmin(cashes)),
            "max_cash": float(np.nanmax(cashes)),
        },
    }


def refit_model_params_expanding(
    base_model: DiscreteLatentMarketModel,
    y_df: pd.DataFrame,
    r_ex: pd.DataFrame,
    z: Optional[pd.DataFrame],
    train_end: int,
    *,
    cross_mode: str = "estimated",
) -> DiscreteLatentMarketModel:
    """
    Expanding-window refit of (a,B,Sigma) and transition (VARX) and Cross,
    **keeping the same PCA/scaler** as base_model to keep the state representation stable.

    The same world-level cross_mode ablation is re-applied after each refit so
    EXPANDING runs stay in the chosen counterfactual world.
    """
    apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
    varx = fit_varx_transition(y_df, z, train_end=train_end, )
    Cross = estimate_cross_cov(apt, varx)

    model = DiscreteLatentMarketModel(
        asset_names=base_model.asset_names,
        state_names=base_model.state_names,
        exog_names=base_model.exog_names,
        a=apt.a,
        B=apt.B,
        Sigma=apt.Sigma,
        c=varx.c,
        A=varx.A,
        G=varx.G,
        Q=varx.Q,
        Cross=Cross,
        pca=base_model.pca,
        scaler_mean=base_model.scaler_mean,
        scaler_std=base_model.scaler_std,
        trans_beta=varx.beta,
        trans_feature_names=varx.feature_names,
        trans_gvarx_cfg=varx.gvarx_cfg,
        trans_y_mean=varx.y_mean,
        trans_y_std=varx.y_std,
        trans_z_mean=varx.z_mean,
        trans_z_std=varx.z_std,
    )
    return _apply_cross_mode_to_model(model, cross_mode)


def refit_model_params_windowed(
    base_model: DiscreteLatentMarketModel,
    y_df: pd.DataFrame,
    r_ex: pd.DataFrame,
    z: Optional[pd.DataFrame],
    train_start: int,
    train_end: int,
    *,
    cross_mode: str = "estimated",
) -> DiscreteLatentMarketModel:
    """Windowed refit that reuses the base PCA/scaler and slices only the
    estimation sample for the APT / VARX parameter updates.
    """
    train_start = int(max(train_start, 0))
    train_end = int(train_end)
    if train_start <= 0:
        return refit_model_params_expanding(
            base_model=base_model,
            y_df=y_df,
            r_ex=r_ex,
            z=z,
            train_end=train_end,
            cross_mode=cross_mode,
        )

    y_sub = y_df.iloc[train_start : train_end + 1].copy()
    r_sub = r_ex.iloc[train_start : train_end + 1].copy()
    z_sub = None if z is None else z.iloc[train_start : train_end + 1].copy()
    rel_end = int(len(y_sub) - 1)
    if rel_end < 5:
        return refit_model_params_expanding(
            base_model=base_model,
            y_df=y_df,
            r_ex=r_ex,
            z=z,
            train_end=train_end,
            cross_mode=cross_mode,
        )

    apt = fit_apt_mu_sigma(y_sub, r_sub, train_end=rel_end, shrink_cov=True)
    varx = fit_varx_transition(y_sub, z_sub, train_end=rel_end)
    Cross = estimate_cross_cov(apt, varx)

    model = DiscreteLatentMarketModel(
        asset_names=base_model.asset_names,
        state_names=base_model.state_names,
        exog_names=base_model.exog_names,
        a=apt.a,
        B=apt.B,
        Sigma=apt.Sigma,
        c=varx.c,
        A=varx.A,
        G=varx.G,
        Q=varx.Q,
        Cross=Cross,
        pca=base_model.pca,
        scaler_mean=base_model.scaler_mean,
        scaler_std=base_model.scaler_std,
        trans_beta=varx.beta,
        trans_feature_names=varx.feature_names,
        trans_gvarx_cfg=varx.gvarx_cfg,
        trans_y_mean=varx.y_mean,
        trans_y_std=varx.y_std,
        trans_z_mean=varx.z_mean,
        trans_z_std=varx.z_std,
    )
    return _apply_cross_mode_to_model(model, cross_mode)


def _resolve_walk_forward_mode_name(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode == "":
        mode = "fixed"
    if mode not in {"fixed", "expanding", "rolling"}:
        raise ValueError(f"Unknown walk-forward mode: {value}")
    return mode


def _resolve_rolling_train_months(value: Any) -> Optional[int]:
    if value is None:
        return None
    months = int(value)
    return None if months <= 0 else months


def _window_bounds_for_refit(
    train_end: int,
    *,
    window_mode: str,
    rolling_train_months: Optional[int],
) -> tuple[int, int]:
    train_end = int(train_end)
    mode = _resolve_walk_forward_mode_name(window_mode)
    months = _resolve_rolling_train_months(rolling_train_months)
    if mode == "rolling" and months is not None:
        train_start = max(0, train_end - months + 1)
    else:
        train_start = 0
    return int(train_start), int(train_end)


def _refresh_residual_policy_baseline_if_supported(policy: Any, torch_model: Any) -> None:
    refresh = getattr(policy, "refresh_baseline", None)
    if callable(refresh):
        refresh(a=torch_model.a, B=torch_model.B, Sigma=torch_model.Sigma)


def simulate_realized_expanding(
    policy: PortfolioPolicy,
    base_model: DiscreteLatentMarketModel,
    y_df: pd.DataFrame,
    r_ex_df: pd.DataFrame,
    rf: np.ndarray,
    start_idx: int,
    H_eval: int,
    H_train: int,
    z_df: Optional[pd.DataFrame],
    device: torch.device,
    dtype: torch.dtype,
    gamma: float,
    retrain_every: int,
    refit_iters: int,
    batch_size: int,
    lr: float,
    *,
    cross_mode: str = "estimated",
    do_ppgdpo: bool = False,
    ppgdpo_cfg: Optional[PPGDPOConfig] = None,
    ppgdpo_variant_specs: Optional[list[dict[str, Any]]] = None,
    # extra model-based benchmarks (computed from model_t.Sigma and model_t.mu)
    model_bench_list: Optional[list[str]] = None,
    bench_w_max: float | None = None,
    rp_max_iter: int = 500,
    rp_tol: float = 1e-8,
    constraints: Optional[PortfolioConstraints] = None,
    window_mode: str = "expanding",
    rolling_train_months: int | None = None,
) -> dict:
    """
    Walk-forward realized evaluation over the last H_eval months:

    At each month j:
      - re-estimate the model parameters on an expanding window up to time t
      - (optional) periodically continue policy training on the updated simulator
      - compute policy weights using y_t and realized next return
      - compute the true myopic benchmark using the same model_t and y_t

    Optional:
      - if do_ppgdpo=True, also compute a P-PGDPO (Pontryagin projection) action each month
        using the *current* model_t and *current* policy, then apply to realized returns.
    """
    window_mode = _resolve_walk_forward_mode_name(window_mode)
    rolling_train_months = _resolve_rolling_train_months(rolling_train_months)
    r_ex_np = r_ex_df.values.astype(float)
    y_all = y_df.values.astype(float)
    constraint_cfg = PortfolioConstraints() if constraints is None else constraints
    constraint_cfg.validate(r_ex_np.shape[1])
    z_all = None if z_df is None else z_df.values.astype(float)

    # hedging-channel time series (based on model_t Cross)
    hedge_mean_abs = []
    hedge_p95_abs = []
    hedge_max_abs = []

    # outputs: baseline policy
    X_pol = 1.0
    wealth_pol = [X_pol]
    rets_pol = []
    pis_pol = []
    cash_pol = []
    to_pol = []
    prev_pol_drift = None

    # outputs: true myopic benchmark
    X_mvo = 1.0
    wealth_mvo = [X_mvo]
    rets_mvo = []
    pis_mvo = []
    cash_mvo = []
    to_mvo = []
    prev_mvo_drift = None

    # outputs: PPGDPO projection variants (optional)
    if do_ppgdpo:
        if ppgdpo_cfg is None:
            raise ValueError("ppgdpo_cfg must be provided when do_ppgdpo=True")
        variant_specs = list(ppgdpo_variant_specs) if ppgdpo_variant_specs else [{
            "name": "ppgdpo",
            "label": "P-PGDPO",
            "cross_scale": 1.0,
            "kind": "main",
        }]
        ppgdpo_tracks: dict[str, dict[str, Any]] = {}
        for var in variant_specs:
            ppgdpo_tracks[str(var["name"])] = {
                "spec": dict(var),
                "X": 1.0,
                "wealth": [1.0],
                "port_rets": [],
                "pi": [],
                "cash": [],
                "turnover": [],
                "prev_pi_drift": None,
                "hedge_term_l1": [],
                "mu_term_l1": [],
                "a_vec_l1": [],
            }
    else:
        variant_specs = []
        ppgdpo_tracks = {}

    # -----------------------------------------------------------------
    # Optional: extra *model-based* benchmarks in walk-forward mode.
    # These are the counterparts of the FIXED-mode "*_model" benchmarks,
    # but re-computed each month using the current refit model_t.
    # -----------------------------------------------------------------
    model_bench_list = list(model_bench_list) if model_bench_list is not None else []
    model_bench: dict[str, dict[str, Any]] = {}

    def _init_track() -> dict[str, Any]:
        return {
            "X": 1.0,
            "wealth": [1.0],
            "port_rets": [],
            "pi": [],
            "cash": [],
            "turnover": [],
            "prev_pi_drift": None,
        }

    if len(model_bench_list) > 0:
        if "gmv" in model_bench_list:
            model_bench["gmv_model"] = _init_track()
        if "risk_parity" in model_bench_list:
            model_bench["risk_parity_model"] = _init_track()
        if "inv_vol" in model_bench_list:
            model_bench["inv_vol_model"] = _init_track()
        if "static_mvo" in model_bench_list:
            model_bench["static_mvo_model"] = _init_track()

    tau_step = 1.0 / float(H_eval)
    sim_cfg_full = DiscreteSimConfig(horizon_steps=int(H_eval), gamma=float(gamma), kappa=1.0)

    # start with the base model at start_idx (already trained)
    model_t = _apply_cross_mode_to_model(base_model, cross_mode)

    for j in range(H_eval):
        t = start_idx + j
        train_end = t
        train_start, train_end = _window_bounds_for_refit(
            train_end,
            window_mode=window_mode,
            rolling_train_months=rolling_train_months,
        )

        # Refit model params on the chosen walk-forward window (past only)
        model_t = refit_model_params_windowed(
            base_model=base_model,
            y_df=y_df,
            r_ex=r_ex_df,
            z=z_df,
            train_start=train_start,
            train_end=train_end,
            cross_mode=cross_mode,
        )
        torch_model = TorchDiscreteLatentModel(model_t, device=device, dtype=dtype)
        _refresh_residual_policy_baseline_if_supported(policy, torch_model)

        # record hedging-channel strength at this step (diagnostic only)
        hstats, _rho, _top = cross_rho_stats(
            model_t.Sigma,
            model_t.Q,
            model_t.Cross,
            asset_names=model_t.asset_names,
            state_names=model_t.state_names,
            top_k=1,
        )
        hedge_mean_abs.append(hstats["mean_abs"])
        hedge_p95_abs.append(hstats["p95_abs"])
        hedge_max_abs.append(hstats["max_abs"])

        # Optional periodic re-training / fine-tuning on updated simulator
        if retrain_every > 0 and j > 0 and (j % retrain_every == 0):
            # Need enough history to sample H_train-month episodes.
            # If exogenous predictors are present (z_all), we need s+H_train <= train_end to slice z_path.
            y_hist_train = y_all[train_start : train_end + 1]
            z_hist_train = None if z_all is None else z_all[train_start : train_end + 1]
            hist_len = int(y_hist_train.shape[0])
            start_max = (hist_len - 1) if z_hist_train is None else (hist_len - 1 - H_train)
            if start_max > 5:
                sampler = EpisodeSampler(
                    y_all=y_hist_train,
                    z_all=z_hist_train,
                    start_max=start_max,
                    horizon=H_train,
                )
                rf_hist = np.asarray(rf[train_start : train_end + 1], dtype=float)
                rf_month = float(np.nanmean(rf_hist))
                sim_cfg = DiscreteSimConfig(horizon_steps=H_train, gamma=float(gamma), kappa=1.0)
                train_cfg = TrainConfig(
                    iters=int(refit_iters),
                    batch_size=int(batch_size),
                    lr=float(lr),
                    weight_decay=0.0,
                    clip_grad_norm=1.0,
                    print_every=max(50, int(refit_iters // 2)),
                )
                mode_label = window_mode.upper()
                print(f"\n[{mode_label}] Re-train @ step={j} (train_start={train_start}, train_end={train_end}) iters={train_cfg.iters} batch={train_cfg.batch_size} lr={train_cfg.lr}")
                train_pgdpo_discrete(torch_model, policy, sampler, rf_month=rf_month, sim_cfg=sim_cfg, train_cfg=train_cfg)
                policy.eval()
            else:
                mode_label = window_mode.upper()
                print(f"\n[{mode_label}] Skip retrain @ step={j} (insufficient history for horizon={H_train} with exog={0 if z_all is None else z_all.shape[1]}).")

        # --- policy action (uses current wealth X_pol)
        with torch.no_grad():
            y_t_torch = torch.tensor(y_all[t], device=device, dtype=dtype).view(1, -1)
            x_t_torch = torch.tensor([[X_pol]], device=device, dtype=dtype)
            tau = torch.tensor([[1.0 - j / H_eval]], device=device, dtype=dtype)
            pi_t, cash_t = policy.weights_and_cash(tau, x_t_torch, y_t_torch)
            pi_pol = pi_t.detach().cpu().numpy().reshape(-1)
            c_pol = float(cash_t.detach().cpu().item()) if cash_t is not None else float(1.0 - pi_pol.sum())

        # --- true myopic action (same utility; ignores continuation / hedging)
        pi_mvo, c_mvo = myopic_from_model(model_t, y_all[t], gamma=float(gamma), ridge=1e-6, w_max=None, constraints=constraint_cfg)

        # --- ppgdpo actions (optional; potentially multiple projection variants)
        ppgdpo_actions: dict[str, dict[str, Any]] = {}
        if do_ppgdpo:
            assert ppgdpo_cfg is not None
            y_pp = torch.tensor(y_all[t], device=device, dtype=dtype).view(1, -1)

            H_rem = H_eval - j
            sim_cfg_rem = DiscreteSimConfig(
                horizon_steps=int(H_rem),
                gamma=float(sim_cfg_full.gamma),
                kappa=float(sim_cfg_full.kappa),
                clamp_wealth_min=float(sim_cfg_full.clamp_wealth_min),
                clamp_state_std_abs=sim_cfg_full.clamp_state_std_abs,
                clamp_port_ret_min=float(sim_cfg_full.clamp_port_ret_min),
                clamp_port_ret_max=float(sim_cfg_full.clamp_port_ret_max),
            )

            z_path = None
            if z_all is not None:
                z_future = z_all[t : t + H_rem, :]
                z_path = torch.tensor(z_future, device=device, dtype=dtype).unsqueeze(0)

            tau0 = 1.0 - j / float(H_eval)
            rf_month_step = float(np.nanmean(rf[: train_end + 1]))

            for var in variant_specs:
                ppgdpo_cfg_step = ppgdpo_cfg
                if ppgdpo_cfg.seed is not None:
                    ppgdpo_cfg_step = PPGDPOConfig(**ppgdpo_cfg.__dict__)
                    ppgdpo_cfg_step.seed = int(ppgdpo_cfg.seed) + int(t)

                tr_var = ppgdpo_tracks[str(var["name"])]
                x_pp_torch = torch.tensor([[float(tr_var["X"])]], device=device, dtype=dtype)
                out_pp = ppgdpo_action(
                    model=torch_model,
                    policy=policy,
                    X=x_pp_torch,
                    Y=y_pp,
                    z_path=z_path,
                    rf_month=rf_month_step,
                    sim_cfg=sim_cfg_rem,
                    ppgdpo_cfg=ppgdpo_cfg_step,
                    tau0=float(tau0),
                    tau_step=float(tau_step),
                    cross_scale=float(var.get("cross_scale", 1.0)),
                    return_debug=True,
                )
                dbg = out_pp.get("debug", {})
                ppgdpo_actions[str(var["name"])] = {
                    "pi": out_pp["u_pp"].detach().cpu().numpy().reshape(-1),
                    "cash": float(out_pp["cash_pp"].detach().cpu().item()),
                    "hedge_term_l1": float(dbg["hedge_term"].abs().sum(dim=1).mean().detach().cpu().item()) if isinstance(dbg.get("hedge_term"), torch.Tensor) else np.nan,
                    "mu_term_l1": float(dbg["mu_term"].abs().sum(dim=1).mean().detach().cpu().item()) if isinstance(dbg.get("mu_term"), torch.Tensor) else np.nan,
                    "a_vec_l1": float(dbg["a_vec"].abs().sum(dim=1).mean().detach().cpu().item()) if isinstance(dbg.get("a_vec"), torch.Tensor) else np.nan,
                }

        # realized next-month return
        r_ex_next = r_ex_np[t + 1]
        rf_next = float(rf[t + 1])

        # turnover to reach the target from last period's *drifted* holdings
        to_pol.append(_turnover_one_way_from_drift(pi_pol, prev_pol_drift))
        to_mvo.append(_turnover_one_way_from_drift(pi_mvo, prev_mvo_drift))

        ret_pol = rf_next + float(np.dot(pi_pol, r_ex_next))
        ret_mvo = rf_next + float(np.dot(pi_mvo, r_ex_next))

        X_pol = X_pol * (1.0 + ret_pol)
        X_mvo = X_mvo * (1.0 + ret_mvo)

        wealth_pol.append(X_pol)
        wealth_mvo.append(X_mvo)
        rets_pol.append(ret_pol)
        rets_mvo.append(ret_mvo)
        pis_pol.append(pi_pol)
        pis_mvo.append(pi_mvo)
        cash_pol.append(c_pol)
        cash_mvo.append(c_mvo)

        # drift weights forward for the next rebalance
        prev_pol_drift, _cash_pol_drift = _drift_risky_weights(pi_pol, c_pol, r_ex_next, rf_next)
        prev_mvo_drift, _cash_mvo_drift = _drift_risky_weights(pi_mvo, c_mvo, r_ex_next, rf_next)

        if do_ppgdpo:
            for name, act in ppgdpo_actions.items():
                tr_var = ppgdpo_tracks[name]
                pi_pp = np.asarray(act["pi"], dtype=float)
                c_pp = float(act["cash"])
                tr_var["turnover"].append(_turnover_one_way_from_drift(pi_pp, tr_var["prev_pi_drift"]))
                ret_pp = rf_next + float(np.dot(pi_pp, r_ex_next))
                tr_var["X"] = float(tr_var["X"]) * (1.0 + ret_pp)
                tr_var["wealth"].append(float(tr_var["X"]))
                tr_var["port_rets"].append(float(ret_pp))
                tr_var["pi"].append(pi_pp)
                tr_var["cash"].append(c_pp)
                tr_var["hedge_term_l1"].append(float(act.get("hedge_term_l1", np.nan)))
                tr_var["mu_term_l1"].append(float(act.get("mu_term_l1", np.nan)))
                tr_var["a_vec_l1"].append(float(act.get("a_vec_l1", np.nan)))
                tr_var["prev_pi_drift"], _cash_pp_drift = _drift_risky_weights(pi_pp, c_pp, r_ex_next, rf_next)

        # -------------------------------------------------------------
        # Extra model-based benchmarks (EXPANDING): weights recomputed
        # each month from the current refit model_t.
        # -------------------------------------------------------------
        if model_bench:
            Sigma_model = np.asarray(model_t.Sigma, dtype=float)

            # For static_mvo_model we use the expanding average of mu_t(y_t)
            # under the *current* refit parameters.
            mu_bar = None
            if "static_mvo_model" in model_bench:
                y_hist = y_all[train_start : train_end + 1, :]
                mu_hist = model_t.a[None, :] + y_hist @ model_t.B.T
                mu_bar = np.nanmean(mu_hist, axis=0)

            # compute + apply each benchmark for this month
            for name, tr in model_bench.items():
                if name == "gmv_model":
                    w_b, c_b = gmv_weights_long_only_cash(Sigma_model, ridge=1e-6, w_max=bench_w_max, constraints=constraint_cfg)
                elif name == "risk_parity_model":
                    w_b, c_b = risk_parity_weights_long_only_cash(
                        Sigma_model, max_iter=int(rp_max_iter), tol=float(rp_tol), ridge=1e-8, w_max=bench_w_max, constraints=constraint_cfg
                    )
                elif name == "inv_vol_model":
                    w_b, c_b = inv_vol_weights_long_only_cash(Sigma_model, w_max=bench_w_max, constraints=constraint_cfg)
                elif name == "static_mvo_model":
                    assert mu_bar is not None
                    w_b, c_b = markowitz_weights_long_only_cash(
                        mu_bar, Sigma_model, risk_aversion=float(gamma), ridge=1e-6, w_max=bench_w_max, constraints=constraint_cfg
                    )
                else:
                    continue

                # turnover (drift-based)
                to_b = _turnover_one_way_from_drift(w_b, tr["prev_pi_drift"])
                tr["turnover"].append(to_b)

                # realized return + wealth update
                ret_b = rf_next + float(np.dot(w_b, r_ex_next))
                tr["X"] = float(tr["X"]) * (1.0 + ret_b)
                tr["wealth"].append(float(tr["X"]))
                tr["port_rets"].append(float(ret_b))
                tr["pi"].append(np.asarray(w_b, dtype=float))
                tr["cash"].append(float(c_b))

                # drift for next rebalance
                tr["prev_pi_drift"], _cash_d = _drift_risky_weights(w_b, c_b, r_ex_next, rf_next)

    wealth_pol = np.asarray(wealth_pol)
    wealth_mvo = np.asarray(wealth_mvo)
    rets_pol = np.asarray(rets_pol)
    rets_mvo = np.asarray(rets_mvo)
    pis_pol = np.asarray(pis_pol)
    pis_mvo = np.asarray(pis_mvo)
    cash_pol = np.asarray(cash_pol)
    cash_mvo = np.asarray(cash_mvo)
    to_pol = np.asarray(to_pol)
    to_mvo = np.asarray(to_mvo)

    rf_seg = rf[start_idx + 1 : start_idx + 1 + H_eval]
    m_pol = compute_metrics(rets_pol, rf=rf_seg, periods_per_year=12, gamma=gamma)
    m_mvo = compute_metrics(rets_mvo, rf=rf_seg, periods_per_year=12, gamma=gamma)

    def _diag(pis: np.ndarray, cash: np.ndarray, turnovers: np.ndarray) -> dict:
        gross = np.sum(np.abs(pis), axis=1)
        net = np.sum(pis, axis=1)
        return {
            "avg_gross": float(np.nanmean(gross)),
            "avg_net": float(np.nanmean(net)),
            "avg_turnover": float(np.nanmean(turnovers)),
            "avg_cash": float(np.nanmean(cash)),
            "min_cash": float(np.nanmin(cash)),
            "max_cash": float(np.nanmax(cash)),
        }

    out = {
        "policy": {
            "wealth": wealth_pol,
            "port_rets": rets_pol,
            "pi": pis_pol,
            "cash": cash_pol,
            "turnover": to_pol,
            "metrics": m_pol,
            "diag": _diag(pis_pol, cash_pol, to_pol),
        },
        "myopic": {
            "wealth": wealth_mvo,
            "port_rets": rets_mvo,
            "pi": pis_mvo,
            "cash": cash_mvo,
            "turnover": to_mvo,
            "metrics": m_mvo,
            "diag": _diag(pis_mvo, cash_mvo, to_mvo),
        },
        "hedging_series": {
            "mean_abs_rho": np.asarray(hedge_mean_abs, dtype=float),
            "p95_abs_rho": np.asarray(hedge_p95_abs, dtype=float),
            "max_abs_rho": np.asarray(hedge_max_abs, dtype=float),
        },
    }

    if do_ppgdpo:
        out_pp_variants: dict[str, dict[str, Any]] = {}
        for name, tr in ppgdpo_tracks.items():
            wealth_pp_arr = np.asarray(tr["wealth"], dtype=float)
            rets_pp_arr = np.asarray(tr["port_rets"], dtype=float)
            pis_pp_arr = np.asarray(tr["pi"], dtype=float)
            cash_pp_arr = np.asarray(tr["cash"], dtype=float)
            to_pp_arr = np.asarray(tr["turnover"], dtype=float)
            hedge_arr = np.asarray(tr.get("hedge_term_l1", []), dtype=float)
            mu_arr = np.asarray(tr.get("mu_term_l1", []), dtype=float)
            a_arr = np.asarray(tr.get("a_vec_l1", []), dtype=float)

            m_pp = compute_metrics(rets_pp_arr, rf=rf_seg, periods_per_year=12, gamma=gamma)
            out_pp_variants[name] = {
                "wealth": wealth_pp_arr,
                "port_rets": rets_pp_arr,
                "pi": pis_pp_arr,
                "cash": cash_pp_arr,
                "turnover": to_pp_arr,
                "metrics": m_pp,
                "cross_scale": float(tr.get("spec", {}).get("cross_scale", 1.0)),
                "label": str(tr.get("spec", {}).get("label", name)),
                "diag": _diag(pis_pp_arr, cash_pp_arr, to_pp_arr),
                "projection_diag": {
                    "hedge_term_l1": hedge_arr,
                    "mu_term_l1": mu_arr,
                    "a_vec_l1": a_arr,
                    "avg_hedge_term_l1": float(np.nanmean(hedge_arr)) if hedge_arr.size else np.nan,
                    "avg_mu_term_l1": float(np.nanmean(mu_arr)) if mu_arr.size else np.nan,
                    "avg_a_vec_l1": float(np.nanmean(a_arr)) if a_arr.size else np.nan,
                    "hedge_to_mu_ratio": float(np.nanmean(hedge_arr / np.maximum(mu_arr, 1e-12))) if hedge_arr.size else np.nan,
                },
            }
        if "ppgdpo" in out_pp_variants:
            out["ppgdpo"] = out_pp_variants["ppgdpo"]
        out["ppgdpo_variants"] = out_pp_variants

    # attach extra model-based benchmarks (if requested)
    if model_bench:
        out_bench: dict[str, dict] = {}
        for name, tr in model_bench.items():
            wealth_b = np.asarray(tr["wealth"], dtype=float)
            rets_b = np.asarray(tr["port_rets"], dtype=float)
            pis_b = np.asarray(tr["pi"], dtype=float)
            cash_b = np.asarray(tr["cash"], dtype=float)
            to_b = np.asarray(tr["turnover"], dtype=float)

            m_b = compute_metrics(rets_b, rf=rf_seg, periods_per_year=12, gamma=gamma)
            out_bench[name] = {
                "wealth": wealth_b,
                "port_rets": rets_b,
                "pi": pis_b,
                "cash": cash_b,
                "turnover": to_b,
                "metrics": m_b,
                "diag": _diag(pis_b, cash_b, to_b),
            }
        out["model_bench"] = out_bench

    return out


# -----------------------------------------------------------------------------
# Spec builders are now defined in pgdpo_yahoo.state_specs.
# This runner imports build_model_for_state_spec() so the v58 training/eval core
# no longer owns spec construction.
# -----------------------------------------------------------------------------

def simulate_realized_expanding_myopic_proxy(
    base_model: DiscreteLatentMarketModel,
    y_df: pd.DataFrame,
    r_ex_df: pd.DataFrame,
    rf: np.ndarray,
    start_idx: int,
    H_eval: int,
    z_df: Optional[pd.DataFrame],
    gamma: float,
    *,
    cross_mode: str = "zero",
    model_bench_list: Optional[list[str]] = None,
    bench_w_max: float | None = None,
    rp_max_iter: int = 500,
    rp_tol: float = 1e-8,
    constraints: Optional[PortfolioConstraints] = None,
    window_mode: str = "expanding",
    rolling_train_months: int | None = None,
) -> dict:
    """Walk-forward realized evaluation using the true myopic benchmark as the
    policy leg in a zero-cross world.

    This is the v63 default when cross_mode=zero: we do not re-train a full
    dynamic policy/PPGDPO, and instead use the same-utility myopic benchmark as
    the policy proxy.
    """
    window_mode = _resolve_walk_forward_mode_name(window_mode)
    rolling_train_months = _resolve_rolling_train_months(rolling_train_months)
    r_ex_np = r_ex_df.values.astype(float)
    y_all = y_df.values.astype(float)
    constraint_cfg = PortfolioConstraints() if constraints is None else constraints
    constraint_cfg.validate(r_ex_np.shape[1])

    hedge_mean_abs = []
    hedge_p95_abs = []
    hedge_max_abs = []

    X_mvo = 1.0
    wealth_mvo = [X_mvo]
    rets_mvo = []
    pis_mvo = []
    cash_mvo = []
    to_mvo = []
    prev_mvo_drift = None

    model_bench_list = list(model_bench_list) if model_bench_list is not None else []
    model_bench: dict[str, dict[str, Any]] = {}

    def _init_track() -> dict[str, Any]:
        return {
            "X": 1.0,
            "wealth": [1.0],
            "port_rets": [],
            "pi": [],
            "cash": [],
            "turnover": [],
            "prev_pi_drift": None,
        }

    if len(model_bench_list) > 0:
        if "gmv" in model_bench_list:
            model_bench["gmv_model"] = _init_track()
        if "risk_parity" in model_bench_list:
            model_bench["risk_parity_model"] = _init_track()
        if "inv_vol" in model_bench_list:
            model_bench["inv_vol_model"] = _init_track()
        if "static_mvo" in model_bench_list:
            model_bench["static_mvo_model"] = _init_track()

    for j in range(H_eval):
        t = start_idx + j
        train_end = t
        train_start, train_end = _window_bounds_for_refit(
            train_end,
            window_mode=window_mode,
            rolling_train_months=rolling_train_months,
        )

        model_t = refit_model_params_windowed(
            base_model=base_model,
            y_df=y_df,
            r_ex=r_ex_df,
            z=z_df,
            train_start=train_start,
            train_end=train_end,
            cross_mode=cross_mode,
        )

        hstats, _rho, _top = cross_rho_stats(
            model_t.Sigma,
            model_t.Q,
            model_t.Cross,
            asset_names=model_t.asset_names,
            state_names=model_t.state_names,
            top_k=1,
        )
        hedge_mean_abs.append(hstats["mean_abs"])
        hedge_p95_abs.append(hstats["p95_abs"])
        hedge_max_abs.append(hstats["max_abs"])

        pi_mvo, c_mvo = myopic_from_model(model_t, y_all[t], gamma=float(gamma), ridge=1e-6, w_max=None, constraints=constraint_cfg)

        r_ex_next = r_ex_np[t + 1]
        rf_next = float(rf[t + 1])

        to_mvo.append(_turnover_one_way_from_drift(pi_mvo, prev_mvo_drift))
        ret_mvo = rf_next + float(np.dot(pi_mvo, r_ex_next))
        X_mvo = X_mvo * (1.0 + ret_mvo)

        wealth_mvo.append(X_mvo)
        rets_mvo.append(ret_mvo)
        pis_mvo.append(pi_mvo)
        cash_mvo.append(c_mvo)

        prev_mvo_drift, _cash_mvo_drift = _drift_risky_weights(pi_mvo, c_mvo, r_ex_next, rf_next)

        if model_bench:
            Sigma_model = np.asarray(model_t.Sigma, dtype=float)
            mu_bar = None
            if "static_mvo_model" in model_bench:
                y_hist = y_all[train_start : train_end + 1, :]
                mu_hist = model_t.a[None, :] + y_hist @ model_t.B.T
                mu_bar = np.nanmean(mu_hist, axis=0)

            for name, tr in model_bench.items():
                if name == "gmv_model":
                    w_b, c_b = gmv_weights_long_only_cash(Sigma_model, ridge=1e-6, w_max=bench_w_max, constraints=constraint_cfg)
                elif name == "risk_parity_model":
                    w_b, c_b = risk_parity_weights_long_only_cash(
                        Sigma_model, max_iter=int(rp_max_iter), tol=float(rp_tol), ridge=1e-8, w_max=bench_w_max, constraints=constraint_cfg
                    )
                elif name == "inv_vol_model":
                    w_b, c_b = inv_vol_weights_long_only_cash(Sigma_model, w_max=bench_w_max, constraints=constraint_cfg)
                elif name == "static_mvo_model":
                    assert mu_bar is not None
                    w_b, c_b = markowitz_weights_long_only_cash(
                        mu_bar, Sigma_model, risk_aversion=float(gamma), ridge=1e-6, w_max=bench_w_max, constraints=constraint_cfg
                    )
                else:
                    continue

                to_b = _turnover_one_way_from_drift(w_b, tr["prev_pi_drift"])
                tr["turnover"].append(to_b)

                ret_b = rf_next + float(np.dot(w_b, r_ex_next))
                tr["X"] = float(tr["X"]) * (1.0 + ret_b)
                tr["wealth"].append(float(tr["X"]))
                tr["port_rets"].append(float(ret_b))
                tr["pi"].append(np.asarray(w_b, dtype=float))
                tr["cash"].append(float(c_b))

                tr["prev_pi_drift"], _cash_d = _drift_risky_weights(w_b, c_b, r_ex_next, rf_next)

    wealth_mvo = np.asarray(wealth_mvo)
    rets_mvo = np.asarray(rets_mvo)
    pis_mvo = np.asarray(pis_mvo)
    cash_mvo = np.asarray(cash_mvo)
    to_mvo = np.asarray(to_mvo)

    rf_seg = rf[start_idx + 1 : start_idx + 1 + H_eval]
    m_mvo = compute_metrics(rets_mvo, rf=rf_seg, periods_per_year=12, gamma=gamma)

    def _diag(pis: np.ndarray, cash: np.ndarray, turnovers: np.ndarray) -> dict:
        gross = np.sum(np.abs(pis), axis=1)
        net = np.sum(pis, axis=1)
        return {
            "avg_gross": float(np.nanmean(gross)),
            "avg_net": float(np.nanmean(net)),
            "avg_turnover": float(np.nanmean(turnovers)),
            "avg_cash": float(np.nanmean(cash)),
            "min_cash": float(np.nanmin(cash)),
            "max_cash": float(np.nanmax(cash)),
        }

    myopic_logs = {
        "wealth": wealth_mvo,
        "port_rets": rets_mvo,
        "pi": pis_mvo,
        "cash": cash_mvo,
        "turnover": to_mvo,
        "metrics": m_mvo,
        "diag": _diag(pis_mvo, cash_mvo, to_mvo),
    }

    out = {
        "policy": _policy_logs_from_myopic_proxy(myopic_logs, proxy_name="myopic_zero_cross_proxy"),
        "myopic": myopic_logs,
        "hedging_series": {
            "mean_abs_rho": np.asarray(hedge_mean_abs, dtype=float),
            "p95_abs_rho": np.asarray(hedge_p95_abs, dtype=float),
            "max_abs_rho": np.asarray(hedge_max_abs, dtype=float),
        },
        "policy_source": "myopic_zero_cross_proxy",
        "zero_cross_policy_proxy": True,
    }

    if model_bench:
        out_bench: dict[str, dict] = {}
        for name, tr in model_bench.items():
            wealth_b = np.asarray(tr["wealth"], dtype=float)
            rets_b = np.asarray(tr["port_rets"], dtype=float)
            pis_b = np.asarray(tr["pi"], dtype=float)
            cash_b = np.asarray(tr["cash"], dtype=float)
            to_b = np.asarray(tr["turnover"], dtype=float)
            m_b = compute_metrics(rets_b, rf=rf_seg, periods_per_year=12, gamma=gamma)
            out_bench[name] = {
                "wealth": wealth_b,
                "port_rets": rets_b,
                "pi": pis_b,
                "cash": cash_b,
                "turnover": to_b,
                "metrics": m_b,
                "diag": _diag(pis_b, cash_b, to_b),
            }
        out["model_bench"] = out_bench

    return out


def _spec_tags(spec: str, args: argparse.Namespace) -> Tuple[str, str]:
    # v36: linear VARX only (no generalized/nonlinear VARX).
    trans_tag = "varx"
    mode_tag = _resolve_walk_forward_mode_name(getattr(args, "walk_forward_mode", "expanding" if args.expanding_window else "fixed"))

    # Make PLS configs explicit in filenames to avoid confusion across runs.
    # v42 note: tuned PLS specs encode H and k in the spec name (e.g. pls_H6_k2),
    # so we should NOT append args.pls_horizon (which may differ).
    import re

    if spec == "pls_only":
        mode_tag += f"_plsh{int(args.pls_horizon)}"
    elif re.fullmatch(r"pls_H\d+_k\d+", spec):
        # Horizon already in spec string; keep mode_tag short.
        pass
    if spec.startswith("pls") and int(args.pls_smooth_span) > 0:
        mode_tag += f"_plsema{int(args.pls_smooth_span)}"

    return trans_tag, mode_tag


def _apply_cross_mode_to_model(
    model: DiscreteLatentMarketModel,
    cross_mode: str = "estimated",
) -> DiscreteLatentMarketModel:
    """Apply the world-level Cross handling used by the empirical ablation.

    - estimated: keep the fitted innovation cross-covariance Cross = Cov(eps, u)
    - zero:      force Cross=0 after each fit/refit, giving a counterfactual
                 no-hedging world for simulation, policy training, and P-PGDPO
    """
    mode = str(cross_mode).lower()
    if mode == "estimated":
        return model
    if mode == "zero":
        model.Cross = np.zeros_like(model.Cross)
        return model
    raise ValueError(f"Unknown cross_mode={cross_mode!r} (expected 'estimated' or 'zero').")


def run_one_state_spec(
    *,
    spec: str,
    r_ex: pd.DataFrame,
    rf: pd.Series,
    macro3: Optional[pd.DataFrame],
    macro7: Optional[pd.DataFrame],
    ff3: Optional[pd.DataFrame],
    ff5: Optional[pd.DataFrame],
    bond_asset_names: list[str],
    train_end: int,
    start_idx: int,
    H_train: int,
    H_eval: int,
    r_ex_np: np.ndarray,
    rf_np: np.ndarray,
    baseline_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    pca_cfg: LatentPCAConfig,
    out_dir: Path,
    eval_label: str = "TEST",
) -> Dict[str, Any]:
    """Run training + evaluation for one state-spec.

    Parameters
    ----------
    eval_label : str
        Label used in console output to distinguish validation vs test blocks.
        (e.g., "VAL" or "TEST").
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # File/label tags used for all outputs
    trans_tag, mode_tag = _spec_tags(spec, args)

    model0, y_df, z_aligned = build_model_for_state_spec(
        spec,
        r_ex=r_ex,
        macro3=macro3,
        macro7=macro7,
        ff3=ff3,
        ff5=ff5,
        train_end=train_end,
        pca_cfg=pca_cfg,
        pls_horizon=int(args.pls_horizon),
        pls_smooth_span=int(args.pls_smooth_span),
        bond_asset_names=bond_asset_names,
        block_eq_k=int(getattr(args, "block_eq_k", 1)),
        block_bond_k=int(getattr(args, "block_bond_k", 1)),
    )
    # Optional ablation: apply the chosen Cross world immediately after fitting.
    # In v58 this is treated as a *world-level* counterfactual, so the same
    # cross_mode is also re-applied after each expanding-window refit.
    cross_mode = str(getattr(args, "cross_mode", "estimated"))
    model0 = _apply_cross_mode_to_model(model0, cross_mode)

    y_all = y_df.values.astype(float)
    z_all = None if z_aligned is None else z_aligned.values.astype(float)

    n_states = int(y_df.shape[1])
    n_exog = 0 if z_aligned is None else int(z_aligned.shape[1])
    trans_tag, mode_tag = _spec_tags(spec, args)

    print("\n================================================================================")
    print(f"MODEL-BASED (DISCRETE) [{spec}]  states={n_states}  exog={n_exog}  trans={trans_tag}")
    print("================================================================================")
    print(f"Model fit on TRAIN: obs={train_end+1} assets={r_ex.shape[1]} states={n_states} exog={n_exog}")
    print(
        "Cross world: "
        + ("estimated fitted Cross" if cross_mode == "estimated" else "counterfactual Cross=0 (no hedging channel)")
    )

    # --- Diagnostics: hedging channel & predictability (TRAIN) ---
    apt0 = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
    apt0._Y_true = r_ex.values.astype(float)[1 : train_end + 1, :]  # type: ignore[attr-defined]
    varx0 = fit_varx_transition(y_df, z_aligned, train_end=train_end, )
    varx0._Y_true = y_df.values.astype(float)[1 : train_end + 1, :]  # type: ignore[attr-defined]
    print_model_hedging_diagnostics(
        title=f"TRAIN diagnostics [{spec}] (helps decide: hedging is weak vs policy not capturing)",
        model=model0,
        apt=apt0,
        varx=varx0,
        top_k=8,
    )

    zero_cross_myopic_proxy = _use_zero_cross_myopic_proxy(args, cross_mode)
    policy_source = "trained_policy_zero_cross" if (str(cross_mode).lower() == "zero" and not zero_cross_myopic_proxy) else "trained_policy"
    policy = None
    torch_model0 = None
    sim_cfg0 = None
    ppgdpo_cfg: Optional[PPGDPOConfig] = None
    rf_month = float(np.nanmean(rf_np[: train_end + 1]))
    constraint_cfg = _build_portfolio_constraints_from_args(args, r_ex.shape[1])
    ppgdpo_L_eff = _resolve_ppgdpo_risky_cap(args, constraint_cfg)
    print(f"[Constraints] {constraint_cfg.summary()}")
    if getattr(args, "ppgdpo_L", None) is not None and abs(float(args.ppgdpo_L) - float(ppgdpo_L_eff)) > 1e-12:
        print(f"[Constraints] P-PGDPO risky cap override clipped by cash_floor: requested L={float(args.ppgdpo_L):.4f}, effective={float(ppgdpo_L_eff):.4f}")

    if zero_cross_myopic_proxy:
        policy_source = "myopic_zero_cross_proxy"
        print("[Zero-cross proxy] cross_mode=zero -> using same-utility myopic benchmark as the policy leg; skipping PG-DPO / P-PGDPO training.")
        if getattr(args, "ppgdpo", False):
            print("[Zero-cross proxy] --ppgdpo ignored because policy is proxied by the true myopic benchmark in cross=0 mode.")
    else:
        if str(cross_mode).lower() == "zero":
            print("[Zero-cross dynamic] cross_mode=zero -> solving the same dynamic problem with Cross=0; Myopic remains a separate same-utility benchmark.")
        # Episode sampler for training:
        # - If we have exogenous predictors (z_all), we need s+H <= train_end to slice z_path.
        # - If exog is None (common for current specs: pca_only/macro3_only/pls_only),
        #   we can sample y0 anywhere in TRAIN and simulate forward under the fitted model.
        if z_all is None:
            start_max = train_end
        else:
            start_max = train_end - H_train
        if start_max <= 5:
            raise RuntimeError(
                "Not enough TRAIN observations to sample training episodes. "
                "Reduce --train_horizon, disable exog, or increase the data window."
            )
        sampler = EpisodeSampler(
            y_all=y_all[: train_end + 1],
            z_all=None if z_all is None else z_all[: train_end + 1],
            start_max=start_max,
            horizon=H_train,
        )

        # Torch model (used for simulation + baseline in residual policy)
        torch_model0 = TorchDiscreteLatentModel(model0, device=device, dtype=dtype)

        # Policy
        if bool(getattr(args, "residual_policy", False)):
            pol_cfg = ResidualPolicyConfig(
                n_assets=r_ex.shape[1],
                n_states=n_states,
                hidden=64,
                depth=2,
                include_time=True,
                softmax_temperature=2.0,
                softmax_epsilon=0.01,
                gamma=float(args.gamma),
                ridge=float(getattr(args, "residual_ridge", 1e-6)),
                w_max=None if getattr(args, "residual_w_max", None) is None else float(args.residual_w_max),
                constraints=constraint_cfg,
                detach_baseline=bool(getattr(args, "residual_detach_baseline", False)),
            )
            policy = ResidualLongOnlyCashPolicy(
                pol_cfg,
                a=torch_model0.a,
                B=torch_model0.B,
                Sigma=torch_model0.Sigma,
            ).to(device=device, dtype=dtype)
            print(f"[Policy] residual_on_myopic: gamma={pol_cfg.gamma} ridge={pol_cfg.ridge} temp={pol_cfg.softmax_temperature} eps={pol_cfg.softmax_epsilon}")
        else:
            pol_cfg = PolicyConfig(
                n_assets=r_ex.shape[1],
                n_states=n_states,
                hidden=64,
                depth=2,
                weight_transform="long_only_cash",
                softmax_temperature=2.0,
                softmax_epsilon=0.01,
                include_time=True,
                constraints=constraint_cfg,
            )
            policy = PortfolioPolicy(pol_cfg).to(device=device, dtype=dtype)
            print(f"[Policy] vanilla: transform={pol_cfg.weight_transform} temp={pol_cfg.softmax_temperature} eps={pol_cfg.softmax_epsilon}")

        # Torch model + train
        sim_cfg0 = DiscreteSimConfig(horizon_steps=H_train, gamma=float(args.gamma), kappa=1.0)
        train_cfg0 = TrainConfig(
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=0.0,
            clip_grad_norm=1.0,
            print_every=200,
        )

        print(f"Training (model-based PG-DPO warm-up) [{spec}]: iters={train_cfg0.iters} batch={train_cfg0.batch_size} lr={train_cfg0.lr}")
        train_pgdpo_discrete(torch_model0, policy, sampler, rf_month=rf_month, sim_cfg=sim_cfg0, train_cfg=train_cfg0)
        policy.eval()

        if getattr(args, "ppgdpo", False):
            ppgdpo_cfg = PPGDPOConfig(
                mc_rollouts=int(args.ppgdpo_mc),
                sub_batch=int(args.ppgdpo_subbatch),
                seed=None if args.ppgdpo_seed is None else int(args.ppgdpo_seed),
                L_cap=float(ppgdpo_L_eff),
                eps_bar=float(args.ppgdpo_eps),
                ridge=float(args.ppgdpo_ridge),
                tau=float(args.ppgdpo_tau),
                armijo=float(args.ppgdpo_armijo),
                backtrack=float(args.ppgdpo_backtrack),
                max_newton=int(args.ppgdpo_newton),
                tol_grad=float(args.ppgdpo_tol_grad),
                max_ls=int(args.ppgdpo_ls),
                constraints=constraint_cfg,
            )

    # ---------------------------------------------------------------------
    # Evaluation: fixed segment OR walk-forward re-estimation
    # ---------------------------------------------------------------------
    walk_mode = _resolve_walk_forward_mode_name(getattr(args, "walk_forward_mode", "expanding" if args.expanding_window else "fixed"))
    if walk_mode == "fixed":
        logs_mvo = simulate_realized_fixed_myopic(
            model0, y_all=y_all, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval,
            gamma=float(args.gamma), ridge=1e-6, w_max=None, constraints=constraint_cfg
        )
        mm = logs_mvo["metrics"]

        if zero_cross_myopic_proxy:
            logs_pol = _policy_logs_from_myopic_proxy(logs_mvo, proxy_name="myopic_zero_cross_proxy")
            pm = logs_pol["metrics"]
        else:
            assert policy is not None
            logs_pol = simulate_realized_fixed_policy(
                policy,
                y_all=y_all,
                r_ex=r_ex_np,
                rf=rf_np,
                start_idx=start_idx,
                H=H_eval,
                gamma=float(args.gamma),
            )
            pm = logs_pol["metrics"]

        # --- Extra model-based benchmarks (use fitted model's mu(y) and Sigma; Cross ignored) ---
        model_bench_list = list(getattr(args, "benchmarks_model", ["gmv", "risk_parity", "inv_vol"]))
        model_bench_logs: dict[str, dict] = {}
        if len(model_bench_list) > 0:
            bench_w_max = getattr(args, "bench_w_max", None)
            rp_max_iter = int(getattr(args, "rp_max_iter", 500))
            rp_tol = float(getattr(args, "rp_tol", 1e-8))
            Sigma_model = np.asarray(model0.Sigma, dtype=float)

            if "gmv" in model_bench_list:
                w_gmv_m, _ = gmv_weights_long_only_cash(Sigma_model, ridge=1e-6, w_max=bench_w_max, constraints=constraint_cfg)
                model_bench_logs["gmv_model"] = simulate_realized_fixed_constant_weights(
                    w_gmv_m, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                )

            if "risk_parity" in model_bench_list:
                w_rp_m, _ = risk_parity_weights_long_only_cash(
                    Sigma_model, max_iter=rp_max_iter, tol=rp_tol, ridge=1e-8, w_max=bench_w_max, constraints=constraint_cfg
                )
                model_bench_logs["risk_parity_model"] = simulate_realized_fixed_constant_weights(
                    w_rp_m, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                )

            if "inv_vol" in model_bench_list:
                w_iv_m, _ = inv_vol_weights_long_only_cash(Sigma_model, w_max=bench_w_max, constraints=constraint_cfg)
                model_bench_logs["inv_vol_model"] = simulate_realized_fixed_constant_weights(
                    w_iv_m, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                )

            if "static_mvo" in model_bench_list:
                y_train = y_all[: train_end + 1, :]
                mu_t_train = model0.a[None, :] + y_train @ model0.B.T
                mu_bar = np.nanmean(mu_t_train, axis=0)
                w_smvo_m, _ = markowitz_weights_long_only_cash(
                    mu_bar, Sigma_model, risk_aversion=float(args.gamma), ridge=1e-6, w_max=bench_w_max, constraints=constraint_cfg
                )
                model_bench_logs["static_mvo_model"] = simulate_realized_fixed_constant_weights(
                    w_smvo_m, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                )

        if model_bench_logs:
            print("\nExtra baselines (model-based mu(y), Sigma from fitted model; Cross ignored):")
            for name, logs_b in model_bench_logs.items():
                print(f"Baseline {name}:")
                for k, v in logs_b["metrics"].items():
                    print(f"  {k:7s}: {v: .4f}")

        pol_label = _policy_leg_label(cross_mode, zero_cross_myopic_proxy)
        print(f"\n{str(eval_label).upper()} metrics ({pol_label}, realized) [FIXED] :", spec)
        for k, v in pm.items():
            print(f"  {k:7s}: {v: .4f}")

        print(f"\n{str(eval_label).upper()} metrics (Myopic, same utility / no continuation, realized) [FIXED] :", spec)
        for k, v in mm.items():
            print(f"  {k:7s}: {v: .4f}")

        logs_pp = None
        logs_pp_variants: dict[str, dict[str, Any]] = {}
        if (not zero_cross_myopic_proxy) and getattr(args, "ppgdpo", False):
            if (H_eval > 120) and (not getattr(args, "ppgdpo_allow_long_horizon", False)):
                print(
                    "\n[WARN] H_eval>120: P-PGDPO can be very slow at long horizons. "
                    "Continuing anyway (pass --ppgdpo_allow_long_horizon to silence this warning)."
                )

            assert ppgdpo_cfg is not None
            for var in _ppgdpo_variant_specs(args, cross_mode):
                logs_v = simulate_realized_fixed_ppgdpo(
                    torch_model=torch_model0,
                    policy=policy,
                    y_all=y_all,
                    z_all=z_all,
                    r_ex=r_ex_np,
                    rf=rf_np,
                    start_idx=start_idx,
                    H=H_eval,
                    rf_month=rf_month,
                    sim_cfg_full=sim_cfg0,
                    ppgdpo_cfg=ppgdpo_cfg,
                    gamma=float(args.gamma),
                    cross_scale=float(var.get("cross_scale", 1.0)),
                )
                logs_v["label"] = str(var.get("label", var["name"]))
                logs_v["cross_scale"] = float(var.get("cross_scale", 1.0))
                logs_pp_variants[str(var["name"])] = logs_v

            logs_pp = logs_pp_variants.get("ppgdpo")
            for name, logs_v in logs_pp_variants.items():
                ppm = logs_v["metrics"]
                vlabel = str(logs_v.get("label", name))
                print(f"\n{str(eval_label).upper()} metrics ({vlabel}, realized) [FIXED] :", spec)
                for k, v in ppm.items():
                    print(f"  {k:7s}: {v: .4f}")
                pdiag = logs_v.get("projection_diag", {})
                if isinstance(pdiag, dict) and ("avg_hedge_term_l1" in pdiag):
                    print(
                        f"  [proj] avg|hedge|={float(pdiag.get('avg_hedge_term_l1', np.nan)):.6f}  "
                        f"avg|mu|={float(pdiag.get('avg_mu_term_l1', np.nan)):.6f}  "
                        f"hedge/mu={float(pdiag.get('hedge_to_mu_ratio', np.nan)):.6f}"
                    )

        # --- Transaction cost sensitivity (turnover-based linear cost) ---
        if getattr(args, "tc_sweep", False):
            tc_bps_list = [float(x) for x in getattr(args, "tc_bps", [0, 5, 10, 25, 50])]
            rf_seg = rf_np[start_idx + 1 : start_idx + 1 + H_eval]

            df_pol_tc = tc_sweep_metrics(
                logs_pol["port_rets"],
                logs_pol["turnover"],
                rf_seg,
                gamma=float(args.gamma),
                tc_bps_list=tc_bps_list,
                periods_per_year=12,
            )
            df_mvo_tc = tc_sweep_metrics(
                logs_mvo["port_rets"],
                logs_mvo["turnover"],
                rf_seg,
                gamma=float(args.gamma),
                tc_bps_list=tc_bps_list,
                periods_per_year=12,
            )
            df_parts = {"policy": df_pol_tc, "myopic": df_mvo_tc}

            for _pp_name, _pp_logs in logs_pp_variants.items():
                df_pp_tc = tc_sweep_metrics(
                    _pp_logs["port_rets"],
                    _pp_logs["turnover"],
                    rf_seg,
                    gamma=float(args.gamma),
                    tc_bps_list=tc_bps_list,
                    periods_per_year=12,
                )
                df_parts[_pp_name] = df_pp_tc

            # --- Include TRAIN-based benchmarks (from baseline_df) when available ---
            try:
                dates_tc = r_ex.index[start_idx + 1 : start_idx + 1 + H_eval]
                base_tc = baseline_df.reindex(dates_tc)
                bench_cols = {
                    "gmv": ("gmv_ret", "gmv_turnover"),
                    "risk_parity": ("rp_ret", "rp_turnover"),
                    "inv_vol": ("inv_vol_ret", "inv_vol_turnover"),
                    "static_mvo": ("static_mvo_ret", "static_mvo_turnover"),
                }
                for bname, (rc, tc_col) in bench_cols.items():
                    if (rc in base_tc.columns) and (tc_col in base_tc.columns):
                        df_parts[bname] = tc_sweep_metrics(
                            base_tc[rc].values,
                            base_tc[tc_col].values,
                            rf_seg,
                            gamma=float(args.gamma),
                            tc_bps_list=tc_bps_list,
                            periods_per_year=12,
                        )
            except Exception:
                pass

            # --- Include model-based benchmarks (FIXED) when available ---
            if model_bench_logs:
                for bname, blog in model_bench_logs.items():
                    df_parts[bname] = tc_sweep_metrics(
                        blog["port_rets"],
                        blog["turnover"],
                        rf_seg,
                        gamma=float(args.gamma),
                        tc_bps_list=tc_bps_list,
                        periods_per_year=12,
                    )

            df_tc = pd.concat(df_parts, names=["strategy", "tc_bps"])
            tc_path = out_dir / f"tc_sweep_{spec}_{trans_tag}_{mode_tag}.csv"
            df_tc.to_csv(tc_path)
            print(f"\nSaved: {tc_path}")

            # Compact log (CER/Sharpe vs tc)
            print("\nTC sweep (net-of-cost) summary [FIXED] (rows=tc bps):")
            for strat, dfi in df_parts.items():
                cols = [c for c in ["cer_ann", "sharpe", "sortino", "ann_ret", "ann_vol", "max_dd", "calmar", "es95", "max_dd_dur", "underwater_frac"] if c in dfi.columns]
                print(f"\n[{strat}] avg_turnover={float(dfi['avg_turnover'].iloc[0]):.4f}")
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(dfi[cols].to_string(float_format=lambda x: f"{x: .4f}"))

        gap = 0.5 * np.sum(np.abs(logs_pol["pi"] - logs_mvo["pi"]), axis=1)
        print("\nPolicy vs Myopic deviation (L1/2 over risky weights) [FIXED]:")
        print(f"  avg={float(np.nanmean(gap)):.4f}  p95={float(np.nanquantile(gap, 0.95)):.4f}  max={float(np.nanmax(gap)):.4f}")

        # Save CSV
        trans_tag, mode_tag = _spec_tags(spec, args)
        years_tag = f"{int(round(H_eval/12))}y"
        asset_tag = str(getattr(args, "asset_universe", "french49"))
        out_path = out_dir / f"monthly_test_{years_tag}_{asset_tag}_{spec}_{trans_tag}_{mode_tag}.csv"
        dates = r_ex.index[start_idx + 1 : start_idx + 1 + H_eval]
        out = baseline_df.reindex(dates).copy()
        out["policy_source"] = policy_source
        out["zero_cross_policy_proxy"] = int(bool(zero_cross_myopic_proxy))
        out["cross_mode"] = cross_mode

        out["policy_port_ret"] = logs_pol["port_rets"]
        out["policy_wealth"] = logs_pol["wealth"][1:]
        out["policy_cash"] = logs_pol["cash"]
        out["policy_turnover"] = logs_pol["turnover"]
        for _pp_name, _pp_logs in logs_pp_variants.items():
            prefix = _pp_name
            out[f"{prefix}_port_ret"] = _pp_logs["port_rets"]
            out[f"{prefix}_wealth"] = _pp_logs["wealth"][1:]
            out[f"{prefix}_cash"] = _pp_logs["cash"]
            out[f"{prefix}_res_cash"] = _pp_logs["cash"] - logs_mvo["cash"]
            out[f"{prefix}_turnover"] = _pp_logs["turnover"]
            pdiag = _pp_logs.get("projection_diag", {})
            if isinstance(pdiag, dict):
                if "hedge_term_l1" in pdiag:
                    out[f"{prefix}_hedge_term_l1"] = pdiag["hedge_term_l1"]
                if "mu_term_l1" in pdiag:
                    out[f"{prefix}_mu_term_l1"] = pdiag["mu_term_l1"]
                if "a_vec_l1" in pdiag:
                    out[f"{prefix}_a_vec_l1"] = pdiag["a_vec_l1"]

        out["myopic_port_ret"] = logs_mvo["port_rets"]
        out["myopic_wealth"] = logs_mvo["wealth"][1:]
        out["myopic_cash"] = logs_mvo["cash"]
        out["res_cash"] = logs_pol["cash"] - logs_mvo["cash"]
        out["myopic_turnover"] = logs_mvo["turnover"]

        # attach model-based constant benchmarks (spec-specific; based on fitted mu/Sigma)
        if "gmv_model" in model_bench_logs:
            out["gmv_model_ret"] = model_bench_logs["gmv_model"]["port_rets"]
            out["gmv_model_wealth"] = model_bench_logs["gmv_model"]["wealth"][1:]
            out["gmv_model_cash"] = model_bench_logs["gmv_model"]["cash"]
            out["gmv_model_turnover"] = model_bench_logs["gmv_model"]["turnover"]
        if "risk_parity_model" in model_bench_logs:
            out["risk_parity_model_ret"] = model_bench_logs["risk_parity_model"]["port_rets"]
            out["risk_parity_model_wealth"] = model_bench_logs["risk_parity_model"]["wealth"][1:]
            out["risk_parity_model_cash"] = model_bench_logs["risk_parity_model"]["cash"]
            out["risk_parity_model_turnover"] = model_bench_logs["risk_parity_model"]["turnover"]
        if "inv_vol_model" in model_bench_logs:
            out["inv_vol_model_ret"] = model_bench_logs["inv_vol_model"]["port_rets"]
            out["inv_vol_model_wealth"] = model_bench_logs["inv_vol_model"]["wealth"][1:]
            out["inv_vol_model_cash"] = model_bench_logs["inv_vol_model"]["cash"]
            out["inv_vol_model_turnover"] = model_bench_logs["inv_vol_model"]["turnover"]
        if "static_mvo_model" in model_bench_logs:
            out["static_mvo_model_ret"] = model_bench_logs["static_mvo_model"]["port_rets"]
            out["static_mvo_model_wealth"] = model_bench_logs["static_mvo_model"]["wealth"][1:]
            out["static_mvo_model_cash"] = model_bench_logs["static_mvo_model"]["cash"]
            out["static_mvo_model_turnover"] = model_bench_logs["static_mvo_model"]["turnover"]


        w_pol = pd.DataFrame(logs_pol["pi"], index=dates, columns=[f"w_{name}" for name in r_ex.columns])
        w_mvo = pd.DataFrame(logs_mvo["pi"], index=dates, columns=[f"myopic_w_{name}" for name in r_ex.columns])
        w_res = pd.DataFrame(logs_pol["pi"] - logs_mvo["pi"], index=dates, columns=[f"res_w_{name}" for name in r_ex.columns])
        w_parts = [out, w_pol, w_mvo, w_res]
        for _pp_name, _pp_logs in logs_pp_variants.items():
            w_pp = pd.DataFrame(_pp_logs["pi"], index=dates, columns=[f"{_pp_name}_w_{name}" for name in r_ex.columns])
            w_pp_res = pd.DataFrame(_pp_logs["pi"] - logs_mvo["pi"], index=dates, columns=[f"{_pp_name}_res_w_{name}" for name in r_ex.columns])
            w_parts.append(w_pp)
            w_parts.append(w_pp_res)
        out = pd.concat(w_parts, axis=1)

        out.to_csv(out_path)
        print(f"Saved: {out_path}")

        if (policy is not None) and (not zero_cross_myopic_proxy):
            pt_path = out_dir / f"policy_{spec}_{trans_tag}_{mode_tag}.pt"
            torch.save(policy.state_dict(), pt_path)
            print(f"Saved: {pt_path}")
        else:
            print("[Zero-cross proxy] Policy checkpoint not written because the policy leg is proxied by the true myopic benchmark.")

        out_ret = {
            "spec": spec,
            "mode": "FIXED",
            "policy": pm,
            "myopic": mm,
            "policy_source": policy_source,
            "cross_mode": cross_mode,
            "zero_cross_policy_proxy": bool(zero_cross_myopic_proxy),
        }
        if logs_pp is not None:
            out_ret["ppgdpo"] = logs_pp["metrics"]
        if logs_pp_variants:
            out_ret["ppgdpo_variants"] = {
                _name: {
                    **_logs["metrics"],
                    "cross_scale": float(_logs.get("cross_scale", 1.0)),
                    "label": str(_logs.get("label", _name)),
                    "avg_hedge_term_l1": float(_logs.get("projection_diag", {}).get("avg_hedge_term_l1", np.nan)),
                    "avg_mu_term_l1": float(_logs.get("projection_diag", {}).get("avg_mu_term_l1", np.nan)),
                    "hedge_to_mu_ratio": float(_logs.get("projection_diag", {}).get("hedge_to_mu_ratio", np.nan)),
                }
                for _name, _logs in logs_pp_variants.items()
            }
        for _name, _logs in model_bench_logs.items():
            out_ret[_name] = _logs["metrics"]
        return out_ret

    # WALK-FORWARD
    print("\n================================================================================")
    print(f"{walk_mode.upper()} WINDOW [{spec}] (walk-forward on TEST segment)")
    print("================================================================================")
    print(f"retrain_every={args.retrain_every} months | refit_iters={args.refit_iters}")

    do_ppgdpo = bool(getattr(args, "ppgdpo", False)) and (not zero_cross_myopic_proxy)
    if do_ppgdpo and (H_eval > 120) and (not getattr(args, "ppgdpo_allow_long_horizon", False)):
        print(
            f"\n[WARN] H_eval>120 in {walk_mode.upper()} mode: P-PGDPO can be very slow. "
            "Continuing anyway (use --ppgdpo_allow_long_horizon to silence this warning)."
        )

    if zero_cross_myopic_proxy:
        logs = simulate_realized_expanding_myopic_proxy(
            base_model=model0,
            y_df=y_df,
            r_ex_df=r_ex,
            rf=rf_np,
            start_idx=start_idx,
            H_eval=H_eval,
            z_df=z_aligned,
            gamma=float(args.gamma),
            cross_mode=str(getattr(args, "cross_mode", "estimated")),
            model_bench_list=list(getattr(args, "benchmarks_model", [])),
            bench_w_max=getattr(args, "bench_w_max", None),
            rp_max_iter=int(getattr(args, "rp_max_iter", 500)),
            rp_tol=float(getattr(args, "rp_tol", 1e-8)),
            constraints=constraint_cfg,
            window_mode=walk_mode,
            rolling_train_months=getattr(args, "rolling_train_months", None),
        )
    else:
        assert policy is not None
        logs = simulate_realized_expanding(
            policy=policy,
            base_model=model0,
            y_df=y_df,
            r_ex_df=r_ex,
            rf=rf_np,
            start_idx=start_idx,
            H_eval=H_eval,
            H_train=H_train,
            z_df=z_aligned,
            device=device,
            dtype=dtype,
            gamma=float(args.gamma),
            retrain_every=int(args.retrain_every),
            refit_iters=int(args.refit_iters),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            cross_mode=str(getattr(args, "cross_mode", "estimated")),
            do_ppgdpo=do_ppgdpo,
            ppgdpo_cfg=ppgdpo_cfg,
            ppgdpo_variant_specs=_ppgdpo_variant_specs(args, str(getattr(args, "cross_mode", "estimated"))) if do_ppgdpo else None,
            model_bench_list=list(getattr(args, "benchmarks_model", [])),
            bench_w_max=getattr(args, "bench_w_max", None),
            rp_max_iter=int(getattr(args, "rp_max_iter", 500)),
            rp_tol=float(getattr(args, "rp_tol", 1e-8)),
            constraints=constraint_cfg,
            window_mode=walk_mode,
            rolling_train_months=getattr(args, "rolling_train_months", None),
        )
    logs_pol = logs["policy"]
    logs_mvo = logs["myopic"]

    pol_label = _policy_leg_label(str(getattr(args, "cross_mode", "estimated")), zero_cross_myopic_proxy)
    walk_mode_upper = walk_mode.upper()
    print(f"\n{str(eval_label).upper()} metrics ({pol_label}, realized) [{walk_mode_upper}]:", spec)
    for k, v in logs_pol["metrics"].items():
        print(f"  {k:7s}: {v: .4f}")

    print(f"\n{str(eval_label).upper()} metrics (Myopic, same utility / no continuation, realized) [{walk_mode_upper}]:", spec)
    for k, v in logs_mvo["metrics"].items():
        print(f"  {k:7s}: {v: .4f}")

    logs_pp = logs.get("ppgdpo", None)
    logs_pp_variants = logs.get("ppgdpo_variants", {})
    if isinstance(logs_pp_variants, dict) and len(logs_pp_variants) > 0:
        for _pp_name, _pp_logs in logs_pp_variants.items():
            print(f"\n{str(eval_label).upper()} metrics ({str(_pp_logs.get('label', _pp_name))}, realized) [{walk_mode_upper}]:", spec)
            for k, v in _pp_logs["metrics"].items():
                print(f"  {k:7s}: {v: .4f}")
            pdiag = _pp_logs.get("projection_diag", {})
            if isinstance(pdiag, dict) and ("avg_hedge_term_l1" in pdiag):
                print(
                    f"  [proj] avg|hedge|={float(pdiag.get('avg_hedge_term_l1', np.nan)):.6f}  "
                    f"avg|mu|={float(pdiag.get('avg_mu_term_l1', np.nan)):.6f}  "
                    f"hedge/mu={float(pdiag.get('hedge_to_mu_ratio', np.nan)):.6f}"
                )

    model_bench_logs = logs.get("model_bench", {})
    if isinstance(model_bench_logs, dict) and len(model_bench_logs) > 0:
        print(f"\nExtra baselines (model-based mu(y), Sigma from refit model_t; Cross ignored) [{walk_mode_upper}]:")
        for name, logs_b in model_bench_logs.items():
            print(f"Baseline {name}:")
            for k, v in logs_b["metrics"].items():
                print(f"  {k:7s}: {v: .4f}")

    # --- Transaction cost sensitivity (turnover-based linear cost) ---
    if getattr(args, "tc_sweep", False):
        tc_bps_list = [float(x) for x in getattr(args, "tc_bps", [0, 5, 10, 25, 50])]
        rf_seg = rf_np[start_idx + 1 : start_idx + 1 + H_eval]

        df_pol_tc = tc_sweep_metrics(
            logs_pol["port_rets"],
            logs_pol["turnover"],
            rf_seg,
            gamma=float(args.gamma),
            tc_bps_list=tc_bps_list,
            periods_per_year=12,
        )
        df_mvo_tc = tc_sweep_metrics(
            logs_mvo["port_rets"],
            logs_mvo["turnover"],
            rf_seg,
            gamma=float(args.gamma),
            tc_bps_list=tc_bps_list,
            periods_per_year=12,
        )
        df_parts = {"policy": df_pol_tc, "myopic": df_mvo_tc}

        for _pp_name, _pp_logs in logs_pp_variants.items():
            df_pp_tc = tc_sweep_metrics(
                _pp_logs["port_rets"],
                _pp_logs["turnover"],
                rf_seg,
                gamma=float(args.gamma),
                tc_bps_list=tc_bps_list,
                periods_per_year=12,
            )
            df_parts[_pp_name] = df_pp_tc

        # --- Include TRAIN-based benchmarks (from baseline_df) when available ---
        try:
            dates_tc = r_ex.index[start_idx + 1 : start_idx + 1 + H_eval]
            base_tc = baseline_df.reindex(dates_tc)
            bench_cols = {
                "gmv": ("gmv_ret", "gmv_turnover"),
                "risk_parity": ("rp_ret", "rp_turnover"),
                "inv_vol": ("inv_vol_ret", "inv_vol_turnover"),
                "static_mvo": ("static_mvo_ret", "static_mvo_turnover"),
            }
            for bname, (rc, tc_col) in bench_cols.items():
                if (rc in base_tc.columns) and (tc_col in base_tc.columns):
                    df_parts[bname] = tc_sweep_metrics(
                        base_tc[rc].values,
                        base_tc[tc_col].values,
                        rf_seg,
                        gamma=float(args.gamma),
                        tc_bps_list=tc_bps_list,
                        periods_per_year=12,
                    )
        except Exception as _e:
            # keep tc sweep robust even if baseline bench columns are missing
            pass

        # --- Include model-based benchmarks (if computed) ---
        if isinstance(model_bench_logs, dict) and len(model_bench_logs) > 0:
            for bname, blog in model_bench_logs.items():
                df_parts[bname] = tc_sweep_metrics(
                    blog["port_rets"],
                    blog["turnover"],
                    rf_seg,
                    gamma=float(args.gamma),
                    tc_bps_list=tc_bps_list,
                    periods_per_year=12,
                )

        df_tc = pd.concat(df_parts, names=["strategy", "tc_bps"])
        tc_path = out_dir / f"tc_sweep_{spec}_{trans_tag}_{mode_tag}.csv"
        df_tc.to_csv(tc_path)
        print(f"\nSaved: {tc_path}")

        print(f"\nTC sweep (net-of-cost) summary [{walk_mode_upper}] (rows=tc bps):")
        for strat, dfi in df_parts.items():
            cols = [c for c in ["cer_ann", "sharpe", "sortino", "ann_ret", "ann_vol", "max_dd", "calmar", "es95", "max_dd_dur", "underwater_frac"] if c in dfi.columns]
            print(f"\n[{strat}] avg_turnover={float(dfi['avg_turnover'].iloc[0]):.4f}")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(dfi[cols].to_string(float_format=lambda x: f"{x: .4f}"))

    gap = 0.5 * np.sum(np.abs(logs_pol["pi"] - logs_mvo["pi"]), axis=1)
    print(f"\nPolicy vs Myopic deviation (L1/2 over risky weights) [{walk_mode_upper}]:")
    print(f"  avg={float(np.nanmean(gap)):.4f}  p95={float(np.nanquantile(gap, 0.95)):.4f}  max={float(np.nanmax(gap)):.4f}")

    hedge = logs.get("hedging_series", {})
    if hedge:
        print(f"\nHedging channel strength over time (Cross -> rho) [{walk_mode_upper}]:")
        print(
            f"  mean(mean|rho|)={float(np.nanmean(hedge['mean_abs_rho'])):.4f}  "
            f"mean(p95|rho|)={float(np.nanmean(hedge['p95_abs_rho'])):.4f}  "
            f"mean(max|rho|)={float(np.nanmean(hedge['max_abs_rho'])):.4f}"
        )

    trans_tag, mode_tag = _spec_tags(spec, args)
    years_tag = f"{int(round(H_eval/12))}y"
    asset_tag = str(getattr(args, "asset_universe", "french49"))
    out_path = out_dir / f"monthly_test_{years_tag}_{asset_tag}_{spec}_{trans_tag}_{mode_tag}.csv"
    dates = r_ex.index[start_idx + 1 : start_idx + 1 + H_eval]
    out = baseline_df.reindex(dates).copy()
    out["policy_source"] = policy_source
    out["zero_cross_policy_proxy"] = int(bool(zero_cross_myopic_proxy))
    out["cross_mode"] = cross_mode

    out["policy_port_ret"] = logs_pol["port_rets"]
    out["policy_wealth"] = logs_pol["wealth"][1:]
    out["policy_cash"] = logs_pol["cash"]
    out["policy_turnover"] = logs_pol["turnover"]
    for _pp_name, _pp_logs in logs_pp_variants.items():
        prefix = _pp_name
        out[f"{prefix}_port_ret"] = _pp_logs["port_rets"]
        out[f"{prefix}_wealth"] = _pp_logs["wealth"][1:]
        out[f"{prefix}_cash"] = _pp_logs["cash"]
        out[f"{prefix}_res_cash"] = _pp_logs["cash"] - logs_mvo["cash"]
        out[f"{prefix}_turnover"] = _pp_logs["turnover"]
        pdiag = _pp_logs.get("projection_diag", {})
        if isinstance(pdiag, dict):
            if "hedge_term_l1" in pdiag:
                out[f"{prefix}_hedge_term_l1"] = pdiag["hedge_term_l1"]
            if "mu_term_l1" in pdiag:
                out[f"{prefix}_mu_term_l1"] = pdiag["mu_term_l1"]
            if "a_vec_l1" in pdiag:
                out[f"{prefix}_a_vec_l1"] = pdiag["a_vec_l1"]

    out["myopic_port_ret"] = logs_mvo["port_rets"]
    out["myopic_wealth"] = logs_mvo["wealth"][1:]
    out["myopic_cash"] = logs_mvo["cash"]
    out["res_cash"] = logs_pol["cash"] - logs_mvo["cash"]
    out["myopic_turnover"] = logs_mvo["turnover"]

    # attach model-based benchmarks (expanding refit)
    if isinstance(model_bench_logs, dict) and len(model_bench_logs) > 0:
        for _name, _logs in model_bench_logs.items():
            out[f"{_name}_ret"] = _logs["port_rets"]
            out[f"{_name}_wealth"] = _logs["wealth"][1:]
            out[f"{_name}_cash"] = _logs["cash"]
            out[f"{_name}_turnover"] = _logs["turnover"]


    if hedge:
        out["hedge_mean_abs_rho"] = hedge["mean_abs_rho"]
        out["hedge_p95_abs_rho"] = hedge["p95_abs_rho"]
        out["hedge_max_abs_rho"] = hedge["max_abs_rho"]

    w_pol = pd.DataFrame(logs_pol["pi"], index=dates, columns=[f"w_{name}" for name in r_ex.columns])
    w_mvo = pd.DataFrame(logs_mvo["pi"], index=dates, columns=[f"myopic_w_{name}" for name in r_ex.columns])
    w_res = pd.DataFrame(logs_pol["pi"] - logs_mvo["pi"], index=dates, columns=[f"res_w_{name}" for name in r_ex.columns])
    w_parts = [out, w_pol, w_mvo, w_res]
    for _pp_name, _pp_logs in logs_pp_variants.items():
        w_pp = pd.DataFrame(_pp_logs["pi"], index=dates, columns=[f"{_pp_name}_w_{name}" for name in r_ex.columns])
        w_pp_res = pd.DataFrame(_pp_logs["pi"] - logs_mvo["pi"], index=dates, columns=[f"{_pp_name}_res_w_{name}" for name in r_ex.columns])
        w_parts.append(w_pp)
        w_parts.append(w_pp_res)
    out = pd.concat(w_parts, axis=1)

    out.to_csv(out_path)
    print(f"Saved: {out_path}")

    if (policy is not None) and (not zero_cross_myopic_proxy):
        pt_path = out_dir / f"policy_{spec}_{trans_tag}_{mode_tag}.pt"
        torch.save(policy.state_dict(), pt_path)
        print(f"Saved: {pt_path}")
    else:
        print("[Zero-cross proxy] Policy checkpoint not written because the policy leg is proxied by the true myopic benchmark.")

    out_ret = {
        "spec": spec,
        "mode": walk_mode_upper,
        "policy": logs_pol["metrics"],
        "myopic": logs_mvo["metrics"],
        "policy_source": policy_source,
        "cross_mode": cross_mode,
        "zero_cross_policy_proxy": bool(zero_cross_myopic_proxy),
    }
    if logs_pp is not None:
        out_ret["ppgdpo"] = logs_pp["metrics"]
    if isinstance(logs_pp_variants, dict) and len(logs_pp_variants) > 0:
        out_ret["ppgdpo_variants"] = {
            _name: {
                **_logs["metrics"],
                "cross_scale": float(_logs.get("cross_scale", 1.0)),
                "label": str(_logs.get("label", _name)),
                "avg_hedge_term_l1": float(_logs.get("projection_diag", {}).get("avg_hedge_term_l1", np.nan)),
                "avg_mu_term_l1": float(_logs.get("projection_diag", {}).get("avg_mu_term_l1", np.nan)),
                "hedge_to_mu_ratio": float(_logs.get("projection_diag", {}).get("hedge_to_mu_ratio", np.nan)),
            }
            for _name, _logs in logs_pp_variants.items()
        }
    if isinstance(model_bench_logs, dict) and len(model_bench_logs) > 0:
        for _name, _logs in model_bench_logs.items():
            out_ret[_name] = _logs["metrics"]
    return out_ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fred_api_key",
        type=str,
        default=None,
        help=(
            "FRED API key (required for FRED-based specs: macro3_only/macro7_only/macro7_pca/ff5_macro7_pca "
            "and/or when --include_bond is enabled)."
        ),
    )
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    # ---------------------------------------------------------------------
    # Training hyperparameters (PG-DPO warm-up)
    # ---------------------------------------------------------------------
    ap.add_argument("--iters", type=int, default=800, help="Number of PG-DPO training iterations.")
    ap.add_argument("--batch_size", type=int, default=512, help="Batch size for PG-DPO training.")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PG-DPO training.")
    ap.add_argument("--gamma", type=float, default=5.0, help="CRRA risk aversion (utility parameter) used in training/evaluation.")
    # ---------------------------------------------------------------------
    # Residual policy parameterization (v54): policy learns residual on top of
    # a constrained same-utility myopic baseline under long-only+cash simplex.
    # ---------------------------------------------------------------------
    ap.add_argument("--residual_policy", dest="residual_policy", action="store_true", default=True,
                    help="Use residual (myopic+delta) long-only+cash policy parameterization (default: on in v54).")
    ap.add_argument("--no_residual_policy", dest="residual_policy", action="store_false",
                    help="Disable residual parameterization and use vanilla long_only_cash softmax policy.")
    ap.add_argument("--residual_ridge", type=float, default=1e-6,
                    help="Ridge added to Sigma for baseline solve in residual policy.")
    ap.add_argument("--residual_w_max", type=float, default=None,
                    help="Optional per-asset cap applied to baseline risky weights (before renorm).")
    ap.add_argument("--residual_detach_baseline", action="store_true",
                    help="Detach baseline weights from autograd (affects costate estimation for P-PGDPO only).")
    # ---------------------------------------------------------------------
    # Portfolio constraints for risky weights + residual cash.
    # Cash is always the residual 1 - sum(risky); negative cash means borrowing.
    # ---------------------------------------------------------------------
    ap.add_argument(
        "--risky_cap",
        type=float,
        default=1.0,
        help="Maximum total risky exposure sum(pi). Use 2.0 for 200%% gross long risky exposure (cash may be -1).",
    )
    ap.add_argument(
        "--cash_floor",
        type=float,
        default=0.0,
        help="Minimum residual cash weight. The effective risky cap is min(risky_cap, 1-cash_floor). Set -1.0 to allow 100%% borrowing.",
    )
    ap.add_argument(
        "--per_asset_cap",
        type=float,
        default=None,
        help="Optional per-asset upper bound on risky weights. Inactive by default.",
    )
    ap.add_argument(
        "--allow_short",
        action="store_true",
        default=False,
        help="Allow risky weights below zero down to --short_floor. Off by default; current mainline experiments use long-only risky weights.",
    )
    ap.add_argument(
        "--short_floor",
        type=float,
        default=0.0,
        help="Per-asset lower bound for risky weights when --allow_short is enabled. Ignored otherwise.",
    )
    ap.add_argument(
        "--latent_k",
        type=int,
        default=2,
        help=(
            "Number of latent factors for PCA/PLS states (pca_only/pls_only) and for PCA-compressed "
            "macro/mixed predictor sets (macro7_pca/ff5_macro7_pca). Ignored for raw macro/factor states "
            "(macro3_only/macro7_only/ff3_only/ff5_only)."
        ),
    )
    # ---------------------------------------------------------------------
    # PLS state construction (only used for pls_* specs)
    # ---------------------------------------------------------------------
    ap.add_argument(
        "--pls_horizon",
        type=int,
        default=12,
        help="(PLS) Horizon H (months) for predictive PLS target: average of next H returns. Use 1 to recover next-month PLS.",
    )
    ap.add_argument(
        "--pls_smooth_span",
        type=int,
        default=6,
        help="(PLS) Optional EWMA smoothing span for PLS scores to increase persistence (0 disables).",
    )
    ap.add_argument(
        "--state_spec",
        type=str,
        default=None,
        choices=list(STATE_SPECS),
        help=(
            "State design. v42 adds an explicit 22-spec sweep (STATE_SPECS_V42) "
            "with tuned names like pca_only_k2 / pls_H12_k3 / ff_mkt_smb. "
            "If omitted, defaults to pca_only_k2."
        ),
    )
    ap.add_argument("--compare_specs", action="store_true",
                    help="Run multiple state specs for comparison (prints a summary table).")
    ap.add_argument(
        "--specs",
        nargs="*",
        default=None,
        help=(
            "(Compare) List of specs to run. Default: STATE_SPECS_V42 (22-spec sweep). "
            "If you pass --specs explicitly, any STATE_SPECS entry is allowed (including v41 aliases)."
        ),
    )
    ap.add_argument(
        "--select_specs_first",
        action="store_true",
        help=(
            "Run method-free spec diagnostics first (predictive R^2 + stability + cross guard) "
            "and optionally shrink the spec list before training."
        ),
    )
    ap.add_argument(
        "--selection_only",
        action="store_true",
        help="Run only the spec-selection diagnostics and exit before any policy training.",
    )
    ap.add_argument(
        "--selection_top_k",
        type=int,
        default=0,
        help="If >0, keep only the top-k ranked specs after method-free spec selection.",
    )
    ap.add_argument(
        "--selection_window_mode",
        choices=["rolling", "expanding"],
        default="rolling",
        help="Window aggregation used in the predictive-R^2 stability check.",
    )
    ap.add_argument(
        "--selection_rolling_window",
        type=int,
        default=60,
        help="Window length (months) used in the predictive-R^2 stability check.",
    )
    ap.add_argument(
        "--selection_return_baseline",
        choices=["train_mean", "expanding_mean"],
        default="expanding_mean",
        help="Benchmark used in OOS return predictive R^2.",
    )
    ap.add_argument(
        "--selection_state_baseline",
        choices=["train_mean", "expanding_mean", "random_walk"],
        default="expanding_mean",
        help="Benchmark used in OOS state-transition predictive R^2.",
    )
    ap.add_argument("--selection_alpha", type=float, default=0.25, help=argparse.SUPPRESS)
    ap.add_argument("--selection_beta", type=float, default=0.50, help=argparse.SUPPRESS)
    ap.add_argument("--selection_gamma", type=float, default=0.25, help=argparse.SUPPRESS)
    ap.add_argument("--selection_return_floor", type=float, default=-0.02, help="Guard only: mark specs as failing when mean OOS return R^2 is at or below this threshold.")
    ap.add_argument("--selection_state_q10_floor", type=float, default=-0.05, help="Guard only: mark specs as failing when rolling state R^2 q10 is at or below this threshold.")
    ap.add_argument("--selection_cross_warn", type=float, default=0.95, help="Guard only: warn when max |rho(eps,u)| exceeds this level.")
    ap.add_argument("--selection_cross_fail", type=float, default=0.98, help="Guard only: mark specs as failing when max |rho(eps,u)| exceeds this level.")
    ap.add_argument("--selection_score_mode", choices=["method_shortlist", "legacy_value"], default="method_shortlist", help="Selection score philosophy. 'method_shortlist' (default) uses a rank-based shortlist score emphasizing return usefulness; 'legacy_value' reproduces the old value score.")
    ap.add_argument("--selection_score_ret_mean_weight", type=float, default=0.45, help="Weight on the mean OOS return R^2 rank in method_shortlist scoring.")
    ap.add_argument("--selection_score_ret_q10_weight", type=float, default=0.20, help="Weight on the rolling return R^2 q10 rank in method_shortlist scoring.")
    ap.add_argument("--selection_score_state_mean_weight", type=float, default=0.20, help="Weight on the mean OOS state R^2 rank in method_shortlist scoring.")
    ap.add_argument("--selection_score_state_q10_weight", type=float, default=0.15, help="Weight on the rolling state R^2 q10 rank in method_shortlist scoring.")
    ap.add_argument("--selection_guarded_only", action="store_true", default=False, help="When using --selection_top_k, restrict the shortlist to specs that pass the guard rails.")
    ap.add_argument(
        "--asset_universe",
        type=str,
        default="ff49ind",
        choices=list(ASSET_UNIVERSE_CHOICES),
        help=(
            "Equity test-asset menu. Bond inclusion remains controlled by --include_bond / --no_bond. "
            "Use ff25_szbm for the v61 mainline (FF 25 Size–B/M)."
        ),
    )

    ap.add_argument(
        "--custom_risky_assets_json",
        type=str,
        default="",
        help=(
            "JSON-encoded list of custom risky-asset definitions. "
            "Only used when --asset_universe custom."
        ),
    )

    ap.add_argument(
        "--bond_name",
        type=str,
        default="UST10Y",
        help="Column name for the added bond asset.",
    )

    ap.add_argument(
        "--include_bond",
        dest="include_bond",
        action="store_true",
        default=True,
        help=(
            "Include a bond asset as an additional risky asset (default: enabled). "
            "In v43 the default bond is a CRSP 10Y Treasury fixed-term monthly return series from CSV."
        ),
    )
    ap.add_argument(
        "--no_bond",
        dest="include_bond",
        action="store_false",
        help="Disable the bond asset.",
    )
    ap.add_argument(
        "--bond_source",
        type=str,
        default="crsp_csv",
        choices=["crsp_csv", "fred_tr"],
        help=(
            "Bond data source. 'crsp_csv' loads monthly returns from a local CSV (default). "
            "'fred_tr' downloads a total return index level from FRED and converts it to monthly returns."
        ),
    )
    ap.add_argument(
        "--bond_csv",
        type=str,
        default="data/bond10y_10y_fixedterm_monthly_monthend_min.csv",
        help=(
            "Path to the CRSP bond CSV when --bond_source crsp_csv. "
            "CSV must contain columns: date, bond10y_ret (decimal monthly returns)."
        ),
    )
    ap.add_argument(
        "--bond_series_id",
        type=str,
        default="BAMLCC0A0CMTRIV",
        help="FRED series ID for the bond total return index (used when --bond_source fred_tr).",
    )
    ap.add_argument(
        "--bond_ret_col",
        type=str,
        default="bond10y_ret",
        help="Return column to read from --bond_csv when --bond_source crsp_csv.",
    )
    ap.add_argument(
        "--bond_csv_specs",
        type=str,
        default="",
        help=(
            "Optional multi-bond CSV panel override. Format: "
            "NAME=path[@ret_col],NAME2=path2[@ret_col2]. "
            "When provided, overrides --bond_name/--bond_csv and loads multiple bond assets."
        ),
    )
    ap.add_argument(
        "--bond_fred_specs",
        type=str,
        default="",
        help=(
            "Optional multi-bond FRED override. Format: NAME=SERIESID,NAME2=SERIESID2. "
            "When provided with --bond_source fred_tr, overrides --bond_name/--bond_series_id."
        ),
    )
    ap.add_argument(
        "--block_eq_k",
        type=int,
        default=1,
        help="For *_eqbond_block specs: number of equity PCA components in the state block.",
    )
    ap.add_argument(
        "--block_bond_k",
        type=int,
        default=1,
        help="For *_eqbond_block specs: number of bond PCA components in the state block.",
    )

    ap.add_argument(
        "--end_date",
        type=str,
        default="2024-12-31",
        help=(
            "Hard end-date (YYYY-MM-DD) for the aligned dataset and evaluation windows. "
            "v43 default is 2024-12-31 so that the bundled CRSP 10Y bond series aligns cleanly."
        ),
    )
    ap.add_argument(
        "--train_pool_start",
        type=str,
        default=None,
        help=(
            "Optional lower bound (YYYY-MM-DD) for the aligned sample used by selection/comparison. "
            "Months before this date are dropped after common-calendar alignment."
        ),
    )

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    ap.add_argument(
        "--eval_mode",
        type=str,
        default="v53",
        choices=["v57", "v56", "v53", "v43", "legacy"],
        help=(
            "Evaluation plan. 'v56' performs rolling time-series CV on the pre-test DEV period "
            "(train+val) using --cv_folds folds, then evaluates on test20y (full last 20y). "
            "'v57' is identical to 'v56' but forces 3 rolling CV folds (coarser regime splits). "
            "'v53' uses a single internal validation split to avoid test leakage: val10y (last 10y "
            "of the pre-test training period) + test20y (full last 20y). "
            "'v43' runs exactly 3 windows ending at --end_date: block1 (first 10y of last20y), "
            "block2 (last 10y), and last20y (full 20y). "
            "'legacy' enables the older flags (--eval_3x10y / --eval_horizons / --eval_30y)."
        ),
    )

    # v56/v57 rolling CV controls (used only when --eval_mode v56 or v57)
    ap.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="(eval_mode=v56/v57) Number of rolling validation folds on the pre-test DEV period (train+val).",
    )
    ap.add_argument(
        "--cv_min_train_months",
        type=int,
        default=204,
        help=(
            "(eval_mode=v56/v57) Minimum number of monthly observations in the initial training set "
            "before fold1 validation starts. Must be >=202 for French49 (train_end index > 200)."
        ),
    )
    ap.add_argument(
        "--cv_val_months",
        type=int,
        default=0,
        help=(
            "(eval_mode=v56/v57) Validation window length (months). If 0, computed automatically to "
            "fit the DEV period into cv_folds sequential blocks (remainder goes to last fold)."
        ),
    )

    # Legacy evaluation flags (kept for backwards compatibility)
    ap.add_argument(
        "--eval_3x10y",
        action="store_true",
        help="Evaluate last 30 years as 3 contiguous 10-year out-of-sample blocks (block1/2/3).",
    )
    ap.add_argument(
        "--eval_horizons",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Evaluate the last N year(s) as separate out-of-sample blocks. "
            "Example: --eval_horizons 10 20 30. "
            "Can be combined with --eval_3x10y (blocks are unioned; overlaps are allowed). "
            "If omitted and no other eval flags are set: default is last 10y."
        ),
    )
    ap.add_argument(
        "--eval_30y",
        action="store_true",
        help="Alias for adding a single last-30y out-of-sample block (equivalent to --eval_horizons 30).",
    )
    ap.add_argument(
        "--force_common_calendar",
        dest="force_common_calendar",
        action="store_true",
        help=(
            "Force all runs (even single-spec runs) to align on a common calendar that includes "
            "the FRED macro predictors when a FRED key is available. This avoids spec-dependent "
            "train/test periods when running specs separately. (default: on)"
        ),
    )
    ap.add_argument(
        "--no_force_common_calendar",
        dest="force_common_calendar",
        action="store_false",
        help="Disable forced common calendar alignment.",
    )
    ap.set_defaults(force_common_calendar=True)

    ap.add_argument(
        "--common_start_mode",
        choices=["macro3", "suite"],
        default="suite",
        help=(
            "When --force_common_calendar is enabled, choose which series define the common aligned "
            "start date. 'macro3' reproduces the older behavior (align only on macro3 when available). "
            "'suite' enforces a single start date across the full v44 22-spec sweep by aligning on "
            "macro7 + FF5 (and bond, if enabled). (default: suite)"
        ),
    )


    ap.add_argument(
        "--train_horizon",
        type=int,
        default=0,
        help=(
            "Training episode horizon (months) for model-based PG-DPO warm-up. "
            "0 means: match each block's test horizon (and auto-cap by available TRAIN length)."
        ),
    )
    ap.add_argument(
        "--benchmarks",
        nargs="*",
        default=["gmv", "risk_parity", "inv_vol"],
        help=(
            "Extra TRAIN-based baselines to compute: gmv risk_parity inv_vol (optional: static_mvo). "
            "In FIXED mode these are constant weights estimated on TRAIN; "
            "in EXPANDING mode they are refit on an expanding window (walk-forward). "
            "Pass none to disable."
        ),
    )
    ap.add_argument(
        "--bench_w_max",
        type=float,
        default=None,
        help="Optional extra per-asset cap for benchmark weights (combined with --per_asset_cap when both are set).",
    )
    ap.add_argument(
        "--rp_max_iter",
        type=int,
        default=500,
        help="(risk_parity) Max iterations for the fixed-point solver.",
    )
    ap.add_argument(
        "--rp_tol",
        type=float,
        default=1e-8,
        help="(risk_parity) Convergence tolerance (max relative RC dispersion).",
    )

    ap.add_argument(
        "--bench_refit_every",
        type=int,
        default=1,
        help=(
            "(EXPANDING mode) Refit TRAIN-based benchmark weights every k months (default: 1 = monthly). "
            "Weights are held fixed between refits, but the portfolio is still rebalanced monthly to the last target."
        ),
    )

    ap.add_argument(
        "--benchmarks_model",
        nargs="*",
        default=["gmv", "risk_parity", "inv_vol"],
        help=(
            "Extra baselines computed from the fitted discrete model's (mu(y), Sigma) used by "
            "PG-DPO/PPGDPO. Cross is ignored (cross=0) by construction. "
            "Choices: gmv risk_parity inv_vol (optional: static_mvo). Pass none to disable."
        ),
    )
    ap.add_argument(
        "--cross_mode",
        choices=["estimated", "zero"],
        default="estimated",
        help=(
            "Innovation cross-covariance handling for the fitted world: 'estimated' keeps the fitted Cross after each fit/refit; "
            "'zero' forces Cross=0 after each fit/refit (counterfactual no-hedging world)."
        ),
    )
    ap.add_argument(
        "--zero_cross_policy_proxy",
        choices=["myopic", "full"],
        default="full",
        help=(
            "When cross_mode=zero, choose how to populate the policy leg: 'full' (default) solves the same dynamic problem in the zero-cross world; "
            "'myopic' uses the same-utility one-step benchmark as a proxy and skips PG-DPO/PPGDPO training."
        ),
    )

    # ---------------------------------------------------------------------
    # Turnover sensitivity: transaction-cost sweep
    # ---------------------------------------------------------------------
    ap.add_argument(
        "--tc_sweep",
        dest="tc_sweep",
        action="store_true",
        default=True,
        help="Compute OOS metrics net of linear transaction costs over a grid (uses realized turnover). (default: enabled)",
    )
    ap.add_argument(
        "--no_tc_sweep",
        dest="tc_sweep",
        action="store_false",
        help="Disable the transaction-cost sweep.",
    )
    ap.add_argument(
        "--tc_bps",
        nargs="*",
        type=float,
        default=[0, 5, 10, 25, 50],
        help="Transaction cost grid in basis points (one-way). Only used when --tc_sweep is set.",
    )

    # ---------------------------------------------------------------------
    # P-PGDPO Pontryagin projection (deployment-time projection; optional)
    # ---------------------------------------------------------------------
    ap.add_argument(
        "--ppgdpo",
        dest="ppgdpo",
        action="store_true",
        default=True,
        help="Evaluate P-PGDPO (Pontryagin projection) actions (in addition to policy + myopic). (default: enabled)",
    )
    ap.add_argument(
        "--no_ppgdpo",
        dest="ppgdpo",
        action="store_false",
        help="Disable P-PGDPO evaluation.",
    )
    ap.add_argument("--ppgdpo_mc", type=int, default=256, help="(P-PGDPO) MC rollouts per decision time for costate estimation.")
    ap.add_argument("--ppgdpo_subbatch", type=int, default=64, help="(P-PGDPO) Sub-batch size for costate MC (controls memory).")
    ap.add_argument("--ppgdpo_seed", type=int, default=123, help="(P-PGDPO) Base RNG seed (step-dependent seeds are derived from this).")
    ap.add_argument("--ppgdpo_L", type=float, default=None, help="(P-PGDPO) Optional override for the total risky cap used in the projection stage. By default it inherits the effective cap implied by --risky_cap / --cash_floor.")
    ap.add_argument("--ppgdpo_eps", type=float, default=1e-6, help="(P-PGDPO) Log-barrier coefficient.")
    ap.add_argument("--ppgdpo_ridge", type=float, default=1e-10, help="(P-PGDPO) Ridge added to Hessian for stability.")
    ap.add_argument("--ppgdpo_tau", type=float, default=0.95, help="(P-PGDPO) Fraction-to-the-boundary for Newton step.")
    ap.add_argument("--ppgdpo_armijo", type=float, default=1e-4, help="(P-PGDPO) Armijo constant (maximize).")
    ap.add_argument("--ppgdpo_backtrack", type=float, default=0.5, help="(P-PGDPO) Backtracking factor for line search.")
    ap.add_argument("--ppgdpo_newton", type=int, default=30, help="(P-PGDPO) Max Newton iterations.")
    ap.add_argument("--ppgdpo_tol_grad", type=float, default=1e-8, help="(P-PGDPO) Gradient infinity-norm tolerance.")
    ap.add_argument("--ppgdpo_ls", type=int, default=20, help="(P-PGDPO) Max line-search steps.")
    ap.add_argument(
        "--ppgdpo_local_zero_hedge",
        action="store_true",
        default=False,
        help=(
            "(P-PGDPO) In estimated-cross runs, also evaluate a local ablation that keeps the trained policy/costates "
            "fixed but sets the projection-stage Cross term to zero."
        ),
    )
    ap.add_argument(
        "--ppgdpo_cross_scales",
        nargs="*",
        type=float,
        default=None,
        help=(
            "(P-PGDPO) Optional extra projection-stage Cross multipliers to evaluate in the current world. "
            "Example: --ppgdpo_cross_scales 0 0.5 2."
        ),
    )

    # Expanding window (walk-forward)
    ap.add_argument(
        "--ppgdpo_allow_long_horizon",
        dest="ppgdpo_allow_long_horizon",
        action="store_true",
        help=(
            "Allow P-PGDPO evaluation for horizons > 120 months (can be VERY slow). "
            "(default: enabled)"
        ),
    )
    ap.add_argument(
        "--ppgdpo_disallow_long_horizon",
        dest="ppgdpo_allow_long_horizon",
        action="store_false",
        help="Disable P-PGDPO evaluation for horizons > 120 months.",
    )
    ap.set_defaults(ppgdpo_allow_long_horizon=True)

    ap.add_argument(
        "--walk_forward_mode",
        choices=["fixed", "expanding", "rolling"],
        default=None,
        help="Walk-forward re-estimation mode on the test segment.",
    )
    ap.add_argument("--expanding_window", action="store_true", help="Legacy alias for --walk_forward_mode expanding.")
    ap.add_argument("--rolling_train_months", type=int, default=0, help="(Rolling) trailing train window in calendar months; 0 uses the full initial train pool length.")
    ap.add_argument("--retrain_every", type=int, default=12, help="(Walk-forward) Continue training policy every N months. Set 0 to disable.")
    ap.add_argument("--refit_iters", type=int, default=200, help="(Walk-forward) # of additional training iterations at each retrain.")
    args = ap.parse_args()
    if args.walk_forward_mode is None:
        args.walk_forward_mode = "expanding" if args.expanding_window else "fixed"
    else:
        args.walk_forward_mode = _resolve_walk_forward_mode_name(args.walk_forward_mode)
        if args.expanding_window and args.walk_forward_mode != "expanding":
            raise ValueError("--expanding_window cannot be combined with walk_forward_mode other than 'expanding'.")
    args.expanding_window = str(args.walk_forward_mode).lower() == "expanding"
    mode = str(args.walk_forward_mode).upper()
    print(f"Walk-forward mode: {mode}")
    # Choose which state specs to run.
    # v42 default:
    #   - single-spec mode: pca_only_k2
    #   - compare mode: the 22-spec sweep plan (STATE_SPECS_V42)
    if args.compare_specs:
        specs = list(args.specs) if args.specs else list(STATE_SPECS_V42)
    else:
        specs = [args.state_spec] if args.state_spec is not None else ["pca_only_k2"]

    # validate
    for s in specs:
        if s not in STATE_SPECS:
            raise ValueError(f"Unknown spec '{s}'. Choose from {STATE_SPECS}.")

    # ------------------------------------------------------------------
    # Spec dependencies (which external series must be available?)
    # ------------------------------------------------------------------
    macro3_needed_for_state = any(spec_requires_macro3(s) for s in specs)
    macro7_needed_for_state = any(spec_requires_macro7(s) for s in specs)
    ff3_needed_for_state = any(spec_requires_ff3(s) for s in specs)
    ff5_needed_for_state = any(spec_requires_ff5(s) for s in specs)
    bond_needed_for_state = any(spec_requires_bond_panel(s) for s in specs)

    fred_key_provided = not (args.fred_api_key is None or str(args.fred_api_key).strip() == "")

    # v44 common-start policy: keep the train start date identical across specs when running the 22-spec sweep
    common_start_mode = str(getattr(args, "common_start_mode", "suite") or "suite")
    if common_start_mode not in ("macro3", "suite"):
        print(f"[WARN] Unknown --common_start_mode={common_start_mode}; defaulting to 'suite'.")
        common_start_mode = "suite"

    force_common = bool(getattr(args, "force_common_calendar", True))
    calendar_suite = bool(force_common and common_start_mode == "suite")
    calendar_macro3 = bool(force_common and common_start_mode == "macro3")

    # FRED key is required when:
    # - macro3_only/macro7_* specs are used, or
    # - bond is sourced from a FRED total-return series, or
    # - force_common_calendar is enabled in 'suite' mode (needs macro7 for the common start date).
    bond_requires_fred = bool(
        getattr(args, "include_bond", False)
        and str(getattr(args, "bond_source", "crsp_csv")) == "fred_tr"
        and (
            str(getattr(args, "bond_fred_specs", "") or "").strip() != ""
            or str(getattr(args, "bond_series_id", "BAMLCC0A0CMTRIV") or "").strip() != ""
        )
    )
    if bond_needed_for_state and (not bool(getattr(args, "include_bond", False))):
        raise RuntimeError(
            "The requested *_eqbond_block state spec requires at least one bond asset, but --no_bond was set."
        )
    fred_required = bool(macro3_needed_for_state or macro7_needed_for_state or bond_requires_fred or calendar_suite)
    if fred_required and not fred_key_provided:
        raise RuntimeError(
            "FRED API key is required when using macro3_only/macro7_* and/or --bond_source fred_tr, "
            "or when --force_common_calendar is enabled with --common_start_mode suite. "
            "Pass --fred_api_key ..."
        )

    # Common-calendar option (v44): enforce a single aligned start date across specs when desired.
    macro_for_calendar = bool(force_common and fred_key_provided)

    # In 'suite' mode we always load macro7 (calendar anchor); in 'macro3' mode we load macro3 unless macro7 is needed.
    want_macro7 = bool(macro7_needed_for_state or (calendar_suite and macro_for_calendar))
    want_macro3 = bool((macro3_needed_for_state or (calendar_macro3 and macro_for_calendar)) and (not want_macro7))
    fred_needed = bool(bond_requires_fred or want_macro3 or want_macro7)

    # In 'suite' mode, also align all runs on FF5 for a single start date (FF5 begins in 1963-07).
    ff5_for_calendar = bool(force_common and common_start_mode == "suite")

    print(f"State specs to run: {specs}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    print("Downloading Ken French F-F Research Factors (monthly)...")
    ff = _clean_returns_decimal(load_ff_factors_monthly())

    # rf and market from ff
    rf = ff["RF"].rename("rf")
    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("mkt")

    asset_universe = str(getattr(args, 'asset_universe', 'ff49ind'))
    if asset_universe == 'custom':
        print("Building custom risky-asset universe from config...")
        ind = _load_custom_risky_assets_from_args(args, ff=ff, mkt=mkt)
        universe_label = f"Custom risky-asset universe ({', '.join(ind.columns)})"
        print(f"Using {universe_label}.")
    else:
        universe_label = describe_asset_universe(asset_universe)
        print(f"Downloading {universe_label} (monthly)...")
        ind = _clean_returns_decimal(load_equity_universe_monthly(asset_universe))

    # Optional factor states
    ff3_state = None
    if ff3_needed_for_state:
        ff3_state = ff[["Mkt-RF", "SMB", "HML"]].copy()

    ff5_state = None
    if ff5_needed_for_state or ff5_for_calendar:
        msg = "Downloading Ken French F-F 5 Factors (monthly)..."
        if ff5_for_calendar and (not ff5_needed_for_state):
            msg += " [calendar-only]"
        print(msg)
        ff5_all = _clean_returns_decimal(load_ff5_factors_monthly())
        # Use only the risky FF5 factors as state (exclude RF).
        ff5_state = ff5_all[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].copy()
        # If we also need FF3, derive it from FF5 so calendars match.
        if ff3_needed_for_state:
            ff3_state = ff5_state[["Mkt-RF", "SMB", "HML"]].copy()

    # optional FRED pulls (macro predictors and/or bond series)
    mac_cfg = (
        FredMacroConfig(api_key=str(args.fred_api_key), end=str(args.end_date))
        if fred_needed
        else None
    )

    # optional macro predictors from FRED
    macro3 = None
    macro7 = None
    if want_macro7:
        if not fred_key_provided:
            # Should be unreachable because we error earlier when macro7 is required.
            raise RuntimeError("FRED API key is required for macro7 specs.")
        msg = (
            "Downloading FRED macro predictors (macro7: infl_yoy / term_spread / default_spread / "
            "ip_yoy / unrate / umcsent / mprime)..."
        )
        if calendar_suite and (not macro7_needed_for_state):
            msg += " [calendar-only]"
        print(msg)
        assert mac_cfg is not None
        macro7 = _clean_macro(build_macro7_monthly(mac_cfg))
        # Keep macro3 as a subset for macro3_only spec compatibility.
        macro3 = macro7[["infl_yoy", "term_spread", "default_spread"]].copy()
    elif want_macro3:
        if not fred_key_provided:
            print("[WARN] --force_common_calendar requested but no --fred_api_key provided; proceeding without macro3.")
        else:
            msg = "Downloading FRED macro predictors (infl_yoy / term_spread / default_spread)..."
            if macro_for_calendar and (not macro3_needed_for_state) and (not macro7_needed_for_state):
                msg += " [calendar-only]"
            print(msg)
            assert mac_cfg is not None
            macro3 = _clean_macro(build_macro3_monthly(mac_cfg))

    # optional bond asset(s) (added as extra risky assets)
    bond_asset_names: list[str] = []
    if getattr(args, "include_bond", False):
        bond_df = _load_bond_panel(args, mac_cfg)
        if bond_df.empty:
            raise RuntimeError("Bond loading returned an empty panel.")
        rename_map: dict[str, str] = {}
        for col in bond_df.columns:
            name = str(col)
            if name in ind.columns:
                rename_map[name] = f"{name}_BOND"
        if len(rename_map) > 0:
            bond_df = bond_df.rename(columns=rename_map)
        bond_asset_names = [str(c) for c in bond_df.columns]
        ind = pd.concat([ind, bond_df], axis=1)

    # Align all data monthly and drop NaNs jointly.
    #
    # NOTE:
    # - We *only* include optional blocks (macro/factors) when they are actually loaded.
    # - If ff5_state is present, we do not additionally align on ff3_state, to avoid
    #   duplicate columns (Mkt-RF/SMB/HML). In that case, ff3_state is derived from ff5_state.
    parts = [ind, rf.to_frame(), mkt.to_frame()]
    if macro7 is not None:
        parts.append(macro7)
    elif macro3 is not None:
        parts.append(macro3)

    if ff5_state is not None:
        parts.append(ff5_state)
    elif ff3_state is not None:
        parts.append(ff3_state)

    aligned = _align_common(*parts)
    ind = aligned[0]
    rf_df = aligned[1]
    mkt_df = aligned[2]

    cursor = 3
    if macro7 is not None:
        macro7 = aligned[cursor]
        cursor += 1
    elif macro3 is not None:
        macro3 = aligned[cursor]
        cursor += 1

    if ff5_state is not None:
        ff5_state = aligned[cursor]
        cursor += 1
    elif ff3_state is not None:
        ff3_state = aligned[cursor]
        cursor += 1

    rf_s = rf_df.iloc[:, 0].rename("rf")
    mkt_s = mkt_df.iloc[:, 0].rename("mkt")

    concat_list = [ind, rf_s, mkt_s]
    if macro7 is not None:
        concat_list.append(macro7)
    elif macro3 is not None:
        concat_list.append(macro3)
    if ff5_state is not None:
        concat_list.append(ff5_state)
    elif ff3_state is not None:
        concat_list.append(ff3_state)

    joint = pd.concat(concat_list, axis=1)
    joint = joint.replace([np.inf, -np.inf], np.nan)

    # Hard end-date (v43 default: 2024-12-31).
    # Normalize to calendar month-end to match the rest of the pipeline.
    end_ts = pd.Timestamp(str(getattr(args, "end_date", "2024-12-31"))) + pd.offsets.MonthEnd(0)
    joint = joint.loc[:end_ts]

    joint = joint.dropna()

    train_pool_start = getattr(args, "train_pool_start", None)
    if train_pool_start is not None and str(train_pool_start).strip() != "":
        start_ts = pd.Timestamp(str(train_pool_start)) + pd.offsets.MonthEnd(0)
        joint = joint.loc[start_ts:]
        if joint.empty:
            raise ValueError(f"No data left after applying --train_pool_start={start_ts.date()}.")
        print(
            f"[TRAIN POOL START] requested_start={start_ts.date()} | clipped_start={joint.index.min().date()} | T={len(joint)}"
        )

    if force_common:
        # This run is a single spec, but in v44 we enforce a common aligned start across specs by default.
        print(
            f"[COMMON START] mode={common_start_mode} | aligned_start={joint.index.min().date()} "
            f"| aligned_end={joint.index.max().date()} | T={len(joint)}"
        )

    ind = joint[ind.columns]
    rf = joint["rf"]
    mkt = joint["mkt"]

    if macro7 is not None:
        macro7 = joint[macro7.columns]
        # Ensure macro3 is always the macro7 subset when macro7 is present.
        macro3 = macro7[["infl_yoy", "term_spread", "default_spread"]].copy()
    elif macro3 is not None:
        macro3 = joint[macro3.columns]

    if ff5_state is not None:
        ff5_state = joint[ff5_state.columns]
        if ff3_needed_for_state:
            ff3_state = ff5_state[["Mkt-RF", "SMB", "HML"]].copy()
    elif ff3_state is not None:
        ff3_state = joint[ff3_state.columns]

    # Excess returns matrix
    r_ex = ind.sub(rf, axis=0)

    # Common arrays for fast slicing
    T = len(r_ex)
    r_ex_np = r_ex.values.astype(float)
    rf_np = rf.values.astype(float)
    mkt_np = mkt.values.astype(float)

    print(
        f"Dataset base (aligned): {T} obs | assets={r_ex.shape[1]} | "
        f"macro3={(0 if macro3 is None else macro3.shape[1])} "
        f"macro7={(0 if macro7 is None else macro7.shape[1])} "
        f"ff3={(0 if ff3_state is None else ff3_state.shape[1])} "
        f"ff5={(0 if ff5_state is None else ff5_state.shape[1])}"
    )
    print(f"Date range (aligned): {r_ex.index[0].date()} .. {r_ex.index[-1].date()}")
    _warn_if_missing_months(r_ex.index, label="aligned dataset")

    # ---------------------------------------------------------------------
    # Evaluation protocol:
    #   - default: last 10y OOS (120 months)
    #   - --eval_3x10y: 3 contiguous 10y blocks within the last 30y (separate re-fit per block)
    #   - --eval_30y: last 30y OOS as one block (Gu–Kelly–Xiu style)
    # Notes:
    #   - training horizon for model-based PG-DPO warm-up is controlled by --train_horizon (months)
    # ---------------------------------------------------------------------
    H10 = 120
    H30 = 360
    H_train_requested = int(getattr(args, "train_horizon", 0))
    if H_train_requested < 0:
        raise ValueError("--train_horizon must be >= 0 (months). Use 0 to auto-match the eval horizon.")

    idx = r_ex.index
    end_global = idx[-1]

    blocks: list[dict[str, int | str]] = []

    def _add_block(tag_prefix: str, test_start_date: pd.Timestamp, test_end_date: pd.Timestamp, *, expected_months: int):
        test_start_date = pd.Timestamp(test_start_date)
        test_end_date = pd.Timestamp(test_end_date)
        test_start_pos = _get_loc_strict(idx, test_start_date, what=f"{tag_prefix}.test_start")
        test_end_pos = _get_loc_strict(idx, test_end_date, what=f"{tag_prefix}.test_end")
        obs = int(test_end_pos - test_start_pos + 1)
        if obs != int(expected_months):
            print(
                f"[WARN] {tag_prefix}: expected {expected_months} monthly observations in TEST window "
                f"but found {obs}. (Data may have missing months.)"
            )

        train_end = int(test_start_pos - 1)
        if train_end <= 200:
            raise RuntimeError(
                f"Not enough data: train_end={train_end} (need >200). "
                f"Requested TEST window {test_start_date.date()}..{test_end_date.date()} ({expected_months}m)."
            )

        start_idx = train_end
        end_idx = int(test_end_pos)
        H_eval = int(end_idx - start_idx)
        tag = f"{tag_prefix}_{test_start_date.strftime('%Y%m')}_{test_end_date.strftime('%Y%m')}"
        blocks.append({"tag": tag, "train_end": train_end, "start_idx": start_idx, "end_idx": end_idx, "H_eval": H_eval})

    # ------------------------------------------------------------------
    # Evaluation plan
    # ------------------------------------------------------------------
    eval_mode = str(getattr(args, "eval_mode", "v53"))

    if eval_mode in ("v56", "v57"):
        # v56/v57: rolling time-series CV on the DEV period (everything before the final test20y),
        # then one final refit/evaluation on the full test20y.
        #
        # v57 is identical to v56 except it forces cv_folds=3 (coarser regime splits).
        #
        # DEV end is the month immediately preceding the test20y start.
        H20 = 240
        last20_end = end_global
        last20_start = last20_end - pd.DateOffset(months=int(H20 - 1))

        dev_end = last20_start - pd.DateOffset(months=1)
        dev_end_pos = _get_loc_strict(idx, dev_end, what=f"{eval_mode}.dev_end")
        dev_len = int(dev_end_pos + 1)

        K = 3 if eval_mode == "v57" else int(getattr(args, "cv_folds", 5))
        if K <= 0:
            raise ValueError(f"({eval_mode}) cv_folds must be positive, got {K}.")

        min_train_obs = int(getattr(args, "cv_min_train_months", 204))
        # _add_block enforces train_end index > 200, i.e. at least 202 obs in TRAIN.
        if min_train_obs < 202:
            print(
                f"[WARN] ({eval_mode}) cv_min_train_months={min_train_obs} is too small; bumping to 202 to satisfy train_end>200."
            )
            min_train_obs = 202
        if min_train_obs >= dev_len:
            raise RuntimeError(
                f"({eval_mode}) Not enough DEV data: cv_min_train_months={min_train_obs} >= dev_len={dev_len} (through {dev_end.date()})."
            )

        val_months_user = int(getattr(args, "cv_val_months", 0) or 0)
        if val_months_user > 0:
            val_months = int(val_months_user)
            needed = int(min_train_obs + val_months * K)
            if needed > dev_len:
                raise RuntimeError(
                    f"({eval_mode}) Requested cv_val_months={val_months} with cv_folds={K} and cv_min_train_months={min_train_obs} "
                    f"requires {needed} DEV months, but only {dev_len} are available (through {dev_end.date()})."
                )
            extra = int(dev_len - needed)
        else:
            val_months = int((dev_len - min_train_obs) // K)
            if val_months <= 0:
                raise RuntimeError(
                    f"({eval_mode}) Not enough DEV data for rolling CV: dev_len={dev_len}, min_train={min_train_obs}, cv_folds={K} -> fold_len={val_months}."
                )
            extra = int(dev_len - (min_train_obs + val_months * K))

        print(
            f"Evaluation plan ({eval_mode}): rolling CV folds on DEV (pre-test), then test20y (full last 20y)"
        )
        print(
            f"  DEV  : {idx[0].date()} .. {dev_end.date()}  (T_dev={dev_len} months)"
        )
        print(
            f"  CV   : folds={K} | min_train={min_train_obs}m | val_months={val_months}m | leftover={extra}m"
        )
        print(
            f"  TEST : {last20_start.date()} .. {last20_end.date()} (H=240 months)"
        )

        train_end_pos = int(min_train_obs - 1)
        for i in range(int(K)):
            test_start_pos = int(train_end_pos + 1)
            if val_months_user > 0:
                # Fixed-size validation windows; any leftover DEV months at the end remain unused for validation.
                test_end_pos = int(test_start_pos + val_months - 1)
            else:
                # Auto-sized windows; extend the last fold to reach dev_end.
                if i < int(K) - 1:
                    test_end_pos = int(test_start_pos + val_months - 1)
                else:
                    test_end_pos = int(dev_end_pos)

            if test_end_pos > int(dev_end_pos):
                test_end_pos = int(dev_end_pos)

            start_date = idx[int(test_start_pos)]
            end_date = idx[int(test_end_pos)]
            expected_months = int(test_end_pos - test_start_pos + 1)

            _add_block(f"valfold{i + 1}", start_date, end_date, expected_months=expected_months)
            train_end_pos = int(test_end_pos)

        _add_block("test20y", last20_start, last20_end, expected_months=int(H20))

    elif eval_mode == "v53":
        # v53: internal validation to avoid test leakage.
        #   - val10y: last 10 years of the *pre-test* training period
        #   - test20y: full last 20 years (final OOS)
        H20 = 240
        last20_end = end_global
        last20_start = last20_end - pd.DateOffset(months=int(H20 - 1))

        # Validation is carved out from the pre-test period.
        # We define it as the 10y window immediately preceding the test start.
        val_end = last20_start - pd.DateOffset(months=1)
        val_start = val_end - pd.DateOffset(months=int(H10 - 1))

        print(
            "Evaluation plan (v53): val10y (last 10y of pre-test) and test20y (full last 20y)"
        )
        _add_block("val10y", val_start, val_end, expected_months=int(H10))
        _add_block("test20y", last20_start, last20_end, expected_months=int(H20))

    elif eval_mode == "v43":
        # v43 default: exactly three windows ending at --end_date:
        #   - block1: first 10y within last20y
        #   - block2: last 10y within last20y
        #   - last20y: full last 20y
        H20 = 240
        last20_end = end_global
        last20_start = last20_end - pd.DateOffset(months=int(H20 - 1))

        block2_end = last20_end
        block2_start = block2_end - pd.DateOffset(months=int(H10 - 1))

        block1_end = block2_start - pd.DateOffset(months=1)
        block1_start = block1_end - pd.DateOffset(months=int(H10 - 1))

        print(
            "Evaluation plan (v43): block1 (first 10y of last20y), block2 (last 10y), and last20y"
        )

        _add_block("block1", block1_start, block1_end, expected_months=int(H10))
        _add_block("block2", block2_start, block2_end, expected_months=int(H10))
        _add_block("last20y", last20_start, last20_end, expected_months=int(H20))

    elif eval_mode == "legacy":
        # ------------------------
        # Legacy evaluation protocol (can be combined)
        # ------------------------
        eval_horizons: list[int] | None = getattr(args, "eval_horizons", None)
        if eval_horizons is not None:
            eval_horizons = [int(x) for x in list(eval_horizons)]
            if len(eval_horizons) == 0:
                eval_horizons = None

        horizons: list[int] = []
        if getattr(args, "eval_30y", False):
            horizons.append(30)
        if eval_horizons is not None:
            horizons.extend(eval_horizons)

        # Default behavior: if no horizons specified and not using 3x10y, run the last 10y.
        if not getattr(args, "eval_3x10y", False) and len(horizons) == 0:
            horizons = [10]

        # De-duplicate horizons (keep order)
        _seen = set()
        horizons = [h for h in horizons if (h not in _seen and not _seen.add(h))]

        if getattr(args, "eval_3x10y", False):
            print("Evaluation plan: 3x10y blocks enabled")
        if len(horizons) > 0:
            print(f"Evaluation plan: horizon blocks enabled -> years={horizons}")

        # 1) 3×10y (contiguous) blocks over the last 30 years.
        if getattr(args, "eval_3x10y", False):
            n_blocks = 3
            for bi in range(n_blocks):
                # block1 is the earliest, block3 is the latest
                months_back_end = (n_blocks - 1 - bi) * H10
                block_end = end_global - pd.DateOffset(months=int(months_back_end))
                block_start = block_end - pd.DateOffset(months=int(H10 - 1))
                _add_block(f"block{bi+1}", block_start, block_end, expected_months=int(H10))

        # 2) Single-window evaluations (e.g., last10y/last20y/last30y). Can be combined with 3×10y.
        for y in horizons:
            if int(y) <= 0:
                raise ValueError(f"Invalid horizon in --eval_horizons: {y} (must be positive years).")
            months = int(y) * 12
            block_end = end_global
            block_start = block_end - pd.DateOffset(months=int(months - 1))
            _add_block(f"last{int(y)}y", block_start, block_end, expected_months=int(months))
    else:
        raise ValueError(
            f"Unknown --eval_mode: {eval_mode} (expected 'v57', 'v56', 'v53', 'v43' or 'legacy')"
        )

    # ---------------------------------------------------------------------
    # Run selected state specs (PCA-only vs Macro-only etc.) for each block
    # ---------------------------------------------------------------------
    pca_cfg = LatentPCAConfig(n_components=int(args.latent_k), standardize=True, random_state=int(args.seed))

    selection_requested = bool(getattr(args, "select_specs_first", False) or getattr(args, "selection_only", False))
    if selection_requested:
        sel_cfg = SpecSelectionConfig(
            window_mode=str(getattr(args, "selection_window_mode", "rolling")),
            rolling_window=int(getattr(args, "selection_rolling_window", 60)),
            return_baseline=str(getattr(args, "selection_return_baseline", "expanding_mean")),
            state_baseline=str(getattr(args, "selection_state_baseline", "expanding_mean")),
            alpha=float(getattr(args, "selection_alpha", 0.25)),
            beta=float(getattr(args, "selection_beta", 0.50)),
            gamma=float(getattr(args, "selection_gamma", 0.25)),
            return_floor=float(getattr(args, "selection_return_floor", 0.0)),
            state_q10_floor=float(getattr(args, "selection_state_q10_floor", -0.05)),
            cross_warn=float(getattr(args, "selection_cross_warn", 0.95)),
            cross_fail=float(getattr(args, "selection_cross_fail", 0.98)),
            score_mode=str(getattr(args, "selection_score_mode", "method_shortlist")),
            score_ret_mean_weight=float(getattr(args, "selection_score_ret_mean_weight", 0.45)),
            score_ret_q10_weight=float(getattr(args, "selection_score_ret_q10_weight", 0.20)),
            score_state_mean_weight=float(getattr(args, "selection_score_state_mean_weight", 0.20)),
            score_state_q10_weight=float(getattr(args, "selection_score_state_q10_weight", 0.15)),
        )
        selection_blocks = [blk for blk in blocks if str(blk["tag"]).lower().startswith("val")]
        if len(selection_blocks) == 0 and len(blocks) > 0:
            selection_blocks = [blocks[0]]

        print("\n" + "=" * 80)
        print("SPEC SELECTION (method-free; predictive R^2 / stability / cross guard)")
        print("=" * 80)
        print(f"Selection blocks: {[str(b['tag']) for b in selection_blocks]}")
        print(f"Candidate specs : {specs}")

        sel_results = []
        for blk in selection_blocks:
            for spec in specs:
                try:
                    sel_results.append(
                        evaluate_spec_predictive_diagnostics(
                            spec=str(spec),
                            block=str(blk["tag"]),
                            r_ex=r_ex,
                            macro3=macro3,
                            macro7=macro7,
                            ff3=ff3_state,
                            ff5=ff5_state,
                            bond_asset_names=bond_asset_names,
                            train_end=int(blk["train_end"]),
                            start_idx=int(blk["start_idx"]),
                            end_idx=int(blk["end_idx"]),
                            pca_cfg=pca_cfg,
                            pls_horizon=int(args.pls_horizon),
                            pls_smooth_span=int(args.pls_smooth_span),
                            block_eq_k=int(getattr(args, "block_eq_k", 1)),
                            block_bond_k=int(getattr(args, "block_bond_k", 1)),
                            config=sel_cfg,
                        )
                    )
                except Exception as e:
                    print(f"[WARN] Spec-selection failed for spec={spec} block={blk['tag']}: {e}")

        df_sel = results_to_frame(sel_results)
        df_rank = rank_specs_from_results(df_sel, config=sel_cfg)

        out_dir_root = Path(args.out_dir)
        out_dir_root.mkdir(parents=True, exist_ok=True)
        sel_path = out_dir_root / "spec_selection_blocks.csv"
        rank_path = out_dir_root / "spec_selection_summary.csv"
        if not df_sel.empty:
            df_sel.to_csv(sel_path, index=False)
            print(f"Saved: {sel_path}")
        if not df_rank.empty:
            df_rank.to_csv(rank_path, index=False)
            print(f"Saved: {rank_path}")
            show_cols = [
                c for c in [
                    "spec",
                    "score",
                    "passes_guard",
                    "recommended",
                    "r2_oos_ret_mean",
                    "r2_roll_ret_q10",
                    "r2_oos_state_mean",
                    "r2_roll_state_q10",
                    "cross_max_abs_rho",
                ] if c in df_rank.columns
            ]
            print("\nSpec ranking summary:")
            with pd.option_context("display.max_rows", None, "display.width", 160, "display.max_colwidth", 40):
                print(df_rank[show_cols].to_string(index=False))
        else:
            print("[WARN] Spec-selection did not produce any valid rows; continuing with the original spec list.")

        if bool(getattr(args, "selection_only", False)):
            print("Selection-only mode complete. Exiting before policy training.")
            return

        top_k = int(getattr(args, "selection_top_k", 0) or 0)
        if top_k > 0 and not df_rank.empty:
            if bool(getattr(args, "selection_guarded_only", False)) and "passes_guard" in df_rank.columns:
                df_pick = df_rank[df_rank["passes_guard"]].copy()
                if df_pick.empty:
                    print("[WARN] No spec passed the selection guard rails; falling back to the full score ranking.")
                    df_pick = df_rank.copy()
            else:
                df_pick = df_rank.copy()
            specs = df_pick["spec"].head(top_k).astype(str).tolist()
            print(f"[SPEC SELECT] keeping top-{len(specs)} specs for training/evaluation (score-ranked): {specs}")

    out_dir_root = Path(args.out_dir)
    out_dir_root.mkdir(parents=True, exist_ok=True)

    multi_blocks = len(blocks) > 1

    all_results: List[Dict[str, Any]] = []
    # Baselines depend only on the TEST window (not on the state_spec),
    # so we compute them once per block and keep them for summary printing/CSV.
    block_baselines: Dict[str, Dict[str, Dict[str, float]]] = {}
    for blk in blocks:
        block_tag = str(blk["tag"])
        train_end = int(blk["train_end"])
        start_idx = int(blk["start_idx"])
        end_idx = int(blk["end_idx"])
        H_eval = int(blk["H_eval"])

        # Per-block training horizon (months) for model-based PG-DPO warm-up.
        #
        # Requested horizon:
        #   - --train_horizon = 0  => auto-match the block test horizon (H_eval)
        #   - --train_horizon > 0  => use that as an upper bound
        #
        # Effective horizon:
        #   - H_train_blk <= H_eval (don't train longer than we evaluate)
        #   - We *may* need to cap further when exog>0 (needs an observed z-path slice)
        train_horizon_req = int(getattr(args, "train_horizon", 0))
        if train_horizon_req <= 0:
            train_horizon_req = int(H_eval)
        H_train_blk = int(min(int(train_horizon_req), int(H_eval)))

        # NOTE: In the discrete simulator, the only reason to require
        # `start_max = train_end - H_train_blk` to be positive is when we need to
        # slice an observed exogenous path z_{t:t+H} from the data (exog>0).
        # For the current specs (pca_only / macro3_only / pls_only), exog=0 and
        # we can safely train with H_train_blk > train_obs by sampling initial
        # states from TRAIN and simulating forward from the fitted model.

        # explicit calendar ranges
        train_start_date = r_ex.index[0]
        train_end_date = r_ex.index[train_end]
        test_start_date = r_ex.index[start_idx + 1]
        test_end_date = r_ex.index[end_idx]

        out_dir = (out_dir_root / block_tag) if multi_blocks else out_dir_root
        out_dir.mkdir(parents=True, exist_ok=True)

        is_val_block = str(block_tag).lower().startswith("val")
        eval_label = "VAL" if is_val_block else "TEST"

        print("\n" + "=" * 80)
        print(
            f"EVAL {block_tag} | "
            f"TRAIN {train_start_date.date()} .. {train_end_date.date()} ({train_end+1} obs) | "
            f"{eval_label} {test_start_date.date()} .. {test_end_date.date()} ({H_eval} obs) | "
            f"H_train={H_train_blk}"
        )
        print("=" * 80)

        # -----------------------------------------------------------------
        # Baselines on realized evaluation window (shared across specs)
        # -----------------------------------------------------------------
        rf_test = rf_np[start_idx + 1 : start_idx + 1 + H_eval]

        # Market baseline (benchmark)
        mkt_rets = mkt_np[start_idx + 1 : start_idx + 1 + H_eval]
        mkt_metrics = compute_metrics(mkt_rets, rf=rf_test, periods_per_year=12, gamma=float(args.gamma))
        print(f"\nBaseline metrics (realized {eval_label}):")
        print("Baseline market (MKT):")
        for k, v in mkt_metrics.items():
            print(f"  {k:7s}: {v: .4f}")

        # Equal-weight across risky assets (cash 0)
        eq_ex = np.mean(r_ex_np[start_idx + 1 : start_idx + 1 + H_eval, :], axis=1)
        eq_rets = rf_test + eq_ex
        eq_metrics = compute_metrics(eq_rets, rf=rf_test, periods_per_year=12, gamma=float(args.gamma))
        print("Baseline equal-weight:")
        for k, v in eq_metrics.items():
            print(f"  {k:7s}: {v: .4f}")

        # Extra lightweight benchmarks (TRAIN-based)
        #
        # FIXED mode: constant weights estimated on TRAIN.
        # EXPANDING mode: re-estimate weights each month on an expanding window up to t.
        bench_list = list(getattr(args, "benchmarks", ["gmv", "risk_parity", "inv_vol"]))
        bench_w_max = getattr(args, "bench_w_max", None)
        rp_max_iter = int(getattr(args, "rp_max_iter", 500))
        rp_tol = float(getattr(args, "rp_tol", 1e-8))

        bench_logs: dict[str, dict] = {}
        if len(bench_list) > 0:
            if walk_mode in {"expanding", "rolling"}:
                # Walk-forward baselines: weights refit each month using realized returns up to t.
                bench_refit_every = int(getattr(args, "bench_refit_every", 1))
                bench_logs = simulate_realized_expanding_benchmarks_from_returns(
                    r_ex=r_ex_np,
                    rf=rf_np,
                    start_idx=start_idx,
                    H=H_eval,
                    bench_list=bench_list,
                    bench_w_max=bench_w_max,
                    rp_max_iter=rp_max_iter,
                    rp_tol=rp_tol,
                    refit_every=bench_refit_every,
                    gamma=float(args.gamma),
                    window_mode=walk_mode,
                    rolling_train_months=getattr(args, "rolling_train_months", None),
                )
            else:
                # Fixed baselines: constant weights from TRAIN.
                train_r_ex = r_ex_np[1 : train_end + 1, :]
                lw = LedoitWolf().fit(train_r_ex)
                Sigma_train = lw.covariance_
                mu_train = np.nanmean(train_r_ex, axis=0)

                if "gmv" in bench_list:
                    w_gmv, _ = gmv_weights_long_only_cash(Sigma_train, ridge=1e-6, w_max=bench_w_max)
                    bench_logs["gmv"] = simulate_realized_fixed_constant_weights(
                        w_gmv, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                    )

                if "risk_parity" in bench_list:
                    w_rp, _ = risk_parity_weights_long_only_cash(
                        Sigma_train, max_iter=rp_max_iter, tol=rp_tol, ridge=1e-8, w_max=bench_w_max
                    )
                    bench_logs["risk_parity"] = simulate_realized_fixed_constant_weights(
                        w_rp, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                    )

                if "inv_vol" in bench_list:
                    w_iv, _ = inv_vol_weights_long_only_cash(Sigma_train, w_max=bench_w_max)
                    bench_logs["inv_vol"] = simulate_realized_fixed_constant_weights(
                        w_iv, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                    )

                if "static_mvo" in bench_list:
                    w_smvo, _ = markowitz_weights_long_only_cash(
                        mu_train, Sigma_train, risk_aversion=float(args.gamma), ridge=1e-6, w_max=bench_w_max
                    )
                    bench_logs["static_mvo"] = simulate_realized_fixed_constant_weights(
                        w_smvo, r_ex=r_ex_np, rf=rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                    )

        if bench_logs:
            if walk_mode == "expanding":
                print("\nExtra baselines (TRAIN-based; refit on expanding window each month):")
            elif walk_mode == "rolling":
                print("\nExtra baselines (TRAIN-based; refit on rolling window each month):")
            else:
                print("\nExtra baselines (constant weights from TRAIN):")
            for name, logs_b in bench_logs.items():
                print(f"Baseline {name}:")
                for k, v in logs_b["metrics"].items():
                    print(f"  {k:7s}: {v: .4f}")

        # Build a baseline DataFrame to be merged into each spec output.
        dates = r_ex.index[start_idx + 1 : start_idx + 1 + H_eval]
        baseline_df = pd.DataFrame(index=dates)
        baseline_df["rf"] = rf_test

        baseline_df["mkt_ret"] = mkt_rets
        baseline_df["mkt_wealth"] = np.cumprod(1.0 + mkt_rets)

        baseline_df["eq_ret"] = eq_rets
        baseline_df["eq_wealth"] = np.cumprod(1.0 + eq_rets)

        # attach extra benchmark series
        if "gmv" in bench_logs:
            baseline_df["gmv_ret"] = bench_logs["gmv"]["port_rets"]
            baseline_df["gmv_wealth"] = bench_logs["gmv"]["wealth"][1:]
            baseline_df["gmv_turnover"] = bench_logs["gmv"]["turnover"]
            baseline_df["gmv_cash"] = bench_logs["gmv"]["cash"]
        if "risk_parity" in bench_logs:
            baseline_df["rp_ret"] = bench_logs["risk_parity"]["port_rets"]
            baseline_df["rp_wealth"] = bench_logs["risk_parity"]["wealth"][1:]
            baseline_df["rp_turnover"] = bench_logs["risk_parity"]["turnover"]
            baseline_df["rp_cash"] = bench_logs["risk_parity"]["cash"]
        if "inv_vol" in bench_logs:
            baseline_df["inv_vol_ret"] = bench_logs["inv_vol"]["port_rets"]
            baseline_df["inv_vol_wealth"] = bench_logs["inv_vol"]["wealth"][1:]
            baseline_df["inv_vol_turnover"] = bench_logs["inv_vol"]["turnover"]
            baseline_df["inv_vol_cash"] = bench_logs["inv_vol"]["cash"]
        if "static_mvo" in bench_logs:
            baseline_df["static_mvo_ret"] = bench_logs["static_mvo"]["port_rets"]
            baseline_df["static_mvo_wealth"] = bench_logs["static_mvo"]["wealth"][1:]
            baseline_df["static_mvo_turnover"] = bench_logs["static_mvo"]["turnover"]
            baseline_df["static_mvo_cash"] = bench_logs["static_mvo"]["cash"]

        baseline_path = out_dir / f"baselines_{block_tag}.csv"
        baseline_df.to_csv(baseline_path)
        print(f"Saved: {baseline_path}")

        # Collect baseline metrics for summary output (same shape as other strategies).
        baseline_metrics: dict[str, dict] = {"mkt": mkt_metrics, "eq": eq_metrics}
        for _name, _obj in bench_logs.items():
            baseline_metrics[_name] = _obj["metrics"]

        # Keep for end-of-run summary (baselines are independent of state_spec).
        block_baselines[block_tag] = baseline_metrics

        # Run specs for this block
        for spec in specs:
            res = run_one_state_spec(
                spec=spec,
                r_ex=r_ex,
                rf=rf,
                macro3=macro3,
                macro7=macro7,
                ff3=ff3_state,
                ff5=ff5_state,
                bond_asset_names=bond_asset_names,
                train_end=train_end,
                start_idx=start_idx,
                H_train=H_train_blk,
                H_eval=H_eval,
                r_ex_np=r_ex_np,
                rf_np=rf_np,
                baseline_df=baseline_df,
                args=args,
                device=device,
                dtype=dtype,
                pca_cfg=pca_cfg,
                out_dir=out_dir,
                eval_label=eval_label,
            )
            res["block"] = block_tag
            res["eval_label"] = eval_label
            res["test_start"] = str(test_start_date.date())
            res["test_end"] = str(test_end_date.date())

            # Attach baselines for consistent summary/CSV across blocks
            res.update(baseline_metrics)
            all_results.append(res)
    if len(all_results) >= 1:
        print("\n================ SUMMARY (realized; validation/test blocks) ================")
        for res in all_results:
            sp = res["spec"]
            blk = res.get("block", "")
            pm = res["policy"]
            mm = res["myopic"]
            pp = res.get("ppgdpo", None)

            cer_pol = pm.get("cer_ann", np.nan)
            cer_mvo = mm.get("cer_ann", np.nan)
            cal_pol = pm.get("calmar", np.nan)
            cal_mvo = mm.get("calmar", np.nan)
            ddur_pol = pm.get("max_dd_dur", np.nan)
            ddur_mvo = mm.get("max_dd_dur", np.nan)
            sor_pol = pm.get("sortino", np.nan)
            sor_mvo = mm.get("sortino", np.nan)

            print(
                f"[{blk} | {sp}] Policy  sharpe={pm['sharpe']:.4f}  sortino={sor_pol:.4f}  "
                f"ann_ret={pm['ann_ret']:.4f}  ann_vol={pm['ann_vol']:.4f}  cer={cer_pol:.4f}  "
                f"max_dd={pm['max_dd']:.4f}  calmar={cal_pol:.4f}  dd_dur={ddur_pol:.0f}  final={pm['final_wealth']:.4f}"
            )
            print(
                f"               Myopic  sharpe={mm['sharpe']:.4f}  sortino={sor_mvo:.4f}  "
                f"ann_ret={mm['ann_ret']:.4f}  ann_vol={mm['ann_vol']:.4f}  cer={cer_mvo:.4f}  "
                f"max_dd={mm['max_dd']:.4f}  calmar={cal_mvo:.4f}  dd_dur={ddur_mvo:.0f}  final={mm['final_wealth']:.4f}"
            )

            if isinstance(pp, dict) and "sharpe" in pp:
                cer_pp = pp.get("cer_ann", np.nan)
                cal_pp = pp.get("calmar", np.nan)
                ddur_pp = pp.get("max_dd_dur", np.nan)
                sor_pp = pp.get("sortino", np.nan)
                print(
                    f"               PPGDPO  sharpe={pp['sharpe']:.4f}  sortino={sor_pp:.4f}  "
                    f"ann_ret={pp['ann_ret']:.4f}  ann_vol={pp['ann_vol']:.4f}  cer={cer_pp:.4f}  "
                    f"max_dd={pp['max_dd']:.4f}  calmar={cal_pp:.4f}  dd_dur={ddur_pp:.0f}  final={pp['final_wealth']:.4f}"
                )
            pp_vars = res.get("ppgdpo_variants", {})
            if isinstance(pp_vars, dict):
                for _name, _met in pp_vars.items():
                    if _name == "ppgdpo" or not isinstance(_met, dict) or ("sharpe" not in _met):
                        continue
                    cer_v = _met.get("cer_ann", np.nan)
                    cal_v = _met.get("calmar", np.nan)
                    ddur_v = _met.get("max_dd_dur", np.nan)
                    sor_v = _met.get("sortino", np.nan)
                    print(
                        f"      {_name:>14s}  sharpe={_met['sharpe']:.4f}  sortino={sor_v:.4f}  "
                        f"ann_ret={_met['ann_ret']:.4f}  ann_vol={_met['ann_vol']:.4f}  cer={cer_v:.4f}  "
                        f"max_dd={_met['max_dd']:.4f}  calmar={cal_v:.4f}  dd_dur={ddur_v:.0f}  final={_met['final_wealth']:.4f}"
                    )

        if getattr(args, "eval_3x10y", False):
            print("\n================ AVERAGE ACROSS 3 BLOCKS (per spec) ================")
            for spec in specs:
                # Only average the 3 non-overlapping 10y blocks (block1/2/3).
                # When running combined evaluation (e.g., 3x10y blocks + last10/20/30),
                # we do NOT want to pool those different test windows.
                rows = [
                    r
                    for r in all_results
                    if (r.get("spec") == spec) and str(r.get("block", "")).startswith("block")
                ]
                if not rows:
                    continue
                pol_sh = float(np.mean([r["policy"]["sharpe"] for r in rows]))
                pol_ret = float(np.mean([r["policy"]["ann_ret"] for r in rows]))
                pol_cer = float(np.mean([r["policy"].get("cer_ann", np.nan) for r in rows]))
                mvo_sh = float(np.mean([r["myopic"]["sharpe"] for r in rows]))
                mvo_ret = float(np.mean([r["myopic"]["ann_ret"] for r in rows]))
                mvo_cer = float(np.mean([r["myopic"].get("cer_ann", np.nan) for r in rows]))

                pp_rows = [r for r in rows if isinstance(r.get("ppgdpo", None), dict) and "sharpe" in r["ppgdpo"]]
                has_pp = len(pp_rows) > 0
                if has_pp:
                    pp_sh = float(np.mean([r["ppgdpo"]["sharpe"] for r in pp_rows]))
                    pp_ret = float(np.mean([r["ppgdpo"]["ann_ret"] for r in pp_rows]))
                    pp_cer = float(np.mean([r["ppgdpo"].get("cer_ann", np.nan) for r in pp_rows]))

                _rows_cross = {str(r.get("cross_mode", "estimated")).lower() for r in rows}
                _rows_proxy = any(bool(r.get("zero_cross_policy_proxy", False)) for r in rows)
                _avg_pol_label = "Dynamic(zero)" if (_rows_cross == {"zero"} and not _rows_proxy) else ("PolicyProxy(zero=myopic)" if _rows_proxy else "Full(est)")
                print(
                    f"[{spec}] {_avg_pol_label} avg sharpe={pol_sh:.4f}  avg ann_ret={pol_ret:.4f}  avg cer={pol_cer:.4f}"
                )
                print(
                    f"      Myopic avg sharpe={mvo_sh:.4f}  avg ann_ret={mvo_ret:.4f}  avg cer={mvo_cer:.4f}"
                )

                if has_pp:
                    print(
                        f"      PPGDPO avg sharpe={pp_sh:.4f}  avg ann_ret={pp_ret:.4f}  avg cer={pp_cer:.4f}"
                    )
                pp_variant_names = sorted({
                    _name
                    for r in rows
                    for _name in (r.get("ppgdpo_variants", {}) or {}).keys()
                    if _name != "ppgdpo"
                })
                for _name in pp_variant_names:
                    _vrows = [r for r in rows if isinstance(r.get("ppgdpo_variants", {}).get(_name), dict)]
                    if not _vrows:
                        continue
                    _sh = float(np.mean([r["ppgdpo_variants"][_name]["sharpe"] for r in _vrows]))
                    _ret = float(np.mean([r["ppgdpo_variants"][_name]["ann_ret"] for r in _vrows]))
                    _cer = float(np.mean([r["ppgdpo_variants"][_name].get("cer_ann", np.nan) for r in _vrows]))
                    print(
                        f"      {_name} avg sharpe={_sh:.4f}  avg ann_ret={_ret:.4f}  avg cer={_cer:.4f}"
                    )

    # -------------------------------------------------------------
    # Save block-level summary CSV (no concatenation)
    # -------------------------------------------------------------
    if all_results:
        mode = _resolve_walk_forward_mode_name(getattr(args, "walk_forward_mode", "expanding" if args.expanding_window else "fixed"))
        rows_out: list[dict] = []
        # Baselines are identical across state specs for a given block.
        # When running multiple specs (compare_specs), avoid duplicating baseline rows.
        seen_baselines: set[tuple[str, str]] = set()
        baseline_order = ["mkt", "eq", "gmv", "risk_parity", "inv_vol"]
        for r in all_results:
            base = {
                "spec": r.get("spec", ""),
                "mode": r.get("mode", ""),
                "block": r.get("block", r.get("tag", "")),
                "eval_label": r.get("eval_label", ""),
                "test_start": r.get("test_start", ""),
                "test_end": r.get("test_end", ""),
                "cross_mode": r.get("cross_mode", ""),
                "policy_source": r.get("policy_source", "trained_policy"),
                "zero_cross_policy_proxy": r.get("zero_cross_policy_proxy", False),
            }
            for strategy_name, key in [
                ("policy", "policy"),
                ("myopic", "myopic"),
                ("ppgdpo", "ppgdpo"),
                ("gmv_model", "gmv_model"),
                ("risk_parity_model", "risk_parity_model"),
                ("inv_vol_model", "inv_vol_model"),
                ("static_mvo_model", "static_mvo_model"),
            ]:
                met = r.get(key)
                if not isinstance(met, dict):
                    continue
                rows_out.append(
                    {
                        **base,
                        "strategy": strategy_name,
                        "ann_ret": met.get("ann_ret", np.nan),
                        "ann_vol": met.get("ann_vol", np.nan),
                        "sharpe": met.get("sharpe", np.nan),
                        "sortino": met.get("sortino", np.nan),
                        "es95": met.get("es95", np.nan),
                        "max_dd": met.get("max_dd", np.nan),
                        "calmar": met.get("calmar", np.nan),
                        "max_dd_dur": met.get("max_dd_dur", np.nan),
                        "underwater_frac": met.get("underwater_frac", np.nan),
                        "final_wealth": met.get("final_wealth", np.nan),
                        "cer_ann": met.get("cer_ann", np.nan),
                    }
                )

            pp_vars = res.get("ppgdpo_variants", {})
            if isinstance(pp_vars, dict):
                for _name, _met in pp_vars.items():
                    if _name == "ppgdpo" or not isinstance(_met, dict):
                        continue
                    rows_out.append(
                        {
                            **base,
                            "strategy": _name,
                            "ann_ret": _met.get("ann_ret", np.nan),
                            "ann_vol": _met.get("ann_vol", np.nan),
                            "sharpe": _met.get("sharpe", np.nan),
                            "sortino": _met.get("sortino", np.nan),
                            "es95": _met.get("es95", np.nan),
                            "max_dd": _met.get("max_dd", np.nan),
                            "calmar": _met.get("calmar", np.nan),
                            "max_dd_dur": _met.get("max_dd_dur", np.nan),
                            "underwater_frac": _met.get("underwater_frac", np.nan),
                            "final_wealth": _met.get("final_wealth", np.nan),
                            "cer_ann": _met.get("cer_ann", np.nan),
                            "cross_scale": _met.get("cross_scale", np.nan),
                            "avg_hedge_term_l1": _met.get("avg_hedge_term_l1", np.nan),
                            "avg_mu_term_l1": _met.get("avg_mu_term_l1", np.nan),
                            "hedge_to_mu_ratio": _met.get("hedge_to_mu_ratio", np.nan),
                            "variant_label": _met.get("label", _name),
                        }
                    )

            # Add baseline strategies once per block.
            # These are already computed and stored in the per-block runner output (keys: mkt/eq/...).
            block_id = str(base.get("block", ""))
            base_baseline = {**base, "spec": "__baseline__"}
            for bname in ["mkt", "eq", "gmv", "risk_parity", "inv_vol", "static_mvo"]:
                met_b = r.get(bname)
                if not isinstance(met_b, dict):
                    continue
                key_b = (block_id, bname)
                if key_b in seen_baselines:
                    continue
                seen_baselines.add(key_b)
                rows_out.append(
                    {
                        **base_baseline,
                        "strategy": bname,
                        "ann_ret": met_b.get("ann_ret", np.nan),
                        "ann_vol": met_b.get("ann_vol", np.nan),
                        "sharpe": met_b.get("sharpe", np.nan),
                        "sortino": met_b.get("sortino", np.nan),
                        "es95": met_b.get("es95", np.nan),
                        "max_dd": met_b.get("max_dd", np.nan),
                        "calmar": met_b.get("calmar", np.nan),
                        "max_dd_dur": met_b.get("max_dd_dur", np.nan),
                        "underwater_frac": met_b.get("underwater_frac", np.nan),
                        "final_wealth": met_b.get("final_wealth", np.nan),
                        "cer_ann": met_b.get("cer_ann", np.nan),
                    }
                )

        if rows_out:
            df_sum = pd.DataFrame(rows_out)
            out_path = out_dir_root / f"summary_blocks_{mode}.csv"
            df_sum.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

            # v56/v57: convenience summaries for rolling CV folds (for later spec selection)
            if eval_mode in ("v56", "v57"):
                try:
                    df_folds = df_sum[
                        df_sum["block"].astype(str).str.startswith("valfold")
                    ].copy()
                    if not df_folds.empty:
                        out_folds = out_dir_root / f"summary_cv_folds_{mode}.csv"
                        df_folds.to_csv(out_folds, index=False)
                        print(f"Saved: {out_folds}")

                        metric_cols = [
                            c
                            for c in [
                                "ann_ret",
                                "ann_vol",
                                "sharpe",
                                "sortino",
                                "es95",
                                "max_dd",
                                "calmar",
                                "max_dd_dur",
                                "underwater_frac",
                                "final_wealth",
                                "cer_ann",
                            ]
                            if c in df_folds.columns
                        ]
                        g = df_folds.groupby(["spec", "strategy"], dropna=False)
                        df_counts = g.size().reset_index(name="n_folds")
                        df_agg = g.agg(
                            {c: ["mean", "median", "min", "max", "std"] for c in metric_cols}
                        )
                        df_agg.columns = [
                            f"{m}_{stat}" for (m, stat) in df_agg.columns
                        ]
                        df_agg = df_agg.reset_index()
                        df_agg = df_agg.merge(
                            df_counts, on=["spec", "strategy"], how="left"
                        )
                        out_agg = out_dir_root / f"summary_cv_agg_{mode}.csv"
                        df_agg.to_csv(out_agg, index=False)
                        print(f"Saved: {out_agg}")
                    else:
                        print(
                            "[WARN] (v56) No valfold blocks found in summary; CV helper CSVs not written."
                        )
                except Exception as e:
                    print(f"[WARN] (v56) Failed to write CV helper summaries: {e}")

if __name__ == "__main__":
    main()
