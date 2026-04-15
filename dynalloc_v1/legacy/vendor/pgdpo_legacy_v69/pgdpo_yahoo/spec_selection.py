"""Method-independent state-spec diagnostics and ranking.

This module adds a *spec-first* workflow on top of the v58 empirical code:
- no portfolio method is trained here;
- specs are ranked by predictive R^2 and stability only;
- cross (Cov(eps,u)) is treated as a diagnostic / risk flag, not as part of
  the ranking signal itself unless it becomes dangerously large.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from .discrete_latent_model import LatentPCAConfig, DiscreteLatentMarketModel, _gvarx_phi_numpy
from .state_specs import build_model_for_state_spec


@dataclass
class SpecSelectionConfig:
    window_mode: str = "rolling"              # rolling | expanding
    rolling_window: int = 60
    return_baseline: str = "expanding_mean"   # train_mean | expanding_mean
    state_baseline: str = "expanding_mean"    # train_mean | expanding_mean | random_walk

    # Legacy value-score weights (kept for score_mode='legacy_value').
    alpha: float = 0.25
    beta: float = 0.50
    gamma: float = 0.25

    # Guard rails. In v62 these are intentionally looser and are treated as
    # filters / warnings rather than as the primary ranking signal.
    return_floor: float = -0.02
    state_q10_floor: float = -0.05
    cross_warn: float = 0.95
    cross_fail: float = 0.98

    # v62 shortlist score: a rank-based aggregation that emphasizes return-side
    # usefulness while keeping state quality as a secondary criterion.
    score_mode: str = "method_shortlist"  # method_shortlist | legacy_value
    score_ret_mean_weight: float = 0.45
    score_ret_q10_weight: float = 0.20
    score_state_mean_weight: float = 0.20
    score_state_q10_weight: float = 0.15


@dataclass
class SpecSelectionResult:
    spec: str
    block: str
    train_end: int
    start_idx: int
    end_idx: int
    horizon: int
    n_assets: int
    n_states: int
    r2_is_ret_mean: float
    r2_is_ret_median: float
    r2_oos_ret_mean: float
    r2_oos_ret_median: float
    r2_roll_ret_q10: float
    r2_roll_ret_min: float
    r2_is_state_mean: float
    r2_is_state_median: float
    r2_oos_state_mean: float
    r2_oos_state_median: float
    r2_roll_state_q10: float
    r2_roll_state_min: float
    cross_mean_abs_rho: float
    cross_p95_abs_rho: float
    cross_max_abs_rho: float


@dataclass
class RankedSpec:
    spec: str
    n_blocks: int
    r2_is_ret_mean: float
    r2_oos_ret_mean: float
    r2_oos_ret_median: float
    r2_roll_ret_q10: float
    r2_is_state_mean: float
    r2_oos_state_mean: float
    r2_oos_state_median: float
    r2_roll_state_q10: float
    cross_mean_abs_rho: float
    cross_max_abs_rho: float
    fail_return: bool
    fail_state: bool
    fail_cross: bool
    warn_cross: bool
    passes_guard: bool
    score_ret_mean_rank: float
    score_ret_q10_rank: float
    score_state_mean_rank: float
    score_state_q10_rank: float
    score: float
    recommended: bool


def cross_rho_stats(
    Sigma: np.ndarray,
    Q: np.ndarray,
    Cross: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    Cross = np.asarray(Cross, dtype=float)
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
    return stats, rho


def _r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray, y_bench: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_bench = np.asarray(y_bench, dtype=float)
    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    sse_b = np.sum((y_true - y_bench) ** 2, axis=0)
    return 1.0 - (sse / np.clip(sse_b, 1e-16, None))


def _ols_style_r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    y_bar = np.mean(y_true, axis=0, keepdims=True)
    sst = np.sum((y_true - y_bar) ** 2, axis=0)
    sse = np.sum(resid ** 2, axis=0)
    return 1.0 - (sse / np.clip(sst, 1e-16, None))


def _predict_returns(model: DiscreteLatentMarketModel, y_hist: np.ndarray) -> np.ndarray:
    return np.asarray(model.a, dtype=float)[None, :] + np.asarray(y_hist, dtype=float) @ np.asarray(model.B, dtype=float).T


def _predict_states(
    model: DiscreteLatentMarketModel,
    y_hist: np.ndarray,
    z_hist: Optional[np.ndarray],
) -> np.ndarray:
    y_hist = np.asarray(y_hist, dtype=float)
    if model.trans_beta is not None and model.trans_gvarx_cfg is not None:
        Phi, _names, _ym, _ys, _zm, _zs = _gvarx_phi_numpy(
            y_hist,
            z_hist,
            model.trans_gvarx_cfg,
            y_mean=model.trans_y_mean,
            y_std=model.trans_y_std,
            z_mean=model.trans_z_mean,
            z_std=model.trans_z_std,
            fit_stats=False,
        )
        return Phi @ np.asarray(model.trans_beta, dtype=float)

    pred = np.asarray(model.c, dtype=float)[None, :] + y_hist @ np.asarray(model.A, dtype=float).T
    if z_hist is not None and np.asarray(model.G).size > 0:
        pred = pred + np.asarray(z_hist, dtype=float) @ np.asarray(model.G, dtype=float).T
    return pred


def _make_oos_baseline(
    target_all: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    mode: str,
    random_walk_source: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct benchmark predictions for targets at indices start_idx+1..end_idx.

    target_all is indexed on the *calendar* scale (same as r_ex or y_df). For time t
    we predict target_{t+1} using information through t.
    """
    target_all = np.asarray(target_all, dtype=float)
    preds = []
    mode = str(mode).lower()
    for t in range(int(start_idx), int(end_idx)):
        if mode == "train_mean":
            hist = target_all[1 : start_idx + 1, :]
            base = np.mean(hist, axis=0)
        elif mode == "expanding_mean":
            hist = target_all[1 : t + 1, :]
            base = np.mean(hist, axis=0)
        elif mode == "random_walk":
            if random_walk_source is None:
                raise ValueError("random_walk baseline requires random_walk_source")
            base = np.asarray(random_walk_source[t, :], dtype=float)
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")
        preds.append(base)
    return np.asarray(preds, dtype=float)


def _window_r2_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_bench: np.ndarray,
    *,
    window: int,
    mode: str = "rolling",
) -> tuple[float, float]:
    T = int(y_true.shape[0])
    if T <= 0:
        return float("nan"), float("nan")

    mode = str(mode).lower()
    if mode not in {"rolling", "expanding"}:
        raise ValueError(f"Unknown selection window mode: {mode}")

    if window <= 1 or T <= window:
        r2 = _r2_per_dim(y_true, y_pred, y_bench)
        val = float(np.mean(r2))
        return val, val

    vals: list[float] = []
    if mode == "rolling":
        for s in range(0, T - window + 1):
            e = s + window
            r2w = _r2_per_dim(y_true[s:e], y_pred[s:e], y_bench[s:e])
            vals.append(float(np.mean(r2w)))
    else:
        min_window = int(max(window, 1))
        for e in range(min_window, T + 1):
            r2w = _r2_per_dim(y_true[:e], y_pred[:e], y_bench[:e])
            vals.append(float(np.mean(r2w)))
    arr = np.asarray(vals, dtype=float)
    return float(np.quantile(arr, 0.10)), float(np.min(arr))


def evaluate_spec_predictive_diagnostics(
    *,
    spec: str,
    block: str,
    r_ex: pd.DataFrame,
    macro3: Optional[pd.DataFrame],
    macro7: Optional[pd.DataFrame],
    ff3: Optional[pd.DataFrame],
    ff5: Optional[pd.DataFrame],
    bond_asset_names: Optional[list[str]],
    train_end: int,
    start_idx: int,
    end_idx: int,
    pca_cfg: LatentPCAConfig,
    pls_horizon: int,
    pls_smooth_span: int,
    block_eq_k: int,
    block_bond_k: int,
    config: SpecSelectionConfig,
) -> SpecSelectionResult:
    model, y_df, z_df = build_model_for_state_spec(
        spec,
        r_ex=r_ex,
        macro3=macro3,
        macro7=macro7,
        ff3=ff3,
        ff5=ff5,
        train_end=train_end,
        pca_cfg=pca_cfg,
        pls_horizon=pls_horizon,
        pls_smooth_span=pls_smooth_span,
        bond_asset_names=bond_asset_names,
        block_eq_k=block_eq_k,
        block_bond_k=block_bond_k,
    )

    y_all = y_df.values.astype(float)
    z_all = None if z_df is None else z_df.values.astype(float)
    rx_all = r_ex.values.astype(float)

    # In-sample diagnostics
    rx_true_is = rx_all[1 : train_end + 1, :]
    rx_pred_is = _predict_returns(model, y_all[:train_end, :])
    r2_is_ret = _ols_style_r2_per_dim(rx_true_is, rx_pred_is)

    y_true_is = y_all[1 : train_end + 1, :]
    z_hist_is = None if z_all is None else z_all[:train_end, :]
    y_pred_is = _predict_states(model, y_all[:train_end, :], z_hist_is)
    r2_is_state = _ols_style_r2_per_dim(y_true_is, y_pred_is)

    # OOS diagnostics
    rx_true_oos = rx_all[start_idx + 1 : end_idx + 1, :]
    rx_pred_oos = _predict_returns(model, y_all[start_idx:end_idx, :])
    rx_bench_oos = _make_oos_baseline(
        rx_all,
        start_idx=start_idx,
        end_idx=end_idx,
        mode=config.return_baseline,
    )
    r2_oos_ret = _r2_per_dim(rx_true_oos, rx_pred_oos, rx_bench_oos)
    r2_roll_ret_q10, r2_roll_ret_min = _window_r2_summary(
        rx_true_oos,
        rx_pred_oos,
        rx_bench_oos,
        window=int(config.rolling_window),
        mode=str(getattr(config, "window_mode", "rolling")),
    )

    y_true_oos = y_all[start_idx + 1 : end_idx + 1, :]
    z_hist_oos = None if z_all is None else z_all[start_idx:end_idx, :]
    y_pred_oos = _predict_states(model, y_all[start_idx:end_idx, :], z_hist_oos)
    y_bench_oos = _make_oos_baseline(
        y_all,
        start_idx=start_idx,
        end_idx=end_idx,
        mode=config.state_baseline,
        random_walk_source=y_all if str(config.state_baseline).lower() == "random_walk" else None,
    )
    r2_oos_state = _r2_per_dim(y_true_oos, y_pred_oos, y_bench_oos)
    r2_roll_state_q10, r2_roll_state_min = _window_r2_summary(
        y_true_oos,
        y_pred_oos,
        y_bench_oos,
        window=int(config.rolling_window),
        mode=str(getattr(config, "window_mode", "rolling")),
    )

    cross_stats, _rho = cross_rho_stats(model.Sigma, model.Q, model.Cross)

    return SpecSelectionResult(
        spec=str(spec),
        block=str(block),
        train_end=int(train_end),
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        horizon=int(end_idx - start_idx),
        n_assets=int(rx_all.shape[1]),
        n_states=int(y_all.shape[1]),
        r2_is_ret_mean=float(np.mean(r2_is_ret)),
        r2_is_ret_median=float(np.median(r2_is_ret)),
        r2_oos_ret_mean=float(np.mean(r2_oos_ret)),
        r2_oos_ret_median=float(np.median(r2_oos_ret)),
        r2_roll_ret_q10=float(r2_roll_ret_q10),
        r2_roll_ret_min=float(r2_roll_ret_min),
        r2_is_state_mean=float(np.mean(r2_is_state)),
        r2_is_state_median=float(np.median(r2_is_state)),
        r2_oos_state_mean=float(np.mean(r2_oos_state)),
        r2_oos_state_median=float(np.median(r2_oos_state)),
        r2_roll_state_q10=float(r2_roll_state_q10),
        r2_roll_state_min=float(r2_roll_state_min),
        cross_mean_abs_rho=float(cross_stats["mean_abs"]),
        cross_p95_abs_rho=float(cross_stats["p95_abs"]),
        cross_max_abs_rho=float(cross_stats["max_abs"]),
    )


def results_to_frame(results: list[SpecSelectionResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def rank_specs_from_results(
    results: list[SpecSelectionResult] | pd.DataFrame,
    *,
    config: SpecSelectionConfig,
) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        df = results.copy()
    else:
        df = results_to_frame(results)
    if df.empty:
        return pd.DataFrame(columns=[f.name for f in RankedSpec.__dataclass_fields__.values()])

    agg_cols = [
        "r2_is_ret_mean",
        "r2_oos_ret_mean",
        "r2_oos_ret_median",
        "r2_roll_ret_q10",
        "r2_is_state_mean",
        "r2_oos_state_mean",
        "r2_oos_state_median",
        "r2_roll_state_q10",
        "cross_mean_abs_rho",
        "cross_max_abs_rho",
    ]
    g = df.groupby("spec", dropna=False)
    out = g[agg_cols].mean().reset_index()
    out["n_blocks"] = g.size().values

    out["fail_return"] = out["r2_oos_ret_mean"] <= float(config.return_floor)
    out["fail_state"] = out["r2_roll_state_q10"] <= float(config.state_q10_floor)
    out["fail_cross"] = out["cross_max_abs_rho"] >= float(config.cross_fail)
    out["warn_cross"] = out["cross_max_abs_rho"] >= float(config.cross_warn)
    out["passes_guard"] = ~(out["fail_return"] | out["fail_state"] | out["fail_cross"])

    mode = str(getattr(config, "score_mode", "method_shortlist")).lower()
    if mode == "legacy_value":
        instab_r = np.maximum(0.0, -out["r2_roll_ret_q10"].to_numpy(dtype=float))
        instab_y = np.maximum(0.0, -out["r2_roll_state_q10"].to_numpy(dtype=float))
        out["score_ret_mean_rank"] = np.nan
        out["score_ret_q10_rank"] = np.nan
        out["score_state_mean_rank"] = np.nan
        out["score_state_q10_rank"] = np.nan
        out["score"] = (
            out["r2_oos_ret_mean"].to_numpy(dtype=float)
            + float(config.alpha) * out["r2_oos_state_mean"].to_numpy(dtype=float)
            - float(config.beta) * instab_r
            - float(config.gamma) * instab_y
        )
    elif mode == "method_shortlist":
        out["score_ret_mean_rank"] = out["r2_oos_ret_mean"].rank(pct=True, method="average").astype(float)
        out["score_ret_q10_rank"] = out["r2_roll_ret_q10"].rank(pct=True, method="average").astype(float)
        out["score_state_mean_rank"] = out["r2_oos_state_mean"].rank(pct=True, method="average").astype(float)
        out["score_state_q10_rank"] = out["r2_roll_state_q10"].rank(pct=True, method="average").astype(float)
        out["score"] = (
            float(config.score_ret_mean_weight) * out["score_ret_mean_rank"].to_numpy(dtype=float)
            + float(config.score_ret_q10_weight) * out["score_ret_q10_rank"].to_numpy(dtype=float)
            + float(config.score_state_mean_weight) * out["score_state_mean_rank"].to_numpy(dtype=float)
            + float(config.score_state_q10_weight) * out["score_state_q10_rank"].to_numpy(dtype=float)
        )
    else:
        raise ValueError(f"Unknown score_mode={config.score_mode!r}")

    out["recommended"] = out["passes_guard"]

    out = out.sort_values(
        by=["score", "r2_oos_ret_mean", "r2_roll_ret_q10", "r2_oos_state_mean", "r2_roll_state_q10"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return out
