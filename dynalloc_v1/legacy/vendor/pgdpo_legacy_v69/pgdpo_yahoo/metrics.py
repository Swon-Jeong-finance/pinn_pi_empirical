"""pgdpo_yahoo.metrics

Simple performance metrics for a backtest equity curve.

v32 additions
------------
* **Certainty-equivalent return (CER)** under CRRA preferences (commonly used
  in empirical portfolio choice as an economic-value metric).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_gross(returns: "np.ndarray | pd.Series") -> np.ndarray:
    """Convert returns to gross returns, guarding against invalid values."""
    r = np.asarray(returns, dtype=float)
    g = 1.0 + r
    # Guard: utility uses log / power of gross; keep strictly positive.
    return np.clip(g, 1e-12, np.inf)


def certainty_equivalent_return(
    returns: "np.ndarray | pd.Series",
    gamma: float,
) -> float:
    """Certainty-equivalent *per-period* return under CRRA.

    Definition (monthly if ``returns`` are monthly):
      CER = CE_gross - 1

    where
      CE_gross = exp(E[log(1+R)])                 if gamma == 1
      CE_gross = (E[(1+R)^(1-gamma)])^(1/(1-gamma)) otherwise

    Notes
    -----
    * This is the standard out-of-sample CER used in many empirical
      portfolio-choice papers (economic value vs Sharpe).
    * Input ``returns`` should be *total* returns (already net-of-cost if you
      apply transaction costs).
    """
    g = _safe_gross(returns)
    if g.size == 0:
        return float("nan")

    if abs(float(gamma) - 1.0) < 1e-12:
        return float(np.exp(np.mean(np.log(g))) - 1.0)

    p = 1.0 - float(gamma)
    m = float(np.mean(g ** p))
    if not np.isfinite(m) or m <= 0.0:
        return float("nan")
    return float(m ** (1.0 / p) - 1.0)


def annualized_certainty_equivalent_return(
    returns: "np.ndarray | pd.Series",
    gamma: float,
    periods_per_year: int,
) -> float:
    """Annualized CER implied by per-period CER."""
    cer = certainty_equivalent_return(returns, gamma=gamma)
    if not np.isfinite(cer):
        return float("nan")
    return float((1.0 + cer) ** float(periods_per_year) - 1.0)


def annualized_mean_return(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized *arithmetic* mean return.

    Notes
    -----
    Many finance reports call this "annualized return" because it matches the
    numerator convention used by the Sharpe ratio (mean × periods_per_year).

    We still report the compounded/geometric annual return separately as ``cagr``.
    """
    r = returns.dropna().values
    if len(r) == 0:
        return float("nan")
    return float(np.mean(r) * periods_per_year)


def annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    """Compounded/geometric annual return (CAGR)."""
    r = returns.dropna().values
    if len(r) == 0:
        return float("nan")
    gross = np.prod(1.0 + r)
    years = len(r) / periods_per_year
    return gross ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna().values
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf: pd.Series, periods_per_year: int) -> float:
    x = (returns - rf).dropna().values
    if len(x) < 2:
        return float("nan")
    mu = np.mean(x) * periods_per_year
    vol = np.std(x, ddof=1) * np.sqrt(periods_per_year)
    return float(mu / (vol + 1e-12))




def sortino_ratio(
    returns: pd.Series,
    mar: "float | pd.Series" = 0.0,
    periods_per_year: int = 12,
) -> float:
    """Sortino ratio using downside deviation.

    Convention
    ----------
    * ``mar`` is the minimum acceptable return (per-period, same frequency as ``returns``).
    * Numerator is annualized mean of (R - MAR).
    * Denominator is annualized downside deviation:
        sqrt(E[min(0, R-MAR)^2]) * sqrt(periods_per_year)

    Notes
    -----
    For our monthly backtests we use MAR=0 by default (so the key is directly
    comparable across runs). If you prefer MAR=Rf or a target return, pass it
    explicitly.
    """
    x = returns.copy()
    if isinstance(mar, pd.Series):
        x = (returns - mar).dropna().values
    else:
        x = (returns - float(mar)).dropna().values
    if len(x) < 1:
        return float("nan")
    mu = float(np.mean(x) * periods_per_year)
    downside = np.minimum(0.0, x)
    dd = float(np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year))
    return float(mu / (dd + 1e-12))


def expected_shortfall(
    returns: pd.Series,
    alpha: float = 0.95,
) -> float:
    """Expected shortfall (CVaR) of returns.

    We report ES as the *average of the worst (1-alpha) returns*.
    For monthly data, ES95 corresponds to the mean of the worst 5% months.

    Returns
    -------
    float
        Typically negative (loss). Comparable in sign convention to max_dd.
    """
    r = returns.dropna().values
    if len(r) == 0:
        return float("nan")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0,1)")
    q = float(np.quantile(r, 1.0 - a))
    tail = r[r <= q]
    if len(tail) == 0:
        return float(q)
    return float(np.mean(tail))


def drawdown_stats(wealth: pd.Series) -> dict:
    """Drawdown path statistics.

    Returns
    -------
    dict with keys:
      * max_dd_dur: maximum drawdown duration (number of periods underwater)
      * underwater_frac: fraction of periods underwater (DD < 0)

    Notes
    -----
    We treat a period as underwater when wealth is strictly below its running peak.
    """
    w = wealth.dropna().values
    if len(w) == 0:
        return {"max_dd_dur": float("nan"), "underwater_frac": float("nan")}

    peak = np.maximum.accumulate(w)
    underwater = w < (peak * (1.0 - 1e-15))

    # fraction of time underwater
    uw_frac = float(np.mean(underwater))

    # max consecutive underwater run length
    max_run = 0
    run = 0
    for u in underwater:
        if u:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0

    return {"max_dd_dur": float(max_run), "underwater_frac": uw_frac}
def max_drawdown(wealth: pd.Series) -> float:
    w = wealth.dropna().values
    if len(w) == 0:
        return float("nan")
    peak = np.maximum.accumulate(w)
    dd = (w / peak) - 1.0
    return float(dd.min())


def summarize(backtest_df: pd.DataFrame, periods_per_year: int) -> dict:
    rets = backtest_df["port_ret"]
    rf = backtest_df["rf"]
    wealth = backtest_df["wealth"]

    max_dd_val = max_drawdown(wealth)
    dd_stats = drawdown_stats(wealth)

    # Return conventions:
    # - ann_ret: arithmetic annualized mean return (mean × periods_per_year)
    # - cagr   : compounded/geometric annual return (CAGR)
    cagr = annualized_return(rets, periods_per_year)
    ann_ret = annualized_mean_return(rets, periods_per_year)
    calmar = float(cagr / (abs(max_dd_val) + 1e-12))

    return {
        "ann_ret": ann_ret,
        "cagr": cagr,
        "ann_vol": annualized_vol(rets, periods_per_year),
        "sharpe": sharpe_ratio(rets, rf, periods_per_year),
        "sortino": sortino_ratio(rets, mar=0.0, periods_per_year=periods_per_year),
        "es95": expected_shortfall(rets, alpha=0.95),
        "max_dd": max_dd_val,
        "calmar": calmar,
        "max_dd_dur": dd_stats["max_dd_dur"],
        "underwater_frac": dd_stats["underwater_frac"],
        "avg_turnover": float(backtest_df["turnover"].mean()),
        "avg_tc": float(backtest_df["tc"].mean()),
        "final_wealth": float(wealth.iloc[-1]) if len(wealth) else float("nan"),
    }


def compute_metrics(
    returns: "np.ndarray | pd.Series",
    rf: "float | np.ndarray | pd.Series | None" = None,
    periods_per_year: int = 12,
    gamma: float | None = None,
) -> dict:
    """Convenience wrapper used by some scripts.

    Parameters
    ----------
    returns:
        1D total-return series (decimal). If you want Sharpe on excess returns,
        pass ``rf``.
    rf:
        Risk-free return series (decimal) aligned with ``returns`` or a scalar.
        If None, Sharpe is computed against 0.
    periods_per_year:
        12 for monthly, 52 for weekly, 252 for daily.
    """
    r = pd.Series(np.asarray(returns, dtype=float))
    if rf is None:
        rf_s = pd.Series(np.zeros_like(r.values))
    else:
        if np.isscalar(rf):
            rf_s = pd.Series(np.full_like(r.values, float(rf)))
        else:
            rf_s = pd.Series(np.asarray(rf, dtype=float))

    # Equity curve including the initial wealth = 1.0 (important for correct MDD).
    wealth = pd.Series(np.concatenate([[1.0], (1.0 + r.fillna(0.0)).cumprod().values]))

    max_dd_val = max_drawdown(wealth)
    dd_stats = drawdown_stats(wealth)

    cagr = annualized_return(r, periods_per_year)
    ann_ret = annualized_mean_return(r, periods_per_year)
    calmar = float(cagr / (abs(max_dd_val) + 1e-12))

    out = {
        "ann_ret": ann_ret,
        "cagr": cagr,
        "ann_vol": annualized_vol(r, periods_per_year),
        "sharpe": sharpe_ratio(r, rf_s, periods_per_year),
        "sortino": sortino_ratio(r, mar=0.0, periods_per_year=periods_per_year),
        "es95": expected_shortfall(r, alpha=0.95),
        "max_dd": max_dd_val,
        "calmar": calmar,
        "max_dd_dur": dd_stats["max_dd_dur"],
        "underwater_frac": dd_stats["underwater_frac"],
        "final_wealth": float(wealth.iloc[-1]) if len(wealth) else float("nan"),
    }

    # Optional economic-value metric (CRRA certainty equivalent).
    if gamma is not None:
        out["cer_ann"] = annualized_certainty_equivalent_return(r.values, gamma=float(gamma), periods_per_year=periods_per_year)

    return out
