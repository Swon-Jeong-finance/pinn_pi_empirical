from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_capped_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    if z <= 0:
        return np.zeros_like(v)
    u = np.maximum(v, 0.0)
    s = float(u.sum())
    if s <= z:
        return u
    x = np.sort(u)[::-1]
    cssv = np.cumsum(x) - z
    ind = np.arange(1, len(x) + 1)
    cond = x - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(u - theta, 0.0)


def annualized_return(r: pd.Series) -> float:
    if r.empty:
        return float('nan')
    wealth = float(np.prod(1.0 + r.values))
    years = len(r) / 12.0
    if years <= 0:
        return float('nan')
    return wealth ** (1.0 / years) - 1.0


def annualized_vol(r: pd.Series) -> float:
    if len(r) < 2:
        return float('nan')
    return float(r.std(ddof=1) * np.sqrt(12.0))


def sharpe_ratio(r: pd.Series) -> float:
    vol = annualized_vol(r)
    if not np.isfinite(vol) or vol <= 0:
        return float('nan')
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12.0))


def certainty_equivalent_annual(r: pd.Series, gamma: float) -> float:
    if r.empty:
        return float('nan')
    mu_m = float(r.mean())
    var_m = float(r.var(ddof=1)) if len(r) > 1 else 0.0
    return 12.0 * (mu_m - 0.5 * gamma * var_m)


def max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float('nan')
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())
