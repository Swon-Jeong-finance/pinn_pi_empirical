from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import DataConfig


@dataclass
class Dataset:
    returns: pd.DataFrame
    states: pd.DataFrame
    factors: pd.DataFrame | None


def _read_panel(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f'{date_col=} not found in {path}')
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def _simulate(periods: int, assets: int, factors: int, seed: int = 17):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2000-01-31', periods=periods, freq='ME')

    slow_value = np.zeros(periods)
    fast_vol = np.zeros(periods)
    curve_slope = np.zeros(periods)
    for t in range(1, periods):
        slow_value[t] = 0.97 * slow_value[t - 1] + 0.08 * rng.normal()
        fast_vol[t] = 0.70 * fast_vol[t - 1] + 0.30 * rng.normal()
        curve_slope[t] = 0.92 * curve_slope[t - 1] + 0.10 * rng.normal()

    states = pd.DataFrame(
        {
            'slow_value': slow_value,
            'fast_vol': fast_vol,
            'curve_slope': curve_slope,
        },
        index=dates,
    )

    mkt = np.zeros(periods)
    value = np.zeros(periods)
    bond = np.zeros(periods)
    corr = np.array([[1.0, 0.25, -0.20], [0.25, 1.0, -0.10], [-0.20, -0.10, 1.0]])
    chol = np.linalg.cholesky(corr)
    for t in range(1, periods):
        h1 = np.exp(-2.9 + 0.80 * np.log(mkt[t - 1] ** 2 + 1e-6) + 0.50 * fast_vol[t - 1])
        h2 = np.exp(-3.1 + 0.65 * np.log(value[t - 1] ** 2 + 1e-6) + 0.35 * slow_value[t - 1])
        h3 = np.exp(-4.0 + 0.55 * np.log(bond[t - 1] ** 2 + 1e-6) - 0.25 * fast_vol[t - 1] + 0.25 * curve_slope[t - 1])
        mean = np.array(
            [
                0.004 + 0.002 * slow_value[t - 1] - 0.001 * fast_vol[t - 1],
                0.002 + 0.0025 * slow_value[t - 1],
                0.001 + 0.001 * curve_slope[t - 1] - 0.0005 * fast_vol[t - 1],
            ]
        )
        z = chol @ rng.normal(size=3)
        shock = np.sqrt(np.maximum([h1, h2, h3], 1e-6)) * z
        mkt[t], value[t], bond[t] = mean + shock

    factor_cols = ['MKT', 'VALUE', 'BOND'][:factors]
    factor_mat = np.column_stack([mkt, value, bond])[:, :factors]
    factors_df = pd.DataFrame(factor_mat, index=dates, columns=factor_cols)

    loadings = rng.normal(size=(assets, factors))
    loadings[:, 0] += rng.uniform(0.6, 1.4, size=assets)
    alpha_state = rng.normal(scale=0.001, size=(assets, 3))
    idio_vol = rng.uniform(0.01, 0.03, size=assets)
    state_mat = states[['slow_value', 'fast_vol', 'curve_slope']].to_numpy(dtype=float)
    rets = 0.001 + state_mat @ alpha_state.T + factor_mat @ loadings.T + rng.normal(
        scale=idio_vol, size=(periods, assets)
    )
    ret_cols = [f'asset_{i + 1:02d}' for i in range(assets)]
    returns_df = pd.DataFrame(rets, index=dates, columns=ret_cols)
    return returns_df, states, factors_df


def load_dataset(cfg: DataConfig) -> Dataset:
    if cfg.mode == 'csv':
        returns = _read_panel(Path(cfg.returns_csv), cfg.date_col)
        states = (
            _read_panel(Path(cfg.states_csv), cfg.date_col)
            if cfg.states_csv
            else pd.DataFrame(index=returns.index)
        )
        factors = _read_panel(Path(cfg.factors_csv), cfg.date_col) if cfg.factors_csv else None
        common = returns.index
        common = common.intersection(states.index) if not states.empty else common
        if factors is not None:
            common = common.intersection(factors.index)
        returns = returns.loc[common]
        states = states.loc[common] if not states.empty else pd.DataFrame(index=common)
        factors = factors.loc[common] if factors is not None else None
        return Dataset(returns=returns, states=states, factors=factors)

    returns, states, factors = _simulate(
        periods=cfg.synthetic.periods,
        assets=cfg.synthetic.assets,
        factors=cfg.synthetic.factors,
        seed=cfg.synthetic.seed,
    )
    return Dataset(returns=returns, states=states, factors=factors)
