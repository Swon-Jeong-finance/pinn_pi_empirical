from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def simulate(periods: int, assets: int, factors: int, seed: int = 17):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2000-01-31', periods=periods, freq='M')

    slow_value = np.zeros(periods)
    fast_vol = np.zeros(periods)
    curve_slope = np.zeros(periods)
    for t in range(1, periods):
        slow_value[t] = 0.97 * slow_value[t-1] + 0.08 * rng.normal()
        fast_vol[t] = 0.70 * fast_vol[t-1] + 0.30 * rng.normal()
        curve_slope[t] = 0.92 * curve_slope[t-1] + 0.10 * rng.normal()

    states = pd.DataFrame({
        'date': dates,
        'slow_value': slow_value,
        'fast_vol': fast_vol,
        'curve_slope': curve_slope,
    })

    mkt = np.zeros(periods)
    value = np.zeros(periods)
    bond = np.zeros(periods)
    h1, h2, h3 = 0.04, 0.03, 0.015
    corr = np.array([[1.0, 0.25, -0.20], [0.25, 1.0, -0.10], [-0.20, -0.10, 1.0]])
    chol = np.linalg.cholesky(corr)

    for t in range(1, periods):
        s = np.array([slow_value[t-1], fast_vol[t-1], curve_slope[t-1]])
        h1 = np.exp(-2.9 + 0.80 * np.log(mkt[t-1] ** 2 + 1e-6) + 0.50 * fast_vol[t-1])
        h2 = np.exp(-3.1 + 0.65 * np.log(value[t-1] ** 2 + 1e-6) + 0.35 * slow_value[t-1])
        h3 = np.exp(-4.0 + 0.55 * np.log(bond[t-1] ** 2 + 1e-6) - 0.25 * fast_vol[t-1] + 0.25 * curve_slope[t-1])
        z = chol @ rng.normal(size=3)
        vols = np.sqrt(np.maximum([h1, h2, h3], 1e-6))
        mean = np.array([
            0.004 + 0.002 * slow_value[t-1] - 0.001 * fast_vol[t-1],
            0.002 + 0.0025 * slow_value[t-1],
            0.001 + 0.001 * curve_slope[t-1] - 0.0005 * fast_vol[t-1],
        ])
        shock = vols * z
        mkt[t], value[t], bond[t] = mean + shock

    factors_df = pd.DataFrame({'date': dates, 'MKT': mkt, 'VALUE': value, 'BOND': bond})

    loadings = rng.normal(size=(assets, factors))
    loadings[:, 0] += rng.uniform(0.6, 1.4, size=assets)
    loadings[:, 2] += rng.uniform(-0.3, 0.3, size=assets)
    alpha_state = rng.normal(scale=0.001, size=(assets, 3))
    idio_vol = rng.uniform(0.01, 0.03, size=assets)

    rets = np.zeros((periods, assets))
    factor_mat = np.column_stack([mkt, value, bond])
    state_mat = np.column_stack([slow_value, fast_vol, curve_slope])
    for t in range(periods):
        rets[t] = 0.001 + state_mat[t] @ alpha_state.T + factor_mat[t] @ loadings.T + rng.normal(scale=idio_vol, size=assets)

    ret_cols = [f'asset_{i+1:02d}' for i in range(assets)]
    returns_df = pd.DataFrame(rets, columns=ret_cols)
    returns_df.insert(0, 'date', dates)

    return returns_df, states, factors_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=Path('demo_data'))
    ap.add_argument('--periods', type=int, default=240)
    ap.add_argument('--assets', type=int, default=12)
    ap.add_argument('--factors', type=int, default=3)
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    returns_df, states_df, factors_df = simulate(args.periods, args.assets, args.factors, args.seed)
    returns_df.to_csv(out / 'returns_panel.csv', index=False)
    states_df.to_csv(out / 'states_panel.csv', index=False)
    factors_df.to_csv(out / 'factors_panel.csv', index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
