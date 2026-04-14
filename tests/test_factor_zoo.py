from __future__ import annotations

import numpy as np
import pandas as pd

from dynalloc_v2.factor_zoo import (
    FactorZooCandidate,
    build_candidate_registry,
    _fit_pls_predictors_to_future_avg_returns,
    _fit_pls_returns_to_future_avg_returns,
)


def _synthetic_panels(periods: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    dates = pd.date_range('2000-01-31', periods=periods, freq='ME')
    base = rng.normal(scale=0.03, size=periods)
    returns = pd.DataFrame(
        {
            f'A{i+1}': 0.4 * base + rng.normal(scale=0.02, size=periods) + 0.01 * (i + 1)
            for i in range(5)
        },
        index=dates,
    )
    macro = pd.DataFrame(
        {
            'infl_yoy': 0.02 + 0.1 * base + rng.normal(scale=0.01, size=periods),
            'term_spread': 0.01 - 0.05 * base + rng.normal(scale=0.01, size=periods),
        },
        index=dates,
    )
    ff3 = pd.DataFrame(
        {
            'Mkt-RF': base,
            'SMB': 0.3 * base + rng.normal(scale=0.01, size=periods),
            'HML': -0.2 * base + rng.normal(scale=0.01, size=periods),
        },
        index=dates,
    )
    ff5 = ff3.assign(
        RMW=0.1 * base + rng.normal(scale=0.01, size=periods),
        CMA=-0.1 * base + rng.normal(scale=0.01, size=periods),
    )
    bond = pd.DataFrame(
        {
            'UST2Y': -0.1 * base + rng.normal(scale=0.01, size=periods),
            'UST10Y': -0.2 * base + rng.normal(scale=0.01, size=periods),
        },
        index=dates,
    )
    return returns, macro, ff3, ff5, bond


def test_pls_only_registry_restores_v1_pure_candidates() -> None:
    registry = build_candidate_registry('pls_only')
    names = [cand.name for cand in registry]
    assert names == [
        'pls_H6_k2',
        'pls_H6_k3',
        'pls_H12_k2',
        'pls_H12_k3',
        'pls_H24_k2',
        'pls_H24_k3',
    ]
    assert all(cand.feature_blocks == ('returns',) for cand in registry)


def test_pure_returns_pls_uses_full_train_end_semantics() -> None:
    returns, _, _, _, _ = _synthetic_panels(periods=24)
    # train_end_pos = 11, horizon = 6 -> legacy recipe should still have 6 usable rows.
    train_dates = returns.index[:12]
    factors = _fit_pls_returns_to_future_avg_returns(
        returns,
        train_dates=train_dates,
        n_components=2,
        horizon=6,
        smooth_span=0,
    )
    assert list(factors.index) == list(returns.index)
    assert factors.shape == (len(returns), 2)
    assert np.isfinite(factors.to_numpy(dtype=float)).all()


def test_predictor_pls_uses_full_train_end_semantics() -> None:
    returns, macro, ff3, ff5, bond = _synthetic_panels(periods=24)
    train_dates = returns.index[:12]
    candidate = FactorZooCandidate(
        name='pls_ret_macro7_H6_k2',
        kind='pls',
        horizon=6,
        n_components=2,
        feature_blocks=('returns', 'macro7'),
    )
    factors = _fit_pls_predictors_to_future_avg_returns(
        candidate,
        returns=returns,
        macro=macro,
        ff3=ff3,
        ff5=ff5,
        bond=bond,
        train_dates=train_dates,
        smooth_span=0,
    )
    assert list(factors.index) == list(returns.index)
    assert factors.shape == (len(returns), 2)
    assert np.isfinite(factors.to_numpy(dtype=float)).all()
