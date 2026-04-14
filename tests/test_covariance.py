from __future__ import annotations

import numpy as np
import pandas as pd

from dynalloc_v2.covariance import AssetADCCCovariance, AssetDCCCovariance, AssetRegimeDCCCovariance, ConstantFactorCovariance, StateDiagonalFactorCovariance


def test_constant_factor_covariance_independent_is_diagonal():
    dates = pd.date_range('2000-01-31', periods=12, freq='ME')
    factors = pd.DataFrame(
        {
            'F1': np.linspace(-0.02, 0.03, len(dates)),
            'F2': np.linspace(0.01, -0.01, len(dates)) + 0.5 * np.linspace(-0.02, 0.03, len(dates)),
        },
        index=dates,
    )
    model = ConstantFactorCovariance(factor_correlation_mode='independent')
    model.fit(pd.DataFrame(index=dates), factors)
    fcov = model.factor_cov_
    assert np.allclose(fcov, np.diag(np.diag(fcov)))


def test_state_diagonal_factor_covariance_independent_predicts_diagonal_factor_cov():
    dates = pd.date_range('2000-01-31', periods=24, freq='ME')
    states = pd.DataFrame({'s1': np.sin(np.linspace(0.0, 2.0, len(dates)))}, index=dates)
    factors = pd.DataFrame(
        {
            'F1': 0.02 * np.sin(np.linspace(0.0, 4.0, len(dates))),
            'F2': 0.015 * np.cos(np.linspace(0.0, 4.0, len(dates))) + 0.25 * 0.02 * np.sin(np.linspace(0.0, 4.0, len(dates))),
        },
        index=dates,
    )
    model = StateDiagonalFactorCovariance(factor_correlation_mode='independent', use_persistence=False)
    model.fit(states, factors)
    loadings = pd.DataFrame([[1.0, 0.2], [0.3, 1.1]], index=['A1', 'A2'], columns=['F1', 'F2'])
    residual_var = pd.Series([0.01, 0.02], index=['A1', 'A2'])
    fc = model.predict(states.iloc[-1], factors.iloc[-1], loadings, residual_var)
    assert np.allclose(fc.factor_cov, np.diag(np.diag(fc.factor_cov)))


def test_asset_dcc_covariance_is_psd_and_updates():
    dates = pd.date_range('2000-01-31', periods=36, freq='ME')
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(rng.normal(scale=0.03, size=(len(dates), 4)), index=dates, columns=['A1', 'A2', 'A3', 'A4'])
    states = pd.DataFrame({'s1': np.linspace(-1.0, 1.0, len(dates))}, index=dates)
    factors = pd.DataFrame({'F1': rng.normal(scale=0.02, size=len(dates))}, index=dates)
    mean_pred = np.zeros_like(returns.to_numpy(dtype=float))
    model = AssetDCCCovariance()
    model.fit(states, factors, asset_returns_tp1=returns, asset_mean_pred=mean_pred)
    loadings = pd.DataFrame([[1.0], [0.8], [0.6], [0.4]], index=returns.columns, columns=['F1'])
    residual_var = pd.Series(0.01, index=returns.columns)
    fc0 = model.predict(states.iloc[-1], factors.iloc[-1], loadings, residual_var)
    vals0 = np.linalg.eigvalsh(fc0.asset_cov)
    assert np.all(vals0 >= -1.0e-10)
    model.update_with_realized(returns.iloc[-1].to_numpy(dtype=float), np.zeros(returns.shape[1]))
    fc1 = model.predict(states.iloc[-1], factors.iloc[-1], loadings, residual_var)
    assert fc1.asset_cov.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(fc1.asset_cov) >= -1.0e-10)
    assert not np.allclose(fc0.asset_cov, fc1.asset_cov)


def test_asset_adcc_covariance_is_psd_and_updates():
    dates = pd.date_range('2000-01-31', periods=36, freq='ME')
    rng = np.random.default_rng(17)
    shocks = rng.normal(scale=0.03, size=(len(dates), 4))
    shocks[:, 1] += 0.5 * np.minimum(shocks[:, 0], 0.0)
    returns = pd.DataFrame(shocks, index=dates, columns=['A1', 'A2', 'A3', 'A4'])
    states = pd.DataFrame({'s1': np.linspace(-1.0, 1.0, len(dates))}, index=dates)
    factors = pd.DataFrame({'F1': rng.normal(scale=0.02, size=len(dates))}, index=dates)
    mean_pred = np.zeros_like(returns.to_numpy(dtype=float))
    model = AssetADCCCovariance(adcc_gamma=0.005)
    model.fit(states, factors, asset_returns_tp1=returns, asset_mean_pred=mean_pred)
    loadings = pd.DataFrame([[1.0], [0.8], [0.6], [0.4]], index=returns.columns, columns=['F1'])
    residual_var = pd.Series(0.01, index=returns.columns)
    fc0 = model.predict(states.iloc[-1], factors.iloc[-1], loadings, residual_var)
    vals0 = np.linalg.eigvalsh(fc0.asset_cov)
    assert np.all(vals0 >= -1.0e-10)
    model.update_with_realized(returns.iloc[-1].to_numpy(dtype=float), np.zeros(returns.shape[1]))
    fc1 = model.predict(states.iloc[-1], factors.iloc[-1], loadings, residual_var)
    assert fc1.asset_cov.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(fc1.asset_cov) >= -1.0e-10)
    assert not np.allclose(fc0.asset_cov, fc1.asset_cov)



def test_asset_regime_dcc_covariance_is_psd_and_updates():
    dates = pd.date_range('2000-01-31', periods=48, freq='ME')
    rng = np.random.default_rng(23)
    base = rng.normal(scale=0.02, size=(len(dates), 4))
    stress = np.zeros_like(base)
    stress[24:, 0] = rng.normal(scale=0.05, size=len(dates) - 24)
    stress[24:, 1] = 0.8 * stress[24:, 0] + rng.normal(scale=0.02, size=len(dates) - 24)
    returns = pd.DataFrame(base + stress, index=dates, columns=['A1', 'A2', 'A3', 'A4'])
    states = pd.DataFrame({'s1': np.linspace(-1.0, 1.0, len(dates))}, index=dates)
    factors = pd.DataFrame({'F1': rng.normal(scale=0.02, size=len(dates))}, index=dates)
    mean_pred = np.zeros_like(returns.to_numpy(dtype=float))
    model = AssetRegimeDCCCovariance(regime_threshold_quantile=0.70, regime_smoothing=0.85, regime_sharpness=6.0)
    model.fit(states, factors, asset_returns_tp1=returns, asset_mean_pred=mean_pred)
    loadings = pd.DataFrame([[1.0], [0.8], [0.6], [0.4]], index=returns.columns, columns=['F1'])
    residual_var = pd.Series(0.01, index=returns.columns)
    fc0 = model.predict(states.iloc[-1], factors.iloc[-1], loadings, residual_var)
    assert fc0.asset_cov.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(fc0.asset_cov) >= -1.0e-10)
    model.update_with_realized(returns.iloc[-1].to_numpy(dtype=float), np.zeros(returns.shape[1]))
    fc1 = model.predict(states.iloc[-1], factors.iloc[-1], loadings, residual_var)
    assert np.all(np.linalg.eigvalsh(fc1.asset_cov) >= -1.0e-10)
    assert not np.allclose(fc0.asset_cov, fc1.asset_cov)
