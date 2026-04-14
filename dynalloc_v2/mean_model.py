from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MeanModelResult:
    kind: str
    coef: np.ndarray
    columns: list[str]
    assets: list[str]
    factor_columns: list[str] | None = None
    asset_alpha: pd.Series | None = None
    loadings: pd.DataFrame | None = None
    regime_beta_low: np.ndarray | None = None
    regime_beta_high: np.ndarray | None = None
    regime_threshold: float | None = None
    regime_sharpness: float | None = None
    regime_ref_var: np.ndarray | None = None
    regime_high_fraction: float | None = None

    def _state_features(self, state_row: pd.Series) -> np.ndarray:
        return np.concatenate([[1.0], state_row[self.columns].to_numpy(dtype=float)])

    def _factor_row(self, latest_factor_return: pd.Series | np.ndarray | None) -> np.ndarray | None:
        if latest_factor_return is None or not self.factor_columns:
            return None
        if isinstance(latest_factor_return, pd.Series):
            arr = latest_factor_return.reindex(self.factor_columns).to_numpy(dtype=float)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.asarray(latest_factor_return, dtype=float).reshape(-1)
        if arr.size == len(self.factor_columns):
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size > len(self.factor_columns):
            return np.nan_to_num(arr[: len(self.factor_columns)], nan=0.0, posinf=0.0, neginf=0.0)
        out = np.zeros(len(self.factor_columns), dtype=float)
        out[: arr.size] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def regime_probability(
        self,
        latest_factor_return: pd.Series | np.ndarray | None = None,
        *,
        regime_weight: float | None = None,
    ) -> float:
        if regime_weight is not None:
            return float(np.clip(float(regime_weight), 0.0, 1.0))
        if self.kind != 'factor_apt_regime':
            return 0.0
        if self.regime_threshold is None or self.regime_sharpness is None or self.regime_ref_var is None:
            return float(np.clip(self.regime_high_fraction if self.regime_high_fraction is not None else 0.5, 0.0, 1.0))
        row = self._factor_row(latest_factor_return)
        if row is None:
            return float(np.clip(self.regime_high_fraction if self.regime_high_fraction is not None else 0.5, 0.0, 1.0))
        scale = np.clip(np.asarray(self.regime_ref_var, dtype=float), 1.0e-12, None)
        energy = float(np.log(max(float(np.mean((row ** 2) / scale)), 1.0e-12)))
        x = np.clip(float(self.regime_sharpness) * (energy - float(self.regime_threshold)), -60.0, 60.0)
        return float(1.0 / (1.0 + np.exp(-x)))

    def predict(
        self,
        state_row: pd.Series,
        *,
        latest_factor_return: pd.Series | np.ndarray | None = None,
        regime_weight: float | None = None,
    ) -> pd.Series:
        x = self._state_features(state_row)
        if self.kind == 'direct_assets':
            mu = x @ self.coef
            return pd.Series(mu, index=self.assets)
        if self.kind == 'factor_apt_regime' and self.regime_beta_low is not None and self.regime_beta_high is not None:
            w_high = self.regime_probability(latest_factor_return, regime_weight=regime_weight)
            factor_mu = (1.0 - w_high) * (x @ self.regime_beta_low) + w_high * (x @ self.regime_beta_high)
        else:
            factor_mu = x @ self.coef
        factor_mu_s = pd.Series(factor_mu, index=self.factor_columns)
        asset_mu = self.asset_alpha + (self.loadings @ factor_mu_s)
        return pd.Series(asset_mu, index=self.assets)

    def predict_factor_means(
        self,
        state_row: pd.Series,
        *,
        latest_factor_return: pd.Series | np.ndarray | None = None,
        regime_weight: float | None = None,
    ) -> pd.Series | None:
        if self.kind not in {'factor_apt', 'factor_apt_regime'}:
            return None
        x = self._state_features(state_row)
        if self.kind == 'factor_apt_regime' and self.regime_beta_low is not None and self.regime_beta_high is not None:
            w_high = self.regime_probability(latest_factor_return, regime_weight=regime_weight)
            factor_mu = (1.0 - w_high) * (x @ self.regime_beta_low) + w_high * (x @ self.regime_beta_high)
        else:
            factor_mu = x @ self.coef
        return pd.Series(factor_mu, index=self.factor_columns)



def _ridge_beta(X: np.ndarray, Y: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    XtX = X.T @ X
    reg = ridge_lambda * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(XtX + reg, X.T @ Y)


def fit_direct_asset_mean(states_t: pd.DataFrame, returns_tp1: pd.DataFrame, ridge_lambda: float = 1e-6) -> MeanModelResult:
    common = states_t.index.intersection(returns_tp1.index)
    X0 = states_t.loc[common]
    Y = returns_tp1.loc[common].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(common)), X0.to_numpy(dtype=float)])
    beta = _ridge_beta(X, Y, ridge_lambda=ridge_lambda)
    return MeanModelResult(kind='direct_assets', coef=beta, columns=list(X0.columns), assets=list(returns_tp1.columns))


def fit_factor_apt_mean(
    states_t: pd.DataFrame,
    factor_returns_tp1: pd.DataFrame,
    loadings: pd.DataFrame,
    asset_alpha: pd.Series,
    ridge_lambda: float = 1e-6,
) -> MeanModelResult:
    common = states_t.index.intersection(factor_returns_tp1.index)
    X0 = states_t.loc[common]
    F = factor_returns_tp1.loc[common].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(common)), X0.to_numpy(dtype=float)])
    beta = _ridge_beta(X, F, ridge_lambda=ridge_lambda)
    return MeanModelResult(
        kind='factor_apt',
        coef=beta,
        columns=list(X0.columns),
        assets=list(loadings.index),
        factor_columns=list(factor_returns_tp1.columns),
        asset_alpha=asset_alpha.reindex(loadings.index),
        loadings=loadings.reindex(index=loadings.index, columns=factor_returns_tp1.columns),
    )


def fit_factor_apt_regime_mean(
    states_t: pd.DataFrame,
    factor_returns_tp1: pd.DataFrame,
    loadings: pd.DataFrame,
    asset_alpha: pd.Series,
    ridge_lambda: float = 1e-6,
    regime_threshold_quantile: float = 0.75,
    regime_sharpness: float = 8.0,
) -> MeanModelResult:
    common = states_t.index.intersection(factor_returns_tp1.index)
    X0 = states_t.loc[common]
    F_df = factor_returns_tp1.loc[common]
    F = F_df.to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(common)), X0.to_numpy(dtype=float)])
    beta_full = _ridge_beta(X, F, ridge_lambda=ridge_lambda)

    if len(common) < 8:
        return MeanModelResult(
            kind='factor_apt_regime',
            coef=beta_full,
            columns=list(X0.columns),
            assets=list(loadings.index),
            factor_columns=list(F_df.columns),
            asset_alpha=asset_alpha.reindex(loadings.index),
            loadings=loadings.reindex(index=loadings.index, columns=F_df.columns),
            regime_beta_low=beta_full,
            regime_beta_high=beta_full,
            regime_threshold=0.0,
            regime_sharpness=float(regime_sharpness),
            regime_ref_var=np.clip(F.var(axis=0, ddof=1), 1.0e-6, None),
            regime_high_fraction=0.5,
        )

    ref_var = np.clip(np.var(F, axis=0, ddof=1), 1.0e-6, None)
    energy = np.array([np.log(max(float(np.mean((row ** 2) / ref_var)), 1.0e-12)) for row in F], dtype=float)
    threshold = float(np.quantile(energy, regime_threshold_quantile))
    high_mask = energy >= threshold
    frac_high = float(np.mean(high_mask)) if len(high_mask) else 0.0
    if frac_high <= 0.10 or frac_high >= 0.90:
        threshold = float(np.median(energy))
        high_mask = energy >= threshold
    if high_mask.all() or (~high_mask).all():
        order = np.argsort(energy)
        split = max(1, len(order) // 2)
        high_mask = np.zeros(len(order), dtype=bool)
        high_mask[order[-split:]] = True
        threshold = float(0.5 * (energy[order[split - 1]] + energy[order[-split]])) if len(order) > 1 else float(energy[0])
    low_mask = ~high_mask

    X_low = X[low_mask]
    F_low = F[low_mask]
    X_high = X[high_mask]
    F_high = F[high_mask]
    beta_low = _ridge_beta(X_low, F_low, ridge_lambda=ridge_lambda) if len(X_low) >= X.shape[1] else beta_full
    beta_high = _ridge_beta(X_high, F_high, ridge_lambda=ridge_lambda) if len(X_high) >= X.shape[1] else beta_full

    return MeanModelResult(
        kind='factor_apt_regime',
        coef=beta_full,
        columns=list(X0.columns),
        assets=list(loadings.index),
        factor_columns=list(F_df.columns),
        asset_alpha=asset_alpha.reindex(loadings.index),
        loadings=loadings.reindex(index=loadings.index, columns=F_df.columns),
        regime_beta_low=beta_low,
        regime_beta_high=beta_high,
        regime_threshold=float(threshold),
        regime_sharpness=float(regime_sharpness),
        regime_ref_var=ref_var,
        regime_high_fraction=float(np.mean(high_mask)),
    )
