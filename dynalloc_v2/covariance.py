from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CovarianceForecast:
    factor_cov: np.ndarray
    asset_cov: np.ndarray
    factor_var: pd.Series


class CovarianceModel:
    def fit(
        self,
        states_t: pd.DataFrame,
        factor_returns_t: pd.DataFrame,
        *,
        asset_returns_tp1: pd.DataFrame | None = None,
        asset_mean_pred: np.ndarray | pd.DataFrame | None = None,
    ):
        raise NotImplementedError

    def predict(
        self,
        state_row: pd.Series,
        latest_factor_return: pd.Series,
        loadings: pd.DataFrame,
        residual_var: pd.Series,
    ) -> CovarianceForecast:
        raise NotImplementedError

    def update_with_realized(
        self,
        realized_return: pd.Series | np.ndarray,
        predicted_mean: pd.Series | np.ndarray | None = None,
    ) -> None:
        return None

    def regime_probability(self) -> float:
        return 0.0


def _make_psd(mat: np.ndarray, *, floor: float = 1.0e-10) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, floor)
    out = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (out + out.T)


def _corr_from_cov(cov: np.ndarray, *, variance_floor: float) -> tuple[np.ndarray, np.ndarray]:
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    d = np.sqrt(np.maximum(np.diag(cov), variance_floor))
    denom = np.outer(d, d)
    corr = np.divide(cov, denom, out=np.eye(len(d), dtype=float), where=denom > 0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return d, corr


def _corr_from_q(q: np.ndarray, *, variance_floor: float, correlation_shrink: float) -> np.ndarray:
    q = _make_psd(q, floor=variance_floor)
    d = np.sqrt(np.maximum(np.diag(q), variance_floor))
    denom = np.outer(d, d)
    corr = np.divide(q, denom, out=np.eye(len(d), dtype=float), where=denom > 0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    if correlation_shrink > 0.0:
        corr = (1.0 - correlation_shrink) * corr + correlation_shrink * np.eye(corr.shape[0])
    np.fill_diagonal(corr, 1.0)
    return _make_psd(corr, floor=variance_floor)


def _weighted_cov(arr: np.ndarray, weights: np.ndarray, *, variance_floor: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if arr.ndim != 2 or arr.shape[0] != len(weights):
        raise ValueError('weighted covariance requires a 2d array aligned with weights')
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(arr.shape[0], dtype=float)
    weights = weights / float(weights.sum())
    mean = np.sum(arr * weights[:, None], axis=0)
    centered = arr - mean
    cov = (centered * weights[:, None]).T @ centered
    return _make_psd(cov, floor=variance_floor)


class ConstantFactorCovariance(CovarianceModel):
    def __init__(
        self,
        variance_floor: float = 1e-6,
        correlation_shrink: float = 0.0,
        factor_correlation_mode: str = 'independent',
    ):
        self.variance_floor = variance_floor
        self.correlation_shrink = correlation_shrink
        self.factor_correlation_mode = factor_correlation_mode
        self.factor_cov_: np.ndarray | None = None

    def fit(
        self,
        states_t: pd.DataFrame,
        factor_returns_t: pd.DataFrame,
        *,
        asset_returns_tp1: pd.DataFrame | None = None,
        asset_mean_pred: np.ndarray | pd.DataFrame | None = None,
    ):
        X = factor_returns_t.to_numpy(dtype=float)
        cov = np.cov(X.T, ddof=1)
        d, corr = _factor_correlation(
            cov,
            variance_floor=self.variance_floor,
            correlation_shrink=self.correlation_shrink,
            factor_correlation_mode=self.factor_correlation_mode,
        )
        self.factor_cov_ = np.outer(d, d) * corr
        return self

    def predict(self, state_row: pd.Series, latest_factor_return: pd.Series, loadings: pd.DataFrame, residual_var: pd.Series) -> CovarianceForecast:
        fcov = self.factor_cov_
        asset_cov = loadings.to_numpy(dtype=float) @ fcov @ loadings.to_numpy(dtype=float).T + np.diag(residual_var.to_numpy(dtype=float))
        return CovarianceForecast(
            factor_cov=fcov,
            asset_cov=asset_cov,
            factor_var=pd.Series(np.diag(fcov), index=loadings.columns),
        )


class StateDiagonalFactorCovariance(CovarianceModel):
    def __init__(
        self,
        ridge_lambda: float = 1e-6,
        variance_floor: float = 1e-6,
        correlation_shrink: float = 0.1,
        factor_correlation_mode: str = 'independent',
        use_persistence: bool = True,
    ):
        self.ridge_lambda = ridge_lambda
        self.variance_floor = variance_floor
        self.correlation_shrink = correlation_shrink
        self.factor_correlation_mode = factor_correlation_mode
        self.use_persistence = use_persistence
        self.beta_: dict[str, np.ndarray] = {}
        self.state_cols_: list[str] = []
        self.factor_cols_: list[str] = []
        self.factor_corr_: np.ndarray | None = None

    def fit(
        self,
        states_t: pd.DataFrame,
        factor_returns_t: pd.DataFrame,
        *,
        asset_returns_tp1: pd.DataFrame | None = None,
        asset_mean_pred: np.ndarray | pd.DataFrame | None = None,
    ):
        common = states_t.index.intersection(factor_returns_t.index)
        states = states_t.loc[common]
        factors = factor_returns_t.loc[common]
        self.state_cols_ = list(states.columns)
        self.factor_cols_ = list(factors.columns)

        Xcorr = factors.to_numpy(dtype=float)
        cov = np.cov(Xcorr.T, ddof=1)
        _, corr = _factor_correlation(
            cov,
            variance_floor=self.variance_floor,
            correlation_shrink=self.correlation_shrink,
            factor_correlation_mode=self.factor_correlation_mode,
        )
        self.factor_corr_ = corr

        if len(common) < 3:
            for col in self.factor_cols_:
                self.beta_[col] = np.array([np.log(np.maximum(factors[col].var(ddof=1), self.variance_floor))])
            return self

        f_arr = factors.to_numpy(dtype=float)
        s_arr = states.to_numpy(dtype=float)
        for j, col in enumerate(self.factor_cols_):
            y = np.log(np.maximum(f_arr[1:, j] ** 2, self.variance_floor))
            parts = [np.ones(len(y))]
            if self.use_persistence:
                parts.append(np.log(np.maximum(f_arr[:-1, j] ** 2, self.variance_floor)))
            if s_arr.shape[1] > 0:
                parts.extend([s_arr[:-1, k] for k in range(s_arr.shape[1])])
            X = np.column_stack(parts)
            XtX = X.T @ X
            reg = self.ridge_lambda * np.eye(X.shape[1])
            reg[0, 0] = 0.0
            beta = np.linalg.solve(XtX + reg, X.T @ y)
            self.beta_[col] = beta
        return self

    def predict(self, state_row: pd.Series, latest_factor_return: pd.Series, loadings: pd.DataFrame, residual_var: pd.Series) -> CovarianceForecast:
        vars_ = []
        for col in self.factor_cols_:
            beta = self.beta_[col]
            feats = [1.0]
            if self.use_persistence and len(beta) > 1:
                feats.append(float(np.log(max(float(latest_factor_return[col]) ** 2, self.variance_floor))))
            for c in self.state_cols_:
                feats.append(float(state_row[c]))
            x = np.array(feats[: len(beta)], dtype=float)
            logh = float(x @ beta)
            vars_.append(float(np.exp(logh)))
        vars_arr = np.maximum(np.array(vars_, dtype=float), self.variance_floor)
        D = np.diag(np.sqrt(vars_arr))
        fcov = D @ self.factor_corr_ @ D
        asset_cov = loadings.to_numpy(dtype=float) @ fcov @ loadings.to_numpy(dtype=float).T + np.diag(residual_var.to_numpy(dtype=float))
        return CovarianceForecast(
            factor_cov=fcov,
            asset_cov=asset_cov,
            factor_var=pd.Series(vars_arr, index=self.factor_cols_),
        )


class AssetDCCCovariance(CovarianceModel):
    def __init__(
        self,
        variance_floor: float = 1.0e-6,
        correlation_shrink: float = 0.10,
        dcc_alpha: float = 0.02,
        dcc_beta: float = 0.97,
        variance_lambda: float = 0.97,
        asset_covariance_shrink: float = 0.10,
    ):
        if dcc_alpha < 0.0 or dcc_beta < 0.0 or dcc_alpha + dcc_beta >= 1.0:
            raise ValueError('dcc_alpha and dcc_beta must satisfy alpha>=0, beta>=0, alpha+beta<1')
        if not (0.0 < variance_lambda < 1.0):
            raise ValueError('variance_lambda must be in (0, 1)')
        self.variance_floor = float(variance_floor)
        self.correlation_shrink = float(correlation_shrink)
        self.dcc_alpha = float(dcc_alpha)
        self.dcc_beta = float(dcc_beta)
        self.variance_lambda = float(variance_lambda)
        self.asset_covariance_shrink = float(asset_covariance_shrink)
        self.asset_cols_: list[str] = []
        self.residual_center_: np.ndarray | None = None
        self.uncond_corr_: np.ndarray | None = None
        self.last_var_: np.ndarray | None = None
        self.last_q_: np.ndarray | None = None

    def _asset_cov_from_state(self) -> np.ndarray:
        std = np.sqrt(np.maximum(self.last_var_, self.variance_floor))
        corr = _corr_from_q(self.last_q_, variance_floor=self.variance_floor, correlation_shrink=self.correlation_shrink)
        cov = np.diag(std) @ corr @ np.diag(std)
        if self.asset_covariance_shrink > 0.0:
            cov = (1.0 - self.asset_covariance_shrink) * cov + self.asset_covariance_shrink * np.diag(np.diag(cov))
        return _make_psd(cov, floor=self.variance_floor)

    def _update_with_centered_residual(self, resid: np.ndarray) -> None:
        std = np.sqrt(np.maximum(self.last_var_, self.variance_floor))
        z = np.divide(resid, std, out=np.zeros_like(resid, dtype=float), where=std > 0)
        self.last_var_ = self.variance_lambda * self.last_var_ + (1.0 - self.variance_lambda) * (resid ** 2)
        self.last_var_ = np.maximum(self.last_var_, self.variance_floor)
        self.last_q_ = (
            (1.0 - self.dcc_alpha - self.dcc_beta) * self.uncond_corr_
            + self.dcc_alpha * np.outer(z, z)
            + self.dcc_beta * self.last_q_
        )
        self.last_q_ = _make_psd(self.last_q_, floor=self.variance_floor)

    def fit(
        self,
        states_t: pd.DataFrame,
        factor_returns_t: pd.DataFrame,
        *,
        asset_returns_tp1: pd.DataFrame | None = None,
        asset_mean_pred: np.ndarray | pd.DataFrame | None = None,
    ):
        if asset_returns_tp1 is None:
            raise ValueError('asset_returns_tp1 is required for asset_dcc covariance')
        asset_df = asset_returns_tp1.copy()
        self.asset_cols_ = list(asset_df.columns)
        mean_df: pd.DataFrame | None = None
        if asset_mean_pred is not None:
            if isinstance(asset_mean_pred, pd.DataFrame):
                mean_df = asset_mean_pred.reindex(index=asset_df.index, columns=asset_df.columns)
            else:
                mean_df = pd.DataFrame(np.asarray(asset_mean_pred, dtype=float), index=asset_df.index, columns=asset_df.columns)
        resid_df = asset_df if mean_df is None else (asset_df - mean_df)
        resid_df = resid_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
        if resid_df.empty:
            raise ValueError('No valid residual rows available for asset_dcc covariance')

        resid_arr = resid_df.to_numpy(dtype=float)
        self.residual_center_ = resid_arr.mean(axis=0)
        centered = resid_arr - self.residual_center_
        sample_cov = np.cov(centered.T, ddof=1) if len(centered) > 1 else np.diag(np.maximum(centered[0] ** 2, self.variance_floor))
        if np.ndim(sample_cov) == 0:
            sample_cov = np.array([[float(sample_cov)]], dtype=float)
        sample_cov = _make_psd(np.asarray(sample_cov, dtype=float), floor=self.variance_floor)
        sample_var = np.maximum(np.diag(sample_cov), self.variance_floor)
        _, sample_corr = _corr_from_cov(sample_cov, variance_floor=self.variance_floor)
        if self.correlation_shrink > 0.0:
            sample_corr = (1.0 - self.correlation_shrink) * sample_corr + self.correlation_shrink * np.eye(sample_corr.shape[0])
        self.uncond_corr_ = _make_psd(sample_corr, floor=self.variance_floor)
        self.last_var_ = sample_var.copy()
        self.last_q_ = self.uncond_corr_.copy()
        for resid in centered:
            self._update_with_centered_residual(resid)
        return self

    def predict(self, state_row: pd.Series, latest_factor_return: pd.Series, loadings: pd.DataFrame, residual_var: pd.Series) -> CovarianceForecast:
        asset_cov = self._asset_cov_from_state()
        return CovarianceForecast(
            factor_cov=np.zeros((0, 0), dtype=float),
            asset_cov=asset_cov,
            factor_var=pd.Series(np.diag(asset_cov), index=self.asset_cols_),
        )

    def update_with_realized(self, realized_return: pd.Series | np.ndarray, predicted_mean: pd.Series | np.ndarray | None = None) -> None:
        realized_arr = np.asarray(realized_return, dtype=float)
        if predicted_mean is None:
            resid = realized_arr
        else:
            resid = realized_arr - np.asarray(predicted_mean, dtype=float)
        centered = resid - self.residual_center_
        self._update_with_centered_residual(centered)


class AssetADCCCovariance(AssetDCCCovariance):
    def __init__(
        self,
        variance_floor: float = 1.0e-6,
        correlation_shrink: float = 0.10,
        dcc_alpha: float = 0.02,
        dcc_beta: float = 0.97,
        adcc_gamma: float = 0.005,
        variance_lambda: float = 0.97,
        asset_covariance_shrink: float = 0.10,
    ):
        if adcc_gamma < 0.0 or dcc_alpha + dcc_beta + adcc_gamma >= 1.0:
            raise ValueError('adcc_gamma must satisfy gamma>=0 and alpha+beta+gamma<1')
        super().__init__(
            variance_floor=variance_floor,
            correlation_shrink=correlation_shrink,
            dcc_alpha=dcc_alpha,
            dcc_beta=dcc_beta,
            variance_lambda=variance_lambda,
            asset_covariance_shrink=asset_covariance_shrink,
        )
        self.adcc_gamma = float(adcc_gamma)
        self.neg_outer_mean_: np.ndarray | None = None

    def _update_with_centered_residual(self, resid: np.ndarray) -> None:
        std = np.sqrt(np.maximum(self.last_var_, self.variance_floor))
        z = np.divide(resid, std, out=np.zeros_like(resid, dtype=float), where=std > 0)
        n = np.minimum(z, 0.0)
        self.last_var_ = self.variance_lambda * self.last_var_ + (1.0 - self.variance_lambda) * (resid ** 2)
        self.last_var_ = np.maximum(self.last_var_, self.variance_floor)
        intercept = (1.0 - self.dcc_alpha - self.dcc_beta) * self.uncond_corr_
        if self.neg_outer_mean_ is not None and self.adcc_gamma > 0.0:
            intercept = intercept - self.adcc_gamma * self.neg_outer_mean_
        self.last_q_ = (
            intercept
            + self.dcc_alpha * np.outer(z, z)
            + self.dcc_beta * self.last_q_
            + self.adcc_gamma * np.outer(n, n)
        )
        self.last_q_ = _make_psd(self.last_q_, floor=self.variance_floor)

    def fit(
        self,
        states_t: pd.DataFrame,
        factor_returns_t: pd.DataFrame,
        *,
        asset_returns_tp1: pd.DataFrame | None = None,
        asset_mean_pred: np.ndarray | pd.DataFrame | None = None,
    ):
        if asset_returns_tp1 is None:
            raise ValueError('asset_returns_tp1 is required for asset_adcc covariance')
        asset_df = asset_returns_tp1.copy()
        self.asset_cols_ = list(asset_df.columns)
        mean_df: pd.DataFrame | None = None
        if asset_mean_pred is not None:
            if isinstance(asset_mean_pred, pd.DataFrame):
                mean_df = asset_mean_pred.reindex(index=asset_df.index, columns=asset_df.columns)
            else:
                mean_df = pd.DataFrame(np.asarray(asset_mean_pred, dtype=float), index=asset_df.index, columns=asset_df.columns)
        resid_df = asset_df if mean_df is None else (asset_df - mean_df)
        resid_df = resid_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
        if resid_df.empty:
            raise ValueError('No valid residual rows available for asset_adcc covariance')

        resid_arr = resid_df.to_numpy(dtype=float)
        self.residual_center_ = resid_arr.mean(axis=0)
        centered = resid_arr - self.residual_center_
        sample_cov = np.cov(centered.T, ddof=1) if len(centered) > 1 else np.diag(np.maximum(centered[0] ** 2, self.variance_floor))
        if np.ndim(sample_cov) == 0:
            sample_cov = np.array([[float(sample_cov)]], dtype=float)
        sample_cov = _make_psd(np.asarray(sample_cov, dtype=float), floor=self.variance_floor)
        sample_var = np.maximum(np.diag(sample_cov), self.variance_floor)
        _, sample_corr = _corr_from_cov(sample_cov, variance_floor=self.variance_floor)
        if self.correlation_shrink > 0.0:
            sample_corr = (1.0 - self.correlation_shrink) * sample_corr + self.correlation_shrink * np.eye(sample_corr.shape[0])
        self.uncond_corr_ = _make_psd(sample_corr, floor=self.variance_floor)
        std = np.sqrt(sample_var)
        z_hist = np.divide(centered, std, out=np.zeros_like(centered, dtype=float), where=std > 0)
        n_hist = np.minimum(z_hist, 0.0)
        if len(n_hist) == 0:
            self.neg_outer_mean_ = np.zeros_like(self.uncond_corr_)
        else:
            neg_outer = np.einsum('ti,tj->ij', n_hist, n_hist) / float(len(n_hist))
            self.neg_outer_mean_ = _make_psd(np.asarray(neg_outer, dtype=float), floor=self.variance_floor)
        self.last_var_ = sample_var.copy()
        self.last_q_ = self.uncond_corr_.copy()
        for resid in centered:
            self._update_with_centered_residual(resid)
        return self


class AssetRegimeDCCCovariance(CovarianceModel):
    def __init__(
        self,
        variance_floor: float = 1.0e-6,
        correlation_shrink: float = 0.10,
        dcc_alpha: float = 0.02,
        dcc_beta: float = 0.97,
        variance_lambda: float = 0.97,
        asset_covariance_shrink: float = 0.10,
        regime_threshold_quantile: float = 0.75,
        regime_smoothing: float = 0.90,
        regime_sharpness: float = 8.0,
    ):
        if dcc_alpha < 0.0 or dcc_beta < 0.0 or dcc_alpha + dcc_beta >= 1.0:
            raise ValueError('dcc_alpha and dcc_beta must satisfy alpha>=0, beta>=0, alpha+beta<1')
        if not (0.0 < variance_lambda < 1.0):
            raise ValueError('variance_lambda must be in (0, 1)')
        if not (0.0 < regime_threshold_quantile < 1.0):
            raise ValueError('regime_threshold_quantile must be in (0, 1)')
        if not (0.0 < regime_smoothing < 1.0):
            raise ValueError('regime_smoothing must be in (0, 1)')
        if regime_sharpness <= 0.0:
            raise ValueError('regime_sharpness must be positive')
        self.variance_floor = float(variance_floor)
        self.correlation_shrink = float(correlation_shrink)
        self.dcc_alpha = float(dcc_alpha)
        self.dcc_beta = float(dcc_beta)
        self.variance_lambda = float(variance_lambda)
        self.asset_covariance_shrink = float(asset_covariance_shrink)
        self.regime_threshold_quantile = float(regime_threshold_quantile)
        self.regime_smoothing = float(regime_smoothing)
        self.regime_sharpness = float(regime_sharpness)
        self.asset_cols_: list[str] = []
        self.residual_center_: np.ndarray | None = None
        self.uncond_corr_low_: np.ndarray | None = None
        self.uncond_corr_high_: np.ndarray | None = None
        self.last_var_low_: np.ndarray | None = None
        self.last_var_high_: np.ndarray | None = None
        self.last_q_low_: np.ndarray | None = None
        self.last_q_high_: np.ndarray | None = None
        self.regime_threshold_: float | None = None
        self.regime_score_: float | None = None

    def _asset_cov_from_components(self, var: np.ndarray, q: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(var, self.variance_floor))
        corr = _corr_from_q(q, variance_floor=self.variance_floor, correlation_shrink=self.correlation_shrink)
        return np.diag(std) @ corr @ np.diag(std)

    def _regime_weight(self, score: float | None = None) -> float:
        if self.regime_threshold_ is None:
            return 0.5
        score_val = float(self.regime_score_ if score is None else score)
        x = np.clip(self.regime_sharpness * (score_val - float(self.regime_threshold_)), -60.0, 60.0)
        return float(1.0 / (1.0 + np.exp(-x)))

    def _blended_var(self) -> np.ndarray:
        w_high = self._regime_weight()
        return (1.0 - w_high) * self.last_var_low_ + w_high * self.last_var_high_

    def _asset_cov_from_state(self) -> np.ndarray:
        w_high = self._regime_weight()
        cov_low = self._asset_cov_from_components(self.last_var_low_, self.last_q_low_)
        cov_high = self._asset_cov_from_components(self.last_var_high_, self.last_q_high_)
        cov = (1.0 - w_high) * cov_low + w_high * cov_high
        if self.asset_covariance_shrink > 0.0:
            cov = (1.0 - self.asset_covariance_shrink) * cov + self.asset_covariance_shrink * np.diag(np.diag(cov))
        return _make_psd(cov, floor=self.variance_floor)

    def _energy_score(self, resid: np.ndarray, ref_var: np.ndarray) -> float:
        scale = np.maximum(ref_var, self.variance_floor)
        energy = float(np.mean((resid ** 2) / scale))
        return float(np.log(max(energy, self.variance_floor)))

    def _update_regime_state(self, resid: np.ndarray, *, regime: str) -> None:
        if regime == 'high':
            var = self.last_var_high_
            q = self.last_q_high_
            uncond = self.uncond_corr_high_
        else:
            var = self.last_var_low_
            q = self.last_q_low_
            uncond = self.uncond_corr_low_
        std = np.sqrt(np.maximum(var, self.variance_floor))
        z = np.divide(resid, std, out=np.zeros_like(resid, dtype=float), where=std > 0)
        new_var = self.variance_lambda * var + (1.0 - self.variance_lambda) * (resid ** 2)
        new_var = np.maximum(new_var, self.variance_floor)
        new_q = (
            (1.0 - self.dcc_alpha - self.dcc_beta) * uncond
            + self.dcc_alpha * np.outer(z, z)
            + self.dcc_beta * q
        )
        new_q = _make_psd(new_q, floor=self.variance_floor)
        if regime == 'high':
            self.last_var_high_ = new_var
            self.last_q_high_ = new_q
        else:
            self.last_var_low_ = new_var
            self.last_q_low_ = new_q

    def fit(
        self,
        states_t: pd.DataFrame,
        factor_returns_t: pd.DataFrame,
        *,
        asset_returns_tp1: pd.DataFrame | None = None,
        asset_mean_pred: np.ndarray | pd.DataFrame | None = None,
    ):
        if asset_returns_tp1 is None:
            raise ValueError('asset_returns_tp1 is required for asset_regime_dcc covariance')
        asset_df = asset_returns_tp1.copy()
        self.asset_cols_ = list(asset_df.columns)
        mean_df: pd.DataFrame | None = None
        if asset_mean_pred is not None:
            if isinstance(asset_mean_pred, pd.DataFrame):
                mean_df = asset_mean_pred.reindex(index=asset_df.index, columns=asset_df.columns)
            else:
                mean_df = pd.DataFrame(np.asarray(asset_mean_pred, dtype=float), index=asset_df.index, columns=asset_df.columns)
        resid_df = asset_df if mean_df is None else (asset_df - mean_df)
        resid_df = resid_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
        if resid_df.empty:
            raise ValueError('No valid residual rows available for asset_regime_dcc covariance')

        resid_arr = resid_df.to_numpy(dtype=float)
        self.residual_center_ = resid_arr.mean(axis=0)
        centered = resid_arr - self.residual_center_
        sample_cov = np.cov(centered.T, ddof=1) if len(centered) > 1 else np.diag(np.maximum(centered[0] ** 2, self.variance_floor))
        if np.ndim(sample_cov) == 0:
            sample_cov = np.array([[float(sample_cov)]], dtype=float)
        sample_cov = _make_psd(np.asarray(sample_cov, dtype=float), floor=self.variance_floor)
        sample_var = np.maximum(np.diag(sample_cov), self.variance_floor)
        energy_hist = np.array([self._energy_score(resid, sample_var) for resid in centered], dtype=float)
        threshold = float(np.quantile(energy_hist, self.regime_threshold_quantile))
        high_mask = energy_hist >= threshold
        frac_high = float(np.mean(high_mask)) if len(high_mask) else 0.0
        if frac_high <= 0.10 or frac_high >= 0.90:
            threshold = float(np.median(energy_hist))
            high_mask = energy_hist >= threshold
        if high_mask.all() or (~high_mask).all():
            order = np.argsort(energy_hist)
            split = max(1, len(order) // 2)
            high_mask = np.zeros(len(order), dtype=bool)
            high_mask[order[-split:]] = True
            threshold = float(0.5 * (energy_hist[order[split - 1]] + energy_hist[order[-split]])) if len(order) > 1 else float(energy_hist[0])
        low_mask = ~high_mask
        cov_low = _weighted_cov(centered, low_mask.astype(float), variance_floor=self.variance_floor)
        cov_high = _weighted_cov(centered, high_mask.astype(float), variance_floor=self.variance_floor)
        var_low = np.maximum(np.diag(cov_low), self.variance_floor)
        var_high = np.maximum(np.diag(cov_high), self.variance_floor)
        _, corr_low = _corr_from_cov(cov_low, variance_floor=self.variance_floor)
        _, corr_high = _corr_from_cov(cov_high, variance_floor=self.variance_floor)
        if self.correlation_shrink > 0.0:
            eye = np.eye(corr_low.shape[0])
            corr_low = (1.0 - self.correlation_shrink) * corr_low + self.correlation_shrink * eye
            corr_high = (1.0 - self.correlation_shrink) * corr_high + self.correlation_shrink * eye
        self.uncond_corr_low_ = _make_psd(corr_low, floor=self.variance_floor)
        self.uncond_corr_high_ = _make_psd(corr_high, floor=self.variance_floor)
        self.last_var_low_ = var_low.copy()
        self.last_var_high_ = var_high.copy()
        self.last_q_low_ = self.uncond_corr_low_.copy()
        self.last_q_high_ = self.uncond_corr_high_.copy()
        self.regime_threshold_ = threshold
        score = float(energy_hist[0]) if len(energy_hist) else 0.0
        for resid, energy in zip(centered, energy_hist):
            score = self.regime_smoothing * score + (1.0 - self.regime_smoothing) * float(energy)
            regime = 'high' if self._regime_weight(score) >= 0.5 else 'low'
            self._update_regime_state(resid, regime=regime)
        self.regime_score_ = score
        return self

    def predict(self, state_row: pd.Series, latest_factor_return: pd.Series, loadings: pd.DataFrame, residual_var: pd.Series) -> CovarianceForecast:
        asset_cov = self._asset_cov_from_state()
        return CovarianceForecast(
            factor_cov=np.zeros((0, 0), dtype=float),
            asset_cov=asset_cov,
            factor_var=pd.Series(np.diag(asset_cov), index=self.asset_cols_),
        )

    def update_with_realized(self, realized_return: pd.Series | np.ndarray, predicted_mean: pd.Series | np.ndarray | None = None) -> None:
        realized_arr = np.asarray(realized_return, dtype=float)
        if predicted_mean is None:
            resid = realized_arr
        else:
            resid = realized_arr - np.asarray(predicted_mean, dtype=float)
        centered = resid - self.residual_center_
        ref_var = self._blended_var()
        energy = self._energy_score(centered, ref_var)
        score_prev = float(self.regime_score_ if self.regime_score_ is not None else energy)
        self.regime_score_ = self.regime_smoothing * score_prev + (1.0 - self.regime_smoothing) * energy
        regime = 'high' if self._regime_weight() >= 0.5 else 'low'
        self._update_regime_state(centered, regime=regime)

    def regime_probability(self) -> float:
        return float(self._regime_weight())


def _factor_correlation(
    cov: np.ndarray,
    *,
    variance_floor: float,
    correlation_shrink: float,
    factor_correlation_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    d = np.sqrt(np.maximum(np.diag(cov), variance_floor))
    m = len(d)
    if factor_correlation_mode == 'independent':
        return d, np.eye(m)
    corr = cov / np.outer(d, d)
    corr = np.nan_to_num(corr)
    if correlation_shrink > 0:
        corr = (1.0 - correlation_shrink) * corr + correlation_shrink * np.eye(m)
    return d, _make_psd(corr, floor=variance_floor)
