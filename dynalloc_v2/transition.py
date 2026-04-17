from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


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
        corr = (1.0 - correlation_shrink) * corr + correlation_shrink * np.eye(corr.shape[0], dtype=float)
    np.fill_diagonal(corr, 1.0)
    return _make_psd(corr, floor=variance_floor)


class DynamicCrossCovariance:
    def __init__(
        self,
        *,
        n_assets: int,
        n_states: int,
        kind: str = 'dcc',
        variance_floor: float = 1.0e-6,
        correlation_shrink: float = 0.10,
        dcc_alpha: float = 0.02,
        dcc_beta: float = 0.97,
        adcc_gamma: float = 0.005,
        variance_lambda: float = 0.97,
        regime_threshold_quantile: float = 0.75,
        regime_smoothing: float = 0.90,
        regime_sharpness: float = 8.0,
    ):
        self.n_assets = int(n_assets)
        self.n_states = int(n_states)
        self.kind = str(kind)
        self.variance_floor = float(variance_floor)
        self.correlation_shrink = float(correlation_shrink)
        self.dcc_alpha = float(dcc_alpha)
        self.dcc_beta = float(dcc_beta)
        self.adcc_gamma = float(adcc_gamma)
        self.variance_lambda = float(variance_lambda)
        self.regime_threshold_quantile = float(regime_threshold_quantile)
        self.regime_smoothing = float(regime_smoothing)
        self.regime_sharpness = float(regime_sharpness)
        self.center_: np.ndarray | None = None
        self.uncond_corr_: np.ndarray | None = None
        self.last_var_: np.ndarray | None = None
        self.last_q_: np.ndarray | None = None
        self.neg_outer_mean_: np.ndarray | None = None
        self.uncond_corr_low_: np.ndarray | None = None
        self.uncond_corr_high_: np.ndarray | None = None
        self.last_var_low_: np.ndarray | None = None
        self.last_var_high_: np.ndarray | None = None
        self.last_q_low_: np.ndarray | None = None
        self.last_q_high_: np.ndarray | None = None
        self.regime_threshold_: float | None = None
        self.regime_score_: float | None = None

    def _joint_cov_from_state(self) -> np.ndarray:
        std = np.sqrt(np.maximum(self.last_var_, self.variance_floor))
        corr = _corr_from_q(self.last_q_, variance_floor=self.variance_floor, correlation_shrink=self.correlation_shrink)
        cov = np.diag(std) @ corr @ np.diag(std)
        return _make_psd(cov, floor=self.variance_floor)

    def _regime_weight(self, score: float | None = None) -> float:
        if self.regime_threshold_ is None:
            return 0.5
        score_val = float(self.regime_score_ if score is None else score)
        x = np.clip(self.regime_sharpness * (score_val - float(self.regime_threshold_)), -60.0, 60.0)
        return float(1.0 / (1.0 + np.exp(-x)))

    def _asset_state_block(self, cov: np.ndarray) -> np.ndarray:
        return cov[: self.n_assets, self.n_assets :]

    def _update_dcc_state(self, resid: np.ndarray) -> None:
        std = np.sqrt(np.maximum(self.last_var_, self.variance_floor))
        z = np.divide(resid, std, out=np.zeros_like(resid, dtype=float), where=std > 0)
        self.last_var_ = self.variance_lambda * self.last_var_ + (1.0 - self.variance_lambda) * (resid ** 2)
        self.last_var_ = np.maximum(self.last_var_, self.variance_floor)
        intercept = (1.0 - self.dcc_alpha - self.dcc_beta) * self.uncond_corr_
        if self.kind == 'adcc' and self.neg_outer_mean_ is not None and self.adcc_gamma > 0.0:
            n = np.minimum(z, 0.0)
            intercept = intercept - self.adcc_gamma * self.neg_outer_mean_
            self.last_q_ = intercept + self.dcc_alpha * np.outer(z, z) + self.dcc_beta * self.last_q_ + self.adcc_gamma * np.outer(n, n)
        else:
            self.last_q_ = intercept + self.dcc_alpha * np.outer(z, z) + self.dcc_beta * self.last_q_
        self.last_q_ = _make_psd(self.last_q_, floor=self.variance_floor)

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
        new_q = (1.0 - self.dcc_alpha - self.dcc_beta) * uncond + self.dcc_alpha * np.outer(z, z) + self.dcc_beta * q
        new_q = _make_psd(new_q, floor=self.variance_floor)
        if regime == 'high':
            self.last_var_high_ = new_var
            self.last_q_high_ = new_q
        else:
            self.last_var_low_ = new_var
            self.last_q_low_ = new_q

    def fit(self, return_resid: np.ndarray, state_innov: np.ndarray) -> 'DynamicCrossCovariance':
        eps = np.asarray(return_resid, dtype=float)
        u = np.asarray(state_innov, dtype=float)
        if eps.ndim != 2 or u.ndim != 2 or len(eps) != len(u):
            raise ValueError('return_resid and state_innov must be 2d arrays with equal row count')
        joint = np.concatenate([eps, u], axis=1)
        if joint.shape[1] != self.n_assets + self.n_states:
            raise ValueError('joint residual width does not match n_assets + n_states')
        self.center_ = joint.mean(axis=0)
        centered = joint - self.center_
        sample_cov = np.cov(centered.T, ddof=1) if len(centered) > 1 else np.diag(np.maximum(centered[0] ** 2, self.variance_floor))
        if np.ndim(sample_cov) == 0:
            sample_cov = np.array([[float(sample_cov)]], dtype=float)
        sample_cov = _make_psd(np.asarray(sample_cov, dtype=float), floor=self.variance_floor)
        sample_var = np.maximum(np.diag(sample_cov), self.variance_floor)

        if self.kind in {'dcc', 'adcc'}:
            _, sample_corr = _corr_from_cov(sample_cov, variance_floor=self.variance_floor)
            if self.correlation_shrink > 0.0:
                sample_corr = (1.0 - self.correlation_shrink) * sample_corr + self.correlation_shrink * np.eye(sample_corr.shape[0])
            self.uncond_corr_ = _make_psd(sample_corr, floor=self.variance_floor)
            self.last_var_ = sample_var.copy()
            self.last_q_ = self.uncond_corr_.copy()
            if self.kind == 'adcc':
                std = np.sqrt(sample_var)
                z_hist = np.divide(centered, std, out=np.zeros_like(centered, dtype=float), where=std > 0)
                n_hist = np.minimum(z_hist, 0.0)
                if len(n_hist) == 0:
                    self.neg_outer_mean_ = np.zeros_like(self.uncond_corr_)
                else:
                    neg_outer = np.einsum('ti,tj->ij', n_hist, n_hist) / float(len(n_hist))
                    self.neg_outer_mean_ = _make_psd(np.asarray(neg_outer, dtype=float), floor=self.variance_floor)
            for resid in centered:
                self._update_dcc_state(resid)
            return self

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
        cov_low = _make_psd((centered[low_mask].T @ centered[low_mask]) / float(max(int(low_mask.sum()) - 1, 1)), floor=self.variance_floor)
        cov_high = _make_psd((centered[high_mask].T @ centered[high_mask]) / float(max(int(high_mask.sum()) - 1, 1)), floor=self.variance_floor)
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

    def current_cross_covariance(self) -> np.ndarray:
        if self.kind in {'dcc', 'adcc'}:
            cov = self._joint_cov_from_state()
            return self._asset_state_block(cov)
        w_high = self._regime_weight()
        cov_low = self._asset_state_block(np.diag(np.sqrt(np.maximum(self.last_var_low_, self.variance_floor))) @ _corr_from_q(self.last_q_low_, variance_floor=self.variance_floor, correlation_shrink=self.correlation_shrink) @ np.diag(np.sqrt(np.maximum(self.last_var_low_, self.variance_floor))))
        cov_high = self._asset_state_block(np.diag(np.sqrt(np.maximum(self.last_var_high_, self.variance_floor))) @ _corr_from_q(self.last_q_high_, variance_floor=self.variance_floor, correlation_shrink=self.correlation_shrink) @ np.diag(np.sqrt(np.maximum(self.last_var_high_, self.variance_floor))))
        return (1.0 - w_high) * cov_low + w_high * cov_high

    def update_with_realized(
        self,
        *,
        realized_return: np.ndarray,
        predicted_return_mean: np.ndarray,
        realized_state: np.ndarray,
        predicted_state: np.ndarray,
    ) -> None:
        eps = np.asarray(realized_return, dtype=float) - np.asarray(predicted_return_mean, dtype=float)
        u = np.asarray(realized_state, dtype=float) - np.asarray(predicted_state, dtype=float)
        resid = np.concatenate([eps, u], axis=0) - self.center_
        if self.kind in {'dcc', 'adcc'}:
            self._update_dcc_state(resid)
            return
        ref_var = (1.0 - self._regime_weight()) * self.last_var_low_ + self._regime_weight() * self.last_var_high_
        energy = self._energy_score(resid, ref_var)
        score_prev = float(self.regime_score_ if self.regime_score_ is not None else energy)
        self.regime_score_ = self.regime_smoothing * score_prev + (1.0 - self.regime_smoothing) * energy
        regime = 'high' if self._regime_weight() >= 0.5 else 'low'
        self._update_regime_state(resid, regime=regime)

@dataclass
class StateTransitionResult:
    coef: np.ndarray
    columns: list[str]
    targets: list[str]

    def predict(self, state_row: pd.Series) -> pd.Series:
        x = np.concatenate([[1.0], state_row[self.columns].to_numpy(dtype=float)])
        pred = x @ self.coef
        return pd.Series(pred, index=self.targets)


@dataclass
class CrossCovarianceEstimate:
    cross: pd.DataFrame
    return_resid_cov: np.ndarray
    state_innov_cov: np.ndarray
    dynamic_model: DynamicCrossCovariance | None = None


def fit_state_transition(states_t: pd.DataFrame, states_tp1: pd.DataFrame, ridge_lambda: float = 1.0e-6) -> StateTransitionResult:
    if len(states_t) != len(states_tp1):
        raise ValueError('states_t and states_tp1 must have equal length.')
    X0 = states_t.copy()
    Y = states_tp1.to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X0)), X0.to_numpy(dtype=float)])
    XtX = X.T @ X
    reg = ridge_lambda * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(XtX + reg, X.T @ Y)
    return StateTransitionResult(coef=beta, columns=list(X0.columns), targets=list(states_tp1.columns))


def estimate_return_state_cross(
    returns_tp1: pd.DataFrame,
    returns_mean_pred: np.ndarray,
    states_t: pd.DataFrame,
    states_tp1: pd.DataFrame,
    transition: StateTransitionResult,
    dynamic_cross_kind: str | None = None,
    variance_floor: float = 1.0e-6,
    correlation_shrink: float = 0.10,
    dcc_alpha: float = 0.02,
    dcc_beta: float = 0.97,
    adcc_gamma: float = 0.005,
    variance_lambda: float = 0.97,
    regime_threshold_quantile: float = 0.75,
    regime_smoothing: float = 0.90,
    regime_sharpness: float = 8.0,
) -> CrossCovarianceEstimate:
    if len(returns_tp1) != len(states_t) or len(states_t) != len(states_tp1):
        raise ValueError('returns_tp1, states_t, states_tp1 must have equal length.')

    pred_states = np.vstack([transition.predict(states_t.iloc[i]).to_numpy(dtype=float) for i in range(len(states_t))])
    eps = returns_tp1.to_numpy(dtype=float) - returns_mean_pred
    u = states_tp1.to_numpy(dtype=float) - pred_states
    denom = max(len(states_t) - 1, 1)
    cross = (eps.T @ u) / denom
    return_resid_cov = (eps.T @ eps) / denom
    state_innov_cov = (u.T @ u) / denom
    dynamic_model: DynamicCrossCovariance | None = None
    if dynamic_cross_kind is not None:
        dynamic_model = DynamicCrossCovariance(
            n_assets=eps.shape[1],
            n_states=u.shape[1],
            kind=str(dynamic_cross_kind),
            variance_floor=variance_floor,
            correlation_shrink=correlation_shrink,
            dcc_alpha=dcc_alpha,
            dcc_beta=dcc_beta,
            adcc_gamma=adcc_gamma,
            variance_lambda=variance_lambda,
            regime_threshold_quantile=regime_threshold_quantile,
            regime_smoothing=regime_smoothing,
            regime_sharpness=regime_sharpness,
        ).fit(eps, u)
    return CrossCovarianceEstimate(
        cross=pd.DataFrame(cross, index=returns_tp1.columns, columns=states_t.columns),
        return_resid_cov=return_resid_cov,
        state_innov_cov=state_innov_cov,
        dynamic_model=dynamic_model,
    )
