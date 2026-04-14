from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


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
    return CrossCovarianceEstimate(
        cross=pd.DataFrame(cross, index=returns_tp1.columns, columns=states_t.columns),
        return_resid_cov=return_resid_cov,
        state_innov_cov=state_innov_cov,
    )
