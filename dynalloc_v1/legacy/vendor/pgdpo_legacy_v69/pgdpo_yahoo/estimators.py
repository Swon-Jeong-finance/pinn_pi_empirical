"""
Estimation for a model-based, differentiable simulator.

We fit (on a rolling window):
- Conditional mean of excess returns:
    r_{t+1}^ex = a + B y_t + eps_t
- Covariance of eps_t using Ledoit-Wolf shrinkage
- Factor VAR(1):
    y_{t+1} = c + A y_t + u_t
  mapped to OU-ish form (approx):
    dY = K(θ - Y) dt + diag(σY) dW^Y

We also estimate cross-cov between asset shocks and factor shocks to form:
    cross = sigma * rho * sigmaY
which is what enters the PMP hedging term.

This module returns a MarketModel object used by the simulator & projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

from .utils import safe_cholesky


@dataclass
class MarketModel:
    asset_tickers: list[str]
    state_names: list[str]
    periods_per_year: int

    # Risk-free (annualized, decimal)
    r_annual: float

    # Excess drift model: mu_excess_annual(y) = a + B y  (annualized)
    a: np.ndarray          # (n,)
    B: np.ndarray          # (n, k)

    # Return covariance (annualized)
    Sigma: np.ndarray      # (n, n)
    sigma: np.ndarray      # (n, n) lower cholesky of Sigma

    # Factor OU parameters (annualized)
    K: np.ndarray          # (k, k)
    theta: np.ndarray      # (k,)
    sigmaY_diag: np.ndarray  # (k,)

    # cross-cov term (instantaneous, annualized): sigma * rho * sigmaY  (n, k)
    cross: np.ndarray      # (n, k)

    def mu_excess_annual(self, y: np.ndarray) -> np.ndarray:
        """
        y: (..., k)
        returns: (..., n)
        """
        y2 = np.atleast_2d(y)
        out = self.a[None, :] + y2 @ self.B.T
        return out.squeeze(0) if y.ndim == 1 else out


def fit_linear_conditional_mean(
    r_excess: np.ndarray,   # (T, n) per-step excess returns (log or simple) 
    y: np.ndarray,          # (T, k) states
    dt_years: float,
    ridge_alpha: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit r_excess_step ≈ a_step + B_step y_t by ridge regression, then convert to annualized:
        mu_excess_annual(y) ≈ a_annual + B_annual y

    Returns:
      a_annual (n,), B_annual (n,k), residuals_step (T,n)
    """
    T, n = r_excess.shape
    k = y.shape[1]
    X = np.hstack([np.ones((T, 1)), y])
    # Ridge on each asset jointly (multi-target)
    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(X, r_excess)  # multi-target
    coef = model.coef_      # (n, 1+k)
    a_step = coef[:, 0]
    B_step = coef[:, 1:]
    pred = X @ coef.T
    resid = r_excess - pred

    # annualize
    a_annual = a_step / dt_years
    B_annual = B_step / dt_years
    return a_annual, B_annual, resid


def fit_shrinkage_cov(resid_step: np.ndarray, dt_years: float) -> np.ndarray:
    """
    Ledoit–Wolf shrinkage covariance, then annualize by /dt.
    """
    lw = LedoitWolf().fit(resid_step)
    cov_step = lw.covariance_
    return cov_step / dt_years


def fit_var1(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    VAR(1): y_{t+1} = c + A y_t + u_t
    returns c (k,), A (k,k), residuals (T-1,k)
    """
    y0 = y[:-1]
    y1 = y[1:]
    Tm1, k = y0.shape
    X = np.hstack([np.ones((Tm1, 1)), y0])
    # OLS: solve for each dimension
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]  # (1+k, k)
    c = beta[0, :]
    A = beta[1:, :].T   # (k,k) so that y1 ≈ c + A y0
    pred = (X @ beta)
    resid = y1 - pred
    return c, A, resid


def var1_to_ou(c: np.ndarray, A: np.ndarray, resid: np.ndarray, dt_years: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate mapping:
      y_{t+dt} = c + A y_t + eps
    to OU form:
      dY = K(θ - Y) dt + diag(sigmaY) dW

    For small dt, use:
      K ≈ (I - A)/dt
      θ ≈ (I - A)^{-1} c
      sigmaY_diag ≈ std(eps)/sqrt(dt)   (diagonal diffusion)
    """
    k = len(c)
    I = np.eye(k)
    # K
    K = (I - A) / dt_years
    # theta
    try:
        theta = np.linalg.solve(I - A, c)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(I - A) @ c

    # diagonal sigmaY from residuals
    std_step = resid.std(axis=0, ddof=1)
    sigmaY_diag = std_step / np.sqrt(dt_years)
    return K, theta, sigmaY_diag


def estimate_cross_term(
    resid_r_step: np.ndarray,   # (T, n) residuals of excess returns (per-step)
    resid_y_step: np.ndarray,   # (T, k) residuals of VAR for factors (per-step)
    Sigma_annual: np.ndarray,   # (n, n)
    sigmaY_diag: np.ndarray,    # (k,)
    dt_years: float,
    clip_rho: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate cross = sigma * rho * sigmaY (annualized) and rho.

    Steps:
      cov_step = Cov(resid_r_step, resid_y_step)  (n,k) per step
      cross_annual = cov_step / dt
      sigma = chol(Sigma_annual)
      rho = sigma^{-1} cross_annual diag(sigmaY)^{-1}

    We clip rho to keep the joint Brownian covariance PSD-ish.
    """
    # covariance per step
    cov_step = np.cov(resid_r_step.T, resid_y_step.T, ddof=1)
    n = resid_r_step.shape[1]
    k = resid_y_step.shape[1]
    cov_xy_step = cov_step[:n, n:]  # (n,k)

    cross_annual = cov_xy_step / dt_years

    sigma = safe_cholesky(Sigma_annual)
    # solve sigma * M = cross_annual  => M = sigma^{-1} cross
    M = np.linalg.solve(sigma, cross_annual)
    # divide by sigmaY
    sigmaY_safe = np.where(sigmaY_diag == 0.0, 1e-12, sigmaY_diag)
    rho = M / sigmaY_safe[None, :]

    rho = np.clip(rho, -clip_rho, clip_rho)
    # rebuild cross with clipped rho to keep internal consistency
    cross = sigma @ (rho * sigmaY_safe[None, :])
    return cross, rho


def fit_market_model(
    log_returns: pd.DataFrame,
    rf_annual: pd.Series,
    states: pd.DataFrame,
    ridge_alpha: float = 1e-4,
) -> MarketModel:
    """
    Fit everything on a window. Uses per-step log returns.
    """
    assert log_returns.shape[0] == states.shape[0] == rf_annual.shape[0]
    dates = log_returns.index
    n = log_returns.shape[1]
    k = states.shape[1]

    freq = (pd.infer_freq(dates) or "").upper()
    if freq.startswith("W"):
        periods_per_year = 52
    elif "M" in freq:
        # Monthly (M/ME/MS)
        periods_per_year = 12
    elif freq.startswith("Q"):
        periods_per_year = 4
    elif freq.startswith("A") or freq.startswith("Y"):
        periods_per_year = 1
    else:
        periods_per_year = 252
    dt_years = 1.0 / periods_per_year

    # excess returns (per-step): r^ex = r - rf*dt (rf in annual terms)
    rf_step = rf_annual.values * dt_years
    r_step = log_returns.values
    r_ex_step = r_step - rf_step[:, None]

    y = states.values

    # conditional mean
    a_annual, B_annual, resid_r_step = fit_linear_conditional_mean(
        r_excess=r_ex_step,
        y=y,
        dt_years=dt_years,
        ridge_alpha=ridge_alpha,
    )

    # covariance
    Sigma_annual = fit_shrinkage_cov(resid_r_step, dt_years=dt_years)
    sigma = safe_cholesky(Sigma_annual)

    # factor VAR(1) -> OU
    c, A, resid_y = fit_var1(y)
    K, theta, sigmaY_diag = var1_to_ou(c, A, resid_y, dt_years=dt_years)

    # cross term
    # align residual lengths: resid_r_step is T, resid_y is T-1; drop first row in resid_r_step
    resid_r_aligned = resid_r_step[1:]
    cross, rho = estimate_cross_term(
        resid_r_step=resid_r_aligned,
        resid_y_step=resid_y,
        Sigma_annual=Sigma_annual,
        sigmaY_diag=sigmaY_diag,
        dt_years=dt_years,
    )

    # risk-free annual: use last (or mean) on window
    r_annual = float(np.nanmean(rf_annual.values))

    return MarketModel(
        asset_tickers=list(log_returns.columns),
        state_names=list(states.columns),
        periods_per_year=periods_per_year,
        r_annual=r_annual,
        a=a_annual,
        B=B_annual,
        Sigma=Sigma_annual,
        sigma=sigma,
        K=K,
        theta=theta,
        sigmaY_diag=sigmaY_diag,
        cross=cross,
    )
