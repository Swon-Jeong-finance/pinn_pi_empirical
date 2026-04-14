from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .utils import project_capped_simplex


@dataclass
class PolicyConfigLite:
    risk_aversion: float
    risky_cap: float
    long_only: bool
    pgd_steps: int
    step_size: float
    turnover_penalty: float


def _sanitize_covariance(cov: np.ndarray, *, ridge: float = 1.0e-8) -> np.ndarray:
    mat = np.asarray(cov, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError('covariance matrix must be square')
    mat = 0.5 * (mat + mat.T)
    diag = np.diag(mat).copy()
    diag = np.where(np.isfinite(diag), np.maximum(diag, ridge), ridge)
    np.fill_diagonal(mat, diag)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, ridge, None)
    return (vecs * vals) @ vecs.T


def solve_mean_variance(mu: np.ndarray, cov: np.ndarray, gamma: float, risky_cap: float, steps: int = 300, step_size: float = 0.05) -> np.ndarray:
    n = len(mu)
    w = np.zeros(n, dtype=float)
    for _ in range(steps):
        grad = mu - gamma * (cov @ w)
        w = project_capped_simplex(w + step_size * grad, risky_cap)
    return w


def solve_projected(mu: np.ndarray, cov: np.ndarray, base_w: np.ndarray, prev_w: np.ndarray, gamma: float, risky_cap: float, turnover_penalty: float, steps: int = 150, step_size: float = 0.05) -> np.ndarray:
    w = base_w.copy()
    for _ in range(steps):
        grad = mu - gamma * (cov @ w) - turnover_penalty * (w - prev_w)
        w = project_capped_simplex(w + step_size * grad, risky_cap)
    return w


def solve_equal_weight(n_assets: int, risky_cap: float) -> np.ndarray:
    if n_assets <= 0 or risky_cap <= 0.0:
        return np.zeros(max(int(n_assets), 0), dtype=float)
    return np.full(int(n_assets), float(risky_cap) / float(n_assets), dtype=float)


def solve_min_variance(cov: np.ndarray, risky_cap: float, *, steps: int = 400, step_size: float | None = None) -> np.ndarray:
    mat = _sanitize_covariance(cov)
    n = mat.shape[0]
    w = solve_equal_weight(n, risky_cap)
    if n == 0 or risky_cap <= 0.0:
        return w
    lipschitz = float(np.max(np.abs(mat).sum(axis=1)))
    eta = float(step_size) if step_size is not None else (1.0 / max(lipschitz, 1.0e-8))
    eta = max(min(eta, 1.0), 1.0e-4)
    for _ in range(max(int(steps), 1)):
        grad = mat @ w
        w = project_capped_simplex(w - eta * grad, risky_cap)
    return w


def solve_risk_parity(cov: np.ndarray, risky_cap: float, *, steps: int = 500, tolerance: float = 1.0e-8) -> np.ndarray:
    mat = _sanitize_covariance(cov)
    n = mat.shape[0]
    w = solve_equal_weight(n, risky_cap)
    if n == 0 or risky_cap <= 0.0:
        return w
    for _ in range(max(int(steps), 1)):
        sigma_w = mat @ w
        total_var = float(w @ sigma_w)
        if (not np.isfinite(total_var)) or total_var <= 1.0e-12:
            return solve_equal_weight(n, risky_cap)
        target_rc = total_var / float(n)
        rc = w * sigma_w
        gap = np.max(np.abs(rc - target_rc)) if len(rc) else 0.0
        if gap <= tolerance:
            break
        scale = np.sqrt(np.clip(target_rc / np.maximum(rc, 1.0e-12), 0.25, 4.0))
        w = project_capped_simplex(w * scale, risky_cap)
        if float(w.sum()) <= 1.0e-12:
            w = solve_equal_weight(n, risky_cap)
    return w


def compute_weights(mu: pd.Series, cov_full: np.ndarray, policy_cfg) -> dict[str, np.ndarray]:
    mu_arr = mu.to_numpy(dtype=float)
    cov_diag = np.diag(np.diag(cov_full))
    base_w = solve_mean_variance(mu_arr, cov_diag, policy_cfg.risk_aversion, policy_cfg.risky_cap, steps=max(250, policy_cfg.pgd_steps), step_size=policy_cfg.step_size)
    out = {
        'myopic_diag': base_w,
        'myopic_full': solve_mean_variance(mu_arr, cov_full, policy_cfg.risk_aversion, policy_cfg.risky_cap, steps=max(250, policy_cfg.pgd_steps), step_size=policy_cfg.step_size),
    }
    return out
