from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import pandas as pd

try:
    from scipy import sparse
    from scipy.sparse.linalg import splu, spsolve
    from scipy.interpolate import RegularGridInterpolator
except Exception as exc:  # pragma: no cover
    sparse = None
    splu = None
    spsolve = None
    RegularGridInterpolator = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

from .mean_model import MeanModelResult
from .transition import CrossCovarianceEstimate, StateTransitionResult
from .ppgdpo import CostateEstimate
from .pipinn_backend import (
    PIPINNEnvFromPPGDPO,
    _select_training_covariance,
    _symmetrize_psd,
    _solve_unconstrained_foc_np,
    _clip_foc_policy_np,
)


@dataclass
class FDMGridSolution:
    """Grid solution for the g-form reduced HJB.

    The solver stores g(tau, z) but exposes grad_log_g = grad_z g / g for
    compatibility with the traditional PINN policy formula.
    """

    value_form: str
    tau_grid: np.ndarray
    z_grids: tuple[np.ndarray, ...]
    g_grid: np.ndarray
    grad_log_g_grids: tuple[np.ndarray, ...]
    diagnostics: dict[str, Any]
    scheme: str
    boundary: str


def _require_scipy() -> None:
    if _SCIPY_IMPORT_ERROR is not None or sparse is None or RegularGridInterpolator is None:
        raise ImportError(
            "FDM backend requires scipy.sparse and scipy.interpolate. "
            f"Import error: {_SCIPY_IMPORT_ERROR!r}"
        )


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _fdm_cfg(cfg: Any) -> Any:
    return getattr(cfg, 'fdm', None)


def _grid_count(fdm_cfg: Any, name: str, default: int) -> int:
    return int(max(int(_cfg_get(fdm_cfg, name, default) or default), 3))


def _flat_index(i: int, j: int | None, n1: int, n2: int | None) -> int:
    if n2 is None:
        return int(i)
    return int(i) * int(n2) + int(j)


def _reflect_idx(k: int, n: int) -> int:
    if n <= 1:
        return 0
    if k < 0:
        return min(-k, n - 1)
    if k >= n:
        return max(2 * (n - 1) - k, 0)
    return k


def _add(rows: list[int], cols: list[int], data: list[float], row: int, col: int, val: float) -> None:
    if val == 0.0 or not np.isfinite(val):
        return
    rows.append(int(row))
    cols.append(int(col))
    data.append(float(val))


def _state_points_from_grids(z_grids: tuple[np.ndarray, ...]) -> np.ndarray:
    if len(z_grids) == 1:
        return z_grids[0].reshape(-1, 1)
    z1, z2 = z_grids
    xx, yy = np.meshgrid(z1, z2, indexing='ij')
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def _state_drift(env: PIPINNEnvFromPPGDPO, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return env.drift_intercept.reshape(1, -1) + x @ env.drift_matrix.T


def _state_mu(env: PIPINNEnvFromPPGDPO, x: np.ndarray) -> np.ndarray:
    return env.mean_map.predict_batch(np.asarray(x, dtype=float))


def _build_linear_operator_1d(env: PIPINNEnvFromPPGDPO, z: np.ndarray, *, drift_scheme: str = 'upwind'):
    _require_scipy()
    n = int(len(z))
    if n < 3:
        raise ValueError('FDM 1D grid requires at least three points')
    dz = float(z[1] - z[0])
    if dz <= 0:
        raise ValueError('FDM z grid must be strictly increasing')
    q11 = float(env.Q[0, 0])
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    x = z.reshape(-1, 1)
    drift = _state_drift(env, x).reshape(-1)
    for i in range(n):
        row = _flat_index(i, None, n, None)
        b = float(drift[i])
        # Drift first derivative.
        if str(drift_scheme).lower() == 'central':
            if 0 < i < n - 1:
                _add(rows, cols, data, row, i - 1, -0.5 * b / dz)
                _add(rows, cols, data, row, i + 1, 0.5 * b / dz)
        else:  # upwind with zero-Neumann at boundaries
            if b >= 0.0:
                if i > 0:
                    _add(rows, cols, data, row, i, b / dz)
                    _add(rows, cols, data, row, i - 1, -b / dz)
            else:
                if i < n - 1:
                    _add(rows, cols, data, row, i + 1, b / dz)
                    _add(rows, cols, data, row, i, -b / dz)
        # Diffusion second derivative with reflecting ghost points.
        coef = 0.5 * q11
        if i == 0:
            _add(rows, cols, data, row, i, coef * (-2.0) / (dz * dz))
            _add(rows, cols, data, row, i + 1, coef * 2.0 / (dz * dz))
        elif i == n - 1:
            _add(rows, cols, data, row, i - 1, coef * 2.0 / (dz * dz))
            _add(rows, cols, data, row, i, coef * (-2.0) / (dz * dz))
        else:
            _add(rows, cols, data, row, i - 1, coef / (dz * dz))
            _add(rows, cols, data, row, i, coef * (-2.0) / (dz * dz))
            _add(rows, cols, data, row, i + 1, coef / (dz * dz))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def _build_linear_operator_2d(env: PIPINNEnvFromPPGDPO, z1: np.ndarray, z2: np.ndarray, *, drift_scheme: str = 'upwind'):
    _require_scipy()
    n1, n2 = int(len(z1)), int(len(z2))
    if n1 < 3 or n2 < 3:
        raise ValueError('FDM 2D grid requires at least three points in each state direction')
    dz1 = float(z1[1] - z1[0])
    dz2 = float(z2[1] - z2[0])
    if dz1 <= 0 or dz2 <= 0:
        raise ValueError('FDM z grids must be strictly increasing')
    q11 = float(env.Q[0, 0])
    q22 = float(env.Q[1, 1])
    q12 = float(env.Q[0, 1])
    points = _state_points_from_grids((z1, z2))
    drift = _state_drift(env, points).reshape(n1, n2, 2)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(n1):
        for j in range(n2):
            row = _flat_index(i, j, n1, n2)
            b1 = float(drift[i, j, 0])
            b2 = float(drift[i, j, 1])
            # Drift in z1.
            if str(drift_scheme).lower() == 'central':
                if 0 < i < n1 - 1:
                    _add(rows, cols, data, row, _flat_index(i - 1, j, n1, n2), -0.5 * b1 / dz1)
                    _add(rows, cols, data, row, _flat_index(i + 1, j, n1, n2), 0.5 * b1 / dz1)
                if 0 < j < n2 - 1:
                    _add(rows, cols, data, row, _flat_index(i, j - 1, n1, n2), -0.5 * b2 / dz2)
                    _add(rows, cols, data, row, _flat_index(i, j + 1, n1, n2), 0.5 * b2 / dz2)
            else:
                if b1 >= 0.0:
                    if i > 0:
                        _add(rows, cols, data, row, _flat_index(i, j, n1, n2), b1 / dz1)
                        _add(rows, cols, data, row, _flat_index(i - 1, j, n1, n2), -b1 / dz1)
                else:
                    if i < n1 - 1:
                        _add(rows, cols, data, row, _flat_index(i + 1, j, n1, n2), b1 / dz1)
                        _add(rows, cols, data, row, _flat_index(i, j, n1, n2), -b1 / dz1)
                if b2 >= 0.0:
                    if j > 0:
                        _add(rows, cols, data, row, _flat_index(i, j, n1, n2), b2 / dz2)
                        _add(rows, cols, data, row, _flat_index(i, j - 1, n1, n2), -b2 / dz2)
                else:
                    if j < n2 - 1:
                        _add(rows, cols, data, row, _flat_index(i, j + 1, n1, n2), b2 / dz2)
                        _add(rows, cols, data, row, _flat_index(i, j, n1, n2), -b2 / dz2)
            # Diffusion z1-z1.
            coef11 = 0.5 * q11
            if i == 0:
                _add(rows, cols, data, row, _flat_index(i, j, n1, n2), coef11 * (-2.0) / (dz1 * dz1))
                _add(rows, cols, data, row, _flat_index(i + 1, j, n1, n2), coef11 * 2.0 / (dz1 * dz1))
            elif i == n1 - 1:
                _add(rows, cols, data, row, _flat_index(i - 1, j, n1, n2), coef11 * 2.0 / (dz1 * dz1))
                _add(rows, cols, data, row, _flat_index(i, j, n1, n2), coef11 * (-2.0) / (dz1 * dz1))
            else:
                _add(rows, cols, data, row, _flat_index(i - 1, j, n1, n2), coef11 / (dz1 * dz1))
                _add(rows, cols, data, row, _flat_index(i, j, n1, n2), coef11 * (-2.0) / (dz1 * dz1))
                _add(rows, cols, data, row, _flat_index(i + 1, j, n1, n2), coef11 / (dz1 * dz1))
            # Diffusion z2-z2.
            coef22 = 0.5 * q22
            if j == 0:
                _add(rows, cols, data, row, _flat_index(i, j, n1, n2), coef22 * (-2.0) / (dz2 * dz2))
                _add(rows, cols, data, row, _flat_index(i, j + 1, n1, n2), coef22 * 2.0 / (dz2 * dz2))
            elif j == n2 - 1:
                _add(rows, cols, data, row, _flat_index(i, j - 1, n1, n2), coef22 * 2.0 / (dz2 * dz2))
                _add(rows, cols, data, row, _flat_index(i, j, n1, n2), coef22 * (-2.0) / (dz2 * dz2))
            else:
                _add(rows, cols, data, row, _flat_index(i, j - 1, n1, n2), coef22 / (dz2 * dz2))
                _add(rows, cols, data, row, _flat_index(i, j, n1, n2), coef22 * (-2.0) / (dz2 * dz2))
                _add(rows, cols, data, row, _flat_index(i, j + 1, n1, n2), coef22 / (dz2 * dz2))
            # Mixed derivative Q12 * g_z1z2 with reflected ghost points.
            if abs(q12) > 1.0e-16:
                for si, sj, sign in ((1, 1, 1.0), (1, -1, -1.0), (-1, 1, -1.0), (-1, -1, 1.0)):
                    ii = _reflect_idx(i + si, n1)
                    jj = _reflect_idx(j + sj, n2)
                    _add(rows, cols, data, row, _flat_index(ii, jj, n1, n2), q12 * sign / (4.0 * dz1 * dz2))
    N = n1 * n2
    return sparse.csr_matrix((data, (rows, cols)), shape=(N, N))


def _grad_g_1d(g: np.ndarray, z: np.ndarray) -> tuple[np.ndarray]:
    dz = float(z[1] - z[0])
    grad = np.zeros_like(g, dtype=float)
    if len(z) > 2:
        grad[..., 1:-1] = (g[..., 2:] - g[..., :-2]) / (2.0 * dz)
    # zero-Neumann at boundaries
    grad[..., 0] = 0.0
    grad[..., -1] = 0.0
    return (grad,)


def _grad_g_2d(g: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dz1 = float(z1[1] - z1[0])
    dz2 = float(z2[1] - z2[0])
    g1 = np.zeros_like(g, dtype=float)
    g2 = np.zeros_like(g, dtype=float)
    if len(z1) > 2:
        g1[:, 1:-1, :] = (g[:, 2:, :] - g[:, :-2, :]) / (2.0 * dz1)
    if len(z2) > 2:
        g2[:, :, 1:-1] = (g[:, :, 2:] - g[:, :, :-2]) / (2.0 * dz2)
    g1[:, 0, :] = 0.0
    g1[:, -1, :] = 0.0
    g2[:, :, 0] = 0.0
    g2[:, :, -1] = 0.0
    return g1, g2


def _grad_single_1d(g: np.ndarray, z: np.ndarray) -> tuple[np.ndarray]:
    return (_grad_g_1d(g.reshape(1, -1), z)[0].reshape(-1),)


def _grad_single_2d(g: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g1, g2 = _grad_g_2d(g.reshape(1, len(z1), len(z2)), z1, z2)
    return g1.reshape(len(z1), len(z2)), g2.reshape(len(z1), len(z2))


def _source_term_g(
    env: PIPINNEnvFromPPGDPO,
    g_flat: np.ndarray,
    z_grids: tuple[np.ndarray, ...],
    *,
    g_floor: float,
) -> np.ndarray:
    g_flat = np.asarray(g_flat, dtype=float).reshape(-1)
    g_safe_flat = np.maximum(g_flat, float(g_floor))
    points = _state_points_from_grids(z_grids)
    mu = _state_mu(env, points)
    if len(z_grids) == 1:
        z = z_grids[0]
        grad = _grad_single_1d(g_flat.reshape(len(z)), z)[0].reshape(-1, 1)
    else:
        z1, z2 = z_grids
        g2d = g_flat.reshape(len(z1), len(z2))
        gg1, gg2 = _grad_single_2d(g2d, z1, z2)
        grad = np.column_stack([gg1.reshape(-1), gg2.reshape(-1)])
    q = grad / g_safe_flat.reshape(-1, 1)
    a = mu + q @ env.C_train.T
    try:
        sig_inv_a = np.linalg.solve(env.Sigma_train, a.T).T
    except np.linalg.LinAlgError:
        sig_inv_a = np.linalg.lstsq(env.Sigma_train, a.T, rcond=None)[0].T
    ham = 0.5 / float(env.gamma) * np.sum(a * sig_inv_a, axis=1)
    r = float(getattr(env, 'r', 0.0))
    src = (1.0 - float(env.gamma)) * g_flat * (r + ham)
    return np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_grad_log_g_grid(g_grid: np.ndarray, z_grids: tuple[np.ndarray, ...], *, g_floor: float) -> tuple[np.ndarray, ...]:
    g_safe = np.maximum(np.asarray(g_grid, dtype=float), float(g_floor))
    if len(z_grids) == 1:
        grad_g = _grad_g_1d(g_grid, z_grids[0])[0]
        return (np.nan_to_num(grad_g / g_safe, nan=0.0, posinf=0.0, neginf=0.0),)
    grad1, grad2 = _grad_g_2d(g_grid, z_grids[0], z_grids[1])
    return (
        np.nan_to_num(grad1 / g_safe, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(grad2 / g_safe, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _boundary_grad_max(grad_log_grids: tuple[np.ndarray, ...]) -> float:
    if not grad_log_grids:
        return 0.0
    masks: list[np.ndarray] = []
    sample = grad_log_grids[0]
    if sample.ndim == 2:  # tau, z1
        mask = np.zeros_like(sample, dtype=bool)
        mask[:, 0] = True
        mask[:, -1] = True
        masks.append(mask)
    elif sample.ndim == 3:
        mask = np.zeros_like(sample, dtype=bool)
        mask[:, 0, :] = True
        mask[:, -1, :] = True
        mask[:, :, 0] = True
        mask[:, :, -1] = True
        masks.append(mask)
    if not masks:
        return 0.0
    m = masks[0]
    vals = [np.asarray(g)[m] for g in grad_log_grids]
    if not vals:
        return 0.0
    return float(np.nanmax(np.abs(np.concatenate([v.reshape(-1) for v in vals]))))


def _residual_l2_grid(
    env: PIPINNEnvFromPPGDPO,
    g_grid: np.ndarray,
    z_grids: tuple[np.ndarray, ...],
    L,
    dtau: float,
    *,
    g_floor: float,
) -> float:
    if g_grid.shape[0] < 2:
        return float('nan')
    vals: list[float] = []
    for n in range(g_grid.shape[0] - 1):
        g_n = g_grid[n].reshape(-1)
        g_np1 = g_grid[n + 1].reshape(-1)
        src = _source_term_g(env, g_n, z_grids, g_floor=g_floor)
        res = (g_np1 - g_n) / float(dtau) - (L @ g_np1 + src)
        vals.append(float(np.nanmean(res * res)))
    return float(np.sqrt(np.nanmean(vals))) if vals else float('nan')


def solve_g_hjb_fdm(env: PIPINNEnvFromPPGDPO, fdm_cfg: Any | None = None) -> FDMGridSolution:
    """Solve the unconstrained traditional-PINN HJB in g-form by FDM.

    The solved PDE is
        g_tau = b' grad g + 0.5 tr(Q D2 g)
                + (1-gamma) g [ r + 0.5/gamma * a' Sigma^{-1} a ],
        a = mu(z) + C grad(log g).
    """
    _require_scipy()
    start = time.perf_counter()
    n_states = int(env.n_states)
    max_state_dim = int(_cfg_get(fdm_cfg, 'max_state_dim', 2) or 2)
    if n_states < 1 or n_states > max_state_dim or n_states > 2:
        raise ValueError(
            f"FDM backend supports one or two state variables; got n_states={n_states}."
        )
    value_form = str(_cfg_get(fdm_cfg, 'value_form', 'g')).lower()
    if value_form != 'g':
        raise ValueError("FDM backend currently solves the g-form HJB only; set fdm.value_form='g'.")
    scheme = str(_cfg_get(fdm_cfg, 'scheme', 'imex')).lower()
    if scheme not in {'imex', 'imex_picard'}:
        raise ValueError("fdm.scheme must be 'imex' or 'imex_picard'.")
    boundary = str(_cfg_get(fdm_cfg, 'boundary', 'neumann')).lower()
    if boundary != 'neumann':
        raise ValueError("FDM backend currently supports boundary='neumann' only.")
    drift_scheme = str(_cfg_get(fdm_cfg, 'drift_scheme', 'upwind')).lower()
    g_floor = float(_cfg_get(fdm_cfg, 'g_floor', 1.0e-10) or 1.0e-10)
    enforce_positive = bool(_cfg_get(fdm_cfg, 'enforce_positive', True))
    picard_iters = int(_cfg_get(fdm_cfg, 'picard_iters', 5) or 5)
    picard_tol = float(_cfg_get(fdm_cfg, 'picard_tol', 1.0e-8) or 1.0e-8)

    n_tau = int(max(int(_cfg_get(fdm_cfg, 'n_tau', 240) or 240), 1))
    tau_grid = np.linspace(0.0, float(env.tau_max), n_tau + 1, dtype=float)
    dtau = float(tau_grid[1] - tau_grid[0]) if len(tau_grid) > 1 else float(env.tau_max)
    x_min = np.asarray(env.x_min, dtype=float).reshape(-1)
    x_max = np.asarray(env.x_max, dtype=float).reshape(-1)
    if n_states == 1:
        n_z1 = _grid_count(fdm_cfg, 'n_z1', 81)
        z1 = np.linspace(float(x_min[0]), float(x_max[0]), n_z1, dtype=float)
        z_grids = (z1,)
        L = _build_linear_operator_1d(env, z1, drift_scheme=drift_scheme)
        shape_space = (n_z1,)
    else:
        n_z1 = _grid_count(fdm_cfg, 'n_z1', 81)
        n_z2 = _grid_count(fdm_cfg, 'n_z2', 81)
        z1 = np.linspace(float(x_min[0]), float(x_max[0]), n_z1, dtype=float)
        z2 = np.linspace(float(x_min[1]), float(x_max[1]), n_z2, dtype=float)
        z_grids = (z1, z2)
        L = _build_linear_operator_2d(env, z1, z2, drift_scheme=drift_scheme)
        shape_space = (n_z1, n_z2)

    N = int(np.prod(shape_space))
    eye = sparse.eye(N, format='csr')
    A = (eye - float(dtau) * L).tocsc()
    lu = None
    try:
        lu = splu(A)
    except Exception:
        lu = None

    def solve_A(rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        if lu is not None:
            return lu.solve(rhs)
        return spsolve(A, rhs)

    g_grid = np.empty((n_tau + 1, *shape_space), dtype=float)
    g_flat = np.ones(N, dtype=float)
    g_grid[0] = g_flat.reshape(shape_space)
    picard_counts: list[int] = []

    for n in range(n_tau):
        if scheme == 'imex':
            src = _source_term_g(env, g_flat, z_grids, g_floor=g_floor)
            rhs = g_flat + float(dtau) * src
            g_new = solve_A(rhs)
            picard_counts.append(0)
        else:
            guess = g_flat.copy()
            count = 0
            for k in range(max(picard_iters, 1)):
                src = _source_term_g(env, guess, z_grids, g_floor=g_floor)
                rhs = g_flat + float(dtau) * src
                cand = solve_A(rhs)
                diff = float(np.nanmax(np.abs(cand - guess)))
                guess = cand
                count = k + 1
                if diff < picard_tol:
                    break
            g_new = guess
            picard_counts.append(count)
        g_new = np.nan_to_num(g_new, nan=g_floor, posinf=g_floor, neginf=g_floor)
        if enforce_positive:
            g_new = np.maximum(g_new, g_floor)
        g_flat = np.asarray(g_new, dtype=float).reshape(-1)
        g_grid[n + 1] = g_flat.reshape(shape_space)

    grad_log = _compute_grad_log_g_grid(g_grid, z_grids, g_floor=g_floor)
    elapsed = time.perf_counter() - start
    q_corr = 0.0
    if n_states >= 2:
        denom = float(np.sqrt(max(float(env.Q[0, 0]) * float(env.Q[1, 1]), 1.0e-24)))
        q_corr = float(abs(float(env.Q[0, 1])) / denom) if denom > 0.0 else 0.0
    diagnostics = {
        'fdm_value_form': 'g',
        'fdm_scheme': scheme,
        'fdm_boundary': boundary,
        'fdm_drift_scheme': drift_scheme,
        'fdm_n_states': int(n_states),
        'fdm_n_z1': int(len(z_grids[0])),
        'fdm_n_z2': int(len(z_grids[1])) if len(z_grids) > 1 else 0,
        'fdm_n_tau': int(n_tau),
        'fdm_dtau': float(dtau),
        'fdm_min_g': float(np.nanmin(g_grid)),
        'fdm_max_g': float(np.nanmax(g_grid)),
        'fdm_max_abs_g': float(np.nanmax(np.abs(g_grid))),
        'fdm_max_abs_grad_log': float(max(np.nanmax(np.abs(g)) for g in grad_log)) if grad_log else 0.0,
        'fdm_max_abs_grad_log_z1': float(np.nanmax(np.abs(grad_log[0]))) if grad_log else 0.0,
        'fdm_max_abs_grad_log_z2': float(np.nanmax(np.abs(grad_log[1]))) if len(grad_log) > 1 else 0.0,
        'fdm_boundary_grad_max': _boundary_grad_max(grad_log),
        'fdm_nan_count': int(np.isnan(g_grid).sum() + sum(np.isnan(g).sum() for g in grad_log)),
        'fdm_inf_count': int(np.isinf(g_grid).sum() + sum(np.isinf(g).sum() for g in grad_log)),
        'fdm_pde_residual_l2_grid': _residual_l2_grid(env, g_grid, z_grids, L, dtau, g_floor=g_floor),
        'fdm_Q_offdiag_corr': q_corr,
        'fdm_state_clip_count': 0,
        'fdm_picard_iters_avg': float(np.mean(picard_counts)) if picard_counts else 0.0,
        'fdm_elapsed_seconds': float(elapsed),
        'fdm_g_floor': float(g_floor),
        'fdm_enforce_positive': bool(enforce_positive),
    }
    return FDMGridSolution(
        value_form='g',
        tau_grid=tau_grid,
        z_grids=z_grids,
        g_grid=g_grid,
        grad_log_g_grids=grad_log,
        diagnostics=diagnostics,
        scheme=scheme,
        boundary=boundary,
    )


class TrainedFDM:
    """Traditional-PINN benchmark with the value PDE solved by FDM.

    The policy/evaluation interface intentionally mirrors TrainedPIPINN. The
    only methodological change relative to the 'pinn' backend is that the value
    gradient is obtained from a finite-difference solution of g(tau, z).
    """

    def __init__(
        self,
        *,
        env: PIPINNEnvFromPPGDPO,
        solution: FDMGridSolution,
        train_objective: float,
        train_seed: int,
        train_history: list[dict[str, Any]] | None = None,
        best_validation_loss: float | None = None,
    ):
        self.env = env
        self.solution = solution
        self.train_objective = float(train_objective)
        self.train_seed = int(train_seed)
        self.state_columns = list(env.state_columns)
        self.asset_columns = list(env.asset_columns)
        self.train_history = list(train_history or [])
        self.best_validation_loss = float(best_validation_loss) if best_validation_loss is not None else float('nan')
        self._state_clip_count = 0
        self._last_state_clipped = False
        self._last_tau_clipped = False
        self._last_z_query = None
        self._last_g_value = float('nan')
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        _require_scipy()
        grids = (self.solution.tau_grid, *self.solution.z_grids)
        self._g_interp = RegularGridInterpolator(grids, self.solution.g_grid, method='linear', bounds_error=False, fill_value=None)
        self._grad_interp = [
            RegularGridInterpolator(grids, arr, method='linear', bounds_error=False, fill_value=None)
            for arr in self.solution.grad_log_g_grids
        ]

    def _state_array(self, state_row: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(state_row, pd.Series):
            arr = state_row[self.state_columns].to_numpy(dtype=float).reshape(-1)
        else:
            arr = np.asarray(state_row, dtype=float).reshape(-1)
        if arr.shape[0] != len(self.state_columns):
            raise ValueError(f'Expected {len(self.state_columns)} state values, got {arr.shape[0]}')
        return arr.astype(float)

    def _clip_query(self, tau: float | None, z: np.ndarray) -> tuple[float, np.ndarray, bool, bool]:
        tau_val = float(self.env.tau_max if tau is None else tau)
        tau_clipped = float(np.clip(tau_val, self.solution.tau_grid[0], self.solution.tau_grid[-1]))
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        mins = np.asarray([grid[0] for grid in self.solution.z_grids], dtype=float)
        maxs = np.asarray([grid[-1] for grid in self.solution.z_grids], dtype=float)
        z_clipped = np.minimum(np.maximum(z_arr, mins), maxs)
        state_clipped = bool(np.any(np.abs(z_clipped - z_arr) > 1.0e-12))
        tau_was_clipped = bool(abs(tau_clipped - tau_val) > 1.0e-12)
        if state_clipped:
            self._state_clip_count += 1
        self._last_state_clipped = state_clipped
        self._last_tau_clipped = tau_was_clipped
        self._last_z_query = z_clipped.copy()
        return tau_clipped, z_clipped, state_clipped, tau_was_clipped

    def grad_u_and_value(self, state_row: pd.Series | np.ndarray, *, tau: float | None = None) -> tuple[np.ndarray, float]:
        z = self._state_array(state_row)
        tau_q, z_q, _, _ = self._clip_query(tau, z)
        point = np.asarray([[tau_q, *z_q]], dtype=float)
        g_val = float(np.asarray(self._g_interp(point)).reshape(-1)[0])
        grad = np.asarray([float(np.asarray(interp(point)).reshape(-1)[0]) for interp in self._grad_interp], dtype=float)
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_g_value = g_val
        return grad, g_val

    def grad_u(self, state_row: pd.Series | np.ndarray, *, tau: float | None = None) -> np.ndarray:
        grad, _ = self.grad_u_and_value(state_row, tau=tau)
        return grad

    def estimate_costates(self, state_row: pd.Series | np.ndarray, *, wealth: float = 1.0, tau0: float | None = None) -> CostateEstimate:
        wealth = float(max(wealth, 1.0e-12))
        grad = self.grad_u(state_row, tau=tau0)
        return CostateEstimate(
            JX=1.0 / wealth,
            JXX=-float(self.env.gamma) / (wealth * wealth),
            JXY=grad / wealth,
            closed_form=True,
        )

    def policy_weights(
        self,
        state_row: pd.Series | np.ndarray,
        *,
        covariance: np.ndarray | None = None,
        cross_mat: np.ndarray | None = None,
        tau: float | None = None,
    ) -> np.ndarray:
        w, _ = self.policy_weights_with_debug(state_row, covariance=covariance, cross_mat=cross_mat, tau=tau)
        return np.asarray(w, dtype=float)

    def policy_weights_with_debug(
        self,
        state_row: pd.Series | np.ndarray,
        *,
        covariance: np.ndarray | None = None,
        cross_mat: np.ndarray | None = None,
        tau: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cov = self.env.Sigma_train if covariance is None else _symmetrize_psd(np.asarray(covariance, dtype=float), floor=1.0e-10)
        cross = self.env.C_train if cross_mat is None else np.asarray(cross_mat, dtype=float)
        if cross.ndim == 1:
            cross = cross.reshape(-1, 1)
        x = self._state_array(state_row).reshape(1, -1)
        mu = self.env.mean_map.predict_batch(x).reshape(-1)
        grad_raw, g_val = self.grad_u_and_value(state_row, tau=tau)
        mu_term = np.asarray(mu, dtype=float).reshape(-1)
        hedge_signal = np.asarray(cross @ grad_raw.reshape(-1), dtype=float).reshape(-1)
        a_vec = mu_term + hedge_signal
        pi_unc = _solve_unconstrained_foc_np(cov, a_vec, gamma=float(self.env.gamma))
        w = _clip_foc_policy_np(
            pi_unc,
            cap=float(self.env.risky_cap),
            long_only=bool(getattr(self.env, 'long_only', True)),
        )
        diag = dict(self.solution.diagnostics)
        diag['fdm_state_clip_count'] = int(self._state_clip_count)
        debug: dict[str, Any] = {
            'hedge_signal': hedge_signal,
            'mu_term': mu_term,
            'hedge_term': hedge_signal,
            'a_vec': a_vec,
            'pi_unc': pi_unc,
            'pi_pre_clip': pi_unc,
            'risky_sum_before_clip': float(np.nansum(pi_unc)),
            'risky_sum_after_clip': float(np.nansum(w)),
            'risky_cap': float(self.env.risky_cap),
            'long_only_clip': bool(getattr(self.env, 'long_only', True)),
            'neg_jxx': float(self.env.gamma),
            'neg_jxx_is_gamma': True,
            'control_update_space': 'foc_clip',
            'closed_form_costates': True,
            'grad_training': np.asarray(grad_raw, dtype=float),
            'grad_raw': np.asarray(grad_raw, dtype=float),
            'fdm_value_form': 'g',
            'fdm_g_value': float(g_val),
            'fdm_state_clipped': bool(self._last_state_clipped),
            'fdm_tau_clipped': bool(self._last_tau_clipped),
            'fdm_z_query': np.asarray(self._last_z_query, dtype=float) if self._last_z_query is not None else np.asarray([], dtype=float),
        }
        debug.update(diag)
        return w, debug


def train_fdm_policy(
    states_t: pd.DataFrame,
    returns_tp1: pd.DataFrame,
    cfg: Any,
    transaction_cost: float,
    *,
    mean_model: MeanModelResult,
    transition: StateTransitionResult,
    cross_est: CrossCovarianceEstimate,
    cov_model: Any,
    factor_repr: Any,
    progress_label: str | None = None,
    tau_max: float | None = None,
    warm_start_from: TrainedFDM | None = None,
) -> TrainedFDM:
    del returns_tp1, transaction_cost, progress_label, warm_start_from
    if states_t.shape[1] <= 0:
        raise ValueError('FDM backend requires at least one state variable')
    train_seed = int(getattr(cfg.ppgdpo, 'train_seed', 17) or 17)
    np.random.seed(train_seed)
    sigma_train = _select_training_covariance(
        cfg=cfg,
        cov_model=cov_model,
        cross_est=cross_est,
        state_train=states_t,
        factor_train=factor_repr.factors if hasattr(factor_repr, 'factors') else pd.DataFrame(index=states_t.index),
        loadings=factor_repr.loadings,
        residual_var=factor_repr.residual_var,
    )
    tau_max_cfg = int(getattr(cfg.ppgdpo, 'horizon_steps', 12) or 12)
    tau_cap = tau_max_cfg if tau_max is None else int(np.ceil(float(tau_max)))
    tau_max_eff = float(max(tau_cap, 1))
    # Reuse the existing HJB environment to keep ordering/domain/covariance logic identical to PINN.
    env = PIPINNEnvFromPPGDPO(
        mean_model=mean_model,
        transition=transition,
        cross_est=cross_est,
        states_t=states_t,
        sigma_train=sigma_train,
        cfg=cfg,
        tau_max=tau_max_eff,
        device='cpu',
        dtype=__import__('torch').float64,
    )
    if str(getattr(env, 'policy_output_mode', 'foc_clip')).lower() != 'foc_clip':
        raise ValueError("optimizer_backend='fdm' requires pipinn.policy_output_mode='foc_clip'.")
    solution = solve_g_hjb_fdm(env, _fdm_cfg(cfg))
    best = float(solution.diagnostics.get('fdm_pde_residual_l2_grid', np.nan))
    history = [dict(solution.diagnostics)]
    return TrainedFDM(
        env=env,
        solution=solution,
        train_objective=best,
        train_seed=train_seed,
        train_history=history,
        best_validation_loss=best,
    )
