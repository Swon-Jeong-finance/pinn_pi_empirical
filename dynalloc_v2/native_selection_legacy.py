from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .bridge_common import _ensure_on_syspath
from .factor_zoo import FactorZooCandidate


PORTED_STAGE1_ENGINE = 'ported_legacy_v2'
EXTERNAL_STAGE1_ENGINE = 'legacy_v1_external_audit'
_STAGE1_AUDIT_METRIC_KEYS: tuple[str, ...] = (
    'r2_oos_ret_mean',
    'r2_oos_ret_median',
    'r2_roll_ret_q10',
    'r2_roll_ret_min',
    'r2_oos_state_mean',
    'r2_oos_state_median',
    'r2_roll_state_q10',
    'r2_roll_state_min',
    'cross_mean_abs_rho',
    'cross_max_abs_rho',
)


def _resolve_legacy_stage1_v1_root(arg: str | Path | None, manifest: dict[str, Any]) -> Path | None:
    if arg is not None and str(arg).strip():
        root = Path(arg).expanduser().resolve()
        return root if root.exists() else None
    env_root = os.environ.get('DYNALLOC_V1_ROOT', '').strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.exists():
            return root
    manifest_root = str((manifest or {}).get('source_v1_root') or '').strip()
    if manifest_root:
        root = Path(manifest_root).expanduser().resolve()
        if root.exists():
            return root
    return None


def _legacy_spec_name_for_candidate(candidate: FactorZooCandidate) -> str | None:
    if candidate.kind == 'pls':
        blocks = tuple(candidate.feature_blocks)
        if blocks == ('returns',):
            return candidate.name
        if blocks == ('returns', 'macro7'):
            return candidate.name
        if blocks == ('returns', 'ff5', 'macro7'):
            return candidate.name
        return None
    if candidate.kind == 'provided':
        mapping = {
            'ff1': 'ff_mkt',
            'ff3': 'ff3_only',
            'ff5': 'ff5_only',
        }
        return mapping.get(str(candidate.provided_source))
    if candidate.kind == 'pca' and candidate.n_components is not None:
        return f'pca_only_k{int(candidate.n_components)}'
    return None


def _load_legacy_stage1_modules(v1_root: Path) -> tuple[Any, Any]:
    _ensure_on_syspath(v1_root / 'legacy' / 'vendor' / 'pgdpo_legacy_v69')
    importlib.invalidate_caches()
    spec_selection = importlib.import_module('pgdpo_yahoo.spec_selection')
    discrete_latent_model = importlib.import_module('pgdpo_yahoo.discrete_latent_model')
    return spec_selection, discrete_latent_model



def _evaluate_stage1_candidate_block_legacy(
    *,
    candidate: FactorZooCandidate,
    legacy_spec_selection: Any,
    legacy_discrete_latent_model: Any,
    macro_full: pd.DataFrame,
    ff3_full: pd.DataFrame,
    ff5_full: pd.DataFrame,
    returns_full: pd.DataFrame,
    block: dict[str, Any],
    rolling_window: int,
    window_mode: str,
    return_baseline: str = 'expanding_mean',
    state_baseline: str = 'expanding_mean',
) -> dict[str, float]:
    """Evaluate the *external* legacy v1 stage-1 implementation.

    This path is kept only for parity checking / audit. The production stage-1
    selection path should use the internal ported evaluator inside v2.
    """

    legacy_spec = _legacy_spec_name_for_candidate(candidate)
    if legacy_spec is None:
        raise KeyError(f'candidate {candidate.name!r} has no legacy stage1 mapping')

    common = returns_full.index
    train_dates = common.intersection(pd.DatetimeIndex(block['train_dates']))
    val_dates = common.intersection(pd.DatetimeIndex(block['val_dates']))
    if len(train_dates) < 13 or len(val_dates) == 0:
        raise ValueError('Not enough aligned dates for legacy stage1 evaluation')

    train_end = int(common.get_loc(train_dates[-1]))
    start_idx = int(train_end)
    end_idx = int(common.get_loc(val_dates[-1]))
    if end_idx <= start_idx:
        raise ValueError('Empty legacy validation horizon')

    macro7 = macro_full.loc[common].copy()
    macro3 = macro7.iloc[:, : min(3, macro7.shape[1])].copy() if not macro7.empty else None
    ff3 = ff3_full.loc[common].copy()
    ff5 = ff5_full.loc[common].copy()
    r_ex = returns_full.loc[common].copy()

    cfg = legacy_spec_selection.SpecSelectionConfig(
        window_mode=str(window_mode),
        rolling_window=int(rolling_window),
        return_baseline=str(return_baseline),
        state_baseline=str(state_baseline),
    )
    pca_cfg = legacy_discrete_latent_model.LatentPCAConfig(
        n_components=max(2, int(candidate.n_components or 2)),
        standardize=True,
        random_state=0,
    )
    result = legacy_spec_selection.evaluate_spec_predictive_diagnostics(
        spec=legacy_spec,
        block=str(block['label']),
        r_ex=r_ex,
        macro3=macro3,
        macro7=macro7,
        ff3=ff3,
        ff5=ff5,
        bond_asset_names=None,
        train_end=int(train_end),
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        pca_cfg=pca_cfg,
        pls_horizon=int(candidate.horizon or 12),
        pls_smooth_span=6,
        block_eq_k=1,
        block_bond_k=1,
        config=cfg,
    )
    return {
        'train_obs': int(train_end),
        'val_obs': int(end_idx - start_idx),
        'r2_oos_ret_mean': float(result.r2_oos_ret_mean),
        'r2_oos_ret_median': float(result.r2_oos_ret_median),
        'r2_roll_ret_q10': float(result.r2_roll_ret_q10),
        'r2_roll_ret_min': float(result.r2_roll_ret_min),
        'r2_oos_state_mean': float(result.r2_oos_state_mean),
        'r2_oos_state_median': float(result.r2_oos_state_median),
        'r2_roll_state_q10': float(result.r2_roll_state_q10),
        'r2_roll_state_min': float(result.r2_roll_state_min),
        'cross_mean_abs_rho': float(result.cross_mean_abs_rho),
        'cross_max_abs_rho': float(result.cross_max_abs_rho),
        'stage1_engine': EXTERNAL_STAGE1_ENGINE,
        'legacy_spec': legacy_spec,
    }



def _build_stage1_external_audit_row(
    *,
    candidate: FactorZooCandidate,
    block: dict[str, Any],
    ported_metrics: dict[str, Any],
    external_metrics: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    legacy_spec = _legacy_spec_name_for_candidate(candidate)
    train_dates = pd.DatetimeIndex(block['train_dates'])
    val_dates = pd.DatetimeIndex(block['val_dates'])
    row: dict[str, Any] = {
        'spec': candidate.name,
        'kind': candidate.kind,
        'block': str(block['label']),
        'legacy_spec': legacy_spec,
        'train_start_date': str(train_dates[0].date()) if len(train_dates) else None,
        'train_end_date': str(train_dates[-1].date()) if len(train_dates) else None,
        'val_start_date': str(val_dates[0].date()) if len(val_dates) else None,
        'val_end_date': str(val_dates[-1].date()) if len(val_dates) else None,
        'stage1_engine_ported': PORTED_STAGE1_ENGINE,
        'stage1_engine_external': EXTERNAL_STAGE1_ENGINE if external_metrics is not None else None,
        'audit_compared': bool(external_metrics is not None and error is None),
        'audit_error': error,
    }
    for key in _STAGE1_AUDIT_METRIC_KEYS:
        ported_val = ported_metrics.get(key, np.nan)
        ext_val = np.nan if external_metrics is None else external_metrics.get(key, np.nan)
        row[f'{key}_ported'] = float(ported_val) if np.isfinite(ported_val) else np.nan
        row[f'{key}_external'] = float(ext_val) if np.isfinite(ext_val) else np.nan
        if np.isfinite(ported_val) and np.isfinite(ext_val):
            row[f'delta_{key}'] = float(ported_val - ext_val)
            row[f'abs_delta_{key}'] = float(abs(ported_val - ext_val))
        else:
            row[f'delta_{key}'] = np.nan
            row[f'abs_delta_{key}'] = np.nan
    return row



def _aggregate_stage1_external_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame(columns=[
            'spec',
            'stage1_external_audit_blocks',
            'stage1_external_audit_error_blocks',
            'stage1_external_audit_ret_mean_abs_delta',
            'stage1_external_audit_state_mean_abs_delta',
            'stage1_external_audit_cross_abs_delta',
        ])

    compared = audit_df[audit_df['audit_compared'].fillna(False)].copy()
    compared_counts = compared.groupby('spec').size().rename('stage1_external_audit_blocks') if not compared.empty else pd.Series(dtype=float)
    error_counts = audit_df.groupby('spec')['audit_error'].apply(lambda s: int(s.notna().sum())).rename('stage1_external_audit_error_blocks')
    frames: list[pd.Series] = [error_counts]
    if not compared.empty:
        frames.extend([
            compared.groupby('spec')['abs_delta_r2_oos_ret_mean'].mean().rename('stage1_external_audit_ret_mean_abs_delta'),
            compared.groupby('spec')['abs_delta_r2_oos_state_mean'].mean().rename('stage1_external_audit_state_mean_abs_delta'),
            compared.groupby('spec')['abs_delta_cross_max_abs_rho'].mean().rename('stage1_external_audit_cross_abs_delta'),
            compared_counts,
        ])
    out = pd.concat(frames, axis=1).reset_index()
    if 'stage1_external_audit_blocks' not in out.columns:
        out['stage1_external_audit_blocks'] = 0
    out['stage1_external_audit_blocks'] = out['stage1_external_audit_blocks'].fillna(0).astype(int)
    out['stage1_external_audit_error_blocks'] = out['stage1_external_audit_error_blocks'].fillna(0).astype(int)
    return out


@dataclass(frozen=True)
class _LegacyStage1LinearModel:
    a: np.ndarray
    B: np.ndarray
    Sigma: np.ndarray
    c: np.ndarray
    A: np.ndarray
    Q: np.ndarray
    Cross: np.ndarray



def _ols_fit(X: np.ndarray, Y: np.ndarray, ridge: float = 1.0e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    XtX = X.T @ X
    XtX = XtX + float(ridge) * np.eye(XtX.shape[0], dtype=float)
    XtY = X.T @ Y
    return np.linalg.solve(XtX, XtY)



def _legacy_cross_rho_stats(Sigma: np.ndarray, Q: np.ndarray, Cross: np.ndarray) -> dict[str, float]:
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    Cross = np.asarray(Cross, dtype=float)
    if Cross.ndim == 1:
        Cross = Cross.reshape(-1, 1)
    sig = np.sqrt(np.clip(np.diag(Sigma), 1.0e-16, None))
    q = np.sqrt(np.clip(np.diag(Q), 1.0e-16, None))
    rho = Cross / (sig[:, None] * q[None, :])
    absrho = np.abs(rho)
    return {
        'mean_abs': float(np.mean(absrho)) if absrho.size else 0.0,
        'max_abs': float(np.max(absrho)) if absrho.size else 0.0,
    }



def _fit_legacy_stage1_linear_model(states_full: pd.DataFrame, returns_full: pd.DataFrame, *, train_end: int) -> _LegacyStage1LinearModel:
    if int(train_end) < 10:
        raise ValueError('Need at least 10 TRAIN observations for legacy stage1 fit')
    common = pd.DatetimeIndex(states_full.index.intersection(returns_full.index)).sort_values()
    y = states_full.loc[common].to_numpy(dtype=float)
    rx = returns_full.loc[common].to_numpy(dtype=float)

    Y_ret = rx[1 : train_end + 1, :]
    X_ret = y[0:train_end, :]
    if Y_ret.shape[0] < 10 or X_ret.shape[0] != Y_ret.shape[0]:
        raise ValueError('Not enough aligned rows for legacy return fit')
    X1_ret = np.concatenate([np.ones((X_ret.shape[0], 1), dtype=float), X_ret], axis=1)
    beta_ret = _ols_fit(X1_ret, Y_ret, ridge=1.0e-8)
    a = beta_ret[0, :]
    B = beta_ret[1:, :].T
    ret_resid = Y_ret - X1_ret @ beta_ret
    if ret_resid.shape[0] >= 2:
        Sigma = LedoitWolf().fit(ret_resid).covariance_
    else:
        Sigma = np.atleast_2d(np.cov(ret_resid, rowvar=False, ddof=0))

    Y_state = y[1 : train_end + 1, :]
    X_state = y[0:train_end, :]
    X1_state = np.concatenate([np.ones((X_state.shape[0], 1), dtype=float), X_state], axis=1)
    beta_state = _ols_fit(X1_state, Y_state, ridge=1.0e-8)
    c = beta_state[0, :]
    A = beta_state[1:, :].T
    state_resid = Y_state - X1_state @ beta_state
    Q = np.atleast_2d(np.cov(state_resid, rowvar=False, ddof=0))

    T = min(ret_resid.shape[0], state_resid.shape[0])
    eps = ret_resid[-T:, :] - ret_resid[-T:, :].mean(axis=0, keepdims=True)
    u = state_resid[-T:, :] - state_resid[-T:, :].mean(axis=0, keepdims=True)
    Cross = (eps.T @ u) / float(max(T, 1))

    return _LegacyStage1LinearModel(a=a, B=B, Sigma=Sigma, c=c, A=A, Q=Q, Cross=Cross)



def _predict_returns_from_legacy_model(model: _LegacyStage1LinearModel, y_hist: np.ndarray) -> np.ndarray:
    y_hist = np.asarray(y_hist, dtype=float)
    return np.asarray(model.a, dtype=float)[None, :] + y_hist @ np.asarray(model.B, dtype=float).T



def _predict_states_from_legacy_model(model: _LegacyStage1LinearModel, y_hist: np.ndarray) -> np.ndarray:
    y_hist = np.asarray(y_hist, dtype=float)
    return np.asarray(model.c, dtype=float)[None, :] + y_hist @ np.asarray(model.A, dtype=float).T
