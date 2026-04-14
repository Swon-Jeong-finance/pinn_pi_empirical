from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


@dataclass
class BaseBundleArtifacts:
    out_dir: Path
    returns_csv: Path
    macro_csv: Path
    ff3_csv: Path
    ff5_csv: Path
    bond_csv: Path
    manifest_yaml: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _ensure_on_syspath(path: Path) -> None:
    s = str(path.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _parse_monthly_dates(values: pd.Series) -> pd.DatetimeIndex:
    raw = values.astype(str).str.strip()
    dt = pd.Series(pd.NaT, index=values.index, dtype='datetime64[ns]')
    mask_6 = raw.str.fullmatch(r'\d{6}')
    mask_8 = raw.str.fullmatch(r'\d{8}')
    if mask_6.any():
        dt.loc[mask_6] = pd.to_datetime(raw.loc[mask_6] + '01', format='%Y%m%d', errors='coerce')
    if mask_8.any():
        dt.loc[mask_8] = pd.to_datetime(raw.loc[mask_8], format='%Y%m%d', errors='coerce')
    other = ~(mask_6 | mask_8)
    if other.any():
        dt.loc[other] = pd.to_datetime(raw.loc[other], errors='coerce')
    if dt.notna().sum() == 0:
        return pd.DatetimeIndex([])
    periods = dt.dt.to_period('M')
    return pd.DatetimeIndex(periods.dt.to_timestamp('M'))


def _guess_date_column(df: pd.DataFrame) -> str:
    preferred = ['date', 'Date', 'DATE', 'month', 'Month', 'MONTH', 'yyyymm', 'YYYYMM']
    for col in preferred:
        if col in df.columns:
            return str(col)
    best_col = str(df.columns[0])
    best_ratio = -1.0
    for col in df.columns[: min(4, len(df.columns))]:
        parsed = _parse_monthly_dates(df[col])
        ratio = 0.0 if len(parsed) == 0 else float(pd.Series(parsed).notna().mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = str(col)
    return best_col


def _read_monthly_panel_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f'Empty CSV: {path}')
    date_col = _guess_date_column(df)
    index = _parse_monthly_dates(df[date_col])
    if len(index) != len(df) or pd.Series(index).notna().sum() == 0:
        raise ValueError(f'Could not parse monthly date column in {path}')
    payload = df.drop(columns=[date_col]).copy()
    payload = payload.loc[:, [c for c in payload.columns if not str(c).startswith('Unnamed:')]]
    for col in payload.columns:
        payload[col] = pd.to_numeric(payload[col], errors='coerce')
    payload = payload.dropna(axis=1, how='all')
    if payload.empty:
        raise ValueError(f'No numeric columns found in {path}')
    payload.index = index
    payload = payload.sort_index()
    payload = payload.groupby(level=0).last()
    return payload


def _build_v2_config_dict(
    *,
    out_dir: Path,
    config_stem: str,
    split_payload: dict[str, str],
    state_cols: list[str],
    factor_cols: list[str],
    risk_aversion: float = 5.0,
    covariance_model_kind: str = 'asset_dcc',
    covariance_factor_correlation_mode: str = 'independent',
    covariance_use_persistence: bool = False,
    covariance_adcc_gamma: float = 0.005,
    covariance_regime_threshold_quantile: float = 0.75,
    covariance_regime_smoothing: float = 0.90,
    covariance_regime_sharpness: float = 8.0,
    ppgdpo_covariance_mode: str = 'full',
    ppgdpo_mc_rollouts: int = 256,
    ppgdpo_mc_sub_batch: int = 256,
    mean_model_kind: str = 'factor_apt',
    comparison_cross_modes: list[str] | None = None,
) -> dict[str, Any]:
    run_output = out_dir / 'outputs' / f'{config_stem}_v2_apt_ppgdpo'
    return {
        'project': {
            'name': f'{config_stem}_v2_apt_ppgdpo',
            'output_dir': str(run_output),
        },
        'experiment': {'kind': 'ppgdpo'},
        'data': {
            'mode': 'csv',
            'returns_csv': str(out_dir / 'returns_panel.csv'),
            'states_csv': str(out_dir / 'states_panel.csv'),
            'factors_csv': str(out_dir / 'factors_panel.csv'),
            'date_col': 'date',
        },
        'split': {
            'train_start': split_payload['train_start'],
            'test_start': split_payload['test_start'],
            'end_date': split_payload['end_date'],
            'refit_every': 1,
            'min_train_months': 72,
            'train_window_mode': 'fixed',
            'rolling_train_months': None,
            'rebalance_every': 1,
            'protocol_label': 'fixed',
        },
        'state': {'columns': list(state_cols)},
        'factor_model': {
            'extractor': 'provided',
            'provided_factor_columns': list(factor_cols),
        },
        'mean_model': {'kind': str(mean_model_kind), 'ridge_lambda': 1.0e-6, 'regime_threshold_quantile': 0.75, 'regime_sharpness': 8.0},
        'covariance_model': {
            'kind': str(covariance_model_kind),
            'ridge_lambda': 1.0e-6,
            'variance_floor': 1.0e-6,
            'correlation_shrink': 0.10,
            'factor_correlation_mode': str(covariance_factor_correlation_mode),
            'use_persistence': bool(covariance_use_persistence),
            'dcc_alpha': 0.02,
            'dcc_beta': 0.97,
            'adcc_gamma': float(covariance_adcc_gamma),
            'variance_lambda': 0.97,
            'asset_covariance_shrink': 0.10,
            'regime_threshold_quantile': float(covariance_regime_threshold_quantile),
            'regime_smoothing': float(covariance_regime_smoothing),
            'regime_sharpness': float(covariance_regime_sharpness),
        },
        'policy': {
            'risk_aversion': float(risk_aversion),
            'risky_cap': 1.0,
            'cash_floor': 0.0,
            'long_only': True,
            'pgd_steps': 120,
            'step_size': 0.05,
            'turnover_penalty': 0.05,
        },
        'ppgdpo': {
            'device': 'cpu',
            'hidden_dim': 32,
            'hidden_layers': 2,
            'epochs': 120,
            'lr': 1.0e-3,
            'utility': 'crra',
            'batch_size': 64,
            'horizon_steps': 12,
            'kappa': 1.0,
            'mc_rollouts': int(ppgdpo_mc_rollouts),
            'mc_sub_batch': int(ppgdpo_mc_sub_batch),
            'clamp_min_return': -0.95,
            'clamp_port_ret_max': 5.0,
            'clamp_wealth_min': 1.0e-8,
            'clamp_state_std_abs': 8.0,
            'covariance_mode': str(ppgdpo_covariance_mode),
            'cross_strength': 1.0,
            'eps_bar': 1.0e-6,
            'newton_ridge': 1.0e-10,
            'newton_tau': 0.95,
            'newton_armijo': 1.0e-4,
            'newton_backtrack': 0.5,
            'max_newton': 30,
            'tol_grad': 1.0e-8,
            'max_line_search': 20,
            'interior_margin': 1.0e-8,
            'clamp_neg_jxx_min': 1.0e-12,
            'train_seed': 17,
            'state_ridge_lambda': 1.0e-6,
        },
        'comparison': {
            'cross_modes': list(comparison_cross_modes or ['estimated', 'zero', 'regime_gated']),
            'transaction_cost_bps': 0.0,
        },
    }
