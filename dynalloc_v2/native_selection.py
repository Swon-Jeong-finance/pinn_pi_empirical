from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .covariance import AssetADCCCovariance, AssetDCCCovariance, AssetRegimeDCCCovariance, ConstantFactorCovariance, StateDiagonalFactorCovariance
from .factor_zoo import FactorZooCandidate, build_candidate_panels, build_candidate_registry
from .native_selection_legacy import (
    PORTED_STAGE1_ENGINE,
    _aggregate_stage1_external_audit,
    _build_stage1_external_audit_row,
    _evaluate_stage1_candidate_block_legacy,
    _fit_legacy_stage1_linear_model,
    _legacy_cross_rho_stats,
    _legacy_spec_name_for_candidate,
    _load_legacy_stage1_modules,
    _predict_returns_from_legacy_model,
    _predict_states_from_legacy_model,
    _resolve_legacy_stage1_v1_root,
)
from .factors import ProvidedFactorExtractor
from .bridge_common import _build_v2_config_dict
from .mean_model import fit_factor_apt_mean, fit_factor_apt_regime_mean
from .experiments import run_experiment
from .policies import solve_mean_variance
from .ppgdpo import solve_ppgdpo_projection, train_warmup_policy
from .transition import estimate_return_state_cross, fit_state_transition
from .selection_splits import (
    SelectionSplitSpec,
    build_cv_blocks as _build_cv_blocks_impl,
    build_selection_blocks as _build_selection_blocks_impl,
    build_trailing_holdout_blocks as _build_trailing_holdout_blocks_impl,
    selection_pool_index as _selection_pool_index_impl,
)
from .experiment_windows import resolve_split_payload as resolve_calendar_window
from .oos_protocols import SELECTED_PROTOCOL, ROLLING_SELECTED_PROTOCOL, apply_oos_protocol, manifest_protocol_payload, protocol_spec
from .schema import Config
from .utils import certainty_equivalent_annual, ensure_dir, sharpe_ratio


SELECTION_REFERENCE_BENCHMARKS: tuple[str, ...] = ('equal_weight', 'min_variance', 'risk_parity')


@dataclass
class NativeBaseBundleArtifacts:
    out_dir: Path
    returns_csv: Path
    macro_csv: Path
    ff3_csv: Path
    ff5_csv: Path
    bond_csv: Path
    manifest_yaml: Path


@dataclass
class NativeSelectionArtifacts:
    out_dir: Path
    manifest_yaml: Path
    selection_summary_csv: Path
    selected_yaml: Path
    entry_count: int



@dataclass(frozen=True)
class SelectionLitePPGDPOConfig:
    optimizer_backend: str = 'ppgdpo'
    rerank_top_n: int = 5
    device: str = 'cpu'
    epochs: int = 40
    hidden_dim: int = 24
    hidden_layers: int = 2
    lr: float = 1.0e-3
    utility: str = 'crra'
    batch_size: int = 8
    horizon_steps: int = 2
    kappa: float = 1.0
    mc_rollouts: int = 256
    mc_sub_batch: int = 256
    covariance_mode: str = 'full'
    covariance_model_kind: str = 'asset_dcc'
    covariance_label: str = 'dcc'
    mean_model_kind: str = 'factor_apt'
    cross_policy_label: str = 'estimated'
    factor_correlation_mode: str = 'independent'
    use_persistence: bool = False
    adcc_gamma: float = 0.005
    regime_threshold_quantile: float = 0.75
    regime_smoothing: float = 0.90
    regime_sharpness: float = 8.0
    cross_strength: float = 1.0
    eps_bar: float = 1.0e-6
    newton_ridge: float = 1.0e-10
    newton_tau: float = 0.95
    newton_armijo: float = 1.0e-4
    newton_backtrack: float = 0.5
    max_newton: int = 30
    tol_grad: float = 1.0e-8
    max_line_search: int = 20
    interior_margin: float = 1.0e-8
    clamp_neg_jxx_min: float = 1.0e-12
    train_seed: int = 17
    state_ridge_lambda: float = 1.0e-6
    pgd_steps: int = 100
    step_size: float = 0.05
    turnover_penalty: float = 0.05
    risky_cap: float = 1.0
    cash_floor: float = 0.0
    clamp_min_return: float = -0.95
    clamp_port_ret_max: float = 5.0
    clamp_wealth_min: float = 1.0e-8
    clamp_state_std_abs: float | None = 8.0
    transaction_cost_bps: float = 0.0
    pipinn_device: str = 'auto'
    pipinn_dtype: str = 'float64'
    pipinn_outer_iters: int = 6
    pipinn_eval_epochs: int = 120
    pipinn_n_train_int: int = 4096
    pipinn_n_train_bc: int = 1024
    pipinn_n_val_int: int = 2048
    pipinn_n_val_bc: int = 512
    pipinn_p_uniform: float = 0.30
    pipinn_p_emp: float = 0.70
    pipinn_p_tau_head: float = 0.50
    pipinn_p_tau_near0: float = 0.20
    pipinn_tau_head_window: int = 0
    pipinn_lr: float = 5.0e-4
    pipinn_grad_clip: float = 1.0
    pipinn_w_bc: float = 20.0
    pipinn_w_bc_dx: float = 5.0
    pipinn_scheduler_factor: float = 0.5
    pipinn_scheduler_patience: int = 3
    pipinn_min_lr: float = 1.0e-5
    pipinn_width: int = 96
    pipinn_depth: int = 4
    pipinn_covariance_train_mode: str = 'dcc_current'
    pipinn_ansatz_mode: str = 'ansatz_normalization_log_transform'
    pipinn_policy_output_mode: str = 'projection'
    pipinn_emit_frozen_traincov_strategy: bool = False
    pipinn_save_training_logs: bool = True
    pipinn_show_progress: bool = False
    pipinn_show_epoch_progress: bool = False


@dataclass(frozen=True)
class SelectionStage2ModelSpec:

    label: str
    covariance_model_kind: str
    factor_correlation_mode: str = 'independent'
    use_persistence: bool = False
    adcc_gamma: float = 0.005
    regime_threshold_quantile: float = 0.75
    regime_smoothing: float = 0.90
    regime_sharpness: float = 8.0
    mean_model_kind: str = 'factor_apt'
    cross_policy_label: str = 'estimated'
    base_covariance_label: str | None = None


@dataclass
class _PairedBlockData:
    decision_dates: pd.DatetimeIndex
    next_dates: pd.DatetimeIndex
    states_t: pd.DataFrame
    states_tp1: pd.DataFrame
    factors_t: pd.DataFrame
    factors_tp1: pd.DataFrame
    returns_tp1: pd.DataFrame


_MISSING_SCORE_PENALTY = -1.0e6

def _parse_stage2_device_pool(value: str | None, *, fallback: str) -> list[str]:
    if value is None:
        tokens: list[str] = []
    else:
        text = str(value).strip()
        if not text:
            tokens = []
        else:
            tokens = [tok.strip() for tok in text.split(',') if tok.strip()]
    if not tokens:
        tokens = [str(fallback)]
    return tokens

def _load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'date' not in df.columns:
        raise ValueError(f'date column missing in {path}')
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()


def load_base_bundle(base_dir: str | Path) -> dict[str, Any]:
    base_dir = Path(base_dir).expanduser().resolve()
    manifest = yaml.safe_load((base_dir / 'base_bundle_manifest.yaml').read_text(encoding='utf-8')) or {}
    returns = _load_panel(base_dir / 'returns_panel.csv')
    macro = _load_panel(base_dir / 'macro_panel.csv')
    ff3 = _load_panel(base_dir / 'ff3_panel.csv')
    ff5 = _load_panel(base_dir / 'ff5_panel.csv')
    bond = _load_panel(base_dir / 'bond_panel.csv')
    common = returns.index.intersection(macro.index).intersection(ff3.index).intersection(ff5.index)
    if not bond.empty:
        common = common.intersection(bond.index)
    return {
        'manifest': manifest,
        'returns': returns.loc[common].copy(),
        'macro': macro.loc[common].copy(),
        'ff3': ff3.loc[common].copy(),
        'ff5': ff5.loc[common].copy(),
        'bond': bond.loc[common].copy(),
        'base_dir': base_dir,
    }


def _r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray, y_bench: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_bench = np.asarray(y_bench, dtype=float)
    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    sse_b = np.sum((y_true - y_bench) ** 2, axis=0)
    return 1.0 - (sse / np.clip(sse_b, 1e-16, None))


def _ols_style_r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    y_bar = np.mean(y_true, axis=0, keepdims=True)
    sst = np.sum((y_true - y_bar) ** 2, axis=0)
    sse = np.sum(resid ** 2, axis=0)
    return 1.0 - (sse / np.clip(sst, 1e-16, None))


def _make_oos_baseline(target_all: np.ndarray, *, start_idx: int, end_idx: int, mode: str, random_walk_source: np.ndarray | None = None) -> np.ndarray:
    target_all = np.asarray(target_all, dtype=float)
    preds: list[np.ndarray] = []
    mode = str(mode).lower()
    for t in range(int(start_idx), int(end_idx)):
        if mode == 'train_mean':
            hist = target_all[1 : start_idx + 1, :]
            base = np.mean(hist, axis=0)
        elif mode == 'expanding_mean':
            hist = target_all[1 : t + 1, :]
            base = np.mean(hist, axis=0)
        elif mode == 'random_walk':
            if random_walk_source is None:
                raise ValueError('random_walk baseline requires random_walk_source')
            base = np.asarray(random_walk_source[t, :], dtype=float)
        else:
            raise ValueError(f'Unknown baseline mode: {mode}')
        preds.append(base)
    return np.asarray(preds, dtype=float)


def _window_r2_summary(y_true: np.ndarray, y_pred: np.ndarray, y_bench: np.ndarray, *, window: int, mode: str = 'rolling') -> tuple[float, float]:
    T = int(y_true.shape[0])
    if T <= 0:
        return float('nan'), float('nan')
    mode = str(mode).lower()
    if mode not in {'rolling', 'expanding'}:
        raise ValueError(f'Unknown selection window mode: {mode}')
    if window <= 1 or T <= window:
        r2 = _r2_per_dim(y_true, y_pred, y_bench)
        val = float(np.mean(r2))
        return val, val
    vals: list[float] = []
    if mode == 'rolling':
        for s in range(0, T - window + 1):
            e = s + window
            r2w = _r2_per_dim(y_true[s:e], y_pred[s:e], y_bench[s:e])
            vals.append(float(np.mean(r2w)))
    else:
        min_window = int(max(window, 1))
        for e in range(min_window, T + 1):
            r2w = _r2_per_dim(y_true[:e], y_pred[:e], y_bench[:e])
            vals.append(float(np.mean(r2w)))
    arr = np.asarray(vals, dtype=float)
    return float(np.quantile(arr, 0.10)), float(np.min(arr))


def _offdiag_corr_metrics(states: pd.DataFrame) -> tuple[float, float]:
    if states.shape[1] <= 1:
        return 0.0, 0.0
    corr = states.corr().to_numpy(dtype=float)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = np.abs(corr[mask])
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.max())


def _selection_pool_index(index: pd.DatetimeIndex, *, train_start: pd.Timestamp | None, train_pool_end: pd.Timestamp) -> pd.DatetimeIndex:
    return _selection_pool_index_impl(index, train_start=train_start, train_pool_end=train_pool_end)


def _build_cv_blocks(index: pd.DatetimeIndex, *, train_start: pd.Timestamp | None = None, train_pool_end: pd.Timestamp, cv_folds: int, min_train_months: int, rolling_window: int, window_mode: str) -> list[dict[str, Any]]:
    return _build_cv_blocks_impl(
        index,
        train_start=train_start,
        train_pool_end=train_pool_end,
        cv_folds=cv_folds,
        min_train_months=min_train_months,
        rolling_window=rolling_window,
        window_mode=window_mode,
    )


def _build_trailing_holdout_blocks(index: pd.DatetimeIndex, *, train_start: pd.Timestamp | None = None, train_pool_end: pd.Timestamp, val_months: int, min_train_months: int) -> list[dict[str, Any]]:
    return _build_trailing_holdout_blocks_impl(
        index,
        train_start=train_start,
        train_pool_end=train_pool_end,
        val_months=val_months,
        min_train_months=min_train_months,
    )


def _build_selection_blocks(index: pd.DatetimeIndex, *, train_start: pd.Timestamp | None = None, train_pool_end: pd.Timestamp, split_mode: str, cv_folds: int, min_train_months: int, selection_val_months: int, rolling_window: int, window_mode: str) -> list[dict[str, Any]]:
    spec = SelectionSplitSpec(
        mode=str(split_mode),
        train_start=train_start,
        train_pool_end=pd.Timestamp(train_pool_end),
        cv_folds=int(cv_folds),
        min_train_months=int(min_train_months),
        selection_val_months=int(selection_val_months),
        rolling_window=int(rolling_window),
        window_mode=str(window_mode),
    )
    return _build_selection_blocks_impl(index, spec)


def _rank_score(df: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series:
    n = len(df)
    if n <= 1:
        return pd.Series(1.0, index=df.index)
    ranks = df[column].rank(ascending=ascending, method='average')
    return 1.0 - (ranks - 1.0) / (n - 1.0)


def _selection_score_ret_first(ret_mean: float, ret_q10: float, state_mean: float, *, missing_penalty: float = _MISSING_SCORE_PENALTY) -> float:
    """Legacy return-first linear score (Recommendation A)."""

    def _clean(x: float) -> float:
        return float(x) if np.isfinite(x) else float(missing_penalty)

    return 0.70 * _clean(ret_mean) + 0.20 * _clean(ret_q10) + 0.10 * _clean(state_mean)


def _selection_score_mean_first(ret_mean: float, ret_q10: float, *, missing_penalty: float = _MISSING_SCORE_PENALTY) -> float:
    """Stage-1 cheap screening score for stage29 (v1-compatible mean-first stage1).

    Mean fit is the anchor. We keep a small tail-return term for stability, but
    state-prediction no longer enters the cheap screen ranking.
    """

    def _clean(x: float) -> float:
        return float(x) if np.isfinite(x) else float(missing_penalty)

    return 0.85 * _clean(ret_mean) + 0.15 * _clean(ret_q10)


def _selection_score_ppgdpo_lite(ce_est: float, ce_gain_myopic: float, ce_gain_zero: float, *, missing_penalty: float = _MISSING_SCORE_PENALTY) -> float:
    """Stage-2 rerank score.

    We anchor on absolute PPGDPO CE and reward incremental gains over the
    zero-cross baseline. The myopic comparison is kept for reporting only and
    does not enter the stage-2 selection score.
    """

    def _clean(x: float) -> float:
        return float(x) if np.isfinite(x) else float(missing_penalty)

    _ = ce_gain_myopic  # report-only; not used in the selection score
    return _clean(ce_est) + _clean(ce_gain_zero)


def _sort_stage2_frame(stage2_df: pd.DataFrame) -> pd.DataFrame:
    return stage2_df.sort_values(
        [
            'stage2_sort_score',
            'ppgdpo_lite_score_mean',
            'ppgdpo_lite_score_q10',
            'ppgdpo_lite_ce_delta_zero_mean',
            'ppgdpo_lite_ce_delta_myopic_mean',
            'ppgdpo_lite_ce_mean',
            'spec',
            'covariance_model_label',
        ],
        ascending=[False, False, False, False, False, False, True, True],
    )


def _annotate_stage2_real_ppgdpo_scores(stage2_df: pd.DataFrame, stage1_selected_specs: list[str]) -> pd.DataFrame:
    stage2_df = stage2_df.copy()
    group_col = 'selection_unit_id' if 'selection_unit_id' in stage2_df.columns else 'spec'
    if stage2_df.empty:
        stage2_df['stage2_rank_myopic_ce'] = pd.Series(dtype=float)
        stage2_df['stage2_rank_ce_gain'] = pd.Series(dtype=float)
        stage2_df['stage2_rank_ppgdpo_score'] = pd.Series(dtype=float)
        stage2_df['stage2_rank_ppgdpo_q10'] = pd.Series(dtype=float)
        stage2_df['stage2_rank_hedging_gain'] = pd.Series(dtype=float)
        stage2_df['stage2_real_ppgdpo_score'] = pd.Series(dtype=float)
        stage2_df['stage2_mean_first_score'] = pd.Series(dtype=float)
        stage2_df['stage2_sort_score'] = pd.Series(dtype=float)
        stage2_df['diagnostic_rank'] = pd.Series(dtype=float)
        stage2_df['selected_stage2_model'] = pd.Series(dtype=bool)
        stage2_df['selected_stage2_model_rank'] = pd.Series(dtype=float)
        return stage2_df

    stage2_df['stage2_rank_myopic_ce'] = _rank_score(stage2_df, 'ppgdpo_lite_myopic_ce_mean', ascending=False)
    stage2_df['stage2_rank_ce_gain'] = _rank_score(stage2_df, 'ppgdpo_lite_ce_delta_myopic_mean', ascending=False)
    stage2_df['stage2_rank_ppgdpo_score'] = _rank_score(stage2_df, 'ppgdpo_lite_score_mean', ascending=False)
    stage2_df['stage2_rank_ppgdpo_q10'] = _rank_score(stage2_df, 'ppgdpo_lite_score_q10', ascending=False)
    stage2_df['stage2_rank_hedging_gain'] = _rank_score(stage2_df, 'ppgdpo_lite_ce_delta_zero_mean', ascending=False)

    for rank_col in ['stage2_rank_ppgdpo_score', 'stage2_rank_ppgdpo_q10', 'stage2_rank_hedging_gain']:
        if rank_col not in stage2_df.columns:
            continue
        series = stage2_df[rank_col]
        if series.isna().all():
            stage2_df[rank_col] = series.fillna(0.5)
        else:
            stage2_df[rank_col] = series.fillna(0.0)

    stage2_df['stage2_real_ppgdpo_score'] = (
        # 0.65 * stage2_df['stage2_rank_ppgdpo_score']
        # + 0.20 * stage2_df['stage2_rank_ppgdpo_q10']
        # + 0.15 * stage2_df['stage2_rank_hedging_gain']
        0.65 * stage2_df['stage2_rank_ppgdpo_score']
        + 0.10 * stage2_df['stage2_rank_ppgdpo_q10']
        + 0.25 * stage2_df['stage2_rank_hedging_gain']
    )
    stage2_df['stage2_mean_first_score'] = stage2_df['stage2_real_ppgdpo_score']
    stage2_df['stage2_sort_score'] = stage2_df['stage2_real_ppgdpo_score'].where(np.isfinite(stage2_df['stage2_real_ppgdpo_score']), -np.inf)
    stage2_df = _sort_stage2_frame(stage2_df).reset_index(drop=True)
    stage2_df['diagnostic_rank'] = np.arange(1, len(stage2_df) + 1, dtype=float)

    stage2_df['selected_stage2_model'] = False
    stage2_df['selected_stage2_model_rank'] = np.nan
    spec_rank_map = {spec_name: rank for rank, spec_name in enumerate(stage1_selected_specs, start=1)}
    for spec_name in stage1_selected_specs:
        spec_positions = stage2_df.index[stage2_df[group_col] == spec_name].tolist()
        if not spec_positions:
            continue
        winner_idx = int(spec_positions[0])
        stage2_df.loc[winner_idx, 'selected_stage2_model'] = True
        stage2_df.loc[winner_idx, 'selected_stage2_model_rank'] = float(spec_rank_map.get(spec_name, np.nan))
    return stage2_df


def _parse_rerank_covariance_models(models: list[str] | tuple[str, ...] | None) -> list[SelectionStage2ModelSpec]:
    requested = [str(x).strip().lower() for x in (models or ['const', 'dcc', 'adcc', 'regime_dcc']) if str(x).strip()]
    if not requested:
        requested = ['const', 'dcc', 'adcc', 'regime_dcc']
    specs: list[SelectionStage2ModelSpec] = []
    seen: set[str] = set()
    for label in requested:
        if label in seen:
            continue
        seen.add(label)
        if label == 'const':
            specs.append(SelectionStage2ModelSpec(label='const', base_covariance_label='const', covariance_model_kind='constant', factor_correlation_mode='sample_shrunk', use_persistence=False))
        elif label == 'diag':
            specs.append(SelectionStage2ModelSpec(label='diag', base_covariance_label='diag', covariance_model_kind='state_only_diagonal', factor_correlation_mode='independent', use_persistence=False))
        elif label == 'dcc':
            specs.append(SelectionStage2ModelSpec(label='dcc', base_covariance_label='dcc', covariance_model_kind='asset_dcc', factor_correlation_mode='independent', use_persistence=False))
        elif label == 'adcc':
            specs.append(SelectionStage2ModelSpec(label='adcc', base_covariance_label='adcc', covariance_model_kind='asset_adcc', factor_correlation_mode='independent', use_persistence=False, adcc_gamma=0.005))
        elif label == 'regime_dcc':
            specs.append(SelectionStage2ModelSpec(label='regime_dcc', base_covariance_label='regime_dcc', covariance_model_kind='asset_regime_dcc', factor_correlation_mode='independent', use_persistence=False, adcc_gamma=0.005, regime_threshold_quantile=0.75, regime_smoothing=0.90, regime_sharpness=8.0))
        else:
            raise KeyError(f'Unsupported rerank covariance model {label!r}')
    return specs


def _expand_stage2_model_specs(cov_specs: list[SelectionStage2ModelSpec]) -> list[SelectionStage2ModelSpec]:
    expanded: list[SelectionStage2ModelSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for cov_spec in cov_specs:
        base_label = str(cov_spec.base_covariance_label or cov_spec.label)
        if base_label == 'regime_dcc':
            variants = [
                (base_label, 'estimated'),
                (f'{base_label}__zero_cross', 'zero'),
                (f'{base_label}__gated_cross', 'regime_gated'),
            ]
        else:
            variants = [
                (base_label, 'estimated'),
                (f'{base_label}__zero_cross', 'zero'),
            ]
        for label, cross_policy in variants:
            key = (label, base_label, cross_policy)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(replace(
                cov_spec,
                label=label,
                base_covariance_label=base_label,
                cross_policy_label=cross_policy,
            ))
    return expanded


def _lite_cfg_for_stage2_model(
    base_cfg: SelectionLitePPGDPOConfig,
    model_spec: SelectionStage2ModelSpec,
    *,
    mean_model_kind: str,
) -> SelectionLitePPGDPOConfig:
    return replace(
        base_cfg,
        covariance_label=str(model_spec.label),
        covariance_model_kind=str(model_spec.covariance_model_kind),
        factor_correlation_mode=str(model_spec.factor_correlation_mode),
        use_persistence=bool(model_spec.use_persistence),
        adcc_gamma=float(model_spec.adcc_gamma),
        regime_threshold_quantile=float(model_spec.regime_threshold_quantile),
        regime_smoothing=float(model_spec.regime_smoothing),
        regime_sharpness=float(model_spec.regime_sharpness),
        mean_model_kind=str(mean_model_kind),
        cross_policy_label=str(model_spec.cross_policy_label),
    )


def _comparison_cross_modes_for_covariance_label(label: str) -> list[str]:
    base = str(label).split('__', 1)[0].lower()
    if base == 'regime_dcc':
        return ['estimated', 'zero', 'regime_gated']
    return ['estimated', 'zero']


def _config_covariance_payload_from_label(label: str) -> dict[str, Any]:
    label = str(label).lower()
    if label == 'const':
        return {
            'kind': 'constant',
            'factor_correlation_mode': 'sample_shrunk',
            'use_persistence': False,
            'cross_covariance_kind': 'sample',
        }
    if label == 'diag':
        return {
            'kind': 'state_only_diagonal',
            'factor_correlation_mode': 'independent',
            'use_persistence': False,
            'cross_covariance_kind': 'sample',
        }
    if label == 'dcc':
        return {
            'kind': 'asset_dcc',
            'factor_correlation_mode': 'independent',
            'use_persistence': False,
            'adcc_gamma': 0.005,
            'cross_covariance_kind': 'dcc',
        }
    if label == 'adcc':
        return {
            'kind': 'asset_adcc',
            'factor_correlation_mode': 'independent',
            'use_persistence': False,
            'adcc_gamma': 0.005,
            'regime_threshold_quantile': 0.75,
            'regime_smoothing': 0.90,
            'regime_sharpness': 8.0,
            'cross_covariance_kind': 'adcc',
        }
    if label == 'regime_dcc':
        return {
            'kind': 'asset_regime_dcc',
            'factor_correlation_mode': 'independent',
            'use_persistence': False,
            'adcc_gamma': 0.005,
            'regime_threshold_quantile': 0.75,
            'regime_smoothing': 0.90,
            'regime_sharpness': 8.0,
            'cross_covariance_kind': 'regime_dcc',
        }
    raise KeyError(f'Unsupported covariance label {label!r}')


def _paired_block_data(
    *,
    states_full: pd.DataFrame,
    factors_full: pd.DataFrame,
    returns_full: pd.DataFrame,
    raw_dates: pd.DatetimeIndex,
) -> _PairedBlockData:
    decisions: list[pd.Timestamp] = []
    nexts: list[pd.Timestamp] = []
    raw_dates = pd.DatetimeIndex(raw_dates)
    if len(raw_dates) >= 2:
        for date_t, next_date in zip(raw_dates[:-1], raw_dates[1:]):
            if (
                date_t in states_full.index
                and next_date in states_full.index
                and date_t in factors_full.index
                and next_date in factors_full.index
                and next_date in returns_full.index
            ):
                decisions.append(pd.Timestamp(date_t))
                nexts.append(pd.Timestamp(next_date))

    decision_idx = pd.DatetimeIndex(decisions)
    next_idx = pd.DatetimeIndex(nexts)
    if len(decision_idx) == 0:
        empty_states = states_full.iloc[0:0].copy()
        empty_factors = factors_full.iloc[0:0].copy()
        empty_returns = returns_full.iloc[0:0].copy()
        return _PairedBlockData(
            decision_dates=decision_idx,
            next_dates=next_idx,
            states_t=empty_states,
            states_tp1=empty_states.copy(),
            factors_t=empty_factors,
            factors_tp1=empty_factors.copy(),
            returns_tp1=empty_returns,
        )

    states_t = states_full.loc[decision_idx].copy()
    states_tp1 = states_full.loc[next_idx].copy()
    states_tp1.index = decision_idx
    factors_t = factors_full.loc[decision_idx].copy()
    factors_tp1 = factors_full.loc[next_idx].copy()
    factors_tp1.index = decision_idx
    returns_tp1 = returns_full.loc[next_idx].copy()
    returns_tp1.index = decision_idx
    return _PairedBlockData(
        decision_dates=decision_idx,
        next_dates=next_idx,
        states_t=states_t,
        states_tp1=states_tp1,
        factors_t=factors_t,
        factors_tp1=factors_tp1,
        returns_tp1=returns_tp1,
    )


def _protocol_row_payload(protocol: str) -> dict[str, Any]:
    spec = protocol_spec(protocol)
    return {
        'selection_protocol_name': str(spec.name),
        'train_window_mode': str(spec.train_window_mode),
        'rolling_train_months': int(spec.rolling_train_months) if spec.rolling_train_months is not None else np.nan,
        'refit_every': int(spec.refit_every) if spec.refit_every is not None else 0,
        'rebalance_every': int(spec.rebalance_every),
    }



def _mean_variant_label(mean_model_kind: str) -> str:
    kind = str(mean_model_kind)
    if kind == 'factor_apt_regime':
        return 'regime_mean'
    return 'baseline_mean'



def _selection_unit_id(spec_name: str, protocol: str, mean_model_kind: str = 'factor_apt') -> str:
    return f"{str(spec_name)}__{str(protocol)}__{_mean_variant_label(mean_model_kind)}"



def _expand_stage1_mean_variants(stage1_df: pd.DataFrame) -> pd.DataFrame:
    if stage1_df.empty:
        out = stage1_df.copy()
        out['mean_model_kind'] = pd.Series(dtype=object)
        out['stage1_mean_variant_label'] = pd.Series(dtype=object)
        return out
    rows: list[dict[str, Any]] = []
    variants = [
        ('baseline_mean', 'factor_apt'),
        ('regime_mean', 'factor_apt_regime'),
    ]
    for _, row in stage1_df.iterrows():
        base = row.to_dict()
        protocol = str(base.get('selection_protocol_name') or '')
        spec_name = str(base.get('spec') or '')
        for label, mean_kind in variants:
            payload = dict(base)
            payload['stage1_mean_variant_label'] = label
            payload['mean_model_kind'] = mean_kind
            payload['selection_unit_id'] = _selection_unit_id(spec_name, protocol, mean_kind)
            rows.append(payload)
    return pd.DataFrame(rows)



def _normalize_selection_protocols(
    selection_protocols: list[str] | tuple[str, ...] | None,
    *,
    select_rolling_oos_window: bool,
    rolling_oos_window_grid: list[int] | tuple[int, ...] | None,
) -> list[str]:
    if selection_protocols is not None:
        raw = [str(x).strip() for x in selection_protocols if str(x).strip()]
    else:
        months_grid = _normalize_rolling_oos_window_grid(rolling_oos_window_grid)
        raw = [f'rolling{int(months)}m_annual' for months in months_grid]
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        protocol_spec(value)
        if value not in seen:
            seen.add(value)
            out.append(value)
    if not out:
        out = ['rolling240m_annual']
    return out



def _evaluate_stage1_candidate_block(
    *,
    states_full: pd.DataFrame,
    returns_full: pd.DataFrame,
    block: dict[str, Any],
    rolling_window: int,
    window_mode: str,
    protocol: str = 'fixed',
    return_baseline: str = 'expanding_mean',
    state_baseline: str = 'expanding_mean',
) -> dict[str, float]:
    common = pd.DatetimeIndex(states_full.index.intersection(returns_full.index)).sort_values()
    train_dates = common.intersection(pd.DatetimeIndex(block['train_dates']))
    val_dates = common.intersection(pd.DatetimeIndex(block['val_dates']))
    if len(train_dates) < 13 or len(val_dates) == 0:
        raise ValueError('Not enough aligned dates for v1-compatible stage1 evaluation')

    states_common = states_full.loc[common].copy()
    returns_common = returns_full.loc[common].copy()
    y_all = states_common.to_numpy(dtype=float)
    rx_all = returns_common.to_numpy(dtype=float)

    train_start_idx = int(common.get_loc(train_dates[0]))
    initial_train_end = int(common.get_loc(train_dates[-1]))
    val_locs = [int(common.get_loc(date)) for date in val_dates if date in common]
    val_locs = [loc for loc in val_locs if loc > 0]
    if not val_locs:
        raise ValueError('Empty v1-compatible validation horizon')

    proto = protocol_spec(protocol)
    decision_locs = [loc - 1 for loc in val_locs]
    train_lengths: list[int] = []
    rx_pred_parts: list[np.ndarray] = []
    state_pred_parts: list[np.ndarray] = []

    if proto.train_window_mode == 'fixed':
        fit_idx = pd.DatetimeIndex(common[train_start_idx : initial_train_end + 1])
        model = _fit_legacy_stage1_linear_model(states_common.loc[fit_idx], returns_common.loc[fit_idx], train_end=len(fit_idx) - 1)
        rx_pred_parts.append(_predict_returns_from_legacy_model(model, y_all[np.asarray(decision_locs, dtype=int), :]))
        state_pred_parts.append(_predict_states_from_legacy_model(model, y_all[np.asarray(decision_locs, dtype=int), :]))
        train_lengths = [len(fit_idx)] * len(decision_locs)
    else:
        refit_every = int(proto.refit_every or 12)
        segment_starts = list(range(0, len(val_locs), max(refit_every, 1)))
        for seg_idx, seg_start in enumerate(segment_starts):
            seg_end = segment_starts[seg_idx + 1] if seg_idx + 1 < len(segment_starts) else len(val_locs)
            seg_decision_locs = decision_locs[seg_start:seg_end]
            fit_end_loc = int(seg_decision_locs[0])
            if proto.train_window_mode == 'expanding':
                fit_start_loc = int(train_start_idx)
            elif proto.train_window_mode == 'rolling':
                rolling_months = int(proto.rolling_train_months or max(rolling_window, 1))
                fit_start_loc = max(int(train_start_idx), int(fit_end_loc - rolling_months + 1))
            else:
                fit_start_loc = int(train_start_idx)
            fit_idx = pd.DatetimeIndex(common[fit_start_loc : fit_end_loc + 1])
            if len(fit_idx) < 13:
                raise ValueError(f'Not enough protocol-specific training rows for stage1 evaluation: protocol={protocol!r}, train_rows={len(fit_idx)}')
            model = _fit_legacy_stage1_linear_model(states_common.loc[fit_idx], returns_common.loc[fit_idx], train_end=len(fit_idx) - 1)
            rx_pred_parts.append(_predict_returns_from_legacy_model(model, y_all[np.asarray(seg_decision_locs, dtype=int), :]))
            state_pred_parts.append(_predict_states_from_legacy_model(model, y_all[np.asarray(seg_decision_locs, dtype=int), :]))
            train_lengths.extend([len(fit_idx)] * len(seg_decision_locs))

    rx_true_oos = returns_common.loc[pd.DatetimeIndex(common[val_locs])].to_numpy(dtype=float)
    rx_pred_oos = np.vstack(rx_pred_parts) if rx_pred_parts else np.zeros_like(rx_true_oos)
    rx_bench_oos = _make_oos_baseline(rx_all, start_idx=decision_locs[0], end_idx=val_locs[-1], mode=return_baseline)
    r2_oos_ret = _r2_per_dim(rx_true_oos, rx_pred_oos, rx_bench_oos)
    r2_roll_ret_q10, r2_roll_ret_min = _window_r2_summary(
        rx_true_oos,
        rx_pred_oos,
        rx_bench_oos,
        window=int(rolling_window),
        mode=str(window_mode),
    )

    state_true_oos = states_common.loc[pd.DatetimeIndex(common[val_locs])].to_numpy(dtype=float)
    state_pred_oos = np.vstack(state_pred_parts) if state_pred_parts else np.zeros_like(state_true_oos)
    state_bench_oos = _make_oos_baseline(
        y_all,
        start_idx=decision_locs[0],
        end_idx=val_locs[-1],
        mode=state_baseline,
        random_walk_source=y_all if str(state_baseline).lower() == 'random_walk' else None,
    )
    r2_oos_state = _r2_per_dim(state_true_oos, state_pred_oos, state_bench_oos)
    r2_roll_state_q10, r2_roll_state_min = _window_r2_summary(
        state_true_oos,
        state_pred_oos,
        state_bench_oos,
        window=int(rolling_window),
        mode=str(window_mode),
    )

    return {
        'train_obs': int(np.mean(train_lengths)) if train_lengths else int(len(train_dates)),
        'train_obs_min': int(np.min(train_lengths)) if train_lengths else int(len(train_dates)),
        'train_obs_max': int(np.max(train_lengths)) if train_lengths else int(len(train_dates)),
        'val_obs': int(len(val_locs)),
        'r2_oos_ret_mean': float(np.mean(r2_oos_ret)),
        'r2_oos_ret_median': float(np.median(r2_oos_ret)),
        'r2_roll_ret_q10': float(r2_roll_ret_q10),
        'r2_roll_ret_min': float(r2_roll_ret_min),
        'r2_oos_state_mean': float(np.mean(r2_oos_state)),
        'r2_oos_state_median': float(np.median(r2_oos_state)),
        'r2_roll_state_q10': float(r2_roll_state_q10),
        'r2_roll_state_min': float(r2_roll_state_min),
    }



def _fit_selection_mean_model(
    *,
    states_t: pd.DataFrame,
    factor_returns_tp1: pd.DataFrame,
    factor_repr,
    lite_cfg: SelectionLitePPGDPOConfig,
):
    if str(lite_cfg.mean_model_kind) == 'factor_apt_regime':
        return fit_factor_apt_regime_mean(
            states_t=states_t,
            factor_returns_tp1=factor_returns_tp1,
            loadings=factor_repr.loadings,
            asset_alpha=factor_repr.asset_alpha,
            ridge_lambda=1.0e-6,
            regime_threshold_quantile=lite_cfg.regime_threshold_quantile,
            regime_sharpness=lite_cfg.regime_sharpness,
        )
    return fit_factor_apt_mean(
        states_t=states_t,
        factor_returns_tp1=factor_returns_tp1,
        loadings=factor_repr.loadings,
        asset_alpha=factor_repr.asset_alpha,
        ridge_lambda=1.0e-6,
    )


def _predict_asset_means_over_sample(mean_model, states_t: pd.DataFrame) -> np.ndarray:
    return np.vstack([mean_model.predict(states_t.iloc[j]).to_numpy(dtype=float) for j in range(len(states_t))])


def _resolve_regime_probability(cov_model: Any, mean_model, latest_factor_return: pd.Series | None) -> float:
    regime_prob = 0.0
    if hasattr(cov_model, 'regime_probability'):
        try:
            regime_prob = float(cov_model.regime_probability())
        except Exception:  # noqa: BLE001
            regime_prob = 0.0
    if (regime_prob <= 0.0) and getattr(mean_model, 'kind', '') == 'factor_apt_regime':
        try:
            regime_prob = float(mean_model.regime_probability(latest_factor_return))
        except Exception:  # noqa: BLE001
            regime_prob = 0.0
    return float(np.clip(regime_prob, 0.0, 1.0))


def _stabilize_covariance_for_loglik(cov: np.ndarray, *, min_eig: float = 1.0e-8) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        jitter = min_eig * np.eye(cov.shape[0], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(cov + jitter)
    eigvals = np.clip(eigvals, min_eig, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _gaussian_quasi_loglik_per_asset(residual: np.ndarray, covariance: np.ndarray) -> float:
    covariance = _stabilize_covariance_for_loglik(covariance)
    n_assets = max(1, int(covariance.shape[0]))
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        covariance = _stabilize_covariance_for_loglik(covariance, min_eig=1.0e-6)
        sign, logdet = np.linalg.slogdet(covariance)
    try:
        mahal = float(residual @ np.linalg.solve(covariance, residual))
    except np.linalg.LinAlgError:
        covariance = _stabilize_covariance_for_loglik(covariance, min_eig=1.0e-6)
        mahal = float(residual @ np.linalg.solve(covariance, residual))
    return float(-0.5 * ((logdet / n_assets) + (mahal / n_assets)))


def _build_selection_cov_model(lite_cfg: SelectionLitePPGDPOConfig):
    if lite_cfg.covariance_model_kind == 'asset_dcc':
        return AssetDCCCovariance(
            variance_floor=1.0e-6,
            correlation_shrink=0.10,
            dcc_alpha=0.02,
            dcc_beta=0.97,
            variance_lambda=0.97,
            asset_covariance_shrink=0.10,
        )
    if lite_cfg.covariance_model_kind == 'asset_adcc':
        return AssetADCCCovariance(
            variance_floor=1.0e-6,
            correlation_shrink=0.10,
            dcc_alpha=0.02,
            dcc_beta=0.97,
            adcc_gamma=float(lite_cfg.adcc_gamma),
            variance_lambda=0.97,
            asset_covariance_shrink=0.10,
        )
    if lite_cfg.covariance_model_kind == 'asset_regime_dcc':
        return AssetRegimeDCCCovariance(
            variance_floor=1.0e-6,
            correlation_shrink=0.10,
            dcc_alpha=0.02,
            dcc_beta=0.97,
            variance_lambda=0.97,
            asset_covariance_shrink=0.10,
            regime_threshold_quantile=float(lite_cfg.regime_threshold_quantile),
            regime_smoothing=float(lite_cfg.regime_smoothing),
            regime_sharpness=float(lite_cfg.regime_sharpness),
        )
    if lite_cfg.covariance_model_kind == 'constant':
        return ConstantFactorCovariance(
            variance_floor=1.0e-6,
            correlation_shrink=0.10,
            factor_correlation_mode=str(lite_cfg.factor_correlation_mode),
        )
    use_persistence = bool(lite_cfg.use_persistence)
    if lite_cfg.covariance_model_kind == 'state_only_diagonal':
        use_persistence = False
    return StateDiagonalFactorCovariance(
        ridge_lambda=1.0e-6,
        variance_floor=1.0e-6,
        correlation_shrink=0.10,
        factor_correlation_mode=str(lite_cfg.factor_correlation_mode),
        use_persistence=use_persistence,
    )



def _make_selection_lite_cfg(*, risk_aversion: float, lite_cfg: SelectionLitePPGDPOConfig):
    return SimpleNamespace(
        optimizer_backend=str(lite_cfg.optimizer_backend),
        policy=SimpleNamespace(
            risky_cap=float(lite_cfg.risky_cap),
            cash_floor=float(lite_cfg.cash_floor),
            risk_aversion=float(risk_aversion),
            pgd_steps=int(lite_cfg.pgd_steps),
            step_size=float(lite_cfg.step_size),
            turnover_penalty=float(lite_cfg.turnover_penalty),
        ),
        ppgdpo=SimpleNamespace(
            device=str(lite_cfg.device),
            hidden_dim=int(lite_cfg.hidden_dim),
            hidden_layers=int(lite_cfg.hidden_layers),
            epochs=int(lite_cfg.epochs),
            lr=float(lite_cfg.lr),
            utility=str(lite_cfg.utility),
            batch_size=int(lite_cfg.batch_size),
            horizon_steps=int(lite_cfg.horizon_steps),
            kappa=float(lite_cfg.kappa),
            mc_rollouts=int(lite_cfg.mc_rollouts),
            mc_sub_batch=int(lite_cfg.mc_sub_batch),
            clamp_min_return=float(lite_cfg.clamp_min_return),
            clamp_port_ret_max=float(lite_cfg.clamp_port_ret_max),
            clamp_wealth_min=float(lite_cfg.clamp_wealth_min),
            clamp_state_std_abs=lite_cfg.clamp_state_std_abs,
            covariance_mode=str(lite_cfg.covariance_mode),
            covariance_model_kind=str(lite_cfg.covariance_model_kind),
            mean_model_kind=str(lite_cfg.mean_model_kind),
            cross_policy_label=str(lite_cfg.cross_policy_label),
            cross_strength=float(lite_cfg.cross_strength),
            eps_bar=float(lite_cfg.eps_bar),
            newton_ridge=float(lite_cfg.newton_ridge),
            newton_tau=float(lite_cfg.newton_tau),
            newton_armijo=float(lite_cfg.newton_armijo),
            newton_backtrack=float(lite_cfg.newton_backtrack),
            max_newton=int(lite_cfg.max_newton),
            tol_grad=float(lite_cfg.tol_grad),
            max_line_search=int(lite_cfg.max_line_search),
            interior_margin=float(lite_cfg.interior_margin),
            clamp_neg_jxx_min=float(lite_cfg.clamp_neg_jxx_min),
            train_seed=int(lite_cfg.train_seed),
            state_ridge_lambda=float(lite_cfg.state_ridge_lambda),
            adcc_gamma=float(lite_cfg.adcc_gamma),
            regime_threshold_quantile=float(lite_cfg.regime_threshold_quantile),
            regime_smoothing=float(lite_cfg.regime_smoothing),
            regime_sharpness=float(lite_cfg.regime_sharpness),
        ),
        pipinn=SimpleNamespace(
            device=str(lite_cfg.pipinn_device),
            dtype=str(lite_cfg.pipinn_dtype),
            outer_iters=int(lite_cfg.pipinn_outer_iters),
            eval_epochs=int(lite_cfg.pipinn_eval_epochs),
            n_train_int=int(lite_cfg.pipinn_n_train_int),
            n_train_bc=int(lite_cfg.pipinn_n_train_bc),
            n_val_int=int(lite_cfg.pipinn_n_val_int),
            n_val_bc=int(lite_cfg.pipinn_n_val_bc),
            p_uniform=float(lite_cfg.pipinn_p_uniform),
            p_emp=float(lite_cfg.pipinn_p_emp),
            p_tau_head=float(lite_cfg.pipinn_p_tau_head),
            p_tau_near0=float(lite_cfg.pipinn_p_tau_near0),
            tau_head_window=int(lite_cfg.pipinn_tau_head_window),
            lr=float(lite_cfg.pipinn_lr),
            grad_clip=float(lite_cfg.pipinn_grad_clip),
            w_bc=float(lite_cfg.pipinn_w_bc),
            w_bc_dx=float(lite_cfg.pipinn_w_bc_dx),
            scheduler_factor=float(lite_cfg.pipinn_scheduler_factor),
            scheduler_patience=int(lite_cfg.pipinn_scheduler_patience),
            min_lr=float(lite_cfg.pipinn_min_lr),
            width=int(lite_cfg.pipinn_width),
            depth=int(lite_cfg.pipinn_depth),
            covariance_train_mode=str(lite_cfg.pipinn_covariance_train_mode),
            ansatz_mode=str(lite_cfg.pipinn_ansatz_mode),
            policy_output_mode=str(lite_cfg.pipinn_policy_output_mode),
            emit_frozen_traincov_strategy=bool(lite_cfg.pipinn_emit_frozen_traincov_strategy),
            save_training_logs=bool(lite_cfg.pipinn_save_training_logs),
            show_progress=bool(lite_cfg.pipinn_show_progress),
            show_epoch_progress=bool(lite_cfg.pipinn_show_epoch_progress),
        ),
    )


def _pipinn_payload_from_lite_cfg(lite_cfg: SelectionLitePPGDPOConfig) -> dict[str, Any]:
    return {
        'device': str(lite_cfg.pipinn_device),
        'dtype': str(lite_cfg.pipinn_dtype),
        'outer_iters': int(lite_cfg.pipinn_outer_iters),
        'eval_epochs': int(lite_cfg.pipinn_eval_epochs),
        'n_train_int': int(lite_cfg.pipinn_n_train_int),
        'n_train_bc': int(lite_cfg.pipinn_n_train_bc),
        'n_val_int': int(lite_cfg.pipinn_n_val_int),
        'n_val_bc': int(lite_cfg.pipinn_n_val_bc),
        'p_uniform': float(lite_cfg.pipinn_p_uniform),
        'p_emp': float(lite_cfg.pipinn_p_emp),
        'p_tau_head': float(lite_cfg.pipinn_p_tau_head),
        'p_tau_near0': float(lite_cfg.pipinn_p_tau_near0),
        'tau_head_window': int(lite_cfg.pipinn_tau_head_window),
        'lr': float(lite_cfg.pipinn_lr),
        'grad_clip': float(lite_cfg.pipinn_grad_clip),
        'w_bc': float(lite_cfg.pipinn_w_bc),
        'w_bc_dx': float(lite_cfg.pipinn_w_bc_dx),
        'scheduler_factor': float(lite_cfg.pipinn_scheduler_factor),
        'scheduler_patience': int(lite_cfg.pipinn_scheduler_patience),
        'min_lr': float(lite_cfg.pipinn_min_lr),
        'width': int(lite_cfg.pipinn_width),
        'depth': int(lite_cfg.pipinn_depth),
        'covariance_train_mode': str(lite_cfg.pipinn_covariance_train_mode),
        'ansatz_mode': str(lite_cfg.pipinn_ansatz_mode),
        'policy_output_mode': str(lite_cfg.pipinn_policy_output_mode),
        'emit_frozen_traincov_strategy': bool(lite_cfg.pipinn_emit_frozen_traincov_strategy),
        'save_training_logs': bool(lite_cfg.pipinn_save_training_logs),
        'show_progress': bool(lite_cfg.pipinn_show_progress),
        'show_epoch_progress': bool(lite_cfg.pipinn_show_epoch_progress),
    }

def _evaluate_ppgdpo_lite_candidate_block(
    *,
    states_full: pd.DataFrame,
    factors_full: pd.DataFrame,
    returns_full: pd.DataFrame,
    block: dict[str, Any],
    risk_aversion: float,
    lite_cfg: SelectionLitePPGDPOConfig,
) -> dict[str, Any]:
    train_pairs = _paired_block_data(
        states_full=states_full,
        factors_full=factors_full,
        returns_full=returns_full,
        raw_dates=block['train_dates'],
    )
    val_pairs = _paired_block_data(
        states_full=states_full,
        factors_full=factors_full,
        returns_full=returns_full,
        raw_dates=block['val_dates'],
    )
    if len(train_pairs.states_t) < 24:
        raise ValueError('Not enough training pairs for PPGDPO-lite rerank')
    if len(val_pairs.states_t) < 1:
        raise ValueError('No validation pairs for PPGDPO-lite rerank')

    factor_cols = list(train_pairs.factors_tp1.columns)
    extractor = ProvidedFactorExtractor(factor_cols)
    factor_repr = extractor.fit(train_pairs.returns_tp1, train_pairs.factors_tp1)
    mean_model = fit_factor_apt_mean(
        states_t=train_pairs.states_t,
        factor_returns_tp1=train_pairs.factors_tp1,
        loadings=factor_repr.loadings,
        asset_alpha=factor_repr.asset_alpha,
        ridge_lambda=1.0e-6,
    )
    mu_train_pred = _predict_asset_means_over_sample(mean_model, train_pairs.states_t)
    cov_model = _build_selection_cov_model(lite_cfg)
    cov_model.fit(
        train_pairs.states_t,
        train_pairs.factors_tp1,
        asset_returns_tp1=train_pairs.returns_tp1,
        asset_mean_pred=mu_train_pred,
    )
    transition = fit_state_transition(train_pairs.states_t, train_pairs.states_tp1, ridge_lambda=lite_cfg.state_ridge_lambda)
    cross_est = estimate_return_state_cross(
        returns_tp1=train_pairs.returns_tp1,
        returns_mean_pred=mu_train_pred,
        states_t=train_pairs.states_t,
        states_tp1=train_pairs.states_tp1,
        transition=transition,
    )
    trainer = train_warmup_policy(
        train_pairs.states_t,
        train_pairs.returns_tp1,
        _make_selection_lite_cfg(risk_aversion=risk_aversion, lite_cfg=lite_cfg),
        transaction_cost=float(lite_cfg.transaction_cost_bps) / 10000.0,
        mean_model=mean_model,
        transition=transition,
        cross_est=cross_est,
    )

    tc = float(lite_cfg.transaction_cost_bps) / 10000.0
    prev_weights = {
        'myopic': np.zeros(returns_full.shape[1], dtype=float),
        'ppgdpo_est': np.zeros(returns_full.shape[1], dtype=float),
        'ppgdpo_zero': np.zeros(returns_full.shape[1], dtype=float),
    }
    rows: list[dict[str, Any]] = []
    cov_loglik_scores: list[float] = []
    prev_mu_for_cov_update: np.ndarray | None = None
    cross_arr = cross_est.cross.to_numpy(dtype=float)
    zero_arr = np.zeros_like(cross_arr)

    for i in range(len(val_pairs.states_t)):
        if prev_mu_for_cov_update is not None:
            cov_model.update_with_realized(val_pairs.returns_tp1.iloc[i - 1].to_numpy(dtype=float), prev_mu_for_cov_update)
        state_row = val_pairs.states_t.iloc[i]
        mu = mean_model.predict(state_row).to_numpy(dtype=float)
        latest_factor_return = val_pairs.factors_t.iloc[i]
        cov_fc = cov_model.predict(state_row, latest_factor_return, factor_repr.loadings, factor_repr.residual_var)
        cov_eval = cov_fc.asset_cov
        if lite_cfg.covariance_mode == 'diag':
            cov_eval = np.diag(np.diag(cov_eval))

        policy_w = trainer.policy_weights(state_row)
        costates = trainer.estimate_costates(state_row)
        myopic_w = solve_mean_variance(
            mu,
            cov_eval,
            risk_aversion,
            lite_cfg.risky_cap,
            steps=max(100, lite_cfg.pgd_steps),
            step_size=lite_cfg.step_size,
        )
        ppgdpo_est_w, _ = solve_ppgdpo_projection(
            mu=mu,
            cov=cov_eval,
            cross_mat=cross_arr,
            costates=costates,
            risky_cap=lite_cfg.risky_cap,
            cash_floor=lite_cfg.cash_floor,
            wealth=1.0,
            cross_scale=lite_cfg.cross_strength,
            eps_bar=lite_cfg.eps_bar,
            ridge=lite_cfg.newton_ridge,
            tau=lite_cfg.newton_tau,
            armijo=lite_cfg.newton_armijo,
            backtrack=lite_cfg.newton_backtrack,
            max_newton=lite_cfg.max_newton,
            tol_grad=lite_cfg.tol_grad,
            max_ls=lite_cfg.max_line_search,
            interior_margin=lite_cfg.interior_margin,
            clamp_neg_jxx_min=lite_cfg.clamp_neg_jxx_min,
        )
        ppgdpo_zero_w, _ = solve_ppgdpo_projection(
            mu=mu,
            cov=cov_eval,
            cross_mat=zero_arr,
            costates=costates,
            risky_cap=lite_cfg.risky_cap,
            cash_floor=lite_cfg.cash_floor,
            wealth=1.0,
            cross_scale=lite_cfg.cross_strength,
            eps_bar=lite_cfg.eps_bar,
            ridge=lite_cfg.newton_ridge,
            tau=lite_cfg.newton_tau,
            armijo=lite_cfg.newton_armijo,
            backtrack=lite_cfg.newton_backtrack,
            max_newton=lite_cfg.max_newton,
            tol_grad=lite_cfg.tol_grad,
            max_ls=lite_cfg.max_line_search,
            interior_margin=lite_cfg.interior_margin,
            clamp_neg_jxx_min=lite_cfg.clamp_neg_jxx_min,
        )
        realized_ret = val_pairs.returns_tp1.iloc[i].to_numpy(dtype=float)
        residual_ret = realized_ret - mu
        cov_loglik_scores.append(_gaussian_quasi_loglik_per_asset(residual_ret, cov_eval))

        for strategy, w in {
            'myopic': myopic_w,
            'ppgdpo_est': ppgdpo_est_w,
            'ppgdpo_zero': ppgdpo_zero_w,
        }.items():
            prev_w = prev_weights[strategy]
            turnover = float(np.abs(w - prev_w).sum())
            gross = float(w @ realized_ret)
            net = gross - tc * turnover
            rows.append({
                'strategy': strategy,
                'net_return': net,
                'turnover': turnover,
            })
            prev_weights[strategy] = w
        prev_mu_for_cov_update = mu.copy()

    monthly = pd.DataFrame(rows)
    out: dict[str, Any] = {
        'months': int(len(val_pairs.states_t)),
        'train_pairs': int(len(train_pairs.states_t)),
        'train_objective': float(trainer.train_objective),
    }
    for strategy in ('myopic', 'ppgdpo_est', 'ppgdpo_zero'):
        series = monthly.loc[monthly['strategy'] == strategy, 'net_return'].astype(float)
        out[f'ce_{strategy}'] = certainty_equivalent_annual(series, risk_aversion)
        out[f'sharpe_{strategy}'] = sharpe_ratio(series)
        out[f'avg_turnover_{strategy}'] = float(monthly.loc[monthly['strategy'] == strategy, 'turnover'].mean())
    out['ce_gain_vs_myopic'] = float(out['ce_ppgdpo_est'] - out['ce_myopic'])
    out['ce_gain_vs_zero'] = float(out['ce_ppgdpo_est'] - out['ce_ppgdpo_zero'])
    out['resid_cov_loglik'] = float(np.nanmean(cov_loglik_scores)) if cov_loglik_scores else np.nan
    # Stage35 real rerank anchor: use the actual stage-2 PPGDPO score so hedging gains
    # enter the selection path instead of being ignored in favor of myopic CE.
    out['ppgdpo_lite_score'] = _selection_score_ppgdpo_lite(
        out['ce_ppgdpo_est'],
        out['ce_gain_vs_myopic'],
        out['ce_gain_vs_zero'],
    )
    return out



def _evaluate_stage2_protocol_covariance_block(
    *,
    selection_unit_id: str,
    candidate: FactorZooCandidate,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    ff3: pd.DataFrame,
    ff5: pd.DataFrame,
    bond: pd.DataFrame,
    block: dict[str, Any],
    protocol: str,
    risk_aversion: float,
    lite_cfg: SelectionLitePPGDPOConfig,
    split_payload: dict[str, Any],
    config_stem: str,
    output_root: Path,
) -> dict[str, Any]:
    panels = build_candidate_panels(
        candidate,
        returns=returns,
        macro=macro,
        ff3=ff3,
        ff5=ff5,
        bond=bond,
        train_dates=block['train_dates'],
    )
    factors_full = panels['factors']
    states_full = panels['states']
    common = returns.index.intersection(states_full.index).intersection(factors_full.index)
    if len(common) < 24:
        raise ValueError('Not enough aligned rows for stage2 protocol evaluation')
    returns_eval = returns.loc[common].copy()
    states_eval = states_full.loc[common].copy()
    factors_eval = factors_full.loc[common].copy()

    protocol_payload = _protocol_row_payload(protocol)
    unit_id = str(selection_unit_id)
    block_label = str(block.get('label') or 'selection_block')
    eval_dir = ensure_dir(output_root / unit_id / block_label / str(lite_cfg.covariance_label))
    returns_csv = eval_dir / 'returns_panel.csv'
    states_csv = eval_dir / 'states_panel.csv'
    factors_csv = eval_dir / 'factors_panel.csv'
    returns_eval.reset_index(names='date').to_csv(returns_csv, index=False)
    states_eval.reset_index(names='date').to_csv(states_csv, index=False)
    factors_eval.reset_index(names='date').to_csv(factors_csv, index=False)

    base_cov_label = str(lite_cfg.covariance_label).split('__', 1)[0]
    cov_payload = _config_covariance_payload_from_label(base_cov_label)
    cfg_payload = _build_v2_config_dict(
        out_dir=eval_dir,
        config_stem=f'{config_stem}_{candidate.name}_{protocol}_{lite_cfg.covariance_label}',
        split_payload=split_payload,
        state_cols=list(states_eval.columns),
        factor_cols=list(factors_eval.columns),
        risk_aversion=risk_aversion,
        covariance_model_kind=str(cov_payload['kind']),
        covariance_factor_correlation_mode=str(cov_payload['factor_correlation_mode']),
        covariance_use_persistence=bool(cov_payload['use_persistence']),
        covariance_adcc_gamma=float(cov_payload.get('adcc_gamma', 0.005)),
        covariance_regime_threshold_quantile=float(cov_payload.get('regime_threshold_quantile', 0.75)),
        covariance_regime_smoothing=float(cov_payload.get('regime_smoothing', 0.90)),
        covariance_regime_sharpness=float(cov_payload.get('regime_sharpness', 8.0)),
        ppgdpo_covariance_mode=str(lite_cfg.covariance_mode),
        ppgdpo_mc_rollouts=int(lite_cfg.mc_rollouts),
        ppgdpo_mc_sub_batch=int(lite_cfg.mc_sub_batch),
        mean_model_kind=str(lite_cfg.mean_model_kind),
        comparison_cross_modes=_comparison_cross_modes_for_covariance_label(base_cov_label),
        optimizer_backend=str(lite_cfg.optimizer_backend),
        pipinn_payload=_pipinn_payload_from_lite_cfg(lite_cfg),
    )
    cfg = Config.model_validate(cfg_payload)
    cfg = _apply_selection_lite_runtime_overrides(cfg, lite_cfg)
    cfg = _set_config_window_from_block(cfg, block)
    cfg = apply_oos_protocol(cfg, protocol)
    train_dates_raw = block.get('train_dates', None)
    train_dates = pd.DatetimeIndex([] if train_dates_raw is None else train_dates_raw)
    if cfg.split.train_window_mode == 'rolling' and cfg.split.rolling_train_months is not None:
        cfg.split.min_train_months = max(
            12,
            min(int(cfg.split.rolling_train_months), int(cfg.split.min_train_months), len(train_dates)),
        )
    else:
        cfg.split.min_train_months = max(12, min(int(cfg.split.min_train_months), len(train_dates)))
    cfg.project.output_dir = eval_dir / 'outputs'
    cfg.project.name = f'{cfg.project.name}_{block_label}_{protocol}_{lite_cfg.covariance_label}'

    artifacts = run_experiment(cfg)
    summary = pd.read_csv(artifacts.summary_with_costs)
    metrics = _extract_validation_protocol_metrics(summary, cross_mode=str(lite_cfg.cross_policy_label))
    return {
        'selection_unit_id': unit_id,
        'spec': str(candidate.name),
        **protocol_payload,
        'model_id': f'{unit_id}__{lite_cfg.covariance_label}',
        'stage2_model_label': str(lite_cfg.covariance_label),
        'covariance_model_label': base_cov_label,
        'covariance_model_kind': str(lite_cfg.covariance_model_kind),
        'mean_model_kind': str(lite_cfg.mean_model_kind),
        'cross_policy_label': str(lite_cfg.cross_policy_label),
        'optimizer_backend': str(lite_cfg.optimizer_backend),
        'block': block_label,
        'selection_split_mode': str(block.get('selection_split_mode') or ''),
        'train_start_date': str(pd.DatetimeIndex(block['train_dates'])[0].date()),
        'train_end_date': str(pd.DatetimeIndex(block['train_dates'])[-1].date()),
        'val_start_date': str(pd.DatetimeIndex(block['val_dates'])[0].date()),
        'val_end_date': str(pd.DatetimeIndex(block['val_dates'])[-1].date()),
        'train_obs': int(len(train_dates)),
        **metrics,
    }



def _normalize_rolling_oos_window_grid(grid: list[int] | tuple[int, ...] | None) -> list[int]:
    payload = list(grid) if grid is not None else [240]
    out: list[int] = []
    seen: set[int] = set()
    for value in payload:
        try:
            months = int(value)
        except Exception:  # noqa: BLE001
            continue
        if months <= 0 or months in seen:
            continue
        seen.add(months)
        out.append(months)
    if not out:
        out = [240]
    return sorted(out)



def _apply_selection_lite_runtime_overrides(cfg: Config, lite_cfg: SelectionLitePPGDPOConfig) -> Config:
    out = cfg.model_copy(deep=True)
    out.optimizer_backend = str(lite_cfg.optimizer_backend)
    out.policy.risky_cap = float(lite_cfg.risky_cap)
    out.policy.cash_floor = float(lite_cfg.cash_floor)
    out.policy.pgd_steps = int(lite_cfg.pgd_steps)
    out.policy.step_size = float(lite_cfg.step_size)
    out.policy.turnover_penalty = float(lite_cfg.turnover_penalty)
    out.ppgdpo.device = str(lite_cfg.device)
    out.ppgdpo.hidden_dim = int(lite_cfg.hidden_dim)
    out.ppgdpo.hidden_layers = int(lite_cfg.hidden_layers)
    out.ppgdpo.epochs = int(lite_cfg.epochs)
    out.ppgdpo.lr = float(lite_cfg.lr)
    out.ppgdpo.utility = str(lite_cfg.utility)
    out.ppgdpo.batch_size = int(lite_cfg.batch_size)
    out.ppgdpo.horizon_steps = int(lite_cfg.horizon_steps)
    out.ppgdpo.kappa = float(lite_cfg.kappa)
    out.ppgdpo.mc_rollouts = int(lite_cfg.mc_rollouts)
    out.ppgdpo.mc_sub_batch = int(lite_cfg.mc_sub_batch)
    out.ppgdpo.clamp_min_return = float(lite_cfg.clamp_min_return)
    out.ppgdpo.clamp_port_ret_max = float(lite_cfg.clamp_port_ret_max)
    out.ppgdpo.clamp_wealth_min = float(lite_cfg.clamp_wealth_min)
    out.ppgdpo.clamp_state_std_abs = lite_cfg.clamp_state_std_abs
    out.ppgdpo.covariance_mode = str(lite_cfg.covariance_mode)
    out.ppgdpo.cross_strength = float(lite_cfg.cross_strength)
    out.ppgdpo.eps_bar = float(lite_cfg.eps_bar)
    out.ppgdpo.newton_ridge = float(lite_cfg.newton_ridge)
    out.ppgdpo.newton_tau = float(lite_cfg.newton_tau)
    out.ppgdpo.newton_armijo = float(lite_cfg.newton_armijo)
    out.ppgdpo.newton_backtrack = float(lite_cfg.newton_backtrack)
    out.ppgdpo.max_newton = int(lite_cfg.max_newton)
    out.ppgdpo.tol_grad = float(lite_cfg.tol_grad)
    out.ppgdpo.max_line_search = int(lite_cfg.max_line_search)
    out.ppgdpo.interior_margin = float(lite_cfg.interior_margin)
    out.ppgdpo.clamp_neg_jxx_min = float(lite_cfg.clamp_neg_jxx_min)
    out.ppgdpo.train_seed = int(lite_cfg.train_seed)
    out.ppgdpo.state_ridge_lambda = float(lite_cfg.state_ridge_lambda)
    out.mean_model.kind = str(lite_cfg.mean_model_kind)
    out.comparison.cross_modes = _comparison_cross_modes_for_covariance_label(str(lite_cfg.covariance_label))
    out.comparison.transaction_cost_bps = float(lite_cfg.transaction_cost_bps)
    if hasattr(out, 'pipinn'):
        out.pipinn.device = str(lite_cfg.pipinn_device)
        out.pipinn.dtype = str(lite_cfg.pipinn_dtype)
        out.pipinn.outer_iters = int(lite_cfg.pipinn_outer_iters)
        out.pipinn.eval_epochs = int(lite_cfg.pipinn_eval_epochs)
        out.pipinn.n_train_int = int(lite_cfg.pipinn_n_train_int)
        out.pipinn.n_train_bc = int(lite_cfg.pipinn_n_train_bc)
        out.pipinn.n_val_int = int(lite_cfg.pipinn_n_val_int)
        out.pipinn.n_val_bc = int(lite_cfg.pipinn_n_val_bc)
        out.pipinn.p_uniform = float(lite_cfg.pipinn_p_uniform)
        out.pipinn.p_emp = float(lite_cfg.pipinn_p_emp)
        out.pipinn.p_tau_head = float(lite_cfg.pipinn_p_tau_head)
        out.pipinn.p_tau_near0 = float(lite_cfg.pipinn_p_tau_near0)
        out.pipinn.tau_head_window = int(lite_cfg.pipinn_tau_head_window)
        out.pipinn.lr = float(lite_cfg.pipinn_lr)
        out.pipinn.grad_clip = float(lite_cfg.pipinn_grad_clip)
        out.pipinn.w_bc = float(lite_cfg.pipinn_w_bc)
        out.pipinn.w_bc_dx = float(lite_cfg.pipinn_w_bc_dx)
        out.pipinn.scheduler_factor = float(lite_cfg.pipinn_scheduler_factor)
        out.pipinn.scheduler_patience = int(lite_cfg.pipinn_scheduler_patience)
        out.pipinn.min_lr = float(lite_cfg.pipinn_min_lr)
        out.pipinn.width = int(lite_cfg.pipinn_width)
        out.pipinn.depth = int(lite_cfg.pipinn_depth)
        out.pipinn.covariance_train_mode = str(lite_cfg.pipinn_covariance_train_mode)
        out.pipinn.ansatz_mode = str(lite_cfg.pipinn_ansatz_mode)
        out.pipinn.policy_output_mode = str(lite_cfg.pipinn_policy_output_mode)
        out.pipinn.emit_frozen_traincov_strategy = bool(lite_cfg.pipinn_emit_frozen_traincov_strategy)
        out.pipinn.save_training_logs = bool(lite_cfg.pipinn_save_training_logs)
        out.pipinn.show_progress = bool(lite_cfg.pipinn_show_progress)
        out.pipinn.show_epoch_progress = bool(lite_cfg.pipinn_show_epoch_progress)
    return out

def _set_config_window_from_block(cfg: Config, block: dict[str, Any]) -> Config:
    out = cfg.model_copy(deep=True)
    train_dates = pd.DatetimeIndex(block.get('train_dates', []))
    val_dates = pd.DatetimeIndex(block.get('val_dates', []))
    if len(train_dates) == 0 or len(val_dates) == 0:
        raise ValueError('selection block is missing train_dates or val_dates')
    out.split.train_start = train_dates[0].date()
    out.split.fixed_train_end = train_dates[-1].date()
    out.split.test_start = val_dates[0].date()
    out.split.end_date = val_dates[-1].date()
    out.split.min_train_months = max(12, min(int(out.split.min_train_months), len(train_dates)))
    return out

def _ppgdpo_strategy_for_cross_mode(cross_mode: str) -> str:
    """Map a cross_mode label to the canonical PPGDPO strategy name.

    `_strategy_metadata` writes the projected strategy as 'ppgdpo' / 'pipinn'
    only when cross_mode='estimated'. For 'zero' and 'regime_gated' the
    strategy column is suffixed ('ppgdpo_zero', 'pipinn_zero', ...), so the
    summary lookup must use the suffixed name to hit the alias table.
    """
    cm = str(cross_mode).lower()
    if cm in {'zero', 'regime_gated'}:
        return f'ppgdpo_{cm}'
    return 'ppgdpo'

def _summary_scalar(summary: pd.DataFrame, *, strategy: str, cross_mode: str, column: str) -> float:
    if summary.empty or column not in summary.columns:
        return float('nan')

    strategy = str(strategy)
    cross_mode = str(cross_mode)
    strategy_aliases = {
        'myopic': ['myopic', 'predictive_static'],
        'predictive_static': ['predictive_static', 'myopic'],
        'policy': ['policy', 'pgdpo'],
        'pgdpo': ['pgdpo', 'policy'],
        'ppgdpo': ['ppgdpo', 'pipinn'],
        'ppgdpo_zero': ['ppgdpo_zero', 'pipinn_zero'],
        'ppgdpo_regime_gated': ['ppgdpo_regime_gated', 'pipinn_regime_gated'],
        'pipinn': ['pipinn', 'ppgdpo'],
        'pipinn_zero': ['pipinn_zero', 'ppgdpo_zero'],
        'pipinn_regime_gated': ['pipinn_regime_gated', 'ppgdpo_regime_gated'],
    }
    cross_aliases = {
        'estimated': ['estimated', 'reference'],
        'reference': ['reference', 'estimated'],
        'zero': ['zero'],
        'regime_gated': ['regime_gated'],
        'benchmark': ['benchmark'],
    }
    strategy_candidates = strategy_aliases.get(strategy, [strategy])
    cross_candidates = cross_aliases.get(cross_mode, [cross_mode])

    for strategy_name in strategy_candidates:
        for cross_name in cross_candidates:
            mask = pd.Series(False, index=summary.index)
            if 'strategy' in summary.columns:
                mask = mask | (summary['strategy'] == str(strategy_name))
            if 'strategy_legacy_label' in summary.columns:
                mask = mask | (summary['strategy_legacy_label'] == str(strategy_name))
            if 'strategy_display' in summary.columns:
                mask = mask | (summary['strategy_display'] == str(strategy_name))
            subset = summary.loc[mask & (summary['cross_mode'] == str(cross_name)), column]
            if subset.empty:
                continue
            value = subset.iloc[0]
            return float(value) if pd.notna(value) else float('nan')
    return float('nan')

def _extract_validation_protocol_metrics(summary: pd.DataFrame, *, cross_mode: str = 'estimated') -> dict[str, Any]:
    cross_mode = str(cross_mode)
    # The projected candidate strategy is written with a cross_mode-dependent
    # suffix (e.g. 'ppgdpo_zero' / 'pipinn_zero'). We must query the
    # summary with the suffixed name so that the alias table kicks in.
    candidate_strategy = _ppgdpo_strategy_for_cross_mode(cross_mode)
    ce_candidate = _summary_scalar(summary, strategy=candidate_strategy, cross_mode=cross_mode, column='cer_ann')
    ce_zero = _summary_scalar(summary, strategy='ppgdpo_zero', cross_mode='zero', column='cer_ann')
    ce_predictive_static = _summary_scalar(summary, strategy='predictive_static', cross_mode='reference', column='cer_ann')
    benchmark_ce = {
        benchmark_name: _summary_scalar(summary, strategy=benchmark_name, cross_mode='benchmark', column='cer_ann')
        for benchmark_name in SELECTION_REFERENCE_BENCHMARKS
    }

    def _delta_vs(reference_value: float) -> float:
        if np.isfinite(ce_candidate) and np.isfinite(reference_value):
            return float(ce_candidate - reference_value)
        return np.nan

    score = _selection_score_ppgdpo_lite(
        ce_candidate,
        _delta_vs(ce_predictive_static),
        _delta_vs(ce_zero),
    )
    out = {
        'validation_months': _summary_scalar(summary, strategy=candidate_strategy, cross_mode=cross_mode, column='months'),
        'validation_ce_est': ce_candidate,
        'validation_ce_predictive_static': ce_predictive_static,
        'validation_ce_myopic': ce_predictive_static,
        'validation_ce_zero': ce_zero,
        'validation_ce_delta_predictive_static': _delta_vs(ce_predictive_static),
        'validation_ce_delta_myopic': _delta_vs(ce_predictive_static),
        'validation_ce_delta_zero': _delta_vs(ce_zero),
        'validation_sharpe_est': _summary_scalar(summary, strategy=candidate_strategy, cross_mode=cross_mode, column='sharpe'),
        'validation_turnover_est': _summary_scalar(summary, strategy=candidate_strategy, cross_mode=cross_mode, column='avg_turnover'),
        'validation_max_drawdown_est': _summary_scalar(summary, strategy=candidate_strategy, cross_mode=cross_mode, column='max_drawdown'),
        'validation_score': score,
        'validation_cross_mode': cross_mode,
    }
    for benchmark_name, reference_value in benchmark_ce.items():
        out[f'validation_ce_{benchmark_name}'] = reference_value
        out[f'validation_ce_delta_{benchmark_name}'] = _delta_vs(reference_value)
    return out


def _evaluate_validation_protocol_block(
    *,
    entry: dict[str, Any],
    block: dict[str, Any],
    protocol: str,
    lite_cfg: SelectionLitePPGDPOConfig,
    output_root: Path,
) -> dict[str, Any]:
    rank = int(entry['rank'])
    model_id = str(entry.get('model_id') or entry.get('spec') or f'rank_{rank:03d}')
    cfg_payload = yaml.safe_load(Path(entry['config_yaml']).read_text(encoding='utf-8')) or {}
    cfg = Config.model_validate(cfg_payload)
    cfg = _apply_selection_lite_runtime_overrides(cfg, lite_cfg)
    cfg = _set_config_window_from_block(cfg, block)
    cfg = apply_oos_protocol(cfg, protocol, entry=entry)
    train_dates_raw = block.get('train_dates', None)
    train_dates = pd.DatetimeIndex([] if train_dates_raw is None else train_dates_raw)
    if cfg.split.train_window_mode == 'rolling' and cfg.split.rolling_train_months is not None:
        cfg.split.min_train_months = max(12, min(int(cfg.split.rolling_train_months), int(cfg.split.min_train_months), len(train_dates)))
    else:
        cfg.split.min_train_months = max(12, min(int(cfg.split.min_train_months), len(train_dates)))
    block_label = str(block.get('label') or 'validation_block')
    cfg.project.output_dir = output_root / f'rank_{rank:03d}' / model_id / block_label / str(protocol)
    cfg.project.name = f'{cfg.project.name}_{block_label}_{protocol}'
    artifacts = run_experiment(cfg)
    summary = pd.read_csv(artifacts.summary_with_costs)
    metrics = _extract_validation_protocol_metrics(summary, cross_mode=str(lite_cfg.cross_policy_label))
    return {
        'rank': rank,
        'spec': str(entry['spec']),
        'model_id': model_id,
        'oos_protocol': str(protocol),
        'block': block_label,
        'train_window_mode': str(cfg.split.train_window_mode),
        'rolling_train_months': int(cfg.split.rolling_train_months) if cfg.split.rolling_train_months is not None else np.nan,
        'refit_every': int(cfg.split.refit_every),
        'rebalance_every': int(cfg.split.rebalance_every),
        'train_start_date': str(pd.DatetimeIndex(block['train_dates'])[0].date()),
        'train_end_date': str(pd.DatetimeIndex(block['train_dates'])[-1].date()),
        'val_start_date': str(pd.DatetimeIndex(block['val_dates'])[0].date()),
        'val_end_date': str(pd.DatetimeIndex(block['val_dates'])[-1].date()),
        **metrics,
        'error': None,
    }


def _evaluate_validation_protocols_for_entries(
    *,
    entries: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    lite_cfg: SelectionLitePPGDPOConfig,
    rolling_oos_window_grid: list[int] | tuple[int, ...],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], int]:
    protocol_names = ['fixed', 'expanding_annual', *[f'rolling{int(months)}m_annual' for months in rolling_oos_window_grid]]
    fallback_months = int(max(rolling_oos_window_grid)) if rolling_oos_window_grid else 240
    block_rows: list[dict[str, Any]] = []
    validation_root = ensure_dir(output_root / 'selection' / 'oos_protocol_validation')
    for entry in entries:
        for block in blocks:
            for protocol in protocol_names:
                try:
                    block_rows.append(
                        _evaluate_validation_protocol_block(
                            entry=entry,
                            block=block,
                            protocol=protocol,
                            lite_cfg=lite_cfg,
                            output_root=validation_root,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    train_dates = pd.DatetimeIndex(block.get('train_dates', []))
                    val_dates = pd.DatetimeIndex(block.get('val_dates', []))
                    rolling_months = np.nan
                    if str(protocol).startswith('rolling'):
                        try:
                            rolling_months = int(''.join(ch for ch in str(protocol) if ch.isdigit()))
                        except Exception:  # noqa: BLE001
                            rolling_months = np.nan
                    block_rows.append({
                        'rank': int(entry['rank']),
                        'spec': str(entry['spec']),
                        'model_id': str(entry.get('model_id') or entry.get('spec')),
                        'oos_protocol': str(protocol),
                        'block': str(block.get('label') or 'validation_block'),
                        'train_window_mode': 'rolling' if str(protocol).startswith('rolling') else ('expanding' if str(protocol) == 'expanding_annual' else 'fixed'),
                        'rolling_train_months': rolling_months,
                        'refit_every': 12 if str(protocol) != 'fixed' else 1,
                        'rebalance_every': 1,
                        'train_start_date': str(train_dates[0].date()) if len(train_dates) else None,
                        'train_end_date': str(train_dates[-1].date()) if len(train_dates) else None,
                        'val_start_date': str(val_dates[0].date()) if len(val_dates) else None,
                        'val_end_date': str(val_dates[-1].date()) if len(val_dates) else None,
                        'validation_months': np.nan,
                        'validation_ce_est': np.nan,
                        'validation_ce_predictive_static': np.nan,
                        'validation_ce_myopic': np.nan,
                        'validation_ce_zero': np.nan,
                        'validation_ce_delta_predictive_static': np.nan,
                        'validation_ce_delta_myopic': np.nan,
                        'validation_ce_delta_zero': np.nan,
                        'validation_sharpe_est': np.nan,
                        'validation_turnover_est': np.nan,
                        'validation_max_drawdown_est': np.nan,
                        'validation_score': np.nan,
                        'error': str(exc),
                    })
    blocks_df = pd.DataFrame(block_rows)
    summary_rows: list[dict[str, Any]] = []
    if not blocks_df.empty:
        group_cols = ['rank', 'spec', 'model_id', 'oos_protocol', 'train_window_mode', 'rolling_train_months']
        for keys, grp in blocks_df.groupby(group_cols, dropna=False, sort=False):
            valid = grp.loc[np.isfinite(pd.to_numeric(grp['validation_score'], errors='coerce'))].copy()
            def _mean(col: str) -> float:
                vals = pd.to_numeric(valid[col], errors='coerce')
                return float(np.nanmean(vals)) if len(valid) else np.nan
            def _q10(col: str) -> float:
                vals = pd.to_numeric(valid[col], errors='coerce')
                arr = vals[np.isfinite(vals)]
                return float(np.nanquantile(arr, 0.10)) if len(arr) else np.nan
            rank, spec, model_id, oos_protocol, train_window_mode, rolling_train_months = keys
            summary_rows.append({
                'rank': int(rank),
                'spec': str(spec),
                'model_id': str(model_id),
                'oos_protocol': str(oos_protocol),
                'train_window_mode': str(train_window_mode),
                'rolling_train_months': float(rolling_train_months) if pd.notna(rolling_train_months) else np.nan,
                'validation_blocks_total': int(len(grp)),
                'validation_blocks_valid': int(len(valid)),
                'validation_score_mean': _mean('validation_score'),
                'validation_score_q10': _q10('validation_score'),
                'validation_ce_mean': _mean('validation_ce_est'),
                'validation_ce_delta_predictive_static_mean': _mean('validation_ce_delta_predictive_static'),
                'validation_ce_delta_myopic_mean': _mean('validation_ce_delta_myopic'),
                'validation_ce_delta_zero_mean': _mean('validation_ce_delta_zero'),
                'validation_sharpe_mean': _mean('validation_sharpe_est'),
                'validation_turnover_mean': _mean('validation_turnover_est'),
                'validation_max_drawdown_mean': _mean('validation_max_drawdown_est'),
                'validation_months_mean': _mean('validation_months'),
                'validation_error_count': int(grp['error'].notna().sum()) if 'error' in grp.columns else 0,
            })
    summary_df = pd.DataFrame(summary_rows)
    if 'selected_rolling_protocol' not in summary_df.columns:
        summary_df['selected_rolling_protocol'] = False
    if 'selected_rolling_protocol_rank' not in summary_df.columns:
        summary_df['selected_rolling_protocol_rank'] = np.nan
    selected_by_model: dict[str, dict[str, Any]] = {}
    if not summary_df.empty:
        all_models = summary_df[['rank', 'spec', 'model_id']].drop_duplicates()
        for _, model_row in all_models.iterrows():
            model_id = str(model_row['model_id'])
            rank = int(model_row['rank'])
            spec = str(model_row['spec'])
            grp = summary_df.loc[(summary_df['model_id'] == model_id) & (summary_df['train_window_mode'] == 'rolling')].copy()
            candidate = None
            if not grp.empty:
                finite = grp.loc[np.isfinite(pd.to_numeric(grp['validation_score_mean'], errors='coerce'))].copy()
                pool = finite if not finite.empty else grp.loc[grp['rolling_train_months'].fillna(-1).astype(float) == float(fallback_months)].copy()
                if pool.empty:
                    pool = grp.copy()
                pool['_sort_score_mean'] = pd.to_numeric(pool['validation_score_mean'], errors='coerce').fillna(-np.inf)
                pool['_sort_score_q10'] = pd.to_numeric(pool['validation_score_q10'], errors='coerce').fillna(-np.inf)
                pool['_sort_ce_zero'] = pd.to_numeric(pool['validation_ce_delta_zero_mean'], errors='coerce').fillna(-np.inf)
                pool['_sort_ce_myopic'] = pd.to_numeric(pool['validation_ce_delta_myopic_mean'], errors='coerce').fillna(-np.inf)
                pool['_sort_ce_mean'] = pd.to_numeric(pool['validation_ce_mean'], errors='coerce').fillna(-np.inf)
                pool['_sort_sharpe'] = pd.to_numeric(pool['validation_sharpe_mean'], errors='coerce').fillna(-np.inf)
                pool['_sort_months'] = pd.to_numeric(pool['rolling_train_months'], errors='coerce').fillna(np.inf)
                pool = pool.sort_values(
                    ['_sort_score_mean', '_sort_score_q10', '_sort_ce_zero', '_sort_ce_myopic', '_sort_ce_mean', '_sort_sharpe', '_sort_months'],
                    ascending=[False, False, False, False, False, False, True],
                )
                if not pool.empty:
                    candidate = pool.iloc[0]
                    summary_df.loc[candidate.name, 'selected_rolling_protocol'] = True
                    summary_df.loc[candidate.name, 'selected_rolling_protocol_rank'] = float(rank)
            if candidate is not None and pd.notna(candidate.get('rolling_train_months')):
                selected_by_model[model_id] = {
                    'name': ROLLING_SELECTED_PROTOCOL,
                    'source_protocol': str(candidate['oos_protocol']),
                    'train_window_mode': 'rolling',
                    'rolling_train_months': int(float(candidate['rolling_train_months'])),
                    'selection_source': 'validation_protocol_selection',
                    'selection_score_mean': float(candidate['validation_score_mean']) if np.isfinite(candidate['validation_score_mean']) else None,
                    'selection_score_q10': float(candidate['validation_score_q10']) if np.isfinite(candidate['validation_score_q10']) else None,
                    'validation_ce_mean': float(candidate['validation_ce_mean']) if np.isfinite(candidate['validation_ce_mean']) else None,
                    'validation_ce_delta_predictive_static_mean': float(candidate['validation_ce_delta_predictive_static_mean']) if np.isfinite(candidate['validation_ce_delta_predictive_static_mean']) else None,
                    'validation_ce_delta_myopic_mean': float(candidate['validation_ce_delta_myopic_mean']) if np.isfinite(candidate['validation_ce_delta_myopic_mean']) else None,
                    'validation_ce_delta_zero_mean': float(candidate['validation_ce_delta_zero_mean']) if np.isfinite(candidate['validation_ce_delta_zero_mean']) else None,
                    'validation_sharpe_mean': float(candidate['validation_sharpe_mean']) if np.isfinite(candidate['validation_sharpe_mean']) else None,
                    'validation_blocks_valid': int(candidate['validation_blocks_valid']) if pd.notna(candidate['validation_blocks_valid']) else 0,
                }
            else:
                selected_by_model[model_id] = {
                    'name': ROLLING_SELECTED_PROTOCOL,
                    'source_protocol': f'rolling{fallback_months}m_annual',
                    'train_window_mode': 'rolling',
                    'rolling_train_months': int(fallback_months),
                    'selection_source': 'validation_protocol_selection_fallback',
                    'selection_score_mean': None,
                    'selection_score_q10': None,
                    'validation_ce_mean': None,
                    'validation_ce_delta_predictive_static_mean': None,
                    'validation_ce_delta_myopic_mean': None,
                    'validation_ce_delta_zero_mean': None,
                    'validation_sharpe_mean': None,
                    'validation_blocks_valid': 0,
                }
    return summary_df, blocks_df, selected_by_model, fallback_months


def _attach_selected_rolling_protocol_columns(summary_df: pd.DataFrame, selected_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = summary_df.copy()
    mapping_months = {k: int(v['rolling_train_months']) for k, v in selected_by_model.items() if v and v.get('rolling_train_months') is not None}
    mapping_score = {k: v.get('selection_score_mean') for k, v in selected_by_model.items()}
    mapping_source = {k: v.get('source_protocol') for k, v in selected_by_model.items()}
    out['selected_rolling_train_months'] = out['model_id'].map(mapping_months)
    out['selected_rolling_protocol_name'] = ROLLING_SELECTED_PROTOCOL
    out['selected_rolling_source_protocol'] = out['model_id'].map(mapping_source)
    out['selected_rolling_validation_score_mean'] = out['model_id'].map(mapping_score)
    return out

def _fail_if_validation_issues(
    *,
    rows_df: pd.DataFrame,
    context: str,
    log_path: Path,
) -> None:
    if rows_df.empty:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        msg = f'[{context}] validation rows are empty'
        log_path.write_text(msg + '\n', encoding='utf-8')
        raise RuntimeError(msg)
    problems: list[str] = []
    if 'error' in rows_df.columns:
        err_df = rows_df.loc[rows_df['error'].notna()].copy()
        if not err_df.empty:
            for _, row in err_df.head(20).iterrows():
                problems.append(
                    f"[{context}] error spec={row.get('spec')} model_id={row.get('model_id')} "
                    f"protocol={row.get('selection_protocol_name', row.get('oos_protocol'))} "
                    f"block={row.get('block')}: {row.get('error')}"
                )
    if 'validation_score' in rows_df.columns:
        score = pd.to_numeric(rows_df['validation_score'], errors='coerce')
        bad = rows_df.loc[~np.isfinite(score)].copy()
        if not bad.empty:
            for _, row in bad.head(20).iterrows():
                problems.append(
                    f"[{context}] NaN validation_score spec={row.get('spec')} model_id={row.get('model_id')} "
                    f"protocol={row.get('selection_protocol_name', row.get('oos_protocol'))} block={row.get('block')}"
                )
    if problems:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('\n'.join(problems) + '\n', encoding='utf-8')
        raise RuntimeError(
            f'Validation failed in {context}: {len(problems)} issue(s). '
            f'See {log_path}'
        )

def _update_entry_metadata_with_selected_protocols(entry: dict[str, Any], selected_payload: dict[str, Any] | None) -> None:
    if not selected_payload:
        return
    entry['selected_rolling_train_months'] = int(selected_payload['rolling_train_months'])
    entry['selected_oos_protocols'] = {ROLLING_SELECTED_PROTOCOL: dict(selected_payload)}
    meta_path = Path(entry['metadata_yaml']).expanduser().resolve()
    payload = yaml.safe_load(meta_path.read_text(encoding='utf-8')) or {}
    payload['selected_rolling_train_months'] = int(selected_payload['rolling_train_months'])
    payload['selected_oos_protocols'] = {ROLLING_SELECTED_PROTOCOL: dict(selected_payload)}
    meta_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def native_select_factor_suite(
    *,
    base_dir: str | Path,
    out_dir: str | Path,
    factor_mode: str = 'ff5_curve_core',
    top_k: int = 2,
    stage1_top_k: int | None = None,
    risk_aversion: float = 5.0,
    cv_folds: int = 3,
    min_train_months: int = 204,
    rolling_window: int = 60,
    window_mode: str = 'rolling',
    return_floor: float = 0.0,
    state_q10_floor: float = -0.05,
    cross_warn: float = 0.95,
    cross_fail: float = 0.98,
    candidate_zoo: str = 'factor_zoo_v1',
    max_candidates: int | None = None,
    rerank_top_n: int = 5,
    selection_split_mode: str = 'trailing_holdout',
    selection_val_months: int = 240,
    selection_device: str = 'cpu',
    stage2_max_parallel: int = 1,
    stage2_devices: str | None = None,
    ppgdpo_lite_epochs: int = 40,
    ppgdpo_lite_mc_rollouts: int = 256,
    ppgdpo_lite_mc_sub_batch: int = 256,
    selection_transaction_cost_bps: float = 0.0,
    ppgdpo_lite_covariance_mode: str = 'full',
    selection_optimizer_backend: str = 'ppgdpo',
    pipinn_device: str = 'auto',
    pipinn_dtype: str = 'float64',
    pipinn_outer_iters: int = 6,
    pipinn_eval_epochs: int = 120,
    pipinn_n_train_int: int = 4096,
    pipinn_n_train_bc: int = 1024,
    pipinn_n_val_int: int = 2048,
    pipinn_n_val_bc: int = 512,
    pipinn_p_uniform: float = 0.30,
    pipinn_p_emp: float = 0.70,
    pipinn_p_tau_head: float = 0.50,
    pipinn_p_tau_near0: float = 0.20,
    pipinn_tau_head_window: int = 0,
    pipinn_lr: float = 5.0e-4,
    pipinn_grad_clip: float = 1.0,
    pipinn_w_bc: float = 20.0,
    pipinn_w_bc_dx: float = 5.0,
    pipinn_scheduler_factor: float = 0.5,
    pipinn_scheduler_patience: int = 3,
    pipinn_min_lr: float = 1.0e-5,
    pipinn_width: int = 96,
    pipinn_depth: int = 4,
    pipinn_covariance_train_mode: str = 'dcc_current',
    pipinn_ansatz_mode: str = 'ansatz_normalization_log_transform',
    pipinn_policy_output_mode: str = 'projection',
    pipinn_emit_frozen_traincov_strategy: bool = False,
    pipinn_save_training_logs: bool = True,
    pipinn_show_progress: bool = False,
    pipinn_show_epoch_progress: bool = False,
    rerank_covariance_models: list[str] | tuple[str, ...] | None = None,
    select_rolling_oos_window: bool = True,
    rolling_oos_window_grid: list[int] | tuple[int, ...] | None = None,
    selection_protocols: list[str] | tuple[str, ...] | None = None,
    fail_on_validation_error: bool = False,
    validation_error_log: str | Path | None = None,
    legacy_stage1_v1_root: str | Path | None = None,
    split_profile_override: str | None = None,
    split_train_start_override: str | None = None,
    split_train_pool_end_override: str | None = None,
    split_test_start_override: str | None = None,
    split_end_date_override: str | None = None,
) -> NativeSelectionArtifacts:
    base = load_base_bundle(base_dir)
    out_dir = ensure_dir(Path(out_dir).expanduser().resolve())
    returns = base['returns']
    macro = base['macro']
    ff3 = base['ff3']
    ff5 = base['ff5']
    bond = base['bond']
    meta = dict(base['manifest'])
    legacy_stage1_root = _resolve_legacy_stage1_v1_root(legacy_stage1_v1_root, meta)
    legacy_stage1_modules: tuple[Any, Any] | None = None
    if legacy_stage1_root is not None:
        try:
            legacy_stage1_modules = _load_legacy_stage1_modules(legacy_stage1_root)
        except Exception:
            legacy_stage1_modules = None

    base_split = dict(meta.get('split') or {})
    if not base_split:
        raise RuntimeError('base_bundle_manifest.yaml missing split payload')
    base_split.setdefault('train_start', base_split.get('train_pool_start', str(returns.index.min().date())))
    base_split.setdefault('test_start', base_split.get('final_test_start', base_split.get('test_start', str(returns.index[int(len(returns)*0.75)].date()))))
    base_split.setdefault('end_date', base_split.get('end_date', str(returns.index.max().date())))
    if 'train_pool_end' not in base_split:
        test_start_ts = pd.Timestamp(base_split['test_start'])
        base_split['train_pool_end'] = str((test_start_ts - pd.offsets.MonthEnd(1)).date())

    split, split_meta = resolve_calendar_window(
        base_profile=meta.get('split_profile'),
        fallback_payload=base_split,
        split_profile_override=split_profile_override,
        train_start_override=split_train_start_override,
        train_pool_end_override=split_train_pool_end_override,
        test_start_override=split_test_start_override,
        end_date_override=split_end_date_override,
    )
    if returns.index.max() < pd.Timestamp(split['end_date']):
        raise RuntimeError(
            f"Base bundle only runs through {returns.index.max().date()}, but requested split end_date={split['end_date']}"
        )
    train_pool_end = pd.Timestamp(split['train_pool_end'])
    train_start_ts = pd.Timestamp(split.get('train_start', split.get('train_pool_start', returns.index.min())))

    blocks = _build_selection_blocks(
        returns.index,
        train_start=train_start_ts,
        train_pool_end=train_pool_end,
        split_mode=selection_split_mode,
        cv_folds=cv_folds,
        min_train_months=min_train_months,
        selection_val_months=selection_val_months,
        rolling_window=rolling_window,
        window_mode=window_mode,
    )
    selection_block_payload = [
        {
            'label': str(block['label']),
            'train_start': str(pd.DatetimeIndex(block['train_dates'])[0].date()),
            'train_end': str(pd.DatetimeIndex(block['train_dates'])[-1].date()),
            'train_months': int(len(block['train_dates'])),
            'validation_start': str(pd.DatetimeIndex(block['val_dates'])[0].date()),
            'validation_end': str(pd.DatetimeIndex(block['val_dates'])[-1].date()),
            'validation_months': int(len(block['val_dates'])),
        }
        for block in blocks
    ]

    lite_cfg = SelectionLitePPGDPOConfig(
        optimizer_backend=str(selection_optimizer_backend),
        rerank_top_n=int(max(0, rerank_top_n)),
        device=str(selection_device),
        epochs=int(ppgdpo_lite_epochs),
        mc_rollouts=int(ppgdpo_lite_mc_rollouts),
        mc_sub_batch=int(ppgdpo_lite_mc_sub_batch),
        covariance_mode=str(ppgdpo_lite_covariance_mode),
        transaction_cost_bps=float(selection_transaction_cost_bps),
        pipinn_device=str(pipinn_device),
        pipinn_dtype=str(pipinn_dtype),
        pipinn_outer_iters=int(pipinn_outer_iters),
        pipinn_eval_epochs=int(pipinn_eval_epochs),
        pipinn_n_train_int=int(pipinn_n_train_int),
        pipinn_n_train_bc=int(pipinn_n_train_bc),
        pipinn_n_val_int=int(pipinn_n_val_int),
        pipinn_n_val_bc=int(pipinn_n_val_bc),
        pipinn_p_uniform=float(pipinn_p_uniform),
        pipinn_p_emp=float(pipinn_p_emp),
        pipinn_p_tau_head=float(pipinn_p_tau_head),
        pipinn_p_tau_near0=float(pipinn_p_tau_near0),
        pipinn_tau_head_window=int(pipinn_tau_head_window),
        pipinn_lr=float(pipinn_lr),
        pipinn_grad_clip=float(pipinn_grad_clip),
        pipinn_w_bc=float(pipinn_w_bc),
        pipinn_w_bc_dx=float(pipinn_w_bc_dx),
        pipinn_scheduler_factor=float(pipinn_scheduler_factor),
        pipinn_scheduler_patience=int(pipinn_scheduler_patience),
        pipinn_min_lr=float(pipinn_min_lr),
        pipinn_width=int(pipinn_width),
        pipinn_depth=int(pipinn_depth),
        pipinn_covariance_train_mode=str(pipinn_covariance_train_mode),
        pipinn_ansatz_mode=str(pipinn_ansatz_mode),
        pipinn_policy_output_mode=str(pipinn_policy_output_mode),
        pipinn_emit_frozen_traincov_strategy=bool(pipinn_emit_frozen_traincov_strategy),
        pipinn_save_training_logs=bool(pipinn_save_training_logs),
        pipinn_show_progress=bool(pipinn_show_progress),
        pipinn_show_epoch_progress=bool(pipinn_show_epoch_progress),
    )
    rerank_cov_specs = _parse_rerank_covariance_models(rerank_covariance_models)
    stage2_model_specs = _expand_stage2_model_specs(rerank_cov_specs)
    rolling_oos_window_grid = _normalize_rolling_oos_window_grid(rolling_oos_window_grid) if select_rolling_oos_window else []
    selection_protocol_candidates = _normalize_selection_protocols(
        selection_protocols,
        select_rolling_oos_window=select_rolling_oos_window,
        rolling_oos_window_grid=rolling_oos_window_grid,
    )

    candidates = build_candidate_registry(candidate_zoo)
    if max_candidates is not None:
        candidates = candidates[: max(1, int(max_candidates))]
    candidate_lookup = {candidate.name: candidate for candidate in candidates}

    summaries: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    stage1_audit_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        legacy_spec = _legacy_spec_name_for_candidate(candidate)
        protocol_block_metrics: dict[str, list[dict[str, Any]]] = {protocol: [] for protocol in selection_protocol_candidates}
        for block in blocks:
            try:
                panels = build_candidate_panels(candidate, returns=returns, macro=macro, ff3=ff3, ff5=ff5, bond=bond, train_dates=block['train_dates'])
                states_full = panels['states']
                cm, cx = _offdiag_corr_metrics(states_full.loc[states_full.index.intersection(block['val_dates'])])
                for protocol in selection_protocol_candidates:
                    proto_meta = _protocol_row_payload(protocol)
                    try:
                        metrics = _evaluate_stage1_candidate_block(
                            states_full=states_full,
                            returns_full=returns,
                            block=block,
                            rolling_window=rolling_window,
                            window_mode=window_mode,
                            protocol=protocol,
                            return_baseline='expanding_mean',
                            state_baseline='expanding_mean',
                        )
                        metrics['stage1_engine'] = PORTED_STAGE1_ENGINE
                        protocol_block_metrics[protocol].append(metrics)
                        block_rows.append({
                            'selection_unit_id': _selection_unit_id(candidate.name, protocol),
                            'spec': candidate.name,
                            'kind': candidate.kind,
                            'block': block['label'],
                            'selection_split_mode': str(selection_split_mode),
                            'train_start_date': str(pd.DatetimeIndex(block['train_dates'])[0].date()),
                            'train_end_date': str(pd.DatetimeIndex(block['train_dates'])[-1].date()),
                            'val_start_date': str(pd.DatetimeIndex(block['val_dates'])[0].date()),
                            'val_end_date': str(pd.DatetimeIndex(block['val_dates'])[-1].date()),
                            **proto_meta,
                            **metrics,
                            'cross_mean_abs_rho': cm,
                            'cross_max_abs_rho': cx,
                        })
                    except Exception as exc:  # noqa: BLE001
                        block_rows.append({
                            'selection_unit_id': _selection_unit_id(candidate.name, protocol),
                            'spec': candidate.name,
                            'kind': candidate.kind,
                            'block': block['label'],
                            'selection_split_mode': str(selection_split_mode),
                            'train_start_date': str(pd.DatetimeIndex(block['train_dates'])[0].date()),
                            'train_end_date': str(pd.DatetimeIndex(block['train_dates'])[-1].date()),
                            'val_start_date': str(pd.DatetimeIndex(block['val_dates'])[0].date()),
                            'val_end_date': str(pd.DatetimeIndex(block['val_dates'])[-1].date()),
                            **proto_meta,
                            'train_obs': 0,
                            'train_obs_min': 0,
                            'train_obs_max': 0,
                            'val_obs': 0,
                            'r2_oos_ret_mean': np.nan,
                            'r2_oos_ret_median': np.nan,
                            'r2_roll_ret_q10': np.nan,
                            'r2_roll_ret_min': np.nan,
                            'r2_oos_state_mean': np.nan,
                            'r2_oos_state_median': np.nan,
                            'r2_roll_state_q10': np.nan,
                            'r2_roll_state_min': np.nan,
                            'cross_mean_abs_rho': np.nan,
                            'cross_max_abs_rho': np.nan,
                            'stage1_engine': 'error',
                            'error': str(exc),
                        })
                if legacy_stage1_modules is not None and legacy_spec is not None:
                    legacy_spec_selection, legacy_discrete_latent_model = legacy_stage1_modules
                    try:
                        external_metrics = _evaluate_stage1_candidate_block_legacy(
                            candidate=candidate,
                            legacy_spec_selection=legacy_spec_selection,
                            legacy_discrete_latent_model=legacy_discrete_latent_model,
                            macro_full=macro,
                            ff3_full=ff3,
                            ff5_full=ff5,
                            returns_full=returns,
                            block=block,
                            rolling_window=rolling_window,
                            window_mode=window_mode,
                            return_baseline='expanding_mean',
                            state_baseline='expanding_mean',
                        )
                        stage1_audit_rows.append(
                            _build_stage1_external_audit_row(
                                candidate=candidate,
                                block=block,
                                ported_metrics={},
                                external_metrics=external_metrics,
                            )
                        )
                    except Exception as audit_exc:  # noqa: BLE001
                        stage1_audit_rows.append(
                            _build_stage1_external_audit_row(
                                candidate=candidate,
                                block=block,
                                ported_metrics={},
                                external_metrics=None,
                                error=str(audit_exc),
                            )
                        )
            except Exception as exc:  # noqa: BLE001
                for protocol in selection_protocol_candidates:
                    proto_meta = _protocol_row_payload(protocol)
                    block_rows.append({
                        'selection_unit_id': _selection_unit_id(candidate.name, protocol),
                        'spec': candidate.name,
                        'kind': candidate.kind,
                        'block': block['label'],
                        'selection_split_mode': str(selection_split_mode),
                        'train_start_date': str(pd.DatetimeIndex(block['train_dates'])[0].date()),
                        'train_end_date': str(pd.DatetimeIndex(block['train_dates'])[-1].date()),
                        'val_start_date': str(pd.DatetimeIndex(block['val_dates'])[0].date()),
                        'val_end_date': str(pd.DatetimeIndex(block['val_dates'])[-1].date()),
                        **proto_meta,
                        'train_obs': 0,
                        'train_obs_min': 0,
                        'train_obs_max': 0,
                        'val_obs': 0,
                        'r2_oos_ret_mean': np.nan,
                        'r2_oos_ret_median': np.nan,
                        'r2_roll_ret_q10': np.nan,
                        'r2_roll_ret_min': np.nan,
                        'r2_oos_state_mean': np.nan,
                        'r2_oos_state_median': np.nan,
                        'r2_roll_state_q10': np.nan,
                        'r2_roll_state_min': np.nan,
                        'cross_mean_abs_rho': np.nan,
                        'cross_max_abs_rho': np.nan,
                        'stage1_engine': 'error',
                        'error': str(exc),
                    })
                if legacy_stage1_modules is not None and legacy_spec is not None:
                    stage1_audit_rows.append(
                        _build_stage1_external_audit_row(
                            candidate=candidate,
                            block=block,
                            ported_metrics={},
                            external_metrics=None,
                            error=f'ported_stage1_error: {exc}',
                        )
                    )

        try:
            full_panels = build_candidate_panels(candidate, returns=returns, macro=macro, ff3=ff3, ff5=ff5, bond=bond, train_dates=returns.index[returns.index <= train_pool_end])
            states_all = full_panels['states']
            factor_dim = int(full_panels['factors'].shape[1])
            state_dim = int(states_all.shape[1])
            common_all = pd.DatetimeIndex(states_all.index.intersection(returns.index)).sort_values()
            if len(common_all) < 14 or train_pool_end not in common_all:
                raise ValueError('Not enough aligned rows for legacy cross diagnostics')
            train_end_all = int(common_all.get_loc(train_pool_end))
            legacy_model_all = _fit_legacy_stage1_linear_model(states_all.loc[common_all], returns.loc[common_all], train_end=train_end_all)
            cross_stats_all = _legacy_cross_rho_stats(legacy_model_all.Sigma, legacy_model_all.Q, legacy_model_all.Cross)
            cm_all, cx_all = float(cross_stats_all['mean_abs']), float(cross_stats_all['max_abs'])
        except Exception:
            cm_all, cx_all = np.nan, np.nan
            factor_dim = np.nan
            state_dim = np.nan

        for protocol in selection_protocol_candidates:
            valid = [m for m in protocol_block_metrics.get(protocol, []) if np.isfinite(m.get('r2_oos_ret_mean', np.nan))]
            proto_meta = _protocol_row_payload(protocol)
            row = {
                'selection_unit_id': _selection_unit_id(candidate.name, protocol),
                'spec': candidate.name,
                'kind': candidate.kind,
                'factor_dim': factor_dim,
                'state_dim': state_dim,
                'horizon': candidate.horizon,
                'latent_k': candidate.n_components,
                'feature_blocks': '|'.join(candidate.feature_blocks),
                'provided_source': candidate.provided_source,
                'residual_base': candidate.residual_base,
                'residual_k': candidate.residual_k,
                'stage1_engine': PORTED_STAGE1_ENGINE,
                'legacy_spec': legacy_spec,
                **proto_meta,
                'r2_oos_ret_mean': float(np.nanmean([m['r2_oos_ret_mean'] for m in valid])) if valid else np.nan,
                'r2_oos_ret_median': float(np.nanmean([m['r2_oos_ret_median'] for m in valid])) if valid else np.nan,
                'r2_roll_ret_q10': float(np.nanmean([m['r2_roll_ret_q10'] for m in valid])) if valid else np.nan,
                'r2_roll_ret_min': float(np.nanmean([m['r2_roll_ret_min'] for m in valid])) if valid else np.nan,
                'r2_oos_state_mean': float(np.nanmean([m['r2_oos_state_mean'] for m in valid])) if valid else np.nan,
                'r2_oos_state_median': float(np.nanmean([m['r2_oos_state_median'] for m in valid])) if valid else np.nan,
                'r2_roll_state_q10': float(np.nanmean([m['r2_roll_state_q10'] for m in valid])) if valid else np.nan,
                'r2_roll_state_min': float(np.nanmean([m['r2_roll_state_min'] for m in valid])) if valid else np.nan,
                'train_obs_mean': float(np.nanmean([m['train_obs'] for m in valid])) if valid else np.nan,
                'train_obs_min_mean': float(np.nanmean([m.get('train_obs_min', m['train_obs']) for m in valid])) if valid else np.nan,
                'train_obs_max_mean': float(np.nanmean([m.get('train_obs_max', m['train_obs']) for m in valid])) if valid else np.nan,
                'cross_mean_abs_rho': cm_all,
                'cross_max_abs_rho': cx_all,
                'n_blocks': int(len(blocks)),
                'selection_split_mode': str(selection_split_mode),
                'selection_val_months_requested': int(selection_val_months),
            }
            row['fail_return'] = bool(np.isnan(row['r2_oos_ret_mean']) or row['r2_oos_ret_mean'] < return_floor)
            row['fail_state'] = bool(np.isnan(row['r2_roll_state_q10']) or row['r2_roll_state_q10'] < state_q10_floor)
            row['fail_cross'] = bool(np.isfinite(row['cross_max_abs_rho']) and row['cross_max_abs_rho'] >= cross_fail)
            row['warn_cross'] = bool(np.isfinite(row['cross_max_abs_rho']) and row['cross_max_abs_rho'] >= cross_warn)
            row['passes_guard'] = bool((not row['fail_return']) and (not row['fail_state']) and (not row['fail_cross']))
            summaries.append(row)

    stage1_df = pd.DataFrame(summaries).sort_values(
        ['r2_oos_ret_mean', 'r2_roll_ret_q10', 'r2_oos_state_mean', 'selection_protocol_name'],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    stage1_audit_df = pd.DataFrame(stage1_audit_rows)
    if not stage1_audit_df.empty:
        stage1_df = stage1_df.merge(_aggregate_stage1_external_audit(stage1_audit_df), on='spec', how='left')
    stage1_df['score_ret_mean_rank'] = _rank_score(stage1_df, 'r2_oos_ret_mean', ascending=False)
    stage1_df['score_ret_q10_rank'] = _rank_score(stage1_df, 'r2_roll_ret_q10', ascending=False)
    stage1_df['score_state_mean_rank'] = _rank_score(stage1_df, 'r2_oos_state_mean', ascending=False)
    stage1_df['score_state_q10_rank'] = _rank_score(stage1_df, 'r2_roll_state_q10', ascending=False)
    stage1_df['score_ret_mean_component'] = 0.85 * stage1_df['r2_oos_ret_mean'].where(np.isfinite(stage1_df['r2_oos_ret_mean']), _MISSING_SCORE_PENALTY)
    stage1_df['score_ret_q10_component'] = 0.15 * stage1_df['r2_roll_ret_q10'].where(np.isfinite(stage1_df['r2_roll_ret_q10']), _MISSING_SCORE_PENALTY)
    stage1_df['score_state_mean_component'] = 0.0
    stage1_df['score'] = stage1_df.apply(
        lambda row: _selection_score_mean_first(
            row['r2_oos_ret_mean'],
            row['r2_roll_ret_q10'],
        ),
        axis=1,
    )
    stage1_df = _expand_stage1_mean_variants(stage1_df)
    stage1_df = stage1_df.sort_values(
        ['score', 'r2_oos_ret_mean', 'r2_roll_ret_q10', 'r2_oos_state_mean', 'cross_max_abs_rho', 'selection_protocol_name', 'stage1_mean_variant_label'],
        ascending=[False, False, False, False, True, True, True],
    ).reset_index(drop=True)
    stage1_df['cheap_rank'] = np.arange(1, len(stage1_df) + 1)
    stage1_df['cheap_selected_for_rerank'] = False
    stage1_df['recommended'] = False

    block_df = pd.DataFrame(block_rows)
    sel_dir = ensure_dir(out_dir / 'selection')
    selection_stage1_csv = sel_dir / 'spec_selection_stage1_summary.csv'
    selection_blocks_csv = sel_dir / 'spec_selection_blocks.csv'
    selection_stage1_audit_csv = sel_dir / 'spec_selection_stage1_external_audit.csv'
    stage1_df.to_csv(selection_stage1_csv, index=False)
    block_df.to_csv(selection_blocks_csv, index=False)
    if not stage1_audit_df.empty:
        stage1_audit_df.to_csv(selection_stage1_audit_csv, index=False)

    final_top_k = max(1, int(top_k))
    diagnostic_top_n = int(max(0, lite_cfg.rerank_top_n))
    if stage1_top_k is None:
        if diagnostic_top_n > 0:
            stage1_buffer_k = max(final_top_k, diagnostic_top_n, 8)
        else:
            stage1_buffer_k = final_top_k
    else:
        stage1_buffer_k = max(final_top_k, int(stage1_top_k))
        if diagnostic_top_n > 0:
            stage1_buffer_k = max(stage1_buffer_k, diagnostic_top_n)
    stage1_buffer_k = min(len(stage1_df), max(1, int(stage1_buffer_k)))

    stage1_selected_units = stage1_df.head(stage1_buffer_k)['selection_unit_id'].tolist()
    stage1_selected_specs = stage1_df.head(stage1_buffer_k)['spec'].tolist()
    diagnostic_units = list(stage1_selected_units) if diagnostic_top_n > 0 else []
    stage1_df['official_selected_stage1'] = stage1_df['selection_unit_id'].isin(stage1_selected_units)
    stage1_df['diagnostic_stage2'] = stage1_df['selection_unit_id'].isin(diagnostic_units)
    stage1_df['cheap_selected_for_rerank'] = stage1_df['diagnostic_stage2']
    stage1_df['recommended'] = False
    stage1_df['stage1_selected_rank'] = stage1_df['selection_unit_id'].map({unit: rank for rank, unit in enumerate(stage1_selected_units, start=1)}).astype(float)
    stage1_df.to_csv(selection_stage1_csv, index=False)

    stage2_rows: list[dict[str, Any]] = []
    stage2_block_rows: list[dict[str, Any]] = []
    if diagnostic_units:
        stage1_unit_lookup = stage1_df.set_index('selection_unit_id')
        stage2_eval_root = ensure_dir(out_dir / 'selection' / 'stage2_protocol_covariance_eval')
        stage2_device_pool = _parse_stage2_device_pool(stage2_devices, fallback=str(lite_cfg.device))
        max_parallel = int(max(1, stage2_max_parallel))
        max_parallel = int(min(max_parallel, len(diagnostic_units)))

        def _evaluate_stage2_for_unit(unit_id: str, assigned_device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            unit_row = stage1_unit_lookup.loc[unit_id]
            spec_name = str(unit_row['spec'])
            protocol = str(unit_row['selection_protocol_name'])
            proto_meta = _protocol_row_payload(protocol)
            candidate = candidate_lookup[spec_name]
            unit_stage2_rows: list[dict[str, Any]] = []
            unit_stage2_block_rows: list[dict[str, Any]] = []
            for model_spec in stage2_model_specs:
                cov_lite_cfg = _lite_cfg_for_stage2_model(lite_cfg, model_spec, mean_model_kind=str(unit_row.get('mean_model_kind') or 'factor_apt'))
                cov_lite_cfg = replace(cov_lite_cfg, device=str(assigned_device), pipinn_device=str(assigned_device))
                block_metrics: list[dict[str, Any]] = []
                for block in blocks:
                    try:
                        metrics = _evaluate_stage2_protocol_covariance_block(
                            selection_unit_id=unit_id,
                            candidate=candidate,
                            returns=returns,
                            macro=macro,
                            ff3=ff3,
                            ff5=ff5,
                            bond=bond,
                            block={**block, 'selection_split_mode': selection_split_mode},
                            protocol=protocol,
                            risk_aversion=risk_aversion,
                            lite_cfg=cov_lite_cfg,
                            split_payload=split,
                            config_stem=str(meta.get('config_stem', 'native')),
                            output_root=stage2_eval_root,
                        )
                        metrics = {**metrics, 'assigned_device': str(assigned_device)}
                        block_metrics.append(metrics)
                        stage2_block_rows.append({**metrics, 'error': None})
                        unit_stage2_block_rows.append({**metrics, 'error': None})
                    except Exception as exc:  # noqa: BLE001
                        train_dates = pd.DatetimeIndex(block.get('train_dates', []))
                        val_dates = pd.DatetimeIndex(block.get('val_dates', []))
                        unit_stage2_block_rows.append({
                            'selection_unit_id': unit_id,
                            'spec': spec_name,
                            **proto_meta,
                            'model_id': f'{unit_id}__{model_spec.label}',
                            'stage2_model_label': model_spec.label,
                            'covariance_model_label': str(model_spec.base_covariance_label or model_spec.label),
                            'covariance_model_kind': cov_lite_cfg.covariance_model_kind,
                            'mean_model_kind': cov_lite_cfg.mean_model_kind,
                            'cross_policy_label': cov_lite_cfg.cross_policy_label,
                            'block': str(block.get('label') or 'selection_block'),
                            'selection_split_mode': str(selection_split_mode),
                            'train_start_date': str(train_dates[0].date()) if len(train_dates) else None,
                            'train_end_date': str(train_dates[-1].date()) if len(train_dates) else None,
                            'val_start_date': str(val_dates[0].date()) if len(val_dates) else None,
                            'val_end_date': str(val_dates[-1].date()) if len(val_dates) else None,
                            'train_obs': int(len(train_dates)),
                            'validation_months': np.nan,
                            'validation_ce_est': np.nan,
                            'validation_ce_predictive_static': np.nan,
                            'validation_ce_myopic': np.nan,
                            'validation_ce_zero': np.nan,
                            'validation_ce_equal_weight': np.nan,
                            'validation_ce_min_variance': np.nan,
                            'validation_ce_risk_parity': np.nan,
                            'validation_ce_delta_predictive_static': np.nan,
                            'validation_ce_delta_myopic': np.nan,
                            'validation_ce_delta_zero': np.nan,
                            'validation_ce_delta_equal_weight': np.nan,
                            'validation_ce_delta_min_variance': np.nan,
                            'validation_ce_delta_risk_parity': np.nan,
                            'validation_sharpe_est': np.nan,
                            'validation_turnover_est': np.nan,
                            'validation_max_drawdown_est': np.nan,
                            'validation_score': np.nan,
                            'assigned_device': str(assigned_device),
                            'error': str(exc),
                        })
                valid = [row for row in block_metrics if np.isfinite(row.get('validation_score', np.nan))]
                unit_stage2_rows.append({
                    'selection_unit_id': unit_id,
                    'spec': spec_name,
                    **proto_meta,
                    'model_id': f'{unit_id}__{model_spec.label}',
                    'stage2_model_label': model_spec.label,
                    'covariance_model_label': str(model_spec.base_covariance_label or model_spec.label),
                    'covariance_model_kind': cov_lite_cfg.covariance_model_kind,
                    'mean_model_kind': cov_lite_cfg.mean_model_kind,
                    'cross_policy_label': cov_lite_cfg.cross_policy_label,
                    'ppgdpo_lite_blocks_valid': int(len(valid)),
                    'ppgdpo_lite_score_mean': float(np.nanmean([row['validation_score'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_score_q10': float(np.nanquantile(np.asarray([row['validation_score'] for row in valid], dtype=float), 0.10)) if valid else np.nan,
                    'ppgdpo_lite_ce_mean': float(np.nanmean([row['validation_ce_est'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_predictive_static_ce_mean': float(np.nanmean([row['validation_ce_predictive_static'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_myopic_ce_mean': float(np.nanmean([row['validation_ce_myopic'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_equal_weight_ce_mean': float(np.nanmean([row['validation_ce_equal_weight'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_min_variance_ce_mean': float(np.nanmean([row['validation_ce_min_variance'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_risk_parity_ce_mean': float(np.nanmean([row['validation_ce_risk_parity'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_ce_delta_predictive_static_mean': float(np.nanmean([row['validation_ce_delta_predictive_static'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_ce_delta_myopic_mean': float(np.nanmean([row['validation_ce_delta_myopic'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_ce_delta_zero_mean': float(np.nanmean([row['validation_ce_delta_zero'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_ce_delta_equal_weight_mean': float(np.nanmean([row['validation_ce_delta_equal_weight'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_ce_delta_min_variance_mean': float(np.nanmean([row['validation_ce_delta_min_variance'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_ce_delta_risk_parity_mean': float(np.nanmean([row['validation_ce_delta_risk_parity'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_sharpe_mean': float(np.nanmean([row['validation_sharpe_est'] for row in valid])) if valid else np.nan,
                    'ppgdpo_lite_train_objective_mean': np.nan,
                    'assigned_device': str(assigned_device),
                })
            return unit_stage2_rows, unit_stage2_block_rows

        if max_parallel <= 1:
            for unit_index, unit_id in enumerate(diagnostic_units):
                assigned_device = stage2_device_pool[unit_index % len(stage2_device_pool)]
                unit_rows, unit_block_rows = _evaluate_stage2_for_unit(str(unit_id), assigned_device)
                stage2_rows.extend(unit_rows)
                stage2_block_rows.extend(unit_block_rows)
        else:
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = []
                for unit_index, unit_id in enumerate(diagnostic_units):
                    assigned_device = stage2_device_pool[unit_index % len(stage2_device_pool)]
                    futures.append(executor.submit(_evaluate_stage2_for_unit, str(unit_id), assigned_device))
                for fut in as_completed(futures):
                    unit_rows, unit_block_rows = fut.result()
                    stage2_rows.extend(unit_rows)
                    stage2_block_rows.extend(unit_block_rows)

    stage2_df = pd.DataFrame(stage2_rows)
    if 'model_id' not in stage2_df.columns:
        stage2_df = pd.DataFrame(columns=[
            'selection_unit_id', 'spec', 'selection_protocol_name', 'train_window_mode', 'rolling_train_months',
            'model_id', 'stage2_model_label', 'covariance_model_label', 'covariance_model_kind', 'mean_model_kind', 'cross_policy_label', 'ppgdpo_lite_blocks_valid',
            'ppgdpo_lite_score_mean', 'ppgdpo_lite_score_q10', 'ppgdpo_lite_ce_mean', 'ppgdpo_lite_predictive_static_ce_mean', 'ppgdpo_lite_myopic_ce_mean',
            'ppgdpo_lite_equal_weight_ce_mean', 'ppgdpo_lite_min_variance_ce_mean', 'ppgdpo_lite_risk_parity_ce_mean',
            'ppgdpo_lite_ce_delta_predictive_static_mean', 'ppgdpo_lite_ce_delta_myopic_mean', 'ppgdpo_lite_ce_delta_zero_mean',
            'ppgdpo_lite_ce_delta_equal_weight_mean', 'ppgdpo_lite_ce_delta_min_variance_mean', 'ppgdpo_lite_ce_delta_risk_parity_mean',
            'ppgdpo_lite_sharpe_mean', 'ppgdpo_lite_train_objective_mean', 'assigned_device',
        ])
    if 'ppgdpo_lite_predictive_static_ce_mean' not in stage2_df.columns and 'ppgdpo_lite_myopic_ce_mean' in stage2_df.columns:
        stage2_df['ppgdpo_lite_predictive_static_ce_mean'] = stage2_df['ppgdpo_lite_myopic_ce_mean']
    if 'ppgdpo_lite_ce_delta_predictive_static_mean' not in stage2_df.columns and 'ppgdpo_lite_ce_delta_myopic_mean' in stage2_df.columns:
        stage2_df['ppgdpo_lite_ce_delta_predictive_static_mean'] = stage2_df['ppgdpo_lite_ce_delta_myopic_mean']
    stage2_df = _annotate_stage2_real_ppgdpo_scores(stage2_df, stage1_selected_units)

    selection_stage2_csv = sel_dir / 'spec_selection_stage2_summary.csv'
    selection_ppgdpo_blocks_csv = sel_dir / 'spec_selection_ppgdpo_blocks.csv'
    stage2_df.to_csv(selection_stage2_csv, index=False)
    # pd.DataFrame(stage2_block_rows).to_csv(selection_ppgdpo_blocks_csv, index=False)
    stage2_block_df = pd.DataFrame(stage2_block_rows)
    stage2_block_df.to_csv(selection_ppgdpo_blocks_csv, index=False)
    if bool(fail_on_validation_error):
        log_path = Path(validation_error_log).expanduser().resolve() if validation_error_log is not None else (sel_dir / 'validation_errors.log')
        _fail_if_validation_issues(rows_df=stage2_block_df, context='stage2_protocol_covariance_eval', log_path=log_path)

    stage1_only_df = stage1_df[~stage1_df['diagnostic_stage2']].copy()
    stage1_only_df['model_id'] = stage1_only_df['selection_unit_id'].map(lambda s: f'{s}__stage1_only')
    stage1_only_df['covariance_model_label'] = 'stage1_only'
    stage1_only_df['covariance_model_kind'] = np.nan
    stage1_only_df['stage2_model_label'] = 'stage1_only'
    if 'mean_model_kind' not in stage1_only_df.columns:
        stage1_only_df['mean_model_kind'] = 'factor_apt'
    stage1_only_df['cross_policy_label'] = np.nan
    stage1_only_df['ppgdpo_lite_blocks_valid'] = np.nan
    stage1_only_df['ppgdpo_lite_score_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_score_q10'] = np.nan
    stage1_only_df['ppgdpo_lite_ce_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_predictive_static_ce_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_myopic_ce_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_equal_weight_ce_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_min_variance_ce_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_risk_parity_ce_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_ce_delta_predictive_static_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_ce_delta_myopic_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_ce_delta_zero_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_ce_delta_equal_weight_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_ce_delta_min_variance_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_ce_delta_risk_parity_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_sharpe_mean'] = np.nan
    stage1_only_df['ppgdpo_lite_train_objective_mean'] = np.nan
    stage1_only_df['stage2_rank_myopic_ce'] = np.nan
    stage1_only_df['stage2_rank_ce_gain'] = np.nan
    stage1_only_df['stage2_rank_ppgdpo_score'] = np.nan
    stage1_only_df['stage2_rank_ppgdpo_q10'] = np.nan
    stage1_only_df['stage2_rank_hedging_gain'] = np.nan
    stage1_only_df['stage2_real_ppgdpo_score'] = np.nan
    stage1_only_df['stage2_mean_first_score'] = np.nan
    stage1_only_df['stage2_sort_score'] = np.nan
    stage1_only_df['diagnostic_rank'] = np.nan
    stage1_only_df['selected_stage2_model'] = False
    stage1_only_df['selected_stage2_model_rank'] = np.nan
    stage1_only_df['official_selected_model'] = stage1_only_df['official_selected_stage1']
    stage1_only_df['official_oos_candidate'] = stage1_only_df['official_selected_model']
    stage1_only_df['recommended'] = stage1_only_df['official_selected_model']
    stage1_only_df['selection_role'] = np.where(stage1_only_df['official_selected_model'], 'stage1_selected_unit', 'stage1_only_nonselected')

    if not stage2_df.empty:
        stage1_meta = stage1_df.set_index('selection_unit_id')
        for col in stage1_df.columns:
            if col == 'selection_unit_id' or col in stage2_df.columns:
                continue
            stage2_df[col] = stage2_df['selection_unit_id'].map(stage1_meta[col].to_dict())
        stage2_df['diagnostic_stage2'] = True
        stage2_df['official_selected_model'] = stage2_df['selected_stage2_model'].fillna(False)
        stage2_df['official_oos_candidate'] = stage2_df['official_selected_model']
        stage2_df['recommended'] = stage2_df['official_selected_model']
        stage2_df['selection_role'] = np.select(
            [stage2_df['official_selected_model'], stage2_df['official_selected_stage1'].fillna(False)],
            ['stage2_selected_model', 'stage2_model_grid_nonselected'],
            default='stage2_diagnostic_only',
        )
        summary_df = pd.concat([stage2_df, stage1_only_df], ignore_index=True, sort=False)
    else:
        summary_df = stage1_only_df.copy()

    rerank_cov_lookup = {spec.label: spec for spec in rerank_cov_specs}
    stage1_unit_lookup = stage1_df.set_index('selection_unit_id') if not stage1_df.empty else pd.DataFrame().set_index(pd.Index([], name='selection_unit_id'))
    candidate_model_grid: list[dict[str, Any]] = []
    if diagnostic_units:
        stage2_candidates = stage2_df.loc[stage2_df['selection_unit_id'].isin(stage1_selected_units)].copy()
        unit_rank_map = {unit: rank for rank, unit in enumerate(stage1_selected_units, start=1)}
        if not stage2_candidates.empty:
            stage2_candidates['stage1_selected_rank'] = stage2_candidates['selection_unit_id'].map(unit_rank_map)
            stage2_candidates = stage2_candidates.sort_values(
                ['stage2_sort_score', 'ppgdpo_lite_score_mean', 'ppgdpo_lite_score_q10', 'ppgdpo_lite_ce_delta_zero_mean', 'ppgdpo_lite_ce_delta_myopic_mean', 'stage1_selected_rank', 'spec', 'selection_protocol_name', 'covariance_model_label'],
                ascending=[False, False, False, False, False, True, True, True, True],
            )
        for unit_id in stage1_selected_units:
            unit_row = stage1_unit_lookup.loc[unit_id]
            stage1_rank = int(float(unit_row['stage1_selected_rank'])) if pd.notna(unit_row.get('stage1_selected_rank', np.nan)) else len(candidate_model_grid) + 1
            spec_name = str(unit_row['spec'])
            protocol_name = str(unit_row['selection_protocol_name'])
            proto_meta = _protocol_row_payload(protocol_name)
            spec_rows = stage2_candidates.loc[stage2_candidates['selection_unit_id'] == unit_id]
            if spec_rows.empty:
                fallback_label = rerank_cov_specs[0].label if rerank_cov_specs else 'stage1_only'
                fallback_kind = rerank_cov_lookup[fallback_label].covariance_model_kind if fallback_label in rerank_cov_lookup else None
                selected_protocol_payload = {
                    'name': SELECTED_PROTOCOL,
                    'source_protocol': protocol_name,
                    'train_window_mode': proto_meta['train_window_mode'],
                    'rolling_train_months': None if pd.isna(proto_meta['rolling_train_months']) else int(float(proto_meta['rolling_train_months'])),
                    'selection_source': 'stage1_protocol_pair_selection',
                    'selection_score_mean': float(unit_row['score']) if np.isfinite(unit_row['score']) else None,
                    'selection_semantics': 'warm_start_rolling_available_history',
                }
                candidate_model_grid.append({
                    'model_id': f'{unit_id}__{fallback_label}',
                    'selection_unit_id': unit_id,
                    'spec': spec_name,
                    'selection_protocol_name': protocol_name,
                    'train_window_mode': proto_meta['train_window_mode'],
                    'rolling_train_months': None if pd.isna(proto_meta['rolling_train_months']) else int(float(proto_meta['rolling_train_months'])),
                    'stage2_model_label': fallback_label,
                    'covariance_model_label': fallback_label,
                    'covariance_model_kind': fallback_kind,
                    'mean_model_kind': 'factor_apt',
                    'cross_policy_label': 'estimated',
                    'stage2_sort_score': None,
                    'ppgdpo_lite_score_mean': None,
                    'ppgdpo_lite_ce_delta_zero_mean': None,
                    'ppgdpo_lite_ce_delta_myopic_mean': None,
                    'stage1_selected_rank': stage1_rank,
                    'stage1_screen_score': float(unit_row['score']) if np.isfinite(unit_row['score']) else None,
                    'selected_oos_protocols': {SELECTED_PROTOCOL: dict(selected_protocol_payload)},
                })
                continue
            for _, row in spec_rows.iterrows():
                selected_protocol_payload = {
                    'name': SELECTED_PROTOCOL,
                    'source_protocol': protocol_name,
                    'train_window_mode': str(row['train_window_mode']),
                    'rolling_train_months': None if pd.isna(row['rolling_train_months']) else int(float(row['rolling_train_months'])),
                    'selection_source': 'stage1_spec_protocol_screen_then_stage2_global_model_selection',
                    'selection_score_mean': float(row['stage2_sort_score']) if np.isfinite(row['stage2_sort_score']) else None,
                    'selection_score_q10': float(row['ppgdpo_lite_score_q10']) if np.isfinite(row['ppgdpo_lite_score_q10']) else None,
                    'validation_ce_mean': float(row['ppgdpo_lite_ce_mean']) if np.isfinite(row['ppgdpo_lite_ce_mean']) else None,
                    'validation_ce_delta_myopic_mean': float(row['ppgdpo_lite_ce_delta_myopic_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_myopic_mean']) else None,
                    'validation_ce_delta_zero_mean': float(row['ppgdpo_lite_ce_delta_zero_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_zero_mean']) else None,
                    'validation_ce_delta_equal_weight_mean': float(row['ppgdpo_lite_ce_delta_equal_weight_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_equal_weight_mean']) else None,
                    'validation_ce_delta_min_variance_mean': float(row['ppgdpo_lite_ce_delta_min_variance_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_min_variance_mean']) else None,
                    'validation_ce_delta_risk_parity_mean': float(row['ppgdpo_lite_ce_delta_risk_parity_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_risk_parity_mean']) else None,
                    'validation_sharpe_mean': float(row['ppgdpo_lite_sharpe_mean']) if np.isfinite(row['ppgdpo_lite_sharpe_mean']) else None,
                    'validation_blocks_valid': int(row['ppgdpo_lite_blocks_valid']) if pd.notna(row['ppgdpo_lite_blocks_valid']) else 0,
                    'selection_semantics': 'warm_start_rolling_available_history',
                }
                candidate_model_grid.append({
                    'model_id': str(row['model_id']),
                    'selection_unit_id': unit_id,
                    'spec': spec_name,
                    'selection_protocol_name': protocol_name,
                    'train_window_mode': str(row['train_window_mode']),
                    'rolling_train_months': None if pd.isna(row['rolling_train_months']) else int(float(row['rolling_train_months'])),
                    'stage2_model_label': str(row.get('stage2_model_label') if pd.notna(row.get('stage2_model_label', np.nan)) else row['covariance_model_label']),
                    'covariance_model_label': str(row['covariance_model_label']),
                    'covariance_model_kind': None if pd.isna(row['covariance_model_kind']) else str(row['covariance_model_kind']),
                    'mean_model_kind': str(row.get('mean_model_kind') or 'factor_apt'),
                    'cross_policy_label': str(row.get('cross_policy_label') or 'estimated'),
                    'stage2_sort_score': float(row['stage2_sort_score']) if np.isfinite(row['stage2_sort_score']) else None,
                    'ppgdpo_lite_score_mean': float(row['ppgdpo_lite_score_mean']) if np.isfinite(row['ppgdpo_lite_score_mean']) else None,
                    'ppgdpo_lite_ce_delta_zero_mean': float(row['ppgdpo_lite_ce_delta_zero_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_zero_mean']) else None,
                    'ppgdpo_lite_ce_delta_myopic_mean': float(row['ppgdpo_lite_ce_delta_myopic_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_myopic_mean']) else None,
                    'ppgdpo_lite_ce_delta_equal_weight_mean': float(row['ppgdpo_lite_ce_delta_equal_weight_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_equal_weight_mean']) else None,
                    'ppgdpo_lite_ce_delta_min_variance_mean': float(row['ppgdpo_lite_ce_delta_min_variance_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_min_variance_mean']) else None,
                    'ppgdpo_lite_ce_delta_risk_parity_mean': float(row['ppgdpo_lite_ce_delta_risk_parity_mean']) if np.isfinite(row['ppgdpo_lite_ce_delta_risk_parity_mean']) else None,
                    'stage1_selected_rank': stage1_rank,
                    'stage1_screen_score': float(unit_row['score']) if np.isfinite(unit_row['score']) else None,
                    'selected_oos_protocols': {SELECTED_PROTOCOL: dict(selected_protocol_payload)},
                })
    else:
        for unit_id in stage1_selected_units:
            unit_row = stage1_unit_lookup.loc[unit_id]
            stage1_rank = int(float(unit_row['stage1_selected_rank'])) if pd.notna(unit_row.get('stage1_selected_rank', np.nan)) else len(candidate_model_grid) + 1
            protocol_name = str(unit_row['selection_protocol_name'])
            proto_meta = _protocol_row_payload(protocol_name)
            selected_protocol_payload = {
                'name': SELECTED_PROTOCOL,
                'source_protocol': protocol_name,
                'train_window_mode': proto_meta['train_window_mode'],
                'rolling_train_months': None if pd.isna(proto_meta['rolling_train_months']) else int(float(proto_meta['rolling_train_months'])),
                'selection_source': 'stage1_spec_protocol_selection',
                'selection_score_mean': float(unit_row['score']) if np.isfinite(unit_row['score']) else None,
                'selection_semantics': 'warm_start_rolling_available_history',
            }
            candidate_model_grid.append({
                'model_id': f'{unit_id}__stage1_only',
                'selection_unit_id': unit_id,
                'spec': str(unit_row['spec']),
                'selection_protocol_name': protocol_name,
                'train_window_mode': proto_meta['train_window_mode'],
                'rolling_train_months': None if pd.isna(proto_meta['rolling_train_months']) else int(float(proto_meta['rolling_train_months'])),
                'stage2_model_label': 'stage1_only',
                'covariance_model_label': 'stage1_only',
                'covariance_model_kind': None,
                'mean_model_kind': 'factor_apt',
                'cross_policy_label': 'estimated',
                'stage2_sort_score': None,
                'ppgdpo_lite_score_mean': None,
                'ppgdpo_lite_ce_delta_zero_mean': None,
                'ppgdpo_lite_ce_delta_myopic_mean': None,
                'stage1_selected_rank': stage1_rank,
                'stage1_screen_score': float(unit_row['score']) if np.isfinite(unit_row['score']) else None,
                'selected_oos_protocols': {SELECTED_PROTOCOL: dict(selected_protocol_payload)},
            })

    selected_model_grid: list[dict[str, Any]] = []
    final_selected_model_ids: set[str] = set()
    if candidate_model_grid:
        candidate_model_df = pd.DataFrame(candidate_model_grid)
        candidate_model_df['_sort_stage2'] = pd.to_numeric(candidate_model_df.get('stage2_sort_score'), errors='coerce').fillna(-np.inf)
        candidate_model_df['_sort_ppgdpo'] = pd.to_numeric(candidate_model_df.get('ppgdpo_lite_score_mean'), errors='coerce').fillna(-np.inf)
        candidate_model_df['_sort_zero'] = pd.to_numeric(candidate_model_df.get('ppgdpo_lite_ce_delta_zero_mean'), errors='coerce').fillna(-np.inf)
        candidate_model_df['_sort_myopic'] = pd.to_numeric(candidate_model_df.get('ppgdpo_lite_ce_delta_myopic_mean'), errors='coerce').fillna(-np.inf)
        candidate_model_df['_sort_stage1'] = pd.to_numeric(candidate_model_df.get('stage1_screen_score'), errors='coerce').fillna(float(_MISSING_SCORE_PENALTY))
        candidate_model_df['_sort_stage1_rank'] = pd.to_numeric(candidate_model_df.get('stage1_selected_rank'), errors='coerce').fillna(np.inf)
        candidate_model_df = candidate_model_df.sort_values(
            ['_sort_stage2', '_sort_ppgdpo', '_sort_zero', '_sort_myopic', '_sort_stage1', '_sort_stage1_rank', 'spec', 'selection_protocol_name', 'covariance_model_label', 'mean_model_kind', 'cross_policy_label'],
            ascending=[False, False, False, False, False, True, True, True, True, True, True],
        ).reset_index(drop=True)
        helper_cols = [c for c in candidate_model_df.columns if c.startswith('_sort_')]
        selected_model_grid = candidate_model_df.head(final_top_k).drop(columns=helper_cols, errors='ignore').to_dict(orient='records')
        final_selected_model_ids = {str(row['model_id']) for row in selected_model_grid}

    if 'official_selected_model' not in summary_df.columns:
        summary_df['official_selected_model'] = False
    summary_df['official_selected_stage1'] = summary_df['selection_unit_id'].isin(stage1_selected_units)
    summary_df['official_selected_model'] = summary_df['model_id'].astype(str).isin(final_selected_model_ids)
    summary_df['official_oos_candidate'] = summary_df['official_selected_model']
    summary_df['recommended'] = summary_df['official_selected_model']
    summary_df['selection_role'] = np.select(
        [
            summary_df['official_selected_model'],
            summary_df['official_selected_stage1'].fillna(False) & summary_df['diagnostic_stage2'].fillna(False),
            summary_df['official_selected_stage1'].fillna(False),
        ],
        [
            'final_selected_model',
            'stage1_survivor_stage2_nonselected',
            'stage1_survivor_nonselected',
        ],
        default=np.where(summary_df['diagnostic_stage2'].fillna(False), 'stage2_diagnostic_only', 'stage1_only_nonselected'),
    )
    summary_df['selection_protocol'] = np.where(
        summary_df['diagnostic_stage2'].fillna(False),
        'stage1_spec_protocol_selection+stage2_protocol_covariance_selection',
        'stage1_spec_protocol_selection_only',
    )
    summary_df['final_score'] = np.where(summary_df['diagnostic_stage2'].fillna(False), summary_df['stage2_sort_score'], summary_df['score'])
    final_rank_map = {str(model['model_id']): rank for rank, model in enumerate(selected_model_grid, start=1)}
    summary_df['final_rank'] = summary_df['model_id'].astype(str).map(final_rank_map).astype(float)
    summary_df = summary_df.sort_values(
        ['official_selected_model', 'final_rank', 'official_selected_stage1', 'selected_stage2_model', 'diagnostic_stage2', 'diagnostic_rank', 'final_score', 'spec', 'selection_protocol_name', 'covariance_model_label'],
        ascending=[False, True, False, False, False, True, False, True, True, True],
    ).reset_index(drop=True)

    selection_summary_csv = sel_dir / 'spec_selection_summary.csv'
    summary_df.to_csv(selection_summary_csv, index=False)

    entries: list[dict[str, Any]] = []
    selected_model_rank_map = {str(model['model_id']): rank for rank, model in enumerate(selected_model_grid, start=1)}
    for selected_model in selected_model_grid:
        unit_id = str(selected_model['selection_unit_id'])
        spec_name = str(selected_model['spec'])
        protocol_name = str(selected_model['selection_protocol_name'])
        selected_model_rank = int(selected_model_rank_map.get(str(selected_model['model_id']), len(entries) + 1))
        stage1_rank = int(selected_model.get('stage1_selected_rank') or selected_model_rank)
        bundle_cov_label = str(selected_model['covariance_model_label'])
        stage2_model_label = str(selected_model.get('stage2_model_label') or bundle_cov_label)
        selected_mean_model_kind = str(selected_model.get('mean_model_kind') or 'factor_apt')
        selected_cross_policy_label = str(selected_model.get('cross_policy_label') or 'estimated')
        candidate = candidate_lookup[spec_name]
        panels = build_candidate_panels(candidate, returns=returns, macro=macro, ff3=ff3, ff5=ff5, bond=bond, train_dates=returns.index[returns.index <= train_pool_end])
        factors_full = panels['factors']
        states_full = panels['states']
        common = returns.index.intersection(states_full.index).intersection(factors_full.index)
        returns_rank = returns.loc[common].copy()
        states_rank = states_full.loc[common].copy()
        factors_rank = factors_full.loc[common].copy()
        rank_dir = ensure_dir(out_dir / f'rank_{len(entries)+1:03d}')
        returns_csv = rank_dir / 'returns_panel.csv'
        states_csv = rank_dir / 'states_panel.csv'
        factors_csv = rank_dir / 'factors_panel.csv'
        returns_rank.reset_index(names='date').to_csv(returns_csv, index=False)
        states_rank.reset_index(names='date').to_csv(states_csv, index=False)
        factors_rank.reset_index(names='date').to_csv(factors_csv, index=False)

        if bundle_cov_label == 'stage1_only':
            cfg_payload = _build_v2_config_dict(
                out_dir=rank_dir,
                config_stem=f"{meta.get('config_stem', 'native')}_{spec_name}_{protocol_name}_{stage2_model_label}",
                split_payload=split,
                state_cols=list(states_rank.columns),
                factor_cols=list(factors_rank.columns),
                risk_aversion=risk_aversion,
                ppgdpo_covariance_mode=str(lite_cfg.covariance_mode),
                mean_model_kind=selected_mean_model_kind,
                comparison_cross_modes=_comparison_cross_modes_for_covariance_label(bundle_cov_label),
                optimizer_backend=str(lite_cfg.optimizer_backend),
                pipinn_payload=_pipinn_payload_from_lite_cfg(lite_cfg),
            )
        else:
            cov_payload = _config_covariance_payload_from_label(bundle_cov_label)
            cfg_payload = _build_v2_config_dict(
                out_dir=rank_dir,
                config_stem=f"{meta.get('config_stem', 'native')}_{spec_name}_{protocol_name}_{stage2_model_label}",
                split_payload=split,
                state_cols=list(states_rank.columns),
                factor_cols=list(factors_rank.columns),
                risk_aversion=risk_aversion,
                covariance_model_kind=str(cov_payload['kind']),
                covariance_factor_correlation_mode=str(cov_payload['factor_correlation_mode']),
                covariance_use_persistence=bool(cov_payload['use_persistence']),
                covariance_adcc_gamma=float(cov_payload.get('adcc_gamma', 0.005)),
                covariance_regime_threshold_quantile=float(cov_payload.get('regime_threshold_quantile', 0.75)),
                covariance_regime_smoothing=float(cov_payload.get('regime_smoothing', 0.90)),
                covariance_regime_sharpness=float(cov_payload.get('regime_sharpness', 8.0)),
                ppgdpo_covariance_mode=str(lite_cfg.covariance_mode),
                mean_model_kind=selected_mean_model_kind,
                comparison_cross_modes=_comparison_cross_modes_for_covariance_label(bundle_cov_label),
                optimizer_backend=str(lite_cfg.optimizer_backend),
                pipinn_payload=_pipinn_payload_from_lite_cfg(lite_cfg),
            )
        cfg_filename = 'config_empirical_pipinn_apt.yaml' if str(lite_cfg.optimizer_backend).lower() == 'pipinn' else 'config_empirical_ppgdpo_apt.yaml'
        cfg_path = rank_dir / cfg_filename
        cfg_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False), encoding='utf-8')
        meta_yaml = rank_dir / 'candidate_metadata.yaml'
        metadata_payload = {
            'stage1_selected_rank': stage1_rank,
            'covariance_grid_rank': 1,
            'selected_model_rank': selected_model_rank,
            'selection_unit_id': unit_id,
            'spec': spec_name,
            'selection_protocol_name': protocol_name,
            'selected_protocol_name': protocol_name,
            'train_window_mode': selected_model['train_window_mode'],
            'selected_rolling_train_months': selected_model.get('rolling_train_months'),
            'selected_oos_protocols': dict(selected_model.get('selected_oos_protocols') or {}),
            'bundle_covariance_model_label': bundle_cov_label,
            'stage2_model_label': stage2_model_label,
            'selected_mean_model_kind': selected_mean_model_kind,
            'selected_cross_policy_label': selected_cross_policy_label,
            'selection_optimizer_backend': str(lite_cfg.optimizer_backend),
            'stage2_sort_score': selected_model.get('stage2_sort_score'),
            'ppgdpo_lite_score_mean': selected_model.get('ppgdpo_lite_score_mean'),
            'ppgdpo_lite_ce_delta_zero_mean': selected_model.get('ppgdpo_lite_ce_delta_zero_mean'),
            'ppgdpo_lite_ce_delta_predictive_static_mean': selected_model.get('ppgdpo_lite_ce_delta_predictive_static_mean'),
            'ppgdpo_lite_ce_delta_myopic_mean': selected_model.get('ppgdpo_lite_ce_delta_myopic_mean'),
            'ppgdpo_lite_ce_delta_equal_weight_mean': selected_model.get('ppgdpo_lite_ce_delta_equal_weight_mean'),
            'ppgdpo_lite_ce_delta_min_variance_mean': selected_model.get('ppgdpo_lite_ce_delta_min_variance_mean'),
            'ppgdpo_lite_ce_delta_risk_parity_mean': selected_model.get('ppgdpo_lite_ce_delta_risk_parity_mean'),
            'candidate': panels['meta'],
        }
        meta_yaml.write_text(yaml.safe_dump(metadata_payload, sort_keys=False), encoding='utf-8')
        entries.append({
            'rank': len(entries) + 1,
            'model_id': str(selected_model['model_id']),
            'selection_unit_id': unit_id,
            'spec': spec_name,
            'selection_protocol_name': protocol_name,
            'selected_protocol_name': protocol_name,
            'stage1_selected_rank': stage1_rank,
            'selected_model_rank': selected_model_rank,
            'covariance_grid_rank': 1,
            'bundle_covariance_model_label': bundle_cov_label,
            'stage2_model_label': stage2_model_label,
            'selected_mean_model_kind': selected_mean_model_kind,
            'selected_cross_policy_label': selected_cross_policy_label,
            'selection_optimizer_backend': str(lite_cfg.optimizer_backend),
            'train_window_mode': selected_model['train_window_mode'],
            'selected_rolling_train_months': selected_model.get('rolling_train_months'),
            'selected_oos_protocols': dict(selected_model.get('selected_oos_protocols') or {}),
            'bundle_dir': str(rank_dir),
            'config_yaml': str(cfg_path),
            'metadata_yaml': str(meta_yaml),
            'returns_csv': str(returns_csv),
            'states_csv': str(states_csv),
            'factors_csv': str(factors_csv),
        })

    selected_oos_protocol_defaults = {}
    if selected_model_grid and selected_model_grid[0].get('selected_oos_protocols'):
        selected_oos_protocol_defaults = {
            SELECTED_PROTOCOL: dict(selected_model_grid[0]['selected_oos_protocols'][SELECTED_PROTOCOL])
        }
    default_oos_protocols = [SELECTED_PROTOCOL]
    validation_protocol_selection = {
        'enabled': False,
        'selection_target': 'integrated_stage1_spec_protocol_then_stage2_model',
        'selection_protocol_candidates': list(selection_protocol_candidates),
        'selection_semantics': 'warm_start_rolling_available_history',
    }

    manifest = {
        'suite_name': f"{meta.get('config_stem', 'native')}_native_top{len(entries)}",
        'source': 'v2_native_selection',
        'base_dir': str(Path(base_dir).expanduser().resolve()),
        'config_stem': meta.get('config_stem', 'native'),
        'factor_mode': factor_mode,
        'candidate_zoo': candidate_zoo,
        'selection_optimizer_backend': str(lite_cfg.optimizer_backend),
        'selection_score_mode': 'stage1_mean_first_spec_protocol_then_stage2_global_real_dynamic_policy' if diagnostic_units else 'stage1_mean_first_spec_protocol_only',
        'selection_protocol': 'stage1_spec_protocol_screen_then_stage2_global_protocol_model_selection' if diagnostic_units else 'stage1_spec_protocol_selection_only',
        'selection_protocol_candidates': list(selection_protocol_candidates),
        'selection_protocol_semantics': 'warm_start_rolling_available_history',
        'selection_protocol_semantics_note': 'rolling protocols may start with the available pre-history and become full-window rolling once enough history accumulates',
        'v55_strategy_label_map': {'myopic': 'predictive_static', 'policy': 'pgdpo'},
        'strategy_label_map': {'myopic': 'predictive_static', 'policy': 'pgdpo'},
        'comparison_benchmark_notes': {
            'predictive_static': 'reference-only, not a primary benchmark',
            'pgdpo': 'warmup direct policy ablation',
            'ppgdpo_variants': ['ppgdpo', 'ppgdpo_zero', 'ppgdpo_regime_gated'],
            'external_benchmarks': list(SELECTION_REFERENCE_BENCHMARKS),
        },
        'stage1_top_k': int(len(stage1_selected_units)),
        'stage1_top_k_requested': None if stage1_top_k is None else int(stage1_top_k),
        'final_top_k_requested': int(final_top_k),
        'diagnostic_stage2_top_n': int(len(diagnostic_units)),
        'rerank_covariance_models': [spec.label for spec in rerank_cov_specs],
        'stage2_model_variants': [spec.label for spec in stage2_model_specs],
        'ppgdpo_lite': {
            'optimizer_backend': str(lite_cfg.optimizer_backend),
            'device': lite_cfg.device,
            'stage2_max_parallel': int(max(1, stage2_max_parallel)),
            'stage2_devices': _parse_stage2_device_pool(stage2_devices, fallback=str(lite_cfg.device)),
            'epochs': lite_cfg.epochs,
            'covariance_mode': lite_cfg.covariance_mode,
            'mc_rollouts': lite_cfg.mc_rollouts,
            'mc_sub_batch': lite_cfg.mc_sub_batch,
            'rerank_covariance_models': [spec.label for spec in rerank_cov_specs],
        'stage2_model_variants': [spec.label for spec in stage2_model_specs],
            'transaction_cost_bps': lite_cfg.transaction_cost_bps,
            'pipinn': _pipinn_payload_from_lite_cfg(lite_cfg) if str(lite_cfg.optimizer_backend).lower() == 'pipinn' else None,
            'score_mode': (
                'stage2 protocol+covariance selection with backend-aware dynamic optimizer; ce_est + 1.0*gain_vs_zero, with gain_vs_myopic kept for reporting only and final winners chosen by global rerank across all stage2 models from stage1 survivors'
                if diagnostic_units else
                'disabled (rerank_top_n=0)'
            ),
        },
        'selection_split_profile': split_meta.get('split_profile'),
        'selection_split': {
            'mode': str(selection_split_mode),
            'split_profile': split_meta.get('split_profile'),
            'split_source': split_meta.get('split_source'),
            'split_description': split_meta.get('split_description'),
            'split_overrides': split_meta.get('split_overrides'),
            'train_pool_start': str(train_start_ts.date()),
            'train_pool_end': str(train_pool_end.date()),
            'validation_months_requested': int(selection_val_months),
            'blocks': selection_block_payload,
            'final_test_start': str(pd.Timestamp(split['test_start']).date()),
            'final_test_end': str(pd.Timestamp(split['end_date']).date()),
            'final_oos_retrain_uses_train_plus_validation': True,
            'final_oos_train_end': str(train_pool_end.date()),
        },
        **manifest_protocol_payload(default_oos_protocols),
        'selected_oos_protocol_defaults': selected_oos_protocol_defaults,
        'validation_protocol_selection': validation_protocol_selection,
        'selection_protocol_validation_summary_csv': None,
        'selection_protocol_validation_blocks_csv': None,
        'top_k': len(entries),
        'selection_summary_csv': str(selection_summary_csv),
        'selection_stage1_csv': str(selection_stage1_csv),
        'selection_stage2_csv': str(selection_stage2_csv),
        'selection_stage1_audit_csv': str(selection_stage1_audit_csv) if not stage1_audit_df.empty else None,
        'stage1_engine': PORTED_STAGE1_ENGINE,
        'stage1_external_audit_enabled': bool(legacy_stage1_modules is not None),
        'stage1_external_audit_v1_root': str(legacy_stage1_root) if legacy_stage1_root is not None else None,
        'entries': entries,
    }
    manifest_yaml = out_dir / 'suite_manifest.yaml'
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')

    selected_yaml = out_dir / 'selected_spec.yaml'
    final_selected_specs = [str(model.get('spec')) for model in selected_model_grid]
    final_selected_stage1_units = [str(model.get('selection_unit_id')) for model in selected_model_grid]
    selected_payload = {
        'selected_stage1_units': stage1_selected_units,
        'primary_selected_stage1_unit': stage1_selected_units[0] if stage1_selected_units else None,
        'selected_stage1_specs': stage1_selected_specs,
        'primary_selected_stage1_spec': stage1_selected_specs[0] if stage1_selected_specs else None,
        'selected_specs': final_selected_specs,
        'primary_selected_spec': final_selected_specs[0] if final_selected_specs else None,
        'selected_final_units': final_selected_stage1_units,
        'primary_selected_final_unit': final_selected_stage1_units[0] if final_selected_stage1_units else None,
        'selected_models': selected_model_grid,
        'primary_selected_model': selected_model_grid[0] if selected_model_grid else None,
        'selection_summary_csv': str(selection_summary_csv),
        'selection_stage1_csv': str(selection_stage1_csv),
        'selection_stage2_csv': str(selection_stage2_csv),
        'selection_stage1_audit_csv': str(selection_stage1_audit_csv) if not stage1_audit_df.empty else None,
        'selection_protocol_validation_summary_csv': None,
        'selection_protocol_validation_blocks_csv': None,
        'validation_protocol_selection': validation_protocol_selection,
        'candidate_count': int(len(stage1_df)),
        'candidate_zoo': candidate_zoo,
        'selection_optimizer_backend': str(lite_cfg.optimizer_backend),
        'selection_score_mode': 'stage1_mean_first_spec_protocol_then_stage2_global_real_dynamic_policy' if diagnostic_units else 'stage1_mean_first_spec_protocol_only',
        'selection_protocol': 'stage1_spec_protocol_screen_then_stage2_global_protocol_model_selection' if diagnostic_units else 'stage1_spec_protocol_selection_only',
        'selection_protocol_candidates': list(selection_protocol_candidates),
        'v55_strategy_label_map': {'myopic': 'predictive_static', 'policy': 'pgdpo'},
        'strategy_label_map': {'myopic': 'predictive_static', 'policy': 'pgdpo'},
        'comparison_benchmark_notes': {
            'predictive_static': 'reference-only, not a primary benchmark',
            'pgdpo': 'warmup direct policy ablation',
            'ppgdpo_variants': ['ppgdpo', 'ppgdpo_zero', 'ppgdpo_regime_gated'],
            'external_benchmarks': list(SELECTION_REFERENCE_BENCHMARKS),
        },
        'stage1_engine': PORTED_STAGE1_ENGINE,
        'stage1_external_audit_enabled': bool(legacy_stage1_modules is not None),
        'stage1_external_audit_v1_root': str(legacy_stage1_root) if legacy_stage1_root is not None else None,
        'stage1_top_k': int(len(stage1_selected_units)),
        'stage1_top_k_requested': None if stage1_top_k is None else int(stage1_top_k),
        'final_top_k_requested': int(final_top_k),
        'diagnostic_stage2_top_n': int(len(diagnostic_units)),
        'rerank_covariance_models': [spec.label for spec in rerank_cov_specs],
        'stage2_model_variants': [spec.label for spec in stage2_model_specs],
        'final_model_grid_count': int(len(selected_model_grid)),
        'ppgdpo_lite': {
            'optimizer_backend': str(lite_cfg.optimizer_backend),
            'device': lite_cfg.device,
            'stage2_max_parallel': int(max(1, stage2_max_parallel)),
            'stage2_devices': _parse_stage2_device_pool(stage2_devices, fallback=str(lite_cfg.device)),
            'epochs': lite_cfg.epochs,
            'covariance_mode': lite_cfg.covariance_mode,
            'mc_rollouts': lite_cfg.mc_rollouts,
            'mc_sub_batch': lite_cfg.mc_sub_batch,
            'rerank_covariance_models': [spec.label for spec in rerank_cov_specs],
        'stage2_model_variants': [spec.label for spec in stage2_model_specs],
            'transaction_cost_bps': lite_cfg.transaction_cost_bps,
            'pipinn': _pipinn_payload_from_lite_cfg(lite_cfg) if str(lite_cfg.optimizer_backend).lower() == 'pipinn' else None,
            'score_mode': (
                'stage2 protocol+covariance selection with backend-aware dynamic optimizer; ce_est + 1.0*gain_vs_zero, with gain_vs_myopic kept for reporting only and final winners chosen by global rerank across all stage2 models from stage1 survivors'
                if diagnostic_units else
                'disabled (rerank_top_n=0)'
            ),
        },
        'selection_blocks': [b['label'] for b in blocks],
        'selection_split_profile': split_meta.get('split_profile'),
        'selection_split': {
            'mode': str(selection_split_mode),
            'split_profile': split_meta.get('split_profile'),
            'split_source': split_meta.get('split_source'),
            'split_description': split_meta.get('split_description'),
            'split_overrides': split_meta.get('split_overrides'),
            'train_pool_start': str(train_start_ts.date()),
            'train_pool_end': str(train_pool_end.date()),
            'validation_months_requested': int(selection_val_months),
            'blocks': selection_block_payload,
            'final_test_start': str(pd.Timestamp(split['test_start']).date()),
            'final_test_end': str(pd.Timestamp(split['end_date']).date()),
            'final_oos_retrain_uses_train_plus_validation': True,
            'final_oos_train_end': str(train_pool_end.date()),
        },
        **manifest_protocol_payload(default_oos_protocols),
        'selected_oos_protocol_defaults': selected_oos_protocol_defaults,
    }
    selected_yaml.write_text(yaml.safe_dump(selected_payload, sort_keys=False), encoding='utf-8')

    return NativeSelectionArtifacts(
        out_dir=out_dir,
        manifest_yaml=manifest_yaml,
        selection_summary_csv=selection_summary_csv,
        selected_yaml=selected_yaml,
        entry_count=len(entries),
    )


# Backward-compatible name
native_select_pls_suite = native_select_factor_suite
