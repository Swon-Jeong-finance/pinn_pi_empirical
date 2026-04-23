from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .schema import Config
from .data import load_dataset
from .factors import ProvidedFactorExtractor, PCAFactorExtractor, FactorRepresentation
from .mean_model import fit_direct_asset_mean, fit_factor_apt_mean, fit_factor_apt_regime_mean, MeanModelResult
from .covariance import AssetADCCCovariance, AssetDCCCovariance, AssetRegimeDCCCovariance, ConstantFactorCovariance, StateDiagonalFactorCovariance
from .policies import solve_equal_weight, solve_mean_variance, solve_min_variance, solve_projected, solve_risk_parity
from .ppgdpo import train_warmup_policy, solve_ppgdpo_projection
from .pipinn_backend import train_pipinn_policy
from .transition import fit_state_transition, estimate_return_state_cross
from .utils import ensure_dir, annualized_return, annualized_vol, sharpe_ratio, certainty_equivalent_annual, max_drawdown


@dataclass
class RunArtifacts:
    output_dir: Path
    summary_zero_cost: Path
    summary_with_costs: Path
    plan_yaml: Path
    benchmark_notes_yaml: Path | None = None


def _select_dates(index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    return index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]


def _build_extractor(cfg: Config):
    if cfg.factor_model.extractor == 'provided':
        return ProvidedFactorExtractor(cfg.factor_model.provided_factor_columns)
    return PCAFactorExtractor(cfg.factor_model.n_factors)


def _build_cov_model(cfg: Config):
    if cfg.covariance_model.kind == 'constant':
        return ConstantFactorCovariance(
            variance_floor=cfg.covariance_model.variance_floor,
            correlation_shrink=cfg.covariance_model.correlation_shrink,
            factor_correlation_mode=cfg.covariance_model.factor_correlation_mode,
        )
    if cfg.covariance_model.kind == 'asset_dcc':
        return AssetDCCCovariance(
            variance_floor=cfg.covariance_model.variance_floor,
            correlation_shrink=cfg.covariance_model.correlation_shrink,
            dcc_alpha=cfg.covariance_model.dcc_alpha,
            dcc_beta=cfg.covariance_model.dcc_beta,
            variance_lambda=cfg.covariance_model.variance_lambda,
            asset_covariance_shrink=cfg.covariance_model.asset_covariance_shrink,
        )
    if cfg.covariance_model.kind == 'asset_adcc':
        return AssetADCCCovariance(
            variance_floor=cfg.covariance_model.variance_floor,
            correlation_shrink=cfg.covariance_model.correlation_shrink,
            dcc_alpha=cfg.covariance_model.dcc_alpha,
            dcc_beta=cfg.covariance_model.dcc_beta,
            adcc_gamma=cfg.covariance_model.adcc_gamma,
            variance_lambda=cfg.covariance_model.variance_lambda,
            asset_covariance_shrink=cfg.covariance_model.asset_covariance_shrink,
        )
    if cfg.covariance_model.kind == 'asset_regime_dcc':
        return AssetRegimeDCCCovariance(
            variance_floor=cfg.covariance_model.variance_floor,
            correlation_shrink=cfg.covariance_model.correlation_shrink,
            dcc_alpha=cfg.covariance_model.dcc_alpha,
            dcc_beta=cfg.covariance_model.dcc_beta,
            variance_lambda=cfg.covariance_model.variance_lambda,
            asset_covariance_shrink=cfg.covariance_model.asset_covariance_shrink,
            regime_threshold_quantile=cfg.covariance_model.regime_threshold_quantile,
            regime_smoothing=cfg.covariance_model.regime_smoothing,
            regime_sharpness=cfg.covariance_model.regime_sharpness,
        )
    use_persistence = cfg.covariance_model.use_persistence
    if cfg.covariance_model.kind == 'state_only_diagonal':
        use_persistence = False
    return StateDiagonalFactorCovariance(
        ridge_lambda=cfg.covariance_model.ridge_lambda,
        variance_floor=cfg.covariance_model.variance_floor,
        correlation_shrink=cfg.covariance_model.correlation_shrink,
        factor_correlation_mode=cfg.covariance_model.factor_correlation_mode,
        use_persistence=use_persistence,
    )

def _resolve_dynamic_cross_kind(cfg: Config) -> str | None:
    cross_kind = str(getattr(cfg.covariance_model, 'cross_covariance_kind', 'sample') or 'sample').lower()
    if cross_kind == 'sample':
        return None
    if cross_kind in {'dcc', 'adcc', 'regime_dcc'}:
        return cross_kind
    cov_kind = str(cfg.covariance_model.kind).lower()
    if cov_kind == 'asset_dcc':
        return 'dcc'
    if cov_kind == 'asset_adcc':
        return 'adcc'
    if cov_kind == 'asset_regime_dcc':
        return 'regime_dcc'
    return None


def _write_plan(cfg: Config, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    plan_yaml = output_dir / 'resolved_config.yaml'
    plan_yaml.write_text(yaml.safe_dump(cfg.model_dump(mode='json'), sort_keys=False), encoding='utf-8')
    return plan_yaml


def _summarize(monthly: pd.DataFrame, gamma: float, return_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in monthly.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mapping = dict(zip(group_cols, keys))
        r = grp[return_col].astype(float)
        row = {
            **mapping,
            'months': int(len(grp)),
            'ann_ret': annualized_return(r),
            'ann_vol': annualized_vol(r),
            'sharpe': sharpe_ratio(r),
            'cer_ann': certainty_equivalent_annual(r, gamma),
            'avg_turnover': float(grp['turnover'].mean()),
            'avg_risky_weight': float(grp['risky_weight'].mean()),
            'max_drawdown': max_drawdown(r),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_windows(cfg: Config, returns: pd.DataFrame):
    all_dates = returns.index
    eval_dates = _select_dates(all_dates, cfg.split.test_start, cfg.split.end_date)
    eval_dates = eval_dates[eval_dates.isin(all_dates[:-1])]
    return all_dates, eval_dates


def _protocol_label(cfg: Config) -> str:
    label = str(cfg.split.protocol_label or '').strip()
    if label:
        return label
    return f"{cfg.split.train_window_mode}_refit{int(cfg.split.refit_every)}_rebalance{int(cfg.split.rebalance_every)}"


def _latest_factor_return_row(cfg: Config, factors: pd.DataFrame | None, factor_repr: FactorRepresentation, date_t: pd.Timestamp) -> pd.Series:
    if cfg.factor_model.extractor == 'provided' and factors is not None and date_t in factors.index:
        return factors.loc[date_t, cfg.factor_model.provided_factor_columns]
    return factor_repr.factors.iloc[-1]


def _fit_mean_model(cfg: Config, states_t: pd.DataFrame, returns_tp1: pd.DataFrame, factor_repr: FactorRepresentation) -> MeanModelResult:
    if cfg.mean_model.kind in {'factor_apt', 'factor_apt_regime'}:
        factor_returns_tp1 = factor_repr.factors.copy()
        if len(factor_returns_tp1) != len(states_t):
            factor_returns_tp1 = factor_returns_tp1.iloc[: len(states_t)].copy()
        factor_returns_tp1.index = states_t.index
        if cfg.mean_model.kind == 'factor_apt_regime':
            return fit_factor_apt_regime_mean(
                states_t=states_t,
                factor_returns_tp1=factor_returns_tp1,
                loadings=factor_repr.loadings,
                asset_alpha=factor_repr.asset_alpha,
                ridge_lambda=cfg.mean_model.ridge_lambda,
                regime_threshold_quantile=cfg.mean_model.regime_threshold_quantile,
                regime_sharpness=cfg.mean_model.regime_sharpness,
            )
        return fit_factor_apt_mean(
            states_t=states_t,
            factor_returns_tp1=factor_returns_tp1,
            loadings=factor_repr.loadings,
            asset_alpha=factor_repr.asset_alpha,
            ridge_lambda=cfg.mean_model.ridge_lambda,
        )
    return fit_direct_asset_mean(states_t, returns_tp1, ridge_lambda=cfg.mean_model.ridge_lambda)


def _predict_asset_means_over_sample(mean_model: MeanModelResult, states_t: pd.DataFrame) -> np.ndarray:
    return np.vstack([mean_model.predict(states_t.iloc[j]).to_numpy(dtype=float) for j in range(len(states_t))])


def _resolve_regime_probability(cov_model: Any, mean_model: MeanModelResult, latest_factor_return: pd.Series | None) -> float:
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


def _fit_models_for_window(cfg: Config, states: pd.DataFrame, returns: pd.DataFrame, factors: pd.DataFrame | None, train_dates: pd.DatetimeIndex):
    train_pair_dates = train_dates[:-1]
    train_next_dates = train_dates[1:]
    state_train = states.loc[train_pair_dates, cfg.state.columns]
    state_train_next = states.loc[train_next_dates, cfg.state.columns]
    ret_train_next = returns.loc[train_next_dates]

    extractor = _build_extractor(cfg)
    if cfg.factor_model.extractor == 'provided':
        factor_train_next = factors.loc[train_next_dates, cfg.factor_model.provided_factor_columns]
        factor_repr = extractor.fit(ret_train_next, factor_train_next)
    else:
        factor_repr = extractor.fit(ret_train_next, None)

    mean_model = _fit_mean_model(cfg, state_train, ret_train_next, factor_repr)
    mu_train_pred = _predict_asset_means_over_sample(mean_model, state_train)
    cov_model = _build_cov_model(cfg)
    cov_model.fit(
        state_train,
        factor_repr.factors,
        asset_returns_tp1=ret_train_next,
        asset_mean_pred=mu_train_pred,
    )
    return state_train, state_train_next, ret_train_next, factor_repr, mean_model, cov_model


def _train_dates_for_decision(cfg: Config, all_dates: pd.DatetimeIndex, decision_pos: int) -> pd.DatetimeIndex:
    train_start_ts = pd.Timestamp(cfg.split.train_start)
    mode = str(cfg.split.train_window_mode)
    if mode == 'fixed':
        fixed_end = pd.Timestamp(cfg.split.fixed_train_end or cfg.split.test_start)
        train_dates = all_dates[(all_dates >= train_start_ts) & (all_dates <= fixed_end)]
    elif mode == 'expanding':
        train_dates = all_dates[: decision_pos + 1]
        train_dates = train_dates[train_dates >= train_start_ts]
    elif mode == 'rolling':
        lookback = int(cfg.split.rolling_train_months or cfg.split.min_train_months)
        start_pos = max(0, decision_pos - lookback + 1)
        train_dates = all_dates[start_pos : decision_pos + 1]
        train_dates = train_dates[train_dates >= train_start_ts]
    else:
        raise ValueError(f'Unsupported split.train_window_mode={mode!r}')
    return pd.DatetimeIndex(train_dates)


def _should_refit(cfg: Config, eval_step: int, cached: Any) -> bool:
    if cached is None:
        return True
    if str(cfg.split.train_window_mode) == 'fixed':
        return False
    return eval_step % max(int(cfg.split.refit_every), 1) == 0


def _should_rebalance(cfg: Config, eval_step: int) -> bool:
    return eval_step % max(int(cfg.split.rebalance_every), 1) == 0


def _drift_holdings(weights: np.ndarray, realized_ret: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    r = np.asarray(realized_ret, dtype=float)
    cash = max(1.0 - float(w.sum()), 0.0)
    asset_growth = w * (1.0 + r)
    wealth_mult = cash + float(asset_growth.sum())
    if (not np.isfinite(wealth_mult)) or wealth_mult <= 1.0e-12:
        return np.zeros_like(w)
    next_w = asset_growth / wealth_mult
    next_w = np.nan_to_num(next_w, nan=0.0, posinf=0.0, neginf=0.0)
    next_w = np.clip(next_w, 0.0, None)
    risky_sum = float(next_w.sum())
    if risky_sum > 1.0 + 1.0e-10:
        next_w = next_w / risky_sum
    return next_w


def _protocol_constants(cfg: Config) -> dict[str, Any]:
    return {
        'oos_protocol': _protocol_label(cfg),
        'train_window_mode': str(cfg.split.train_window_mode),
        'refit_every': int(cfg.split.refit_every),
        'rebalance_every': int(cfg.split.rebalance_every),
        'rolling_train_months': int(cfg.split.rolling_train_months) if cfg.split.rolling_train_months is not None else None,
        'fixed_train_end': str(cfg.split.fixed_train_end) if cfg.split.fixed_train_end is not None else None,
    }

def _format_output_tag_value(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return 'nan'
        return f'{value:.6g}'.replace('+', '')
    return str(value)


def _resolve_pipinn_output_dir(cfg: Config) -> Path:
    base_dir = Path(cfg.project.output_dir)
    if _optimizer_backend(cfg) != 'pipinn':
        return base_dir
    if not bool(getattr(cfg.pipinn, 'auto_output_subdir', False)):
        return base_dir

    default_fields = [
        'outer_iters',
        'eval_epochs',
        'n_train_int',
        'n_train_bc',
        'p_uniform',
        'p_emp',
        'lr',
        'w_bc',
        'w_bc_dx',
        'width',
        'depth',
    ]
    fields = list(getattr(cfg.pipinn, 'output_tag_fields', []) or default_fields)
    parts: list[str] = []
    for field in fields:
        if not hasattr(cfg.pipinn, field):
            continue
        raw = getattr(cfg.pipinn, field)
        value = _format_output_tag_value(raw)
        safe_value = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in value)
        parts.append(f'{field}-{safe_value}')
    if not parts:
        return base_dir / 'pipinn_cfg'
    return base_dir / ('pipinn_' + '__'.join(parts))


def _fit_dynamic_policy_backend(
    cfg: Config,
    *,
    state_train: pd.DataFrame,
    state_train_next: pd.DataFrame,
    ret_train_next: pd.DataFrame,
    factor_repr: FactorRepresentation,
    mean_model: MeanModelResult,
    cov_model: Any,
    transaction_cost: float,
    progress_label: str | None = None,
    tau_max: float | None = None,
    prev_trainer: Any = None,
) -> tuple[Any, Any, Any]:
    transition = fit_state_transition(state_train, state_train_next, ridge_lambda=cfg.ppgdpo.state_ridge_lambda)
    mu_train_pred = _predict_asset_means_over_sample(mean_model, state_train)
    dynamic_cross_kind = _resolve_dynamic_cross_kind(cfg)
    cross_est = estimate_return_state_cross(
        returns_tp1=ret_train_next,
        returns_mean_pred=mu_train_pred,
        states_t=state_train,
        states_tp1=state_train_next,
        transition=transition,
        dynamic_cross_kind=dynamic_cross_kind,
        variance_floor=cfg.covariance_model.variance_floor,
        correlation_shrink=cfg.covariance_model.correlation_shrink,
        dcc_alpha=cfg.covariance_model.dcc_alpha,
        dcc_beta=cfg.covariance_model.dcc_beta,
        adcc_gamma=cfg.covariance_model.adcc_gamma,
        variance_lambda=cfg.covariance_model.variance_lambda,
        regime_threshold_quantile=cfg.covariance_model.regime_threshold_quantile,
        regime_smoothing=cfg.covariance_model.regime_smoothing,
        regime_sharpness=cfg.covariance_model.regime_sharpness,
    )
    backend = str(getattr(cfg, 'optimizer_backend', 'ppgdpo')).lower()
    if backend == 'pipinn':
        trainer = train_pipinn_policy(
            state_train,
            ret_train_next,
            cfg,
            transaction_cost=transaction_cost,
            mean_model=mean_model,
            transition=transition,
            cross_est=cross_est,
            cov_model=cov_model,
            factor_repr=factor_repr,
            progress_label=progress_label,
            tau_max=tau_max,
            warm_start_from=prev_trainer,
        )
    else:
        trainer = train_warmup_policy(
            state_train,
            ret_train_next,
            cfg,
            transaction_cost=transaction_cost,
            mean_model=mean_model,
            transition=transition,
            cross_est=cross_est,
        )
    return transition, cross_est, trainer


def _optimizer_backend(cfg: Config) -> str:
    return str(getattr(cfg, 'optimizer_backend', 'ppgdpo')).lower()

def _emit_pipinn_frozen_traincov_strategy(cfg: Config) -> bool:
    return _optimizer_backend(cfg) == 'pipinn' and bool(getattr(cfg.pipinn, 'emit_frozen_traincov_strategy', False))

def _augment_summary(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = summary.copy()
    meta = _protocol_constants(cfg)
    for col, value in reversed(list(meta.items())):
        out.insert(0, col, value)
    return out


def _effective_risky_cap(cfg: Config) -> float:
    cash_floor = float(getattr(cfg.policy, 'cash_floor', 0.0) or 0.0)
    return float(np.clip(min(float(cfg.policy.risky_cap), 1.0 - cash_floor), 0.0, 1.0))


def _sample_covariance(returns_tp1: pd.DataFrame) -> np.ndarray:
    arr = returns_tp1.to_numpy(dtype=float)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    if arr.shape[0] < 2:
        var = np.nanvar(arr, axis=0, ddof=0) if arr.size else np.asarray([], dtype=float)
        cov = np.diag(np.maximum(np.nan_to_num(var, nan=1.0e-6), 1.0e-6))
    else:
        cov = np.cov(arr, rowvar=False, ddof=1)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov = np.atleast_2d(cov)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov).copy()
    diag = np.where(np.isfinite(diag), np.maximum(diag, 1.0e-8), 1.0e-8)
    np.fill_diagonal(cov, diag)
    return cov


def _normalize_name(name: str) -> str:
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum())


def _detect_market_factor_column(factors: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if factors is None or factors.empty:
        return None
    columns = list(factors.columns)
    lookup = {_normalize_name(col): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return str(candidate)
        hit = lookup.get(_normalize_name(candidate))
        if hit is not None:
            return hit
    return None

def _relative_output_path(path: Path | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(output_dir.resolve()))
    except Exception:  # noqa: BLE001
        return str(path)


def _write_pipinn_training_log(
    output_dir: Path,
    trainer: Any,
    *,
    decision_date: pd.Timestamp,
    train_dates: pd.DatetimeIndex,
    refit_index: int,
) -> tuple[Path | None, dict[str, Any] | None]:
    history = list(getattr(trainer, 'train_history', []) or [])
    if not history:
        return None, None
    log_dir = ensure_dir(output_dir / 'training_logs' / 'pipinn')
    csv_path = log_dir / f"refit_{int(refit_index):03d}_{pd.Timestamp(decision_date).strftime('%Y%m%d')}.csv"
    train_start = pd.Timestamp(train_dates[0]) if len(train_dates) else pd.NaT
    train_end = pd.Timestamp(train_dates[-1]) if len(train_dates) else pd.NaT
    hist_df = pd.DataFrame(history)
    hist_df.insert(0, 'refit_index', int(refit_index))
    hist_df.insert(1, 'decision_date', pd.Timestamp(decision_date))
    hist_df.insert(2, 'train_window_start', train_start)
    hist_df.insert(3, 'train_window_end', train_end)
    hist_df.to_csv(csv_path, index=False)
    manifest_row = {
        'refit_index': int(refit_index),
        'decision_date': str(pd.Timestamp(decision_date).date()),
        'train_window_start': str(train_start.date()) if pd.notna(train_start) else None,
        'train_window_end': str(train_end.date()) if pd.notna(train_end) else None,
        'epochs_logged': int(len(hist_df)),
        'train_objective': float(getattr(trainer, 'train_objective', np.nan)),
        'best_validation_loss': float(getattr(trainer, 'best_validation_loss', np.nan)),
        'history_csv': _relative_output_path(csv_path, output_dir),
    }
    return csv_path, manifest_row


def _write_pipinn_training_manifest(output_dir: Path, manifest_rows: list[dict[str, Any]]) -> Path | None:
    if not manifest_rows:
        return None
    manifest_path = output_dir / 'training_logs' / 'pipinn_refit_manifest.csv'
    ensure_dir(manifest_path.parent)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def _strategy_metadata(strategy: str, cross_mode: str, *, backend: str, market_source: str = '') -> dict[str, Any]:
    strategy = str(strategy)
    cross_mode = str(cross_mode)
    backend = str(backend).lower()
    if strategy == 'predictive_static':
        return {
            'strategy': 'myopic',
            'cross_mode': cross_mode,
            'strategy_display': 'myopic',
            'strategy_legacy_label': 'predictive_static',
            'comparison_role': 'reference_only',
            'benchmark_primary': False,
            'benchmark_note': 'reference-only myopic benchmark',
            'benchmark_source': '',
        }
    if strategy == 'pgdpo':
        if backend == 'pipinn':
            return {
                'strategy': 'pipinn_traincov_diag',
                'cross_mode': 'estimated',
                'strategy_display': 'pipinn_traincov_diag',
                'strategy_legacy_label': 'pgdpo',
                'comparison_role': 'method_diagnostic',
                'benchmark_primary': False,
                'benchmark_note': 'PI-PINN trainer-default projection with frozen train-window covariance and estimated cross',
                'benchmark_source': '',
            }
        return {
            'strategy': 'pgdpo',
            'cross_mode': cross_mode,
            'strategy_display': 'pgdpo',
            'strategy_legacy_label': 'pgdpo',
            'comparison_role': 'method_ablation',
            'benchmark_primary': False,
            'benchmark_note': 'warmup direct policy output before Pontryagin projection',
            'benchmark_source': '',
        }
    if strategy == 'ppgdpo':
        if backend == 'pipinn':
            mapping = {
                'estimated': ('pipinn', 'pipinn', 'ppgdpo', 'PI-PINN value-gradient projection with estimated cross'),
                'zero': ('pipinn_zero', 'pipinn_zero', 'ppgdpo_zero', 'PI-PINN value-gradient projection with zero cross'),
                'regime_gated': ('pipinn_regime_gated', 'pipinn_regime_gated', 'ppgdpo_regime_gated', 'PI-PINN value-gradient projection with regime-gated cross'),
            }
            out_name, out_display, legacy_name, note = mapping.get(
                cross_mode,
                (f'pipinn_{cross_mode}', f'pipinn_{cross_mode}', f'ppgdpo_{cross_mode}', 'PI-PINN projected strategy'),
            )
            return {
                'strategy': out_name,
                'cross_mode': cross_mode,
                'strategy_display': out_display,
                'strategy_legacy_label': legacy_name,
                'comparison_role': 'method_candidate',
                'benchmark_primary': False,
                'benchmark_note': note,
                'benchmark_source': '',
            }
        mapping = {
            'estimated': ('ppgdpo', 'ppgdpo', 'Pontryagin projection candidate with estimated cross'),
            'zero': ('ppgdpo_zero', 'ppgdpo_zero', 'Pontryagin projection candidate with zero cross'),
            'regime_gated': ('ppgdpo_regime_gated', 'ppgdpo_regime_gated', 'Pontryagin projection candidate with regime-gated cross'),
        }
        out_name, out_display, note = mapping.get(
            cross_mode,
            (f'ppgdpo_{cross_mode}', f'ppgdpo_{cross_mode}', 'Pontryagin projection candidate'),
        )
        return {
            'strategy': out_name,
            'cross_mode': cross_mode,
            'strategy_display': out_display,
            'strategy_legacy_label': out_name,
            'comparison_role': 'method_candidate',
            'benchmark_primary': False,
            'benchmark_note': note,
            'benchmark_source': '',
        }
    note = f'market benchmark source: {market_source or "unspecified"}' if strategy == 'market' else 'standard external benchmark'
    return {
        'strategy': strategy,
        'cross_mode': cross_mode,
        'strategy_display': strategy,
        'strategy_legacy_label': strategy,
        'comparison_role': 'external_benchmark',
        'benchmark_primary': True,
        'benchmark_note': note,
        'benchmark_source': market_source if strategy == 'market' else '',
    }

def _write_benchmark_notes(
    output_dir: Path,
    *,
    include_standard_benchmarks: bool,
    standard_benchmarks: list[str],
    reference_cross_mode: str,
    benchmark_cross_mode: str,
    market_source: str,
    backend: str = 'ppgdpo',
    pipinn_ansatz_mode: str | None = None,
    pipinn_policy_output_mode: str | None = None,
) -> Path:
    benchmark_roles = {
        'predictive_static': 'reference_only',
        'pgdpo': 'method_ablation',
        'ppgdpo': 'method_candidate',
    }
    benchmark_roles.update({name: 'external_benchmark' for name in standard_benchmarks})

    strategy_label_map = {
        'myopic': 'predictive_static',
        'policy': 'pgdpo',
    }
    if str(backend).lower() == 'pipinn':
        strategy_label_map.update({
            'pipinn': 'ppgdpo',
            'pipinn_zero': 'ppgdpo_zero',
            'pipinn_regime_gated': 'ppgdpo_regime_gated',
            'pipinn_traincov_diag': 'pgdpo',
        })
    else:
        strategy_label_map.update({
            'ppgdpo': 'ppgdpo',
            'ppgdpo_zero': 'ppgdpo_zero',
            'ppgdpo_regime_gated': 'ppgdpo_regime_gated',
        })
        
    payload = {
        'version': 'v56',
        'optimizer_backend': str(backend).lower(),
        'strategy_label_map': strategy_label_map,
        'comparison_roles': benchmark_roles,
        'reference_cross_mode_label': reference_cross_mode,
        'benchmark_cross_mode_label': benchmark_cross_mode,
        'include_standard_benchmarks': bool(include_standard_benchmarks),
        'standard_benchmarks': list(standard_benchmarks),
        'notes': [
            'predictive_static is reference-only and should not be treated as a primary benchmark',
            'pgdpo denotes the warmup direct policy output before Pontryagin projection',
            'ppgdpo / ppgdpo_zero / ppgdpo_regime_gated remain the main mechanism-comparison strategies',
        ],
    }
    if str(backend).lower() == 'pipinn':
        payload['pipinn_ansatz_mode'] = str(pipinn_ansatz_mode or 'ansatz_log_transform')
        payload['pipinn_policy_output_mode'] = str(pipinn_policy_output_mode or 'pure_qp')
    if 'market' in set(standard_benchmarks):
        payload['market_benchmark_source'] = market_source
    path = output_dir / 'benchmark_notes.yaml'
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return path


def _run_factorcov_experiment(cfg: Config) -> RunArtifacts:
    data = load_dataset(cfg.data)
    returns = data.returns.copy()
    states = data.states.copy()
    factors = data.factors.copy() if data.factors is not None else None

    missing_states = [c for c in cfg.state.columns if c not in states.columns]
    if missing_states:
        raise ValueError(f'Missing state columns: {missing_states}')

    # output_dir = ensure_dir(Path(cfg.project.output_dir))
    output_dir = ensure_dir(_resolve_pipinn_output_dir(cfg))
    cfg.project.output_dir = output_dir
    ensure_dir(output_dir / 'monthly')
    plan_yaml = _write_plan(cfg, output_dir)
    all_dates, eval_dates = _prepare_windows(cfg, returns)

    backend = _optimizer_backend(cfg)
    tc = cfg.comparison.transaction_cost_bps / 10000.0
    protocol_meta = _protocol_constants(cfg)
    monthly_rows: list[dict[str, Any]] = []
    prev_mu_for_cov_update: pd.Series | None = None
    holdings: dict[tuple[str, str], np.ndarray] = {
        ('myopic', 'diag'): np.zeros(returns.shape[1], dtype=float),
        ('myopic', 'full'): np.zeros(returns.shape[1], dtype=float),
        ('projected', 'diag'): np.zeros(returns.shape[1], dtype=float),
        ('projected', 'full'): np.zeros(returns.shape[1], dtype=float),
    }
    cached = None
    last_meta: dict[str, Any] = {
        'mu_l1': np.nan,
        'factor_var_json': json.dumps({}),
        'factor_mu_json': json.dumps({}),
        'mean_model_kind': cfg.mean_model.kind,
    }

    for i, date_t in enumerate(eval_dates):
        pos = all_dates.get_loc(date_t)
        train_dates = _train_dates_for_decision(cfg, all_dates, pos)
        if len(train_dates) < cfg.split.min_train_months:
            continue

        refit_now = _should_refit(cfg, i, cached)
        if refit_now:
            cached = _fit_models_for_window(cfg, states, returns, factors, train_dates)
            prev_mu_for_cov_update = None
        _, _, _, factor_repr, mean_model, cov_model = cached
        if (not refit_now) and prev_mu_for_cov_update is not None:
            cov_model.update_with_realized(returns.loc[date_t], prev_mu_for_cov_update)

        state_row = states.loc[date_t, cfg.state.columns]
        latest_factor_return = _latest_factor_return_row(cfg, factors, factor_repr, pd.Timestamp(date_t))
        regime_prob = _resolve_regime_probability(cov_model, mean_model, latest_factor_return)
        mu = mean_model.predict(state_row, latest_factor_return=latest_factor_return, regime_weight=regime_prob if mean_model.kind == 'factor_apt_regime' else None)
        cov_fc = cov_model.predict(state_row, latest_factor_return, factor_repr.loadings, factor_repr.residual_var)
        cov_full = cov_fc.asset_cov
        cov_diag = np.diag(np.diag(cov_full))
        mu_arr = mu.to_numpy(dtype=float)
        factor_mu = mean_model.predict_factor_means(state_row, latest_factor_return=latest_factor_return, regime_weight=regime_prob if mean_model.kind == 'factor_apt_regime' else None)

        rebalance_now = _should_rebalance(cfg, i)
        targets: dict[tuple[str, str], np.ndarray] = {}
        if rebalance_now:
            base_w = solve_mean_variance(
                mu_arr,
                cov_diag,
                cfg.policy.risk_aversion,
                cfg.policy.risky_cap,
                steps=max(250, cfg.policy.pgd_steps),
                step_size=cfg.policy.step_size,
            )
            myopic_diag = base_w
            myopic_full = solve_mean_variance(
                mu_arr,
                cov_full,
                cfg.policy.risk_aversion,
                cfg.policy.risky_cap,
                steps=max(250, cfg.policy.pgd_steps),
                step_size=cfg.policy.step_size,
            )
            projected_diag = solve_projected(
                mu_arr,
                cov_diag,
                base_w,
                holdings[('projected', 'diag')],
                cfg.policy.risk_aversion,
                cfg.policy.risky_cap,
                cfg.policy.turnover_penalty,
                steps=cfg.policy.pgd_steps,
                step_size=cfg.policy.step_size,
            )
            projected_full = solve_projected(
                mu_arr,
                cov_full,
                base_w,
                holdings[('projected', 'full')],
                cfg.policy.risk_aversion,
                cfg.policy.risky_cap,
                cfg.policy.turnover_penalty,
                steps=cfg.policy.pgd_steps,
                step_size=cfg.policy.step_size,
            )
            targets = {
                ('myopic', 'diag'): myopic_diag,
                ('myopic', 'full'): myopic_full,
                ('projected', 'diag'): projected_diag,
                ('projected', 'full'): projected_full,
            }
            last_meta = {
                'mu_l1': float(np.abs(mu_arr).sum()),
                'factor_var_json': json.dumps({k: float(v) for k, v in cov_fc.factor_var.items()}),
                'factor_mu_json': json.dumps({k: float(v) for k, v in (factor_mu if factor_mu is not None else pd.Series(dtype=float)).items()}),
                'mean_model_kind': cfg.mean_model.kind,
            }

        next_date = all_dates[pos + 1]
        realized_ret = returns.loc[next_date].to_numpy(dtype=float)

        for (strategy, cov_mode), current_holdings in list(holdings.items()):
            if rebalance_now:
                weights_for_return = np.asarray(targets[(strategy, cov_mode)], dtype=float)
                turnover = float(np.abs(weights_for_return - current_holdings).sum())
            else:
                weights_for_return = np.asarray(current_holdings, dtype=float)
                turnover = 0.0
            gross = float(weights_for_return @ realized_ret)
            net = gross - tc * turnover
            monthly_rows.append({
                **protocol_meta,
                'decision_date': pd.Timestamp(date_t),
                'return_date': pd.Timestamp(next_date),
                'strategy': strategy,
                'cov_mode': cov_mode,
                'gross_return': gross,
                'net_return': net,
                'turnover': turnover,
                'risky_weight': float(weights_for_return.sum()),
                'weights_json': json.dumps({asset: float(weight) for asset, weight in zip(returns.columns, weights_for_return)}),
                **last_meta,
            })
            holdings[(strategy, cov_mode)] = _drift_holdings(weights_for_return, realized_ret)
        prev_mu_for_cov_update = mu.reindex(returns.columns)

    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        raise ValueError('No backtest rows were produced. Check split dates, data alignment, and min_train_months.')

    monthly.to_csv(output_dir / 'monthly' / 'monthly_paths.csv', index=False)
    zero_summary = _augment_summary(_summarize(monthly, cfg.policy.risk_aversion, return_col='gross_return', group_cols=['strategy', 'cov_mode']), cfg)
    zero_summary.to_csv(output_dir / 'comparison_cov_modes_zero_cost_summary.csv', index=False)
    all_costs_summary = _augment_summary(_summarize(monthly, cfg.policy.risk_aversion, return_col='net_return', group_cols=['strategy', 'cov_mode']), cfg)
    all_costs_summary.to_csv(output_dir / 'comparison_cov_modes_all_costs_summary.csv', index=False)
    monthly.to_csv(output_dir / 'comparison_results.csv', index=False)

    return RunArtifacts(
        output_dir=output_dir,
        summary_zero_cost=output_dir / 'comparison_cov_modes_zero_cost_summary.csv',
        summary_with_costs=output_dir / 'comparison_cov_modes_all_costs_summary.csv',
        plan_yaml=plan_yaml,
    )


def _run_ppgdpo_experiment(cfg: Config) -> RunArtifacts:
    data = load_dataset(cfg.data)
    returns = data.returns.copy()
    states = data.states.copy()
    factors = data.factors.copy() if data.factors is not None else None

    missing_states = [c for c in cfg.state.columns if c not in states.columns]
    if missing_states:
        raise ValueError(f'Missing state columns: {missing_states}')

    # output_dir = ensure_dir(Path(cfg.project.output_dir))
    output_dir = ensure_dir(_resolve_pipinn_output_dir(cfg))
    cfg.project.output_dir = output_dir
    ensure_dir(output_dir / 'monthly')
    plan_yaml = _write_plan(cfg, output_dir)
    all_dates, eval_dates = _prepare_windows(cfg, returns)

    backend = _optimizer_backend(cfg)
    emit_pipinn_frozen_traincov = _emit_pipinn_frozen_traincov_strategy(cfg)
    save_training_logs = backend == 'pipinn' and bool(getattr(cfg.pipinn, 'save_training_logs', False))
    show_progress = backend == 'pipinn' and bool(getattr(cfg.pipinn, 'show_progress', False))
    
    tc = cfg.comparison.transaction_cost_bps / 10000.0
    protocol_meta = _protocol_constants(cfg)
    cross_modes = list(dict.fromkeys(str(mode) for mode in cfg.comparison.cross_modes))
    reference_cross_mode = str(getattr(cfg.comparison, 'reference_cross_mode_label', 'reference') or 'reference')
    benchmark_cross_mode = str(getattr(cfg.comparison, 'benchmark_cross_mode_label', 'benchmark') or 'benchmark')
    include_standard_benchmarks = bool(getattr(cfg.comparison, 'include_standard_benchmarks', True))
    standard_benchmarks = [str(item) for item in getattr(cfg.comparison, 'standard_benchmarks', ['equal_weight', 'min_variance', 'risk_parity'])]
    benchmark_set = set(standard_benchmarks if include_standard_benchmarks else [])
    effective_risky_cap = _effective_risky_cap(cfg)

    market_factor_col = _detect_market_factor_column(factors, list(getattr(cfg.comparison, 'market_factor_candidates', [])))
    market_source = f'factor:{market_factor_col}' if market_factor_col is not None else 'equal_weight_proxy'

    weight_strategy_slots: list[tuple[str, str]] = [('predictive_static', reference_cross_mode)]
    if backend == 'ppgdpo' or emit_pipinn_frozen_traincov:
        weight_strategy_slots.append(('pgdpo', reference_cross_mode))
    weight_strategy_slots.extend([('ppgdpo', cross_mode) for cross_mode in cross_modes])
    for benchmark_name in ['equal_weight', 'min_variance', 'risk_parity']:
        if benchmark_name in benchmark_set:
            weight_strategy_slots.append((benchmark_name, benchmark_cross_mode))
    if ('market' in benchmark_set) and (market_factor_col is None):
        weight_strategy_slots.append(('market', benchmark_cross_mode))
    direct_strategy_slots: list[tuple[str, str]] = []
    if ('market' in benchmark_set) and (market_factor_col is not None):
        direct_strategy_slots.append(('market', benchmark_cross_mode))

    holdings: dict[tuple[str, str], np.ndarray] = {
        slot: np.zeros(returns.shape[1], dtype=float)
        for slot in weight_strategy_slots
    }
    monthly_rows: list[dict[str, object]] = []
    prev_mu_for_cov_update: pd.Series | None = None
    prev_state_pred_for_cross: pd.Series | None = None
    cached = None
    last_costates = None
    last_proj_debug: dict[str, dict[str, Any]] = {mode: {'hedge_signal': np.zeros(returns.shape[1], dtype=float)} for mode in cross_modes}
    last_train_objective = np.nan
    last_refit_step: int = 0
    last_meta: dict[str, Any] = {
        'mu_l1': np.nan,
        'factor_var_json': json.dumps({}),
        'factor_mu_json': json.dumps({}),
        'mean_model_kind': cfg.mean_model.kind,
    }

    refit_counter = 0
    training_manifest_rows: list[dict[str, Any]] = []
    last_training_log_csv: str | None = None
    last_training_window_start: str | None = None
    last_training_window_end: str | None = None

    eval_iterable = tqdm(eval_dates, desc=f"{backend}:{_protocol_label(cfg)}", unit='month') if show_progress else eval_dates
    eval_horizon_months = int(max(len(eval_dates), 1))
    for i, date_t in enumerate(eval_iterable):
        pos = all_dates.get_loc(date_t)
        train_dates = _train_dates_for_decision(cfg, all_dates, pos)
        if len(train_dates) < cfg.split.min_train_months:
            continue

        refit_now = _should_refit(cfg, i, cached)
        tau_remaining = float(max(eval_horizon_months - i, 1))
        if refit_now:
            refit_counter += 1
            last_refit_step = i
            
            state_train, state_train_next, ret_train_next, factor_repr, mean_model, cov_model = _fit_models_for_window(
                cfg, states, returns, factors, train_dates
            )
            progress_label = f"{pd.Timestamp(date_t).strftime('%Y-%m')} [{pd.Timestamp(train_dates[0]).strftime('%Y-%m')}→{pd.Timestamp(train_dates[-1]).strftime('%Y-%m')}]"
            
            # extract previous trainer from cache for warm-start (None on first refit)
            prev_trainer_for_warm = cached[5] if cached is not None else None
            _transition, cross_est, trainer = _fit_dynamic_policy_backend(
                cfg,
                state_train=state_train,
                state_train_next=state_train_next,
                ret_train_next=ret_train_next,
                factor_repr=factor_repr,
                mean_model=mean_model,
                cov_model=cov_model,
                transaction_cost=tc,
                progress_label=progress_label,
                tau_max=tau_remaining if backend == 'pipinn' else None,
                prev_trainer=prev_trainer_for_warm,
            )
            sample_cov_train = _sample_covariance(ret_train_next)
            # cached = (factor_repr, mean_model, cov_model, cross_est, trainer, sample_cov_train)
            cached = (factor_repr, mean_model, cov_model, _transition, cross_est, trainer, sample_cov_train)
            prev_mu_for_cov_update = None
            prev_state_pred_for_cross = None
            last_training_window_start = str(pd.Timestamp(train_dates[0]).date()) if len(train_dates) else None
            last_training_window_end = str(pd.Timestamp(train_dates[-1]).date()) if len(train_dates) else None
            if save_training_logs:
                training_log_path, manifest_row = _write_pipinn_training_log(
                    output_dir,
                    trainer,
                    decision_date=pd.Timestamp(date_t),
                    train_dates=train_dates,
                    refit_index=refit_counter,
                )
                last_training_log_csv = _relative_output_path(training_log_path, output_dir)
                if manifest_row is not None:
                    training_manifest_rows.append(manifest_row)
        # factor_repr, mean_model, cov_model, cross_est, trainer, sample_cov_train = cached
        factor_repr, mean_model, cov_model, transition_model, cross_est, trainer, sample_cov_train = cached
        
        if (not refit_now) and prev_mu_for_cov_update is not None:
            cov_model.update_with_realized(returns.loc[date_t], prev_mu_for_cov_update)
            if cross_est.dynamic_model is not None and prev_state_pred_for_cross is not None:
                cross_est.dynamic_model.update_with_realized(
                    realized_return=returns.loc[date_t].to_numpy(dtype=float),
                    predicted_return_mean=prev_mu_for_cov_update.to_numpy(dtype=float),
                    realized_state=states.loc[date_t, cfg.state.columns].to_numpy(dtype=float),
                    predicted_state=prev_state_pred_for_cross.to_numpy(dtype=float),
                )

        state_row = states.loc[date_t, cfg.state.columns]
        latest_factor_return = _latest_factor_return_row(cfg, factors, factor_repr, pd.Timestamp(date_t))
        regime_prob = _resolve_regime_probability(cov_model, mean_model, latest_factor_return)
        mu = mean_model.predict(state_row, latest_factor_return=latest_factor_return, regime_weight=regime_prob if mean_model.kind == 'factor_apt_regime' else None)
        cov_fc = cov_model.predict(state_row, latest_factor_return, factor_repr.loadings, factor_repr.residual_var)
        cov_full = cov_fc.asset_cov
        cov_diag = np.diag(np.diag(cov_full))
        cov_eval = cov_full if cfg.ppgdpo.covariance_mode == 'full' else cov_diag
        mu_arr = mu.to_numpy(dtype=float)
        factor_mu = mean_model.predict_factor_means(state_row, latest_factor_return=latest_factor_return, regime_weight=regime_prob if mean_model.kind == 'factor_apt_regime' else None)

        rebalance_now = _should_rebalance(cfg, i)
        targets: dict[tuple[str, str], np.ndarray] = {}
        if rebalance_now:
            if backend == 'pipinn':
                pgdpo_w = trainer.policy_weights(state_row, tau=tau_remaining)
            else:
                pgdpo_w = trainer.policy_weights(state_row)
            # pgdpo_w = trainer.policy_weights(state_row)
            # horizon_steps = int(cfg.ppgdpo.horizon_steps)
            # tau_remaining = float(max(horizon_steps - (i - last_refit_step), 1))
            # last_costates = trainer.estimate_costates(state_row, tau0=tau_remaining)
            # last_costates = trainer.estimate_costates(state_row)
            if backend == 'pipinn':
                last_costates = trainer.estimate_costates(state_row, tau0=tau_remaining)
            else:
                last_costates = trainer.estimate_costates(state_row)
            last_train_objective = float(trainer.train_objective)
            predictive_static_w = solve_mean_variance(
                mu_arr,
                cov_eval,
                cfg.policy.risk_aversion,
                effective_risky_cap,
                steps=max(250, cfg.policy.pgd_steps),
                step_size=cfg.policy.step_size,
            )
            cross_sample = cross_est.cross.to_numpy(dtype=float)
            cross_base = cross_est.dynamic_model.current_cross_covariance() if cross_est.dynamic_model is not None else cross_sample
            zero_cross = np.zeros_like(cross_base)
            regime_gated_cross = (1.0 - regime_prob) * cross_base
            cross_lookup = {
                'estimated': cross_base,
                'zero': zero_cross,
                'regime_gated': regime_gated_cross,
            }
            targets[('predictive_static', reference_cross_mode)] = predictive_static_w
            targets[('pgdpo', reference_cross_mode)] = pgdpo_w
            for cross_mode in cross_modes:
                cross_mat = cross_lookup.get(str(cross_mode), cross_base)
                # ppgdpo_w, proj_debug = solve_ppgdpo_projection(
                #     mu=mu_arr,
                #     cov=cov_eval,
                #     cross_mat=cross_mat,
                #     costates=last_costates,
                #     risky_cap=cfg.policy.risky_cap,
                #     cash_floor=cfg.policy.cash_floor,
                #     wealth=1.0,
                #     cross_scale=cfg.ppgdpo.cross_strength,
                #     eps_bar=cfg.ppgdpo.eps_bar,
                #     ridge=cfg.ppgdpo.newton_ridge,
                #     tau=cfg.ppgdpo.newton_tau,
                #     armijo=cfg.ppgdpo.newton_armijo,
                #     backtrack=cfg.ppgdpo.newton_backtrack,
                #     max_newton=cfg.ppgdpo.max_newton,
                #     tol_grad=cfg.ppgdpo.tol_grad,
                #     max_ls=cfg.ppgdpo.max_line_search,
                #     interior_margin=cfg.ppgdpo.interior_margin,
                #     clamp_neg_jxx_min=cfg.ppgdpo.clamp_neg_jxx_min,
                # )
                if backend == 'pipinn' and str(getattr(cfg.pipinn, 'policy_output_mode', 'projection')).lower() == 'pure_qp':
                    ppgdpo_w, proj_debug = trainer.policy_weights_with_debug(
                        state_row,
                        covariance=cov_eval,
                        cross_mat=cross_mat,
                        tau=tau_remaining,
                    )
                else:
                    ppgdpo_w, proj_debug = solve_ppgdpo_projection(
                        mu=mu_arr,
                        cov=cov_eval,
                        cross_mat=cross_mat,
                        costates=last_costates,
                        risky_cap=cfg.policy.risky_cap,
                        cash_floor=cfg.policy.cash_floor,
                        wealth=1.0,
                        cross_scale=cfg.ppgdpo.cross_strength,
                        eps_bar=cfg.ppgdpo.eps_bar,
                        ridge=cfg.ppgdpo.newton_ridge,
                        tau=cfg.ppgdpo.newton_tau,
                        armijo=cfg.ppgdpo.newton_armijo,
                        backtrack=cfg.ppgdpo.newton_backtrack,
                        max_newton=cfg.ppgdpo.max_newton,
                        tol_grad=cfg.ppgdpo.tol_grad,
                        max_ls=cfg.ppgdpo.max_line_search,
                        interior_margin=cfg.ppgdpo.interior_margin,
                        clamp_neg_jxx_min=cfg.ppgdpo.clamp_neg_jxx_min,
                    )
                last_proj_debug[cross_mode] = proj_debug
                targets[('ppgdpo', cross_mode)] = ppgdpo_w
            if 'equal_weight' in benchmark_set:
                targets[('equal_weight', benchmark_cross_mode)] = solve_equal_weight(returns.shape[1], effective_risky_cap)
            if 'min_variance' in benchmark_set:
                targets[('min_variance', benchmark_cross_mode)] = solve_min_variance(sample_cov_train, effective_risky_cap, steps=max(400, cfg.policy.pgd_steps))
            if 'risk_parity' in benchmark_set:
                targets[('risk_parity', benchmark_cross_mode)] = solve_risk_parity(sample_cov_train, effective_risky_cap, steps=max(500, cfg.policy.pgd_steps * 3))
            if ('market' in benchmark_set) and (market_factor_col is None):
                targets[('market', benchmark_cross_mode)] = solve_equal_weight(returns.shape[1], effective_risky_cap)
            last_meta = {
                'mu_l1': float(np.abs(mu_arr).sum()),
                'factor_var_json': json.dumps({k: float(v) for k, v in cov_fc.factor_var.items()}),
                'factor_mu_json': json.dumps({k: float(v) for k, v in (factor_mu if factor_mu is not None else pd.Series(dtype=float)).items()}),
                'mean_model_kind': cfg.mean_model.kind,
            }

        next_date = all_dates[pos + 1]
        realized_ret = returns.loc[next_date].to_numpy(dtype=float)

        for (strategy, cross_mode), current_holdings in list(holdings.items()):
            if rebalance_now:
                weights_for_return = np.asarray(targets[(strategy, cross_mode)], dtype=float)
                turnover = float(np.abs(weights_for_return - current_holdings).sum())
            else:
                weights_for_return = np.asarray(current_holdings, dtype=float)
                turnover = 0.0
            gross = float(weights_for_return @ realized_ret)
            net = gross - tc * turnover
            proj_debug = last_proj_debug.get(cross_mode, {'hedge_signal': np.zeros(returns.shape[1], dtype=float)}) if strategy == 'ppgdpo' else {'hedge_signal': np.zeros(returns.shape[1], dtype=float)}
            meta = _strategy_metadata(strategy, cross_mode, backend=backend, market_source=market_source)
            monthly_rows.append({
                **protocol_meta,
                'decision_date': pd.Timestamp(date_t),
                'return_date': pd.Timestamp(next_date),
                **meta,
                'gross_return': gross,
                'net_return': net,
                'turnover': turnover,
                'risky_weight': float(weights_for_return.sum()),
                'stored_weights_space': 'pi',
                'weights_json': json.dumps({asset: float(weight) for asset, weight in zip(returns.columns, weights_for_return)}),
                **last_meta,
                'hedge_signal_l2': float(np.linalg.norm(np.asarray(proj_debug.get('hedge_signal', np.zeros_like(weights_for_return)), dtype=float))),
                'costate_jxy_l2': float(np.linalg.norm(last_costates.JXY)) if last_costates is not None else np.nan,
                'costate_jx': float(last_costates.JX) if last_costates is not None else np.nan,
                'costate_neg_jxx': float(max(-last_costates.JXX, 0.0)) if last_costates is not None else np.nan,
                'costate_closed_form': bool(last_costates.closed_form) if last_costates is not None else False,
                'train_objective': float(last_train_objective) if np.isfinite(last_train_objective) else np.nan,
                'regime_probability': float(regime_prob),
                'training_log_csv': last_training_log_csv,
                'training_window_start': last_training_window_start,
                'training_window_end': last_training_window_end,
                'refit_index': int(refit_counter),
            })
            holdings[(strategy, cross_mode)] = _drift_holdings(weights_for_return, realized_ret)

        for strategy, cross_mode in direct_strategy_slots:
            market_excess = np.nan
            if factors is not None and market_factor_col is not None and next_date in factors.index:
                market_excess = float(factors.loc[next_date, market_factor_col])
            gross = float(effective_risky_cap * market_excess) if np.isfinite(market_excess) else np.nan
            meta = _strategy_metadata(strategy, cross_mode, backend=backend, market_source=market_source)
            monthly_rows.append({
                **protocol_meta,
                'decision_date': pd.Timestamp(date_t),
                'return_date': pd.Timestamp(next_date),
                **meta,
                'gross_return': gross,
                'net_return': gross,
                'turnover': 0.0,
                'risky_weight': float(effective_risky_cap),
                'stored_weights_space': 'pi',
                'weights_json': json.dumps({}),
                **last_meta,
                'hedge_signal_l2': 0.0,
                'costate_jxy_l2': np.nan,
                'costate_jx': np.nan,
                'costate_neg_jxx': np.nan,
                'costate_closed_form': False,
                'train_objective': np.nan,
                'regime_probability': float(regime_prob),
                'training_log_csv': last_training_log_csv,
                'training_window_start': last_training_window_start,
                'training_window_end': last_training_window_end,
                'refit_index': int(refit_counter),
            })
        prev_mu_for_cov_update = mu.reindex(returns.columns)
        prev_state_pred_for_cross = transition_model.predict(state_row).reindex(cfg.state.columns)

    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        raise ValueError('No backtest rows were produced. Check split dates, data alignment, and min_train_months.')

    benchmark_notes_yaml = _write_benchmark_notes(
        output_dir,
        include_standard_benchmarks=include_standard_benchmarks,
        standard_benchmarks=standard_benchmarks,
        reference_cross_mode=reference_cross_mode,
        benchmark_cross_mode=benchmark_cross_mode,
        market_source=market_source,
        backend=backend,
        pipinn_ansatz_mode=str(getattr(cfg.pipinn, 'ansatz_mode', 'ansatz_log_transform')) if str(backend).lower() == 'pipinn' else None,
        pipinn_policy_output_mode=str(getattr(cfg.pipinn, 'policy_output_mode', 'pure_qp')) if str(backend).lower() == 'pipinn' else None,
    )
    

    group_cols = [
        'strategy_display',
        'strategy',
        'strategy_legacy_label',
        'comparison_role',
        'benchmark_primary',
        'benchmark_note',
        'benchmark_source',
        'cross_mode',
    ]
    monthly.to_csv(output_dir / 'monthly' / 'monthly_paths.csv', index=False)
    zero_summary = _augment_summary(_summarize(monthly, cfg.policy.risk_aversion, return_col='gross_return', group_cols=group_cols), cfg)
    zero_summary.to_csv(output_dir / 'comparison_cross_modes_zero_cost_summary.csv', index=False)
    all_costs_summary = _augment_summary(_summarize(monthly, cfg.policy.risk_aversion, return_col='net_return', group_cols=group_cols), cfg)
    all_costs_summary.to_csv(output_dir / 'comparison_cross_modes_all_costs_summary.csv', index=False)
    monthly.to_csv(output_dir / 'comparison_results.csv', index=False)
    if save_training_logs:
        _write_pipinn_training_manifest(output_dir, training_manifest_rows)

    return RunArtifacts(
        output_dir=output_dir,
        summary_zero_cost=output_dir / 'comparison_cross_modes_zero_cost_summary.csv',
        summary_with_costs=output_dir / 'comparison_cross_modes_all_costs_summary.csv',
        plan_yaml=plan_yaml,
        benchmark_notes_yaml=benchmark_notes_yaml,
    )


def run_experiment(cfg: Config) -> RunArtifacts:
    if cfg.experiment.kind == 'ppgdpo':
        return _run_ppgdpo_experiment(cfg)
    return _run_factorcov_experiment(cfg)
