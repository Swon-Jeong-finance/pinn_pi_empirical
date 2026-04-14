from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .bridge_common import (
    BaseBundleArtifacts,
    _build_v2_config_dict,
    _ensure_on_syspath,
    _load_yaml,
    _read_monthly_panel_csv,
)
from .experiment_windows import profile_payload, resolve_split_payload as resolve_calendar_window
from .workspace import default_french_cache_dir, default_fred_cache_dir


@dataclass
class BridgeArtifacts:
    out_dir: Path
    returns_csv: Path
    states_csv: Path
    factors_csv: Path
    config_yaml: Path
    metadata_yaml: Path


@dataclass
class BaseGridArtifacts:
    out_dir: Path
    manifest_yaml: Path
    entry_count: int


DEFAULT_MARKET_UNIVERSES: tuple[str, ...] = ('ff1', 'ff6', 'ff25', 'ff38', 'ff49', 'ff100')


def _seed_shared_french_cache(v1_root: Path, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_dir = v1_root / 'legacy' / 'vendor' / 'pgdpo_legacy_v69' / '_cache_french'
    if not src_dir.exists():
        return
    for path in src_dir.glob('*.zip'):
        dst = cache_dir / path.name
        if not dst.exists():
            try:
                dst.write_bytes(path.read_bytes())
            except Exception:
                continue


def _patch_french_cache_dir(modules: _ImportedV1, cache_dir: Path) -> None:
    french_data = modules.french_data
    if french_data is None or not hasattr(french_data, 'FrenchDownloadConfig'):
        return
    cls = french_data.FrenchDownloadConfig

    def __init__(self, cache_dir: Path = cache_dir, refresh: bool = False):
        self.cache_dir = Path(cache_dir)
        self.refresh = bool(refresh)

    cls.__init__ = __init__


def _prepare_shared_legacy_caches(v1_root: Path, modules: _ImportedV1) -> tuple[Path, Path]:
    french_cache = default_french_cache_dir()
    fred_cache = default_fred_cache_dir()
    _seed_shared_french_cache(v1_root, french_cache)
    _patch_french_cache_dir(modules, french_cache)
    return french_cache, fred_cache


def _resolve_selected_spec(v1_root: Path, config_stem: str, *, selected_rank: int = 1, spec: str | None = None) -> str:
    if spec:
        return str(spec)
    path = v1_root / 'outputs' / config_stem / 'selected_spec.yaml'
    payload = _load_yaml(path)
    specs = payload.get('selected_specs') or []
    if not specs:
        spec0 = payload.get('primary_selected_spec')
        if not spec0:
            raise RuntimeError(f'No selected spec found in {path}')
        return str(spec0)
    idx = max(0, int(selected_rank) - 1)
    if idx >= len(specs):
        raise IndexError(f'selected_rank={selected_rank} exceeds selected_specs length={len(specs)} in {path}')
    return str(specs[idx])


@dataclass
class _ImportedV1:
    asset_universes: Any | None = None
    french_data: Any | None = None
    crsp_bond: Any | None = None
    state_specs: Any | None = None
    discrete_latent_model: Any | None = None
    macro_pool: Any | None = None
    fred_macro: Any | None = None


def _import_v1_modules(
    v1_root: Path,
    *,
    include_asset_universes: bool = True,
    include_state_modules: bool = True,
) -> _ImportedV1:
    _ensure_on_syspath(v1_root / 'src')
    _ensure_on_syspath(v1_root / 'legacy' / 'vendor' / 'pgdpo_legacy_v69')
    return _ImportedV1(
        asset_universes=importlib.import_module('pgdpo_yahoo.asset_universes') if include_asset_universes else None,
        french_data=importlib.import_module('pgdpo_yahoo.french_data'),
        crsp_bond=importlib.import_module('pgdpo_yahoo.crsp_bond'),
        state_specs=importlib.import_module('pgdpo_yahoo.state_specs') if include_state_modules else None,
        discrete_latent_model=importlib.import_module('pgdpo_yahoo.discrete_latent_model') if include_state_modules else None,
        macro_pool=importlib.import_module('dynalloc.macro_pool'),
        fred_macro=importlib.import_module('pgdpo_yahoo.fred_macro'),
    )


def _macro_cfg_from_profile(v1_root: Path, profile_name: str) -> SimpleNamespace:
    payload = _load_yaml(v1_root / 'profiles' / 'macro' / f'{profile_name}.yaml')
    feature_ids = list(payload.get('feature_ids') or [])
    return SimpleNamespace(
        pool=str(payload.get('pool', 'custom')),
        feature_ids=feature_ids,
        effective_feature_ids=feature_ids,
        macro3_columns=list(payload.get('macro3_columns') or ['infl_yoy', 'term_spread', 'default_spread']),
    )


def _resolve_asset_universe(v1_root: Path, profile_name: str) -> str:
    payload = _load_yaml(v1_root / 'profiles' / 'universe' / f'{profile_name}.yaml')
    asset_universe = payload.get('asset_universe')
    if not asset_universe:
        raise RuntimeError(f'asset_universe missing in profile {profile_name}')
    return str(asset_universe)




def _legacy_loader_asset_universe_name(asset_universe: str) -> str:
    universe = str(asset_universe).strip().lower()
    if universe == 'ff25':
        return 'ff25_szbm'
    return universe

def _effective_asset_universe(v1_root: Path, profile_name: str, asset_universe_override: str | None = None) -> str:
    if asset_universe_override is None:
        return _resolve_asset_universe(v1_root, profile_name)
    universe = str(asset_universe_override).strip().lower()
    if universe not in _NATIVE_UNIVERSE_SPECS:
        supported = ', '.join(sorted(_NATIVE_UNIVERSE_SPECS))
        raise ValueError(f'Unsupported asset_universe_override={asset_universe_override!r}. Supported: {supported}')
    return universe


def _normalize_asset_universe_list(asset_universes: list[str] | tuple[str, ...] | None) -> list[str]:
    values = list(asset_universes or DEFAULT_MARKET_UNIVERSES)
    normalized: list[str] = []
    seen: set[str] = set()
    supported = set(_NATIVE_UNIVERSE_SPECS)
    for value in values:
        universe = str(value).strip().lower()
        if not universe:
            continue
        if universe not in supported:
            supported_text = ', '.join(sorted(supported))
            raise ValueError(f'Unsupported asset universe {value!r}. Supported: {supported_text}')
        if universe in seen:
            continue
        seen.add(universe)
        normalized.append(universe)
    if not normalized:
        raise ValueError('asset_universes resolved to an empty list')
    return normalized


def _resolve_split_payload(split_profile: str) -> dict[str, str]:
    return dict(profile_payload(split_profile))


_NATIVE_UNIVERSE_SPECS: dict[str, dict[str, Any]] = {
    'ff1': {'patterns': (r'ff1(?!\d)', r'1x1'), 'min_cols': 1, 'max_cols': 2},
    'ff6': {'patterns': (r'ff6(?!\d)', r'2x3', r'6[_-]?port'), 'min_cols': 6, 'max_cols': 8},
    'ff25': {'patterns': (r'ff25(?!\d)', r'5x5', r'25[_-]?port'), 'min_cols': 25, 'max_cols': 30},
    'ff38': {'patterns': (r'ff38(?!\d)', r'38[_-]?port'), 'min_cols': 38, 'max_cols': 45},
    'ff49': {'patterns': (r'ff49(?!\d)', r'49[_-]?ind', r'49[_-]?port'), 'min_cols': 49, 'max_cols': 55},
    'ff100': {'patterns': (r'ff100(?!\d)', r'10x10', r'100[_-]?port'), 'min_cols': 100, 'max_cols': 110},
}


def _score_native_universe_file(path: Path, asset_universe: str) -> float:
    spec = _NATIVE_UNIVERSE_SPECS.get(asset_universe)
    if spec is None:
        return float('-inf')
    name = path.stem.lower()
    score = float('-inf')
    for idx, pattern in enumerate(spec['patterns']):
        if re.search(pattern, name):
            score = max(score, 100.0 - 10.0 * idx)
    return score


def _try_load_equity_universe_native(v1_root: Path, asset_universe: str) -> tuple[pd.DataFrame, Path] | None:
    universe = str(asset_universe).lower()
    spec = _NATIVE_UNIVERSE_SPECS.get(universe)
    if spec is None:
        return None
    data_dir = v1_root / 'data'
    if not data_dir.exists():
        return None
    candidates: list[tuple[float, int, Path, pd.DataFrame]] = []
    for path in sorted(data_dir.rglob('*.csv')):
        score = _score_native_universe_file(path, universe)
        if not np.isfinite(score):
            continue
        try:
            panel = _read_monthly_panel_csv(path)
        except Exception:
            continue
        ncols = int(panel.shape[1])
        if ncols < int(spec['min_cols']) or ncols > int(spec['max_cols']):
            continue
        closeness = abs(ncols - int(spec['min_cols']))
        candidates.append((score, -closeness, path, panel))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], -len(x[2].name)), reverse=True)
    _score, _closeness, path, panel = candidates[0]
    return panel, path


def _load_bond_excess_panel(v1_root: Path, modules: _ImportedV1, bond_hook: str, rf: pd.Series) -> pd.DataFrame:
    hook = str(bond_hook).lower()
    data_dir = v1_root / 'data'
    if hook in {'none', 'no_bond', ''}:
        return pd.DataFrame(index=rf.index)
    if hook == 'ust10y':
        spec_text = f'UST10Y={data_dir / "bond10y.csv"}@bond10y_ret'
    elif hook == 'curve_core':
        spec_text = (
            f'UST2Y={data_dir / "bond2y.csv"}@ret2y,'
            f'UST5Y={data_dir / "bond5y.csv"}@ret5y,'
            f'UST10Y={data_dir / "bond10y.csv"}@bond10y_ret'
        )
    else:
        raise KeyError(f'Unsupported bond_hook for bridge: {bond_hook}')
    panel = modules.crsp_bond.load_crsp_bond_panel_from_spec_text(spec_text)
    common = panel.index.intersection(rf.index)
    if len(common) == 0:
        raise RuntimeError(f'No common dates between bond panel and RF for bond_hook={bond_hook}')
    panel = panel.loc[common]
    rf = rf.loc[common]
    return panel.sub(rf, axis=0)


def _choose_factor_panel(ff3: pd.DataFrame, ff5: pd.DataFrame, bond_excess: pd.DataFrame, factor_mode: str) -> pd.DataFrame:
    mode = str(factor_mode)
    if mode == 'ff5_curve_core':
        return pd.concat([ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], bond_excess], axis=1)
    if mode == 'ff3_curve_core':
        return pd.concat([ff3[['Mkt-RF', 'SMB', 'HML']], bond_excess], axis=1)
    if mode == 'ff5_only':
        return ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()
    raise KeyError(f'Unsupported factor_mode={factor_mode!r}')


def build_v1_lane_bundle(
    *,
    v1_root: str | Path,
    config_stem: str,
    out_dir: str | Path,
    fred_api_key: str | None = None,
    selected_rank: int = 1,
    spec: str | None = None,
    factor_mode: str = 'ff5_curve_core',
    refresh_fred: bool = False,
    risk_aversion: float = 5.0,
    asset_universe_override: str | None = None,
    split_profile_override: str | None = None,
    split_train_start_override: str | None = None,
    split_train_pool_end_override: str | None = None,
    split_test_start_override: str | None = None,
    split_end_date_override: str | None = None,
) -> BridgeArtifacts:
    v1_root = Path(v1_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_yaml(v1_root / 'configs' / f'{config_stem}.yaml')

    universe_profile = str(cfg.get('universe', {}).get('profile'))
    bond_hook = str(cfg.get('universe', {}).get('bond_hook', 'none'))
    split_profile = str(cfg.get('split', {}).get('profile'))
    macro_profile = str(cfg.get('macro', {}).get('profile'))
    split_payload, split_meta = resolve_calendar_window(
        base_profile=split_profile,
        split_profile_override=split_profile_override,
        train_start_override=split_train_start_override,
        train_pool_end_override=split_train_pool_end_override,
        test_start_override=split_test_start_override,
        end_date_override=split_end_date_override,
    )
    end_date = pd.Timestamp(split_payload['end_date'])
    selected_spec = _resolve_selected_spec(v1_root, config_stem, selected_rank=selected_rank, spec=spec)

    if not fred_api_key:
        fred_api_key = os.environ.get('FRED_API_KEY')
    if not fred_api_key:
        raise RuntimeError('A FRED API key is required. Pass --fred-api-key or set FRED_API_KEY.')

    asset_universe = _effective_asset_universe(v1_root, universe_profile, asset_universe_override)
    native_universe = _try_load_equity_universe_native(v1_root, asset_universe)
    modules = _import_v1_modules(v1_root, include_asset_universes=native_universe is None, include_state_modules=True)
    _shared_french_cache_dir, shared_fred_cache_dir = _prepare_shared_legacy_caches(v1_root, modules)
    if native_universe is None:
        if modules.asset_universes is None:
            raise RuntimeError('legacy asset_universes module was not imported')
        legacy_loader_universe = _legacy_loader_asset_universe_name(asset_universe)
        eq_total = modules.asset_universes.load_equity_universe_monthly(legacy_loader_universe)
        universe_loader = 'legacy_v1_asset_universes'
        native_universe_path = None
    else:
        eq_total, native_universe_path = native_universe
        universe_loader = 'v2_native_csv'

    ff3_all = modules.french_data.load_ff_factors_monthly()
    ff5_all = modules.french_data.load_ff5_factors_monthly()
    common_ff = ff3_all.index.intersection(ff5_all.index)
    ff3_all = ff3_all.loc[common_ff]
    ff5_all = ff5_all.loc[common_ff]
    rf = ff5_all['RF'].astype(float).rename('RF')

    common_eq = eq_total.index.intersection(rf.index)
    eq_total = eq_total.loc[common_eq]
    rf = rf.loc[common_eq]
    eq_excess = eq_total.sub(rf, axis=0)

    bond_excess = _load_bond_excess_panel(v1_root, modules, bond_hook, rf)
    returns_excess = pd.concat([eq_excess, bond_excess], axis=1)

    macro_cfg = _macro_cfg_from_profile(v1_root, macro_profile)
    fred_cfg = modules.fred_macro.FredMacroConfig(
        api_key=str(fred_api_key),
        cache_dir=shared_fred_cache_dir,
        refresh=bool(refresh_fred),
    )
    macro_full = modules.macro_pool.build_macro_pool_monthly(fred_cfg=fred_cfg, macro_config=macro_cfg)
    macro3 = macro_full.loc[:, list(macro_cfg.macro3_columns)].copy()
    macro7 = macro_full.copy()

    ff3 = ff3_all[['Mkt-RF', 'SMB', 'HML']].copy()
    ff5 = ff5_all[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()

    eq_total = eq_total.loc[eq_total.index <= end_date]
    rf = rf.loc[rf.index <= end_date]
    eq_excess = eq_excess.loc[eq_excess.index <= end_date]
    returns_excess = returns_excess.loc[returns_excess.index <= end_date]
    if not bond_excess.empty:
        bond_excess = bond_excess.loc[bond_excess.index <= end_date]
    macro3 = macro3.loc[macro3.index <= end_date]
    macro7 = macro7.loc[macro7.index <= end_date]
    ff3 = ff3.loc[ff3.index <= end_date]
    ff5 = ff5.loc[ff5.index <= end_date]

    common = returns_excess.index
    common = common.intersection(macro3.index).intersection(macro7.index).intersection(ff3.index).intersection(ff5.index)
    if not bond_excess.empty:
        common = common.intersection(bond_excess.index)
    if len(common) == 0:
        raise RuntimeError('No common monthly dates across returns, macro, and factors.')
    returns_excess = returns_excess.loc[common].sort_index()
    macro3 = macro3.loc[common].sort_index()
    macro7 = macro7.loc[common].sort_index()
    ff3 = ff3.loc[common].sort_index()
    ff5 = ff5.loc[common].sort_index()
    bond_excess = bond_excess.loc[common].sort_index() if not bond_excess.empty else pd.DataFrame(index=common)

    train_pool_end = pd.Timestamp(split_payload['train_pool_end'])
    train_candidates = returns_excess.index[returns_excess.index <= train_pool_end]
    if len(train_candidates) == 0:
        raise RuntimeError(f'No train dates at or before train_pool_end={train_pool_end.date()}')
    train_end = int(returns_excess.index.get_loc(train_candidates[-1]))

    pca_cfg = modules.discrete_latent_model.LatentPCAConfig(n_components=3, standardize=True, random_state=0)
    model, y_df, _z_df = modules.state_specs.build_model_for_state_spec(
        selected_spec,
        r_ex=returns_excess,
        macro3=macro3,
        macro7=macro7,
        ff3=ff3,
        ff5=ff5,
        train_end=train_end,
        pca_cfg=pca_cfg,
        pls_horizon=12,
        pls_smooth_span=6,
        bond_asset_names=list(bond_excess.columns),
    )
    _ = model  # kept for parity / future metadata

    factors_panel = _choose_factor_panel(ff3, ff5, bond_excess, factor_mode=factor_mode)
    common2 = returns_excess.index.intersection(y_df.index).intersection(factors_panel.index)
    returns_excess = returns_excess.loc[common2].copy()
    y_df = y_df.loc[common2].copy()
    factors_panel = factors_panel.loc[common2].copy()

    returns_out = out_dir / 'returns_panel.csv'
    states_out = out_dir / 'states_panel.csv'
    factors_out = out_dir / 'factors_panel.csv'
    returns_excess.reset_index(names='date').to_csv(returns_out, index=False)
    y_df.reset_index(names='date').to_csv(states_out, index=False)
    factors_panel.reset_index(names='date').to_csv(factors_out, index=False)

    metadata = {
        'config_stem': config_stem,
        'selected_rank': int(selected_rank),
        'selected_spec': selected_spec,
        'asset_universe': asset_universe,
        'asset_universe_override': str(asset_universe_override).lower() if asset_universe_override is not None else None,
        'legacy_loader_asset_universe': _legacy_loader_asset_universe_name(asset_universe),
        'equity_loader': universe_loader,
        'native_universe_csv': str(native_universe_path) if native_universe_path is not None else None,
        'bond_hook': bond_hook,
        'split_profile': split_meta.get('split_profile') or split_profile,
        'split_profile_source': split_meta.get('split_source'),
        'split_description': split_meta.get('split_description'),
        'split_overrides': split_meta.get('split_overrides'),
        'split': split_payload,
        'macro_profile': macro_profile,
        'macro_features': list(macro7.columns),
        'factor_mode': factor_mode,
        'factor_columns': list(factors_panel.columns),
        'state_columns': list(y_df.columns),
        'notes': 'returns_panel contains monthly excess returns so cash=0 maps to the RF benchmark in v2.',
    }
    metadata_yaml = out_dir / 'bridge_metadata.yaml'
    metadata_yaml.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')

    cfg_payload = _build_v2_config_dict(
        out_dir=out_dir,
        config_stem=config_stem,
        split_payload=split_payload,
        state_cols=list(y_df.columns),
        factor_cols=list(factors_panel.columns),
        risk_aversion=risk_aversion,
    )
    cfg_path = out_dir / 'config_empirical_ppgdpo_apt.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False), encoding='utf-8')

    return BridgeArtifacts(
        out_dir=out_dir,
        returns_csv=returns_out,
        states_csv=states_out,
        factors_csv=factors_out,
        config_yaml=cfg_path,
        metadata_yaml=metadata_yaml,
    )


def build_v1_base_bundle(
    *,
    v1_root: str | Path,
    config_stem: str,
    out_dir: str | Path,
    fred_api_key: str | None = None,
    refresh_fred: bool = False,
    asset_universe_override: str | None = None,
    split_profile_override: str | None = None,
    split_train_start_override: str | None = None,
    split_train_pool_end_override: str | None = None,
    split_test_start_override: str | None = None,
    split_end_date_override: str | None = None,
) -> BaseBundleArtifacts:
    v1_root = Path(v1_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_yaml(v1_root / 'configs' / f'{config_stem}.yaml')

    universe_profile = str(cfg.get('universe', {}).get('profile'))
    bond_hook = str(cfg.get('universe', {}).get('bond_hook', 'none'))
    split_profile = str(cfg.get('split', {}).get('profile'))
    macro_profile = str(cfg.get('macro', {}).get('profile'))
    split_payload, split_meta = resolve_calendar_window(
        base_profile=split_profile,
        split_profile_override=split_profile_override,
        train_start_override=split_train_start_override,
        train_pool_end_override=split_train_pool_end_override,
        test_start_override=split_test_start_override,
        end_date_override=split_end_date_override,
    )
    end_date = pd.Timestamp(split_payload['end_date'])

    if not fred_api_key:
        fred_api_key = os.environ.get('FRED_API_KEY')
    if not fred_api_key:
        raise RuntimeError('A FRED API key is required. Pass --fred-api-key or set FRED_API_KEY.')

    asset_universe = _effective_asset_universe(v1_root, universe_profile, asset_universe_override)
    native_universe = _try_load_equity_universe_native(v1_root, asset_universe)
    modules = _import_v1_modules(v1_root, include_asset_universes=native_universe is None, include_state_modules=False)
    _shared_french_cache_dir, shared_fred_cache_dir = _prepare_shared_legacy_caches(v1_root, modules)
    if native_universe is None:
        if modules.asset_universes is None:
            raise RuntimeError('legacy asset_universes module was not imported')
        eq_total = modules.asset_universes.load_equity_universe_monthly(asset_universe)
        universe_loader = 'legacy_v1_asset_universes'
        native_universe_path = None
    else:
        eq_total, native_universe_path = native_universe
        universe_loader = 'v2_native_csv'

    ff3_all = modules.french_data.load_ff_factors_monthly()
    ff5_all = modules.french_data.load_ff5_factors_monthly()
    common_ff = ff3_all.index.intersection(ff5_all.index)
    ff3_all = ff3_all.loc[common_ff]
    ff5_all = ff5_all.loc[common_ff]
    rf = ff5_all['RF'].astype(float).rename('RF')

    common_eq = eq_total.index.intersection(rf.index)
    eq_total = eq_total.loc[common_eq]
    rf = rf.loc[common_eq]
    eq_excess = eq_total.sub(rf, axis=0)

    bond_excess = _load_bond_excess_panel(v1_root, modules, bond_hook, rf)
    returns_excess = pd.concat([eq_excess, bond_excess], axis=1)

    macro_cfg = _macro_cfg_from_profile(v1_root, macro_profile)
    fred_cfg = modules.fred_macro.FredMacroConfig(
        api_key=str(fred_api_key),
        cache_dir=shared_fred_cache_dir,
        refresh=bool(refresh_fred),
    )
    macro_full = modules.macro_pool.build_macro_pool_monthly(fred_cfg=fred_cfg, macro_config=macro_cfg)
    macro7 = macro_full.copy()

    ff3 = ff3_all[['Mkt-RF', 'SMB', 'HML']].copy()
    ff5 = ff5_all[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()

    returns_excess = returns_excess.loc[returns_excess.index <= end_date]
    if not bond_excess.empty:
        bond_excess = bond_excess.loc[bond_excess.index <= end_date]
    macro7 = macro7.loc[macro7.index <= end_date]
    ff3 = ff3.loc[ff3.index <= end_date]
    ff5 = ff5.loc[ff5.index <= end_date]

    common = returns_excess.index.intersection(macro7.index).intersection(ff3.index).intersection(ff5.index)
    if not bond_excess.empty:
        common = common.intersection(bond_excess.index)
    if len(common) == 0:
        raise RuntimeError('No common monthly dates across returns, macro, and factors for base bundle.')

    returns_excess = returns_excess.loc[common].sort_index()
    macro7 = macro7.loc[common].sort_index()
    ff3 = ff3.loc[common].sort_index()
    ff5 = ff5.loc[common].sort_index()
    bond_excess = bond_excess.loc[common].sort_index() if not bond_excess.empty else pd.DataFrame(index=common)

    returns_out = out_dir / 'returns_panel.csv'
    macro_out = out_dir / 'macro_panel.csv'
    ff3_out = out_dir / 'ff3_panel.csv'
    ff5_out = out_dir / 'ff5_panel.csv'
    bond_out = out_dir / 'bond_panel.csv'

    returns_excess.reset_index(names='date').to_csv(returns_out, index=False)
    macro7.reset_index(names='date').to_csv(macro_out, index=False)
    ff3.reset_index(names='date').to_csv(ff3_out, index=False)
    ff5.reset_index(names='date').to_csv(ff5_out, index=False)
    bond_excess.reset_index(names='date').to_csv(bond_out, index=False)

    manifest = {
        'config_stem': config_stem,
        'asset_universe': asset_universe,
        'asset_universe_override': str(asset_universe_override).lower() if asset_universe_override is not None else None,
        'equity_loader': universe_loader,
        'native_universe_csv': str(native_universe_path) if native_universe_path is not None else None,
        'universe_profile': universe_profile,
        'bond_hook': bond_hook,
        'split_profile': split_meta.get('split_profile') or split_profile,
        'split_profile_source': split_meta.get('split_source'),
        'split_description': split_meta.get('split_description'),
        'split_overrides': split_meta.get('split_overrides'),
        'split': split_payload,
        'macro_profile': macro_profile,
        'macro_columns': list(macro7.columns),
        'ff3_columns': list(ff3.columns),
        'ff5_columns': list(ff5.columns),
        'bond_columns': list(bond_excess.columns),
    }
    manifest_yaml = out_dir / 'base_bundle_manifest.yaml'
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')

    return BaseBundleArtifacts(
        out_dir=out_dir,
        returns_csv=returns_out,
        macro_csv=macro_out,
        ff3_csv=ff3_out,
        ff5_csv=ff5_out,
        bond_csv=bond_out,
        manifest_yaml=manifest_yaml,
    )



def build_v1_base_grid(
    *,
    v1_root: str | Path,
    config_stem: str,
    out_dir: str | Path,
    fred_api_key: str | None = None,
    refresh_fred: bool = False,
    asset_universes: list[str] | tuple[str, ...] | None = None,
    split_profile_override: str | None = None,
    split_train_start_override: str | None = None,
    split_train_pool_end_override: str | None = None,
    split_test_start_override: str | None = None,
    split_end_date_override: str | None = None,
) -> BaseGridArtifacts:
    v1_root = Path(v1_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    universes = _normalize_asset_universe_list(asset_universes)
    entries: list[dict[str, Any]] = []
    for universe in universes:
        bundle_dir = out_dir / universe
        artifacts = build_v1_base_bundle(
            v1_root=v1_root,
            config_stem=config_stem,
            out_dir=bundle_dir,
            fred_api_key=fred_api_key,
            refresh_fred=refresh_fred,
            asset_universe_override=universe,
            split_profile_override=split_profile_override,
            split_train_start_override=split_train_start_override,
            split_train_pool_end_override=split_train_pool_end_override,
            split_test_start_override=split_test_start_override,
            split_end_date_override=split_end_date_override,
        )
        entries.append({
            'asset_universe': universe,
            'bundle_dir': str(artifacts.out_dir),
            'manifest_yaml': str(artifacts.manifest_yaml),
            'returns_csv': str(artifacts.returns_csv),
            'macro_csv': str(artifacts.macro_csv),
            'ff3_csv': str(artifacts.ff3_csv),
            'ff5_csv': str(artifacts.ff5_csv),
            'bond_csv': str(artifacts.bond_csv),
        })

    manifest = {
        'grid_name': f'{config_stem}_market_universe_grid',
        'source_v1_root': str(v1_root),
        'config_stem': config_stem,
        'split_profile_override': split_profile_override,
        'split_overrides': {
            'train_start': split_train_start_override,
            'train_pool_end': split_train_pool_end_override,
            'test_start': split_test_start_override,
            'end_date': split_end_date_override,
        },
        'asset_universes': universes,
        'entries': entries,
    }
    manifest_yaml = out_dir / 'market_universe_grid_manifest.yaml'
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')
    return BaseGridArtifacts(out_dir=out_dir, manifest_yaml=manifest_yaml, entry_count=len(entries))
