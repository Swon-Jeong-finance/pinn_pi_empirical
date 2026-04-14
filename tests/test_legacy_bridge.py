from pathlib import Path
import pandas as pd
import yaml

from dynalloc_v2.legacy_bridge import (
    BaseBundleArtifacts,
    _build_v2_config_dict,
    _effective_asset_universe,
    _legacy_loader_asset_universe_name,
    _normalize_asset_universe_list,
    _resolve_selected_spec,
    _resolve_split_payload,
    _try_load_equity_universe_native,
    build_v1_base_grid,
)


def test_resolve_split_payload_cv2000():
    out = _resolve_split_payload('cv2000_final20y')
    assert out['test_start'] == '2000-01-01'
    assert out['end_date'] == '2019-12-31'


def test_resolve_selected_spec_from_yaml(tmp_path: Path):
    root = tmp_path
    d = root / 'outputs' / 'lane'
    d.mkdir(parents=True)
    payload = {
        'selected_specs': ['spec_a', 'spec_b'],
        'primary_selected_spec': 'spec_a',
    }
    (d / 'selected_spec.yaml').write_text(yaml.safe_dump(payload), encoding='utf-8')
    assert _resolve_selected_spec(root, 'lane', selected_rank=1) == 'spec_a'
    assert _resolve_selected_spec(root, 'lane', selected_rank=2) == 'spec_b'
    assert _resolve_selected_spec(root, 'lane', spec='forced_spec') == 'forced_spec'


def test_build_v2_config_dict(tmp_path: Path):
    out_dir = tmp_path / 'bundle'
    out_dir.mkdir()
    cfg = _build_v2_config_dict(
        out_dir=out_dir,
        config_stem='ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed',
        split_payload={
            'train_start': '1964-01-01',
            'test_start': '2000-01-01',
            'end_date': '2019-12-31',
        },
        state_cols=['pls1', 'pls2', 'pls3'],
        factor_cols=['Mkt-RF', 'SMB', 'HML', 'UST2Y'],
        risk_aversion=5.0,
    )
    assert cfg['experiment']['kind'] == 'ppgdpo'
    assert cfg['state']['columns'] == ['pls1', 'pls2', 'pls3']
    assert cfg['factor_model']['provided_factor_columns'][-1] == 'UST2Y'
    assert cfg['covariance_model']['factor_correlation_mode'] == 'independent'



def _write_monthly_panel(path: Path, n_assets: int, *, prefix: str = 'asset'):
    dates = pd.date_range('2000-01-31', periods=4, freq='ME')
    df = pd.DataFrame({'month': [d.strftime('%Y%m') for d in dates]})
    for i in range(n_assets):
        df[f'{prefix}_{i+1:03d}'] = 0.01 * (i + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def test_try_load_equity_universe_native_ff25(tmp_path: Path):
    _write_monthly_panel(tmp_path / 'data' / 'ff25_portfolios_monthly.csv', 25)
    out = _try_load_equity_universe_native(tmp_path, 'ff25')
    assert out is not None
    panel, source = out
    assert source.name == 'ff25_portfolios_monthly.csv'
    assert panel.shape == (4, 25)
    assert panel.index[0].day >= 28



def test_try_load_equity_universe_native_ff1_does_not_match_ff100(tmp_path: Path):
    _write_monthly_panel(tmp_path / 'data' / 'ff100_portfolios_monthly.csv', 100, prefix='ff100')
    _write_monthly_panel(tmp_path / 'data' / 'ff1_market_monthly.csv', 1, prefix='ff1')
    out = _try_load_equity_universe_native(tmp_path, 'ff1')
    assert out is not None
    panel, source = out
    assert source.name == 'ff1_market_monthly.csv'
    assert panel.shape[1] == 1


def test_normalize_asset_universe_list_defaults_and_dedupes():
    out = _normalize_asset_universe_list(['ff25', 'FF6', 'ff25', 'ff1'])
    assert out == ['ff25', 'ff6', 'ff1']


def test_effective_asset_universe_override(tmp_path: Path):
    profile_dir = tmp_path / 'profiles' / 'universe'
    profile_dir.mkdir(parents=True)
    (profile_dir / 'demo.yaml').write_text(yaml.safe_dump({'asset_universe': 'ff49'}), encoding='utf-8')
    assert _effective_asset_universe(tmp_path, 'demo') == 'ff49'
    assert _effective_asset_universe(tmp_path, 'demo', 'ff6') == 'ff6'


def test_build_v1_base_grid_calls_builder(monkeypatch, tmp_path: Path):
    calls = []

    def fake_build_v1_base_bundle(**kwargs):
        calls.append(kwargs['asset_universe_override'])
        bundle_dir = Path(kwargs['out_dir'])
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest = bundle_dir / 'base_bundle_manifest.yaml'
        manifest.write_text('asset_universe: ' + str(kwargs['asset_universe_override']) + '\n', encoding='utf-8')
        return BaseBundleArtifacts(
            out_dir=bundle_dir,
            returns_csv=bundle_dir / 'returns_panel.csv',
            macro_csv=bundle_dir / 'macro_panel.csv',
            ff3_csv=bundle_dir / 'ff3_panel.csv',
            ff5_csv=bundle_dir / 'ff5_panel.csv',
            bond_csv=bundle_dir / 'bond_panel.csv',
            manifest_yaml=manifest,
        )

    monkeypatch.setattr('dynalloc_v2.legacy_bridge.build_v1_base_bundle', fake_build_v1_base_bundle)
    artifacts = build_v1_base_grid(
        v1_root=tmp_path,
        config_stem='demo_cfg',
        out_dir=tmp_path / 'grid',
        fred_api_key='dummy',
        asset_universes=['ff1', 'ff25', 'ff100'],
    )
    assert calls == ['ff1', 'ff25', 'ff100']
    assert artifacts.entry_count == 3
    payload = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    assert payload['asset_universes'] == ['ff1', 'ff25', 'ff100']


def test_legacy_loader_asset_universe_name_maps_ff25():
    assert _legacy_loader_asset_universe_name('ff25') == 'ff25_szbm'
    assert _legacy_loader_asset_universe_name('ff6') == 'ff6'
