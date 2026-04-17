from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import yaml

from dynalloc_v2.factor_zoo import build_candidate_registry
from dynalloc_v2.native_selection import (
    _annotate_stage2_real_ppgdpo_scores,
    _build_cv_blocks,
    _build_selection_blocks,
    _config_covariance_payload_from_label,
    _comparison_cross_modes_for_covariance_label,
    _expand_stage2_model_specs,
    _legacy_spec_name_for_candidate,
    _selection_score_mean_first,
    _selection_score_ppgdpo_lite,
    _selection_score_ret_first,
    SelectionStage2ModelSpec,
    native_select_factor_suite,
)


def _write_panel(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(names='date').to_csv(path, index=False)


def _build_synthetic_base_bundle(base_dir: Path) -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range('1990-01-31', periods=240, freq='ME')

    base_factor = rng.normal(scale=0.03, size=len(dates))
    value_factor = 0.4 * base_factor + rng.normal(scale=0.02, size=len(dates))
    quality_factor = -0.2 * base_factor + rng.normal(scale=0.015, size=len(dates))
    bond_2y = -0.1 * base_factor + rng.normal(scale=0.01, size=len(dates))
    bond_5y = -0.15 * base_factor + rng.normal(scale=0.012, size=len(dates))
    bond_10y = -0.25 * base_factor + rng.normal(scale=0.014, size=len(dates))

    macro = pd.DataFrame(
        {
            'infl_yoy': 0.02 + 0.01 * np.sin(np.linspace(0, 12, len(dates))),
            'term_spread': 0.01 + 0.2 * bond_10y,
            'default_spread': 0.02 + 0.3 * np.abs(base_factor),
            'indpro_yoy': 0.01 + 0.2 * np.cos(np.linspace(0, 10, len(dates))),
            'unrate_chg': 0.1 * rng.normal(size=len(dates)),
            'short_rate': 0.02 + 0.15 * bond_2y,
            'gs5': 0.03 + bond_5y,
            'gs10': 0.035 + bond_10y,
        },
        index=dates,
    )
    ff3 = pd.DataFrame({'Mkt-RF': base_factor, 'SMB': value_factor, 'HML': quality_factor}, index=dates)
    ff5 = ff3.assign(RMW=quality_factor + rng.normal(scale=0.01, size=len(dates)), CMA=-0.1 * value_factor + rng.normal(scale=0.01, size=len(dates)))
    bond = pd.DataFrame({'UST2Y': bond_2y, 'UST5Y': bond_5y, 'UST10Y': bond_10y}, index=dates)

    loadings = rng.normal(size=(6, 5)) * 0.2
    ff_mat = ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].to_numpy(dtype=float)
    bond_mat = bond[['UST2Y', 'UST5Y', 'UST10Y']].to_numpy(dtype=float)
    signal = ff_mat @ loadings[:, :5].T / 3.0 + bond_mat @ rng.normal(size=(3, 6)) * 0.05
    returns = pd.DataFrame(signal + rng.normal(scale=0.03, size=signal.shape), index=dates, columns=[f'A{i+1}' for i in range(signal.shape[1])])

    _write_panel(base_dir / 'returns_panel.csv', returns)
    _write_panel(base_dir / 'macro_panel.csv', macro)
    _write_panel(base_dir / 'ff3_panel.csv', ff3)
    _write_panel(base_dir / 'ff5_panel.csv', ff5)
    _write_panel(base_dir / 'bond_panel.csv', bond)
    manifest = {
        'config_stem': 'ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed',
        'split': {
            'train_pool_start': '1964-01-01',
            'train_pool_end': '1999-12-31',
            'test_start': '2000-01-01',
            'end_date': '2009-12-31',
        },
    }
    (base_dir / 'base_bundle_manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')


def _build_fake_stage1_v1_root(v1_root: Path) -> Path:
    pkg_dir = v1_root / 'legacy' / 'vendor' / 'pgdpo_legacy_v69' / 'pgdpo_yahoo'
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / '__init__.py').write_text('', encoding='utf-8')
    (pkg_dir / 'discrete_latent_model.py').write_text(
        "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass LatentPCAConfig:\n    n_components: int\n    standardize: bool = True\n    random_state: int = 0\n",
        encoding='utf-8',
    )
    (pkg_dir / 'spec_selection.py').write_text(
        "from dataclasses import dataclass\nfrom types import SimpleNamespace\n\n\n@dataclass(frozen=True)\nclass SpecSelectionConfig:\n    window_mode: str\n    rolling_window: int\n    return_baseline: str\n    state_baseline: str\n\n\ndef evaluate_spec_predictive_diagnostics(*, spec, **kwargs):\n    base = 123.0 + float(len(str(spec)))\n    return SimpleNamespace(\n        r2_oos_ret_mean=base,\n        r2_oos_ret_median=base + 1.0,\n        r2_roll_ret_q10=base + 2.0,\n        r2_roll_ret_min=base + 3.0,\n        r2_oos_state_mean=base + 4.0,\n        r2_oos_state_median=base + 5.0,\n        r2_roll_state_q10=base + 6.0,\n        r2_roll_state_min=base + 7.0,\n        cross_mean_abs_rho=0.75,\n        cross_max_abs_rho=0.85,\n    )\n",
        encoding='utf-8',
    )
    return v1_root


def test_factor_zoo_registry_is_large_enough():
    zoo = build_candidate_registry('factor_zoo_v1')
    assert len(zoo) >= 30
    names = {c.name for c in zoo}
    assert 'ff5' in names
    assert 'pca_k4' in names
    assert 'pls_ret_ff5_macro7_H6_k2' in names
    assert 'ff5_pcares_k2' in names

def test_cross_mode_mapping_and_stage2_expansion_and_cross_estimator_payload():
    assert _comparison_cross_modes_for_covariance_label('dcc') == ['estimated', 'zero']
    assert _comparison_cross_modes_for_covariance_label('adcc') == ['estimated', 'zero']
    assert _comparison_cross_modes_for_covariance_label('regime_dcc') == ['estimated', 'zero', 'regime_gated']

    specs = _expand_stage2_model_specs([
        SelectionStage2ModelSpec(
            label='adcc',
            base_covariance_label='adcc',
            covariance_model_kind='asset_adcc',
            factor_correlation_mode='independent',
            use_persistence=False,
        ),
        SelectionStage2ModelSpec(
            label='regime_dcc',
            base_covariance_label='regime_dcc',
            covariance_model_kind='asset_regime_dcc',
            factor_correlation_mode='independent',
            use_persistence=False,
        ),
    ])
    by_base: dict[str, set[str]] = {}
    for spec in specs:
        by_base.setdefault(str(spec.base_covariance_label), set()).add(str(spec.cross_policy_label))
    assert by_base['adcc'] == {'estimated', 'zero'}
    assert by_base['regime_dcc'] == {'estimated', 'zero', 'regime_gated'}

    assert _config_covariance_payload_from_label('const')['cross_covariance_kind'] == 'sample'
    assert _config_covariance_payload_from_label('dcc')['cross_covariance_kind'] == 'dcc'
    assert _config_covariance_payload_from_label('adcc')['cross_covariance_kind'] == 'adcc'
    assert _config_covariance_payload_from_label('regime_dcc')['cross_covariance_kind'] == 'regime_dcc'


def test_legacy_spec_name_mapping_matches_v1_specs():
    zoo = build_candidate_registry('factor_zoo_v1')
    mapping = {c.name: _legacy_spec_name_for_candidate(c) for c in zoo}
    assert mapping['ff1'] == 'ff_mkt'
    assert mapping['ff3'] == 'ff3_only'
    assert mapping['ff5'] == 'ff5_only'
    assert mapping['pca_k1'] == 'pca_only_k1'
    assert mapping['pls_H12_k3'] == 'pls_H12_k3'
    assert mapping['pls_ret_macro7_H12_k2'] == 'pls_ret_macro7_H12_k2'
    assert mapping['pls_ret_ff5_macro7_H24_k3'] == 'pls_ret_ff5_macro7_H24_k3'

def test_native_factor_zoo_builds_suite(tmp_path: Path):
    base_dir = tmp_path / 'base_bundle'
    _build_synthetic_base_bundle(base_dir)

    artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite',
        candidate_zoo='factor_zoo_v1',
        max_candidates=8,
        top_k=2,
        min_train_months=60,
        rolling_window=36,
        rerank_top_n=0,
        selection_split_mode='trailing_holdout',
        selection_val_months=60,
        select_rolling_oos_window=False,
    )
    assert artifacts.selection_summary_csv.exists()
    summary = pd.read_csv(artifacts.selection_summary_csv)
    assert len(summary) == 8
    selected = yaml.safe_load(artifacts.selected_yaml.read_text(encoding='utf-8'))
    assert len(selected['selected_specs']) == 2
    assert selected['selection_split']['mode'] == 'trailing_holdout'
    assert selected['selection_split']['validation_months_requested'] == 60
    assert selected['selection_split']['final_oos_retrain_uses_train_plus_validation'] is True
    manifest_payload = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    assert len(manifest_payload['entries']) == 2
    assert selected['selection_protocol'] == 'stage1_only'


def test_native_factor_zoo_rerank_builds_suite(tmp_path: Path):
    base_dir = tmp_path / 'base_bundle'
    _build_synthetic_base_bundle(base_dir)

    artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite_rerank',
        candidate_zoo='factor_zoo_v1',
        max_candidates=2,
        top_k=1,
        min_train_months=60,
        rolling_window=36,
        rerank_top_n=1,
        selection_split_mode='trailing_holdout',
        selection_val_months=60,
        select_rolling_oos_window=False,
        rerank_covariance_models=['const', 'diag'],
        ppgdpo_lite_epochs=2,
        ppgdpo_lite_mc_rollouts=1,
        ppgdpo_lite_mc_sub_batch=1,
        selection_device='cpu',
    )
    summary = pd.read_csv(artifacts.selection_summary_csv)
    assert len(summary) == 3
    assert 'covariance_model_label' in summary.columns
    assert 'model_id' in summary.columns
    assert 'ppgdpo_lite_score_mean' in summary.columns
    assert 'ppgdpo_lite_myopic_ce_mean' in summary.columns
    assert 'ppgdpo_lite_predictive_static_ce_mean' in summary.columns
    assert 'ppgdpo_lite_equal_weight_ce_mean' in summary.columns
    assert 'ppgdpo_lite_min_variance_ce_mean' in summary.columns
    assert 'ppgdpo_lite_risk_parity_ce_mean' in summary.columns
    assert 'ppgdpo_lite_ce_delta_predictive_static_mean' in summary.columns
    assert 'ppgdpo_lite_ce_delta_equal_weight_mean' in summary.columns
    assert 'ppgdpo_lite_ce_delta_min_variance_mean' in summary.columns
    assert 'ppgdpo_lite_ce_delta_risk_parity_mean' in summary.columns
    assert 'stage2_mean_first_score' in summary.columns
    assert 'stage2_real_ppgdpo_score' in summary.columns
    assert 'selected_stage2_model' in summary.columns
    assert 'official_selected_model' in summary.columns
    assert 'official_selected_stage1' in summary.columns
    assert 'diagnostic_stage2' in summary.columns
    selected = yaml.safe_load(artifacts.selected_yaml.read_text(encoding='utf-8'))
    assert selected['selection_protocol'] == 'stage1_spec_selection_then_stage2_real_ppgdpo_model_selection'
    assert selected['selection_split']['mode'] == 'trailing_holdout'
    assert selected['selection_split']['validation_months_requested'] == 60
    assert selected['selection_split']['final_oos_retrain_uses_train_plus_validation'] is True
    assert len(selected['selected_specs']) == 1
    assert len(selected['selected_models']) == 1
    assert selected['selection_score_mode'] == 'stage1_mean_first_then_stage2_real_ppgdpo'
    assert selected['ppgdpo_lite']['rerank_covariance_models'] == ['const', 'diag']
    assert 'stage2 real selection' in selected['ppgdpo_lite']['score_mode']
    assert selected['final_model_grid_count'] == 1
    assert Path(selected['selection_stage1_csv']).exists()
    assert Path(selected['selection_stage2_csv']).exists()
    stage2 = pd.read_csv(selected['selection_stage2_csv'])
    assert set(stage2['covariance_model_label']) == {'const', 'diag'}
    assert int(stage2['selected_stage2_model'].sum()) == 1
    assert int(summary['official_selected_model'].sum()) == 1
    manifest_payload = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    assert manifest_payload['selection_protocol'] == 'stage1_spec_selection_then_stage2_real_ppgdpo_model_selection'
    assert manifest_payload['v55_strategy_label_map'] == {'myopic': 'predictive_static', 'policy': 'pgdpo'}
    assert manifest_payload['strategy_label_map'] == {'myopic': 'predictive_static', 'policy': 'pgdpo'}
    assert manifest_payload['comparison_benchmark_notes']['external_benchmarks'] == ['equal_weight', 'min_variance', 'risk_parity']
    assert manifest_payload['selection_split']['mode'] == 'trailing_holdout'
    assert manifest_payload['selection_split']['final_oos_retrain_uses_train_plus_validation'] is True
    assert len(manifest_payload['entries']) == 1
    cfg_payload = yaml.safe_load(Path(manifest_payload['entries'][0]['config_yaml']).read_text(encoding='utf-8'))
    assert cfg_payload['split']['test_start'] == '2000-01-01'


def test_selection_score_is_linear_recommendation_a():
    ret_mean = -0.001
    ret_q10 = -0.002
    state_mean = 0.400
    out = _selection_score_ret_first(ret_mean, ret_q10, state_mean)
    assert out == pytest.approx(0.70 * ret_mean + 0.20 * ret_q10 + 0.10 * state_mean)


def test_selection_score_mean_first_stage28():
    ret_mean = 0.012
    ret_q10 = -0.010
    out = _selection_score_mean_first(ret_mean, ret_q10)
    assert out == pytest.approx(0.85 * ret_mean + 0.15 * ret_q10)


def test_selection_score_ppgdpo_lite_prefers_ce_and_zero_cross_hedging_gains():
    ce_est = 0.10
    gain_myopic = 0.03
    gain_zero = 0.02
    out = _selection_score_ppgdpo_lite(ce_est, gain_myopic, gain_zero)
    assert out == pytest.approx(ce_est + gain_zero)


def test_stage2_annotation_prefers_real_ppgdpo_score_over_myopic_anchor():
    stage2 = pd.DataFrame(
        {
            'spec': ['spec_a', 'spec_a', 'spec_b', 'spec_b'],
            'model_id': ['spec_a__const', 'spec_a__dcc', 'spec_b__const', 'spec_b__diag'],
            'covariance_model_label': ['const', 'dcc', 'const', 'diag'],
            'covariance_model_kind': ['constant', 'asset_dcc', 'constant', 'state_only_diagonal'],
            'ppgdpo_lite_score_mean': [0.02, 0.12, 0.08, 0.03],
            'ppgdpo_lite_score_q10': [0.01, 0.11, 0.05, 0.02],
            'ppgdpo_lite_ce_mean': [0.02, 0.10, 0.07, 0.04],
            'ppgdpo_lite_myopic_ce_mean': [0.09, 0.03, 0.06, 0.07],
            'ppgdpo_lite_ce_delta_myopic_mean': [-0.07, 0.07, 0.01, -0.03],
            'ppgdpo_lite_ce_delta_zero_mean': [-0.05, 0.08, 0.03, -0.01],
        }
    )
    out = _annotate_stage2_real_ppgdpo_scores(stage2, ['spec_a', 'spec_b'])
    winners = out.loc[out['selected_stage2_model']].sort_values('selected_stage2_model_rank')
    assert list(winners['model_id']) == ['spec_a__dcc', 'spec_b__const']
    assert list(winners['selected_stage2_model_rank']) == [1.0, 2.0]


def test_build_cv_blocks_restores_v1_expanding_train_semantics():
    dates = pd.date_range('2000-01-31', periods=120, freq='ME')
    blocks = _build_cv_blocks(
        dates,
        train_pool_end=dates[-1],
        cv_folds=3,
        min_train_months=60,
        rolling_window=24,
        window_mode='rolling',
    )
    assert len(blocks) == 3
    assert len(blocks[0]['train_dates']) == 60
    # In v1 semantics, training history expands across folds even when window_mode='rolling'.
    assert len(blocks[1]['train_dates']) > 60
    assert len(blocks[2]['train_dates']) > len(blocks[1]['train_dates'])



def test_build_selection_blocks_trailing_holdout_uses_single_back_window():
    dates = pd.date_range('2000-01-31', periods=180, freq='ME')
    blocks = _build_selection_blocks(
        dates,
        train_start=dates[0],
        train_pool_end=dates[-1],
        split_mode='trailing_holdout',
        cv_folds=3,
        min_train_months=60,
        selection_val_months=48,
        rolling_window=24,
        window_mode='rolling',
    )
    assert len(blocks) == 1
    block = blocks[0]
    assert len(block['train_dates']) == 132
    assert len(block['val_dates']) == 48
    assert block['val_dates'][0] == dates[132]
    assert block['val_dates'][-1] == dates[-1]


def test_native_factor_zoo_can_override_calendar_window(tmp_path: Path):
    base_dir = tmp_path / 'base_bundle_override'
    _build_synthetic_base_bundle(base_dir)

    artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite_override',
        candidate_zoo='factor_zoo_v1',
        max_candidates=4,
        top_k=1,
        min_train_months=36,
        rerank_top_n=0,
        selection_split_mode='trailing_holdout',
        selection_val_months=24,
        split_test_start_override='2008-01-01',
        split_end_date_override='2009-12-31',
        split_train_pool_end_override='2007-12-31',
    )
    selected = yaml.safe_load(artifacts.selected_yaml.read_text(encoding='utf-8'))
    assert selected['selection_split']['final_test_start'] == '2008-01-01'
    assert selected['selection_split']['final_test_end'] == '2009-12-31'
    manifest_payload = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    assert manifest_payload['selection_split']['final_test_start'] == '2008-01-01'
    cfg_payload = yaml.safe_load(Path(manifest_payload['entries'][0]['config_yaml']).read_text(encoding='utf-8'))
    assert cfg_payload['split']['test_start'] == '2008-01-01'
    assert cfg_payload['split']['end_date'] == '2009-12-31'


def test_native_factor_zoo_external_stage1_root_is_audit_only(tmp_path: Path):
    base_dir = tmp_path / 'base_bundle_audit'
    _build_synthetic_base_bundle(base_dir)
    fake_v1_root = _build_fake_stage1_v1_root(tmp_path / 'fake_v1_root')

    base_artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite_no_audit',
        candidate_zoo='factor_zoo_v1',
        max_candidates=3,
        top_k=1,
        min_train_months=60,
        rolling_window=36,
        rerank_top_n=0,
        selection_split_mode='trailing_holdout',
        selection_val_months=60,
        select_rolling_oos_window=False,
    )
    audit_artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite_with_audit',
        candidate_zoo='factor_zoo_v1',
        max_candidates=3,
        top_k=1,
        min_train_months=60,
        rolling_window=36,
        rerank_top_n=0,
        selection_split_mode='trailing_holdout',
        selection_val_months=60,
        select_rolling_oos_window=False,
        legacy_stage1_v1_root=fake_v1_root,
    )

    base_selected = yaml.safe_load(base_artifacts.selected_yaml.read_text(encoding='utf-8'))
    audit_selected = yaml.safe_load(audit_artifacts.selected_yaml.read_text(encoding='utf-8'))
    assert audit_selected['stage1_engine'] == 'ported_legacy_v2'
    assert audit_selected['stage1_external_audit_enabled'] is True
    assert audit_selected['selected_specs'] == base_selected['selected_specs']

    summary = pd.read_csv(audit_artifacts.selection_summary_csv)
    assert set(summary['stage1_engine'].dropna()) == {'ported_legacy_v2'}
    assert 'stage1_external_audit_blocks' in summary.columns
    assert int(summary['stage1_external_audit_blocks'].fillna(0).max()) >= 1

    audit_csv = Path(audit_selected['selection_stage1_audit_csv'])
    assert audit_csv.exists()
    audit_df = pd.read_csv(audit_csv)
    assert audit_df['audit_compared'].fillna(False).any()
    assert audit_df['r2_oos_ret_mean_external'].max() > 100.0
    assert audit_df['abs_delta_r2_oos_ret_mean'].max() > 100.0


def test_native_factor_zoo_without_external_stage1_root_skips_audit_csv(tmp_path: Path):
    base_dir = tmp_path / 'base_bundle_no_audit'
    _build_synthetic_base_bundle(base_dir)

    artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite_no_external_root',
        candidate_zoo='factor_zoo_v1',
        max_candidates=3,
        top_k=1,
        min_train_months=60,
        rolling_window=36,
        rerank_top_n=0,
        selection_split_mode='trailing_holdout',
        selection_val_months=60,
        select_rolling_oos_window=False,
    )
    selected = yaml.safe_load(artifacts.selected_yaml.read_text(encoding='utf-8'))
    assert selected['stage1_engine'] == 'ported_legacy_v2'
    assert selected['selection_stage1_audit_csv'] is None
    manifest = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    assert manifest['selection_stage1_audit_csv'] is None
    assert manifest['stage1_external_audit_enabled'] is False


def test_native_factor_zoo_rerank_accepts_adcc(tmp_path: Path):
    base_dir = tmp_path / 'base_bundle_adcc'
    _build_synthetic_base_bundle(base_dir)

    artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite_rerank_adcc',
        candidate_zoo='factor_zoo_v1',
        max_candidates=2,
        top_k=1,
        min_train_months=60,
        rolling_window=36,
        rerank_top_n=1,
        selection_split_mode='trailing_holdout',
        selection_val_months=60,
        select_rolling_oos_window=False,
        rerank_covariance_models=['adcc'],
        ppgdpo_lite_epochs=2,
        ppgdpo_lite_mc_rollouts=1,
        ppgdpo_lite_mc_sub_batch=1,
        selection_device='cpu',
    )
    selected = yaml.safe_load(artifacts.selected_yaml.read_text(encoding='utf-8'))
    assert selected['ppgdpo_lite']['rerank_covariance_models'] == ['adcc']
    stage2 = pd.read_csv(selected['selection_stage2_csv'])
    assert set(stage2['covariance_model_label']) == {'adcc'}
    manifest_payload = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    cfg_payload = yaml.safe_load(Path(manifest_payload['entries'][0]['config_yaml']).read_text(encoding='utf-8'))
    assert cfg_payload['covariance_model']['kind'] == 'asset_adcc'
    assert cfg_payload['covariance_model']['adcc_gamma'] == pytest.approx(0.005)



def test_native_factor_zoo_rerank_accepts_regime_dcc(tmp_path: Path):
    base_dir = tmp_path / 'base_bundle_regime_dcc'
    _build_synthetic_base_bundle(base_dir)

    artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite_rerank_regime_dcc',
        candidate_zoo='factor_zoo_v1',
        max_candidates=2,
        top_k=1,
        min_train_months=60,
        rolling_window=36,
        rerank_top_n=1,
        selection_split_mode='trailing_holdout',
        selection_val_months=60,
        select_rolling_oos_window=False,
        rerank_covariance_models=['regime_dcc'],
        ppgdpo_lite_epochs=2,
        ppgdpo_lite_mc_rollouts=1,
        ppgdpo_lite_mc_sub_batch=1,
        selection_device='cpu',
    )
    selected = yaml.safe_load(artifacts.selected_yaml.read_text(encoding='utf-8'))
    assert selected['ppgdpo_lite']['rerank_covariance_models'] == ['regime_dcc']
    stage2 = pd.read_csv(selected['selection_stage2_csv'])
    assert set(stage2['covariance_model_label']) == {'regime_dcc'}
    manifest_payload = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    cfg_payload = yaml.safe_load(Path(manifest_payload['entries'][0]['config_yaml']).read_text(encoding='utf-8'))
    assert cfg_payload['covariance_model']['kind'] == 'asset_regime_dcc'
    assert cfg_payload['covariance_model']['regime_threshold_quantile'] == pytest.approx(0.75)


def test_native_factor_zoo_selects_validation_rolling_window(tmp_path: Path):
    base_dir = tmp_path / 'base_bundle_roll_select'
    _build_synthetic_base_bundle(base_dir)

    artifacts = native_select_factor_suite(
        base_dir=base_dir,
        out_dir=tmp_path / 'suite_roll_select',
        candidate_zoo='factor_zoo_v1',
        max_candidates=3,
        top_k=1,
        min_train_months=36,
        rolling_window=24,
        rerank_top_n=0,
        selection_split_mode='trailing_holdout',
        selection_val_months=24,
        select_rolling_oos_window=True,
        rolling_oos_window_grid=[24, 36],
        ppgdpo_lite_epochs=1,
        ppgdpo_lite_mc_rollouts=1,
        ppgdpo_lite_mc_sub_batch=1,
        selection_device='cpu',
    )

    selected = yaml.safe_load(artifacts.selected_yaml.read_text(encoding='utf-8'))
    manifest_payload = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    summary = pd.read_csv(artifacts.selection_summary_csv)

    assert selected['validation_protocol_selection']['enabled'] is True
    assert selected['oos_protocols_default'] == ['fixed', 'expanding_annual', 'rolling_selected_annual']
    assert manifest_payload['oos_protocols_default'] == ['fixed', 'expanding_annual', 'rolling_selected_annual']
    assert Path(selected['selection_protocol_validation_summary_csv']).exists()
    assert Path(selected['selection_protocol_validation_blocks_csv']).exists()
    assert 'selected_rolling_train_months' in summary.columns

    entry = manifest_payload['entries'][0]
    assert entry['selected_rolling_train_months'] in {24, 36}
    assert entry['selected_oos_protocols']['rolling_selected_annual']['rolling_train_months'] in {24, 36}

    metadata = yaml.safe_load(Path(entry['metadata_yaml']).read_text(encoding='utf-8'))
    assert metadata['selected_rolling_train_months'] in {24, 36}
    assert metadata['selected_oos_protocols']['rolling_selected_annual']['rolling_train_months'] in {24, 36}
