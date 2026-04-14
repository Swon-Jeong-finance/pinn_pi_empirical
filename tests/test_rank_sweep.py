from __future__ import annotations

from pathlib import Path
import pandas as pd
from datetime import date
import yaml

from dynalloc_v2.rank_sweep import run_rank_sweep
from dynalloc_v2.cli import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_run_rank_sweep_smoke(tmp_path: Path):
    base_cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    base_cfg.data.synthetic.periods = 60
    base_cfg.split.train_start = date(2000, 1, 31)
    base_cfg.split.test_start = date(2003, 1, 31)
    base_cfg.split.end_date = date(2004, 12, 31)
    base_cfg.split.min_train_months = 24
    base_cfg.split.refit_every = 24
    base_cfg.ppgdpo.epochs = 2
    base_cfg.ppgdpo.hidden_dim = 8
    base_cfg.ppgdpo.batch_size = 8
    base_cfg.ppgdpo.horizon_steps = 2
    base_cfg.ppgdpo.mc_rollouts = 1
    base_cfg.ppgdpo.mc_sub_batch = 1
    base_cfg.policy.pgd_steps = 20

    suite_dir = tmp_path / 'suite'
    entries = []
    for rank in [1, 2]:
        cfg = base_cfg.model_copy(deep=True)
        cfg.project.name = f'demo_rank_{rank}'
        cfg.project.output_dir = tmp_path / f'run_rank_{rank}'
        rank_dir = suite_dir / f'rank_{rank:03d}'
        rank_dir.mkdir(parents=True)
        cfg_path = rank_dir / 'config_empirical_ppgdpo_apt.yaml'
        cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode='json'), sort_keys=False), encoding='utf-8')
        metadata_path = rank_dir / 'bridge_metadata.yaml'
        metadata_path.write_text(yaml.safe_dump({'rank': rank}, sort_keys=False), encoding='utf-8')
        entries.append({'rank': rank, 'spec': f'spec_{rank}', 'config_yaml': str(cfg_path), 'metadata_yaml': str(metadata_path)})

    manifest_path = suite_dir / 'suite_manifest.yaml'
    manifest_path.write_text(yaml.safe_dump({'suite_name': 'demo_suite', 'entries': entries}, sort_keys=False), encoding='utf-8')

    artifacts = run_rank_sweep(manifest_path)
    assert artifacts.zero_cost_summary.exists()
    assert artifacts.all_costs_summary.exists()
    assert artifacts.results_csv.exists()
    assert (suite_dir / 'rank_sweep' / 'rank_001' / 'protocols' / 'fixed' / 'comparison' / 'comparison_cross_modes_zero_cost_summary.csv').exists()
    assert (suite_dir / 'rank_sweep' / 'rank_002' / 'protocols' / 'fixed' / 'comparison' / 'comparison_cross_modes_zero_cost_summary.csv').exists()
    assert (suite_dir / 'rank_sweep' / 'rank_001' / 'protocols' / 'expanding_annual' / 'comparison' / 'comparison_cross_modes_zero_cost_summary.csv').exists()
    assert (suite_dir / 'rank_sweep' / 'rank_001' / 'protocols' / 'rolling20y_annual' / 'comparison' / 'comparison_cross_modes_zero_cost_summary.csv').exists()
    assert not (suite_dir / 'rank_sweep' / 'rank_001' / 'comparison' / 'comparison_cross_modes_zero_cost_summary.csv').exists()
    report = yaml.safe_load((suite_dir / 'rank_sweep' / 'stage21_rank_sweep_report.yaml').read_text(encoding='utf-8'))
    assert report['oos_protocols'] == ['fixed', 'expanding_annual', 'rolling20y_annual']
    assert report['emit_legacy_fixed_layout'] is False
    agg = pd.read_csv(artifacts.results_csv)
    assert set(agg['oos_protocol']) == {'fixed', 'expanding_annual', 'rolling20y_annual'}
    by_protocol = agg.groupby('oos_protocol')['rebalance_every'].unique().to_dict()
    assert all(list(v) == [1] for v in by_protocol.values())
    annual_refits = agg[agg['oos_protocol'].isin(['expanding_annual', 'rolling20y_annual'])]
    assert set(annual_refits['refit_every']) == {12}
    assert set(annual_refits.loc[annual_refits['oos_protocol'] == 'rolling20y_annual', 'rolling_train_months']) == {240.0}



def test_run_rank_sweep_with_device_override(tmp_path: Path):
    base_cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    base_cfg.data.synthetic.periods = 48
    base_cfg.split.train_start = date(2000, 1, 31)
    base_cfg.split.test_start = date(2002, 1, 31)
    base_cfg.split.end_date = date(2003, 12, 31)
    base_cfg.split.min_train_months = 24
    base_cfg.split.refit_every = 24
    base_cfg.ppgdpo.epochs = 1
    base_cfg.ppgdpo.hidden_dim = 8
    base_cfg.ppgdpo.batch_size = 8
    base_cfg.ppgdpo.horizon_steps = 2
    base_cfg.ppgdpo.mc_rollouts = 1
    base_cfg.ppgdpo.mc_sub_batch = 1
    base_cfg.policy.pgd_steps = 10

    suite_dir = tmp_path / 'suite'
    entries = []
    cfg = base_cfg.model_copy(deep=True)
    cfg.project.name = 'demo_rank_1'
    cfg.project.output_dir = tmp_path / 'run_rank_1'
    rank_dir = suite_dir / 'rank_001'
    rank_dir.mkdir(parents=True)
    cfg_path = rank_dir / 'config_empirical_ppgdpo_apt.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode='json'), sort_keys=False), encoding='utf-8')
    metadata_path = rank_dir / 'bridge_metadata.yaml'
    metadata_path.write_text(yaml.safe_dump({'rank': 1}, sort_keys=False), encoding='utf-8')
    entries.append({'rank': 1, 'spec': 'spec_1', 'config_yaml': str(cfg_path), 'metadata_yaml': str(metadata_path)})
    manifest_path = suite_dir / 'suite_manifest.yaml'
    manifest_path.write_text(yaml.safe_dump({'suite_name': 'demo_suite', 'entries': entries}, sort_keys=False), encoding='utf-8')
    artifacts = run_rank_sweep(manifest_path, device_override='cpu', oos_protocols=['fixed'])
    assert artifacts.results_csv.exists()




def test_run_rank_sweep_can_emit_legacy_fixed_layout(tmp_path: Path):
    base_cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    base_cfg.data.synthetic.periods = 48
    base_cfg.split.train_start = date(2000, 1, 31)
    base_cfg.split.test_start = date(2002, 1, 31)
    base_cfg.split.end_date = date(2003, 12, 31)
    base_cfg.split.min_train_months = 24
    base_cfg.split.refit_every = 24
    base_cfg.ppgdpo.epochs = 1
    base_cfg.ppgdpo.hidden_dim = 8
    base_cfg.ppgdpo.batch_size = 8
    base_cfg.ppgdpo.horizon_steps = 2
    base_cfg.ppgdpo.mc_rollouts = 1
    base_cfg.ppgdpo.mc_sub_batch = 1
    base_cfg.policy.pgd_steps = 10

    suite_dir = tmp_path / 'suite_legacy_layout'
    cfg = base_cfg.model_copy(deep=True)
    cfg.project.name = 'demo_rank_1'
    cfg.project.output_dir = tmp_path / 'run_rank_legacy_layout'
    rank_dir = suite_dir / 'rank_001'
    rank_dir.mkdir(parents=True)
    cfg_path = rank_dir / 'config_empirical_ppgdpo_apt.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode='json'), sort_keys=False), encoding='utf-8')
    metadata_path = rank_dir / 'bridge_metadata.yaml'
    metadata_path.write_text(yaml.safe_dump({'rank': 1}, sort_keys=False), encoding='utf-8')
    manifest_path = suite_dir / 'suite_manifest.yaml'
    manifest_path.write_text(yaml.safe_dump({'suite_name': 'demo_suite', 'entries': [{'rank': 1, 'spec': 'spec_1', 'config_yaml': str(cfg_path), 'metadata_yaml': str(metadata_path)}]}, sort_keys=False), encoding='utf-8')

    artifacts = run_rank_sweep(manifest_path, oos_protocols=['fixed'], emit_legacy_fixed_layout=True)
    assert artifacts.results_csv.exists()
    assert (suite_dir / 'rank_sweep' / 'rank_001' / 'comparison' / 'comparison_cross_modes_zero_cost_summary.csv').exists()
    report = yaml.safe_load((suite_dir / 'rank_sweep' / 'stage21_rank_sweep_report.yaml').read_text(encoding='utf-8'))
    assert report['emit_legacy_fixed_layout'] is True


def test_run_rank_sweep_legacy_manifest_defaults_upgrade(tmp_path: Path):
    base_cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    base_cfg.data.synthetic.periods = 48
    base_cfg.split.train_start = date(2000, 1, 31)
    base_cfg.split.test_start = date(2002, 1, 31)
    base_cfg.split.end_date = date(2003, 12, 31)
    base_cfg.split.min_train_months = 24
    base_cfg.split.refit_every = 24
    base_cfg.ppgdpo.epochs = 1
    base_cfg.ppgdpo.hidden_dim = 8
    base_cfg.ppgdpo.batch_size = 8
    base_cfg.ppgdpo.horizon_steps = 2
    base_cfg.ppgdpo.mc_rollouts = 1
    base_cfg.ppgdpo.mc_sub_batch = 1
    base_cfg.policy.pgd_steps = 10

    suite_dir = tmp_path / 'suite_legacy'
    cfg = base_cfg.model_copy(deep=True)
    cfg.project.name = 'demo_rank_1'
    cfg.project.output_dir = tmp_path / 'run_rank_legacy'
    rank_dir = suite_dir / 'rank_001'
    rank_dir.mkdir(parents=True)
    cfg_path = rank_dir / 'config_empirical_ppgdpo_apt.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode='json'), sort_keys=False), encoding='utf-8')
    metadata_path = rank_dir / 'bridge_metadata.yaml'
    metadata_path.write_text(yaml.safe_dump({'rank': 1}, sort_keys=False), encoding='utf-8')
    manifest_path = suite_dir / 'suite_manifest.yaml'
    manifest_path.write_text(yaml.safe_dump({
        'suite_name': 'legacy_suite',
        'oos_protocols_default': ['fixed', 'expanding_annual'],
        'entries': [{'rank': 1, 'spec': 'spec_1', 'config_yaml': str(cfg_path), 'metadata_yaml': str(metadata_path)}],
    }, sort_keys=False), encoding='utf-8')

    artifacts = run_rank_sweep(manifest_path)
    report = yaml.safe_load((suite_dir / 'rank_sweep' / 'stage21_rank_sweep_report.yaml').read_text(encoding='utf-8'))
    assert report['oos_protocols'] == ['fixed', 'expanding_annual', 'rolling20y_annual']
    agg = pd.read_csv(artifacts.results_csv)
    assert set(agg['oos_protocol']) == {'fixed', 'expanding_annual', 'rolling20y_annual'}
    by_protocol = agg.groupby('oos_protocol')['rebalance_every'].unique().to_dict()
    assert all(list(v) == [1] for v in by_protocol.values())
    annual_refits = agg[agg['oos_protocol'].isin(['expanding_annual', 'rolling20y_annual'])]
    assert set(annual_refits['refit_every']) == {12}
    assert set(annual_refits.loc[annual_refits['oos_protocol'] == 'rolling20y_annual', 'rolling_train_months']) == {240.0}


def test_run_rank_sweep_resolves_selected_rolling_protocol(tmp_path: Path):
    base_cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    base_cfg.data.synthetic.periods = 48
    base_cfg.split.train_start = date(2000, 1, 31)
    base_cfg.split.test_start = date(2002, 1, 31)
    base_cfg.split.end_date = date(2003, 12, 31)
    base_cfg.split.min_train_months = 24
    base_cfg.split.refit_every = 24
    base_cfg.ppgdpo.epochs = 1
    base_cfg.ppgdpo.hidden_dim = 8
    base_cfg.ppgdpo.batch_size = 8
    base_cfg.ppgdpo.horizon_steps = 2
    base_cfg.ppgdpo.mc_rollouts = 1
    base_cfg.ppgdpo.mc_sub_batch = 1
    base_cfg.policy.pgd_steps = 10

    suite_dir = tmp_path / 'suite_selected_rolling'
    cfg = base_cfg.model_copy(deep=True)
    cfg.project.name = 'demo_rank_selected_rolling'
    cfg.project.output_dir = tmp_path / 'run_rank_selected_rolling'
    rank_dir = suite_dir / 'rank_001'
    rank_dir.mkdir(parents=True)
    cfg_path = rank_dir / 'config_empirical_ppgdpo_apt.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode='json'), sort_keys=False), encoding='utf-8')
    metadata_path = rank_dir / 'bridge_metadata.yaml'
    metadata_path.write_text(yaml.safe_dump({'rank': 1}, sort_keys=False), encoding='utf-8')
    manifest_path = suite_dir / 'suite_manifest.yaml'
    manifest_path.write_text(yaml.safe_dump({
        'suite_name': 'selected_rolling_suite',
        'oos_protocols_default': ['fixed', 'expanding_annual', 'rolling_selected_annual'],
        'validation_protocol_selection': {
            'default_rolling_train_months': 84,
            'fallback_rolling_train_months': 120,
        },
        'entries': [{
            'rank': 1,
            'model_id': 'spec_1__const',
            'spec': 'spec_1',
            'config_yaml': str(cfg_path),
            'metadata_yaml': str(metadata_path),
            'selected_rolling_train_months': 84,
            'selected_oos_protocols': {
                'rolling_selected_annual': {
                    'rolling_train_months': 84,
                }
            },
        }],
    }, sort_keys=False), encoding='utf-8')

    artifacts = run_rank_sweep(manifest_path)
    agg = pd.read_csv(artifacts.results_csv)
    assert set(agg['oos_protocol']) == {'fixed', 'expanding_annual', 'rolling_selected_annual'}
    selected = agg.loc[agg['oos_protocol'] == 'rolling_selected_annual', 'rolling_train_months']
    assert set(selected.dropna()) == {84.0}
