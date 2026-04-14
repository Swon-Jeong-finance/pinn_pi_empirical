from __future__ import annotations

from pathlib import Path
import pandas as pd

from dynalloc_v2.cli import load_config
from dynalloc_v2.experiments import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_demo_factorcov_smoke(tmp_path: Path):
    cfg = load_config(ROOT / 'configs' / 'demo_synthetic.yaml')
    cfg.project.output_dir = tmp_path / 'demo_factorcov'
    artifacts = run_experiment(cfg)
    assert artifacts.summary_zero_cost.exists()
    summary = pd.read_csv(artifacts.summary_zero_cost)
    assert {'strategy', 'cov_mode', 'ann_ret', 'cer_ann'}.issubset(summary.columns)
    assert set(summary['cov_mode']) == {'diag', 'full'}


def test_demo_ppgdpo_smoke(tmp_path: Path):
    cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    cfg.project.output_dir = tmp_path / 'demo_ppgdpo'
    cfg.data.synthetic.periods = 48
    cfg.split.train_start = pd.Timestamp('2000-01-31').date()
    cfg.split.test_start = pd.Timestamp('2002-01-31').date()
    cfg.split.end_date = pd.Timestamp('2003-12-31').date()
    cfg.split.min_train_months = 24
    cfg.ppgdpo.epochs = 6
    cfg.ppgdpo.hidden_dim = 16
    cfg.ppgdpo.batch_size = 16
    cfg.ppgdpo.horizon_steps = 2
    cfg.ppgdpo.mc_rollouts = 1
    cfg.ppgdpo.mc_sub_batch = 1
    cfg.split.refit_every = 24
    artifacts = run_experiment(cfg)
    assert artifacts.summary_zero_cost.exists()
    summary = pd.read_csv(artifacts.summary_zero_cost)
    assert {'strategy', 'strategy_display', 'comparison_role', 'cross_mode', 'ann_ret', 'cer_ann'}.issubset(summary.columns)
    assert {'estimated', 'zero', 'reference', 'benchmark'}.issubset(set(summary['cross_mode']))
    assert {'predictive_static', 'pgdpo', 'ppgdpo', 'equal_weight', 'min_variance', 'risk_parity'}.issubset(set(summary['strategy']))
    assert 'market' not in set(summary['strategy'])
    assert artifacts.benchmark_notes_yaml is not None and artifacts.benchmark_notes_yaml.exists()


def test_csv_template_validation():
    cfg = load_config(ROOT / 'configs' / 'csv_template_provided.yaml')
    assert cfg.factor_model.extractor == 'provided'
    assert cfg.mean_model.kind == 'factor_apt'


def test_csv_ppgdpo_template_validation():
    cfg = load_config(ROOT / 'configs' / 'csv_template_ppgdpo.yaml')
    assert cfg.experiment.kind == 'ppgdpo'
    assert cfg.factor_model.extractor == 'provided'
    assert cfg.mean_model.kind == 'factor_apt'


def test_csv_pca_template_validation():
    cfg = load_config(ROOT / 'configs' / 'csv_template_pca.yaml')
    assert cfg.factor_model.extractor == 'pca'
    assert cfg.mean_model.kind == 'factor_apt'


def test_demo_ppgdpo_adcc_smoke(tmp_path: Path):
    cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    cfg.project.output_dir = tmp_path / 'demo_ppgdpo_adcc'
    cfg.data.synthetic.periods = 48
    cfg.split.train_start = pd.Timestamp('2000-01-31').date()
    cfg.split.test_start = pd.Timestamp('2002-01-31').date()
    cfg.split.end_date = pd.Timestamp('2003-12-31').date()
    cfg.split.min_train_months = 24
    cfg.ppgdpo.epochs = 4
    cfg.ppgdpo.hidden_dim = 16
    cfg.ppgdpo.batch_size = 16
    cfg.ppgdpo.horizon_steps = 2
    cfg.ppgdpo.mc_rollouts = 1
    cfg.ppgdpo.mc_sub_batch = 1
    cfg.split.refit_every = 24
    cfg.covariance_model.kind = 'asset_adcc'
    cfg.covariance_model.adcc_gamma = 0.005
    artifacts = run_experiment(cfg)
    assert artifacts.summary_zero_cost.exists()
    summary = pd.read_csv(artifacts.summary_zero_cost)
    assert {'strategy', 'strategy_display', 'comparison_role', 'cross_mode', 'ann_ret', 'cer_ann'}.issubset(summary.columns)
    assert {'predictive_static', 'pgdpo', 'ppgdpo', 'equal_weight', 'min_variance', 'risk_parity'}.issubset(set(summary['strategy']))
    assert 'market' not in set(summary['strategy'])
    assert artifacts.benchmark_notes_yaml is not None and artifacts.benchmark_notes_yaml.exists()
