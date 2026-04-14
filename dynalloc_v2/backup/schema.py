from __future__ import annotations

from pathlib import Path
from typing import Literal
from datetime import date
from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    name: str
    output_dir: Path


class SyntheticDataConfig(BaseModel):
    periods: int = 240
    assets: int = 12
    factors: int = 3
    seed: int = 17


class DataConfig(BaseModel):
    mode: Literal['synthetic', 'csv'] = 'csv'
    returns_csv: Path | None = None
    states_csv: Path | None = None
    factors_csv: Path | None = None
    date_col: str = 'date'
    synthetic: SyntheticDataConfig | None = None

    @model_validator(mode='after')
    def _validate_mode(self):
        if self.mode == 'csv' and self.returns_csv is None:
            raise ValueError('returns_csv is required when data.mode == "csv"')
        if self.mode == 'synthetic' and self.synthetic is None:
            self.synthetic = SyntheticDataConfig()
        return self


class SplitConfig(BaseModel):
    train_start: date
    test_start: date
    end_date: date
    fixed_train_end: date | None = None
    refit_every: int = 1
    min_train_months: int = 60
    train_window_mode: Literal['fixed', 'expanding', 'rolling'] = 'fixed'
    rolling_train_months: int | None = None
    rebalance_every: int = 1
    protocol_label: str | None = None


class StateConfig(BaseModel):
    source: Literal['states'] = 'states'
    columns: list[str] = Field(default_factory=list)


class FactorModelConfig(BaseModel):
    extractor: Literal['provided', 'pca'] = 'provided'
    n_factors: int = 3
    provided_factor_columns: list[str] = Field(default_factory=list)


class MeanModelConfig(BaseModel):
    kind: Literal['direct_assets', 'factor_apt', 'factor_apt_regime'] = 'factor_apt'
    ridge_lambda: float = 1.0e-6
    regime_threshold_quantile: float = 0.75
    regime_sharpness: float = 8.0


class CovarianceModelConfig(BaseModel):
    kind: Literal['constant', 'state_diagonal', 'state_only_diagonal', 'asset_dcc', 'asset_adcc', 'asset_regime_dcc'] = 'asset_dcc'
    ridge_lambda: float = 1.0e-6
    variance_floor: float = 1.0e-6
    correlation_shrink: float = 0.10
    factor_correlation_mode: Literal['sample_shrunk', 'independent'] = 'independent'
    use_persistence: bool = False
    dcc_alpha: float = 0.02
    dcc_beta: float = 0.97
    adcc_gamma: float = 0.005
    variance_lambda: float = 0.97
    asset_covariance_shrink: float = 0.10
    regime_threshold_quantile: float = 0.75
    regime_smoothing: float = 0.90
    regime_sharpness: float = 8.0


class PolicyConfig(BaseModel):
    risk_aversion: float = 8.0
    risky_cap: float = 1.0
    cash_floor: float = 0.0
    long_only: bool = True
    pgd_steps: int = 150
    step_size: float = 0.05
    turnover_penalty: float = 0.05


class ComparisonConfig(BaseModel):
    cov_modes: list[Literal['diag', 'full']] = Field(default_factory=lambda: ['diag', 'full'])
    cross_modes: list[Literal['estimated', 'zero', 'regime_gated']] = Field(default_factory=lambda: ['estimated', 'zero', 'regime_gated'])
    transaction_cost_bps: float = 0.0
    include_standard_benchmarks: bool = True
    standard_benchmarks: list[Literal['equal_weight', 'market', 'min_variance', 'risk_parity']] = Field(
        default_factory=lambda: ['equal_weight', 'min_variance', 'risk_parity']
    )
    reference_cross_mode_label: str = 'reference'
    benchmark_cross_mode_label: str = 'benchmark'
    market_factor_candidates: list[str] = Field(
        default_factory=lambda: ['MKT', 'Mkt-RF', 'Mkt_RF', 'MKT_RF', 'MARKET', 'market', 'market_excess']
    )


class ExperimentConfig(BaseModel):
    kind: Literal['factorcov', 'ppgdpo'] = 'factorcov'


class PPGDPOConfig(BaseModel):
    device: str = 'cpu'
    hidden_dim: int = 32
    hidden_layers: int = 2
    epochs: int = 120
    lr: float = 1.0e-3
    utility: Literal['log', 'crra'] = 'crra'
    batch_size: int = 64
    horizon_steps: int = 12
    kappa: float = 1.0
    mc_rollouts: int = 256
    mc_sub_batch: int = 256
    clamp_min_return: float = -0.95
    clamp_port_ret_max: float = 5.0
    clamp_wealth_min: float = 1.0e-8
    clamp_state_std_abs: float | None = 8.0
    covariance_mode: Literal['diag', 'full'] = 'full'
    cross_strength: float = 1.0  # treated as Cross scaling in stage34 projection
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


class PIPINNConfig(BaseModel):
    device: str = 'auto'
    dtype: Literal['float32', 'float64'] = 'float64'
    outer_iters: int = 6
    eval_epochs: int = 120
    n_train_int: int = 4096
    n_train_bc: int = 1024
    n_val_int: int = 2048
    n_val_bc: int = 512
    p_uniform: float = 0.30
    p_emp: float = 0.70
    p_tau_near0: float = 0.30
    lr: float = 5.0e-4
    grad_clip: float = 1.0
    w_bc: float = 20.0
    w_bc_dx: float = 5.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 1.0e-5
    width: int = 96
    depth: int = 4
    covariance_train_mode: Literal['dcc_current', 'cross_resid'] = 'dcc_current'
    x_domain_quantile_low: float = 0.001
    x_domain_quantile_high: float = 0.999
    x_domain_buffer: float = 0.20
    emit_frozen_traincov_strategy: bool = False
    save_training_logs: bool = True
    show_progress: bool = True
    show_epoch_progress: bool = False



class Config(BaseModel):
    optimizer_backend: Literal['ppgdpo', 'pipinn'] = 'ppgdpo'
    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    state: StateConfig
    factor_model: FactorModelConfig
    mean_model: MeanModelConfig = Field(default_factory=MeanModelConfig)
    covariance_model: CovarianceModelConfig = Field(default_factory=CovarianceModelConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    ppgdpo: PPGDPOConfig = Field(default_factory=PPGDPOConfig)
    pipinn: PIPINNConfig = Field(default_factory=PIPINNConfig)
