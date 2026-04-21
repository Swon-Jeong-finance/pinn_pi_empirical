
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


BOND_HOOK_CHOICES = ("none", "ust10y", "curve_core")
ASSET_UNIVERSE_ALIASES = {
    "fama_market": "ff_mkt",
    "ff1": "ff_mkt",
    "ff6": "ff6_szbm",
    "ff17": "ff17ind",
    "ff30": "ff30ind",
    "ff38": "ff38ind",
    "ff100": "ff100_szbm",
    "ff25": "ff25_szbm",
    "ff49": "ff49ind",
}


def _canonical_asset_universe(value: str) -> str:
    token = str(value).strip().lower()
    return ASSET_UNIVERSE_ALIASES.get(token, token)


def _bond_assets_for_hook(hook: str) -> list[BondAssetConfig]:
    hook = str(hook).strip().lower()
    if hook == "ust10y":
        return [
            BondAssetConfig(
                name="UST10Y",
                source="crsp_csv",
                csv="data/bond10y.csv",
                ret_col="bond10y_ret",
            )
        ]
    if hook == "curve_core":
        return [
            BondAssetConfig(
                name="UST2Y",
                source="crsp_csv",
                csv="data/bond2y.csv",
                ret_col="ret2y",
            ),
            BondAssetConfig(
                name="UST5Y",
                source="crsp_csv",
                csv="data/bond5y.csv",
                ret_col="ret5y",
            ),
            BondAssetConfig(
                name="UST10Y",
                source="crsp_csv",
                csv="data/bond10y.csv",
                ret_col="bond10y_ret",
            ),
        ]
    if hook == "none":
        return []
    raise ValueError(f"Unknown bond_hook={hook!r}. Choose from {list(BOND_HOOK_CHOICES)}.")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentMeta(StrictModel):
    name: str
    output_dir: str = "outputs/default"
    notes: str | None = None


class BondAssetConfig(StrictModel):
    name: str
    source: Literal["crsp_csv", "fred_tr"] = "crsp_csv"
    csv: str | None = None
    ret_col: str | None = None
    date_col: str = "date"
    series_id: str | None = None

    @model_validator(mode="after")
    def validate_bond_asset(self) -> "BondAssetConfig":
        if self.source == "crsp_csv":
            if not self.csv:
                raise ValueError("Bond CSV path is required when source='crsp_csv'.")
            if not self.ret_col:
                raise ValueError("Bond ret_col is required when source='crsp_csv'.")
        elif self.source == "fred_tr":
            if not self.series_id:
                raise ValueError("Bond series_id is required when source='fred_tr'.")
        return self


class CustomRiskyAssetConfig(StrictModel):
    name: str
    source: Literal["market_total_return", "equity_universe_column", "csv"] = "market_total_return"
    base_universe: Literal["ff6_szbm", "ff17ind", "ff25_szbm", "ff30ind", "ff38ind", "ff49ind", "ff100_szbm"] | None = None
    column: str | None = None
    csv: str | None = None
    ret_col: str | None = None
    date_col: str = "date"

    @model_validator(mode="after")
    def validate_custom_risky_asset(self) -> "CustomRiskyAssetConfig":
        if self.source == "market_total_return":
            return self
        if self.source == "equity_universe_column":
            if not self.base_universe:
                raise ValueError("base_universe is required when source='equity_universe_column'.")
            if not self.column:
                raise ValueError("column is required when source='equity_universe_column'.")
            return self
        if not self.csv:
            raise ValueError("csv is required when source='csv'.")
        if not self.ret_col:
            raise ValueError("ret_col is required when source='csv'.")
        return self


class UniverseConfig(StrictModel):
    asset_universe: str = "ff25_szbm"
    bond_hook: Literal["none", "ust10y", "curve_core"] | None = None
    include_bond: bool = True
    bond_source: str = "crsp_csv"
    bond_csv: str = "data/bond10y.csv"
    bond_name: str = "UST10Y"
    bond_series_id: str = "BAMLCC0A0CMTRIV"
    bond_ret_col: str = "bond10y_ret"
    bond_assets: list[BondAssetConfig] = Field(default_factory=list)
    custom_risky_assets: list[CustomRiskyAssetConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_universe(self) -> "UniverseConfig":
        self.asset_universe = _canonical_asset_universe(self.asset_universe)
        if self.bond_hook is not None:
            hook = str(self.bond_hook).strip().lower()
            self.bond_hook = hook
            if hook == "none":
                self.include_bond = False
                self.bond_assets = []
            else:
                self.include_bond = True
                hooked_assets = _bond_assets_for_hook(hook)
                self.bond_assets = hooked_assets
                first = hooked_assets[0]
                self.bond_source = str(first.source)
                self.bond_name = str(first.name)
                self.bond_csv = str(first.csv) if first.csv else self.bond_csv
                self.bond_ret_col = str(first.ret_col) if first.ret_col else self.bond_ret_col
                if first.series_id:
                    self.bond_series_id = str(first.series_id)
        if self.asset_universe == "custom":
            risky_names = [a.name for a in self.custom_risky_assets]
            if len(risky_names) != len(set(risky_names)):
                raise ValueError("custom risky asset names must be unique.")
            if not self.include_bond and len(self.custom_risky_assets) == 0:
                raise ValueError(
                    "asset_universe='custom' requires at least one custom risky asset or include_bond=True."
                )
        elif self.custom_risky_assets:
            raise ValueError("custom_risky_assets are only supported when asset_universe='custom'.")
        if not self.include_bond:
            return self
        assets = self.normalized_bond_assets
        if len(assets) == 0:
            raise ValueError("include_bond=True but no bond asset could be resolved.")
        names = [a.name for a in assets]
        if len(names) != len(set(names)):
            raise ValueError("bond asset names must be unique.")
        custom_names = {a.name for a in self.custom_risky_assets}
        overlap = custom_names.intersection(names)
        if overlap:
            raise ValueError(f"custom risky asset names collide with bond names: {sorted(overlap)}")
        sources = {a.source for a in assets}
        if len(sources) != 1:
            raise ValueError(
                "All bond assets must use the same source within one experiment (all crsp_csv or all fred_tr)."
            )
        return self

    @property
    def normalized_bond_assets(self) -> list[BondAssetConfig]:
        if not self.include_bond:
            return []
        if self.bond_assets:
            return list(self.bond_assets)
        return [
            BondAssetConfig(
                name=self.bond_name,
                source=("fred_tr" if str(self.bond_source) == "fred_tr" else "crsp_csv"),
                csv=self.bond_csv if str(self.bond_source) == "crsp_csv" else None,
                ret_col=self.bond_ret_col if str(self.bond_source) == "crsp_csv" else None,
                series_id=self.bond_series_id if str(self.bond_source) == "fred_tr" else None,
            )
        ]

    @property
    def bond_source_kind(self) -> str | None:
        assets = self.normalized_bond_assets
        if not assets:
            return None
        return assets[0].source

    @property
    def bond_count(self) -> int:
        return len(self.normalized_bond_assets)

    @property
    def custom_risky_asset_names(self) -> list[str]:
        return [a.name for a in self.custom_risky_assets]


class MacroConfig(StrictModel):
    pool: Literal["legacy7", "bond_curve_core", "bond_curve_extended", "custom"] = "legacy7"
    feature_ids: list[str] = Field(default_factory=list)
    macro3_columns: list[str] = Field(
        default_factory=lambda: ["infl_yoy", "term_spread", "default_spread"]
    )

    @model_validator(mode="after")
    def validate_macro(self) -> "MacroConfig":
        if len(self.macro3_columns) < 1:
            raise ValueError("macro3_columns must contain at least one column name.")
        if self.pool == "custom" and not self.feature_ids:
            raise ValueError("macro.feature_ids must be provided when macro.pool='custom'.")
        features = self.effective_feature_ids
        for col in self.macro3_columns:
            if col not in features:
                raise ValueError(
                    f"macro3 column '{col}' must be present in the macro feature pool."
                )
        return self

    @property
    def effective_feature_ids(self) -> list[str]:
        if self.feature_ids:
            return list(self.feature_ids)
        if self.pool == "legacy7":
            return [
                "infl_yoy",
                "term_spread",
                "default_spread",
                "indpro_yoy",
                "unrate_chg",
                "umcsent",
                "mprime",
            ]
        if self.pool == "bond_curve_core":
            return [
                "infl_yoy",
                "term_spread",
                "default_spread",
                "indpro_yoy",
                "unrate_chg",
                "short_rate",
                "gs2",
                "gs5",
                "gs10",
                "slope_10y_2y",
                "curvature_2_5_10",
                "umcsent",
            ]
        if self.pool == "custom":
            return list(self.feature_ids)
        return [
            "infl_yoy",
            "term_spread",
            "default_spread",
            "core_infl_yoy",
            "indpro_yoy",
            "unrate_chg",
            "short_rate",
            "mprime",
            "gs2",
            "gs5",
            "gs10",
            "gs20",
            "slope_10y_2y",
            "slope_20y_5y",
            "curvature_2_5_10",
            "curvature_5_10_20",
            "umcsent",
        ]


class ModelConfig(StrictModel):
    state_spec: str = "pca_only_k2"
    latent_k: int = 2
    pls_horizon: int = 12
    pls_smooth_span: int = 6
    cross_mode: Literal["estimated", "zero"] = "estimated"
    zero_cross_policy_proxy: Literal["full", "myopic"] = "full"

    @model_validator(mode="after")
    def validate_model(self) -> "ModelConfig":
        if self.latent_k < 1:
            raise ValueError("latent_k must be >= 1.")
        if self.pls_horizon < 1:
            raise ValueError("pls_horizon must be >= 1.")
        if self.pls_smooth_span < 0:
            raise ValueError("pls_smooth_span must be >= 0.")
        return self


class ConstraintConfig(StrictModel):
    enabled: bool = True
    risky_cap: float = 1.0
    cash_floor: float = 0.0
    per_asset_cap: float | None = None
    allow_short: bool = False
    short_floor: float = 0.0

    @model_validator(mode="after")
    def validate_constraints(self) -> "ConstraintConfig":
        if self.risky_cap <= 0:
            raise ValueError("risky_cap must be positive.")
        if self.per_asset_cap is not None and self.per_asset_cap <= 0:
            raise ValueError("per_asset_cap must be positive when provided.")
        if not self.allow_short and self.short_floor != 0.0:
            raise ValueError(
                "short_floor must stay at 0.0 when allow_short is False."
            )
        if self.allow_short and self.short_floor > 0.0:
            raise ValueError("short_floor must be <= 0.0 when shorting is enabled.")
        if self.cash_floor >= 1.0:
            raise ValueError("cash_floor must be < 1.0.")
        return self

    @property
    def effective_risky_cap(self) -> float:
        return min(self.risky_cap, 1.0 - self.cash_floor)

    @property
    def simplex_fast_path(self) -> bool:
        return self.enabled and (not self.allow_short) and (self.per_asset_cap is None)


class MethodConfig(StrictModel):
    train_policy: bool = True
    evaluate_ppgdpo: bool = True
    tc_sweep: bool = True
    tc_bps: list[float] = Field(default_factory=lambda: [0.0, 5.0, 10.0, 25.0, 50.0])


class SplitConfig(StrictModel):
    train_pool_start: str | None = None
    train_pool_end: str = "2004-12-31"
    final_test_start: str = "2005-01-01"
    end_date: str = "2024-12-31"
    selection_cv_folds: int = 3
    selection_min_train_months: int = 204
    selection_val_months: int = 0
    force_common_calendar: bool = True
    common_start_mode: Literal["macro3", "suite"] = "suite"

    @staticmethod
    def _parse_date(value: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date '{value}'.") from exc

    @staticmethod
    def _month_index(d: date) -> int:
        return d.year * 12 + d.month

    @property
    def train_pool_start_date(self) -> date | None:
        if self.train_pool_start is None:
            return None
        return self._parse_date(self.train_pool_start)

    @property
    def train_pool_end_date(self) -> date:
        return self._parse_date(self.train_pool_end)

    @property
    def final_test_start_date(self) -> date:
        return self._parse_date(self.final_test_start)

    @property
    def end_date_value(self) -> date:
        return self._parse_date(self.end_date)

    @property
    def final_test_months(self) -> int:
        start = self.final_test_start_date
        end = self.end_date_value
        return (end.year - start.year) * 12 + (end.month - start.month) + 1

    @property
    def final_test_years(self) -> int:
        months = self.final_test_months
        if months % 12 != 0:
            raise ValueError(
                "final_test_start..end_date does not span a whole number of years."
            )
        return months // 12

    @model_validator(mode="after")
    def validate_split(self) -> "SplitConfig":
        train_start = self.train_pool_start_date
        train_end = self.train_pool_end_date
        test_start = self.final_test_start_date
        end = self.end_date_value
        if self.selection_cv_folds < 1:
            raise ValueError("selection_cv_folds must be >= 1.")
        if self.selection_min_train_months < 202:
            raise ValueError("selection_min_train_months must be >= 202.")
        if self.selection_val_months < 0:
            raise ValueError("selection_val_months must be >= 0.")
        if train_start is not None and train_start > train_end:
            raise ValueError("train_pool_start must be on or before train_pool_end.")
        if end <= train_end:
            raise ValueError("end_date must be after train_pool_end.")
        if test_start <= train_end:
            raise ValueError("final_test_start must be after train_pool_end.")
        if self._month_index(test_start) - self._month_index(train_end) != 1:
            raise ValueError(
                "train_pool_end must be the month immediately preceding final_test_start."
            )
        months = self.final_test_months
        if months <= 0:
            raise ValueError("final test span must be positive.")
        if months % 12 != 0:
            raise ValueError(
                "final_test_start..end_date must span a whole number of years for the evaluation."
            )
        return self


class SelectionConfig(StrictModel):
    enabled: bool = True
    candidate_mode: Literal["manual", "grid"] = "manual"
    candidate_specs: list[str] = Field(default_factory=list)
    include_static_specs: bool = False
    include_ff_split_specs: bool = False
    include_alias_specs: bool = False
    include_eqbond_block_specs: Literal["auto", "never", "always"] = "auto"
    pca_prefixes: list[str] = Field(default_factory=list)
    pca_components: list[int] = Field(default_factory=list)
    pls_prefixes: list[str] = Field(default_factory=list)
    pls_horizons: list[int] = Field(default_factory=list)
    pls_components: list[int] = Field(default_factory=list)
    top_k: int = 1
    guarded_only: bool = False
    window_mode: Literal["rolling", "expanding"] = "rolling"
    rolling_window: int = 60
    return_baseline: Literal["train_mean", "expanding_mean"] = "expanding_mean"
    state_baseline: Literal["train_mean", "expanding_mean", "random_walk"] = "expanding_mean"
    return_floor: float = 0.0
    state_q10_floor: float = -0.05
    cross_warn: float = 0.95
    cross_fail: float = 0.98
    score_mode: Literal["method_shortlist", "legacy_value"] = "method_shortlist"
    score_ret_mean_weight: float = 0.45
    score_ret_q10_weight: float = 0.20
    score_state_mean_weight: float = 0.20
    score_state_q10_weight: float = 0.15
    output_subdir: str = "selection"

    @property
    def uses_expanding_diagnostics(self) -> bool:
        return self.window_mode == "expanding"

    @model_validator(mode="after")
    def validate_selection(self) -> "SelectionConfig":
        if self.top_k < 1:
            raise ValueError("selection.top_k must be >= 1.")
        if self.rolling_window < 1:
            raise ValueError("selection.rolling_window must be >= 1.")
        if self.candidate_mode not in {"manual", "grid"}:
            raise ValueError("selection.candidate_mode must be 'manual' or 'grid'.")
        if self.window_mode not in {"rolling", "expanding"}:
            raise ValueError("selection.window_mode must be 'rolling' or 'expanding'.")
        for field_name in ("pca_components", "pls_horizons", "pls_components"):
            values = getattr(self, field_name)
            if any(int(v) < 1 for v in values):
                raise ValueError(f"selection.{field_name} values must all be >= 1.")
        if self.include_eqbond_block_specs not in {"auto", "never", "always"}:
            raise ValueError(
                "selection.include_eqbond_block_specs must be 'auto', 'never', or 'always'."
            )
        return self


class ComparisonConfig(StrictModel):
    enabled: bool = True
    spec_source: Literal["selected", "fixed"] = "selected"
    selected_rank: int = 1
    use_selected_spec: bool | None = None
    fixed_spec: str | None = None
    cross_modes: list[Literal["estimated", "zero"]] = Field(default_factory=list)
    output_subdir: str = "comparison"

    @model_validator(mode="after")
    def validate_comparison(self) -> "ComparisonConfig":
        if self.use_selected_spec is not None:
            self.spec_source = "selected" if self.use_selected_spec else "fixed"
        self.use_selected_spec = self.spec_source == "selected"
        if self.selected_rank < 1:
            raise ValueError("comparison.selected_rank must be >= 1.")
        deduped: list[str] = []
        seen: set[str] = set()
        for mode in self.cross_modes:
            if mode in seen:
                continue
            deduped.append(mode)
            seen.add(mode)
        self.cross_modes = deduped
        return self


class RankSweepConfig(StrictModel):
    enabled: bool = False
    start_rank: int = 1
    end_rank: int | None = None
    max_ranks: int | None = None
    resume: bool = True
    stop_on_error: bool = False
    output_subdir: str = "rank_sweep"
    poll_seconds: float = 2.0
    lock_timeout_seconds: float = 600.0

    @model_validator(mode="after")
    def validate_rank_sweep(self) -> "RankSweepConfig":
        if self.start_rank < 1:
            raise ValueError("rank_sweep.start_rank must be >= 1.")
        if self.end_rank is not None and self.end_rank < self.start_rank:
            raise ValueError("rank_sweep.end_rank must be >= rank_sweep.start_rank.")
        if self.max_ranks is not None and self.max_ranks < 1:
            raise ValueError("rank_sweep.max_ranks must be >= 1 when provided.")
        if self.poll_seconds <= 0:
            raise ValueError("rank_sweep.poll_seconds must be positive.")
        if self.lock_timeout_seconds <= 0:
            raise ValueError("rank_sweep.lock_timeout_seconds must be positive.")
        return self


class TrainingConfig(StrictModel):
    iters: int = 800
    batch_size: int = 512
    lr: float = 3e-4
    gamma: float = 5.0
    residual_policy: bool = True
    residual_ridge: float = 1e-6
    residual_w_max: float | None = None
    residual_detach_baseline: bool = False
    ppgdpo_mc: int = 256
    ppgdpo_subbatch: int = 64
    ppgdpo_seed: int = 123
    ppgdpo_L: float | None = None
    ppgdpo_eps: float = 1e-6
    ppgdpo_ridge: float = 1e-10
    ppgdpo_tau: float = 0.95
    ppgdpo_armijo: float = 1e-4
    ppgdpo_backtrack: float = 0.5
    ppgdpo_newton: int = 30
    ppgdpo_tol_grad: float = 1e-8
    ppgdpo_ls: int = 20

    @model_validator(mode="after")
    def validate_training(self) -> "TrainingConfig":
        if self.iters < 0:
            raise ValueError("iters must be >= 0.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if self.ppgdpo_mc < 1:
            raise ValueError("ppgdpo_mc must be >= 1.")
        if self.ppgdpo_subbatch < 1:
            raise ValueError("ppgdpo_subbatch must be >= 1.")
        return self


class EvaluationConfig(StrictModel):
    walk_forward_mode: Literal["fixed", "expanding", "rolling"] = "fixed"
    expanding_window: bool = False
    rolling_train_months: int | None = None
    retrain_every: int = 12
    refit_iters: int = 200
    ppgdpo_local_zero_hedge: bool = False
    ppgdpo_cross_scales: list[float] | None = None
    ppgdpo_allow_long_horizon: bool = True

    @property
    def is_expanding(self) -> bool:
        return self.walk_forward_mode == "expanding"

    @property
    def is_rolling(self) -> bool:
        return self.walk_forward_mode == "rolling"

    @property
    def is_fixed(self) -> bool:
        return self.walk_forward_mode == "fixed"

    @model_validator(mode="after")
    def validate_eval(self) -> "EvaluationConfig":
        if self.expanding_window and self.walk_forward_mode not in {"fixed", "expanding"}:
            raise ValueError(
                "evaluation.expanding_window cannot be combined with walk_forward_mode='rolling'."
            )
        if self.expanding_window:
            self.walk_forward_mode = "expanding"
        self.expanding_window = self.walk_forward_mode == "expanding"
        if self.walk_forward_mode not in {"fixed", "expanding", "rolling"}:
            raise ValueError(
                "evaluation.walk_forward_mode must be 'fixed', 'expanding', or 'rolling'."
            )
        if self.rolling_train_months is not None and self.rolling_train_months < 1:
            raise ValueError("evaluation.rolling_train_months must be >= 1 when provided.")
        if self.retrain_every < 0:
            raise ValueError("retrain_every must be >= 0.")
        if self.refit_iters < 0:
            raise ValueError("refit_iters must be >= 0.")
        return self


class RuntimeConfig(StrictModel):
    seed: int = 0
    device: str = "cuda"
    dtype: Literal["float32", "float64"] = "float64"
    backend: Literal["plan", "legacy_bridge", "native_stage3a", "native_stage3b", "native_stage4", "native_stage5", "native_stage6", "native_stage7"] = "plan"
    legacy_entrypoint: str | None = None
    legacy_workdir: str | None = None
    legacy_python: str | None = None
    fred_api_key_env: str | None = "FRED_API_KEY"
    pass_through_args: list[str] = Field(default_factory=list)
    tolerate_legacy_tail_errors: bool = True
    persist_legacy_logs: bool = True

    @property
    def legacy_entrypoint_path(self) -> Path | None:
        if self.legacy_entrypoint is None:
            return None
        return Path(self.legacy_entrypoint).expanduser().resolve()

    @property
    def legacy_workdir_path(self) -> Path | None:
        if self.legacy_workdir is None:
            return None
        return Path(self.legacy_workdir).expanduser().resolve()


class UniverseOverride(StrictModel):
    profile: str | None = None
    asset_universe: str | None = None
    bond_hook: Literal["none", "ust10y", "curve_core"] | None = None
    include_bond: bool | None = None
    bond_source: str | None = None
    bond_csv: str | None = None
    bond_name: str | None = None
    bond_series_id: str | None = None
    bond_ret_col: str | None = None
    bond_assets: list[BondAssetConfig] | None = None
    custom_risky_assets: list[CustomRiskyAssetConfig] | None = None


class MacroOverride(StrictModel):
    profile: str | None = None
    pool: Literal["legacy7", "bond_curve_core", "bond_curve_extended", "custom"] | None = None
    feature_ids: list[str] | None = None
    macro3_columns: list[str] | None = None


class ModelOverride(StrictModel):
    profile: str | None = None
    state_spec: str | None = None
    latent_k: int | None = None
    pls_horizon: int | None = None
    pls_smooth_span: int | None = None
    cross_mode: Literal["estimated", "zero"] | None = None
    zero_cross_policy_proxy: Literal["full", "myopic"] | None = None


class ConstraintOverride(StrictModel):
    profile: str | None = None
    enabled: bool | None = None
    risky_cap: float | None = None
    cash_floor: float | None = None
    per_asset_cap: float | None = None
    allow_short: bool | None = None
    short_floor: float | None = None


class MethodOverride(StrictModel):
    profile: str | None = None
    train_policy: bool | None = None
    evaluate_ppgdpo: bool | None = None
    tc_sweep: bool | None = None
    tc_bps: list[float] | None = None


class SplitOverride(StrictModel):
    profile: str | None = None
    train_pool_start: str | None = None
    train_pool_end: str | None = None
    final_test_start: str | None = None
    end_date: str | None = None
    selection_cv_folds: int | None = None
    selection_min_train_months: int | None = None
    selection_val_months: int | None = None
    force_common_calendar: bool | None = None
    common_start_mode: Literal["macro3", "suite"] | None = None


class SelectionOverride(StrictModel):
    profile: str | None = None
    enabled: bool | None = None
    candidate_mode: Literal["manual", "grid"] | None = None
    candidate_specs: list[str] | None = None
    include_static_specs: bool | None = None
    include_ff_split_specs: bool | None = None
    include_alias_specs: bool | None = None
    include_eqbond_block_specs: Literal["auto", "never", "always"] | None = None
    pca_prefixes: list[str] | None = None
    pca_components: list[int] | None = None
    pls_prefixes: list[str] | None = None
    pls_horizons: list[int] | None = None
    pls_components: list[int] | None = None
    top_k: int | None = None
    guarded_only: bool | None = None
    window_mode: Literal["rolling", "expanding"] | None = None
    rolling_window: int | None = None
    return_baseline: Literal["train_mean", "expanding_mean"] | None = None
    state_baseline: Literal["train_mean", "expanding_mean", "random_walk"] | None = None
    return_floor: float | None = None
    state_q10_floor: float | None = None
    cross_warn: float | None = None
    cross_fail: float | None = None
    score_mode: Literal["method_shortlist", "legacy_value"] | None = None
    score_ret_mean_weight: float | None = None
    score_ret_q10_weight: float | None = None
    score_state_mean_weight: float | None = None
    score_state_q10_weight: float | None = None
    output_subdir: str | None = None


class ComparisonOverride(StrictModel):
    profile: str | None = None
    enabled: bool | None = None
    spec_source: Literal["selected", "fixed"] | None = None
    selected_rank: int | None = None
    use_selected_spec: bool | None = None
    fixed_spec: str | None = None
    cross_modes: list[Literal["estimated", "zero"]] | None = None
    output_subdir: str | None = None


class RankSweepOverride(StrictModel):
    enabled: bool | None = None
    start_rank: int | None = None
    end_rank: int | None = None
    max_ranks: int | None = None
    resume: bool | None = None
    stop_on_error: bool | None = None
    output_subdir: str | None = None
    poll_seconds: float | None = None
    lock_timeout_seconds: float | None = None


class TrainingOverride(StrictModel):
    iters: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    gamma: float | None = None
    residual_policy: bool | None = None
    residual_ridge: float | None = None
    residual_w_max: float | None = None
    residual_detach_baseline: bool | None = None
    ppgdpo_mc: int | None = None
    ppgdpo_subbatch: int | None = None
    ppgdpo_seed: int | None = None
    ppgdpo_L: float | None = None
    ppgdpo_eps: float | None = None
    ppgdpo_ridge: float | None = None
    ppgdpo_tau: float | None = None
    ppgdpo_armijo: float | None = None
    ppgdpo_backtrack: float | None = None
    ppgdpo_newton: int | None = None
    ppgdpo_tol_grad: float | None = None
    ppgdpo_ls: int | None = None


class EvaluationOverride(StrictModel):
    walk_forward_mode: Literal["fixed", "expanding", "rolling"] | None = None
    expanding_window: bool | None = None
    rolling_train_months: int | None = None
    retrain_every: int | None = None
    refit_iters: int | None = None
    ppgdpo_local_zero_hedge: bool | None = None
    ppgdpo_cross_scales: list[float] | None = None
    ppgdpo_allow_long_horizon: bool | None = None


class RuntimeOverride(StrictModel):
    seed: int | None = None
    device: str | None = None
    dtype: Literal["float32", "float64"] | None = None
    backend: Literal["plan", "legacy_bridge", "native_stage3a", "native_stage3b", "native_stage4", "native_stage5", "native_stage6", "native_stage7"] | None = None
    legacy_entrypoint: str | None = None
    legacy_workdir: str | None = None
    legacy_python: str | None = None
    fred_api_key_env: str | None = None
    pass_through_args: list[str] | None = None
    tolerate_legacy_tail_errors: bool | None = None
    persist_legacy_logs: bool | None = None


class RawExperimentConfig(StrictModel):
    experiment: ExperimentMeta
    universe: UniverseOverride = Field(default_factory=UniverseOverride)
    macro: MacroOverride = Field(default_factory=MacroOverride)
    model: ModelOverride = Field(default_factory=ModelOverride)
    constraints: ConstraintOverride = Field(default_factory=ConstraintOverride)
    method: MethodOverride = Field(default_factory=MethodOverride)
    split: SplitOverride = Field(default_factory=SplitOverride)
    selection: SelectionOverride = Field(default_factory=SelectionOverride)
    comparison: ComparisonOverride = Field(default_factory=ComparisonOverride)
    rank_sweep: RankSweepOverride = Field(default_factory=RankSweepOverride)
    training: TrainingOverride = Field(default_factory=TrainingOverride)
    evaluation: EvaluationOverride = Field(default_factory=EvaluationOverride)
    runtime: RuntimeOverride = Field(default_factory=RuntimeOverride)


class ResolvedExperimentConfig(StrictModel):
    experiment: ExperimentMeta
    universe: UniverseConfig
    macro: MacroConfig
    model: ModelConfig
    constraints: ConstraintConfig
    method: MethodConfig
    split: SplitConfig
    selection: SelectionConfig
    comparison: ComparisonConfig
    rank_sweep: RankSweepConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    profile_source: str
    profile_root: str

    @property
    def slug(self) -> str:
        raw = self.experiment.name.strip().lower().replace(" ", "_")
        return "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})

    @property
    def selection_output_dir(self) -> Path:
        return Path(self.experiment.output_dir) / self.selection.output_subdir

    @property
    def comparison_output_dir(self) -> Path:
        return Path(self.experiment.output_dir) / self.comparison.output_subdir

    @property
    def manifest_path(self) -> Path:
        return Path(self.experiment.output_dir) / "stage2_manifest.yaml"

    @property
    def stage3a_manifest_path(self) -> Path:
        return Path(self.experiment.output_dir) / "stage3a_manifest.yaml"

    @property
    def stage3b_manifest_path(self) -> Path:
        return Path(self.experiment.output_dir) / "stage3b_manifest.yaml"

    @property
    def stage4_manifest_path(self) -> Path:
        return Path(self.experiment.output_dir) / "stage4_manifest.yaml"

    @property
    def stage5_manifest_path(self) -> Path:
        return Path(self.experiment.output_dir) / "stage5_manifest.yaml"

    @property
    def stage6_manifest_path(self) -> Path:
        return Path(self.experiment.output_dir) / "stage6_manifest.yaml"

    @property
    def stage7_manifest_path(self) -> Path:
        return Path(self.experiment.output_dir) / "stage7_manifest.yaml"

    @property
    def rank_sweep_output_dir(self) -> Path:
        return Path(self.experiment.output_dir) / self.rank_sweep.output_subdir

    @property
    def rank_sweep_queue_dir(self) -> Path:
        return self.rank_sweep_output_dir / "_queue"

    @property
    def stage7_rank_sweep_report_path(self) -> Path:
        return self.rank_sweep_output_dir / "stage7_rank_sweep_report.yaml"

    @property
    def rank_sweep_progress_path(self) -> Path:
        return self.rank_sweep_output_dir / "rank_sweep_progress.csv"

    @property
    def rank_sweep_results_path(self) -> Path:
        return self.rank_sweep_output_dir / "rank_sweep_results.csv"

    @property
    def rank_sweep_results_jsonl_path(self) -> Path:
        return self.rank_sweep_output_dir / "rank_sweep_results.jsonl"

    @property
    def rank_sweep_zero_cost_summary_path(self) -> Path:
        return self.rank_sweep_output_dir / "rank_sweep_cross_modes_zero_cost_summary.csv"

    @property
    def rank_sweep_all_costs_summary_path(self) -> Path:
        return self.rank_sweep_output_dir / "rank_sweep_cross_modes_all_costs_summary.csv"

    @property
    def stage7_selection_report_path(self) -> Path:
        return self.selection_output_dir / "stage7_selection_report.yaml"

    @property
    def logs_dir(self) -> Path:
        return Path(self.experiment.output_dir) / "logs"

    @property
    def selected_spec_path(self) -> Path:
        return Path(self.experiment.output_dir) / "selected_spec.yaml"

    @property
    def selection_report_path(self) -> Path:
        return self.selection_output_dir / "stage3a_selection_report.yaml"

    @property
    def stage3b_selection_report_path(self) -> Path:
        return self.selection_output_dir / "stage3b_selection_report.yaml"

    @property
    def stage4_selection_report_path(self) -> Path:
        return self.selection_output_dir / "stage4_selection_report.yaml"

    @property
    def stage5_selection_report_path(self) -> Path:
        return self.selection_output_dir / "stage5_selection_report.yaml"

    @property
    def stage6_selection_report_path(self) -> Path:
        return self.selection_output_dir / "stage6_selection_report.yaml"

    @property
    def comparison_report_path(self) -> Path:
        return self.comparison_output_dir / "stage3a_comparison_report.yaml"

    @property
    def stage3b_comparison_report_path(self) -> Path:
        return self.comparison_output_dir / "stage3b_comparison_report.yaml"

    @property
    def stage4_comparison_report_path(self) -> Path:
        return self.comparison_output_dir / "stage4_comparison_report.yaml"

    @property
    def stage5_comparison_report_path(self) -> Path:
        return self.comparison_output_dir / "stage5_comparison_report.yaml"

    @property
    def stage6_comparison_report_path(self) -> Path:
        return self.comparison_output_dir / "stage6_comparison_report.yaml"

    @property
    def comparison_summary_from_log_path(self) -> Path:
        return self.comparison_output_dir / "comparison_summary_from_log.csv"

    @property
    def comparison_zero_cost_summary_path(self) -> Path:
        return self.comparison_output_dir / "comparison_zero_cost_summary.csv"

    @property
    def comparison_all_costs_summary_path(self) -> Path:
        return self.comparison_output_dir / "comparison_all_costs_summary.csv"

    @property
    def comparison_cross_modes_zero_cost_summary_path(self) -> Path:
        return self.comparison_output_dir / "comparison_cross_modes_zero_cost_summary.csv"

    @property
    def comparison_cross_modes_all_costs_summary_path(self) -> Path:
        return self.comparison_output_dir / "comparison_cross_modes_all_costs_summary.csv"

    @property
    def effective_rolling_train_months(self) -> int | None:
        if self.evaluation.rolling_train_months is not None:
            return int(self.evaluation.rolling_train_months)
        if self.split.train_pool_start is None:
            return None
        start = self.split.train_pool_start_date
        end = self.split.train_pool_end_date
        if start is None:
            return None
        return (end.year - start.year) * 12 + (end.month - start.month) + 1

    def plan_summary(self) -> dict[str, object]:
        return {
            "name": self.experiment.name,
            "output_dir": self.experiment.output_dir,
            "profile_root": self.profile_root,
            "asset_universe": self.universe.asset_universe,
            "custom_risky_assets": list(self.universe.custom_risky_asset_names),
            "include_bond": self.universe.include_bond,
            "bond_count": self.universe.bond_count,
            "macro_pool": self.macro.pool,
            "macro_feature_count": len(self.macro.effective_feature_ids),
            "base_state_spec": self.model.state_spec,
            "cross_mode": self.model.cross_mode,
            "split_train_pool_start": self.split.train_pool_start,
            "split_train_pool_end": self.split.train_pool_end,
            "split_final_test_start": self.split.final_test_start,
            "split_end_date": self.split.end_date,
            "split_force_common_calendar": self.split.force_common_calendar,
            "split_common_start_mode": self.split.common_start_mode,
            "selection_enabled": self.selection.enabled,
            "selection_folds": self.split.selection_cv_folds,
            "selection_top_k": self.selection.top_k,
            "selection_window_mode": self.selection.window_mode,
            "comparison_enabled": self.comparison.enabled,
            "comparison_use_selected_spec": self.comparison.use_selected_spec,
            "comparison_spec_source": self.comparison.spec_source,
            "comparison_selected_rank": self.comparison.selected_rank,
            "comparison_cross_modes": list(self.comparison.cross_modes),
            "rank_sweep_enabled": self.rank_sweep.enabled,
            "rank_sweep_start_rank": self.rank_sweep.start_rank,
            "rank_sweep_end_rank": self.rank_sweep.end_rank,
            "rank_sweep_max_ranks": self.rank_sweep.max_ranks,
            "rank_sweep_output_dir": str(self.rank_sweep_output_dir),
            "effective_risky_cap": self.constraints.effective_risky_cap,
            "simplex_fast_path": self.constraints.simplex_fast_path,
            "train_policy": self.method.train_policy,
            "evaluate_ppgdpo": self.method.evaluate_ppgdpo,
            "walk_forward_mode": self.evaluation.walk_forward_mode,
            "rolling_train_months": self.evaluation.rolling_train_months,
            "backend": self.runtime.backend,
        }
