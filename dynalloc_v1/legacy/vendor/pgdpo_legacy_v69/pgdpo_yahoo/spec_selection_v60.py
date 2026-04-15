"""v60 model-selection utilities.

Compared with v59's spec-selection module, this version keeps the ranking stage
strictly *method-free* while broadening the **handcrafted / supervised spec menu**.
The learned recurrent `vrnn_*` line was retired after repeated empirical failure.

Ranking remains based on predictive diagnostics and stability / cross-shock guards.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from .discrete_latent_model import LatentPCAConfig
from .model_specs_v60 import infer_spec_family
from .spec_selection import (
    SpecSelectionConfig,
    evaluate_spec_predictive_diagnostics,
    rank_specs_from_results as _rank_specs_from_results_v59,
)


@dataclass
class ModelSelectionResult:
    spec: str
    family: str
    block: str
    train_end: int
    start_idx: int
    end_idx: int
    horizon: int
    n_assets: int
    n_states: int
    r2_is_ret_mean: float
    r2_is_ret_median: float
    r2_oos_ret_mean: float
    r2_oos_ret_median: float
    r2_roll_ret_q10: float
    r2_roll_ret_min: float
    r2_is_state_mean: float
    r2_is_state_median: float
    r2_oos_state_mean: float
    r2_oos_state_median: float
    r2_roll_state_q10: float
    r2_roll_state_min: float
    cross_mean_abs_rho: float
    cross_p95_abs_rho: float
    cross_max_abs_rho: float
    train_objective: float = float("nan")
    train_recon_loss: float = float("nan")
    train_return_loss: float = float("nan")
    train_state_loss: float = float("nan")
    train_kl_loss: float = float("nan")
    obs_dim: int = 0
    hidden_dim: int = 0
    model_input_dim: int = 0


def _legacy_to_model_result(legacy: Any, *, family: str) -> ModelSelectionResult:
    d = asdict(legacy)
    return ModelSelectionResult(family=str(family), **d)


def evaluate_model_predictive_diagnostics(
    *,
    spec: str,
    block: str,
    r_ex: pd.DataFrame,
    macro3: pd.DataFrame | None,
    macro7: pd.DataFrame | None,
    ff3: pd.DataFrame | None,
    ff5: pd.DataFrame | None,
    bond_asset_names: list[str] | None,
    train_end: int,
    start_idx: int,
    end_idx: int,
    pca_cfg: LatentPCAConfig,
    pls_horizon: int,
    pls_smooth_span: int,
    block_eq_k: int,
    block_bond_k: int,
    config: SpecSelectionConfig,
) -> ModelSelectionResult:
    legacy = evaluate_spec_predictive_diagnostics(
        spec=spec,
        block=block,
        r_ex=r_ex,
        macro3=macro3,
        macro7=macro7,
        ff3=ff3,
        ff5=ff5,
        bond_asset_names=bond_asset_names,
        train_end=train_end,
        start_idx=start_idx,
        end_idx=end_idx,
        pca_cfg=pca_cfg,
        pls_horizon=pls_horizon,
        pls_smooth_span=pls_smooth_span,
        block_eq_k=block_eq_k,
        block_bond_k=block_bond_k,
        config=config,
    )
    return _legacy_to_model_result(legacy, family=infer_spec_family(spec))


def results_to_frame_v60(results: list[ModelSelectionResult] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame([asdict(r) for r in results])


def rank_models_from_results(
    results: list[ModelSelectionResult] | pd.DataFrame,
    *,
    config: SpecSelectionConfig,
) -> pd.DataFrame:
    df = results_to_frame_v60(results)
    if df.empty:
        return df

    family_map = df.groupby("spec", dropna=False)["family"].first().reset_index()
    ranked = _rank_specs_from_results_v59(df, config=config)
    ranked = ranked.merge(family_map, on="spec", how="left")

    front_cols = [c for c in ["spec", "family", "score", "passes_guard", "recommended"] if c in ranked.columns]
    other_cols = [c for c in ranked.columns if c not in front_cols]
    return ranked[front_cols + other_cols]
