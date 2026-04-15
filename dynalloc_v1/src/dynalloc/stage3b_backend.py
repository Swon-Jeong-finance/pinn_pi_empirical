
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .artifacts import (
    discover_comparison_artifacts,
    discover_selection_artifacts,
    normalize_tc_sweep_csv,
    render_artifact_snapshot,
    summarize_monthly_csv,
)
from .legacy_bridge import (
    LegacyRunSpec,
    _phase_specs_for_render,
    build_comparison_run_spec,
    build_selection_run_spec,
    resolve_comparison_spec,
)
from .schema import ResolvedExperimentConfig
from .selection import choose_selected_specs, resolve_candidate_specs, write_selected_spec_artifact


class Stage3BError(RuntimeError):
    pass


def _macro3_subset_columns(args: Any) -> list[str]:
    cols = list(getattr(args, "stage4_macro3_columns", ["infl_yoy", "term_spread", "default_spread"]))
    if len(cols) < 1:
        raise Stage3BError("stage4 macro3 subset must contain at least one column.")
    return cols


def _stage4_macro_features(args: Any) -> list[str]:
    return list(getattr(args, "stage4_macro_features", []))


def _stage4_macro_requires_fred(args: Any) -> bool:
    features = _stage4_macro_features(args)
    if not features:
        return True
    no_fred = {"rv_mkt_1m", "rv_mkt_3m", "log_rv_mkt_1m", "log_rv_mkt_3m"}
    return any(feat not in no_fred for feat in features)


@dataclass(frozen=True)
class Stage3BRunResult:
    selected_spec: str | None
    selected_specs: list[str]
    completed_phases: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class Stage3BPhaseSnapshot:
    phase: str
    command: str
    log_path: str
    returncode: int
    backend: str = "native_stage3b"
    artifacts: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage3BManifest:
    experiment: str
    backend: str
    completed_phases: list[str]
    selected_specs: list[str]
    primary_selected_spec: str | None
    warnings: list[str]
    snapshots: list[Stage3BPhaseSnapshot]

    def dump(self, path: Path) -> None:
        payload = {
            "experiment": self.experiment,
            "backend": self.backend,
            "completed_phases": self.completed_phases,
            "selected_specs": self.selected_specs,
            "primary_selected_spec": self.primary_selected_spec,
            "warnings": self.warnings,
            "snapshots": [
                {
                    "phase": snap.phase,
                    "command": snap.command,
                    "log_path": snap.log_path,
                    "returncode": snap.returncode,
                    "backend": snap.backend,
                    "artifacts": snap.artifacts,
                }
                for snap in self.snapshots
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False))


@dataclass
class VendorContext:
    module: Any
    workdir: Path
    dataset_cache: dict[tuple[str, ...], "Stage3BDataset"] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage3BDataset:
    specs: tuple[str, ...]
    r_ex: pd.DataFrame
    rf: pd.Series
    mkt: pd.Series
    macro3: pd.DataFrame | None
    macro7: pd.DataFrame | None
    ff3_state: pd.DataFrame | None
    ff5_state: pd.DataFrame | None
    bond_asset_names: list[str]
    r_ex_np: np.ndarray
    rf_np: np.ndarray
    mkt_np: np.ndarray
    device: torch.device
    dtype: torch.dtype
    pca_cfg: Any


@dataclass(frozen=True)
class EvalBlock:
    tag: str
    train_end: int
    start_idx: int
    end_idx: int
    H_eval: int


class _TeeStream:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def _tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        tee_out = _TeeStream(sys.__stdout__, fh)
        tee_err = _TeeStream(sys.__stderr__, fh)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            yield


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _scoped_log_path(config: ResolvedExperimentConfig, *, output_dir: Path, filename: str) -> Path:
    root = Path(config.experiment.output_dir).resolve()
    target = output_dir.resolve()
    try:
        rel = target.relative_to(root)
        return config.logs_dir / rel / filename
    except ValueError:
        return config.logs_dir / filename


def _selection_log_path(config: ResolvedExperimentConfig) -> Path:
    return _scoped_log_path(
        config,
        output_dir=config.selection_output_dir,
        filename="selection_stage3b.log",
    )


def _comparison_log_path(config: ResolvedExperimentConfig, spec: str) -> Path:
    safe = "".join(ch for ch in spec if ch.isalnum() or ch in {"_", "-"})
    return _scoped_log_path(
        config,
        output_dir=config.comparison_output_dir,
        filename=f"comparison_{safe}_stage3b.log",
    )


def _fred_value(config: ResolvedExperimentConfig) -> str | None:
    if not config.runtime.fred_api_key_env:
        return None
    value = os.environ.get(config.runtime.fred_api_key_env)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _import_vendor_module(config: ResolvedExperimentConfig) -> VendorContext:
    entrypoint = config.runtime.legacy_entrypoint_path
    if entrypoint is None:
        raise Stage3BError("runtime.legacy_entrypoint must be set for native_stage3b.")
    workdir = config.runtime.legacy_workdir_path or entrypoint.parent
    if not workdir.exists():
        raise Stage3BError(f"legacy workdir does not exist: {workdir}")
    workdir_str = str(workdir)
    if workdir_str not in sys.path:
        sys.path.insert(0, workdir_str)

    module_name = f"_dynalloc_vendor_runner_{abs(hash(str(entrypoint.resolve())))}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, entrypoint)
        if spec is None or spec.loader is None:
            raise Stage3BError(f"Could not import vendor entrypoint: {entrypoint}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return VendorContext(module=module, workdir=workdir)


def _bond_arg_values(config: ResolvedExperimentConfig) -> dict[str, Any]:
    bond_assets = config.universe.normalized_bond_assets
    payload: dict[str, Any] = {
        "include_bond": bool(config.universe.include_bond and len(bond_assets) > 0),
        "bond_source": str(config.universe.bond_source_kind or config.universe.bond_source),
        "bond_csv_specs": "",
        "bond_fred_specs": "",
        "bond_name": str(config.universe.bond_name),
        "bond_csv": str(config.universe.bond_csv),
        "bond_series_id": str(config.universe.bond_series_id),
        "bond_ret_col": str(config.universe.bond_ret_col),
    }
    if not bond_assets:
        return payload
    first = bond_assets[0]
    payload["bond_name"] = str(first.name)
    if first.source == "crsp_csv":
        payload["bond_csv"] = str(first.csv)
        payload["bond_ret_col"] = str(first.ret_col)
        payload["bond_csv_specs"] = ",".join(
            f"{asset.name}={asset.csv}@{asset.ret_col}" for asset in bond_assets
        )
    else:
        payload["bond_series_id"] = str(first.series_id)
        payload["bond_fred_specs"] = ",".join(
            f"{asset.name}={asset.series_id}" for asset in bond_assets
        )
    return payload



def _build_args(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    state_spec: str,
    specs: list[str],
    out_dir: Path,
) -> argparse.Namespace:
    # Defaults mirror the legacy runner closely enough for the code paths used by stage3b.
    vals: dict[str, Any] = {
        "fred_api_key": _fred_value(config),
        "out_dir": str(out_dir.resolve()),
        "seed": int(config.runtime.seed),
        "device": str(config.runtime.device),
        "dtype": str(config.runtime.dtype),
        "iters": int(config.training.iters if config.method.train_policy else 0),
        "batch_size": int(config.training.batch_size),
        "lr": float(config.training.lr),
        "gamma": float(config.training.gamma),
        "residual_policy": bool(config.training.residual_policy),
        "residual_ridge": float(config.training.residual_ridge),
        "residual_w_max": config.training.residual_w_max,
        "residual_detach_baseline": bool(config.training.residual_detach_baseline),
        "risky_cap": float(config.constraints.risky_cap),
        "cash_floor": float(config.constraints.cash_floor),
        "per_asset_cap": config.constraints.per_asset_cap,
        "allow_short": bool(config.constraints.allow_short),
        "short_floor": float(config.constraints.short_floor),
        "latent_k": int(config.model.latent_k),
        "pls_horizon": int(config.model.pls_horizon),
        "pls_smooth_span": int(config.model.pls_smooth_span),
        "state_spec": str(state_spec),
        "compare_specs": bool(len(specs) > 1),
        "specs": list(specs) if len(specs) > 1 else None,
        "select_specs_first": False,
        "selection_only": phase == "selection",
        "selection_top_k": int(config.selection.top_k),
        "selection_window_mode": str(config.selection.window_mode),
        "selection_rolling_window": int(config.selection.rolling_window),
        "selection_return_baseline": str(config.selection.return_baseline),
        "selection_state_baseline": str(config.selection.state_baseline),
        "selection_alpha": 0.25,
        "selection_beta": 0.50,
        "selection_gamma": 0.25,
        "selection_return_floor": float(config.selection.return_floor),
        "selection_state_q10_floor": float(config.selection.state_q10_floor),
        "selection_cross_warn": float(config.selection.cross_warn),
        "selection_cross_fail": float(config.selection.cross_fail),
        "selection_score_mode": str(config.selection.score_mode),
        "selection_score_ret_mean_weight": float(config.selection.score_ret_mean_weight),
        "selection_score_ret_q10_weight": float(config.selection.score_ret_q10_weight),
        "selection_score_state_mean_weight": float(config.selection.score_state_mean_weight),
        "selection_score_state_q10_weight": float(config.selection.score_state_q10_weight),
        "selection_guarded_only": bool(config.selection.guarded_only),
        "asset_universe": str(config.universe.asset_universe),
        "custom_risky_assets": [asset.model_dump() for asset in config.universe.custom_risky_assets],
        "custom_risky_assets_json": json.dumps([asset.model_dump() for asset in config.universe.custom_risky_assets], separators=(",", ":")) if config.universe.custom_risky_assets else "",
        **_bond_arg_values(config),
        "end_date": str(config.split.end_date),
        "train_pool_start": config.split.train_pool_start,
        "eval_mode": "v57" if phase == "selection" else "legacy",
        "eval_horizons": [int(config.split.final_test_years)] if phase != "selection" else None,
        "eval_3x10y": False,
        "eval_30y": False,
        "force_common_calendar": bool(config.split.force_common_calendar),
        "common_start_mode": str(config.split.common_start_mode),
        "train_horizon": 0,
        "benchmarks": ["gmv", "risk_parity", "inv_vol"],
        "bench_w_max": None,
        "rp_max_iter": 500,
        "rp_tol": 1e-8,
        "bench_refit_every": 1,
        "benchmarks_model": ["gmv", "risk_parity", "inv_vol"],
        "cross_mode": str(config.model.cross_mode),
        "zero_cross_policy_proxy": str(config.model.zero_cross_policy_proxy),
        "tc_sweep": bool(config.method.tc_sweep),
        "tc_bps": [float(x) for x in config.method.tc_bps],
        "ppgdpo": bool(config.method.evaluate_ppgdpo),
        "ppgdpo_mc": int(config.training.ppgdpo_mc),
        "ppgdpo_subbatch": int(config.training.ppgdpo_subbatch),
        "ppgdpo_seed": int(config.training.ppgdpo_seed),
        "ppgdpo_L": config.training.ppgdpo_L,
        "ppgdpo_eps": float(config.training.ppgdpo_eps),
        "ppgdpo_ridge": float(config.training.ppgdpo_ridge),
        "ppgdpo_tau": float(config.training.ppgdpo_tau),
        "ppgdpo_armijo": float(config.training.ppgdpo_armijo),
        "ppgdpo_backtrack": float(config.training.ppgdpo_backtrack),
        "ppgdpo_newton": int(config.training.ppgdpo_newton),
        "ppgdpo_tol_grad": float(config.training.ppgdpo_tol_grad),
        "ppgdpo_ls": int(config.training.ppgdpo_ls),
        "ppgdpo_local_zero_hedge": bool(config.evaluation.ppgdpo_local_zero_hedge),
        "ppgdpo_cross_scales": None if config.evaluation.ppgdpo_cross_scales is None else [float(x) for x in config.evaluation.ppgdpo_cross_scales],
        "ppgdpo_allow_long_horizon": bool(config.evaluation.ppgdpo_allow_long_horizon),
        "walk_forward_mode": str(config.evaluation.walk_forward_mode),
        "expanding_window": bool(config.evaluation.is_expanding),
        "rolling_train_months": config.effective_rolling_train_months,
        "retrain_every": int(config.evaluation.retrain_every),
        "refit_iters": int(config.evaluation.refit_iters),
        "cv_folds": int(config.split.selection_cv_folds),
        "cv_min_train_months": int(config.split.selection_min_train_months),
        "cv_val_months": int(config.split.selection_val_months),
        "block_eq_k": 1,
        "block_bond_k": 1,
    }
    return argparse.Namespace(**vals)


def _dataset_cache_key(args: argparse.Namespace, specs: list[str]) -> tuple[str, ...]:
    # include the spec list, the modeled cross-world, the risky-asset menu, and the bond menu.
    asset_universe = str(getattr(args, "asset_universe", "ff49ind"))
    custom_json = str(getattr(args, "custom_risky_assets_json", "") or "")
    bond_key = "|".join(
        [
            str(int(bool(getattr(args, "include_bond", False)))),
            str(getattr(args, "bond_source", "")),
            str(getattr(args, "bond_csv_specs", "") or getattr(args, "bond_csv", "")),
            str(getattr(args, "bond_fred_specs", "") or getattr(args, "bond_series_id", "")),
        ]
    )
    return tuple(
        list(specs)
        + [
            f"cross={getattr(args, 'cross_mode', 'estimated')}",
            f"asset_universe={asset_universe}",
            f"custom_assets={custom_json}",
            f"bond_assets={bond_key}",
        ]
    )


def _get_or_prepare_dataset(ctx: VendorContext, args: argparse.Namespace, specs: list[str]) -> Stage3BDataset:
    key = _dataset_cache_key(args, specs)
    if key not in ctx.dataset_cache:
        with _pushd(ctx.workdir):
            ctx.dataset_cache[key] = _prepare_dataset(ctx.module, args, specs)
    return ctx.dataset_cache[key]


def _custom_risky_assets_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = getattr(args, "custom_risky_assets", None)
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    raw_json = str(getattr(args, "custom_risky_assets_json", "") or "").strip()
    if raw_json == "":
        return []
    try:
        payload = json.loads(raw_json)
    except Exception as exc:
        raise Stage3BError(f"Could not parse custom_risky_assets_json: {exc}") from exc
    if not isinstance(payload, list):
        raise Stage3BError("custom_risky_assets_json must decode to a list of asset mappings.")
    return [dict(item) for item in payload]


def _load_custom_risky_assets(v: Any, args: argparse.Namespace, ff: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    asset_specs = _custom_risky_assets_from_args(args)
    if len(asset_specs) == 0:
        raise Stage3BError("asset_universe='custom' requires custom_risky_assets in the config.")
    frames: list[pd.Series] = []
    seen_names: set[str] = set()
    base_cache: dict[str, pd.DataFrame] = {}
    for spec in asset_specs:
        name = str(spec.get("name", "") or "").strip()
        if name == "":
            raise Stage3BError("Every custom risky asset needs a non-empty name.")
        if name in seen_names:
            raise Stage3BError(f"Duplicate custom risky asset name '{name}'.")
        seen_names.add(name)
        source = str(spec.get("source", "market_total_return") or "market_total_return").strip()
        if source == "market_total_return":
            series = mkt.rename(name)
        elif source == "equity_universe_column":
            base_universe = str(spec.get("base_universe", "") or "").strip()
            column = str(spec.get("column", "") or "").strip()
            if base_universe == "" or column == "":
                raise Stage3BError(
                    f"Custom risky asset '{name}' with source='equity_universe_column' requires base_universe and column."
                )
            if base_universe not in base_cache:
                base_cache[base_universe] = v._clean_returns_decimal(v.load_equity_universe_monthly(base_universe))
            base_df = base_cache[base_universe]
            if column not in base_df.columns:
                raise Stage3BError(
                    f"Custom risky asset '{name}' requested column '{column}' from {base_universe}, but it was not found."
                )
            series = base_df[column].rename(name)
        elif source == "csv":
            csv_path = Path(str(spec.get("csv", "") or "")).expanduser()
            if not csv_path.is_absolute():
                csv_path = (Path.cwd() / csv_path).resolve()
            ret_col = str(spec.get("ret_col", "") or "").strip()
            date_col = str(spec.get("date_col", "date") or "date").strip()
            if ret_col == "":
                raise Stage3BError(f"Custom risky asset '{name}' with source='csv' requires ret_col.")
            if not csv_path.exists():
                raise Stage3BError(f"Custom risky asset '{name}' CSV not found: {csv_path}")
            df = pd.read_csv(csv_path)
            if date_col not in df.columns or ret_col not in df.columns:
                raise Stage3BError(
                    f"Custom risky asset '{name}' CSV must contain columns '{date_col}' and '{ret_col}'."
                )
            raw = df[[date_col, ret_col]].copy()
            raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce") + pd.offsets.MonthEnd(0)
            raw = raw.dropna(subset=[date_col])
            raw = raw.set_index(date_col).sort_index()
            series = v._clean_returns_decimal(raw[[ret_col]]).iloc[:, 0].rename(name)
        else:
            raise Stage3BError(
                f"Unknown custom risky asset source '{source}' for asset '{name}'."
            )
        frames.append(series)
    risky = pd.concat([s.to_frame() for s in frames], axis=1)
    risky = risky.replace([np.inf, -np.inf], np.nan)
    return risky


def _prepare_dataset(v: Any, args: argparse.Namespace, specs: list[str]) -> Stage3BDataset:
    for s in specs:
        if s not in v.STATE_SPECS:
            raise Stage3BError(f"Unknown spec '{s}'. Choose from {list(v.STATE_SPECS)}.")

    macro3_needed_for_state = any(v.spec_requires_macro3(s) for s in specs)
    macro7_needed_for_state = any(v.spec_requires_macro7(s) for s in specs)
    ff3_needed_for_state = any(v.spec_requires_ff3(s) for s in specs)
    ff5_needed_for_state = any(v.spec_requires_ff5(s) for s in specs)
    bond_needed_for_state = any(v.spec_requires_bond_panel(s) for s in specs)

    fred_key_provided = not (args.fred_api_key is None or str(args.fred_api_key).strip() == "")

    common_start_mode = str(getattr(args, "common_start_mode", "suite") or "suite")
    if common_start_mode not in ("macro3", "suite"):
        print(f"[WARN] Unknown --common_start_mode={common_start_mode}; defaulting to 'suite'.")
        common_start_mode = "suite"

    force_common = bool(getattr(args, "force_common_calendar", True))
    calendar_suite = bool(force_common and common_start_mode == "suite")
    calendar_macro3 = bool(force_common and common_start_mode == "macro3")

    bond_requires_fred = bool(
        getattr(args, "include_bond", False)
        and str(getattr(args, "bond_source", "crsp_csv")) == "fred_tr"
        and (
            str(getattr(args, "bond_fred_specs", "") or "").strip() != ""
            or str(getattr(args, "bond_series_id", "BAMLCC0A0CMTRIV") or "").strip() != ""
        )
    )
    if bond_needed_for_state and (not bool(getattr(args, "include_bond", False))):
        raise Stage3BError(
            "The requested *_eqbond_block state spec requires at least one bond asset, but include_bond is false."
        )
    macro_requires_fred = _stage4_macro_requires_fred(args)
    fred_required = bool(bond_requires_fred or ((macro3_needed_for_state or macro7_needed_for_state or calendar_suite) and macro_requires_fred))
    if fred_required and not fred_key_provided:
        raise Stage3BError(
            "FRED API key is required when using FRED-backed macro3/macro7 specs and/or fred bond data, "
            "or when the common calendar suite alignment needs FRED-backed macro features."
        )

    macro_for_calendar = bool(force_common and ((fred_key_provided and macro_requires_fred) or (not macro_requires_fred)))
    want_macro7 = bool(macro7_needed_for_state or (calendar_suite and macro_for_calendar))
    want_macro3 = bool((macro3_needed_for_state or (calendar_macro3 and macro_for_calendar)) and (not want_macro7))
    fred_needed = bool(bond_requires_fred or want_macro3 or want_macro7)
    ff5_for_calendar = bool(force_common and common_start_mode == "suite")

    print(f"State specs to run: {specs}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    print("Downloading Ken French F-F Research Factors (monthly)...")
    ff = v._clean_returns_decimal(v.load_ff_factors_monthly())

    rf = ff["RF"].rename("rf")
    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("mkt")

    asset_universe = str(getattr(args, "asset_universe", "ff49ind"))
    if asset_universe == "custom":
        print("Building custom risky-asset universe from config...")
        ind = _load_custom_risky_assets(v, args, ff, mkt)
        universe_label = f"Custom risky-asset universe ({', '.join(ind.columns)})"
        print(f"Using {universe_label}.")
    else:
        universe_label = v.describe_asset_universe(asset_universe)
        print(f"Downloading {universe_label} (monthly)...")
        ind = v._clean_returns_decimal(v.load_equity_universe_monthly(asset_universe))

    ff3_state = None
    if ff3_needed_for_state:
        ff3_state = ff[["Mkt-RF", "SMB", "HML"]].copy()

    ff5_state = None
    if ff5_needed_for_state or ff5_for_calendar:
        msg = "Downloading Ken French F-F 5 Factors (monthly)..."
        if ff5_for_calendar and (not ff5_needed_for_state):
            msg += " [calendar-only]"
        print(msg)
        ff5_all = v._clean_returns_decimal(v.load_ff5_factors_monthly())
        ff5_state = ff5_all[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].copy()
        if ff3_needed_for_state:
            ff3_state = ff5_state[["Mkt-RF", "SMB", "HML"]].copy()

    mac_cfg = v.FredMacroConfig(api_key=str(args.fred_api_key), end=str(args.end_date)) if fred_needed else None

    macro3 = None
    macro7 = None
    if want_macro7:
        if macro_requires_fred and not fred_key_provided:
            raise Stage3BError("FRED API key is required for FRED-backed macro7 specs.")
        msg = (
            "Downloading FRED/custom macro predictors (macro7-compatible pool)..."
            if macro_requires_fred else
            "Building custom macro predictors without FRED dependency..."
        )
        if calendar_suite and (not macro7_needed_for_state):
            msg += " [calendar-only]"
        print(msg)
        assert mac_cfg is not None
        macro7 = v._clean_macro(v.build_macro7_monthly(mac_cfg))
        macro3 = macro7[_macro3_subset_columns(args)].copy()
    elif want_macro3:
        if macro_requires_fred and not fred_key_provided:
            print("[WARN] common calendar requested but no FRED key provided; proceeding without macro3.")
        else:
            msg = ("Downloading FRED/custom macro predictors (macro3-compatible subset)..."
                   if macro_requires_fred else
                   "Building custom macro3-compatible predictors without FRED dependency...")
            if macro_for_calendar and (not macro3_needed_for_state) and (not macro7_needed_for_state):
                msg += " [calendar-only]"
            print(msg)
            assert mac_cfg is not None
            macro3 = v._clean_macro(v.build_macro3_monthly(mac_cfg))

    bond_asset_names: list[str] = []
    if getattr(args, "include_bond", False):
        bond_df = v._load_bond_panel(args, mac_cfg)
        if bond_df.empty:
            raise Stage3BError("Bond loading returned an empty panel.")
        rename_map: dict[str, str] = {}
        for col in bond_df.columns:
            name = str(col)
            if name in ind.columns:
                rename_map[name] = f"{name}_BOND"
        if len(rename_map) > 0:
            bond_df = bond_df.rename(columns=rename_map)
        bond_asset_names = [str(c) for c in bond_df.columns]
        ind = pd.concat([ind, bond_df], axis=1)

    parts = [ind, rf.to_frame(), mkt.to_frame()]
    if macro7 is not None:
        parts.append(macro7)
    elif macro3 is not None:
        parts.append(macro3)
    if ff5_state is not None:
        parts.append(ff5_state)
    elif ff3_state is not None:
        parts.append(ff3_state)

    aligned = v._align_common(*parts)
    ind = aligned[0]
    rf_df = aligned[1]
    mkt_df = aligned[2]

    cursor = 3
    if macro7 is not None:
        macro7 = aligned[cursor]
        cursor += 1
    elif macro3 is not None:
        macro3 = aligned[cursor]
        cursor += 1

    if ff5_state is not None:
        ff5_state = aligned[cursor]
        cursor += 1
    elif ff3_state is not None:
        ff3_state = aligned[cursor]
        cursor += 1

    rf_s = rf_df.iloc[:, 0].rename("rf")
    mkt_s = mkt_df.iloc[:, 0].rename("mkt")

    concat_list: list[Any] = [ind, rf_s, mkt_s]
    if macro7 is not None:
        concat_list.append(macro7)
    elif macro3 is not None:
        concat_list.append(macro3)
    if ff5_state is not None:
        concat_list.append(ff5_state)
    elif ff3_state is not None:
        concat_list.append(ff3_state)

    joint = pd.concat(concat_list, axis=1)
    joint = joint.replace([np.inf, -np.inf], np.nan)

    end_ts = pd.Timestamp(str(getattr(args, "end_date", "2024-12-31"))) + pd.offsets.MonthEnd(0)
    joint = joint.loc[:end_ts]
    joint = joint.dropna()

    train_pool_start = getattr(args, "train_pool_start", None)
    if train_pool_start is not None and str(train_pool_start).strip():
        start_ts = pd.Timestamp(str(train_pool_start)) + pd.offsets.MonthEnd(0)
        joint = joint.loc[start_ts:]
        if joint.empty:
            raise Stage3BError(
                f"No data left after applying split.train_pool_start={start_ts.date()}."
            )
        print(
            f"[TRAIN POOL START] requested_start={start_ts.date()} | clipped_start={joint.index.min().date()} | T={len(joint)}"
        )

    if force_common:
        print(
            f"[COMMON START] mode={common_start_mode} | aligned_start={joint.index.min().date()} "
            f"| aligned_end={joint.index.max().date()} | T={len(joint)}"
        )

    ind = joint[ind.columns]
    rf = joint["rf"]
    mkt = joint["mkt"]

    if macro7 is not None:
        macro7 = joint[macro7.columns]
        macro3 = macro7[_macro3_subset_columns(args)].copy()
    elif macro3 is not None:
        macro3 = joint[macro3.columns]

    if ff5_state is not None:
        ff5_state = joint[ff5_state.columns]
        if ff3_needed_for_state:
            ff3_state = ff5_state[["Mkt-RF", "SMB", "HML"]].copy()
    elif ff3_state is not None:
        ff3_state = joint[ff3_state.columns]

    r_ex = ind.sub(rf, axis=0)
    r_ex_np = r_ex.values.astype(float)
    rf_np = rf.values.astype(float)
    mkt_np = mkt.values.astype(float)

    print(
        f"Dataset base (aligned): {len(r_ex)} obs | assets={r_ex.shape[1]} | "
        f"macro3={(0 if macro3 is None else macro3.shape[1])} "
        f"macro7={(0 if macro7 is None else macro7.shape[1])} "
        f"ff3={(0 if ff3_state is None else ff3_state.shape[1])} "
        f"ff5={(0 if ff5_state is None else ff5_state.shape[1])}"
    )
    print(f"Date range (aligned): {r_ex.index[0].date()} .. {r_ex.index[-1].date()}")
    v._warn_if_missing_months(r_ex.index, label="aligned dataset")

    pca_cfg = v.LatentPCAConfig(
        n_components=int(args.latent_k),
        standardize=True,
        random_state=int(args.seed),
    )
    return Stage3BDataset(
        specs=tuple(specs),
        r_ex=r_ex,
        rf=rf,
        mkt=mkt,
        macro3=macro3,
        macro7=macro7,
        ff3_state=ff3_state,
        ff5_state=ff5_state,
        bond_asset_names=bond_asset_names,
        r_ex_np=r_ex_np,
        rf_np=rf_np,
        mkt_np=mkt_np,
        device=device,
        dtype=dtype,
        pca_cfg=pca_cfg,
    )


def _make_block(v: Any, idx: pd.DatetimeIndex, *, tag_prefix: str, test_start_date: Any, test_end_date: Any, expected_months: int) -> EvalBlock:
    test_start_date = pd.Timestamp(test_start_date) + pd.offsets.MonthEnd(0)
    test_end_date = pd.Timestamp(test_end_date) + pd.offsets.MonthEnd(0)
    test_start_pos = v._get_loc_strict(idx, test_start_date, what=f"{tag_prefix}.test_start")
    test_end_pos = v._get_loc_strict(idx, test_end_date, what=f"{tag_prefix}.test_end")
    obs = int(test_end_pos - test_start_pos + 1)
    if obs != int(expected_months):
        print(
            f"[WARN] {tag_prefix}: expected {expected_months} monthly observations in TEST window "
            f"but found {obs}. (Data may have missing months.)"
        )
    train_end = int(test_start_pos - 1)
    if train_end <= 200:
        raise Stage3BError(
            f"Not enough data: train_end={train_end} (need >200). "
            f"Requested TEST window {test_start_date.date()}..{test_end_date.date()} ({expected_months}m)."
        )
    start_idx = train_end
    end_idx = int(test_end_pos)
    H_eval = int(end_idx - start_idx)
    tag = f"{tag_prefix}_{test_start_date.strftime('%Y%m')}_{test_end_date.strftime('%Y%m')}"
    return EvalBlock(tag=tag, train_end=train_end, start_idx=start_idx, end_idx=end_idx, H_eval=H_eval)


def _selection_blocks(v: Any, dataset: Stage3BDataset, config: ResolvedExperimentConfig) -> list[EvalBlock]:
    idx = dataset.r_ex.index
    dev_end = pd.Timestamp(config.split.train_pool_end) + pd.offsets.MonthEnd(0)
    dev_end_pos = v._get_loc_strict(idx, dev_end, what="selection.dev_end")
    dev_len = int(dev_end_pos + 1)

    K = int(config.split.selection_cv_folds)
    min_train_obs = max(int(config.split.selection_min_train_months), 202)
    if min_train_obs >= dev_len:
        raise Stage3BError(
            f"Not enough DEV data: selection_min_train_months={min_train_obs} >= dev_len={dev_len} "
            f"(through {dev_end.date()})."
        )
    val_months_user = int(config.split.selection_val_months or 0)
    if val_months_user > 0:
        val_months = int(val_months_user)
        needed = int(min_train_obs + val_months * K)
        if needed > dev_len:
            raise Stage3BError(
                f"Requested selection_val_months={val_months} with folds={K} and min_train={min_train_obs} "
                f"requires {needed} DEV months, but only {dev_len} are available."
            )
        extra = int(dev_len - needed)
    else:
        val_months = int((dev_len - min_train_obs) // K)
        if val_months <= 0:
            raise Stage3BError(
                f"Not enough DEV data for rolling CV: dev_len={dev_len}, min_train={min_train_obs}, folds={K}."
            )
        extra = int(dev_len - (min_train_obs + val_months * K))

    print("Evaluation plan (stage3b selection): rolling CV folds on DEV (pre-test only)")
    print(f"  DEV  : {idx[0].date()} .. {dev_end.date()}  (T_dev={dev_len} months)")
    print(
        f"  CV   : folds={K} | min_train={min_train_obs}m | val_months={val_months}m | leftover={extra}m"
    )

    blocks: list[EvalBlock] = []
    train_end_pos = int(min_train_obs - 1)
    for i in range(K):
        test_start_pos = int(train_end_pos + 1)
        if val_months_user > 0 or i < K - 1:
            test_end_pos = int(test_start_pos + val_months - 1)
        else:
            test_end_pos = int(dev_end_pos)
        if test_end_pos > dev_end_pos:
            test_end_pos = int(dev_end_pos)
        start_date = idx[int(test_start_pos)]
        end_date = idx[int(test_end_pos)]
        expected_months = int(test_end_pos - test_start_pos + 1)
        block = _make_block(
            v,
            idx,
            tag_prefix=f"valfold{i + 1}",
            test_start_date=start_date,
            test_end_date=end_date,
            expected_months=expected_months,
        )
        blocks.append(block)
        train_end_pos = int(test_end_pos)
    return blocks


def _comparison_block(v: Any, dataset: Stage3BDataset, config: ResolvedExperimentConfig) -> EvalBlock:
    idx = dataset.r_ex.index
    final_test_start = pd.Timestamp(config.split.final_test_start) + pd.offsets.MonthEnd(0)
    end_date = pd.Timestamp(config.split.end_date) + pd.offsets.MonthEnd(0)
    expected_months = int(config.split.final_test_months)
    block = _make_block(
        v,
        idx,
        tag_prefix=f"last{config.split.final_test_years}y",
        test_start_date=final_test_start,
        test_end_date=end_date,
        expected_months=expected_months,
    )
    print(f"Evaluation plan (stage3b comparison): final test only -> years=[{config.split.final_test_years}]")
    return block


def _selection_spec_config(v: Any, config: ResolvedExperimentConfig) -> Any:
    return v.SpecSelectionConfig(
        window_mode=str(config.selection.window_mode),
        rolling_window=int(config.selection.rolling_window),
        return_baseline=str(config.selection.return_baseline),
        state_baseline=str(config.selection.state_baseline),
        alpha=0.25,
        beta=0.50,
        gamma=0.25,
        return_floor=float(config.selection.return_floor),
        state_q10_floor=float(config.selection.state_q10_floor),
        cross_warn=float(config.selection.cross_warn),
        cross_fail=float(config.selection.cross_fail),
        score_mode=str(config.selection.score_mode),
        score_ret_mean_weight=float(config.selection.score_ret_mean_weight),
        score_ret_q10_weight=float(config.selection.score_ret_q10_weight),
        score_state_mean_weight=float(config.selection.score_state_mean_weight),
        score_state_q10_weight=float(config.selection.score_state_q10_weight),
    )


def _build_baselines(
    v: Any,
    dataset: Stage3BDataset,
    block: EvalBlock,
    args: argparse.Namespace,
    out_dir: Path,
    *,
    eval_label: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], Path]:
    start_idx = int(block.start_idx)
    end_idx = int(block.end_idx)
    H_eval = int(block.H_eval)
    train_end = int(block.train_end)

    rf_test = dataset.rf_np[start_idx + 1 : start_idx + 1 + H_eval]

    mkt_rets = dataset.mkt_np[start_idx + 1 : start_idx + 1 + H_eval]
    mkt_metrics = v.compute_metrics(mkt_rets, rf=rf_test, periods_per_year=12, gamma=float(args.gamma))
    print(f"\nBaseline metrics (realized {eval_label}):")
    print("Baseline market (MKT):")
    for k, val in mkt_metrics.items():
        print(f"  {k:7s}: {val: .4f}")

    eq_ex = np.mean(dataset.r_ex_np[start_idx + 1 : start_idx + 1 + H_eval, :], axis=1)
    eq_rets = rf_test + eq_ex
    eq_metrics = v.compute_metrics(eq_rets, rf=rf_test, periods_per_year=12, gamma=float(args.gamma))
    print("Baseline equal-weight:")
    for k, val in eq_metrics.items():
        print(f"  {k:7s}: {val: .4f}")

    bench_list = list(getattr(args, "benchmarks", ["gmv", "risk_parity", "inv_vol"]))
    bench_w_max = getattr(args, "bench_w_max", None)
    rp_max_iter = int(getattr(args, "rp_max_iter", 500))
    rp_tol = float(getattr(args, "rp_tol", 1e-8))

    walk_forward_mode = str(getattr(args, "walk_forward_mode", "expanding" if getattr(args, "expanding_window", False) else "fixed")).lower()
    bench_logs: dict[str, dict[str, Any]] = {}
    if len(bench_list) > 0:
        if walk_forward_mode in {"expanding", "rolling"}:
            bench_refit_every = int(getattr(args, "bench_refit_every", 1))
            bench_logs = v.simulate_realized_expanding_benchmarks_from_returns(
                r_ex=dataset.r_ex_np,
                rf=dataset.rf_np,
                start_idx=start_idx,
                H=H_eval,
                bench_list=bench_list,
                bench_w_max=bench_w_max,
                rp_max_iter=rp_max_iter,
                rp_tol=rp_tol,
                refit_every=bench_refit_every,
                gamma=float(args.gamma),
                window_mode=walk_forward_mode,
                rolling_train_months=getattr(args, "rolling_train_months", None),
            )
        else:
            train_r_ex = dataset.r_ex_np[1 : train_end + 1, :]
            lw = v.LedoitWolf().fit(train_r_ex)
            Sigma_train = lw.covariance_
            mu_train = np.nanmean(train_r_ex, axis=0)

            if "gmv" in bench_list:
                w_gmv, _ = v.gmv_weights_long_only_cash(Sigma_train, ridge=1e-6, w_max=bench_w_max)
                bench_logs["gmv"] = v.simulate_realized_fixed_constant_weights(
                    w_gmv, r_ex=dataset.r_ex_np, rf=dataset.rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                )

            if "risk_parity" in bench_list:
                w_rp, _ = v.risk_parity_weights_long_only_cash(
                    Sigma_train, max_iter=rp_max_iter, tol=rp_tol, ridge=1e-8, w_max=bench_w_max
                )
                bench_logs["risk_parity"] = v.simulate_realized_fixed_constant_weights(
                    w_rp, r_ex=dataset.r_ex_np, rf=dataset.rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                )

            if "inv_vol" in bench_list:
                w_iv, _ = v.inv_vol_weights_long_only_cash(Sigma_train, w_max=bench_w_max)
                bench_logs["inv_vol"] = v.simulate_realized_fixed_constant_weights(
                    w_iv, r_ex=dataset.r_ex_np, rf=dataset.rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                )

            if "static_mvo" in bench_list:
                w_smvo, _ = v.markowitz_weights_long_only_cash(
                    mu_train, Sigma_train, risk_aversion=float(args.gamma), ridge=1e-6, w_max=bench_w_max
                )
                bench_logs["static_mvo"] = v.simulate_realized_fixed_constant_weights(
                    w_smvo, r_ex=dataset.r_ex_np, rf=dataset.rf_np, start_idx=start_idx, H=H_eval, gamma=float(args.gamma)
                )

    if bench_logs:
        if walk_forward_mode == "expanding":
            print("\nExtra baselines (TRAIN-based; refit on expanding window each month):")
        elif walk_forward_mode == "rolling":
            print("\nExtra baselines (TRAIN-based; refit on rolling window each month):")
        else:
            print("\nExtra baselines (constant weights from TRAIN):")
        for name, logs_b in bench_logs.items():
            print(f"Baseline {name}:")
            for k, val in logs_b["metrics"].items():
                print(f"  {k:7s}: {val: .4f}")

    dates = dataset.r_ex.index[start_idx + 1 : start_idx + 1 + H_eval]
    baseline_df = pd.DataFrame(index=dates)
    baseline_df["rf"] = rf_test
    baseline_df["mkt_ret"] = mkt_rets
    baseline_df["mkt_wealth"] = np.cumprod(1.0 + mkt_rets)
    baseline_df["eq_ret"] = eq_rets
    baseline_df["eq_wealth"] = np.cumprod(1.0 + eq_rets)

    if "gmv" in bench_logs:
        baseline_df["gmv_ret"] = bench_logs["gmv"]["port_rets"]
        baseline_df["gmv_wealth"] = bench_logs["gmv"]["wealth"][1:]
        baseline_df["gmv_turnover"] = bench_logs["gmv"]["turnover"]
        baseline_df["gmv_cash"] = bench_logs["gmv"]["cash"]
    if "risk_parity" in bench_logs:
        baseline_df["rp_ret"] = bench_logs["risk_parity"]["port_rets"]
        baseline_df["rp_wealth"] = bench_logs["risk_parity"]["wealth"][1:]
        baseline_df["rp_turnover"] = bench_logs["risk_parity"]["turnover"]
        baseline_df["rp_cash"] = bench_logs["risk_parity"]["cash"]
    if "inv_vol" in bench_logs:
        baseline_df["inv_vol_ret"] = bench_logs["inv_vol"]["port_rets"]
        baseline_df["inv_vol_wealth"] = bench_logs["inv_vol"]["wealth"][1:]
        baseline_df["inv_vol_turnover"] = bench_logs["inv_vol"]["turnover"]
        baseline_df["inv_vol_cash"] = bench_logs["inv_vol"]["cash"]
    if "static_mvo" in bench_logs:
        baseline_df["static_mvo_ret"] = bench_logs["static_mvo"]["port_rets"]
        baseline_df["static_mvo_wealth"] = bench_logs["static_mvo"]["wealth"][1:]
        baseline_df["static_mvo_turnover"] = bench_logs["static_mvo"]["turnover"]
        baseline_df["static_mvo_cash"] = bench_logs["static_mvo"]["cash"]

    baseline_path = out_dir / f"baselines_{block.tag}.csv"
    baseline_df.to_csv(baseline_path)
    print(f"Saved: {baseline_path}")

    baseline_metrics: dict[str, dict[str, float]] = {"mkt": mkt_metrics, "eq": eq_metrics}
    for name, obj in bench_logs.items():
        baseline_metrics[name] = obj["metrics"]
    return baseline_df, baseline_metrics, baseline_path


def _result_rows(
    result: dict[str, Any],
    *,
    block_tag: str,
    selected_spec: str,
    baseline_metrics: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(strategy: str, metrics: dict[str, Any], label: str | None = None, extra: dict[str, Any] | None = None) -> None:
        row = {
            "block": block_tag,
            "spec": selected_spec,
            "strategy": strategy,
            "strategy_label": label or strategy,
        }
        row.update(metrics)
        if extra:
            row.update(extra)
        rows.append(row)

    for name, metrics in baseline_metrics.items():
        add_row(name, metrics, label=name)

    if isinstance(result.get("policy"), dict):
        add_row("policy", result["policy"], label="Policy")
    if isinstance(result.get("myopic"), dict):
        add_row("myopic", result["myopic"], label="Myopic")
    if isinstance(result.get("ppgdpo"), dict):
        add_row("ppgdpo", result["ppgdpo"], label="PPGDPO")

    for name in ["gmv_model", "risk_parity_model", "inv_vol_model", "static_mvo_model"]:
        if isinstance(result.get(name), dict):
            add_row(name, result[name], label=name)

    variants = result.get("ppgdpo_variants", {})
    if isinstance(variants, dict):
        for name, metrics in variants.items():
            if isinstance(metrics, dict):
                add_row(str(name), metrics, label=str(metrics.get("label", name)))

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_cols = [c for c in ["cer_ann", "sharpe", "ann_ret"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return df


def _write_tc_summaries(config: ResolvedExperimentConfig, *, selected_spec: str) -> dict[str, Path | None]:
    artifacts = discover_comparison_artifacts(config, selected_spec=selected_spec)
    written: dict[str, Path | None] = {
        "comparison_all_costs_summary_csv": None,
        "comparison_zero_cost_summary_csv": None,
    }
    if artifacts.tc_sweep_csv is None:
        return written
    tc_df = normalize_tc_sweep_csv(artifacts.tc_sweep_csv)
    tc_df.to_csv(config.comparison_all_costs_summary_path, index=False)
    written["comparison_all_costs_summary_csv"] = config.comparison_all_costs_summary_path
    zero_df = tc_df[tc_df["tc_bps"].fillna(-1).eq(0)].copy()
    if not zero_df.empty:
        sort_cols = [c for c in ["cer_ann", "sharpe", "ann_ret"] if c in zero_df.columns]
        if sort_cols:
            zero_df = zero_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
        zero_df.to_csv(config.comparison_zero_cost_summary_path, index=False)
        written["comparison_zero_cost_summary_csv"] = config.comparison_zero_cost_summary_path
    return written


def _write_stage3b_selection_report(
    config: ResolvedExperimentConfig,
    *,
    selected_specs: list[str],
    log_path: Path,
    command: str,
    blocks: list[EvalBlock],
) -> Path:
    artifacts = discover_selection_artifacts(config)
    payload = {
        "experiment": config.experiment.name,
        "backend": "native_stage3b",
        "phase": "selection",
        "selected_specs": selected_specs,
        "primary_selected_spec": selected_specs[0] if selected_specs else None,
        "summary_csv": str(artifacts.summary_csv) if artifacts.summary_csv else None,
        "blocks_csv": str(artifacts.blocks_csv) if artifacts.blocks_csv else None,
        "log_path": str(log_path),
        "command": command,
        "selection_blocks": [block.tag for block in blocks],
    }
    path = config.stage3b_selection_report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _write_stage3b_comparison_report(
    config: ResolvedExperimentConfig,
    *,
    selected_spec: str,
    log_path: Path,
    command: str,
    summary_df: pd.DataFrame,
    artifacts_written: dict[str, Path | None],
) -> Path:
    artifacts = discover_comparison_artifacts(config, selected_spec=selected_spec)
    monthly_summary = summarize_monthly_csv(artifacts.monthly_csv) if artifacts.monthly_csv is not None else None
    payload = {
        "experiment": config.experiment.name,
        "backend": "native_stage3b",
        "phase": "comparison",
        "selected_spec": selected_spec,
        "returncode": 0,
        "log_path": str(log_path),
        "command": command,
        "artifacts": {
            "monthly_csv": str(artifacts.monthly_csv) if artifacts.monthly_csv else None,
            "tc_sweep_csv": str(artifacts.tc_sweep_csv) if artifacts.tc_sweep_csv else None,
            "baselines_csv": str(artifacts.baselines_csv) if artifacts.baselines_csv else None,
            "policy_pt": str(artifacts.policy_pt) if artifacts.policy_pt else None,
            "comparison_summary_from_log_csv": str(config.comparison_summary_from_log_path) if config.comparison_summary_from_log_path.exists() else None,
            "comparison_all_costs_summary_csv": str(artifacts_written.get("comparison_all_costs_summary_csv")) if artifacts_written.get("comparison_all_costs_summary_csv") else None,
            "comparison_zero_cost_summary_csv": str(artifacts_written.get("comparison_zero_cost_summary_csv")) if artifacts_written.get("comparison_zero_cost_summary_csv") else None,
        },
        "monthly_summary": monthly_summary,
        "summary_rows": int(len(summary_df)),
    }
    path = config.stage3b_comparison_report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _selection_phase_native(
    config: ResolvedExperimentConfig,
    ctx: VendorContext,
) -> tuple[list[str], Stage3BPhaseSnapshot]:
    candidate_specs = resolve_candidate_specs(config)
    args = _build_args(
        config,
        phase="selection",
        state_spec=candidate_specs[0],
        specs=candidate_specs,
        out_dir=config.selection_output_dir,
    )
    dataset = _get_or_prepare_dataset(ctx, args, candidate_specs)
    blocks = _selection_blocks(ctx.module, dataset, config)
    sel_cfg = _selection_spec_config(ctx.module, config)
    command = build_selection_run_spec(config).shell_command(redact_secrets=True)
    log_path = _selection_log_path(config)

    with _pushd(ctx.workdir), _tee_output(log_path):
        print(f"# stage3b backend: selection\n# cwd: {ctx.workdir}\n# command: {command}\n")
        print("\n" + "=" * 80)
        print("SPEC SELECTION (method-free; predictive R^2 / stability / cross guard)")
        print("=" * 80)
        print(f"Selection blocks: {[b.tag for b in blocks]}")
        print(f"Candidate specs : {candidate_specs}")
        sel_results = []
        for blk in blocks:
            for spec in candidate_specs:
                try:
                    sel_results.append(
                        ctx.module.evaluate_spec_predictive_diagnostics(
                            spec=str(spec),
                            block=str(blk.tag),
                            r_ex=dataset.r_ex,
                            macro3=dataset.macro3,
                            macro7=dataset.macro7,
                            ff3=dataset.ff3_state,
                            ff5=dataset.ff5_state,
                            bond_asset_names=dataset.bond_asset_names,
                            train_end=int(blk.train_end),
                            start_idx=int(blk.start_idx),
                            end_idx=int(blk.end_idx),
                            pca_cfg=dataset.pca_cfg,
                            pls_horizon=int(args.pls_horizon),
                            pls_smooth_span=int(args.pls_smooth_span),
                            block_eq_k=int(getattr(args, "block_eq_k", 1)),
                            block_bond_k=int(getattr(args, "block_bond_k", 1)),
                            config=sel_cfg,
                        )
                    )
                except Exception as exc:
                    print(f"[WARN] Spec-selection failed for spec={spec} block={blk.tag}: {exc}")

        df_sel = ctx.module.results_to_frame(sel_results)
        df_rank = ctx.module.rank_specs_from_results(df_sel, config=sel_cfg)

        out_dir_root = config.selection_output_dir
        out_dir_root.mkdir(parents=True, exist_ok=True)
        sel_path = out_dir_root / "spec_selection_blocks.csv"
        rank_path = out_dir_root / "spec_selection_summary.csv"
        if not df_sel.empty:
            df_sel.to_csv(sel_path, index=False)
            print(f"Saved: {sel_path}")
        if not df_rank.empty:
            df_rank.to_csv(rank_path, index=False)
            print(f"Saved: {rank_path}")
            show_cols = [
                c for c in [
                    "spec",
                    "score",
                    "passes_guard",
                    "recommended",
                    "r2_oos_ret_mean",
                    "r2_roll_ret_q10",
                    "r2_oos_state_mean",
                    "r2_roll_state_q10",
                    "cross_max_abs_rho",
                ] if c in df_rank.columns
            ]
            print("\nSpec ranking summary:")
            with pd.option_context("display.max_rows", None, "display.width", 160, "display.max_colwidth", 40):
                print(df_rank[show_cols].to_string(index=False))
        else:
            raise Stage3BError("Selection did not produce any valid ranking rows.")

    selected_specs = choose_selected_specs(config)
    write_selected_spec_artifact(config, selected_specs)
    report_path = _write_stage3b_selection_report(
        config,
        selected_specs=selected_specs,
        log_path=log_path,
        command=command,
        blocks=blocks,
    )
    snapshot = Stage3BPhaseSnapshot(
        phase="selection",
        command=command,
        log_path=str(log_path),
        returncode=0,
        artifacts={
            "selection_summary_csv": str(config.selection_output_dir / "spec_selection_summary.csv"),
            "selection_blocks_csv": str(config.selection_output_dir / "spec_selection_blocks.csv"),
            "selected_spec_yaml": str(config.selected_spec_path),
            "stage3b_selection_report_yaml": str(report_path),
        },
    )
    return selected_specs, snapshot


def _comparison_phase_native(
    config: ResolvedExperimentConfig,
    ctx: VendorContext,
    *,
    selected_spec: str,
) -> tuple[str, list[str], Stage3BPhaseSnapshot]:
    args = _build_args(
        config,
        phase="comparison",
        state_spec=selected_spec,
        specs=[selected_spec],
        out_dir=config.comparison_output_dir,
    )
    dataset = _get_or_prepare_dataset(ctx, args, [selected_spec])
    block = _comparison_block(ctx.module, dataset, config)
    command = build_comparison_run_spec(config, selected_spec=selected_spec).shell_command(redact_secrets=True)
    log_path = _comparison_log_path(config, selected_spec)
    warnings: list[str] = []

    with _pushd(ctx.workdir), _tee_output(log_path):
        print(f"# stage3b backend: comparison\n# cwd: {ctx.workdir}\n# command: {command}\n")
        train_horizon_req = int(getattr(args, "train_horizon", 0))
        if train_horizon_req <= 0:
            train_horizon_req = int(block.H_eval)
        H_train_blk = int(min(int(train_horizon_req), int(block.H_eval)))

        train_start_date = dataset.r_ex.index[0]
        train_end_date = dataset.r_ex.index[block.train_end]
        test_start_date = dataset.r_ex.index[block.start_idx + 1]
        test_end_date = dataset.r_ex.index[block.end_idx]
        eval_label = "TEST"

        out_dir = config.comparison_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(
            f"EVAL {block.tag} | "
            f"TRAIN {train_start_date.date()} .. {train_end_date.date()} ({block.train_end + 1} obs) | "
            f"{eval_label} {test_start_date.date()} .. {test_end_date.date()} ({block.H_eval} obs) | "
            f"H_train={H_train_blk}"
        )
        print("=" * 80)

        baseline_df, baseline_metrics, baseline_path = _build_baselines(
            ctx.module,
            dataset,
            block,
            args,
            out_dir,
            eval_label=eval_label,
        )

        result = ctx.module.run_one_state_spec(
            spec=selected_spec,
            r_ex=dataset.r_ex,
            rf=dataset.rf,
            macro3=dataset.macro3,
            macro7=dataset.macro7,
            ff3=dataset.ff3_state,
            ff5=dataset.ff5_state,
            bond_asset_names=dataset.bond_asset_names,
            train_end=int(block.train_end),
            start_idx=int(block.start_idx),
            H_train=int(H_train_blk),
            H_eval=int(block.H_eval),
            r_ex_np=dataset.r_ex_np,
            rf_np=dataset.rf_np,
            baseline_df=baseline_df,
            args=args,
            device=dataset.device,
            dtype=dataset.dtype,
            pca_cfg=dataset.pca_cfg,
            out_dir=out_dir,
            eval_label=eval_label,
        )

        result["block"] = block.tag
        result["eval_label"] = eval_label
        result["test_start"] = str(test_start_date.date())
        result["test_end"] = str(test_end_date.date())
        result.update(baseline_metrics)

    summary_df = _result_rows(
        result,
        block_tag=block.tag,
        selected_spec=selected_spec,
        baseline_metrics=baseline_metrics,
    )
    if not summary_df.empty:
        summary_df.to_csv(config.comparison_summary_from_log_path, index=False)

    artifacts_written = _write_tc_summaries(config, selected_spec=selected_spec)
    report_path = _write_stage3b_comparison_report(
        config,
        selected_spec=selected_spec,
        log_path=log_path,
        command=command,
        summary_df=summary_df,
        artifacts_written=artifacts_written,
    )
    artifacts = discover_comparison_artifacts(config, selected_spec=selected_spec)
    snapshot = Stage3BPhaseSnapshot(
        phase="comparison",
        command=command,
        log_path=str(log_path),
        returncode=0,
        artifacts={
            "baselines_csv": str(artifacts.baselines_csv) if artifacts.baselines_csv else str(baseline_path),
            "monthly_csv": str(artifacts.monthly_csv) if artifacts.monthly_csv else None,
            "tc_sweep_csv": str(artifacts.tc_sweep_csv) if artifacts.tc_sweep_csv else None,
            "policy_pt": str(artifacts.policy_pt) if artifacts.policy_pt else None,
            "comparison_summary_csv": str(config.comparison_summary_from_log_path) if config.comparison_summary_from_log_path.exists() else None,
            "comparison_all_costs_summary_csv": str(artifacts_written.get("comparison_all_costs_summary_csv")) if artifacts_written.get("comparison_all_costs_summary_csv") else None,
            "comparison_zero_cost_summary_csv": str(artifacts_written.get("comparison_zero_cost_summary_csv")) if artifacts_written.get("comparison_zero_cost_summary_csv") else None,
            "stage3b_comparison_report_yaml": str(report_path),
        },
    )
    return selected_spec, warnings, snapshot


def write_stage3b_manifest(
    config: ResolvedExperimentConfig,
    *,
    completed_phases: list[str],
    selected_specs: list[str],
    warnings: list[str],
    snapshots: list[Stage3BPhaseSnapshot],
) -> None:
    manifest = Stage3BManifest(
        experiment=config.experiment.name,
        backend="native_stage3b",
        completed_phases=completed_phases,
        selected_specs=selected_specs,
        primary_selected_spec=selected_specs[0] if selected_specs else None,
        warnings=warnings,
        snapshots=snapshots,
    )
    manifest.dump(config.stage3b_manifest_path)


def run_stage3b(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    dry_run: bool = False,
    selected_spec: str | None = None,
) -> Stage3BRunResult | list[LegacyRunSpec]:
    if dry_run:
        chosen = selected_spec
        if phase in {"comparison", "all"} and chosen is None and config.comparison.enabled:
            try:
                chosen = resolve_comparison_spec(config, selected_spec=selected_spec)
            except Exception:
                chosen = None
        return _phase_specs_for_render(config, phase=phase, selected_spec=chosen)

    ctx = _import_vendor_module(config)

    completed_phases: list[str] = []
    warnings: list[str] = []
    selected_specs: list[str] = []
    snapshots: list[Stage3BPhaseSnapshot] = []

    if phase in {"selection", "all"} and config.selection.enabled:
        picked, snapshot = _selection_phase_native(config, ctx)
        selected_specs = picked
        completed_phases.append("selection")
        snapshots.append(snapshot)
    if phase in {"comparison", "all"} and config.comparison.enabled:
        if selected_spec is None:
            from .legacy_bridge import resolve_comparison_spec
            selected_spec = resolve_comparison_spec(config)
        final_spec, phase_warnings, snapshot = _comparison_phase_native(
            config,
            ctx,
            selected_spec=selected_spec,
        )
        if not selected_specs:
            selected_specs = [final_spec]
        warnings.extend(phase_warnings)
        completed_phases.append("comparison")
        snapshots.append(snapshot)

    write_stage3b_manifest(
        config,
        completed_phases=completed_phases,
        selected_specs=selected_specs,
        warnings=warnings,
        snapshots=snapshots,
    )
    return Stage3BRunResult(
        selected_spec=selected_specs[0] if selected_specs else None,
        selected_specs=selected_specs,
        completed_phases=completed_phases,
        warnings=warnings,
    )


def artifact_snapshot(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    selected_spec: str | None = None,
) -> str:
    if selected_spec is None and config.selected_spec_path.exists():
        try:
            payload = yaml.safe_load(config.selected_spec_path.read_text()) or {}
            selected_spec = payload.get("primary_selected_spec") or payload.get("selected_spec")
        except Exception:
            selected_spec = None
    if selected_spec is None and config.comparison.fixed_spec:
        selected_spec = config.comparison.fixed_spec
    return render_artifact_snapshot(config, phase=phase, selected_spec=selected_spec)
