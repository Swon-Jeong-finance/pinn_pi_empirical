
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import yaml

from . import stage3b_backend as base
from .legacy_bridge import LegacyRunSpec
from .macro_pool import build_macro_pool_monthly, describe_macro_pool
from .schema import ResolvedExperimentConfig

Stage4Error = base.Stage3BError
Stage4RunResult = base.Stage3BRunResult


@contextlib.contextmanager
def _patched_stage3b(config: ResolvedExperimentConfig):
    original_build_args = base._build_args
    original_prepare_dataset = base._prepare_dataset

    def patched_build_args(
        cfg: ResolvedExperimentConfig,
        *,
        phase: str,
        state_spec: str,
        specs: list[str],
        out_dir: Path,
    ):
        args = original_build_args(cfg, phase=phase, state_spec=state_spec, specs=specs, out_dir=out_dir)
        bond_assets = cfg.universe.normalized_bond_assets
        args.include_bond = bool(cfg.universe.include_bond and len(bond_assets) > 0)
        if bond_assets:
            args.bond_source = str(cfg.universe.bond_source_kind or bond_assets[0].source)
            first = bond_assets[0]
            args.bond_name = str(first.name)
            if first.source == "crsp_csv":
                args.bond_csv = str(first.csv)
                args.bond_ret_col = str(first.ret_col)
                args.bond_csv_specs = ",".join(
                    f"{asset.name}={asset.csv}@{asset.ret_col}" for asset in bond_assets
                )
                args.bond_fred_specs = ""
            else:
                args.bond_series_id = str(first.series_id)
                args.bond_csv_specs = ""
                args.bond_fred_specs = ",".join(
                    f"{asset.name}={asset.series_id}" for asset in bond_assets
                )
        else:
            args.bond_csv_specs = ""
            args.bond_fred_specs = ""
        args.stage4_macro_pool = str(cfg.macro.pool)
        args.stage4_macro_features = list(cfg.macro.effective_feature_ids)
        args.stage4_macro3_columns = list(cfg.macro.macro3_columns)
        return args

    def patched_prepare_dataset(v: Any, args: Any, specs: list[str]):
        original_builder = getattr(v, "build_macro7_monthly")
        original_builder3 = getattr(v, "build_macro3_monthly")

        def stage4_builder(fred_cfg: Any):
            print(f"[stage4] macro pool: {describe_macro_pool(config.macro)}")
            if config.macro.pool == "legacy7" and not config.macro.feature_ids:
                return original_builder(fred_cfg)
            return build_macro_pool_monthly(fred_cfg=fred_cfg, macro_config=config.macro)

        def stage4_builder3(fred_cfg: Any):
            if config.macro.pool == "legacy7" and not config.macro.feature_ids:
                return original_builder3(fred_cfg)
            macro = build_macro_pool_monthly(fred_cfg=fred_cfg, macro_config=config.macro)
            return macro[list(config.macro.macro3_columns)].copy()

        setattr(v, "build_macro7_monthly", stage4_builder)
        setattr(v, "build_macro3_monthly", stage4_builder3)
        try:
            return original_prepare_dataset(v, args, specs)
        finally:
            setattr(v, "build_macro7_monthly", original_builder)
            setattr(v, "build_macro3_monthly", original_builder3)

    base._build_args = patched_build_args
    base._prepare_dataset = patched_prepare_dataset
    try:
        yield
    finally:
        base._build_args = original_build_args
        base._prepare_dataset = original_prepare_dataset


def _copy_yaml_with_backend(src: Path, dst: Path, backend: str, extra: dict[str, Any] | None = None) -> None:
    if not src.exists():
        return
    payload = yaml.safe_load(src.read_text()) or {}
    if not isinstance(payload, dict):
        return
    payload["backend"] = backend
    if extra:
        payload.update(extra)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_stage4_manifest(config: ResolvedExperimentConfig, result: Stage4RunResult) -> None:
    payload = {
        "experiment": config.experiment.name,
        "backend": "native_stage4",
        "completed_phases": list(result.completed_phases),
        "selected_specs": list(result.selected_specs),
        "primary_selected_spec": result.selected_spec,
        "warnings": list(result.warnings),
        "macro_pool": config.macro.pool,
        "macro_features": list(config.macro.effective_feature_ids),
        "bond_assets": [asset.model_dump(mode="json") for asset in config.universe.normalized_bond_assets],
        "selection_report": str(config.stage4_selection_report_path) if config.stage4_selection_report_path.exists() else None,
        "comparison_report": str(config.stage4_comparison_report_path) if config.stage4_comparison_report_path.exists() else None,
        "stage3b_manifest_path": str(config.stage3b_manifest_path) if config.stage3b_manifest_path.exists() else None,
    }
    config.stage4_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.stage4_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def run_stage4(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    dry_run: bool = False,
    selected_spec: str | None = None,
) -> Stage4RunResult | list[LegacyRunSpec]:
    if dry_run:
        return base.run_stage3b(config, phase=phase, dry_run=True, selected_spec=selected_spec)

    with _patched_stage3b(config):
        result = base.run_stage3b(config, phase=phase, dry_run=False, selected_spec=selected_spec)

    extra = {
        "macro_pool": config.macro.pool,
        "macro_features": list(config.macro.effective_feature_ids),
        "bond_assets": [asset.model_dump(mode="json") for asset in config.universe.normalized_bond_assets],
    }
    _copy_yaml_with_backend(
        config.stage3b_selection_report_path,
        config.stage4_selection_report_path,
        "native_stage4",
        extra,
    )
    _copy_yaml_with_backend(
        config.stage3b_comparison_report_path,
        config.stage4_comparison_report_path,
        "native_stage4",
        extra,
    )
    _write_stage4_manifest(config, result)
    return result


def artifact_snapshot(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    selected_spec: str | None = None,
) -> str:
    return base.artifact_snapshot(config, phase=phase, selected_spec=selected_spec)
