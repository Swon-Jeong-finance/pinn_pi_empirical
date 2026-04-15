from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import stage5_backend as base
from .legacy_bridge import LegacyRunSpec, resolve_comparison_spec
from .schema import ResolvedExperimentConfig

Stage6Error = base.Stage5Error


@dataclass(frozen=True)
class Stage6RunResult:
    selected_spec: str | None
    selected_specs: list[str]
    completed_phases: list[str]
    warnings: list[str]


def _copy_yaml_with_backend(
    src: Path, dst: Path, backend: str, extra: dict[str, Any] | None = None
) -> None:
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


def _effective_cross_modes(config: ResolvedExperimentConfig) -> list[str]:
    modes = list(config.comparison.cross_modes)
    if not modes:
        modes = [str(config.model.cross_mode)]
    out: list[str] = []
    seen: set[str] = set()
    for mode in modes:
        s = str(mode)
        if s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _comparison_config_for_mode(
    config: ResolvedExperimentConfig,
    *,
    mode: str,
    multi_mode: bool,
) -> ResolvedExperimentConfig:
    cfg = config.model_copy(deep=True)
    cfg.model.cross_mode = mode
    base_subdir = str(config.comparison.output_subdir).rstrip("/") or "comparison"
    cfg.comparison.output_subdir = (
        f"{base_subdir}/{mode}" if multi_mode else base_subdir
    )
    return cfg


def _mode_artifacts(cfg: ResolvedExperimentConfig) -> dict[str, str | None]:
    stage3b_report = None
    stage3b_log_path = None
    if cfg.stage3b_comparison_report_path.exists():
        stage3b_report = str(cfg.stage3b_comparison_report_path)
        payload = yaml.safe_load(cfg.stage3b_comparison_report_path.read_text()) or {}
        if isinstance(payload, dict):
            raw_log_path = payload.get("log_path")
            if raw_log_path:
                stage3b_log_path = str(raw_log_path)
    payload = {
        "comparison_output_dir": str(cfg.comparison_output_dir),
        "comparison_summary_from_log_csv": str(cfg.comparison_summary_from_log_path)
        if cfg.comparison_summary_from_log_path.exists()
        else None,
        "comparison_zero_cost_summary_csv": str(cfg.comparison_zero_cost_summary_path)
        if cfg.comparison_zero_cost_summary_path.exists()
        else None,
        "comparison_all_costs_summary_csv": str(cfg.comparison_all_costs_summary_path)
        if cfg.comparison_all_costs_summary_path.exists()
        else None,
        "stage3b_comparison_report_yaml": stage3b_report,
        "stage3b_log_path": stage3b_log_path,
        "stage5_comparison_report_yaml": str(cfg.stage5_comparison_report_path)
        if cfg.stage5_comparison_report_path.exists()
        else None,
    }
    return payload


def _concat_mode_summaries(
    config: ResolvedExperimentConfig,
    *,
    mode_configs: list[tuple[str, ResolvedExperimentConfig]],
) -> dict[str, str | None]:
    zero_frames: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []
    for mode, cfg in mode_configs:
        if cfg.comparison_zero_cost_summary_path.exists():
            df = pd.read_csv(cfg.comparison_zero_cost_summary_path)
            df.insert(0, "cross_mode", mode)
            zero_frames.append(df)
        if cfg.comparison_all_costs_summary_path.exists():
            df = pd.read_csv(cfg.comparison_all_costs_summary_path)
            df.insert(0, "cross_mode", mode)
            all_frames.append(df)

    written: dict[str, str | None] = {
        "comparison_cross_modes_zero_cost_summary_csv": None,
        "comparison_cross_modes_all_costs_summary_csv": None,
    }
    if zero_frames:
        zero_df = pd.concat(zero_frames, ignore_index=True)
        config.comparison_cross_modes_zero_cost_summary_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        zero_df.to_csv(config.comparison_cross_modes_zero_cost_summary_path, index=False)
        written["comparison_cross_modes_zero_cost_summary_csv"] = str(
            config.comparison_cross_modes_zero_cost_summary_path
        )
    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)
        config.comparison_cross_modes_all_costs_summary_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        all_df.to_csv(config.comparison_cross_modes_all_costs_summary_path, index=False)
        written["comparison_cross_modes_all_costs_summary_csv"] = str(
            config.comparison_cross_modes_all_costs_summary_path
        )
    return written


def _write_stage6_manifest(
    config: ResolvedExperimentConfig,
    *,
    completed_phases: list[str],
    selected_specs: list[str],
    warnings: list[str],
    cross_modes: list[str],
    mode_reports: dict[str, dict[str, str | None]],
) -> None:
    payload = {
        "experiment": config.experiment.name,
        "backend": "native_stage6",
        "completed_phases": list(completed_phases),
        "selected_specs": list(selected_specs),
        "primary_selected_spec": selected_specs[0] if selected_specs else None,
        "comparison": {
            "spec_source": config.comparison.spec_source,
            "selected_rank": config.comparison.selected_rank,
            "cross_modes": list(cross_modes),
        },
        "warnings": list(warnings),
        "selection_report": str(config.stage6_selection_report_path)
        if config.stage6_selection_report_path.exists()
        else None,
        "comparison_report": str(config.stage6_comparison_report_path)
        if config.stage6_comparison_report_path.exists()
        else None,
        "mode_reports": mode_reports,
    }
    config.stage6_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.stage6_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_stage6_comparison_report(
    config: ResolvedExperimentConfig,
    *,
    selected_spec: str,
    mode_reports: dict[str, dict[str, str | None]],
    combined: dict[str, str | None],
    warnings: list[str],
) -> None:
    payload = {
        "experiment": config.experiment.name,
        "backend": "native_stage6",
        "phase": "comparison",
        "selected_spec": selected_spec,
        "selected_rank": config.comparison.selected_rank,
        "spec_source": config.comparison.spec_source,
        "cross_modes": list(_effective_cross_modes(config)),
        "per_mode": mode_reports,
        "combined_summaries": combined,
        "warnings": list(warnings),
    }
    config.stage6_comparison_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.stage6_comparison_report_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_stage6_selection_report(
    config: ResolvedExperimentConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    _copy_yaml_with_backend(
        config.stage5_selection_report_path,
        config.stage6_selection_report_path,
        "native_stage6",
        extra,
    )


def run_stage6(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    dry_run: bool = False,
    selected_spec: str | None = None,
    selected_rank: int | None = None,
) -> Stage6RunResult | list[LegacyRunSpec]:
    rank = config.comparison.selected_rank if selected_rank is None else int(selected_rank)
    cross_modes = _effective_cross_modes(config)
    multi_mode = len(cross_modes) > 1

    if dry_run:
        rendered: list[LegacyRunSpec] = []
        if phase in {"selection", "all"} and config.selection.enabled:
            rendered.extend(base.run_stage5(config, phase="selection", dry_run=True))
        chosen = selected_spec
        if phase in {"comparison", "all"} and config.comparison.enabled:
            if chosen is None:
                try:
                    chosen = resolve_comparison_spec(config, selected_spec=None, selected_rank=rank)
                except Exception:
                    chosen = None
            if chosen is not None:
                for mode in cross_modes:
                    cfg_mode = _comparison_config_for_mode(
                        config, mode=mode, multi_mode=multi_mode
                    )
                    rendered.extend(
                        base.run_stage5(
                            cfg_mode,
                            phase="comparison",
                            dry_run=True,
                            selected_spec=chosen,
                        )
                    )
        return rendered

    completed_phases: list[str] = []
    warnings: list[str] = []
    selected_specs: list[str] = []

    if phase in {"selection", "all"} and config.selection.enabled:
        sel_result = base.run_stage5(config, phase="selection", dry_run=False)
        completed_phases.extend(sel_result.completed_phases)
        selected_specs = list(sel_result.selected_specs)
        warnings.extend(getattr(sel_result, "warnings", []))
        _write_stage6_selection_report(
            config,
            extra={
                "selected_rank": rank,
                "comparison_cross_modes": cross_modes,
            },
        )

    chosen = selected_spec
    if phase in {"comparison", "all"} and config.comparison.enabled:
        if chosen is None:
            chosen = resolve_comparison_spec(config, selected_spec=None, selected_rank=rank)
        mode_reports: dict[str, dict[str, str | None]] = {}
        mode_cfgs: list[tuple[str, ResolvedExperimentConfig]] = []
        for mode in cross_modes:
            cfg_mode = _comparison_config_for_mode(config, mode=mode, multi_mode=multi_mode)
            mode_cfgs.append((mode, cfg_mode))
            cmp_result = base.run_stage5(
                cfg_mode,
                phase="comparison",
                dry_run=False,
                selected_spec=chosen,
            )
            warnings.extend(getattr(cmp_result, "warnings", []))
            mode_reports[mode] = _mode_artifacts(cfg_mode)
        combined = _concat_mode_summaries(config, mode_configs=mode_cfgs)
        _write_stage6_comparison_report(
            config,
            selected_spec=chosen,
            mode_reports=mode_reports,
            combined=combined,
            warnings=warnings,
        )
        completed_phases.append("comparison")
        if not selected_specs:
            selected_specs = [chosen]
        elif chosen not in selected_specs:
            selected_specs.insert(0, chosen)
        _write_stage6_manifest(
            config,
            completed_phases=completed_phases,
            selected_specs=selected_specs,
            warnings=warnings,
            cross_modes=cross_modes,
            mode_reports=mode_reports,
        )
    else:
        _write_stage6_manifest(
            config,
            completed_phases=completed_phases,
            selected_specs=selected_specs,
            warnings=warnings,
            cross_modes=cross_modes,
            mode_reports={},
        )

    return Stage6RunResult(
        selected_spec=selected_specs[0] if selected_specs else chosen,
        selected_specs=selected_specs,
        completed_phases=completed_phases,
        warnings=warnings,
    )


def artifact_snapshot(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    selected_spec: str | None = None,
    selected_rank: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "experiment": config.experiment.name,
        "backend": "native_stage6",
        "phase": phase,
        "selected_rank": config.comparison.selected_rank if selected_rank is None else int(selected_rank),
        "selected_spec_override": selected_spec,
        "selection_report": str(config.stage6_selection_report_path)
        if config.stage6_selection_report_path.exists()
        else None,
        "comparison_report": str(config.stage6_comparison_report_path)
        if config.stage6_comparison_report_path.exists()
        else None,
        "stage6_manifest": str(config.stage6_manifest_path)
        if config.stage6_manifest_path.exists()
        else None,
        "comparison_cross_modes_zero_cost_summary": str(config.comparison_cross_modes_zero_cost_summary_path)
        if config.comparison_cross_modes_zero_cost_summary_path.exists()
        else None,
        "comparison_cross_modes_all_costs_summary": str(config.comparison_cross_modes_all_costs_summary_path)
        if config.comparison_cross_modes_all_costs_summary_path.exists()
        else None,
        "cross_modes": _effective_cross_modes(config),
    }
    return yaml.safe_dump(payload, sort_keys=False)
