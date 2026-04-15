from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import stage4_backend as base
from .selection import resolve_candidate_specs
from .schema import ResolvedExperimentConfig

Stage5Error = base.Stage4Error
Stage5RunResult = base.Stage4RunResult


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


def _write_stage5_manifest(config: ResolvedExperimentConfig, result: Stage5RunResult) -> None:
    payload = {
        "experiment": config.experiment.name,
        "backend": "native_stage5",
        "completed_phases": list(result.completed_phases),
        "selected_specs": list(result.selected_specs),
        "primary_selected_spec": result.selected_spec,
        "warnings": list(result.warnings),
        "selection_candidate_mode": config.selection.candidate_mode,
        "selection_candidate_count": len(resolve_candidate_specs(config)),
        "selection_report": str(config.stage5_selection_report_path) if config.stage5_selection_report_path.exists() else None,
        "comparison_report": str(config.stage5_comparison_report_path) if config.stage5_comparison_report_path.exists() else None,
        "stage4_manifest_path": str(config.stage4_manifest_path) if config.stage4_manifest_path.exists() else None,
    }
    config.stage5_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.stage5_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def run_stage5(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    dry_run: bool = False,
    selected_spec: str | None = None,
) -> Stage5RunResult | list[Any]:
    if dry_run:
        return base.run_stage4(config, phase=phase, dry_run=True, selected_spec=selected_spec)

    result = base.run_stage4(config, phase=phase, dry_run=False, selected_spec=selected_spec)

    extra = {
        "selection_candidate_mode": config.selection.candidate_mode,
        "selection_candidate_count": len(resolve_candidate_specs(config)),
    }
    _copy_yaml_with_backend(
        config.stage4_selection_report_path,
        config.stage5_selection_report_path,
        "native_stage5",
        extra,
    )
    _copy_yaml_with_backend(
        config.stage4_comparison_report_path,
        config.stage5_comparison_report_path,
        "native_stage5",
        extra,
    )
    _write_stage5_manifest(config, result)
    return result


def artifact_snapshot(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    selected_spec: str | None = None,
) -> str:
    return base.artifact_snapshot(config, phase=phase, selected_spec=selected_spec)
