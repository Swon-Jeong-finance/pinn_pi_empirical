from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .artifacts import (
    discover_comparison_artifacts,
    discover_selection_artifacts,
    parse_summary_metrics_from_log_path,
    render_artifact_snapshot,
    write_comparison_reports,
    write_selection_report,
)
from .legacy_bridge import (
    LegacyRunSpec,
    build_comparison_run_spec,
    build_selection_run_spec,
    resolve_comparison_spec,
)
from .schema import ResolvedExperimentConfig
from .selection import choose_selected_specs, resolve_candidate_specs, write_selected_spec_artifact


class Stage3AError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoggedCommandResult:
    returncode: int
    log_path: Path
    elapsed_sec: float


@dataclass(frozen=True)
class Stage3ARunResult:
    selected_spec: str | None
    selected_specs: list[str]
    completed_phases: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class Stage3APhaseSnapshot:
    phase: str
    command: str
    log_path: str
    returncode: int
    soft_tolerated: bool = False
    soft_reason: str | None = None


@dataclass(frozen=True)
class Stage3AManifest:
    experiment: str
    backend: str
    completed_phases: list[str]
    selected_specs: list[str]
    primary_selected_spec: str | None
    warnings: list[str]
    snapshots: list[Stage3APhaseSnapshot]

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
                    "soft_tolerated": snap.soft_tolerated,
                    "soft_reason": snap.soft_reason,
                }
                for snap in self.snapshots
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _run_logged(spec: LegacyRunSpec, log_path: Path) -> LoggedCommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"# cwd: {spec.cwd}\n")
        log_fh.write(f"# command: {spec.shell_command(redact_secrets=True)}\n\n")
        proc = subprocess.Popen(
            spec.command,
            cwd=spec.cwd,
            env=spec.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_fh.write(line)
        proc.wait()
        elapsed = time.time() - start
        log_fh.write(f"\n# returncode: {proc.returncode}\n")
        log_fh.write(f"# elapsed_sec: {elapsed:.3f}\n")
    return LoggedCommandResult(returncode=int(proc.returncode), log_path=log_path, elapsed_sec=elapsed)


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
        filename="selection_legacy.log",
    )


def _comparison_log_path(config: ResolvedExperimentConfig, spec: str) -> Path:
    safe = "".join(ch for ch in spec if ch.isalnum() or ch in {"_", "-"})
    return _scoped_log_path(
        config,
        output_dir=config.comparison_output_dir,
        filename=f"comparison_{safe}_legacy.log",
    )


def _selection_phase_native(config: ResolvedExperimentConfig) -> tuple[list[str], Stage3APhaseSnapshot]:
    spec = build_selection_run_spec(config)
    run = _run_logged(spec, _selection_log_path(config))
    if run.returncode != 0:
        raise subprocess.CalledProcessError(run.returncode, spec.command)

    artifacts = discover_selection_artifacts(config)
    if artifacts.summary_csv is None:
        raise Stage3AError(
            "Selection completed but spec_selection_summary.csv was not produced."
        )

    selected_specs = choose_selected_specs(config)
    write_selected_spec_artifact(config, selected_specs)
    write_selection_report(
        config,
        selected_specs=selected_specs,
        command=spec.shell_command(redact_secrets=True),
        log_path=run.log_path,
    )
    snapshot = Stage3APhaseSnapshot(
        phase="selection",
        command=spec.shell_command(redact_secrets=True),
        log_path=str(run.log_path),
        returncode=run.returncode,
    )
    return selected_specs, snapshot


def _comparison_soft_success(
    config: ResolvedExperimentConfig,
    *,
    selected_spec: str,
    returncode: int,
    log_path: Path,
) -> tuple[bool, str | None]:
    if returncode == 0:
        return False, None
    artifacts = discover_comparison_artifacts(config, selected_spec=selected_spec)
    summary_df = parse_summary_metrics_from_log_path(log_path)
    core_ok = artifacts.monthly_csv is not None and artifacts.baselines_csv is not None
    if config.method.tc_sweep:
        core_ok = core_ok and (artifacts.tc_sweep_csv is not None)
    if core_ok and (not summary_df.empty or artifacts.tc_sweep_csv is not None):
        return True, (
            "legacy comparison returned non-zero after producing core artifacts; "
            "stage3a treated this as a soft-tolerated tail failure."
        )
    return False, None


def _comparison_phase_native(
    config: ResolvedExperimentConfig,
    *,
    selected_spec: str,
) -> tuple[str, list[str], Stage3APhaseSnapshot]:
    spec = build_comparison_run_spec(config, selected_spec=selected_spec)
    run = _run_logged(spec, _comparison_log_path(config, selected_spec))
    soft_tolerated, soft_reason = _comparison_soft_success(
        config,
        selected_spec=selected_spec,
        returncode=run.returncode,
        log_path=run.log_path,
    )
    if run.returncode != 0 and not (config.runtime.tolerate_legacy_tail_errors and soft_tolerated):
        raise subprocess.CalledProcessError(run.returncode, spec.command)

    write_comparison_reports(
        config,
        selected_spec=selected_spec,
        command=spec.shell_command(redact_secrets=True),
        log_path=run.log_path,
        returncode=run.returncode,
        soft_tolerated=bool(config.runtime.tolerate_legacy_tail_errors and soft_tolerated),
        soft_reason=soft_reason,
    )
    warnings: list[str] = []
    if soft_tolerated and config.runtime.tolerate_legacy_tail_errors:
        warnings.append(soft_reason or "soft-tolerated legacy tail failure")
    snapshot = Stage3APhaseSnapshot(
        phase="comparison",
        command=spec.shell_command(redact_secrets=True),
        log_path=str(run.log_path),
        returncode=run.returncode,
        soft_tolerated=bool(config.runtime.tolerate_legacy_tail_errors and soft_tolerated),
        soft_reason=soft_reason,
    )
    return selected_spec, warnings, snapshot


def write_stage3a_manifest(
    config: ResolvedExperimentConfig,
    *,
    completed_phases: list[str],
    selected_specs: list[str],
    warnings: list[str],
    snapshots: list[Stage3APhaseSnapshot],
) -> Path:
    manifest = Stage3AManifest(
        experiment=config.experiment.name,
        backend="native_stage3a",
        completed_phases=completed_phases,
        selected_specs=selected_specs,
        primary_selected_spec=selected_specs[0] if selected_specs else None,
        warnings=warnings,
        snapshots=snapshots,
    )
    manifest.dump(config.stage3a_manifest_path)
    return config.stage3a_manifest_path


def run_stage3a(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    dry_run: bool = False,
    selected_spec: str | None = None,
) -> Stage3ARunResult | list[LegacyRunSpec]:
    if dry_run:
        from .legacy_bridge import _phase_specs_for_render, resolve_comparison_spec

        chosen = selected_spec
        if phase in {"comparison", "all"} and chosen is None and config.comparison.enabled:
            try:
                chosen = resolve_comparison_spec(config, selected_spec=selected_spec)
            except Exception:
                chosen = None
        return _phase_specs_for_render(config, phase=phase, selected_spec=chosen)

    completed_phases: list[str] = []
    warnings: list[str] = []
    selected_specs: list[str] = []
    snapshots: list[Stage3APhaseSnapshot] = []

    if phase in {"selection", "all"} and config.selection.enabled:
        picked, snapshot = _selection_phase_native(config)
        selected_specs = picked
        completed_phases.append("selection")
        snapshots.append(snapshot)
    if phase in {"comparison", "all"} and config.comparison.enabled:
        if selected_spec is None:
            selected_spec = resolve_comparison_spec(config)
        final_spec, phase_warnings, snapshot = _comparison_phase_native(
            config,
            selected_spec=selected_spec,
        )
        if not selected_specs:
            selected_specs = [final_spec]
        warnings.extend(phase_warnings)
        completed_phases.append("comparison")
        snapshots.append(snapshot)

    write_stage3a_manifest(
        config,
        completed_phases=completed_phases,
        selected_specs=selected_specs,
        warnings=warnings,
        snapshots=snapshots,
    )
    return Stage3ARunResult(
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
