from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import stage5_backend as base
from . import stage6_backend as stage6
from .legacy_bridge import LegacyRunSpec, resolve_comparison_spec
from .schema import ResolvedExperimentConfig
from .selection import read_selection_summary

Stage7Error = base.Stage5Error

_PROGRESS_FIELDS = [
    "timestamp",
    "event",
    "status",
    "rank",
    "spec",
    "worker_id",
    "device",
    "elapsed_sec",
    "error",
    "rank_dir",
]

_RESULT_FIELDS = [
    "timestamp",
    "rank",
    "spec",
    "status",
    "worker_id",
    "device",
    "elapsed_sec",
    "warning_count",
    "cross_modes",
    "rank_dir",
    "stage6_comparison_report",
    "combined_zero_cost_summary",
    "combined_all_costs_summary",
    "error",
]


@dataclass(frozen=True)
class Stage7RunResult:
    selected_spec: str | None
    selected_specs: list[str]
    completed_phases: list[str]
    warnings: list[str]
    processed_ranks: list[int]
    failed_ranks: list[int]
    worker_id: str


@dataclass(frozen=True)
class RankTask:
    rank: int
    spec: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _filtered_selection_rows(config: ResolvedExperimentConfig) -> list[dict[str, str]]:
    rows = read_selection_summary(config)
    if config.selection.guarded_only:
        guarded = [row for row in rows if _truthy(row.get("passes_guard"))]
        if guarded:
            rows = guarded
    return rows


def _planned_rank_tasks(
    config: ResolvedExperimentConfig,
    *,
    selected_rank: int | None = None,
    selected_spec: str | None = None,
) -> list[RankTask]:
    if selected_spec is not None:
        rank = config.comparison.selected_rank if selected_rank is None else int(selected_rank)
        return [RankTask(rank=rank, spec=selected_spec)]

    if not config.comparison.use_selected_spec:
        spec = config.comparison.fixed_spec or config.model.state_spec
        rank = config.comparison.selected_rank if selected_rank is None else int(selected_rank)
        return [RankTask(rank=rank, spec=spec)]

    if selected_rank is not None:
        spec = resolve_comparison_spec(config, selected_spec=None, selected_rank=selected_rank)
        return [RankTask(rank=int(selected_rank), spec=spec)]

    rows = _filtered_selection_rows(config)
    if not rows:
        raise Stage7Error("Selection summary is empty; no rank sweep tasks can be planned.")
    start = int(config.rank_sweep.start_rank)
    stop = len(rows) if config.rank_sweep.end_rank is None else min(int(config.rank_sweep.end_rank), len(rows))
    if start > stop:
        return []
    ranks = list(range(start, stop + 1))
    if config.rank_sweep.max_ranks is not None:
        ranks = ranks[: int(config.rank_sweep.max_ranks)]
    tasks: list[RankTask] = []
    for rank in ranks:
        row = rows[rank - 1]
        spec = str(row.get("spec") or "").strip()
        if not spec:
            continue
        tasks.append(RankTask(rank=rank, spec=spec))
    return tasks


def _rank_label(rank: int) -> str:
    return f"rank_{int(rank):03d}"


def _rank_dir(config: ResolvedExperimentConfig, rank: int) -> Path:
    return config.rank_sweep_output_dir / _rank_label(rank)


def _rank_done_path(config: ResolvedExperimentConfig, rank: int) -> Path:
    return _rank_dir(config, rank) / "_done.yaml"


def _rank_failed_path(config: ResolvedExperimentConfig, rank: int) -> Path:
    return _rank_dir(config, rank) / "_failed.yaml"


def _rank_report_path(config: ResolvedExperimentConfig, rank: int) -> Path:
    return _rank_dir(config, rank) / "stage7_rank_report.yaml"


def _rank_claim_dir(config: ResolvedExperimentConfig, rank: int) -> Path:
    return config.rank_sweep_queue_dir / "claims" / _rank_label(rank)


def _selection_lock_dir(config: ResolvedExperimentConfig) -> Path:
    return config.rank_sweep_queue_dir / "selection.lock"


def _selection_ready_path(config: ResolvedExperimentConfig) -> Path:
    return config.rank_sweep_queue_dir / "selection.ready.yaml"


def _selection_failed_path(config: ResolvedExperimentConfig) -> Path:
    return config.rank_sweep_queue_dir / "selection.failed.yaml"


def _lock_dir(config: ResolvedExperimentConfig, name: str) -> Path:
    return config.rank_sweep_queue_dir / "locks" / f"{name}.lock"


@contextmanager
def _dir_lock(path: Path, *, timeout_seconds: float, poll_seconds: float):
    start = time.monotonic()
    while True:
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            if time.monotonic() - start > timeout_seconds:
                raise Stage7Error(f"Timed out waiting for lock: {path}")
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _named_lock(config: ResolvedExperimentConfig, name: str):
    with _dir_lock(
        _lock_dir(config, name),
        timeout_seconds=float(config.rank_sweep.lock_timeout_seconds),
        poll_seconds=float(config.rank_sweep.poll_seconds),
    ):
        yield


def _append_csv_row(config: ResolvedExperimentConfig, path: Path, fieldnames: list[str], row: dict[str, Any], *, lock_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _named_lock(config, lock_name):
        write_header = not path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in fieldnames})


def _append_jsonl(config: ResolvedExperimentConfig, path: Path, payload: dict[str, Any], *, lock_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _named_lock(config, lock_name):
        with path.open("a") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _append_dataframe(config: ResolvedExperimentConfig, path: Path, df: pd.DataFrame, *, lock_name: str) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _named_lock(config, lock_name):
        write_header = not path.exists()
        df.to_csv(path, mode="a", header=write_header, index=False)


def _active_claims(config: ResolvedExperimentConfig) -> list[dict[str, Any]]:
    claims_dir = config.rank_sweep_queue_dir / "claims"
    out: list[dict[str, Any]] = []
    if not claims_dir.exists():
        return out
    for path in sorted(claims_dir.glob("rank_*")):
        meta_path = path / "claim.yaml"
        payload = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("rank", path.name)
        payload.setdefault("path", str(path))
        out.append(payload)
    return out


def _refresh_rank_sweep_report(
    config: ResolvedExperimentConfig,
    *,
    planned_tasks: list[RankTask],
    warnings: list[str],
) -> None:
    planned_ranks = [task.rank for task in planned_tasks]
    completed: list[int] = []
    failed: list[int] = []
    for rank in planned_ranks:
        if _rank_done_path(config, rank).exists():
            completed.append(rank)
        elif _rank_failed_path(config, rank).exists():
            failed.append(rank)
    payload = {
        "experiment": config.experiment.name,
        "backend": "native_stage7",
        "rank_sweep": {
            "enabled": bool(config.rank_sweep.enabled),
            "start_rank": int(config.rank_sweep.start_rank),
            "end_rank": config.rank_sweep.end_rank,
            "max_ranks": config.rank_sweep.max_ranks,
            "resume": bool(config.rank_sweep.resume),
            "stop_on_error": bool(config.rank_sweep.stop_on_error),
            "output_dir": str(config.rank_sweep_output_dir),
        },
        "planned_ranks": planned_ranks,
        "planned_count": len(planned_ranks),
        "completed_ranks": completed,
        "failed_ranks": failed,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "pending_ranks": [rank for rank in planned_ranks if rank not in completed and rank not in failed],
        "active_claims": _active_claims(config),
        "selection_summary_csv": str(config.selection_output_dir / "spec_selection_summary.csv") if (config.selection_output_dir / "spec_selection_summary.csv").exists() else None,
        "stage7_selection_report": str(config.stage7_selection_report_path) if config.stage7_selection_report_path.exists() else None,
        "progress_csv": str(config.rank_sweep_progress_path) if config.rank_sweep_progress_path.exists() else None,
        "results_csv": str(config.rank_sweep_results_path) if config.rank_sweep_results_path.exists() else None,
        "results_jsonl": str(config.rank_sweep_results_jsonl_path) if config.rank_sweep_results_jsonl_path.exists() else None,
        "combined_zero_cost_summary_csv": str(config.rank_sweep_zero_cost_summary_path) if config.rank_sweep_zero_cost_summary_path.exists() else None,
        "combined_all_costs_summary_csv": str(config.rank_sweep_all_costs_summary_path) if config.rank_sweep_all_costs_summary_path.exists() else None,
        "warnings": list(dict.fromkeys(warnings)),
        "updated_at": _now_utc(),
    }
    config.stage7_rank_sweep_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.stage7_rank_sweep_report_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_stage7_manifest(
    config: ResolvedExperimentConfig,
    *,
    completed_phases: list[str],
    selected_specs: list[str],
    warnings: list[str],
    processed_ranks: list[int],
    failed_ranks: list[int],
    worker_id: str,
) -> None:
    payload = {
        "experiment": config.experiment.name,
        "backend": "native_stage7",
        "completed_phases": list(completed_phases),
        "selected_specs": list(selected_specs),
        "primary_selected_spec": selected_specs[0] if selected_specs else None,
        "warnings": list(dict.fromkeys(warnings)),
        "processed_ranks": list(processed_ranks),
        "failed_ranks": list(failed_ranks),
        "worker_id": worker_id,
        "selection_report": str(config.stage7_selection_report_path) if config.stage7_selection_report_path.exists() else None,
        "rank_sweep_report": str(config.stage7_rank_sweep_report_path) if config.stage7_rank_sweep_report_path.exists() else None,
        "progress_csv": str(config.rank_sweep_progress_path) if config.rank_sweep_progress_path.exists() else None,
        "results_csv": str(config.rank_sweep_results_path) if config.rank_sweep_results_path.exists() else None,
        "results_jsonl": str(config.rank_sweep_results_jsonl_path) if config.rank_sweep_results_jsonl_path.exists() else None,
        "combined_zero_cost_summary_csv": str(config.rank_sweep_zero_cost_summary_path) if config.rank_sweep_zero_cost_summary_path.exists() else None,
        "combined_all_costs_summary_csv": str(config.rank_sweep_all_costs_summary_path) if config.rank_sweep_all_costs_summary_path.exists() else None,
        "updated_at": _now_utc(),
    }
    config.stage7_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.stage7_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_stage7_selection_report(
    config: ResolvedExperimentConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    stage6._copy_yaml_with_backend(
        config.stage5_selection_report_path,
        config.stage7_selection_report_path,
        "native_stage7",
        extra,
    )


def _selection_ready(config: ResolvedExperimentConfig) -> bool:
    return (config.selection_output_dir / "spec_selection_summary.csv").exists()


def _ensure_selection_ready(
    config: ResolvedExperimentConfig,
    *,
    worker_id: str,
    warnings: list[str],
) -> list[str]:
    if not config.selection.enabled:
        return []
    config.rank_sweep_queue_dir.mkdir(parents=True, exist_ok=True)
    selected_specs: list[str] = []
    if config.rank_sweep.resume and _selection_ready(config):
        if config.stage5_selection_report_path.exists() and not config.stage7_selection_report_path.exists():
            _write_stage7_selection_report(
                config,
                extra={
                    "rank_sweep_enabled": bool(config.rank_sweep.enabled),
                    "rank_sweep_output_dir": str(config.rank_sweep_output_dir),
                    "worker_id": worker_id,
                    "selection_status": "reused",
                },
            )
        return selected_specs

    lock_dir = _selection_lock_dir(config)
    try:
        with _dir_lock(
            lock_dir,
            timeout_seconds=float(config.rank_sweep.lock_timeout_seconds),
            poll_seconds=float(config.rank_sweep.poll_seconds),
        ):
            if config.rank_sweep.resume and _selection_ready(config):
                return selected_specs
            sel_result = base.run_stage5(config, phase="selection", dry_run=False)
            selected_specs = list(sel_result.selected_specs)
            warnings.extend(getattr(sel_result, "warnings", []))
            _write_stage7_selection_report(
                config,
                extra={
                    "rank_sweep_enabled": bool(config.rank_sweep.enabled),
                    "rank_sweep_output_dir": str(config.rank_sweep_output_dir),
                    "worker_id": worker_id,
                    "selection_status": "completed",
                },
            )
            _selection_ready_path(config).write_text(
                yaml.safe_dump(
                    {
                        "worker_id": worker_id,
                        "completed_at": _now_utc(),
                        "selection_summary_csv": str(config.selection_output_dir / "spec_selection_summary.csv"),
                    },
                    sort_keys=False,
                )
            )
            if _selection_failed_path(config).exists():
                _selection_failed_path(config).unlink()
            return selected_specs
    except Stage7Error:
        raise
    except Exception as exc:  # pragma: no cover - best-effort coordination path
        _selection_failed_path(config).write_text(
            yaml.safe_dump({"worker_id": worker_id, "failed_at": _now_utc(), "error": str(exc)}, sort_keys=False)
        )
        raise



def _wait_for_selection(config: ResolvedExperimentConfig) -> None:
    if _selection_ready(config):
        return
    failed_path = _selection_failed_path(config)
    while True:
        if _selection_ready(config):
            return
        if failed_path.exists():
            payload = yaml.safe_load(failed_path.read_text()) or {}
            raise Stage7Error(f"Selection phase failed in another worker: {payload}")
        time.sleep(float(config.rank_sweep.poll_seconds))


def _rank_base_config(
    config: ResolvedExperimentConfig,
    *,
    rank: int,
    device_override: str | None,
) -> ResolvedExperimentConfig:
    cfg = config.model_copy(deep=True)
    cfg.comparison.selected_rank = int(rank)
    base_subdir = str(config.comparison.output_subdir).rstrip("/") or "comparison"
    cfg.comparison.output_subdir = f"{config.rank_sweep.output_subdir}/{_rank_label(rank)}/{base_subdir}"
    if device_override is not None:
        cfg.runtime.device = str(device_override)
    return cfg


def _try_claim_rank(config: ResolvedExperimentConfig, *, rank: int, worker_id: str, spec: str, device: str) -> bool:
    if config.rank_sweep.resume and (_rank_done_path(config, rank).exists() or _rank_failed_path(config, rank).exists()):
        return False
    claim_dir = _rank_claim_dir(config, rank)
    claim_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False
    meta = {
        "rank": int(rank),
        "spec": spec,
        "worker_id": worker_id,
        "device": device,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "claimed_at": _now_utc(),
    }
    (claim_dir / "claim.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return True


def _release_rank_claim(config: ResolvedExperimentConfig, rank: int) -> None:
    shutil.rmtree(_rank_claim_dir(config, rank), ignore_errors=True)


def _append_progress(
    config: ResolvedExperimentConfig,
    *,
    event: str,
    status: str,
    rank: int,
    spec: str,
    worker_id: str,
    device: str,
    elapsed_sec: float | None,
    error: str | None,
) -> None:
    _append_csv_row(
        config,
        config.rank_sweep_progress_path,
        _PROGRESS_FIELDS,
        {
            "timestamp": _now_utc(),
            "event": event,
            "status": status,
            "rank": int(rank),
            "spec": spec,
            "worker_id": worker_id,
            "device": device,
            "elapsed_sec": None if elapsed_sec is None else round(float(elapsed_sec), 6),
            "error": error,
            "rank_dir": str(_rank_dir(config, rank)),
        },
        lock_name="progress_csv",
    )


def _append_rank_result(
    config: ResolvedExperimentConfig,
    *,
    rank: int,
    spec: str,
    status: str,
    worker_id: str,
    device: str,
    elapsed_sec: float,
    warnings: list[str],
    cross_modes: list[str],
    report_path: Path | None,
    zero_summary_path: Path | None,
    all_costs_path: Path | None,
    error: str | None,
) -> None:
    row = {
        "timestamp": _now_utc(),
        "rank": int(rank),
        "spec": spec,
        "status": status,
        "worker_id": worker_id,
        "device": device,
        "elapsed_sec": round(float(elapsed_sec), 6),
        "warning_count": len(list(dict.fromkeys(warnings))),
        "cross_modes": ",".join(cross_modes),
        "rank_dir": str(_rank_dir(config, rank)),
        "stage6_comparison_report": str(report_path) if report_path else None,
        "combined_zero_cost_summary": str(zero_summary_path) if zero_summary_path else None,
        "combined_all_costs_summary": str(all_costs_path) if all_costs_path else None,
        "error": error,
    }
    _append_csv_row(config, config.rank_sweep_results_path, _RESULT_FIELDS, row, lock_name="results_csv")
    _append_jsonl(
        config,
        config.rank_sweep_results_jsonl_path,
        {
            **row,
            "warnings": list(dict.fromkeys(warnings)),
        },
        lock_name="results_jsonl",
    )


def _append_rank_summaries(
    config: ResolvedExperimentConfig,
    *,
    rank: int,
    spec: str,
    worker_id: str,
    device: str,
    rank_cfg: ResolvedExperimentConfig,
) -> None:
    if rank_cfg.comparison_cross_modes_zero_cost_summary_path.exists():
        df = pd.read_csv(rank_cfg.comparison_cross_modes_zero_cost_summary_path)
        df.insert(0, "rank", int(rank))
        df.insert(1, "spec", spec)
        df.insert(2, "worker_id", worker_id)
        df.insert(3, "device", device)
        _append_dataframe(
            config,
            config.rank_sweep_zero_cost_summary_path,
            df,
            lock_name="zero_cost_summary",
        )
    if rank_cfg.comparison_cross_modes_all_costs_summary_path.exists():
        df = pd.read_csv(rank_cfg.comparison_cross_modes_all_costs_summary_path)
        df.insert(0, "rank", int(rank))
        df.insert(1, "spec", spec)
        df.insert(2, "worker_id", worker_id)
        df.insert(3, "device", device)
        _append_dataframe(
            config,
            config.rank_sweep_all_costs_summary_path,
            df,
            lock_name="all_costs_summary",
        )


def _write_rank_completion_marker(
    config: ResolvedExperimentConfig,
    *,
    rank: int,
    spec: str,
    worker_id: str,
    device: str,
    elapsed_sec: float,
    warnings: list[str],
    status: str,
    error: str | None,
) -> None:
    path = _rank_done_path(config, rank) if status == "completed" else _rank_failed_path(config, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "rank": int(rank),
                "spec": spec,
                "worker_id": worker_id,
                "device": device,
                "status": status,
                "elapsed_sec": round(float(elapsed_sec), 6),
                "warnings": list(dict.fromkeys(warnings)),
                "error": error,
                "updated_at": _now_utc(),
            },
            sort_keys=False,
        )
    )


def _write_rank_report(
    rank_cfg: ResolvedExperimentConfig,
    *,
    rank: int,
    spec: str,
    worker_id: str,
    device: str,
    mode_reports: dict[str, dict[str, str | None]],
    combined: dict[str, str | None],
    warnings: list[str],
    elapsed_sec: float,
) -> Path:
    path = _rank_report_path(rank_cfg, rank)
    payload = {
        "experiment": rank_cfg.experiment.name,
        "backend": "native_stage7",
        "rank": int(rank),
        "selected_spec": spec,
        "selected_rank": int(rank),
        "worker_id": worker_id,
        "device": device,
        "cross_modes": stage6._effective_cross_modes(rank_cfg),
        "elapsed_sec": round(float(elapsed_sec), 6),
        "per_mode": mode_reports,
        "combined_summaries": combined,
        "warnings": list(dict.fromkeys(warnings)),
        "stage6_comparison_report": str(rank_cfg.stage6_comparison_report_path)
        if rank_cfg.stage6_comparison_report_path.exists()
        else None,
        "updated_at": _now_utc(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _run_one_rank(
    config: ResolvedExperimentConfig,
    *,
    task: RankTask,
    worker_id: str,
    device: str,
) -> tuple[ResolvedExperimentConfig, Path | None, list[str]]:
    rank_cfg = _rank_base_config(config, rank=task.rank, device_override=device)
    cross_modes = stage6._effective_cross_modes(rank_cfg)
    multi_mode = len(cross_modes) > 1
    mode_reports: dict[str, dict[str, str | None]] = {}
    mode_cfgs: list[tuple[str, ResolvedExperimentConfig]] = []
    warnings: list[str] = []
    for mode in cross_modes:
        cfg_mode = stage6._comparison_config_for_mode(rank_cfg, mode=mode, multi_mode=multi_mode)
        mode_cfgs.append((mode, cfg_mode))
        cmp_result = base.run_stage5(cfg_mode, phase="comparison", dry_run=False, selected_spec=task.spec)
        warnings.extend(getattr(cmp_result, "warnings", []))
        mode_reports[mode] = stage6._mode_artifacts(cfg_mode)
    combined = stage6._concat_mode_summaries(rank_cfg, mode_configs=mode_cfgs)
    stage6._write_stage6_comparison_report(
        rank_cfg,
        selected_spec=task.spec,
        mode_reports=mode_reports,
        combined=combined,
        warnings=warnings,
    )
    report_path = _write_rank_report(
        rank_cfg,
        rank=task.rank,
        spec=task.spec,
        worker_id=worker_id,
        device=device,
        mode_reports=mode_reports,
        combined=combined,
        warnings=warnings,
        elapsed_sec=0.0,
    )
    return rank_cfg, report_path, warnings


def _run_rank_task(
    config: ResolvedExperimentConfig,
    *,
    task: RankTask,
    worker_id: str,
    device: str,
    warnings: list[str],
    planned_tasks: list[RankTask],
) -> tuple[bool, str | None]:
    start = time.monotonic()
    _append_progress(
        config,
        event="start",
        status="running",
        rank=task.rank,
        spec=task.spec,
        worker_id=worker_id,
        device=device,
        elapsed_sec=None,
        error=None,
    )
    try:
        rank_cfg = _rank_base_config(config, rank=task.rank, device_override=device)
        cross_modes = stage6._effective_cross_modes(rank_cfg)
        multi_mode = len(cross_modes) > 1
        mode_reports: dict[str, dict[str, str | None]] = {}
        mode_cfgs: list[tuple[str, ResolvedExperimentConfig]] = []
        local_warnings: list[str] = []
        for mode in cross_modes:
            cfg_mode = stage6._comparison_config_for_mode(rank_cfg, mode=mode, multi_mode=multi_mode)
            mode_cfgs.append((mode, cfg_mode))
            cmp_result = base.run_stage5(cfg_mode, phase="comparison", dry_run=False, selected_spec=task.spec)
            local_warnings.extend(getattr(cmp_result, "warnings", []))
            mode_reports[mode] = stage6._mode_artifacts(cfg_mode)
        combined = stage6._concat_mode_summaries(rank_cfg, mode_configs=mode_cfgs)
        stage6._write_stage6_comparison_report(
            rank_cfg,
            selected_spec=task.spec,
            mode_reports=mode_reports,
            combined=combined,
            warnings=local_warnings,
        )
        elapsed = time.monotonic() - start
        report_path = _write_rank_report(
            rank_cfg,
            rank=task.rank,
            spec=task.spec,
            worker_id=worker_id,
            device=device,
            mode_reports=mode_reports,
            combined=combined,
            warnings=local_warnings,
            elapsed_sec=elapsed,
        )
        _append_rank_summaries(
            config,
            rank=task.rank,
            spec=task.spec,
            worker_id=worker_id,
            device=device,
            rank_cfg=rank_cfg,
        )
        _write_rank_completion_marker(
            config,
            rank=task.rank,
            spec=task.spec,
            worker_id=worker_id,
            device=device,
            elapsed_sec=elapsed,
            warnings=local_warnings,
            status="completed",
            error=None,
        )
        _append_rank_result(
            config,
            rank=task.rank,
            spec=task.spec,
            status="completed",
            worker_id=worker_id,
            device=device,
            elapsed_sec=elapsed,
            warnings=local_warnings,
            cross_modes=cross_modes,
            report_path=report_path,
            zero_summary_path=rank_cfg.comparison_cross_modes_zero_cost_summary_path
            if rank_cfg.comparison_cross_modes_zero_cost_summary_path.exists()
            else None,
            all_costs_path=rank_cfg.comparison_cross_modes_all_costs_summary_path
            if rank_cfg.comparison_cross_modes_all_costs_summary_path.exists()
            else None,
            error=None,
        )
        _append_progress(
            config,
            event="finish",
            status="completed",
            rank=task.rank,
            spec=task.spec,
            worker_id=worker_id,
            device=device,
            elapsed_sec=elapsed,
            error=None,
        )
        warnings.extend(local_warnings)
        _refresh_rank_sweep_report(config, planned_tasks=planned_tasks, warnings=warnings)
        return True, None
    except Exception as exc:
        elapsed = time.monotonic() - start
        error = str(exc)
        _write_rank_completion_marker(
            config,
            rank=task.rank,
            spec=task.spec,
            worker_id=worker_id,
            device=device,
            elapsed_sec=elapsed,
            warnings=[],
            status="failed",
            error=error,
        )
        _append_rank_result(
            config,
            rank=task.rank,
            spec=task.spec,
            status="failed",
            worker_id=worker_id,
            device=device,
            elapsed_sec=elapsed,
            warnings=[],
            cross_modes=stage6._effective_cross_modes(_rank_base_config(config, rank=task.rank, device_override=device)),
            report_path=None,
            zero_summary_path=None,
            all_costs_path=None,
            error=error,
        )
        _append_progress(
            config,
            event="finish",
            status="failed",
            rank=task.rank,
            spec=task.spec,
            worker_id=worker_id,
            device=device,
            elapsed_sec=elapsed,
            error=error,
        )
        _refresh_rank_sweep_report(config, planned_tasks=planned_tasks, warnings=warnings)
        return False, error
    finally:
        _release_rank_claim(config, task.rank)


def run_stage7(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    dry_run: bool = False,
    selected_spec: str | None = None,
    selected_rank: int | None = None,
    worker_id: str | None = None,
    runtime_device: str | None = None,
) -> Stage7RunResult | list[LegacyRunSpec]:
    if (not config.rank_sweep.enabled) or (selected_rank is not None):
        cfg = config.model_copy(deep=True)
        if runtime_device is not None:
            cfg.runtime.device = str(runtime_device)
        return stage6.run_stage6(
            cfg,
            phase=phase,
            dry_run=dry_run,
            selected_spec=selected_spec,
            selected_rank=selected_rank,
        )

    device = str(runtime_device or config.runtime.device)
    worker_id = str(worker_id or f"pid-{os.getpid()}")

    if dry_run:
        rendered: list[LegacyRunSpec] = []
        if phase in {"selection", "all"} and config.selection.enabled:
            rendered.extend(base.run_stage5(config, phase="selection", dry_run=True))
        if phase in {"comparison", "all"} and config.comparison.enabled:
            try:
                tasks = _planned_rank_tasks(config, selected_rank=selected_rank, selected_spec=selected_spec)
            except Exception:
                tasks = []
            for task in tasks:
                rank_cfg = _rank_base_config(config, rank=task.rank, device_override=device)
                rendered.extend(
                    stage6.run_stage6(
                        rank_cfg,
                        phase="comparison",
                        dry_run=True,
                        selected_spec=task.spec,
                        selected_rank=task.rank,
                    )
                )
        return rendered

    completed_phases: list[str] = []
    warnings: list[str] = []
    processed_ranks: list[int] = []
    failed_ranks: list[int] = []
    selected_specs: list[str] = []

    config.rank_sweep_output_dir.mkdir(parents=True, exist_ok=True)
    config.rank_sweep_queue_dir.mkdir(parents=True, exist_ok=True)

    if phase in {"selection", "all"} and config.selection.enabled:
        selected_specs = _ensure_selection_ready(config, worker_id=worker_id, warnings=warnings)
        if not _selection_ready(config):
            _wait_for_selection(config)
        completed_phases.append("selection")
        _refresh_rank_sweep_report(config, planned_tasks=[], warnings=warnings)
        _write_stage7_manifest(
            config,
            completed_phases=completed_phases,
            selected_specs=selected_specs,
            warnings=warnings,
            processed_ranks=processed_ranks,
            failed_ranks=failed_ranks,
            worker_id=worker_id,
        )

    if phase in {"comparison", "all"} and config.comparison.enabled:
        if config.comparison.use_selected_spec and phase == "comparison" and not _selection_ready(config):
            raise Stage7Error(
                "Selection summary is missing for the rank sweep. Run the selection phase first, "
                "or launch one worker with --phase all to bootstrap selection."
            )
        if config.comparison.use_selected_spec:
            _wait_for_selection(config)
        tasks = _planned_rank_tasks(config, selected_rank=selected_rank, selected_spec=selected_spec)
        _refresh_rank_sweep_report(config, planned_tasks=tasks, warnings=warnings)
        task_map = {task.rank: task for task in tasks}
        while True:
            claimed: RankTask | None = None
            for task in tasks:
                if config.rank_sweep.resume and (_rank_done_path(config, task.rank).exists() or _rank_failed_path(config, task.rank).exists()):
                    continue
                if _try_claim_rank(config, rank=task.rank, worker_id=worker_id, spec=task.spec, device=device):
                    claimed = task
                    break
            if claimed is None:
                break
            ok, error = _run_rank_task(
                config,
                task=claimed,
                worker_id=worker_id,
                device=device,
                warnings=warnings,
                planned_tasks=tasks,
            )
            if ok:
                processed_ranks.append(claimed.rank)
                if claimed.spec not in selected_specs:
                    selected_specs.append(claimed.spec)
            else:
                failed_ranks.append(claimed.rank)
                if not config.rank_sweep.stop_on_error:
                    warnings.append(f"rank {claimed.rank} failed: {error}")
                else:
                    _write_stage7_manifest(
                        config,
                        completed_phases=completed_phases,
                        selected_specs=selected_specs,
                        warnings=warnings,
                        processed_ranks=processed_ranks,
                        failed_ranks=failed_ranks,
                        worker_id=worker_id,
                    )
                    raise Stage7Error(f"Rank sweep stopped on error at rank {claimed.rank}: {error}")
        completed_phases.append("comparison")
        _refresh_rank_sweep_report(config, planned_tasks=tasks, warnings=warnings)

    _write_stage7_manifest(
        config,
        completed_phases=completed_phases,
        selected_specs=selected_specs,
        warnings=warnings,
        processed_ranks=processed_ranks,
        failed_ranks=failed_ranks,
        worker_id=worker_id,
    )
    return Stage7RunResult(
        selected_spec=selected_specs[0] if selected_specs else None,
        selected_specs=selected_specs,
        completed_phases=completed_phases,
        warnings=list(dict.fromkeys(warnings)),
        processed_ranks=processed_ranks,
        failed_ranks=failed_ranks,
        worker_id=worker_id,
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
        "backend": "native_stage7",
        "phase": phase,
        "selected_spec_override": selected_spec,
        "selected_rank_override": selected_rank,
        "rank_sweep_enabled": bool(config.rank_sweep.enabled),
        "rank_sweep_output_dir": str(config.rank_sweep_output_dir),
        "stage7_manifest": str(config.stage7_manifest_path) if config.stage7_manifest_path.exists() else None,
        "stage7_selection_report": str(config.stage7_selection_report_path) if config.stage7_selection_report_path.exists() else None,
        "stage7_rank_sweep_report": str(config.stage7_rank_sweep_report_path) if config.stage7_rank_sweep_report_path.exists() else None,
        "rank_sweep_progress_csv": str(config.rank_sweep_progress_path) if config.rank_sweep_progress_path.exists() else None,
        "rank_sweep_results_csv": str(config.rank_sweep_results_path) if config.rank_sweep_results_path.exists() else None,
        "rank_sweep_results_jsonl": str(config.rank_sweep_results_jsonl_path) if config.rank_sweep_results_jsonl_path.exists() else None,
        "rank_sweep_zero_cost_summary": str(config.rank_sweep_zero_cost_summary_path) if config.rank_sweep_zero_cost_summary_path.exists() else None,
        "rank_sweep_all_costs_summary": str(config.rank_sweep_all_costs_summary_path) if config.rank_sweep_all_costs_summary_path.exists() else None,
        "active_claims": _active_claims(config),
    }
    return yaml.safe_dump(payload, sort_keys=False)
