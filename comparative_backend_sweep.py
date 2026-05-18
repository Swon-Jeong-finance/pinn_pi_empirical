#!/usr/bin/env python3
"""Run cross-backend comparative analysis from existing rank_001 resolved configs.

Typical use case
----------------
You already have selected rank_001 outputs such as:

    experiments/pipinn/ff6_pls_tau1_declining/rank_001/.../resolved_config.yaml
    experiments/pinn/ff6_pls_tau1_declining/rank_001/.../resolved_config.yaml
    experiments/fdm/ff6_pls_tau1_declining/rank_001/.../resolved_config.yaml

This script uses each selected rank config as a template, changes only the
optimizer backend (plus backend-required policy extraction settings), reruns
``dynalloc_v2.experiments.run_experiment``, and writes aggregate comparison CSVs.

It intentionally does NOT re-run native selection and does NOT re-apply OOS
protocols. The split/protocol already stored in resolved_config.yaml is kept.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

import pandas as pd
import yaml


VALUE_BACKENDS = {"pipinn", "pinn", "fdm"}
DEFAULT_SOURCE_BACKENDS = ["pipinn", "pinn", "fdm"]
DEFAULT_TARGET_BACKENDS = ["pipinn", "pinn", "fdm"]


@dataclass(frozen=True)
class SourceConfig:
    backend: str
    path: Path


@dataclass
class RunRecord:
    suite: str
    rank: str
    selected_by_backend: str
    run_as_backend: str
    source_config_yaml: str
    target_output_dir: str
    status: str
    error: str | None = None
    zero_cost_summary: str | None = None
    all_costs_summary: str | None = None
    results_csv: str | None = None


def _prepend_repo_root(repo_root: str | None) -> None:
    if not repo_root:
        return
    root = str(Path(repo_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_backend_path_overrides(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"--source-configs expects backend=path, got: {raw!r}")
        backend, path = raw.split("=", 1)
        backend = backend.strip().lower()
        if not backend:
            raise ValueError(f"Empty backend in --source-configs value: {raw!r}")
        out[backend] = Path(path).expanduser().resolve()
    return out


def _filter_matches(matches: list[Path], config_match: str | None) -> list[Path]:
    if not config_match:
        return matches
    token = str(config_match)
    return [p for p in matches if token in str(p)]


def _choose_config(matches: list[Path], *, pick_config: str, context: str) -> Path:
    matches = sorted({p.expanduser().resolve() for p in matches})
    if not matches:
        raise FileNotFoundError(f"No resolved_config.yaml found for {context}")
    if len(matches) == 1:
        return matches[0]
    if pick_config == "error":
        sample = "\n".join(f"  - {p}" for p in matches[:20])
        more = "" if len(matches) <= 20 else f"\n  ... and {len(matches) - 20} more"
        raise RuntimeError(
            f"Multiple config files found for {context}. Use --config-glob, --config-match, "
            f"--source-configs backend=/path, or --pick-config latest/first/shortest.\n{sample}{more}"
        )
    if pick_config == "latest":
        return max(matches, key=lambda p: p.stat().st_mtime)
    if pick_config == "first":
        return matches[0]
    if pick_config == "shortest":
        return min(matches, key=lambda p: len(str(p)))
    raise ValueError(f"Unknown pick_config={pick_config!r}")


def find_source_config(
    *,
    experiments_root: Path,
    suite: str,
    rank: str,
    backend: str,
    config_glob: str,
    config_match: str | None,
    pick_config: str,
    overrides: dict[str, Path],
) -> SourceConfig:
    backend = str(backend).lower()
    if backend in overrides:
        path = overrides[backend]
        if not path.exists():
            raise FileNotFoundError(f"Override config for backend={backend!r} does not exist: {path}")
        return SourceConfig(backend=backend, path=path)

    base = experiments_root / backend / suite / rank
    matches = list(base.glob(config_glob)) if base.exists() else []
    matches = _filter_matches(matches, config_match)
    path = _choose_config(matches, pick_config=pick_config, context=f"backend={backend}, base={base}")
    return SourceConfig(backend=backend, path=path)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _safe_setattr(obj: Any, attr: str, value: Any) -> None:
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def _model_dump(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump(mode="json")
    if hasattr(cfg, "dict"):
        return cfg.dict()
    raise TypeError(f"Unsupported config object type: {type(cfg)!r}")


def _apply_backend_overrides(
    cfg: Any,
    *,
    target_backend: str,
    output_dir: Path,
    selected_by_backend: str,
    source_config: Path,
    device: str | None,
    pipinn_policy_output_mode: str | None,
    pipinn_pde_form: str | None,
    fdm_value_form: str | None,
    fdm_scheme: str | None,
) -> Any:
    target_backend = str(target_backend).strip().lower()
    cfg = cfg.model_copy(deep=True) if hasattr(cfg, "model_copy") else copy.deepcopy(cfg)

    cfg.optimizer_backend = target_backend
    cfg.project.output_dir = output_dir
    cfg.project.name = f"{cfg.project.name}__selected_by_{selected_by_backend}__run_as_{target_backend}"

    # Device overrides. These are intentionally broad because the exact schema
    # can differ across code versions.
    if device:
        if hasattr(cfg, "ppgdpo"):
            cfg.ppgdpo.device = str(device)
        if hasattr(cfg, "pipinn"):
            cfg.pipinn.device = str(device)
        if hasattr(cfg, "fdm"):
            _safe_setattr(cfg.fdm, "device", str(device))

    # Backend compatibility. In the current code path, pinn/fdm require foc_clip.
    if hasattr(cfg, "pipinn"):
        if target_backend in {"pinn", "fdm"}:
            cfg.pipinn.policy_output_mode = "foc_clip"
        elif target_backend == "pipinn":
            cfg.pipinn.policy_output_mode = str(pipinn_policy_output_mode or "pure_qp")
        elif pipinn_policy_output_mode:
            cfg.pipinn.policy_output_mode = str(pipinn_policy_output_mode)

        if pipinn_pde_form:
            cfg.pipinn.pde_form = str(pipinn_pde_form)

    # Optional FDM-specific overrides, if this codebase has cfg.fdm.
    if hasattr(cfg, "fdm"):
        if fdm_value_form:
            _safe_setattr(cfg.fdm, "value_form", str(fdm_value_form))
        if fdm_scheme:
            _safe_setattr(cfg.fdm, "scheme", str(fdm_scheme))

    # Attach metadata to project name only; keep actual model/data/split/covariance intact.
    # Some Config schemas forbid extra fields, so do not add arbitrary config keys here.
    _ = source_config
    return cfg


def _read_run_outputs(output_dir: Path) -> tuple[Path, Path, Path]:
    zero = output_dir / "comparison_cross_modes_zero_cost_summary.csv"
    costs = output_dir / "comparison_cross_modes_all_costs_summary.csv"
    results = output_dir / "comparison_results.csv"
    missing = [p for p in [zero, costs, results] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing run output(s): " + ", ".join(str(p) for p in missing))
    return zero, costs, results


def _insert_metadata(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for key, value in reversed(list(meta.items())):
        if key in out.columns:
            out[key] = value
        else:
            out.insert(0, key, value)
    return out


def _copy_or_read_existing_self_outputs(source_cfg_payload: dict[str, Any]) -> Path:
    project = source_cfg_payload.get("project") or {}
    output_dir = project.get("output_dir")
    if not output_dir:
        raise ValueError("Source config does not contain project.output_dir; cannot --reuse-self")
    return Path(output_dir).expanduser().resolve()


def run_one(
    *,
    args: argparse.Namespace,
    Config: Any,
    run_experiment: Any,
    source: SourceConfig,
    target_backend: str,
    run_root: Path,
) -> tuple[RunRecord, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    target_backend = str(target_backend).strip().lower()
    source_payload = _load_yaml(source.path)
    meta = {
        "suite": args.suite,
        "rank": args.rank,
        "selected_by_backend": source.backend,
        "run_as_backend": target_backend,
        "source_config_yaml": str(source.path),
    }

    if args.reuse_self and source.backend == target_backend:
        output_dir = _copy_or_read_existing_self_outputs(source_payload)
        zero, costs, results = _read_run_outputs(output_dir)
        record = RunRecord(
            suite=args.suite,
            rank=args.rank,
            selected_by_backend=source.backend,
            run_as_backend=target_backend,
            source_config_yaml=str(source.path),
            target_output_dir=str(output_dir),
            status="reused",
            zero_cost_summary=str(zero),
            all_costs_summary=str(costs),
            results_csv=str(results),
        )
        meta["target_output_dir"] = str(output_dir)
        return (
            record,
            _insert_metadata(pd.read_csv(zero), meta),
            _insert_metadata(pd.read_csv(costs), meta),
            _insert_metadata(pd.read_csv(results), meta),
        )

    output_dir = run_root / f"selected_by_{source.backend}" / args.rank / f"run_as_{target_backend}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and args.skip_existing:
        zero, costs, results = _read_run_outputs(output_dir)
        record = RunRecord(
            suite=args.suite,
            rank=args.rank,
            selected_by_backend=source.backend,
            run_as_backend=target_backend,
            source_config_yaml=str(source.path),
            target_output_dir=str(output_dir),
            status="skipped_existing",
            zero_cost_summary=str(zero),
            all_costs_summary=str(costs),
            results_csv=str(results),
        )
        meta["target_output_dir"] = str(output_dir)
        return (
            record,
            _insert_metadata(pd.read_csv(zero), meta),
            _insert_metadata(pd.read_csv(costs), meta),
            _insert_metadata(pd.read_csv(results), meta),
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Target output directory already exists and is not empty: {output_dir}. "
            "Use --overwrite, --skip-existing, or a new --run-name."
        )

    cfg = Config.model_validate(source_payload)
    cfg = _apply_backend_overrides(
        cfg,
        target_backend=target_backend,
        output_dir=output_dir,
        selected_by_backend=source.backend,
        source_config=source.path,
        device=args.device,
        pipinn_policy_output_mode=args.pipinn_policy_output_mode,
        pipinn_pde_form=args.pipinn_pde_form,
        fdm_value_form=args.fdm_value_form,
        fdm_scheme=args.fdm_scheme,
    )

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = output_dir / "comparative_resolved_config_preview.yaml"
        preview_path.write_text(yaml.safe_dump(_model_dump(cfg), sort_keys=False), encoding="utf-8")
        record = RunRecord(
            suite=args.suite,
            rank=args.rank,
            selected_by_backend=source.backend,
            run_as_backend=target_backend,
            source_config_yaml=str(source.path),
            target_output_dir=str(output_dir),
            status="dry_run",
        )
        return record, None, None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparative_source.yaml").write_text(
        yaml.safe_dump(
            {
                "suite": args.suite,
                "rank": args.rank,
                "selected_by_backend": source.backend,
                "run_as_backend": target_backend,
                "source_config_yaml": str(source.path),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    artifacts = run_experiment(cfg)
    zero = Path(artifacts.summary_zero_cost)
    costs = Path(artifacts.summary_with_costs)
    results = Path(artifacts.output_dir) / "comparison_results.csv"
    if not results.exists():
        results = output_dir / "comparison_results.csv"

    record = RunRecord(
        suite=args.suite,
        rank=args.rank,
        selected_by_backend=source.backend,
        run_as_backend=target_backend,
        source_config_yaml=str(source.path),
        target_output_dir=str(artifacts.output_dir),
        status="done",
        zero_cost_summary=str(zero),
        all_costs_summary=str(costs),
        results_csv=str(results),
    )
    meta["target_output_dir"] = str(artifacts.output_dir)
    return (
        record,
        _insert_metadata(pd.read_csv(zero), meta),
        _insert_metadata(pd.read_csv(costs), meta),
        _insert_metadata(pd.read_csv(results), meta),
    )


def _write_aggregate_outputs(
    *,
    run_root: Path,
    records: list[RunRecord],
    zero_frames: list[pd.DataFrame],
    cost_frames: list[pd.DataFrame],
    result_frames: list[pd.DataFrame],
    sources: list[SourceConfig],
    args: argparse.Namespace,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)

    records_df = pd.DataFrame([r.__dict__ for r in records])
    records_df.to_csv(run_root / "comparative_runs.csv", index=False)

    if zero_frames:
        pd.concat(zero_frames, ignore_index=True).to_csv(run_root / "comparative_summary_zero_cost.csv", index=False)
    if cost_frames:
        cost_all = pd.concat(cost_frames, ignore_index=True)
        cost_all.to_csv(run_root / "comparative_summary_all_costs.csv", index=False)

        # A compact candidate-only view for the common comparison table.
        candidate_mask = pd.Series(True, index=cost_all.index)
        if "comparison_role" in cost_all.columns:
            candidate_mask &= cost_all["comparison_role"].astype(str).eq("method_candidate")
        if "cross_mode" in cost_all.columns:
            candidate_mask &= cost_all["cross_mode"].astype(str).eq("estimated")
        compact_cols = [
            "selected_by_backend",
            "run_as_backend",
            "rank",
            "strategy_display",
            "strategy",
            "strategy_legacy_label",
            "cross_mode",
            "months",
            "ann_ret",
            "ann_vol",
            "sharpe",
            "cer_ann",
            "avg_turnover",
            "avg_risky_weight",
            "max_drawdown",
            "target_output_dir",
            "source_config_yaml",
        ]
        compact_cols = [c for c in compact_cols if c in cost_all.columns]
        cost_all.loc[candidate_mask, compact_cols].to_csv(run_root / "comparative_candidate_estimated_all_costs.csv", index=False)

    if result_frames:
        pd.concat(result_frames, ignore_index=True).to_csv(run_root / "comparative_results.csv", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "suite": args.suite,
        "rank": args.rank,
        "experiments_root": str(Path(args.experiments_root).expanduser().resolve()),
        "run_root": str(run_root),
        "source_backends": list(args.source_backends),
        "target_backends": list(args.target_backends),
        "source_configs": [{"backend": s.backend, "path": str(s.path)} for s in sources],
        "records_csv": str(run_root / "comparative_runs.csv"),
        "summary_zero_cost_csv": str(run_root / "comparative_summary_zero_cost.csv"),
        "summary_all_costs_csv": str(run_root / "comparative_summary_all_costs.csv"),
        "candidate_estimated_all_costs_csv": str(run_root / "comparative_candidate_estimated_all_costs.csv"),
        "results_csv": str(run_root / "comparative_results.csv"),
        "dry_run": bool(args.dry_run),
        "reuse_self": bool(args.reuse_self),
    }
    (run_root / "comparative_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comparative_backend_sweep",
        description=(
            "Run pipinn/pinn/fdm cross-backend comparisons from existing rank resolved_config.yaml files. "
            "This does not rerun selection and does not re-apply OOS protocols."
        ),
    )
    p.add_argument("--repo-root", default=None, help="Optional repository root to prepend to PYTHONPATH before importing dynalloc_v2.")
    p.add_argument("--experiments-root", default="experiments", help="Root containing experiments/{backend}/{suite}/{rank}/...")
    p.add_argument("--suite", required=True, help="Suite directory name, e.g. ff6_pls_tau1_declining")
    p.add_argument("--rank", default="rank_001", help="Rank directory to compare. Default: rank_001")
    p.add_argument("--source-backends", nargs="+", default=DEFAULT_SOURCE_BACKENDS, help="Backends whose selected specs are used as templates.")
    p.add_argument("--target-backends", nargs="+", default=DEFAULT_TARGET_BACKENDS, help="Backends to run each selected spec as.")
    p.add_argument("--source-configs", nargs="*", default=None, help="Optional explicit backend=resolved_config.yaml overrides.")
    p.add_argument("--config-glob", default="**/resolved_config.yaml", help="Glob under experiments/{backend}/{suite}/{rank}. Default: **/resolved_config.yaml")
    p.add_argument("--config-match", default=None, help="Optional substring filter for discovered config paths, useful when multiple protocols exist.")
    p.add_argument("--pick-config", choices=["error", "latest", "first", "shortest"], default="error", help="How to choose if config discovery finds multiple files. Default: error.")
    p.add_argument("--out-root", default=None, help="Aggregate output root. Default: {experiments-root}/comparative_backend_sweep")
    p.add_argument("--run-name", default=None, help="Run subdirectory name. Default: timestamp.")
    p.add_argument("--device", default=None, help="Optional device override applied to ppgdpo/pipinn/fdm configs, e.g. cuda:0")
    p.add_argument("--pipinn-policy-output-mode", choices=["projection", "pure_qp", "foc_clip"], default=None, help="Override for run_as=pipinn. Default for pipinn target is pure_qp.")
    p.add_argument("--pipinn-pde-form", choices=["log_g", "g"], default=None, help="Optional override for cfg.pipinn.pde_form when present.")
    p.add_argument("--fdm-value-form", default=None, help="Optional override for cfg.fdm.value_form when present.")
    p.add_argument("--fdm-scheme", default=None, help="Optional override for cfg.fdm.scheme when present.")
    p.add_argument("--reuse-self", action="store_true", help="For selected_by=X/run_as=X, reuse the source output instead of rerunning.")
    p.add_argument("--skip-existing", action="store_true", help="If a target output already has results, read them instead of rerunning.")
    p.add_argument("--overwrite", action="store_true", help="Delete target output directories before rerunning.")
    p.add_argument("--fail-fast", action="store_true", help="Stop immediately on the first failed run.")
    p.add_argument("--dry-run", action="store_true", help="Do not run training; only write preview configs and manifest.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.source_backends = [str(x).lower() for x in args.source_backends]
    args.target_backends = [str(x).lower() for x in args.target_backends]
    experiments_root = Path(args.experiments_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else experiments_root / "comparative_backend_sweep"
    
    if args.run_name:
        run_root = out_root / args.suite / args.run_name
    else:
        run_root = out_root / args.suite

    _prepend_repo_root(args.repo_root)
    try:
        from dynalloc_v2.schema import Config
        from dynalloc_v2.experiments import run_experiment
    except Exception as exc:  # noqa: BLE001
        print("Failed to import dynalloc_v2. Use --repo-root or run from the repository root.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    overrides = _parse_backend_path_overrides(args.source_configs)
    sources: list[SourceConfig] = []
    try:
        for backend in args.source_backends:
            sources.append(
                find_source_config(
                    experiments_root=experiments_root,
                    suite=args.suite,
                    rank=args.rank,
                    backend=backend,
                    config_glob=args.config_glob,
                    config_match=args.config_match,
                    pick_config=args.pick_config,
                    overrides=overrides,
                )
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Config discovery failed: {exc}", file=sys.stderr)
        return 2

    records: list[RunRecord] = []
    zero_frames: list[pd.DataFrame] = []
    cost_frames: list[pd.DataFrame] = []
    result_frames: list[pd.DataFrame] = []

    for source in sources:
        for target_backend in args.target_backends:
            print(f"[comparative] selected_by={source.backend} run_as={target_backend}", flush=True)
            try:
                record, zero_df, cost_df, results_df = run_one(
                    args=args,
                    Config=Config,
                    run_experiment=run_experiment,
                    source=source,
                    target_backend=target_backend,
                    run_root=run_root,
                )
                records.append(record)
                if zero_df is not None:
                    zero_frames.append(zero_df)
                if cost_df is not None:
                    cost_frames.append(cost_df)
                if results_df is not None:
                    result_frames.append(results_df)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[comparative] FAILED selected_by={source.backend} run_as={target_backend}: {exc}", file=sys.stderr, flush=True)
                record = RunRecord(
                    suite=args.suite,
                    rank=args.rank,
                    selected_by_backend=source.backend,
                    run_as_backend=str(target_backend).lower(),
                    source_config_yaml=str(source.path),
                    target_output_dir=str(run_root / f"selected_by_{source.backend}" / args.rank / f"run_as_{target_backend}"),
                    status="failed",
                    error=tb,
                )
                records.append(record)
                if args.fail_fast:
                    _write_aggregate_outputs(
                        run_root=run_root,
                        records=records,
                        zero_frames=zero_frames,
                        cost_frames=cost_frames,
                        result_frames=result_frames,
                        sources=sources,
                        args=args,
                    )
                    return 1

    _write_aggregate_outputs(
        run_root=run_root,
        records=records,
        zero_frames=zero_frames,
        cost_frames=cost_frames,
        result_frames=result_frames,
        sources=sources,
        args=args,
    )
    print(f"[comparative] wrote outputs to: {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
