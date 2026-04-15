from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .schema import ResolvedExperimentConfig


@dataclass(frozen=True)
class SelectionArtifacts:
    summary_csv: Path | None
    blocks_csv: Path | None


@dataclass(frozen=True)
class ComparisonArtifacts:
    monthly_csv: Path | None
    tc_sweep_csv: Path | None
    baselines_csv: Path | None
    policy_pt: Path | None


def _latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: (p.stat().st_mtime, str(p)))


def discover_selection_artifacts(config: ResolvedExperimentConfig) -> SelectionArtifacts:
    out_dir = config.selection_output_dir
    return SelectionArtifacts(
        summary_csv=(out_dir / "spec_selection_summary.csv") if (out_dir / "spec_selection_summary.csv").exists() else None,
        blocks_csv=(out_dir / "spec_selection_blocks.csv") if (out_dir / "spec_selection_blocks.csv").exists() else None,
    )


def discover_comparison_artifacts(
    config: ResolvedExperimentConfig,
    *,
    selected_spec: str,
) -> ComparisonArtifacts:
    out_dir = config.comparison_output_dir
    asset_tag = config.universe.asset_universe
    monthly = _latest(list(out_dir.glob(f"monthly_test_*_{asset_tag}_{selected_spec}_*.csv")))
    tc_sweep = _latest(list(out_dir.glob(f"tc_sweep_{selected_spec}_*.csv")))
    policy_pt = _latest(list(out_dir.glob(f"policy_{selected_spec}_*.pt")))
    baselines = _latest(list(out_dir.glob("baselines_*.csv")))
    return ComparisonArtifacts(
        monthly_csv=monthly,
        tc_sweep_csv=tc_sweep,
        baselines_csv=baselines,
        policy_pt=policy_pt,
    )


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_tc_sweep_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)
    if "strategy" not in df.columns and len(cols) >= 1:
        first = cols[0]
        if first.startswith("Unnamed") or first == "strategy":
            df = df.rename(columns={first: "strategy"})
    if "tc_bps" not in df.columns and len(cols) >= 2:
        second = list(df.columns)[1]
        if second.startswith("Unnamed") or second == "tc_bps":
            df = df.rename(columns={second: "tc_bps"})
    if "strategy" not in df.columns or "tc_bps" not in df.columns:
        raise ValueError(f"Could not identify strategy/tc_bps columns in {path}")
    df["tc_bps"] = pd.to_numeric(df["tc_bps"], errors="coerce")
    return df


_FIRST_LINE_RE = re.compile(r"^\[(?P<block>[^|\]]+)\|(?P<spec>[^\]]+)\]\s+(?P<strategy>\S+)\s+(?P<body>.*)$")
_CONT_LINE_RE = re.compile(r"^\s+(?P<strategy>\S+)\s+(?P<body>.*)$")
_METRIC_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<value>[-+]?\d+(?:\.\d+)?)")


def parse_summary_metrics_from_log_text(text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_block: str | None = None
    current_spec: str | None = None
    in_summary = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if "================ SUMMARY" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if line.strip() == "":
            continue
        first = _FIRST_LINE_RE.match(line)
        cont = _CONT_LINE_RE.match(line)
        if first:
            current_block = first.group("block").strip()
            current_spec = first.group("spec").strip()
            strategy = first.group("strategy").strip()
            body = first.group("body")
        elif cont and current_block and current_spec:
            strategy = cont.group("strategy").strip()
            body = cont.group("body")
        else:
            continue

        metrics: dict[str, Any] = {
            "block": current_block,
            "spec": current_spec,
            "strategy": strategy.lower(),
            "strategy_label": strategy,
        }
        for match in _METRIC_RE.finditer(body):
            key = match.group("key")
            value = float(match.group("value"))
            if key == "cer":
                key = "cer_ann"
            elif key == "dd_dur":
                key = "max_dd_dur"
            elif key == "final":
                key = "final_wealth"
            metrics[key] = value
        if len(metrics) > 4:
            rows.append(metrics)
    return pd.DataFrame(rows)


def parse_summary_metrics_from_log_path(path: Path) -> pd.DataFrame:
    return parse_summary_metrics_from_log_text(path.read_text())


def summarize_monthly_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    date_col = df.columns[0] if len(df.columns) > 0 else None
    start = None
    end = None
    if date_col is not None and len(df) > 0:
        start = str(df.iloc[0, 0])
        end = str(df.iloc[-1, 0])
    strategy_cols = [
        col
        for col in df.columns
        if col.endswith("_port_ret") or col.endswith("_wealth") or col.endswith("_turnover")
    ]
    return {
        "path": str(path),
        "rows": int(len(df)),
        "start": start,
        "end": end,
        "date_column": date_col,
        "strategy_columns": strategy_cols,
    }


def write_selection_report(
    config: ResolvedExperimentConfig,
    *,
    selected_specs: list[str],
    command: str,
    log_path: Path,
) -> Path:
    artifacts = discover_selection_artifacts(config)
    payload = {
        "experiment": config.experiment.name,
        "phase": "selection",
        "selected_specs": selected_specs,
        "primary_selected_spec": selected_specs[0] if selected_specs else None,
        "summary_csv": str(artifacts.summary_csv) if artifacts.summary_csv else None,
        "blocks_csv": str(artifacts.blocks_csv) if artifacts.blocks_csv else None,
        "log_path": str(log_path),
        "command": command,
    }
    path = config.selection_report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def write_comparison_reports(
    config: ResolvedExperimentConfig,
    *,
    selected_spec: str,
    command: str,
    log_path: Path,
    returncode: int,
    soft_tolerated: bool,
    soft_reason: str | None,
) -> dict[str, Path | None]:
    artifacts = discover_comparison_artifacts(config, selected_spec=selected_spec)
    written: dict[str, Path | None] = {}

    summary_from_log = parse_summary_metrics_from_log_path(log_path)
    if not summary_from_log.empty:
        summary_from_log.to_csv(config.comparison_summary_from_log_path, index=False)
        written["comparison_summary_from_log_csv"] = config.comparison_summary_from_log_path
    else:
        written["comparison_summary_from_log_csv"] = None

    if artifacts.tc_sweep_csv is not None:
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
        else:
            written["comparison_zero_cost_summary_csv"] = None
    else:
        written["comparison_all_costs_summary_csv"] = None
        written["comparison_zero_cost_summary_csv"] = None

    monthly_summary: dict[str, Any] | None = None
    if artifacts.monthly_csv is not None:
        monthly_summary = summarize_monthly_csv(artifacts.monthly_csv)

    payload = {
        "experiment": config.experiment.name,
        "phase": "comparison",
        "selected_spec": selected_spec,
        "returncode": int(returncode),
        "soft_tolerated": bool(soft_tolerated),
        "soft_reason": soft_reason,
        "command": command,
        "log_path": str(log_path),
        "artifacts": {
            "monthly_csv": str(artifacts.monthly_csv) if artifacts.monthly_csv else None,
            "tc_sweep_csv": str(artifacts.tc_sweep_csv) if artifacts.tc_sweep_csv else None,
            "baselines_csv": str(artifacts.baselines_csv) if artifacts.baselines_csv else None,
            "policy_pt": str(artifacts.policy_pt) if artifacts.policy_pt else None,
            **{k: (str(v) if v else None) for k, v in written.items()},
        },
        "monthly_summary": monthly_summary,
        "log_summary_rows": int(len(summary_from_log)),
    }
    path = config.comparison_report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return {**written, "comparison_report_yaml": path}


def render_artifact_snapshot(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    selected_spec: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "experiment": config.experiment.name,
        "phase": phase,
        "selection": {
            "summary_csv": None,
            "blocks_csv": None,
            "selected_spec_yaml": str(config.selected_spec_path) if config.selected_spec_path.exists() else None,
            "selection_report_yaml": str(config.selection_report_path) if config.selection_report_path.exists() else None,
            "stage3b_selection_report_yaml": str(config.stage3b_selection_report_path) if config.stage3b_selection_report_path.exists() else None,
        },
        "comparison": {
            "selected_spec": selected_spec,
            "monthly_csv": None,
            "tc_sweep_csv": None,
            "baselines_csv": None,
            "policy_pt": None,
            "comparison_report_yaml": str(config.comparison_report_path) if config.comparison_report_path.exists() else None,
            "stage3b_comparison_report_yaml": str(config.stage3b_comparison_report_path) if config.stage3b_comparison_report_path.exists() else None,
            "comparison_summary_from_log_csv": str(config.comparison_summary_from_log_path) if config.comparison_summary_from_log_path.exists() else None,
            "comparison_zero_cost_summary_csv": str(config.comparison_zero_cost_summary_path) if config.comparison_zero_cost_summary_path.exists() else None,
            "comparison_all_costs_summary_csv": str(config.comparison_all_costs_summary_path) if config.comparison_all_costs_summary_path.exists() else None,
        },
        "manifests": {
            "stage2_manifest_yaml": str(config.manifest_path) if config.manifest_path.exists() else None,
            "stage3a_manifest_yaml": str(config.stage3a_manifest_path) if config.stage3a_manifest_path.exists() else None,
            "stage3b_manifest_yaml": str(config.stage3b_manifest_path) if config.stage3b_manifest_path.exists() else None,
        },
        "logs_dir": str(config.logs_dir),
    }
    sel = discover_selection_artifacts(config)
    if sel.summary_csv:
        payload["selection"]["summary_csv"] = str(sel.summary_csv)
    if sel.blocks_csv:
        payload["selection"]["blocks_csv"] = str(sel.blocks_csv)
    if selected_spec:
        cmp = discover_comparison_artifacts(config, selected_spec=selected_spec)
        if cmp.monthly_csv:
            payload["comparison"]["monthly_csv"] = str(cmp.monthly_csv)
        if cmp.tc_sweep_csv:
            payload["comparison"]["tc_sweep_csv"] = str(cmp.tc_sweep_csv)
        if cmp.baselines_csv:
            payload["comparison"]["baselines_csv"] = str(cmp.baselines_csv)
        if cmp.policy_pt:
            payload["comparison"]["policy_pt"] = str(cmp.policy_pt)
    return yaml.safe_dump(payload, sort_keys=False)
