from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from .schema import ResolvedExperimentConfig


class SelectionError(RuntimeError):
    pass


STATIC_SPECS = (
    "macro3_only",
    "macro7_only",
    "ff3_only",
    "ff5_only",
)

FF_SPLIT_SPECS = (
    "ff_mkt",
    "ff_smb",
    "ff_hml",
    "ff_mkt_smb",
    "ff_mkt_hml",
    "ff_smb_hml",
)

EQBOND_BLOCK_SPECS = (
    "macro3_eqbond_block",
    "macro7_eqbond_block",
)

ALIAS_SPECS = (
    "pca_only",
    "macro7_pca",
    "ff5_macro7_pca",
    "pls_only",
)

VALID_PCA_PREFIXES = (
    "pca_only",
    "macro7_pca",
    "ff5_macro7_pca",
)

VALID_PLS_PREFIXES = (
    "pls",
    "pls_macro7",
    "pls_ff5_macro7",
    "pls_ret_macro7",
    "pls_ret_ff5_macro7",
    "pls_bal_ret_macro7",
    "pls_bal_ret_ff5_macro7",
)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = str(value).strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _normalize_plain_pls_prefix(prefix: str) -> str:
    s = str(prefix).strip()
    if s in {"pls", "plain_pls", "pls_only"}:
        return "pls"
    return s


def _grid_generated_specs(config: ResolvedExperimentConfig) -> list[str]:
    sel = config.selection
    specs: list[str] = []

    specs.extend(str(spec) for spec in sel.candidate_specs if str(spec).strip())

    if sel.include_static_specs:
        specs.extend(STATIC_SPECS)
    if sel.include_ff_split_specs:
        specs.extend(FF_SPLIT_SPECS)
    if sel.include_alias_specs:
        specs.extend(ALIAS_SPECS)

    pca_prefixes = _dedupe_keep_order([str(x) for x in sel.pca_prefixes])
    pca_components = [int(x) for x in sel.pca_components]
    for prefix in pca_prefixes:
        if prefix not in VALID_PCA_PREFIXES:
            raise SelectionError(
                f"Unknown PCA prefix '{prefix}'. Valid prefixes: {list(VALID_PCA_PREFIXES)}"
            )
        for k in pca_components:
            specs.append(f"{prefix}_k{k}")

    pls_prefixes = _dedupe_keep_order([_normalize_plain_pls_prefix(x) for x in sel.pls_prefixes])
    pls_horizons = [int(x) for x in sel.pls_horizons]
    pls_components = [int(x) for x in sel.pls_components]
    for prefix in pls_prefixes:
        if prefix not in VALID_PLS_PREFIXES:
            raise SelectionError(
                f"Unknown PLS prefix '{prefix}'. Valid prefixes: {list(VALID_PLS_PREFIXES)}"
            )
        for horizon in pls_horizons:
            for k in pls_components:
                if prefix == "pls":
                    specs.append(f"pls_H{horizon}_k{k}")
                else:
                    specs.append(f"{prefix}_H{horizon}_k{k}")

    include_eqbond = False
    if sel.include_eqbond_block_specs == "always":
        include_eqbond = True
    elif sel.include_eqbond_block_specs == "auto":
        include_eqbond = bool(config.universe.include_bond and config.universe.bond_count > 0)
    if include_eqbond:
        specs.extend(EQBOND_BLOCK_SPECS)

    specs = _dedupe_keep_order(specs)
    if not specs:
        raise SelectionError(
            "selection.candidate_mode='grid' produced no candidate specs. "
            "Provide candidate_specs and/or grid prefixes/components."
        )
    return specs


def resolve_candidate_specs(config: ResolvedExperimentConfig) -> list[str]:
    if config.selection.candidate_mode == "grid":
        return _grid_generated_specs(config)

    specs = [str(spec) for spec in config.selection.candidate_specs if str(spec).strip()]
    if specs:
        return _dedupe_keep_order(specs)
    return [config.model.state_spec]


def selection_summary_csv_path(config: ResolvedExperimentConfig) -> Path:
    return config.selection_output_dir / "spec_selection_summary.csv"


def selection_blocks_csv_path(config: ResolvedExperimentConfig) -> Path:
    return config.selection_output_dir / "spec_selection_blocks.csv"


def read_selection_summary(config: ResolvedExperimentConfig) -> list[dict[str, str]]:
    path = selection_summary_csv_path(config)
    if not path.exists():
        raise SelectionError(
            f"Selection summary CSV not found: {path}. Run the selection phase first."
        )
    with path.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SelectionError(f"Selection summary CSV is empty: {path}")
    return rows


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def choose_selected_specs(
    config: ResolvedExperimentConfig,
    *,
    rows: list[dict[str, str]] | None = None,
    top_k: int | None = None,
) -> list[str]:
    rows = rows if rows is not None else read_selection_summary(config)
    if config.selection.guarded_only:
        guarded = [row for row in rows if _truthy(row.get("passes_guard"))]
        if guarded:
            rows = guarded
    limit = config.selection.top_k if top_k is None else int(top_k)
    if limit < 1:
        raise SelectionError("top_k must be >= 1.")
    picked = [str(row["spec"]) for row in rows[:limit] if row.get("spec")]
    if not picked:
        raise SelectionError("No selected spec could be determined from the selection summary.")
    return picked


def choose_selected_spec(config: ResolvedExperimentConfig, *, rank: int = 1) -> str:
    if rank < 1:
        raise SelectionError("rank must be >= 1.")
    rows = read_selection_summary(config)
    if config.selection.guarded_only:
        guarded = [row for row in rows if _truthy(row.get("passes_guard"))]
        if guarded:
            rows = guarded
    if rank > len(rows):
        raise SelectionError(
            f"Requested selected rank {rank}, but only {len(rows)} ranked rows are available."
        )
    row = rows[rank - 1]
    spec = str(row.get("spec") or "").strip()
    if not spec:
        raise SelectionError(f"No spec found at selected rank {rank}.")
    return spec


def write_selected_spec_artifact(config: ResolvedExperimentConfig, selected_specs: list[str]) -> None:
    config.selected_spec_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": config.experiment.name,
        "selected_specs": selected_specs,
        "primary_selected_spec": selected_specs[0],
        "selection_summary_csv": str(selection_summary_csv_path(config)),
        "candidate_count": len(resolve_candidate_specs(config)),
    }
    config.selected_spec_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def load_selected_spec_artifact(config: ResolvedExperimentConfig) -> dict[str, Any]:
    if not config.selected_spec_path.exists():
        raise SelectionError(
            f"Selected spec artifact not found: {config.selected_spec_path}."
        )
    payload = yaml.safe_load(config.selected_spec_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise SelectionError(
            f"Selected spec artifact must be a mapping: {config.selected_spec_path}"
        )
    return payload
