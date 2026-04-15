
from __future__ import annotations

from .resolver import dump_yaml
from .schema import ResolvedExperimentConfig
from .selection import resolve_candidate_specs


def render_resolved_config(config: ResolvedExperimentConfig) -> str:
    payload = config.model_dump(mode="json")
    payload["derived"] = {
        "slug": config.slug,
        "effective_risky_cap": config.constraints.effective_risky_cap,
        "simplex_fast_path": config.constraints.simplex_fast_path,
        "final_test_years": config.split.final_test_years,
        "candidate_specs": resolve_candidate_specs(config),
        "candidate_count": len(resolve_candidate_specs(config)),
        "selection_candidate_mode": config.selection.candidate_mode,
        "selection_output_dir": str(config.selection_output_dir),
        "comparison_output_dir": str(config.comparison_output_dir),
        "manifest_path": str(config.manifest_path),
        "stage3a_manifest_path": str(config.stage3a_manifest_path),
        "stage3b_manifest_path": str(config.stage3b_manifest_path),
        "stage4_manifest_path": str(config.stage4_manifest_path),
        "stage5_manifest_path": str(config.stage5_manifest_path),
        "stage6_manifest_path": str(config.stage6_manifest_path),
        "stage7_manifest_path": str(config.stage7_manifest_path),
        "rank_sweep_output_dir": str(config.rank_sweep_output_dir),
        "stage7_rank_sweep_report_path": str(config.stage7_rank_sweep_report_path),
        "logs_dir": str(config.logs_dir),
        "selected_spec_path": str(config.selected_spec_path),
        "comparison_spec_source": config.comparison.spec_source,
        "comparison_selected_rank": config.comparison.selected_rank,
        "comparison_cross_modes": list(config.comparison.cross_modes),
        "bond_assets": [asset.model_dump(mode="json") for asset in config.universe.normalized_bond_assets],
        "macro_effective_features": list(config.macro.effective_feature_ids),
        "plan_summary": config.plan_summary(),
    }
    return dump_yaml(payload)
