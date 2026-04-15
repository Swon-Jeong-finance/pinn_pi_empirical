from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml

from dynalloc.resolver import load_and_resolve
from dynalloc.stage5_backend import Stage5RunResult
from dynalloc.stage7_backend import run_stage7

ROOT = Path(__file__).resolve().parents[1]


def _write_selection_summary(config, rows: list[dict[str, str]]) -> None:
    path = config.selection_output_dir / "spec_selection_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["spec", "score", "passes_guard", "recommended"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_stage7_config_resolves() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage7_curve_core_rank_sweep.yaml")
    assert config.runtime.backend == "native_stage7"
    assert config.rank_sweep.enabled is True
    assert config.rank_sweep.start_rank == 1



def test_stage7_dry_run_renders_all_rank_cross_mode_pairs(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff25_stage7_curve_core_rank_sweep.yaml")
    config.rank_sweep.end_rank = 2
    _write_selection_summary(
        config,
        [
            {"spec": "best_spec", "score": "1.0", "passes_guard": "True", "recommended": "True"},
            {"spec": "second_spec", "score": "0.9", "passes_guard": "True", "recommended": "True"},
        ],
    )
    specs = run_stage7(config, phase="comparison", dry_run=True)
    assert len(specs) == 4
    shell = "\n".join(spec.shell_command() for spec in specs)
    assert "--state_spec best_spec" in shell
    assert "--state_spec second_spec" in shell
    assert "rank_sweep/rank_001/comparison/estimated" in shell
    assert "rank_sweep/rank_002/comparison/zero" in shell



def test_stage7_serial_comparison_writes_rank_sweep_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff25_stage7_curve_core_rank_sweep.yaml")
    config.experiment.output_dir = str(tmp_path / "outputs")
    config.rank_sweep.end_rank = 2
    _write_selection_summary(
        config,
        [
            {"spec": "best_spec", "score": "1.0", "passes_guard": "True", "recommended": "True"},
            {"spec": "second_spec", "score": "0.9", "passes_guard": "True", "recommended": "True"},
        ],
    )

    def fake_run_stage5(cfg, *, phase, dry_run=False, selected_spec=None):
        assert dry_run is False
        assert phase == "comparison"
        out_dir = cfg.comparison_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "strategy": "ppgdpo",
                    "tc_bps": 0.0,
                    "cer_ann": 1.0,
                    "sharpe": 0.5,
                }
            ]
        ).to_csv(cfg.comparison_zero_cost_summary_path, index=False)
        pd.DataFrame(
            [
                {
                    "strategy": "ppgdpo",
                    "tc_bps": 0.0,
                    "cer_ann": 1.0,
                    "sharpe": 0.5,
                }
            ]
        ).to_csv(cfg.comparison_all_costs_summary_path, index=False)
        cfg.stage5_comparison_report_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.stage5_comparison_report_path.write_text(
            yaml.safe_dump(
                {
                    "selected_spec": selected_spec,
                    "cross_mode": cfg.model.cross_mode,
                },
                sort_keys=False,
            )
        )
        return Stage5RunResult(
            selected_spec=selected_spec,
            selected_specs=[selected_spec] if selected_spec else [],
            completed_phases=[phase],
            warnings=[f"warn:{selected_spec}:{cfg.model.cross_mode}"],
        )

    monkeypatch.setattr("dynalloc.stage7_backend.base.run_stage5", fake_run_stage5)

    result = run_stage7(
        config,
        phase="comparison",
        dry_run=False,
        worker_id="gpu2-worker",
        runtime_device="cuda:2",
    )

    assert result.processed_ranks == [1, 2]
    assert result.failed_ranks == []
    assert (config.rank_sweep_progress_path).exists()
    assert (config.rank_sweep_results_path).exists()
    assert (config.rank_sweep_zero_cost_summary_path).exists()
    assert (config.rank_sweep_all_costs_summary_path).exists()
    assert (config.stage7_rank_sweep_report_path).exists()
    assert (config.stage7_manifest_path).exists()

    progress_df = pd.read_csv(config.rank_sweep_progress_path)
    assert len(progress_df) == 4
    assert set(progress_df["status"]) == {"running", "completed"}

    results_df = pd.read_csv(config.rank_sweep_results_path)
    assert list(results_df["rank"]) == [1, 2]
    assert set(results_df["device"]) == {"cuda:2"}

    zero_df = pd.read_csv(config.rank_sweep_zero_cost_summary_path)
    all_df = pd.read_csv(config.rank_sweep_all_costs_summary_path)
    assert len(zero_df) == 4
    assert len(all_df) == 4
    assert set(zero_df["rank"]) == {1, 2}
    assert set(zero_df["cross_mode"]) == {"estimated", "zero"}

    rank_report = tmp_path / "outputs" / "rank_sweep" / "rank_001" / "stage7_rank_report.yaml"
    stage6_report = tmp_path / "outputs" / "rank_sweep" / "rank_001" / "comparison" / "stage6_comparison_report.yaml"
    assert rank_report.exists()
    assert stage6_report.exists()
