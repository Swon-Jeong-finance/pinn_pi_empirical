from __future__ import annotations

import csv
from pathlib import Path

from dynalloc.resolver import load_and_resolve
from dynalloc.selection import choose_selected_spec
from dynalloc.stage6_backend import run_stage6

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


def test_choose_selected_spec_supports_rank() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage6_curve_core_selected_pair.yaml")
    _write_selection_summary(
        config,
        [
            {"spec": "best_spec", "score": "1.0", "passes_guard": "True", "recommended": "True"},
            {"spec": "second_spec", "score": "0.9", "passes_guard": "True", "recommended": "True"},
            {"spec": "third_spec", "score": "0.8", "passes_guard": "True", "recommended": "False"},
        ],
    )
    assert choose_selected_spec(config, rank=1) == "best_spec"
    assert choose_selected_spec(config, rank=2) == "second_spec"


def test_stage6_dry_run_comparison_uses_selected_rank_and_cross_modes(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff25_stage6_curve_core_selected_pair_rank2.yaml")
    _write_selection_summary(
        config,
        [
            {"spec": "best_spec", "score": "1.0", "passes_guard": "True", "recommended": "True"},
            {"spec": "second_spec", "score": "0.9", "passes_guard": "True", "recommended": "True"},
        ],
    )
    specs = run_stage6(config, phase="comparison", dry_run=True)
    assert len(specs) == 2
    shell = "\n".join(spec.shell_command() for spec in specs)
    assert "--state_spec second_spec" in shell
    assert "comparison/estimated" in shell
    assert "comparison/zero" in shell
    assert "--cross_mode estimated" in shell
    assert "--cross_mode zero" in shell


def test_stage6_ff49_config_resolves() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff49_stage6_curve_core_selected_pair.yaml")
    assert config.runtime.backend == "native_stage6"
    assert config.comparison.spec_source == "selected"
    assert config.comparison.selected_rank == 1
    assert config.comparison.cross_modes == ["estimated", "zero"]
    assert config.universe.asset_universe == "ff49ind"
