from __future__ import annotations

from pathlib import Path

from dynalloc.legacy_bridge import build_selection_run_spec
from dynalloc.resolver import load_and_resolve
from dynalloc.schema import SplitConfig
from dynalloc.stage3b_backend import _comparison_log_path

ROOT = Path(__file__).resolve().parents[1]


def test_stage10_cv_config_resolves() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage10_curve_core_rank_sweep_cv1985.yaml")
    assert config.runtime.backend == "native_stage7"
    assert config.split.train_pool_start == "1964-01-01"
    assert config.split.train_pool_end == "1984-12-31"
    assert config.split.final_test_start == "1985-01-01"
    assert config.split.final_test_years == 15
    assert config.split.force_common_calendar is True
    assert config.split.common_start_mode == "suite"


def test_selection_command_includes_stage10_split_hooks(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff25_stage10_curve_core_rank_sweep_cv1985.yaml")
    spec = build_selection_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--train_pool_start 1964-01-01" in shell
    assert "--force_common_calendar" in shell
    assert "--common_start_mode suite" in shell


def test_stage3b_comparison_logs_are_scoped_by_rank_and_mode(tmp_path: Path) -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage10_curve_core_rank_sweep_cv1985.yaml")
    config.experiment.output_dir = str(tmp_path / "outputs")
    config.comparison.output_subdir = "rank_sweep/rank_002/comparison/estimated"
    estimated = _comparison_log_path(config, "pls_ff5_macro7_H24_k2")
    config.comparison.output_subdir = "rank_sweep/rank_002/comparison/zero"
    zero = _comparison_log_path(config, "pls_ff5_macro7_H24_k2")

    assert estimated != zero
    assert str(estimated).endswith(
        "logs/rank_sweep/rank_002/comparison/estimated/comparison_pls_ff5_macro7_H24_k2_stage3b.log"
    )
    assert str(zero).endswith(
        "logs/rank_sweep/rank_002/comparison/zero/comparison_pls_ff5_macro7_H24_k2_stage3b.log"
    )


def test_split_config_rejects_train_pool_start_after_end() -> None:
    try:
        SplitConfig(
            train_pool_start="1985-01-01",
            train_pool_end="1984-12-31",
            final_test_start="1985-01-01",
            end_date="1999-12-31",
        )
    except ValueError as exc:
        assert "train_pool_start" in str(exc)
    else:
        raise AssertionError("Expected SplitConfig to reject invalid train_pool_start")
