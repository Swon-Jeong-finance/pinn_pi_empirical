from __future__ import annotations

from pathlib import Path

from dynalloc.legacy_bridge import build_comparison_run_spec, build_selection_run_spec
from dynalloc.resolver import load_and_resolve


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ff25_stage2_selection_suite.yaml"


def test_selection_command_contains_stage2_split_logic(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(CONFIG)
    spec = build_selection_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--eval_mode v57" in shell
    assert "--cv_folds 3" in shell
    assert "--selection_only" in shell
    assert "--compare_specs" in shell
    assert "--specs pca_only_k2 pls_H12_k3 macro7_only ff5_macro7_pca_k3" in shell


def test_comparison_command_is_final_test_only(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(CONFIG)
    spec = build_comparison_run_spec(config, selected_spec="pls_H12_k3")
    shell = spec.shell_command(redact_secrets=True)
    assert "--eval_mode legacy" in shell
    assert "--eval_horizons 20" in shell
    assert "--state_spec pls_H12_k3" in shell
    assert "--selection_only" not in shell
