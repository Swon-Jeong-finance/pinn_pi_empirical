from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dynalloc.legacy_bridge import build_comparison_run_spec, build_selection_run_spec
from dynalloc.resolver import load_and_resolve
from dynalloc.schema import UniverseConfig

ROOT = Path(__file__).resolve().parents[1]

FIXED_CONFIG_NAMES = [
    "ff25_stage12_rank_sweep_cv1985_no_bond.yaml",
    "ff25_stage12_rank_sweep_cv1985_curve_core.yaml",
    "ff25_stage12_rank_sweep_cv1985_ust10y.yaml",
    "ff49_stage12_rank_sweep_cv1985_no_bond.yaml",
    "ff49_stage12_rank_sweep_cv1985_curve_core.yaml",
    "ff49_stage12_rank_sweep_cv1985_ust10y.yaml",
    "fama_market_stage12_rank_sweep_cv1985_no_bond.yaml",
    "fama_market_stage12_rank_sweep_cv1985_curve_core.yaml",
    "fama_market_stage12_rank_sweep_cv1985_ust10y.yaml",
    "ff25_stage12_rank_sweep_cv2000_no_bond.yaml",
    "ff25_stage12_rank_sweep_cv2000_curve_core.yaml",
    "ff25_stage12_rank_sweep_cv2000_ust10y.yaml",
    "ff49_stage12_rank_sweep_cv2000_no_bond.yaml",
    "ff49_stage12_rank_sweep_cv2000_curve_core.yaml",
    "ff49_stage12_rank_sweep_cv2000_ust10y.yaml",
    "fama_market_stage12_rank_sweep_cv2000_no_bond.yaml",
    "fama_market_stage12_rank_sweep_cv2000_curve_core.yaml",
    "fama_market_stage12_rank_sweep_cv2000_ust10y.yaml",
]

ROLLING_CONFIG_NAMES = [
    "ff25_stage12_rank_sweep_cv2000_no_bond_rolling.yaml",
    "ff25_stage12_rank_sweep_cv2000_curve_core_rolling.yaml",
    "ff25_stage12_rank_sweep_cv2000_ust10y_rolling.yaml",
    "ff49_stage12_rank_sweep_cv2000_no_bond_rolling.yaml",
    "ff49_stage12_rank_sweep_cv2000_curve_core_rolling.yaml",
    "ff49_stage12_rank_sweep_cv2000_ust10y_rolling.yaml",
    "fama_market_stage12_rank_sweep_cv2000_no_bond_rolling.yaml",
    "fama_market_stage12_rank_sweep_cv2000_curve_core_rolling.yaml",
    "fama_market_stage12_rank_sweep_cv2000_ust10y_rolling.yaml",
]

CONFIG_NAMES = FIXED_CONFIG_NAMES + ROLLING_CONFIG_NAMES


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_stage12_bond_hook_configs_resolve(config_name: str) -> None:
    config = load_and_resolve(ROOT / "configs" / config_name)
    assert config.selection.top_k == 2
    assert config.selection.window_mode == "rolling"
    assert config.rank_sweep.enabled is True
    assert config.rank_sweep.start_rank == 1
    assert config.rank_sweep.end_rank == 2


@pytest.mark.parametrize("config_name", ROLLING_CONFIG_NAMES)
def test_stage12_rolling_configs_use_rolling_walk_forward(config_name: str) -> None:
    config = load_and_resolve(ROOT / "configs" / config_name)
    assert config.evaluation.walk_forward_mode == "rolling"
    assert config.evaluation.is_rolling is True
    assert config.evaluation.expanding_window is False
    assert config.effective_rolling_train_months == 432


def test_cv2000_split_profile_has_expected_window() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage12_rank_sweep_cv2000_ust10y.yaml")
    assert config.split.train_pool_end == "1999-12-31"
    assert config.split.final_test_start == "2000-01-01"
    assert config.split.end_date == "2019-12-31"
    assert config.split.final_test_years == 20


def test_universe_config_canonicalizes_fama_market_curve_hook() -> None:
    config = UniverseConfig(asset_universe="fama_market", bond_hook="curve_core")
    assert config.asset_universe == "ff_mkt"
    assert config.include_bond is True
    assert [asset.name for asset in config.normalized_bond_assets] == ["UST2Y", "UST5Y", "UST10Y"]


def test_universe_config_none_hook_disables_bonds() -> None:
    config = UniverseConfig(asset_universe="ff25_szbm", bond_hook="none")
    assert config.include_bond is False
    assert config.bond_count == 0


def test_selection_command_uses_curve_core_bond_specs(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff49_stage12_rank_sweep_cv2000_curve_core.yaml")
    spec = build_selection_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--bond_csv_specs" in shell
    assert "UST2Y=" in shell
    assert "UST5Y=" in shell
    assert "UST10Y=" in shell
    assert "--selection_top_k 2" in shell
    assert "--selection_window_mode rolling" in shell



def test_selection_command_uses_no_bond_flag(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff25_stage12_rank_sweep_cv2000_no_bond.yaml")
    spec = build_selection_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--asset_universe ff25_szbm" in shell
    assert "--no_bond" in shell



def test_selection_command_canonicalizes_fama_market_alias(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "fama_market_stage12_rank_sweep_cv2000_ust10y.yaml")
    spec = build_selection_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--asset_universe ff_mkt" in shell
    assert "--bond_csv_specs" in shell



def test_comparison_command_uses_rolling_walk_forward(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff49_stage12_rank_sweep_cv2000_curve_core_rolling.yaml")
    spec = build_comparison_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--walk_forward_mode rolling" in shell
    assert "--rolling_train_months 432" in shell
    assert "--expanding_window" not in shell



def test_selection_expanding_profile_can_be_enabled(tmp_path: Path) -> None:
    src = ROOT / "configs" / "ff49_stage12_rank_sweep_cv2000_curve_core.yaml"
    data = yaml.safe_load(src.read_text())
    data["experiment"]["name"] = data["experiment"]["name"] + "_sel_expanding_test"
    data["experiment"]["output_dir"] = data["experiment"]["output_dir"] + "_sel_expanding_test"
    data["selection"] = {"profile": "cv2000_3fold_fullgrid_core_top2_expanding"}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))

    config = load_and_resolve(cfg_path)
    assert config.selection.window_mode == "expanding"
    assert config.selection.uses_expanding_diagnostics is True
