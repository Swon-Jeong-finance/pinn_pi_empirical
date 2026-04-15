from __future__ import annotations

from pathlib import Path

from dynalloc.legacy_bridge import build_selection_run_spec
from dynalloc.resolver import load_and_resolve
from dynalloc.schema import UniverseConfig
from dynalloc.stage3b_backend import run_stage3b

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "mkt10y_stage12_rank_sweep_cv1985.yaml"


def test_custom_universe_requires_assets_or_bond() -> None:
    try:
        UniverseConfig(asset_universe="custom", include_bond=False, custom_risky_assets=[])
    except ValueError as exc:
        assert "custom" in str(exc)
    else:
        raise AssertionError("Expected empty custom universe to fail")


def test_mkt10y_config_resolves_custom_universe() -> None:
    config = load_and_resolve(CONFIG)
    assert config.universe.asset_universe == "custom"
    assert config.universe.custom_risky_asset_names == ["MKT"]
    assert config.universe.bond_count == 1
    assert config.universe.normalized_bond_assets[0].name == "UST10Y"


def test_selection_command_includes_custom_risky_assets(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(CONFIG)
    spec = build_selection_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--asset_universe custom" in shell
    assert "--custom_risky_assets_json" in shell
    assert '"name":"MKT"' in shell or "MKT" in shell


def test_stage3b_dry_run_supports_custom_universe(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(CONFIG)
    specs = run_stage3b(config, phase="selection", dry_run=True)
    assert len(specs) == 1
    assert "--asset_universe" in specs[0].command
    assert "custom" in specs[0].command
