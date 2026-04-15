from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "legacy" / "vendor" / "pgdpo_legacy_v69"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from dynalloc.legacy_bridge import build_selection_run_spec
from dynalloc.resolver import load_and_resolve
from dynalloc.schema import UniverseConfig
from pgdpo_yahoo import asset_universes as au


def test_market_only_universe_loader_uses_total_return(monkeypatch) -> None:
    idx = pd.to_datetime(["2000-01-31", "2000-02-29"])
    ff = pd.DataFrame({"Mkt-RF": [0.01, 0.02], "RF": [0.001, 0.001]}, index=idx)
    monkeypatch.setattr(au, "load_ff_factors_monthly", lambda cfg=None: ff)
    out = au.load_equity_universe_monthly("ff_mkt")
    assert list(out.columns) == ["EQ_MKT"]
    assert out.iloc[0, 0] == 0.011
    assert out.iloc[1, 0] == 0.021


def test_market_bond_profile_resolves() -> None:
    config = load_and_resolve(ROOT / "configs" / "mkt_ust10y_stage12_rank_sweep_cv1985.yaml")
    assert config.universe.asset_universe == "ff_mkt"
    assert config.universe.include_bond is True
    assert config.universe.bond_count == 1
    assert config.universe.normalized_bond_assets[0].name == "UST10Y"


def test_market_two_bond_profile_resolves() -> None:
    config = load_and_resolve(ROOT / "configs" / "mkt_ust2y_ust10y_stage12_rank_sweep_cv1985.yaml")
    assert config.universe.asset_universe == "ff_mkt"
    assert config.universe.include_bond is True
    assert config.universe.bond_count == 2
    assert [a.name for a in config.universe.normalized_bond_assets] == ["UST2Y", "UST10Y"]


def test_selection_command_uses_market_only_asset_universe(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "mkt_ust10y_stage12_rank_sweep_cv1985.yaml")
    spec = build_selection_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--asset_universe ff_mkt" in shell


def test_universe_config_accepts_market_only_no_bond() -> None:
    u = UniverseConfig(asset_universe="ff_mkt", include_bond=False)
    assert u.bond_count == 0
