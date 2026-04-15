from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd

from dynalloc.legacy_bridge import build_comparison_run_spec
from dynalloc.resolver import load_and_resolve
from dynalloc.schema import UniverseConfig

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "legacy" / "vendor" / "pgdpo_legacy_v69"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from pgdpo_yahoo import asset_universes as au


def test_universe_aliases_canonicalize_new_stage13_tokens() -> None:
    assert UniverseConfig(asset_universe="ff1").asset_universe == "ff_mkt"
    assert UniverseConfig(asset_universe="ff6").asset_universe == "ff6_szbm"
    assert UniverseConfig(asset_universe="ff100").asset_universe == "ff100_szbm"


def test_ff6_loader_route(monkeypatch) -> None:
    idx = pd.to_datetime(["2000-01-31", "2000-02-29"])
    fake = pd.DataFrame({f"p{i}": [0.01 + i * 0.001, 0.02 + i * 0.001] for i in range(6)}, index=idx)
    monkeypatch.setattr(au, "load_6_size_bm_portfolios_monthly", lambda cfg=None: fake)
    out = au.load_equity_universe_monthly("ff6_szbm")
    assert out.equals(fake)


def test_ff100_loader_route(monkeypatch) -> None:
    idx = pd.to_datetime(["2000-01-31", "2000-02-29"])
    fake = pd.DataFrame({f"p{i}": [0.01, 0.02] for i in range(100)}, index=idx)
    monkeypatch.setattr(au, "load_100_size_bm_portfolios_monthly", lambda cfg=None: fake)
    out = au.load_equity_universe_monthly("ff100_szbm")
    assert out.equals(fake)


def test_stage13_generator_writes_expected_54_configs(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "generate_stage13_symmetric_matrix.py"
    subprocess.run(
        [sys.executable, str(script), "--config-dir", str(tmp_path), "--universes", "ff1", "ff6", "ff100"],
        check=True,
        cwd=str(ROOT),
    )
    files = sorted(tmp_path.glob("*.yaml"))
    assert len(files) == 54
    assert (tmp_path / "ff6_stage13_rank_sweep_cv1985_curve_core_expanding.yaml").exists()
    assert (tmp_path / "ff100_stage13_rank_sweep_cv2000_ust10y_rolling.yaml").exists()


def test_stage13_expanding_config_resolves() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff6_stage13_rank_sweep_cv1985_curve_core_expanding.yaml")
    assert config.universe.asset_universe == "ff6_szbm"
    assert config.evaluation.walk_forward_mode == "expanding"
    assert config.evaluation.expanding_window is True
    assert [asset.name for asset in config.universe.normalized_bond_assets] == ["UST2Y", "UST5Y", "UST10Y"]


def test_stage13_rolling_config_uses_explicit_rolling_flags(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff100_stage13_rank_sweep_cv2000_no_bond_rolling.yaml")
    spec = build_comparison_run_spec(config)
    shell = spec.shell_command(redact_secrets=True)
    assert "--asset_universe ff100_szbm" in shell
    assert "--walk_forward_mode rolling" in shell
    assert "--rolling_train_months 432" in shell
    assert "--no_bond" in shell
