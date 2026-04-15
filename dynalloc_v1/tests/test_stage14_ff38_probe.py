from __future__ import annotations

from pathlib import Path
import py_compile
import subprocess
import sys

import pandas as pd

from dynalloc.resolver import load_and_resolve
from dynalloc.schema import UniverseConfig

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "legacy" / "vendor" / "pgdpo_legacy_v69"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from pgdpo_yahoo import asset_universes as au


def test_universe_aliases_canonicalize_ff38() -> None:
    assert UniverseConfig(asset_universe="ff38").asset_universe == "ff38ind"


def test_ff38_loader_route(monkeypatch) -> None:
    idx = pd.to_datetime(["2000-01-31", "2000-02-29"])
    fake = pd.DataFrame({f"p{i}": [0.01, 0.02] for i in range(38)}, index=idx)
    monkeypatch.setattr(au, "load_38_industry_portfolios_monthly", lambda cfg=None: fake)
    out = au.load_equity_universe_monthly("ff38ind")
    assert out.equals(fake)


def test_stage14_generator_writes_expected_3_configs(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "generate_stage14_ff38_probe.py"
    subprocess.run([sys.executable, str(script), "--config-dir", str(tmp_path)], check=True, cwd=str(ROOT))
    files = sorted(tmp_path.glob("*.yaml"))
    assert len(files) == 3
    assert (tmp_path / "ff38_stage14_rank_sweep_cv2000_no_bond_fixed.yaml").exists()
    assert (tmp_path / "ff38_stage14_rank_sweep_cv2000_curve_core_fixed.yaml").exists()


def test_stage14_fixed_config_resolves() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff38_stage14_rank_sweep_cv2000_curve_core_fixed.yaml")
    assert config.universe.asset_universe == "ff38ind"
    assert config.evaluation.walk_forward_mode == "fixed"
    assert [asset.name for asset in config.universe.normalized_bond_assets] == ["UST2Y", "UST5Y", "UST10Y"]


def test_legacy_runner_compiles_after_stage14_patch() -> None:
    py_compile.compile(str(ROOT / "legacy" / "vendor" / "pgdpo_legacy_v69" / "run_french49_10y_model_based_latent_varx_fred.py"), doraise=True)
