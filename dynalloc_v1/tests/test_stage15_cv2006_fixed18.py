from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from dynalloc.resolver import load_and_resolve

ROOT = Path(__file__).resolve().parents[1]


def test_stage15_generator_writes_expected_18_configs(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "generate_stage15_cv2006_fixed18.py"
    subprocess.run([sys.executable, str(script), "--config-dir", str(tmp_path)], check=True, cwd=str(ROOT))
    files = sorted(tmp_path.glob("*.yaml"))
    assert len(files) == 18
    assert (tmp_path / "ff38_stage15_rank_sweep_cv2006_curve_core_fixed.yaml").exists()
    assert (tmp_path / "ff100_stage15_rank_sweep_cv2006_ust10y_fixed.yaml").exists()


def test_stage15_fixed_config_resolves_new_split() -> None:
    config_path = ROOT / "configs" / "ff49_stage15_rank_sweep_cv2006_curve_core_fixed.yaml"
    config = load_and_resolve(config_path)
    raw = config_path.read_text()
    assert config.split.final_test_start == "2006-01-01"
    assert config.split.end_date == "2025-12-31"
    assert 'profile: cv2006_3fold_fullgrid_core_top2' in raw
    assert config.evaluation.walk_forward_mode == "fixed"
    assert [asset.name for asset in config.universe.normalized_bond_assets] == ["UST2Y", "UST5Y", "UST10Y"]
