from __future__ import annotations

from pathlib import Path

from dynalloc.resolver import load_and_resolve
from dynalloc.selection import resolve_candidate_specs

ROOT = Path(__file__).resolve().parents[1]


def test_stage17_config_resolves() -> None:
    cfg = load_and_resolve(ROOT / "configs" / "ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed.yaml")
    assert cfg.universe.asset_universe == "ff49ind"
    assert cfg.universe.bond_hook == "curve_core"
    assert cfg.evaluation.walk_forward_mode == "fixed"
    assert cfg.selection.top_k == 2


def test_stage17_candidate_grid_is_pls_only() -> None:
    cfg = load_and_resolve(ROOT / "configs" / "ff38_stage17_rank_sweep_cv2006_curve_core_pls_fixed.yaml")
    specs = resolve_candidate_specs(cfg)
    assert len(specs) == 12
    assert all(spec.startswith(("pls_H", "pls_ret_macro7_H", "pls_ret_ff5_macro7_H")) for spec in specs)
    assert not any("macro7_pca" in spec for spec in specs)
    assert not any(spec in {"macro7_only", "ff3_only", "ff5_only"} for spec in specs)


def test_stage17_generated_configs_exclude_ff1_and_use_curve_core() -> None:
    names = (ROOT / "stage17_generated_configs.txt").read_text().splitlines()
    assert len(names) == 10
    assert not any(name.startswith("ff1_stage17") for name in names)
    assert all("curve_core" in name for name in names)
