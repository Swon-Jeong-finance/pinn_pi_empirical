from __future__ import annotations

from pathlib import Path

from dynalloc.resolver import load_and_resolve
from dynalloc.selection import resolve_candidate_specs

ROOT = Path(__file__).resolve().parents[1]


def test_fullgrid_selection_profile_expands_candidates() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage5_curve_core_selection_fullgrid.yaml")
    specs = resolve_candidate_specs(config)
    assert len(specs) > 50
    assert "pls_H6_k1" in specs
    assert "pls_H24_k4" in specs
    assert "ff5_macro7_pca_k4" in specs
    assert "macro7_eqbond_block" in specs


def test_fullgrid_selection_ff49_resolves() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff49_stage5_curve_core_selection_fullgrid.yaml")
    specs = resolve_candidate_specs(config)
    assert config.runtime.backend == "native_stage5"
    assert config.universe.asset_universe == "ff49ind"
    assert len(specs) > 50
