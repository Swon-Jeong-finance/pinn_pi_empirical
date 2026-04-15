
from __future__ import annotations

from pathlib import Path

from dynalloc.resolver import load_and_resolve
from dynalloc import stage3b_backend as base
from dynalloc.stage4_backend import _patched_stage3b, run_stage4

ROOT = Path(__file__).resolve().parents[1]


def test_stage4_core_curve_resolves_bonds_and_macro() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage4_curve_core_fixed_suite.yaml")
    assert config.runtime.backend == "native_stage4"
    assert config.universe.bond_count == 3
    assert [asset.name for asset in config.universe.normalized_bond_assets] == ["UST2Y", "UST5Y", "UST10Y"]
    assert config.macro.pool == "bond_curve_core"
    assert "gs2" in config.macro.effective_feature_ids
    assert "curvature_2_5_10" in config.macro.effective_feature_ids


def test_run_stage4_dry_run(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff25_stage4_curve_core_selection_suite.yaml")
    specs = run_stage4(config, phase="selection", dry_run=True)
    assert len(specs) == 1
    assert "--eval_mode" in specs[0].command
    assert "v57" in specs[0].command


def test_stage4_patches_bond_panel_args() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage4_curve_core_fixed_suite.yaml")
    with _patched_stage3b(config):
        args = base._build_args(
            config,
            phase="comparison",
            state_spec="ff5_macro7_pca_k3",
            specs=["ff5_macro7_pca_k3"],
            out_dir=config.comparison_output_dir,
        )
    assert args.bond_csv_specs
    assert "UST2Y=" in args.bond_csv_specs
    assert "UST5Y=" in args.bond_csv_specs
    assert "UST10Y=" in args.bond_csv_specs
    assert args.stage4_macro_pool == "bond_curve_core"
