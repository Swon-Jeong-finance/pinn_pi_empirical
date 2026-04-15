from __future__ import annotations

from pathlib import Path

from dynalloc.resolver import load_and_resolve
from dynalloc.schema import MacroConfig

ROOT = Path(__file__).resolve().parents[1]


def test_macro_config_custom_requires_features() -> None:
    try:
        MacroConfig(pool="custom", feature_ids=[], macro3_columns=["infl_yoy", "term_spread", "default_spread"])
    except ValueError as exc:
        assert "feature_ids" in str(exc)
    else:
        raise AssertionError("Expected custom macro pool without feature_ids to fail")


def test_cv_macro_profile_resolves_custom_pool() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage10_curve_core_rank_sweep_cv1985.yaml")
    assert config.macro.pool == "custom"
    assert config.macro.effective_feature_ids == [
        "infl_yoy",
        "term_spread",
        "default_spread",
        "indpro_yoy",
        "unrate_chg",
        "short_rate",
        "gs5",
        "gs10",
    ]


def test_stage11_alias_config_resolves_custom_pool() -> None:
    config = load_and_resolve(ROOT / "configs" / "ff25_stage11_curve_core_rank_sweep_cv1985.yaml")
    assert config.macro.pool == "custom"
    assert config.runtime.backend == "native_stage7"
