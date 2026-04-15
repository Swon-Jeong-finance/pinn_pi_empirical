from __future__ import annotations

from pathlib import Path

from dynalloc.resolver import load_and_resolve


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ff25_stage2_selection_suite.yaml"


def test_resolve_stage2_config() -> None:
    config = load_and_resolve(CONFIG)
    assert config.split.train_pool_end == "2004-12-31"
    assert config.split.final_test_start == "2005-01-01"
    assert config.split.final_test_years == 20
    assert config.selection.enabled is True
    assert config.comparison.use_selected_spec is True
    assert config.profile_root.endswith("profiles") or "profile_data" in config.profile_root
    assert config.constraints.effective_risky_cap == 1.0
    assert config.constraints.simplex_fast_path is True
