from __future__ import annotations

from pathlib import Path

from dynalloc.legacy_bridge import render_phase_commands
from dynalloc.resolver import load_and_resolve


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ff25_stage2_selection_suite.yaml"


def test_render_all_phase_without_selected_spec(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(CONFIG)
    rendered = render_phase_commands(config, phase="all", redact_secrets=True)
    assert "--selection_only" in rendered
    assert "comparison command omitted" in rendered
