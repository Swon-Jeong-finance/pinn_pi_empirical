from __future__ import annotations

from pathlib import Path

from dynalloc.resolver import load_and_resolve
from dynalloc.stage5_backend import run_stage5

ROOT = Path(__file__).resolve().parents[1]


def test_stage5_dry_run(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(ROOT / "configs" / "ff25_stage5_curve_core_selection_fullgrid.yaml")
    specs = run_stage5(config, phase="selection", dry_run=True)
    assert len(specs) == 1
    assert "--selection_only" in specs[0].command
    assert "--specs" in specs[0].command
