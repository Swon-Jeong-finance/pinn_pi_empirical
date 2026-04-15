from __future__ import annotations

from pathlib import Path

from dynalloc.resolver import load_and_resolve
from dynalloc.selection import choose_selected_spec, write_selected_spec_artifact


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ff25_stage2_selection_suite.yaml"


def test_choose_selected_spec_from_csv(tmp_path) -> None:
    config = load_and_resolve(CONFIG)
    config.experiment.output_dir = str(tmp_path / "outputs")
    config.selection_output_dir.mkdir(parents=True, exist_ok=True)
    (config.selection_output_dir / "spec_selection_summary.csv").write_text(
        "spec,score,passes_guard\n"
        "macro7_only,0.6,False\n"
        "pls_H12_k3,0.9,True\n"
    )
    config.selection.guarded_only = True
    assert choose_selected_spec(config) == "pls_H12_k3"
    write_selected_spec_artifact(config, ["pls_H12_k3"])
    assert config.selected_spec_path.exists()
