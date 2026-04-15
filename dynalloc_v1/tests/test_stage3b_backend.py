
from __future__ import annotations

from pathlib import Path

from dynalloc.resolver import load_and_resolve
from dynalloc.stage3b_backend import _result_rows, run_stage3b

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ff25_stage3b_selection_suite.yaml"


def test_run_stage3b_dry_run(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(CONFIG)
    specs = run_stage3b(config, phase="selection", dry_run=True)
    assert len(specs) == 1
    assert "--eval_mode" in specs[0].command
    assert "v57" in specs[0].command
    assert "run_french49_10y_model_based_latent_varx_fred.py" in " ".join(specs[0].command)


def test_stage3b_result_rows_include_ppgdpo_variant() -> None:
    result = {
        "policy": {"cer_ann": -0.01, "sharpe": 0.4},
        "myopic": {"cer_ann": -0.02, "sharpe": 0.39},
        "ppgdpo": {"cer_ann": 0.02, "sharpe": 0.47},
        "ppgdpo_variants": {
            "ppgdpo_cross0": {"cer_ann": 0.01, "sharpe": 0.45, "label": "PPGDPO_cross0"}
        },
        "gmv_model": {"cer_ann": 0.03, "sharpe": 0.5},
    }
    baseline_metrics = {"mkt": {"cer_ann": 0.01, "sharpe": 0.42}}
    df = _result_rows(result, block_tag="last20y_200501_202412", selected_spec="ff5_macro7_pca_k3", baseline_metrics=baseline_metrics)
    strategies = set(df["strategy"].tolist())
    assert {"policy", "myopic", "ppgdpo", "ppgdpo_cross0", "gmv_model", "mkt"} <= strategies
