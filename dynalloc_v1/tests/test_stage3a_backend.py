from __future__ import annotations

from pathlib import Path

import pandas as pd

from dynalloc.artifacts import parse_summary_metrics_from_log_text, write_comparison_reports
from dynalloc.resolver import load_and_resolve
from dynalloc.stage3a_backend import _comparison_soft_success, run_stage3a


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ff25_stage3a_selection_suite.yaml"


def _fake_log_text() -> str:
    return """
================ SUMMARY (realized; validation/test blocks) ================
[last20y_200501_202412 | ff5_macro7_pca_k3] Policy  sharpe=0.4416  sortino=0.7880  ann_ret=0.1133  ann_vol=0.2212  cer=-0.0172  max_dd=-0.6222  calmar=0.1485  dd_dur=67  final=5.8547
               Myopic  sharpe=0.4424  sortino=0.8012  ann_ret=0.1141  ann_vol=0.2225  cer=-0.0162  max_dd=-0.6181  calmar=0.1506  dd_dur=67  final=5.9292
               PPGDPO  sharpe=0.4742  sortino=0.8478  ann_ret=0.0957  ann_vol=0.1691  cer=0.0209  max_dd=-0.5158  calmar=0.1638  dd_dur=42  final=5.0623
""".strip()


def test_parse_summary_metrics_from_log_text() -> None:
    df = parse_summary_metrics_from_log_text(_fake_log_text())
    assert list(df["strategy"]) == ["policy", "myopic", "ppgdpo"]
    assert float(df.loc[df["strategy"] == "ppgdpo", "cer_ann"].iloc[0]) == 0.0209



def test_run_stage3a_dry_run(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(CONFIG)
    specs = run_stage3a(config, phase="selection", dry_run=True)
    assert len(specs) == 1
    assert "--selection_only" in specs[0].shell_command(redact_secrets=True)



def test_soft_tolerated_comparison_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    config = load_and_resolve(CONFIG)
    config.experiment.output_dir = str(tmp_path / "outputs")
    config.comparison_output_dir.mkdir(parents=True, exist_ok=True)
    spec = "ff5_macro7_pca_k3"

    monthly = config.comparison_output_dir / f"monthly_test_20y_{config.universe.asset_universe}_{spec}_varx_fixed.csv"
    pd.DataFrame({"date": ["2005-01-31", "2005-02-28"], "policy_port_ret": [0.01, -0.02]}).to_csv(monthly, index=False)
    baselines = config.comparison_output_dir / "baselines_last20y_200501_202412.csv"
    pd.DataFrame({"date": ["2005-01-31"], "rf": [0.001]}).to_csv(baselines, index=False)
    tc = config.comparison_output_dir / f"tc_sweep_{spec}_varx_fixed.csv"
    pd.DataFrame(
        {
            "strategy": ["policy", "myopic", "ppgdpo"],
            "tc_bps": [0.0, 0.0, 0.0],
            "cer_ann": [-0.0172, -0.0162, 0.0209],
            "sharpe": [0.4416, 0.4424, 0.4742],
        }
    ).to_csv(tc, index=False)
    log_path = config.logs_dir / "comparison_ff5_macro7_pca_k3_legacy.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(_fake_log_text())

    tolerated, reason = _comparison_soft_success(
        config,
        selected_spec=spec,
        returncode=1,
        log_path=log_path,
    )
    assert tolerated is True
    assert reason is not None

    written = write_comparison_reports(
        config,
        selected_spec=spec,
        command="dummy",
        log_path=log_path,
        returncode=1,
        soft_tolerated=True,
        soft_reason=reason,
    )
    assert config.comparison_report_path.exists()
    assert config.comparison_zero_cost_summary_path.exists()
    zero = pd.read_csv(config.comparison_zero_cost_summary_path)
    assert zero.iloc[0]["strategy"] == "ppgdpo"
    assert written["comparison_report_yaml"] == config.comparison_report_path
