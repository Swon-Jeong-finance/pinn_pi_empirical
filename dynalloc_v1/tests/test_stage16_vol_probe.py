from __future__ import annotations

from pathlib import Path

import pandas as pd

from dynalloc.macro_pool import build_macro_pool_monthly
from dynalloc.resolver import load_and_resolve
from dynalloc.schema import MacroConfig
from dynalloc.stage3b_backend import _macro3_subset_columns

ROOT = Path(__file__).resolve().parents[1]


def test_macro_config_allows_single_macro3_column() -> None:
    cfg = MacroConfig(pool="custom", feature_ids=["log_rv_mkt_1m"], macro3_columns=["log_rv_mkt_1m"])
    assert cfg.effective_feature_ids == ["log_rv_mkt_1m"]


def test_macro3_subset_columns_honors_stage4_args() -> None:
    class Args:
        stage4_macro3_columns = ["log_rv_mkt_1m"]

    assert _macro3_subset_columns(Args()) == ["log_rv_mkt_1m"]


def test_build_macro_pool_monthly_supports_log_rv_feature(monkeypatch) -> None:
    import dynalloc.macro_pool as mp

    idx = pd.date_range("2000-01-31", periods=6, freq="ME")

    class FredModule:
        @staticmethod
        def download_fred_series_monthly(_cfg, series_id: str):
            base = pd.Series(range(1, 7), index=idx, dtype=float)
            if series_id == "CPIAUCSL":
                return base + 100.0
            if series_id == "TB3MS":
                return base + 1.0
            if series_id == "GS10":
                return base + 2.0
            if series_id == "BAA":
                return base + 3.0
            if series_id == "AAA":
                return base + 2.5
            return base

    class FrenchCfg:
        def __init__(self):
            pass

    class FrenchModule:
        FrenchDownloadConfig = FrenchCfg

        @staticmethod
        def load_ff_factors_daily(_cfg=None):
            didx = pd.date_range("2000-01-03", periods=120, freq="B")
            out = pd.DataFrame(index=didx)
            out["Mkt-RF"] = 0.001
            out["RF"] = 0.0001
            out["SMB"] = 0.0
            out["HML"] = 0.0
            return out

    monkeypatch.setattr(mp, "_get_fred_module", lambda: FredModule)
    monkeypatch.setattr(mp, "_get_french_module", lambda: FrenchModule)

    cfg = MacroConfig(pool="custom", feature_ids=["log_rv_mkt_1m"], macro3_columns=["log_rv_mkt_1m"])
    out = build_macro_pool_monthly(fred_cfg=object(), macro_config=cfg)
    assert list(out.columns) == ["log_rv_mkt_1m"]
    assert len(out) >= 1


def test_stage16_probe_config_resolves() -> None:
    cfg = load_and_resolve(ROOT / "configs" / "ff49_stage16_probe_cv2000_curve_core_volonly_fixedpair.yaml")
    assert cfg.selection.enabled is False
    assert cfg.comparison.fixed_spec == "macro7_only"
    assert cfg.macro.effective_feature_ids == ["log_rv_mkt_1m"]
