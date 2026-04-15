
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .schema import MacroConfig

DEFAULT_FRED_SERIES: dict[str, str] = {
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "gs2": "GS2",
    "gs5": "GS5",
    "gs10": "GS10",
    "gs20": "GS20",
    "tb3ms": "TB3MS",
    "baa": "BAA",
    "aaa": "AAA",
    "indpro": "INDPRO",
    "unrate": "UNRATE",
    "umcsent": "UMCSENT",
    "mprime": "MPRIME",
}


def _get_fred_module() -> Any:
    return importlib.import_module("pgdpo_yahoo.fred_macro")


def _get_french_module() -> Any:
    return importlib.import_module("pgdpo_yahoo.french_data")


def _build_daily_market_realized_vol_features() -> dict[str, pd.Series]:
    fd = _get_french_module()
    daily = fd.load_ff_factors_daily(fd.FrenchDownloadConfig())
    mkt_total = (daily["Mkt-RF"].astype(float) + daily["RF"].astype(float)).rename("mkt_total")
    rv_1m = mkt_total.pow(2).resample("ME").sum().rename("rv_mkt_1m")
    rv_3m = rv_1m.rolling(3, min_periods=3).sum().rename("rv_mkt_3m")
    eps = 1.0e-12
    return {
        "rv_mkt_1m": rv_1m,
        "rv_mkt_3m": rv_3m,
        "log_rv_mkt_1m": np.log(rv_1m.clip(lower=eps)).rename("log_rv_mkt_1m"),
        "log_rv_mkt_3m": np.log(rv_3m.clip(lower=eps)).rename("log_rv_mkt_3m"),
    }


def _download_many(fred_cfg: Any, series_keys: list[str]) -> dict[str, pd.Series]:
    fm = _get_fred_module()
    out: dict[str, pd.Series] = {}
    for key in series_keys:
        out[key] = fm.download_fred_series_monthly(fred_cfg, DEFAULT_FRED_SERIES[key])
    return out


def _ffill_monthly(series_map: dict[str, pd.Series]) -> dict[str, pd.Series]:
    if not series_map:
        return {}
    common_start = max(s.index.min() for s in series_map.values())
    common_end = min(s.index.max() for s in series_map.values())
    idx = pd.date_range(start=common_start, end=common_end, freq="ME")
    out: dict[str, pd.Series] = {}
    for name, s in series_map.items():
        s2 = s.reindex(idx).astype(float)
        s2 = s2.replace([np.inf, -np.inf], np.nan).ffill()
        out[name] = s2
    return out


def build_macro_pool_monthly(*, fred_cfg: Any, macro_config: MacroConfig) -> pd.DataFrame:
    fm = _get_fred_module()
    if macro_config.pool == "legacy7" and not macro_config.feature_ids:
        return fm.build_macro7_monthly(fred_cfg)

    features = list(macro_config.effective_feature_ids)
    required: set[str] = set()

    if "infl_yoy" in features:
        required.add("cpi")
    if "core_infl_yoy" in features:
        required.add("core_cpi")
    if "term_spread" in features:
        required.update({"tb3ms", "gs10"})
    if "default_spread" in features:
        required.update({"baa", "aaa"})
    if "short_rate" in features:
        required.add("tb3ms")
    if "gs10" in features:
        required.add("gs10")
    if any(name in features for name in {"gs2", "slope_10y_2y", "curvature_2_5_10"}):
        required.add("gs2")
    if any(name in features for name in {"gs5", "slope_20y_5y", "curvature_2_5_10", "curvature_5_10_20"}):
        required.add("gs5")
    if any(name in features for name in {"gs20", "slope_20y_5y", "curvature_5_10_20"}):
        required.add("gs20")
    if "indpro_yoy" in features:
        required.add("indpro")
    if "unrate_chg" in features:
        required.add("unrate")
    if "umcsent" in features:
        required.add("umcsent")
    if "mprime" in features:
        required.add("mprime")

    series = _download_many(fred_cfg, sorted(required)) if required else {}
    series = _ffill_monthly(series)

    computed: dict[str, pd.Series] = {}

    if "infl_yoy" in features and "cpi" in series:
        computed["infl_yoy"] = np.log(series["cpi"]).diff(12).shift(1).rename("infl_yoy")
    if "term_spread" in features and {"tb3ms", "gs10"}.issubset(series):
        computed["term_spread"] = ((series["gs10"] - series["tb3ms"]) / 100.0).rename("term_spread")
    if "default_spread" in features and {"baa", "aaa"}.issubset(series):
        computed["default_spread"] = ((series["baa"] - series["aaa"]) / 100.0).rename("default_spread")
    if "short_rate" in features and "tb3ms" in series:
        computed["short_rate"] = (series["tb3ms"] / 100.0).rename("short_rate")
    if "gs10" in features and "gs10" in series:
        computed["gs10"] = (series["gs10"] / 100.0).rename("gs10")
    if "core_infl_yoy" in features and "core_cpi" in series:
        computed["core_infl_yoy"] = np.log(series["core_cpi"]).diff(12).shift(1).rename("core_infl_yoy")
    if "gs2" in features and "gs2" in series:
        computed["gs2"] = (series["gs2"] / 100.0).rename("gs2")
    if "slope_10y_2y" in features and {"gs2", "gs10"}.issubset(series):
        computed["slope_10y_2y"] = ((series["gs10"] - series["gs2"]) / 100.0).rename("slope_10y_2y")
    if "gs5" in features and "gs5" in series:
        computed["gs5"] = (series["gs5"] / 100.0).rename("gs5")
    if "gs20" in features and "gs20" in series:
        computed["gs20"] = (series["gs20"] / 100.0).rename("gs20")
    if "curvature_2_5_10" in features and {"gs2", "gs5", "gs10"}.issubset(series):
        computed["curvature_2_5_10"] = ((2.0 * series["gs5"] - series["gs2"] - series["gs10"]) / 100.0).rename("curvature_2_5_10")
    if "slope_20y_5y" in features and {"gs5", "gs20"}.issubset(series):
        computed["slope_20y_5y"] = ((series["gs20"] - series["gs5"]) / 100.0).rename("slope_20y_5y")
    if "curvature_5_10_20" in features and {"gs5", "gs10", "gs20"}.issubset(series):
        computed["curvature_5_10_20"] = ((2.0 * series["gs10"] - series["gs5"] - series["gs20"]) / 100.0).rename("curvature_5_10_20")
    if "indpro_yoy" in features and "indpro" in series:
        computed["indpro_yoy"] = np.log(series["indpro"]).diff(12).shift(1).rename("indpro_yoy")
    if "unrate_chg" in features and "unrate" in series:
        computed["unrate_chg"] = (series["unrate"] / 100.0).diff().shift(1).rename("unrate_chg")
    if "umcsent" in features and "umcsent" in series:
        computed["umcsent"] = series["umcsent"].shift(1).rename("umcsent")
    if "mprime" in features and "mprime" in series:
        computed["mprime"] = (series["mprime"] / 100.0).rename("mprime")

    if any(name in features for name in {"rv_mkt_1m", "rv_mkt_3m", "log_rv_mkt_1m", "log_rv_mkt_3m"}):
        computed.update(_build_daily_market_realized_vol_features())

    missing = [feat for feat in features if feat not in computed]
    if missing:
        raise RuntimeError(f"Macro pool requested unknown/unavailable features: {missing}")

    columns = list(features)
    for col in macro_config.macro3_columns:
        if col not in columns:
            columns.append(col)
    df = pd.concat([computed[name] for name in columns], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def describe_macro_pool(macro_config: MacroConfig) -> str:
    features = macro_config.effective_feature_ids
    return f"{macro_config.pool} ({len(features)} features: {', '.join(features)})"
