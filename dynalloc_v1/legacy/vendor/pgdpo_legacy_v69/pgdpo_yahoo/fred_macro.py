"""
FRED macro predictors (official API).

Provides monthly macro state variables commonly used in Campbell-style return predictability:
- Inflation (CPI YoY log-change, lagged)
- Term spread (10Y - 3M, annual decimal)
- Default spread (BAA - AAA, annual decimal)

Optional (macro7 extension):
- Industrial production growth (INDPRO YoY log-change, lagged)
- Unemployment rate (UNRATE, lagged)
- Consumer sentiment (UMCSENT / UMCSENT1, lagged)
- Prime rate (MPRIME)

This module does NOT depend on requests; uses urllib + json only.
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"  # FRED endpoint


@dataclass
class FredMacroConfig:
    api_key: str
    cache_dir: Path = Path("./_cache_fred")
    refresh: bool = False

    # Observation window (YYYY-MM-DD); None means no restriction
    start: Optional[str] = None
    end: Optional[str] = None

    # Series IDs (monthly where possible)
    cpi_series: str = "CPIAUCSL"   # CPI, seasonally adjusted, monthly
    gs10_series: str = "GS10"      # 10Y Treasury constant maturity, monthly (%)
    tb3ms_series: str = "TB3MS"    # 3M T-bill secondary market rate, monthly (%)
    baa_series: str = "BAA"        # Moody's BAA yield, monthly (%)
    aaa_series: str = "AAA"        # Moody's AAA yield, monthly (%)

    # Additional monthly macro series (used for macro7)
    indpro_series: str = "INDPRO"    # Industrial production index (monthly)
    unrate_series: str = "UNRATE"    # Unemployment rate (monthly, %)
    umcsent_series: str = "UMCSENT"  # Univ. of Michigan Consumer Sentiment (monthly)
    mprime_series: str = "MPRIME"    # Bank prime loan rate (monthly, %)

    # Transform settings
    infl_yoy_months: int = 12
    infl_lag_months: int = 1

    # Macro7 transforms / lags
    indpro_yoy_months: int = 12
    indpro_lag_months: int = 1      # IP released with a delay (safe default)
    unrate_lag_months: int = 1      # UNRATE released with a delay (safe default)
    umcsent_lag_months: int = 1     # sentiment released with a delay (safe default)

    # Rates/spreads are typically observable at (or near) month-end.
    # Keep lag=0 by default to match the existing macro3 design.
    mprime_lag_months: int = 0
    rate_lag_months: int = 0


def _cache_path(cfg: FredMacroConfig, series_id: str) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{series_id}|{cfg.start}|{cfg.end}".encode("utf-8")).hexdigest()
    return cfg.cache_dir / f"fred_{series_id}_{key}.parquet"


def _fred_get_json(params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    url = f"{FRED_API_BASE}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PGDPO empirical loader)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_fred_series_monthly(cfg: FredMacroConfig, series_id: str) -> pd.Series:
    """
    Download a FRED series via API and return a monthly Series indexed by month-end Timestamp.

    FRED values are strings; '.' indicates missing.
    """
    path = _cache_path(cfg, series_id)
    if path.exists() and (not cfg.refresh):
        try:
            df = pd.read_parquet(path)
            s = df["value"]
            s.index = pd.to_datetime(df["date"])
            return s
        except Exception:
            pass

    params = {
        "series_id": series_id,
        "api_key": cfg.api_key,
        "file_type": "json",
    }
    if cfg.start is not None:
        params["observation_start"] = cfg.start
    if cfg.end is not None:
        params["observation_end"] = cfg.end

    js = _fred_get_json(params)
    obs = js.get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED returned no observations for series_id={series_id}")

    dates = []
    vals = []
    for o in obs:
        d = o.get("date")
        v = o.get("value")
        if d is None:
            continue
        dates.append(pd.to_datetime(d))
        if v is None or v == ".":
            vals.append(np.nan)
        else:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)

    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=series_id).astype(float)
    # Convert to month-end index (FRED monthly series often use first of month; normalize)
    s.index = (s.index + pd.offsets.MonthEnd(0))
    s = s.sort_index().groupby(s.index).last()  # in case of duplicates
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    # cache
    df = pd.DataFrame({"date": s.index, "value": s.values})
    df.to_parquet(path)
    return s


def build_macro3_monthly(cfg: FredMacroConfig) -> pd.DataFrame:
    """
    Return month-end DataFrame with columns:
      - infl_yoy (log CPI_t - log CPI_{t-12}), lagged by infl_lag_months
      - term_spread ((GS10 - TB3MS)/100)
      - default_spread ((BAA - AAA)/100)
    """
    cpi = download_fred_series_monthly(cfg, cfg.cpi_series)
    gs10 = download_fred_series_monthly(cfg, cfg.gs10_series)
    tb3 = download_fred_series_monthly(cfg, cfg.tb3ms_series)
    baa = download_fred_series_monthly(cfg, cfg.baa_series)
    aaa = download_fred_series_monthly(cfg, cfg.aaa_series)

    # Ensure a complete month-end calendar across series.
    # IMPORTANT (look-ahead safety): we only forward-fill missing observations.
    # We do NOT use backward-fill or interpolation, which would leak future
    # information into the past.
    common_start = max(s.index.min() for s in [cpi, gs10, tb3, baa, aaa])
    common_end = min(s.index.max() for s in [cpi, gs10, tb3, baa, aaa])
    # NOTE: pandas deprecated "M" in favor of "ME" (month-end). We use month-end throughout.
    idx_full = pd.date_range(start=common_start, end=common_end, freq="ME")

    def _fill_monthly(s: pd.Series) -> pd.Series:
        s2 = s.reindex(idx_full).astype(float)
        s2 = s2.replace([np.inf, -np.inf], np.nan)
        # Forward-fill only (causal).
        s2 = s2.ffill()
        return s2

    cpi = _fill_monthly(cpi)
    gs10 = _fill_monthly(gs10)
    tb3 = _fill_monthly(tb3)
    baa = _fill_monthly(baa)
    aaa = _fill_monthly(aaa)

    infl = np.log(cpi).diff(cfg.infl_yoy_months).rename("infl_yoy")
    if cfg.infl_lag_months and cfg.infl_lag_months > 0:
        infl = infl.shift(cfg.infl_lag_months)

    term = ((gs10 - tb3) / 100.0).rename("term_spread")
    default = ((baa - aaa) / 100.0).rename("default_spread")
    if getattr(cfg, "rate_lag_months", 0) and int(cfg.rate_lag_months) > 0:
        term = term.shift(int(cfg.rate_lag_months))
        default = default.shift(int(cfg.rate_lag_months))

    df = pd.concat([infl, term, default], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def build_macro7_monthly(cfg: FredMacroConfig) -> pd.DataFrame:
    """Return an expanded monthly macro set (month-end).

    The design goal is to add *plausible* additional macro/financial predictors
    **without shrinking the sample** relative to macro3 (i.e., choose series that
    start no later than GS10's start).

    Output columns (7):
      - infl_yoy (lagged)
      - term_spread
      - default_spread
      - indpro_yoy (log Industrial Production YoY change, lagged)
      - unrate_chg (unemployment rate monthly change, lagged)
      - umcsent (consumer sentiment index, lagged)
      - mprime (prime rate, decimal)

    Look-ahead safety:
      - We only forward-fill missing values after reindexing to a month-end grid.
      - No backward-fill, no interpolation.
    """
    # base macro3 series
    cpi = download_fred_series_monthly(cfg, cfg.cpi_series)
    gs10 = download_fred_series_monthly(cfg, cfg.gs10_series)
    tb3 = download_fred_series_monthly(cfg, cfg.tb3ms_series)
    baa = download_fred_series_monthly(cfg, cfg.baa_series)
    aaa = download_fred_series_monthly(cfg, cfg.aaa_series)

    # extra macro series
    indpro = download_fred_series_monthly(cfg, cfg.indpro_series)
    unrate = download_fred_series_monthly(cfg, cfg.unrate_series)
    umcsent = download_fred_series_monthly(cfg, cfg.umcsent_series)
    mprime = download_fred_series_monthly(cfg, cfg.mprime_series)

    series = [cpi, gs10, tb3, baa, aaa, indpro, unrate, umcsent, mprime]

    common_start = max(s.index.min() for s in series)
    common_end = min(s.index.max() for s in series)
    idx_full = pd.date_range(start=common_start, end=common_end, freq="ME")

    def _fill_monthly_ffill(s: pd.Series) -> pd.Series:
        s2 = s.reindex(idx_full).astype(float)
        s2 = s2.replace([np.inf, -np.inf], np.nan)
        return s2.ffill()

    cpi = _fill_monthly_ffill(cpi)
    gs10 = _fill_monthly_ffill(gs10)
    tb3 = _fill_monthly_ffill(tb3)
    baa = _fill_monthly_ffill(baa)
    aaa = _fill_monthly_ffill(aaa)
    indpro = _fill_monthly_ffill(indpro)
    unrate = _fill_monthly_ffill(unrate)
    umcsent = _fill_monthly_ffill(umcsent)
    mprime = _fill_monthly_ffill(mprime)

    infl = np.log(cpi).diff(cfg.infl_yoy_months).rename("infl_yoy")
    if cfg.infl_lag_months and cfg.infl_lag_months > 0:
        infl = infl.shift(cfg.infl_lag_months)

    term = ((gs10 - tb3) / 100.0).rename("term_spread")
    default = ((baa - aaa) / 100.0).rename("default_spread")
    if cfg.rate_lag_months and cfg.rate_lag_months > 0:
        term = term.shift(cfg.rate_lag_months)
        default = default.shift(cfg.rate_lag_months)

    # Real activity proxy: Industrial Production YoY growth (log-diff)
    indpro_yoy = np.log(indpro).diff(cfg.indpro_yoy_months).rename("indpro_yoy")
    if cfg.indpro_lag_months and cfg.indpro_lag_months > 0:
        indpro_yoy = indpro_yoy.shift(cfg.indpro_lag_months)

    # Labor market: unemployment rate change (decimal units)
    unrate_dec = (unrate / 100.0)
    unrate_chg = unrate_dec.diff().rename("unrate_chg")
    if cfg.unrate_lag_months and cfg.unrate_lag_months > 0:
        unrate_chg = unrate_chg.shift(cfg.unrate_lag_months)

    # Sentiment (index level)
    umcsent_s = umcsent.rename("umcsent")
    if cfg.umcsent_lag_months and cfg.umcsent_lag_months > 0:
        umcsent_s = umcsent_s.shift(cfg.umcsent_lag_months)

    # Credit conditions proxy: prime rate in decimal units
    mprime_d = (mprime / 100.0).rename("mprime")
    if cfg.mprime_lag_months and cfg.mprime_lag_months > 0:
        mprime_d = mprime_d.shift(cfg.mprime_lag_months)

    df = pd.concat([infl, term, default, indpro_yoy, unrate_chg, umcsent_s, mprime_d], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def build_total_return_index_returns_monthly(
    cfg: FredMacroConfig,
    series_id: str,
    *,
    name: str | None = None,
    log_returns: bool = False,
) -> pd.Series:
    """Compute monthly returns from a FRED *index level* series.

    Many bond indices on FRED (e.g., ICE BofA total return index values) are
    provided as **index levels** (often daily). This helper:
      1) downloads the series via FRED API,
      2) aggregates to month-end ("last" observation each month),
      3) converts levels to monthly simple returns.

    Parameters
    ----------
    cfg:
        FRED API config.
    series_id:
        FRED series id (e.g., "BAMLCC0A0CMTRIV").
    name:
        Optional output series name. Defaults to ``series_id``.
    log_returns:
        If True, uses log-differences and converts to simple returns.

    Returns
    -------
    pd.Series
        Monthly simple returns (decimal), indexed by month-end.
    """
    lvl = download_fred_series_monthly(cfg, series_id).sort_index().astype(float)

    # Some FRED "index value" series have occasional missing month-ends.
    # Look-ahead safety: fill missing levels with *forward-fill only*.
    # (Interpolation/backfill would leak future information.)
    if len(lvl) > 1:
        # NOTE: pandas deprecated "M" in favor of "ME" (month-end).
        full_idx = pd.date_range(start=lvl.index[0], end=lvl.index[-1], freq="ME")
        lvl = lvl.reindex(full_idx)
        lvl = lvl.ffill()

    if log_returns:
        # log return -> convert to simple
        lr = np.log(lvl).diff()
        ret = np.exp(lr) - 1.0
    else:
        ret = lvl.pct_change()

    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    ret.name = series_id if name is None else str(name)
    return ret
