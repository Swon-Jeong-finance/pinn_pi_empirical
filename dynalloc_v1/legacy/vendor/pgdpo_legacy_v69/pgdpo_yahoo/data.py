"""
Yahoo Finance data access + caching.

Design goal:
- "good enough" for fast prototyping
- deterministic snapshots via local cache
- later: swap this module with CRSP/Bloomberg without touching the rest
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from .utils import ensure_dir, has_pyarrow


@dataclass
class YahooDownloadConfig:
    start: str
    end: str
    interval: str = "1d"     # '1d' is the most robust for Yahoo
    auto_adjust: bool = False
    group_by: str = "column"
    threads: bool = True


class YahooDataClient:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)

    @staticmethod
    def _key(tickers: Iterable[str], cfg: YahooDownloadConfig) -> str:
        tickers_sorted = ",".join(sorted(list(tickers)))
        raw = f"{tickers_sorted}|{cfg.start}|{cfg.end}|{cfg.interval}|{cfg.auto_adjust}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def download(self, tickers: Iterable[str], cfg: YahooDownloadConfig, refresh: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame in the same format as yfinance.download.
        Cached to disk for speed and snapshot reproducibility.
        """
        tickers = list(tickers)
        key = self._key(tickers, cfg)

        # Cache format: parquet if available; else pickle
        if has_pyarrow():
            path = self.cache_dir / f"yahoo_{key}.parquet"
            if path.exists() and not refresh:
                return pd.read_parquet(path)
        else:
            path = self.cache_dir / f"yahoo_{key}.pkl"
            if path.exists() and not refresh:
                return pd.read_pickle(path)

        # Lazy import so that the package can be imported even when yfinance is not installed
        import yfinance as yf  # type: ignore

        df = yf.download(
            tickers=tickers,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
            auto_adjust=cfg.auto_adjust,
            group_by=cfg.group_by,
            threads=cfg.threads,
            progress=False,
        )
        if df is None or len(df) == 0:
            raise RuntimeError("Yahoo download returned empty data. Check tickers/date range.")

        # Save snapshot
        if has_pyarrow():
            df.to_parquet(path)
        else:
            df.to_pickle(path)

        return df

    @staticmethod
    def extract_adj_close(raw: pd.DataFrame, tickers: Iterable[str], allow_missing: bool = False) -> pd.DataFrame:
        """
        Extract adjusted close prices for multiple tickers from yfinance's wide MultiIndex columns.
        """
        tickers = list(tickers)
        if isinstance(raw.columns, pd.MultiIndex):
            # Typical yfinance format: columns (field, ticker)
            if ("Adj Close" in raw.columns.get_level_values(0)) or ("Adj Close" in raw.columns.levels[0]):
                px = raw["Adj Close"].copy()
            elif ("Close" in raw.columns.get_level_values(0)) or ("Close" in raw.columns.levels[0]):
                # fallback when Adj Close missing
                px = raw["Close"].copy()
            else:
                raise ValueError("Cannot find Adj Close/Close in yfinance output.")
        else:
            # Single ticker: columns not multiindex
            if "Adj Close" in raw.columns:
                px = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
            elif "Close" in raw.columns:
                px = raw[["Close"]].rename(columns={"Close": tickers[0]})
            else:
                raise ValueError("Cannot find Adj Close/Close in yfinance output.")

        # Ensure requested tickers present (optionally drop missing)
        missing = [t for t in tickers if t not in px.columns]
        if missing and (not allow_missing):
            raise ValueError(f"Missing tickers in price data: {missing}")
        if missing and allow_missing:
            tickers = [t for t in tickers if t in px.columns]

        px = px[tickers].sort_index()
        px.index = pd.to_datetime(px.index)
        return px
