"""
Feature engineering:
- price -> returns
- build state vector Y_t from (a) macro proxies or (b) PCA factors
- build risk-free series from Yahoo yield tickers (heuristic scaling)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.astype(float)
    lr = np.log(px).diff()
    lr = lr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return lr


def to_simple_returns_from_log(log_returns: pd.DataFrame) -> pd.DataFrame:
    return np.expm1(log_returns)


def resample_prices(prices: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    # Use last observation in each period
    # Pandas deprecation: 'M' -> 'ME' (month-end). Keep backward compatibility.
    if freq == "M":
        freq = "ME"
    return prices.resample(freq).last().dropna(how="all")


def align_on_intersection(*dfs: pd.DataFrame) -> List[pd.DataFrame]:
    idx = None
    for df in dfs:
        if idx is None:
            idx = df.index
        else:
            idx = idx.intersection(df.index)
    if idx is None:
        return list(dfs)
    return [df.loc[idx].copy() for df in dfs]


def _heuristic_yield_to_decimal(y: pd.Series) -> pd.Series:
    """
    Yahoo yields have inconsistent scaling depending on ticker (e.g., ^TNX often ~ 10x).
    This heuristic aims to map to a reasonable **annual decimal yield**.

    For empirical RL prototypes, you often standardize states anyway.
    Here we only need rf in a reasonable ballpark for wealth dynamics/backtests.
    """
    s = y.astype(float).copy()
    med = np.nanmedian(s.values)
    # Typical:
    # - if med ~ 40: ^TNX-like (10-year yield * 10) -> divide by 10
    # - if med ~ 400: sometimes *100 -> divide by 100
    if med > 200:
        s = s / 100.0
    elif med > 20:
        s = s / 10.0
    # percent -> decimal
    return s / 100.0


@dataclass
class DatasetConfig:
    freq: str = "W-FRI"               # resample frequency for empirical backtest
    use_pca_states: bool = True
    n_pca_factors: int = 5
    standardize_states: bool = True
    pca_fit_end_date: Optional[str] = None  # fit PCA only using dates <= this (avoid look-ahead)
    scaler_fit_end_date: Optional[str] = None  # fit scaler only using dates <= this (avoid look-ahead)
    strict_no_lookahead: bool = True  # if True, require fit_end_date(s) to be set (avoid silent full-sample fit)
    allow_bfill_states: bool = False  # if True, allow backward-fill (NOT recommended for OOS)
    rf_ticker: str = "^IRX"           # 13-week T-bill yield proxy on Yahoo
    vix_ticker: str = "^VIX"          # optional macro proxy
    include_vix_if_available: bool = False


class MarketDataset:
    """
    Container holding aligned returns, rf, states, and metadata.
    """
    def __init__(
        self,
        dates: pd.DatetimeIndex,
        asset_tickers: List[str],
        state_names: List[str],
        log_returns: pd.DataFrame,
        rf_annual: pd.Series,
        states: pd.DataFrame,
        periods_per_year: int,
    ):
        self.dates = dates
        self.asset_tickers = asset_tickers
        self.state_names = state_names
        self.log_returns = log_returns
        self.rf_annual = rf_annual
        self.states = states
        self.periods_per_year = periods_per_year

    @property
    def simple_returns(self) -> pd.DataFrame:
        return to_simple_returns_from_log(self.log_returns)

    def split_train_test(self, split_date: str) -> Tuple["MarketDataset", "MarketDataset"]:
        d = pd.to_datetime(split_date)
        train_mask = self.dates <= d
        test_mask = self.dates > d
        return self._subset(train_mask), self._subset(test_mask)

    def _subset(self, mask: np.ndarray) -> "MarketDataset":
        dates = self.dates[mask]
        return MarketDataset(
            dates=dates,
            asset_tickers=self.asset_tickers,
            state_names=self.state_names,
            log_returns=self.log_returns.loc[dates],
            rf_annual=self.rf_annual.loc[dates],
            states=self.states.loc[dates],
            periods_per_year=self.periods_per_year,
        )


def build_dataset(
    asset_prices: pd.DataFrame,
    rf_prices: Optional[pd.DataFrame],
    state_prices: Optional[pd.DataFrame],
    cfg: DatasetConfig,
) -> MarketDataset:
    """
    Build a weekly (or chosen freq) dataset.
    Inputs are price-like series (Adj Close) from Yahoo:
    - asset_prices: (T x n)
    - rf_prices: (T x 1) for yield tickers like ^IRX
    - state_prices: (T x m) for macro proxies like ^VIX, ^TNX, etc.
    """
    # 1) resample prices
    asset_px = resample_prices(asset_prices, cfg.freq)
    parts = [asset_px]

    rf_px = None
    if rf_prices is not None:
        rf_px = resample_prices(rf_prices, cfg.freq)
        parts.append(rf_px)

    st_px = None
    if state_prices is not None and len(state_prices.columns) > 0:
        st_px = resample_prices(state_prices, cfg.freq)
        parts.append(st_px)

    # 2) align indices
    aligned = align_on_intersection(*parts)
    asset_px = aligned[0]
    cursor = 1
    if rf_px is not None:
        rf_px = aligned[cursor]
        cursor += 1
    if st_px is not None:
        st_px = aligned[cursor]
        cursor += 1

    # 3) returns
    log_ret = to_log_returns(asset_px).dropna()
    # Align rf/states to return index (because of diff)
    dates = log_ret.index
    if rf_px is not None:
        rf_px = rf_px.loc[dates]
    if st_px is not None:
        st_px = st_px.loc[dates]

    # 4) rf annual series
    if rf_px is None:
        rf_annual = pd.Series(0.0, index=dates, name="rf_annual")
    else:
        rf_raw = rf_px.iloc[:, 0]
        rf_annual = _heuristic_yield_to_decimal(rf_raw).rename("rf_annual").reindex(dates).ffill()

    # 5) states
    states_list = []
    state_names = []

    if cfg.use_pca_states:
        # PCA on asset returns.
        # IMPORTANT for OOS: fit PCA loadings using only *past* data (no look-ahead).
        X_all = log_ret.values
        X_all = np.nan_to_num(X_all, nan=0.0)

        fit_idx = dates
        # [v47] strict look-ahead guard: require an explicit fit window for PCA in OOS settings
        if cfg.strict_no_lookahead and (cfg.pca_fit_end_date is None):
            raise ValueError(
                "DatasetConfig.pca_fit_end_date must be set when strict_no_lookahead=True "
                "(prevents silent full-sample PCA fit / look-ahead)."
            )
        if cfg.pca_fit_end_date is not None:
            fit_end = pd.to_datetime(cfg.pca_fit_end_date)
            fit_idx = dates[dates <= fit_end]
            if len(fit_idx) < max(30, cfg.n_pca_factors + 5):
                # [v47] Never fall back to full-sample fit in strict mode (would be look-ahead).
                if cfg.strict_no_lookahead:
                    raise ValueError(
                        "PCA fit window too short under strict_no_lookahead. "
                        "Choose an earlier pca_fit_end_date or reduce n_pca_factors."
                    )
                fit_idx = dates

        X_fit = log_ret.loc[fit_idx].values
        X_fit = np.nan_to_num(X_fit, nan=0.0)

        n_comp = int(min(cfg.n_pca_factors, X_fit.shape[0], X_fit.shape[1]))
        if n_comp < 1:
            raise ValueError(f"PCA needs at least 1 component; got n_comp={n_comp}.")
        if n_comp < cfg.n_pca_factors:
            # Avoid crashing when assets < requested factors
            # (common in prototyping with small universes).
            pass
        pca = PCA(n_components=n_comp)
        pca.fit(X_fit)
        factors = pca.transform(X_all)

        for i in range(n_comp):
            states_list.append(factors[:, i])
            state_names.append(f"pca_{i+1}")
        states = pd.DataFrame(np.column_stack(states_list), index=dates, columns=state_names)
    else:
        if st_px is None:
            raise ValueError("cfg.use_pca_states=False but no state_prices provided.")
        # Default: use log of macro proxy levels as states
        states = np.log(st_px.astype(float)).replace([np.inf, -np.inf], np.nan).ffill()  # NOTE: no bfill to avoid look-ahead (v47)
        state_names = list(states.columns)

    # [v47] ensure no look-ahead from backfilling macro/state proxies
    # We do NOT bfill by default. If early rows are NaN (e.g., proxy starts later),
    # we drop those rows and align returns/rf accordingly.
    if cfg.allow_bfill_states:
        if cfg.strict_no_lookahead:
            raise ValueError("allow_bfill_states=True is incompatible with strict_no_lookahead=True (would leak future via backfill).")
        # Explicit opt-in only (unsafe for strict OOS)
        states = states.bfill()

    # Drop any rows with missing state values (prevents leaking future via bfill)
    valid = states.notna().all(axis=1)
    if not bool(valid.all()):
        states = states.loc[valid].copy()
        # Align returns/rf to the trimmed state index
        log_ret = log_ret.loc[states.index]
        rf_annual = rf_annual.loc[states.index]
        dates = states.index
        if st_px is not None:
            st_px = st_px.loc[dates]
        if rf_px is not None:
            rf_px = rf_px.loc[dates]


    # Optionally include VIX if available
    if cfg.include_vix_if_available and (st_px is not None) and (cfg.vix_ticker in st_px.columns):
        vix = np.log(st_px[cfg.vix_ticker].astype(float)).rename("log_vix")
        states = pd.concat([states, vix], axis=1)
        state_names = list(states.columns)

    # [v47] post-vix NaN trim (keeps builder robust without bfill)
    if cfg.allow_bfill_states:
        if cfg.strict_no_lookahead:
            raise ValueError("allow_bfill_states=True is incompatible with strict_no_lookahead=True (would leak future via backfill).")
        states = states.bfill()
    valid2 = states.notna().all(axis=1)
    if not bool(valid2.all()):
        states = states.loc[valid2].copy()
        log_ret = log_ret.loc[states.index]
        rf_annual = rf_annual.loc[states.index]
        dates = states.index
        if st_px is not None:
            st_px = st_px.loc[dates]
        if rf_px is not None:
            rf_px = rf_px.loc[dates]

    if cfg.standardize_states:
        scaler = StandardScaler(with_mean=True, with_std=True)

        fit_idx2 = dates
        fit_end2 = cfg.scaler_fit_end_date or cfg.pca_fit_end_date
        if cfg.strict_no_lookahead and (fit_end2 is None):
            raise ValueError(
                "DatasetConfig.scaler_fit_end_date (or pca_fit_end_date) must be set when "
                "strict_no_lookahead=True (prevents silent full-sample scaler fit / look-ahead)."
            )
        if fit_end2 is not None:
            fit_end2_dt = pd.to_datetime(fit_end2)
            fit_idx2 = dates[dates <= fit_end2_dt]
            if len(fit_idx2) < max(30, states.shape[1] + 5):
                # [v47] Never fall back to full-sample fit in strict mode (would be look-ahead).
                if cfg.strict_no_lookahead:
                    raise ValueError(
                        "Scaler fit window too short under strict_no_lookahead. "
                        "Choose an earlier scaler_fit_end_date (or pca_fit_end_date)."
                    )
                fit_idx2 = dates

        scaler.fit(states.loc[fit_idx2].values)
        states_arr = scaler.transform(states.values)
        states = pd.DataFrame(states_arr, index=dates, columns=state_names)

    # periods_per_year (heuristic)
    # For W-FRI, roughly 52. For daily, ~252.
    if cfg.freq.startswith("W"):
        ppy = 52
    elif cfg.freq.startswith("M"):
        ppy = 12
    else:
        ppy = 252

    return MarketDataset(
        dates=dates,
        asset_tickers=list(asset_px.columns),
        state_names=state_names,
        log_returns=log_ret,
        rf_annual=rf_annual,
        states=states,
        periods_per_year=ppy,
    )


def build_dataset_from_returns(
    simple_returns: pd.DataFrame,
    rf_monthly: Optional[pd.Series],
    cfg: DatasetConfig,
    periods_per_year: int = 12,
) -> MarketDataset:
    """
    Build MarketDataset directly from *monthly* simple returns (decimal), e.g. Ken French tables.

    Inputs:
      - simple_returns: DataFrame indexed by month-end timestamps, columns = assets, values = decimal simple returns
      - rf_monthly: Series indexed by month-end, decimal simple risk-free return for that month (same convention)
      - cfg: controls PCA/scaler fit window, etc.
      - periods_per_year: 12 for monthly

    Conventions:
      - We treat row at date t as the return realized over (t-1 -> t), known at t.
      - Per-step rf used by simulator is rf_step[t] = rf_monthly[t].
      - We store rf_annual := rf_monthly * periods_per_year so that rf_step = rf_annual * (1/periods_per_year).
    """
    # Align on intersection of available dates
    parts = [simple_returns.copy()]
    if rf_monthly is not None:
        parts.append(rf_monthly.to_frame("rf"))

    aligned = align_on_intersection(*parts)
    ret = aligned[0].astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    dates = ret.index

    if rf_monthly is None:
        rf_annual = pd.Series(0.0, index=dates, name="rf_annual")
    else:
        rf_m = aligned[1].iloc[:, 0].astype(float).reindex(dates).ffill().fillna(0.0)
        rf_annual = (rf_m * float(periods_per_year)).rename("rf_annual")

    # log returns
    log_ret = np.log1p(ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # states: reuse the same PCA/scaler logic as build_dataset()
    if cfg.use_pca_states:
        X_all = log_ret.values
        X_all = np.nan_to_num(X_all, nan=0.0)

        fit_idx = dates
        # [v47] strict look-ahead guard: require an explicit fit window for PCA in OOS settings
        if cfg.strict_no_lookahead and (cfg.pca_fit_end_date is None):
            raise ValueError(
                "DatasetConfig.pca_fit_end_date must be set when strict_no_lookahead=True "
                "(prevents silent full-sample PCA fit / look-ahead)."
            )
        if cfg.pca_fit_end_date is not None:
            fit_end = pd.to_datetime(cfg.pca_fit_end_date)
            fit_idx = dates[dates <= fit_end]
            if len(fit_idx) < max(30, cfg.n_pca_factors + 5):
                # [v47] Never fall back to full-sample fit in strict mode (would be look-ahead).
                if cfg.strict_no_lookahead:
                    raise ValueError(
                        "PCA fit window too short under strict_no_lookahead. "
                        "Choose an earlier pca_fit_end_date or reduce n_pca_factors."
                    )
                fit_idx = dates

        X_fit = log_ret.loc[fit_idx].values
        X_fit = np.nan_to_num(X_fit, nan=0.0)

        n_comp = int(min(cfg.n_pca_factors, X_fit.shape[0], X_fit.shape[1]))
        if n_comp < 1:
            raise ValueError(f"PCA needs at least 1 component; got n_comp={n_comp}.")
        pca = PCA(n_components=n_comp)
        pca.fit(X_fit)
        factors = pca.transform(X_all)
        state_names = [f"pca_{i+1}" for i in range(n_comp)]
        states = pd.DataFrame(factors[:, :n_comp], index=dates, columns=state_names)
    else:
        raise ValueError("build_dataset_from_returns currently supports only cfg.use_pca_states=True.")

    if cfg.standardize_states:
        scaler = StandardScaler(with_mean=True, with_std=True)
        fit_idx2 = dates
        fit_end2 = cfg.scaler_fit_end_date or cfg.pca_fit_end_date
        if cfg.strict_no_lookahead and (fit_end2 is None):
            raise ValueError(
                "DatasetConfig.scaler_fit_end_date (or pca_fit_end_date) must be set when "
                "strict_no_lookahead=True (prevents silent full-sample scaler fit / look-ahead)."
            )
        if fit_end2 is not None:
            fit_end2_dt = pd.to_datetime(fit_end2)
            fit_idx2 = dates[dates <= fit_end2_dt]
            if len(fit_idx2) < max(30, states.shape[1] + 5):
                # [v47] Never fall back to full-sample fit in strict mode (would be look-ahead).
                if cfg.strict_no_lookahead:
                    raise ValueError(
                        "Scaler fit window too short under strict_no_lookahead. "
                        "Choose an earlier scaler_fit_end_date (or pca_fit_end_date)."
                    )
                fit_idx2 = dates
        scaler.fit(states.loc[fit_idx2].values)
        states_arr = scaler.transform(states.values)
        states = pd.DataFrame(states_arr, index=dates, columns=state_names)

    return MarketDataset(
        dates=dates,
        asset_tickers=list(ret.columns),
        state_names=list(states.columns),
        log_returns=log_ret,
        rf_annual=rf_annual,
        states=states,
        periods_per_year=int(periods_per_year),
    )