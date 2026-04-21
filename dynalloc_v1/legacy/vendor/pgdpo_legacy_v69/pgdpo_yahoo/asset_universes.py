"""Asset-universe loaders for v61.

The key v61 experiment changes the equity test-asset menu from French 49
industries to the canonical Ken French 25 Size–Book-to-Market portfolios,
while keeping the rest of the empirical skeleton (RF, FF factors, macro,
and optional bond asset) unchanged.
"""
from __future__ import annotations

import pandas as pd

from .french_data import (
    load_6_size_bm_portfolios_monthly,
    load_17_industry_portfolios_monthly,
    load_25_size_bm_portfolios_monthly,
    load_30_industry_portfolios_monthly,
    load_38_industry_portfolios_monthly,
    load_49_industry_portfolios_monthly,
    load_100_size_bm_portfolios_monthly,
    load_ff_factors_monthly,
)

ASSET_UNIVERSE_CHOICES: tuple[str, ...] = (
    'ff17ind',
    'ff30ind',
    'ff38ind',
    'ff49ind',
    'ff25_szbm',
    'ff6_szbm',
    'ff100_szbm',
    'ff_mkt',
    'ff1',
    'ff6',
    'ff17',
    'ff30',
    'ff38',
    'ff100',
    'fama_market',
    'custom',
)


def describe_asset_universe(asset_universe: str) -> str:
    s = str(asset_universe).lower()
    if s in {'ff17ind', 'ff17'}:
        return 'Ken French 17 Industry Portfolios'
    if s in {'ff30ind', 'ff30'}:
        return 'Ken French 30 Industry Portfolios'
    if s in {'ff38ind', 'ff38'}:
        return 'Ken French 38 Industry Portfolios'
    if s == 'ff49ind':
        return 'Ken French 49 Industry Portfolios'
    if s == 'ff25_szbm':
        return 'Ken French 25 Size–Book-to-Market Portfolios (5x5)'
    if s in {'ff6_szbm', 'ff6'}:
        return 'Ken French 6 Size–Book-to-Market Portfolios (2x3)'
    if s in {'ff100_szbm', 'ff100'}:
        return 'Ken French 100 Size–Book-to-Market Portfolios (10x10)'
    if s in {'ff_mkt', 'fama_market', 'ff1'}:
        return 'Ken French market total-return proxy (Mkt-RF + RF)'
    if s == 'custom':
        return 'Custom risky-asset universe'
    raise ValueError(
        f'Unknown asset_universe={asset_universe!r}. '
        f'Choose from {list(ASSET_UNIVERSE_CHOICES)}.'
    )


def load_equity_universe_monthly(asset_universe: str) -> pd.DataFrame:
    s = str(asset_universe).lower()
    if s in {'ff17ind', 'ff17'}:
        return load_17_industry_portfolios_monthly()
    if s in {'ff30ind', 'ff30'}:
        return load_30_industry_portfolios_monthly()
    if s in {'ff38ind', 'ff38'}:
        return load_38_industry_portfolios_monthly()
    if s == 'ff49ind':
        return load_49_industry_portfolios_monthly()
    if s == 'ff25_szbm':
        return load_25_size_bm_portfolios_monthly()
    if s in {'ff6_szbm', 'ff6'}:
        return load_6_size_bm_portfolios_monthly()
    if s in {'ff100_szbm', 'ff100'}:
        return load_100_size_bm_portfolios_monthly()
    if s in {'ff_mkt', 'fama_market', 'ff1'}:
        ff = load_ff_factors_monthly()
        out = (ff['Mkt-RF'] + ff['RF']).rename('EQ_MKT').to_frame()
        out.index.name = ff.index.name
        return out
    if s == 'custom':
        raise ValueError(
            "asset_universe='custom' must be resolved by the caller using custom_risky_assets; "
            "load_equity_universe_monthly('custom') is not supported directly."
        )
    raise ValueError(
        f'Unknown asset_universe={asset_universe!r}. '
        f'Choose from {list(ASSET_UNIVERSE_CHOICES)}.'
    )
