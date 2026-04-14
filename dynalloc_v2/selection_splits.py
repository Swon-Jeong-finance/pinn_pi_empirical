from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

@dataclass(frozen=True)
class SelectionSplitSpec:
    mode: str
    train_pool_end: pd.Timestamp
    train_start: pd.Timestamp | None = None
    cv_folds: int = 3
    min_train_months: int = 60
    selection_val_months: int = 240
    rolling_window: int = 60
    window_mode: str = 'rolling'

def selection_pool_index(index: pd.DatetimeIndex, *, train_start: pd.Timestamp | None, train_pool_end: pd.Timestamp) -> pd.DatetimeIndex:
    pool = pd.DatetimeIndex(index)
    if train_start is not None:
        pool = pool[pool >= pd.Timestamp(train_start)]
    pool = pool[pool <= pd.Timestamp(train_pool_end)]
    return pd.DatetimeIndex(pool).sort_values()

def build_cv_blocks(index: pd.DatetimeIndex, *, train_start: pd.Timestamp | None = None, train_pool_end: pd.Timestamp, cv_folds: int = 3, min_train_months: int = 60, rolling_window: int = 60, window_mode: str = 'rolling') -> list[dict[str, Any]]:
    del rolling_window, window_mode  # preserve legacy v1 semantics: train expands across folds.
    train_pool = selection_pool_index(index, train_start=train_start, train_pool_end=train_pool_end)
    dev_len = int(len(train_pool))
    K = int(max(1, cv_folds))
    min_train_obs = int(max(12, min_train_months))
    if min_train_obs >= dev_len:
        raise RuntimeError(f'Not enough DEV data: selection_min_train_months={min_train_obs} >= dev_len={dev_len}.')
    val_months = int((dev_len - min_train_obs) // K)
    if val_months <= 0:
        raise RuntimeError(f'Not enough DEV data for rolling CV: dev_len={dev_len}, min_train={min_train_obs}, folds={K}.')
    blocks: list[dict[str, Any]] = []
    train_end_pos = int(min_train_obs - 1)
    for i in range(K):
        test_start_pos = int(train_end_pos + 1)
        test_end_pos = int(dev_len - 1) if i == K - 1 else int(min(dev_len - 1, test_start_pos + val_months - 1))
        train_dates = train_pool[:test_start_pos]
        val_dates = train_pool[test_start_pos : test_end_pos + 1]
        if len(train_dates) < 12 or len(val_dates) == 0:
            continue
        blocks.append({
            'label': f'valfold{i+1}_{val_dates[0].strftime("%Y%m")}_{val_dates[-1].strftime("%Y%m")}',
            'train_dates': train_dates,
            'val_dates': val_dates,
        })
        train_end_pos = int(test_end_pos)
    if not blocks:
        raise RuntimeError('No valid native CV blocks were generated')
    return blocks

def build_trailing_holdout_blocks(index: pd.DatetimeIndex, *, train_start: pd.Timestamp | None = None, train_pool_end: pd.Timestamp, val_months: int = 240, min_train_months: int = 60) -> list[dict[str, Any]]:
    train_pool = selection_pool_index(index, train_start=train_start, train_pool_end=train_pool_end)
    dev_len = int(len(train_pool))
    min_train_obs = int(max(12, min_train_months))
    val_obs = int(max(1, val_months))
    if val_obs >= dev_len:
        raise RuntimeError(f'Not enough DEV data for trailing holdout: val_months={val_obs} >= dev_len={dev_len}.')
    train_obs = int(dev_len - val_obs)
    if train_obs < min_train_obs:
        raise RuntimeError(
            f'Not enough TRAIN data left after trailing holdout: train_obs={train_obs}, '
            f'min_train_months={min_train_obs}, val_months={val_obs}, dev_len={dev_len}.'
        )
    train_dates = train_pool[:train_obs]
    val_dates = train_pool[train_obs:]
    if len(train_dates) < 12 or len(val_dates) == 0:
        raise RuntimeError('No valid trailing holdout block was generated')
    return [{
        'label': f'valholdout{len(val_dates)}_{val_dates[0].strftime("%Y%m")}_{val_dates[-1].strftime("%Y%m")}',
        'train_dates': train_dates,
        'val_dates': val_dates,
    }]

def build_selection_blocks(index: pd.DatetimeIndex, spec: SelectionSplitSpec) -> list[dict[str, Any]]:
    mode = str(spec.mode).strip().lower()
    if mode in {'trailing_holdout', 'trailing_holdout_1fold', 'single_holdout', 'single_trailing_validation'}:
        return build_trailing_holdout_blocks(
            index,
            train_start=spec.train_start,
            train_pool_end=spec.train_pool_end,
            val_months=int(spec.selection_val_months),
            min_train_months=int(spec.min_train_months),
        )
    if mode in {'expanding_cv', 'cv', 'rolling_cv'}:
        return build_cv_blocks(
            index,
            train_start=spec.train_start,
            train_pool_end=spec.train_pool_end,
            cv_folds=int(spec.cv_folds),
            min_train_months=int(spec.min_train_months),
            rolling_window=int(spec.rolling_window),
            window_mode=str(spec.window_mode),
        )
    raise ValueError(f'Unknown selection split mode: {spec.mode!r}')
