from __future__ import annotations

import pandas as pd

from dynalloc_v2.selection_splits import SelectionSplitSpec, build_selection_blocks


def test_selection_split_trailing_holdout_matches_stage39_behavior():
    dates = pd.date_range('2000-01-31', periods=180, freq='ME')
    spec = SelectionSplitSpec(
        mode='trailing_holdout',
        train_start=dates[0],
        train_pool_end=dates[-1],
        cv_folds=3,
        min_train_months=60,
        selection_val_months=48,
        rolling_window=24,
        window_mode='rolling',
    )
    blocks = build_selection_blocks(dates, spec)
    assert len(blocks) == 1
    block = blocks[0]
    assert len(block['train_dates']) == 132
    assert len(block['val_dates']) == 48
    assert block['label'] == 'valholdout48_201101_201412'
    assert block['val_dates'][0] == dates[132]
    assert block['val_dates'][-1] == dates[-1]


def test_selection_split_expanding_cv_matches_stage39_behavior():
    dates = pd.date_range('2000-01-31', periods=180, freq='ME')
    spec = SelectionSplitSpec(
        mode='expanding_cv',
        train_start=dates[0],
        train_pool_end=dates[-1],
        cv_folds=3,
        min_train_months=60,
        selection_val_months=48,
        rolling_window=24,
        window_mode='rolling',
    )
    blocks = build_selection_blocks(dates, spec)
    assert len(blocks) == 3
    assert blocks[0]['label'] == 'valfold1_200501_200804'
    assert blocks[1]['label'] == 'valfold2_200805_201108'
    assert blocks[2]['label'] == 'valfold3_201109_201412'
    assert len(blocks[0]['train_dates']) == 60
    assert len(blocks[1]['train_dates']) > 60
    assert len(blocks[2]['train_dates']) > len(blocks[1]['train_dates'])
