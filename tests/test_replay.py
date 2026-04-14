from pathlib import Path
import yaml

from dynalloc_v2.replay import _window_from_selection_split


def test_window_from_selection_split_trailing_holdout():
    manifest = {
        'selection_split': {
            'train_pool_start': '1964-01-01',
            'final_oos_train_end': '1999-12-31',
            'blocks': [
                {
                    'label': 'valholdout240_198001_199912',
                    'train_start': '1964-01-31',
                    'train_end': '1979-12-31',
                    'validation_start': '1980-01-31',
                    'validation_end': '1999-12-31',
                }
            ],
        }
    }
    fit_start, fit_end, replay_start, replay_end = _window_from_selection_split(manifest, 'selection_validation')
    assert fit_start == '1964-01-01'
    assert fit_end == '1999-12-31'
    assert replay_start == '1980-01-31'
    assert replay_end == '1999-12-31'


def test_window_from_selection_split_requires_block_label_for_multi_block():
    manifest = {
        'selection_split': {
            'train_pool_start': '1964-01-01',
            'final_oos_train_end': '1999-12-31',
            'blocks': [
                {
                    'label': 'valfold1',
                    'train_start': '1964-01-31',
                    'train_end': '1973-12-31',
                    'validation_start': '1974-01-31',
                    'validation_end': '1982-08-31',
                },
                {
                    'label': 'valfold2',
                    'train_start': '1964-01-31',
                    'train_end': '1982-08-31',
                    'validation_start': '1982-09-30',
                    'validation_end': '1991-04-30',
                },
            ],
        }
    }
    try:
        _window_from_selection_split(manifest, 'selection_validation')
    except RuntimeError as exc:
        assert 'block-label' in str(exc)
    else:
        raise AssertionError('Expected RuntimeError for ambiguous validation sample')
