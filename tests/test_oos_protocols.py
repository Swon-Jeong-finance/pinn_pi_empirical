from __future__ import annotations

from datetime import date
from pathlib import Path

from dynalloc_v2.cli import load_config
from dynalloc_v2.oos_protocols import (
    DEFAULT_OOS_PROTOCOLS,
    apply_oos_protocol,
    manifest_protocol_payload,
    resolve_oos_protocols,
)

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_protocol_payload_matches_stage39_defaults():
    payload = manifest_protocol_payload()
    assert payload['oos_protocols_default'] == ['fixed', 'expanding_annual', 'rolling20y_annual']
    assert payload['oos_protocol_descriptions']['fixed'].startswith('fit once on the pre-test pool')
    assert payload['oos_protocol_descriptions']['expanding_annual'].startswith('refit on an expanding window')
    assert payload['oos_protocol_descriptions']['rolling20y_annual'].startswith('refit on a rolling 20-year window')


def test_resolve_oos_protocols_upgrades_legacy_defaults():
    manifest = {'oos_protocols_default': ['fixed', 'expanding_annual']}
    assert resolve_oos_protocols(manifest, None) == list(DEFAULT_OOS_PROTOCOLS)


def test_apply_oos_protocol_preserves_stage39_behavior():
    cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    cfg.split.train_start = date(2000, 1, 31)
    cfg.split.test_start = date(2003, 1, 31)
    cfg.split.end_date = date(2004, 12, 31)
    cfg.split.refit_every = 24

    fixed = apply_oos_protocol(cfg, 'fixed')
    assert fixed.split.train_window_mode == 'fixed'
    assert fixed.split.refit_every == 24
    assert fixed.split.rebalance_every == 1
    assert fixed.split.rolling_train_months is None

    expanding = apply_oos_protocol(cfg, 'expanding_annual')
    assert expanding.split.train_window_mode == 'expanding'
    assert expanding.split.refit_every == 12
    assert expanding.split.rebalance_every == 1
    assert expanding.split.rolling_train_months is None

    rolling = apply_oos_protocol(cfg, 'rolling20y_annual')
    assert rolling.split.train_window_mode == 'rolling'
    assert rolling.split.refit_every == 12
    assert rolling.split.rebalance_every == 1
    assert rolling.split.rolling_train_months == 240


def test_manifest_protocol_payload_accepts_selected_rolling_default():
    payload = manifest_protocol_payload(['fixed', 'expanding_annual', 'rolling_selected_annual'])
    assert payload['oos_protocols_default'] == ['fixed', 'expanding_annual', 'rolling_selected_annual']
    assert payload['oos_protocol_descriptions']['rolling_selected_annual'].startswith('refit on a validation-selected rolling window')


def test_apply_oos_protocol_resolves_selected_rolling_from_entry():
    cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    cfg.split.train_start = date(2000, 1, 31)
    cfg.split.test_start = date(2003, 1, 31)
    cfg.split.end_date = date(2004, 12, 31)

    manifest = {'validation_protocol_selection': {'default_rolling_train_months': 120}}
    entry = {
        'selected_rolling_train_months': 84,
        'selected_oos_protocols': {
            'rolling_selected_annual': {
                'rolling_train_months': 84,
            }
        },
    }
    rolling = apply_oos_protocol(cfg, 'rolling_selected_annual', manifest=manifest, entry=entry)
    assert rolling.split.train_window_mode == 'rolling'
    assert rolling.split.refit_every == 12
    assert rolling.split.rebalance_every == 1
    assert rolling.split.rolling_train_months == 84


def test_apply_oos_protocol_supports_dynamic_rolling_months():
    cfg = load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')
    cfg.split.train_start = date(2000, 1, 31)
    cfg.split.test_start = date(2003, 1, 31)
    cfg.split.end_date = date(2004, 12, 31)

    rolling = apply_oos_protocol(cfg, 'rolling84m_annual')
    assert rolling.split.train_window_mode == 'rolling'
    assert rolling.split.refit_every == 12
    assert rolling.split.rebalance_every == 1
    assert rolling.split.rolling_train_months == 84
