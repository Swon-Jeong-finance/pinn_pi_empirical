from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


DEFAULT_SPLIT_PROFILE = 'cv2000_final20y'


@dataclass(frozen=True)
class ExperimentWindowSpec:
    profile: str
    train_start: str
    train_pool_end: str
    test_start: str
    end_date: str
    description: str


_WINDOW_SPECS: dict[str, ExperimentWindowSpec] = {
    'cv2000_final20y': ExperimentWindowSpec(
        profile='cv2000_final20y',
        train_start='1964-01-01',
        train_pool_end='1999-12-31',
        test_start='2000-01-01',
        end_date='2019-12-31',
        description='validation 1980-1999, final OOS 2000-2019',
    ),
    'cv2006_final20y': ExperimentWindowSpec(
        profile='cv2006_final20y',
        train_start='1964-01-01',
        train_pool_end='2005-12-31',
        test_start='2006-01-01',
        end_date='2025-12-31',
        description='validation 1986-2005, final OOS 2006-2025',
    ),
}


def available_split_profiles() -> list[str]:
    return list(_WINDOW_SPECS.keys())


def split_profile_spec(profile: str) -> ExperimentWindowSpec:
    key = str(profile).strip()
    if key not in _WINDOW_SPECS:
        raise ValueError(f'Unsupported split profile: {profile!r}')
    return _WINDOW_SPECS[key]


def _coerce_date_str(value: Any | None) -> str | None:
    if value is None:
        return None
    return str(pd.Timestamp(value).date())


def normalize_split_payload(payload: dict[str, Any]) -> dict[str, str]:
    out = {
        'train_start': _coerce_date_str(payload.get('train_start')),
        'train_pool_end': _coerce_date_str(payload.get('train_pool_end')),
        'test_start': _coerce_date_str(payload.get('test_start')),
        'end_date': _coerce_date_str(payload.get('end_date')),
    }
    if out['train_pool_end'] is None and out['test_start'] is not None:
        test_start_ts = pd.Timestamp(out['test_start'])
        out['train_pool_end'] = str((test_start_ts - pd.offsets.MonthEnd(1)).date())
    missing = [key for key, value in out.items() if value is None]
    if missing:
        raise ValueError(f'Incomplete split payload; missing {missing}')
    train_start = pd.Timestamp(out['train_start'])
    train_pool_end = pd.Timestamp(out['train_pool_end'])
    test_start = pd.Timestamp(out['test_start'])
    end_date = pd.Timestamp(out['end_date'])
    if train_start > train_pool_end:
        raise ValueError('train_start must be on or before train_pool_end')
    if train_pool_end >= test_start:
        raise ValueError('train_pool_end must be strictly before test_start')
    if test_start > end_date:
        raise ValueError('test_start must be on or before end_date')
    return out


def profile_payload(profile: str) -> dict[str, str]:
    spec = split_profile_spec(profile)
    return normalize_split_payload({
        'train_start': spec.train_start,
        'train_pool_end': spec.train_pool_end,
        'test_start': spec.test_start,
        'end_date': spec.end_date,
    })


def resolve_split_payload(
    *,
    base_profile: str | None = None,
    fallback_payload: dict[str, Any] | None = None,
    split_profile_override: str | None = None,
    train_start_override: Any | None = None,
    train_pool_end_override: Any | None = None,
    test_start_override: Any | None = None,
    end_date_override: Any | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    profile_key = str(split_profile_override or base_profile or '').strip() or None
    if profile_key is not None:
        payload = profile_payload(profile_key)
        source = 'profile'
    elif fallback_payload:
        payload = normalize_split_payload(dict(fallback_payload))
        source = 'fallback_payload'
    else:
        profile_key = DEFAULT_SPLIT_PROFILE
        payload = profile_payload(profile_key)
        source = 'default_profile'

    overrides = {
        'train_start': _coerce_date_str(train_start_override),
        'train_pool_end': _coerce_date_str(train_pool_end_override),
        'test_start': _coerce_date_str(test_start_override),
        'end_date': _coerce_date_str(end_date_override),
    }
    requested = {k: v for k, v in overrides.items() if v is not None}
    if requested:
        payload = dict(payload)
        payload.update(requested)
        payload = normalize_split_payload(payload)

    meta = {
        'split_profile': profile_key,
        'split_source': source,
        'split_overrides': requested,
        'split_description': split_profile_spec(profile_key).description if profile_key in _WINDOW_SPECS else None,
    }
    return payload, meta
