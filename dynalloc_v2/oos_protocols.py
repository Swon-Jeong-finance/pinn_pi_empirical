from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from .schema import Config

DEFAULT_OOS_PROTOCOLS = ['fixed', 'expanding_annual', 'rolling20y_annual']
ROLLING_SELECTED_PROTOCOL = 'rolling_selected_annual'
SELECTED_PROTOCOL = 'selected_protocol'

_ROLLING_MONTHS_RE = re.compile(r'^rolling(?P<months>\d+)m_annual$')
_ROLLING_YEARS_RE = re.compile(r'^rolling(?P<years>\d+)y_annual$')


@dataclass(frozen=True)
class OOSProtocolSpec:
    name: str
    description: str
    train_window_mode: Literal['fixed', 'expanding', 'rolling']
    rebalance_every: int
    refit_every: int | None = None
    rolling_train_months: int | None = None


_PROTOCOL_SPECS: dict[str, OOSProtocolSpec] = {
    'fixed': OOSProtocolSpec(
        name='fixed',
        description='fit once on the pre-test pool and rebalance monthly without expanding re-estimation',
        train_window_mode='fixed',
        rebalance_every=1,
        refit_every=None,
        rolling_train_months=None,
    ),
    'expanding_annual': OOSProtocolSpec(
        name='expanding_annual',
        description='refit on an expanding window once per year and rebalance monthly between refits',
        train_window_mode='expanding',
        rebalance_every=1,
        refit_every=12,
        rolling_train_months=None,
    ),
    'rolling20y_annual': OOSProtocolSpec(
        name='rolling20y_annual',
        description='refit on a rolling 20-year window once per year (warm-starting from the available history until the full window is reached) and rebalance monthly between refits',
        train_window_mode='rolling',
        rebalance_every=1,
        refit_every=12,
        rolling_train_months=240,
    ),
    ROLLING_SELECTED_PROTOCOL: OOSProtocolSpec(
        name=ROLLING_SELECTED_PROTOCOL,
        description='refit on a validation-selected rolling window once per year (warm-starting from the available history until the full window is reached) and rebalance monthly between refits',
        train_window_mode='rolling',
        rebalance_every=1,
        refit_every=12,
        rolling_train_months=None,
    ),
    SELECTED_PROTOCOL: OOSProtocolSpec(
        name=SELECTED_PROTOCOL,
        description='use the protocol selected for this entry during native selection',
        train_window_mode='fixed',
        rebalance_every=1,
        refit_every=None,
        rolling_train_months=None,
    ),
}


def _dynamic_rolling_spec(name: str) -> OOSProtocolSpec | None:
    value = str(name).strip()
    month_match = _ROLLING_MONTHS_RE.fullmatch(value)
    if month_match is not None:
        months = int(month_match.group('months'))
        return OOSProtocolSpec(
            name=value,
            description=f'refit on a rolling {months}-month window once per year (warm-starting from the available history until the full window is reached) and rebalance monthly between refits',
            train_window_mode='rolling',
            rebalance_every=1,
            refit_every=12,
            rolling_train_months=months,
        )
    year_match = _ROLLING_YEARS_RE.fullmatch(value)
    if year_match is not None:
        years = int(year_match.group('years'))
        return OOSProtocolSpec(
            name=value,
            description=f'refit on a rolling {years}-year window once per year (warm-starting from the available history until the full window is reached) and rebalance monthly between refits',
            train_window_mode='rolling',
            rebalance_every=1,
            refit_every=12,
            rolling_train_months=12 * years,
        )
    return None


def protocol_spec(name: str) -> OOSProtocolSpec:
    value = str(name).strip()
    dynamic = _dynamic_rolling_spec(value)
    if dynamic is not None:
        return dynamic
    if value not in _PROTOCOL_SPECS:
        raise ValueError(f'Unsupported OOS protocol: {value!r}')
    return _PROTOCOL_SPECS[value]


def protocol_descriptions(names: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    payload = list(DEFAULT_OOS_PROTOCOLS if names is None else names)
    return {name: protocol_spec(name).description for name in payload}


def manifest_protocol_payload(names: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    payload = list(DEFAULT_OOS_PROTOCOLS if names is None else names)
    return {
        'oos_protocols_default': list(payload),
        'oos_protocol_descriptions': protocol_descriptions(payload),
    }


def resolve_oos_protocols(manifest: dict[str, Any], requested: list[str] | tuple[str, ...] | None) -> list[str]:
    if requested is not None:
        payload = list(requested)
    else:
        payload = list(manifest.get('oos_protocols_default') or DEFAULT_OOS_PROTOCOLS)
        legacy_defaults = ['fixed', 'expanding_annual']
        if payload == legacy_defaults:
            payload = list(DEFAULT_OOS_PROTOCOLS)
    resolved: list[str] = []
    for protocol in payload:
        value = str(protocol).strip()
        protocol_spec(value)
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        raise ValueError('At least one OOS protocol is required.')
    return resolved


def _selected_protocol_payload(manifest: dict[str, Any] | None, entry: dict[str, Any] | None) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if entry is not None:
        selected = entry.get('selected_oos_protocols') or {}
        for key in (SELECTED_PROTOCOL, ROLLING_SELECTED_PROTOCOL):
            payload = selected.get(key)
            if isinstance(payload, dict):
                candidates.append(dict(payload))
        if entry.get('selected_protocol_name') is not None:
            candidates.append(
                {
                    'name': SELECTED_PROTOCOL,
                    'source_protocol': entry.get('selected_protocol_name'),
                    'train_window_mode': entry.get('train_window_mode') or entry.get('selection_protocol_train_window_mode'),
                    'rolling_train_months': entry.get('selected_rolling_train_months') or entry.get('selection_protocol_rolling_train_months'),
                }
            )
    if manifest is not None:
        selected_defaults = manifest.get('selected_oos_protocol_defaults') or {}
        payload = selected_defaults.get(SELECTED_PROTOCOL)
        if isinstance(payload, dict):
            candidates.append(dict(payload))
        payload = selected_defaults.get(ROLLING_SELECTED_PROTOCOL)
        if isinstance(payload, dict):
            candidates.append(dict(payload))
        validation = manifest.get('validation_protocol_selection') or {}
        payload = validation.get('default_selected_protocol')
        if isinstance(payload, dict):
            candidates.append(dict(payload))
    for payload in candidates:
        if payload:
            return payload
    return {}


def _selected_rolling_train_months(manifest: dict[str, Any] | None, entry: dict[str, Any] | None) -> int:
    candidates: list[Any] = []
    payload = _selected_protocol_payload(manifest, entry)
    if payload:
        candidates.append(payload.get('rolling_train_months'))
    if entry is not None:
        candidates.append(entry.get('selected_rolling_train_months'))
    if manifest is not None:
        validation = manifest.get('validation_protocol_selection') or {}
        candidates.append(validation.get('default_rolling_train_months'))
        candidates.append(validation.get('fallback_rolling_train_months'))
    candidates.append(240)
    for candidate in candidates:
        if candidate is None:
            continue
        months = int(candidate)
        if months > 0:
            return months
    return 240


def _apply_protocol_fields(out: Config, spec: OOSProtocolSpec, *, protocol_label: str | None = None) -> Config:
    out.split.protocol_label = protocol_label or spec.name
    out.split.train_window_mode = spec.train_window_mode
    out.split.rolling_train_months = spec.rolling_train_months
    out.split.rebalance_every = int(spec.rebalance_every)
    if spec.refit_every is None:
        out.split.refit_every = max(int(out.split.refit_every), 1)
    else:
        out.split.refit_every = int(spec.refit_every)
    return out


def apply_oos_protocol(
    cfg: Config,
    protocol: str,
    *,
    manifest: dict[str, Any] | None = None,
    entry: dict[str, Any] | None = None,
) -> Config:
    out = cfg.model_copy(deep=True)
    spec = protocol_spec(protocol)
    if spec.name == ROLLING_SELECTED_PROTOCOL:
        out.split.protocol_label = spec.name
        out.split.train_window_mode = 'rolling'
        out.split.rolling_train_months = _selected_rolling_train_months(manifest, entry)
        out.split.rebalance_every = int(spec.rebalance_every)
        out.split.refit_every = int(spec.refit_every or 12)
        return out
    if spec.name == SELECTED_PROTOCOL:
        payload = _selected_protocol_payload(manifest, entry)
        source_protocol = str(payload.get('source_protocol') or payload.get('actual_protocol') or payload.get('name') or '').strip()
        if source_protocol and source_protocol not in {SELECTED_PROTOCOL, ROLLING_SELECTED_PROTOCOL}:
            selected_spec = protocol_spec(source_protocol)
            out = _apply_protocol_fields(out, selected_spec, protocol_label=source_protocol)
        else:
            mode = str(payload.get('train_window_mode') or 'fixed').strip().lower()
            if mode not in {'fixed', 'expanding', 'rolling'}:
                mode = 'fixed'
            out.split.protocol_label = source_protocol or SELECTED_PROTOCOL
            out.split.train_window_mode = mode  # type: ignore[assignment]
            rolling_months = payload.get('rolling_train_months')
            out.split.rolling_train_months = int(rolling_months) if rolling_months is not None else None
            out.split.rebalance_every = int(payload.get('rebalance_every') or 1)
            if mode == 'fixed':
                out.split.refit_every = max(int(out.split.refit_every), 1)
            else:
                out.split.refit_every = int(payload.get('refit_every') or 12)
        rolling_months = payload.get('rolling_train_months')
        if rolling_months is not None and out.split.train_window_mode == 'rolling':
            out.split.rolling_train_months = int(rolling_months)
        return out
    return _apply_protocol_fields(out, spec)
