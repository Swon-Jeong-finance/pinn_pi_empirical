from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .bridge_common import _load_yaml
from .legacy_bridge import build_v1_lane_bundle
from .oos_protocols import manifest_protocol_payload


@dataclass
class SuiteBridgeArtifacts:
    out_dir: Path
    manifest_yaml: Path
    entry_count: int


def _read_selected_specs(v1_root: Path, config_stem: str) -> list[str]:
    payload = _load_yaml(v1_root / 'outputs' / config_stem / 'selected_spec.yaml')
    specs = list(payload.get('selected_specs') or [])
    primary = payload.get('primary_selected_spec')
    if not specs and primary:
        specs = [str(primary)]
    return [str(x) for x in specs]


def build_v1_lane_suite(
    *,
    v1_root: str | Path,
    config_stem: str,
    out_dir: str | Path,
    fred_api_key: str | None = None,
    top_k: int = 2,
    factor_mode: str = 'ff5_curve_core',
    refresh_fred: bool = False,
    risk_aversion: float = 5.0,
    asset_universe_override: str | None = None,
    split_profile_override: str | None = None,
    split_train_start_override: str | None = None,
    split_train_pool_end_override: str | None = None,
    split_test_start_override: str | None = None,
    split_end_date_override: str | None = None,
) -> SuiteBridgeArtifacts:
    v1_root = Path(v1_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _read_selected_specs(v1_root, config_stem)
    if not specs:
        raise RuntimeError(f'No selected_specs found for {config_stem!r}')
    specs = specs[: max(1, int(top_k))]

    entries: list[dict[str, Any]] = []
    for rank, spec in enumerate(specs, start=1):
        rank_dir = out_dir / f'rank_{rank:03d}'
        artifacts = build_v1_lane_bundle(
            v1_root=v1_root,
            config_stem=config_stem,
            out_dir=rank_dir,
            fred_api_key=fred_api_key,
            selected_rank=rank,
            spec=spec,
            factor_mode=factor_mode,
            refresh_fred=refresh_fred,
            risk_aversion=risk_aversion,
            asset_universe_override=asset_universe_override,
            split_profile_override=split_profile_override,
            split_train_start_override=split_train_start_override,
            split_train_pool_end_override=split_train_pool_end_override,
            split_test_start_override=split_test_start_override,
            split_end_date_override=split_end_date_override,
        )
        entries.append({
            'rank': rank,
            'spec': spec,
            'bundle_dir': str(artifacts.out_dir),
            'config_yaml': str(artifacts.config_yaml),
            'metadata_yaml': str(artifacts.metadata_yaml),
            'returns_csv': str(artifacts.returns_csv),
            'states_csv': str(artifacts.states_csv),
            'factors_csv': str(artifacts.factors_csv),
        })

    manifest = {
        'suite_name': f'{config_stem}_top{len(entries)}',
        'source_v1_root': str(v1_root),
        'config_stem': config_stem,
        'factor_mode': factor_mode,
        'asset_universe_override': str(asset_universe_override).lower() if asset_universe_override is not None else None,
        'split_profile_override': split_profile_override,
        'split_overrides': {
            'train_start': split_train_start_override,
            'train_pool_end': split_train_pool_end_override,
            'test_start': split_test_start_override,
            'end_date': split_end_date_override,
        },
        'top_k': len(entries),
        **manifest_protocol_payload(),
        'entries': entries,
    }
    manifest_yaml = out_dir / 'suite_manifest.yaml'
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')
    return SuiteBridgeArtifacts(out_dir=out_dir, manifest_yaml=manifest_yaml, entry_count=len(entries))
