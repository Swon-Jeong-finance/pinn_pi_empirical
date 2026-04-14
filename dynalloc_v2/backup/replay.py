from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .experiments import run_experiment
from .oos_protocols import apply_oos_protocol
from .schema import Config
from .workspace import default_experiments_root


@dataclass
class ReplayArtifacts:
    output_dir: Path
    manifest_path: Path
    sample: str
    protocol: str
    fit_start: str
    fit_end: str
    replay_start: str
    replay_end: str
    zero_cost_summary: Path
    all_costs_summary: Path
    results_csv: Path


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    return p, yaml.safe_load(p.read_text(encoding='utf-8')) or {}


def _load_entry_config(entry: dict[str, Any]) -> Config:
    cfg_path = Path(entry['config_yaml']).expanduser().resolve()
    payload = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
    return Config.model_validate(payload)


def _entry_for_rank(manifest: dict[str, Any], rank: int) -> dict[str, Any]:
    for entry in manifest.get('entries') or []:
        if int(entry['rank']) == int(rank):
            return dict(entry)
    raise KeyError(f'rank={rank} not found in manifest entries')


def _window_from_selection_split(manifest: dict[str, Any], sample: str, *, block_label: str | None = None) -> tuple[str, str, str, str]:
    split = dict(manifest.get('selection_split') or {})
    if not split:
        raise RuntimeError('selection_split missing in manifest; cannot derive replay windows')
    fit_start = str(split['train_pool_start'])
    fit_end = str(split['final_oos_train_end'])
    sample_key = str(sample).strip().lower()
    if sample_key == 'insample_full':
        return fit_start, fit_end, fit_start, fit_end

    blocks = list(split.get('blocks') or [])
    if sample_key in {'selection_train', 'selection_validation'}:
        if not blocks:
            raise RuntimeError('selection_split.blocks missing in manifest')
        if block_label is not None:
            matches = [b for b in blocks if str(b.get('label')) == str(block_label)]
            if not matches:
                labels = ', '.join(str(b.get('label')) for b in blocks)
                raise KeyError(f'block_label={block_label!r} not found in selection_split.blocks. Available: {labels}')
            block = matches[0]
        elif len(blocks) == 1:
            block = blocks[0]
        else:
            labels = ', '.join(str(b.get('label')) for b in blocks)
            raise RuntimeError(
                f'sample={sample_key!r} is ambiguous because selection_split has multiple blocks; '
                f'pass --block-label. Available: {labels}'
            )
        if sample_key == 'selection_train':
            return fit_start, fit_end, str(block['train_start']), str(block['train_end'])
        return fit_start, fit_end, str(block['validation_start']), str(block['validation_end'])

    raise KeyError(f'Unsupported sample={sample!r}. Use insample_full, selection_train, or selection_validation.')


def _default_replay_output_dir(*, manifest_path: Path, rank: int, sample: str, protocol: str) -> Path:
    suite_name = manifest_path.stem
    return default_experiments_root() / 'replays' / suite_name / f'rank_{int(rank):03d}' / f'{sample}_{protocol}'


def replay_manifest_sample(
    *,
    manifest_path: str | Path,
    rank: int = 1,
    sample: str = 'insample_full',
    protocol: str = 'fixed',
    block_label: str | None = None,
    out_dir: str | Path | None = None,
    device_override: str | None = None,
    mc_rollouts_override: int | None = None,
    mc_sub_batch_override: int | None = None,
) -> ReplayArtifacts:
    manifest_resolved, manifest = _load_manifest(manifest_path)
    entry = _entry_for_rank(manifest, rank)
    cfg = _load_entry_config(entry)
    requested_protocol = str(protocol)
    cfg = apply_oos_protocol(cfg, requested_protocol, manifest=manifest, entry=entry)
    effective_protocol = str(cfg.split.protocol_label or requested_protocol)

    fit_start, fit_end, replay_start, replay_end = _window_from_selection_split(manifest, sample, block_label=block_label)
    cfg.split.train_window_mode = 'fixed'
    cfg.split.refit_every = 1
    cfg.split.rebalance_every = 1
    cfg.split.train_start = date.fromisoformat(fit_start[:10])
    cfg.split.fixed_train_end = date.fromisoformat(fit_end[:10])
    cfg.split.test_start = date.fromisoformat(replay_start[:10])
    cfg.split.end_date = date.fromisoformat(replay_end[:10])
    cfg.split.protocol_label = f'{effective_protocol}_replay_{sample}'

    if device_override is not None:
        cfg.ppgdpo.device = device_override
    if mc_rollouts_override is not None:
        cfg.ppgdpo.mc_rollouts = int(mc_rollouts_override)
    if mc_sub_batch_override is not None:
        cfg.ppgdpo.mc_sub_batch = int(mc_sub_batch_override)

    output_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else _default_replay_output_dir(
        manifest_path=manifest_resolved,
        rank=rank,
        sample=sample,
        protocol=effective_protocol,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.project.output_dir = output_dir
    cfg.project.name = f'{cfg.project.name}_{sample}_{effective_protocol}'

    artifacts = run_experiment(cfg)
    return ReplayArtifacts(
        output_dir=artifacts.output_dir,
        manifest_path=manifest_resolved,
        sample=sample,
        protocol=effective_protocol,
        fit_start=fit_start,
        fit_end=fit_end,
        replay_start=replay_start,
        replay_end=replay_end,
        zero_cost_summary=artifacts.summary_zero_cost,
        all_costs_summary=artifacts.summary_with_costs,
        results_csv=artifacts.output_dir / 'comparison_results.csv',
    )
