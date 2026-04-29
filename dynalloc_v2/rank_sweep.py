from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import json
import shutil
from typing import Any

import pandas as pd
import yaml

from .schema import Config
from .experiments import run_experiment
from .oos_protocols import apply_oos_protocol, resolve_oos_protocols


@dataclass
class RankSweepArtifacts:
    out_dir: Path
    manifest_yaml: Path
    zero_cost_summary: Path
    all_costs_summary: Path
    progress_csv: Path
    results_csv: Path

def _resolve_manifest_transaction_cost_bps(
    manifest: dict[str, Any],
    entry: dict[str, Any],
) -> float | None:
    """Look for a transaction_cost_bps value in the manifest/entry.

    Precedence (first non-None wins):
      1. entry-level 'comparison_transaction_cost_bps'
      2. entry-level 'transaction_cost_bps'
      3. manifest 'comparison.transaction_cost_bps'
      4. manifest top-level 'comparison_transaction_cost_bps'
      5. manifest 'ppgdpo_lite.transaction_cost_bps'
    Returns None if none of these are set, in which case the per-rank YAML
    value is left untouched.
    """
    candidates: list[Any] = [
        entry.get('comparison_transaction_cost_bps'),
        entry.get('transaction_cost_bps'),
        (manifest.get('comparison') or {}).get('transaction_cost_bps'),
        manifest.get('comparison_transaction_cost_bps'),
        (manifest.get('ppgdpo_lite') or {}).get('transaction_cost_bps'),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        return float(candidate)
    return None

def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def _protocol_output_dir(base_output_dir: Path, protocol: str) -> Path:
    return base_output_dir.parent / f'{base_output_dir.name}_{protocol}'

def _parse_device_pool(value: str | None) -> list[str]:
    if value is None:
        return []
    raw = str(value).replace(',', ' ').strip()
    if not raw:
        return []
    return [token for token in raw.split() if token]

def _run_single_rank_protocol(
    *,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    requested_protocol: str,
    rank_root: str,
    device_override: str | None,
    mc_rollouts_override: int | None,
    mc_sub_batch_override: int | None,
    transaction_cost_bps_override: float | None,
    risky_cap_override: float | None,
    cash_floor_override: float | None,
    emit_legacy_fixed_layout: bool,
) -> dict[str, Any]:
    rank = int(entry['rank'])
    spec = str(entry['spec'])
    cfg_path = Path(entry['config_yaml'])
    payload = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
    base_cfg = Config.model_validate(payload)
    base_output_dir = Path(base_cfg.project.output_dir)
    rank_dir = Path(rank_root) / f'rank_{rank:03d}'
    rank_dir.mkdir(parents=True, exist_ok=True)

    cfg = apply_oos_protocol(base_cfg, requested_protocol, manifest=manifest, entry=entry)
    effective_protocol = str(cfg.split.protocol_label or requested_protocol)
    cfg.project.output_dir = _protocol_output_dir(base_output_dir, effective_protocol)
    cfg.project.name = f'{cfg.project.name}_{effective_protocol}'
    if device_override is not None:
        cfg.ppgdpo.device = device_override
        if hasattr(cfg, 'pipinn'):
            cfg.pipinn.device = device_override
    if mc_rollouts_override is not None:
        cfg.ppgdpo.mc_rollouts = int(mc_rollouts_override)
    if mc_sub_batch_override is not None:
        cfg.ppgdpo.mc_sub_batch = int(mc_sub_batch_override)
    if transaction_cost_bps_override is not None:
        cfg.comparison.transaction_cost_bps = float(transaction_cost_bps_override)
    else:
        manifest_tc = _resolve_manifest_transaction_cost_bps(manifest, entry)
        if manifest_tc is not None:
            cfg.comparison.transaction_cost_bps = float(manifest_tc)
    if risky_cap_override is not None:
        cfg.policy.risky_cap = float(risky_cap_override)
    if cash_floor_override is not None:
        cfg.policy.cash_floor = float(cash_floor_override)

    artifacts = run_experiment(cfg)

    protocol_dir = rank_dir / 'protocols' / effective_protocol
    comp_dir = protocol_dir / 'comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)
    zero_dst = comp_dir / 'comparison_cross_modes_zero_cost_summary.csv'
    cost_dst = comp_dir / 'comparison_cross_modes_all_costs_summary.csv'
    res_dst = comp_dir / 'comparison_results.csv'
    _safe_copy(artifacts.summary_zero_cost, zero_dst)
    _safe_copy(artifacts.summary_with_costs, cost_dst)
    _safe_copy(artifacts.output_dir / 'comparison_results.csv', res_dst)
    _safe_copy(artifacts.plan_yaml, protocol_dir / 'resolved_config.yaml')

    done_payload = {
        'rank': rank,
        'spec': spec,
        'oos_protocol_requested': requested_protocol,
        'oos_protocol': effective_protocol,
        'output_dir': str(artifacts.output_dir),
        'device': device_override or (cfg.pipinn.device if str(getattr(cfg, 'optimizer_backend', 'ppgdpo')).lower() == 'pipinn' else cfg.ppgdpo.device),
        'optimizer_backend': str(getattr(cfg, 'optimizer_backend', 'ppgdpo')),
        'mc_rollouts': int(cfg.ppgdpo.mc_rollouts),
        'mc_sub_batch': int(cfg.ppgdpo.mc_sub_batch),
        'train_window_mode': str(cfg.split.train_window_mode),
        'refit_every': int(cfg.split.refit_every),
        'rebalance_every': int(cfg.split.rebalance_every),
        'rolling_train_months': int(cfg.split.rolling_train_months) if cfg.split.rolling_train_months is not None else None,
        'transaction_cost_bps': float(cfg.comparison.transaction_cost_bps),
        'risky_cap': float(cfg.policy.risky_cap),
        'cash_floor': float(cfg.policy.cash_floor),
    }
    (protocol_dir / '_done.yaml').write_text(yaml.safe_dump(done_payload, sort_keys=False), encoding='utf-8')
    (protocol_dir / 'stage21_rank_report.yaml').write_text(yaml.safe_dump(done_payload, sort_keys=False), encoding='utf-8')

    if effective_protocol == 'fixed' and emit_legacy_fixed_layout:
        legacy_comp_dir = rank_dir / 'comparison'
        legacy_comp_dir.mkdir(parents=True, exist_ok=True)
        _safe_copy(zero_dst, legacy_comp_dir / 'comparison_cross_modes_zero_cost_summary.csv')
        _safe_copy(cost_dst, legacy_comp_dir / 'comparison_cross_modes_all_costs_summary.csv')
        _safe_copy(res_dst, legacy_comp_dir / 'comparison_results.csv')
        _safe_copy(artifacts.plan_yaml, rank_dir / 'resolved_config.yaml')
        (rank_dir / '_done.yaml').write_text(yaml.safe_dump(done_payload, sort_keys=False), encoding='utf-8')
        (rank_dir / 'stage21_rank_report.yaml').write_text(yaml.safe_dump(done_payload, sort_keys=False), encoding='utf-8')

    return {
        'rank': rank,
        'spec': spec,
        'requested_protocol': requested_protocol,
        'effective_protocol': effective_protocol,
        'zero_dst': str(zero_dst),
        'cost_dst': str(cost_dst),
        'res_dst': str(res_dst),
        'output_dir': str(artifacts.output_dir),
        'train_window_mode': str(cfg.split.train_window_mode),
        'refit_every': int(cfg.split.refit_every),
        'rebalance_every': int(cfg.split.rebalance_every),
        'rolling_train_months': int(cfg.split.rolling_train_months) if cfg.split.rolling_train_months is not None else None,
        'mc_rollouts': int(cfg.ppgdpo.mc_rollouts),
        'mc_sub_batch': int(cfg.ppgdpo.mc_sub_batch),
        'transaction_cost_bps': float(cfg.comparison.transaction_cost_bps),
        'risky_cap': float(cfg.policy.risky_cap),
        'cash_floor': float(cfg.policy.cash_floor),
    }

def run_rank_sweep(
    manifest_path: str | Path,
    device_override: str | None = None,
    mc_rollouts_override: int | None = None,
    mc_sub_batch_override: int | None = None,
    transaction_cost_bps_override: float | None = None,
    risky_cap_override: float | None = None,
    cash_floor_override: float | None = None,
    oos_protocols: list[str] | tuple[str, ...] | None = None,
    emit_legacy_fixed_layout: bool = False,
    max_parallel: int = 1,
    parallel_backend: str = 'process',
) -> RankSweepArtifacts:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8')) or {}
    entries = list(manifest.get('entries') or [])
    if not entries:
        raise RuntimeError(f'No entries found in manifest: {manifest_path}')
    protocols_requested = resolve_oos_protocols(manifest, oos_protocols)
    emit_legacy_fixed_layout = bool(emit_legacy_fixed_layout)

    suite_root = manifest_path.parent
    rank_root = suite_root / 'rank_sweep'
    rank_root.mkdir(parents=True, exist_ok=True)

    progress_rows: list[dict[str, Any]] = []
    zero_rows: list[pd.DataFrame] = []
    cost_rows: list[pd.DataFrame] = []
    result_rows: list[pd.DataFrame] = []

    tasks: list[dict[str, Any]] = []
    max_workers = max(1, int(max_parallel))
    device_pool = _parse_device_pool(device_override)
    for idx, entry in enumerate(sorted(entries, key=lambda x: int(x['rank']))):
        for requested_protocol in protocols_requested:
            if not device_pool:
                assigned_device = None
            elif max_workers > 1:
                assigned_device = device_pool[idx % len(device_pool)]
            else:
                assigned_device = device_pool[0]
            tasks.append({
                'entry': entry,
                'manifest': manifest,
                'requested_protocol': requested_protocol,
                'rank_root': str(rank_root),
                'device_override': assigned_device,
                'mc_rollouts_override': mc_rollouts_override,
                'mc_sub_batch_override': mc_sub_batch_override,
                'transaction_cost_bps_override': transaction_cost_bps_override,
                'risky_cap_override': risky_cap_override,
                'cash_floor_override': cash_floor_override,
                'emit_legacy_fixed_layout': emit_legacy_fixed_layout,
            })

    if parallel_backend not in {'process', 'thread'}:
        raise ValueError(f'Unknown parallel_backend={parallel_backend!r}. Use process or thread.')
    executor_cls: type[Executor] = ProcessPoolExecutor if parallel_backend == 'process' else ThreadPoolExecutor

    if max_workers == 1:
        completed = [_run_single_rank_protocol(**task) for task in tasks]
    else:
        completed = []
        executor_kwargs: dict[str, Any] = {'max_workers': max_workers}
        if parallel_backend == 'process':
            executor_kwargs['mp_context'] = mp.get_context('spawn')
        with executor_cls(**executor_kwargs) as ex:
            futures = [ex.submit(_run_single_rank_protocol, **task) for task in tasks]
            for fut in futures:
                completed.append(fut.result())

    for result in completed:
        rank = int(result['rank'])
        spec = str(result['spec'])
        effective_protocol = str(result['effective_protocol'])
        requested_protocol = str(result['requested_protocol'])
        zero_df = pd.read_csv(result['zero_dst'])
        cost_df = pd.read_csv(result['cost_dst'])
        res_df = pd.read_csv(result['res_dst'])
        for df in (zero_df, cost_df, res_df):
            if 'oos_protocol' not in df.columns:
                df.insert(0, 'oos_protocol', effective_protocol)
            else:
                df['oos_protocol'] = effective_protocol
            if 'oos_protocol_requested' not in df.columns:
                df.insert(1, 'oos_protocol_requested', requested_protocol)
            else:
                df['oos_protocol_requested'] = requested_protocol
            df.insert(0, 'spec', spec)
            df.insert(0, 'rank', rank)

        zero_rows.append(zero_df)
        cost_rows.append(cost_df)
        result_rows.append(res_df)
        progress_rows.append({
            'rank': rank,
            'spec': spec,
            'oos_protocol': effective_protocol,
            'oos_protocol_requested': requested_protocol,
            'status': 'done',
            'output_dir': str(result['output_dir']),
            'train_window_mode': str(result['train_window_mode']),
            'refit_every': int(result['refit_every']),
            'rebalance_every': int(result['rebalance_every']),
            'rolling_train_months': int(result['rolling_train_months']) if result['rolling_train_months'] is not None else None,
            'mc_rollouts': int(result['mc_rollouts']),
            'mc_sub_batch': int(result['mc_sub_batch']),
            'transaction_cost_bps': float(result['transaction_cost_bps']),
        })

    zero_all = pd.concat(zero_rows, ignore_index=True)
    cost_all = pd.concat(cost_rows, ignore_index=True)
    results_all = pd.concat(result_rows, ignore_index=True)
    progress_df = pd.DataFrame(progress_rows)

    zero_path = rank_root / 'rank_sweep_cross_modes_zero_cost_summary.csv'
    cost_path = rank_root / 'rank_sweep_cross_modes_all_costs_summary.csv'
    results_path = rank_root / 'rank_sweep_results.csv'
    progress_path = rank_root / 'rank_sweep_progress.csv'
    results_jsonl = rank_root / 'rank_sweep_results.jsonl'

    zero_all.to_csv(zero_path, index=False)
    cost_all.to_csv(cost_path, index=False)
    results_all.to_csv(results_path, index=False)
    progress_df.to_csv(progress_path, index=False)
    with results_jsonl.open('w', encoding='utf-8') as f:
        for row in results_all.to_dict(orient='records'):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    actual_protocols = list(dict.fromkeys(results_all['oos_protocol'].astype(str).tolist()))
    for protocol in actual_protocols:
        zero_all[zero_all['oos_protocol'] == protocol].to_csv(rank_root / f'rank_sweep_{protocol}_cross_modes_zero_cost_summary.csv', index=False)
        cost_all[cost_all['oos_protocol'] == protocol].to_csv(rank_root / f'rank_sweep_{protocol}_cross_modes_all_costs_summary.csv', index=False)
        results_all[results_all['oos_protocol'] == protocol].to_csv(rank_root / f'rank_sweep_{protocol}_results.csv', index=False)

    report = {
        'suite_name': manifest.get('suite_name'),
        'entries': [{'rank': int(x['rank']), 'spec': str(x['spec'])} for x in entries],
        'oos_protocols_requested': protocols_requested,
        'oos_protocols': actual_protocols,
        'zero_cost_summary': str(zero_path),
        'all_costs_summary': str(cost_path),
        'results_csv': str(results_path),
        'emit_legacy_fixed_layout': emit_legacy_fixed_layout,
        'protocol_outputs': {
            protocol: {
                'zero_cost_summary': str(rank_root / f'rank_sweep_{protocol}_cross_modes_zero_cost_summary.csv'),
                'all_costs_summary': str(rank_root / f'rank_sweep_{protocol}_cross_modes_all_costs_summary.csv'),
                'results_csv': str(rank_root / f'rank_sweep_{protocol}_results.csv'),
            }
            for protocol in actual_protocols
        },
    }
    report_path = rank_root / 'stage21_rank_sweep_report.yaml'
    report_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')

    return RankSweepArtifacts(
        out_dir=suite_root,
        manifest_yaml=manifest_path,
        zero_cost_summary=zero_path,
        all_costs_summary=cost_path,
        progress_csv=progress_path,
        results_csv=results_path,
    )
