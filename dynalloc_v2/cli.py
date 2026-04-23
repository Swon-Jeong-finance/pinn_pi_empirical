from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from .experiment_windows import DEFAULT_SPLIT_PROFILE, available_split_profiles
from .experiments import run_experiment
from .legacy_cli import normalize_legacy_alias_argv, register_legacy_parsers
from .replay import replay_manifest_sample
from .workspace import init_workspace
from .native_selection import native_select_factor_suite
from .rank_sweep import run_rank_sweep
from .raw_bundle import build_ff49_curve_core_bundle
from .schema import Config


def load_config(path: str | Path) -> Config:
    p = Path(path)
    payload = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    return Config.model_validate(payload)


def _add_split_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--split-profile', choices=available_split_profiles(), default=None)
    parser.add_argument('--split-train-start', default=None)
    parser.add_argument('--split-train-pool-end', default=None)
    parser.add_argument('--split-test-start', default=None)
    parser.add_argument('--split-end-date', default=None)


def cmd_init_workspace(args):
    payload = init_workspace(root=args.root)
    for key, value in payload.items():
        print(f'{key}: {value}')
    return 0


def cmd_replay_sample(args):
    artifacts = replay_manifest_sample(
        manifest_path=args.manifest,
        rank=args.rank,
        sample=args.sample,
        protocol=args.protocol,
        block_label=args.block_label,
        out_dir=args.out_dir,
        device_override=args.device,
        mc_rollouts_override=args.ppgdpo_mc_rollouts,
        mc_sub_batch_override=args.ppgdpo_mc_sub_batch,
    )
    print(f'output_dir: {artifacts.output_dir}')
    print(f'fit_start: {artifacts.fit_start}')
    print(f'fit_end: {artifacts.fit_end}')
    print(f'replay_start: {artifacts.replay_start}')
    print(f'replay_end: {artifacts.replay_end}')
    print(f'zero_cost_summary: {artifacts.zero_cost_summary}')
    print(f'all_costs_summary: {artifacts.all_costs_summary}')
    print(f'results_csv: {artifacts.results_csv}')
    return 0


def cmd_validate(args):
    cfg = load_config(args.config)
    print('config ok')
    print(cfg.model_dump_json(indent=2))
    return 0


def cmd_plan(args):
    cfg = load_config(args.config)
    print(yaml.safe_dump(cfg.model_dump(mode='json'), sort_keys=False))
    return 0


def cmd_run(args):
    cfg = load_config(args.config)
    artifacts = run_experiment(cfg)
    print(f'output_dir: {artifacts.output_dir}')
    print(f'summary_zero_cost: {artifacts.summary_zero_cost}')
    print(f'summary_with_costs: {artifacts.summary_with_costs}')
    return 0


def cmd_run_rank_sweep(args):
    artifacts = run_rank_sweep(
        args.manifest,
        device_override=args.device,
        mc_rollouts_override=args.ppgdpo_mc_rollouts,
        mc_sub_batch_override=args.ppgdpo_mc_sub_batch,
        oos_protocols=args.oos_protocols,
        emit_legacy_fixed_layout=bool(args.emit_legacy_fixed_layout),
    )
    print(f'output_dir: {artifacts.out_dir}')
    print(f'progress_csv: {artifacts.progress_csv}')
    print(f'zero_cost_summary: {artifacts.zero_cost_summary}')
    print(f'all_costs_summary: {artifacts.all_costs_summary}')
    print(f'results_csv: {artifacts.results_csv}')
    return 0


def cmd_build_ff49_base_bundle(args):
    artifacts = build_ff49_curve_core_bundle(
        out_dir=args.out_dir,
        search_root=args.search_root,
        ff49_zip=args.ff49_zip,
        ff3_zip=args.ff3_zip,
        ff5_zip=args.ff5_zip,
        bond2y_csv=args.bond2y_csv,
        bond5y_csv=args.bond5y_csv,
        bond10y_csv=args.bond10y_csv,
        fred_cache_dir=args.fred_cache_dir,
        fred_api_key=args.fred_api_key,
        refresh_fred=bool(args.refresh_fred),
        macro_panel_csv=args.macro_panel_csv,
        panel_start_date=args.panel_start_date,
        panel_end_date=args.panel_end_date,
        manifest_split_profile=args.manifest_split_profile,
    )
    print(f'base_out_dir: {artifacts.out_dir}')
    print(f'returns_csv: {artifacts.returns_csv}')
    print(f'macro_csv: {artifacts.macro_csv}')
    print(f'ff3_csv: {artifacts.ff3_csv}')
    print(f'ff5_csv: {artifacts.ff5_csv}')
    print(f'bond_csv: {artifacts.bond_csv}')
    print(f'manifest_yaml: {artifacts.manifest_yaml}')
    return 0


def cmd_select_native_suite(args):
    artifacts = native_select_factor_suite(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        factor_mode=args.factor_mode,
        top_k=args.top_k,
        stage1_top_k=args.stage1_top_k,
        risk_aversion=args.risk_aversion,
        cv_folds=args.cv_folds,
        min_train_months=args.min_train_months,
        rolling_window=args.rolling_window,
        window_mode=args.window_mode,
        candidate_zoo=args.candidate_zoo,
        max_candidates=args.max_candidates,
        rerank_top_n=args.rerank_top_n,
        selection_split_mode=args.selection_split_mode,
        selection_val_months=args.selection_val_months,
        selection_device=args.selection_device,
        stage2_max_parallel=args.stage2_max_parallel,
        stage2_devices=args.stage2_devices,
        ppgdpo_lite_epochs=args.ppgdpo_lite_epochs,
        ppgdpo_lite_mc_rollouts=args.ppgdpo_lite_mc_rollouts,
        ppgdpo_lite_mc_sub_batch=args.ppgdpo_lite_mc_sub_batch,
        selection_transaction_cost_bps=args.selection_transaction_cost_bps,
        ppgdpo_lite_covariance_mode=args.ppgdpo_lite_covariance_mode,
        selection_optimizer_backend=args.selection_optimizer_backend,
        pipinn_device=args.pipinn_device,
        pipinn_dtype=args.pipinn_dtype,
        pipinn_outer_iters=args.pipinn_outer_iters,
        pipinn_eval_epochs=args.pipinn_eval_epochs,
        pipinn_n_train_int=args.pipinn_n_train_int,
        pipinn_n_train_bc=args.pipinn_n_train_bc,
        pipinn_n_val_int=args.pipinn_n_val_int,
        pipinn_n_val_bc=args.pipinn_n_val_bc,
        pipinn_p_uniform=args.pipinn_p_uniform,
        pipinn_p_emp=args.pipinn_p_emp,
        pipinn_p_tau_head=args.pipinn_p_tau_head,
        pipinn_p_tau_near0=args.pipinn_p_tau_near0,
        pipinn_tau_head_window=args.pipinn_tau_head_window,
        pipinn_lr=args.pipinn_lr,
        pipinn_grad_clip=args.pipinn_grad_clip,
        pipinn_w_bc=args.pipinn_w_bc,
        pipinn_w_bc_dx=args.pipinn_w_bc_dx,
        pipinn_scheduler_factor=args.pipinn_scheduler_factor,
        pipinn_scheduler_patience=args.pipinn_scheduler_patience,
        pipinn_min_lr=args.pipinn_min_lr,
        pipinn_width=args.pipinn_width,
        pipinn_depth=args.pipinn_depth,
        pipinn_covariance_train_mode=args.pipinn_covariance_train_mode,
        pipinn_ansatz_mode=args.pipinn_ansatz_mode,
        pipinn_policy_output_mode=args.pipinn_policy_output_mode,
        pipinn_emit_frozen_traincov_strategy=bool(args.pipinn_emit_frozen_traincov_strategy),
        pipinn_save_training_logs=bool(not args.disable_pipinn_save_training_logs),
        pipinn_show_progress=bool(args.pipinn_show_progress),
        pipinn_show_epoch_progress=bool(args.pipinn_show_epoch_progress),
        rerank_covariance_models=args.rerank_covariance_models,
        select_rolling_oos_window=bool(not args.disable_rolling_oos_window_selection),
        rolling_oos_window_grid=args.rolling_oos_window_grid,
        selection_protocols=args.selection_protocols,
        legacy_stage1_v1_root=args.legacy_stage1_v1_root,
        split_profile_override=args.split_profile,
        split_train_start_override=args.split_train_start,
        split_train_pool_end_override=args.split_train_pool_end,
        split_test_start_override=args.split_test_start,
        split_end_date_override=args.split_end_date,
    )
    print(f'suite_out_dir: {artifacts.out_dir}')
    print(f'manifest_yaml: {artifacts.manifest_yaml}')
    print(f'selection_summary_csv: {artifacts.selection_summary_csv}')
    print(f'selected_yaml: {artifacts.selected_yaml}')
    print(f'entry_count: {artifacts.entry_count}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='dynalloc-v2')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_init = sub.add_parser('init-workspace')
    p_init.add_argument('--root', default=None)
    p_init.set_defaults(func=cmd_init_workspace)

    p_replay = sub.add_parser('replay-sample')
    p_replay.add_argument('--manifest', required=True)
    p_replay.add_argument('--rank', type=int, default=1)
    p_replay.add_argument('--sample', choices=['insample_full', 'selection_train', 'selection_validation'], default='insample_full')
    p_replay.add_argument(
        '--protocol',
        default='fixed',
        help='OOS protocol label for replay metadata. Supports fixed, expanding_annual, rolling20y_annual, rolling_selected_annual, selected_protocol, and rolling{N}m_annual.',
    )
    p_replay.add_argument('--block-label', default=None)
    p_replay.add_argument('--out-dir', default=None)
    p_replay.add_argument('--device', default=None)
    p_replay.add_argument('--ppgdpo-mc-rollouts', type=int, default=None)
    p_replay.add_argument('--ppgdpo-mc-sub-batch', type=int, default=None)
    p_replay.set_defaults(func=cmd_replay_sample)

    p_validate = sub.add_parser('validate')
    p_validate.add_argument('--config', required=True)
    p_validate.set_defaults(func=cmd_validate)

    p_plan = sub.add_parser('plan')
    p_plan.add_argument('--config', required=True)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser('run')
    p_run.add_argument('--config', required=True)
    p_run.set_defaults(func=cmd_run)

    p_raw_base = sub.add_parser('build-ff49-base-bundle')
    p_raw_base.add_argument('--out-dir', required=True)
    p_raw_base.add_argument('--search-root', default=None)
    p_raw_base.add_argument('--ff49-zip', default=None)
    p_raw_base.add_argument('--ff3-zip', default=None)
    p_raw_base.add_argument('--ff5-zip', default=None)
    p_raw_base.add_argument('--bond2y-csv', default=None)
    p_raw_base.add_argument('--bond5y-csv', default=None)
    p_raw_base.add_argument('--bond10y-csv', default=None)
    p_raw_base.add_argument('--fred-cache-dir', default=None)
    p_raw_base.add_argument('--fred-api-key', default=None)
    p_raw_base.add_argument('--refresh-fred', action='store_true')
    p_raw_base.add_argument('--macro-panel-csv', default=None)
    p_raw_base.add_argument('--panel-start-date', default=None)
    p_raw_base.add_argument('--panel-end-date', default=None)
    p_raw_base.add_argument('--manifest-split-profile', choices=available_split_profiles(), default=DEFAULT_SPLIT_PROFILE)
    p_raw_base.set_defaults(func=cmd_build_ff49_base_bundle)

    p_native = sub.add_parser('select-native-suite')
    p_native.add_argument('--base-dir', required=True)
    p_native.add_argument('--out-dir', required=True)
    p_native.add_argument('--factor-mode', choices=['ff5_curve_core', 'ff3_curve_core', 'ff5_only'], default='ff5_curve_core')
    p_native.add_argument('--top-k', type=int, default=5, help='Final number of selected models written to the manifest after global stage2 reranking.')
    p_native.add_argument('--stage1-top-k', type=int, default=None, help='How many spec-protocol pairs survive stage1 into stage2. Default: auto=max(top-k, rerank-top-n, 8).')
    p_native.add_argument('--risk-aversion', type=float, default=5.0)
    p_native.add_argument('--cv-folds', type=int, default=3)
    p_native.add_argument('--min-train-months', type=int, default=192)
    p_native.add_argument('--rolling-window', type=int, default=60)
    p_native.add_argument('--window-mode', choices=['rolling', 'expanding'], default='rolling')
    p_native.add_argument('--candidate-zoo', choices=['pls_only', 'factor_zoo_v1', 'factor_zoo_v2'], default='factor_zoo_v2')
    p_native.add_argument('--max-candidates', type=int, default=None)
    p_native.add_argument('--rerank-top-n', type=int, default=8)
    p_native.add_argument('--selection-split-mode', choices=['trailing_holdout', 'expanding_cv'], default='trailing_holdout')
    p_native.add_argument('--selection-val-months', type=int, default=240)
    p_native.add_argument('--selection-device', default='cpu')
    p_native.add_argument('--stage2-max-parallel', type=int, default=1, help='Number of stage2 unit_id evaluations to run concurrently.')
    p_native.add_argument('--stage2-devices', default=None, help='Comma-separated device list for stage2 unit_id scheduling. Example: cuda:0,cuda:1')
    p_native.add_argument('--ppgdpo-lite-epochs', type=int, default=40)
    p_native.add_argument('--ppgdpo-lite-mc-rollouts', type=int, default=256)
    p_native.add_argument('--ppgdpo-lite-mc-sub-batch', type=int, default=256)
    p_native.add_argument('--selection-transaction-cost-bps', type=float, default=0.0)
    p_native.add_argument('--ppgdpo-lite-covariance-mode', choices=['full', 'diag'], default='full')
    
    p_native.add_argument('--selection-optimizer-backend', choices=['ppgdpo', 'pipinn'], default='pipinn')
    p_native.add_argument('--pipinn-device', default='auto')
    p_native.add_argument('--pipinn-dtype', choices=['float32', 'float64'], default='float64')
    p_native.add_argument('--pipinn-outer-iters', type=int, default=10)
    p_native.add_argument('--pipinn-eval-epochs', type=int, default=50)
    p_native.add_argument('--pipinn-n-train-int', type=int, default=4096)
    p_native.add_argument('--pipinn-n-train-bc', type=int, default=1024)
    p_native.add_argument('--pipinn-n-val-int', type=int, default=2048)
    p_native.add_argument('--pipinn-n-val-bc', type=int, default=512)
    p_native.add_argument('--pipinn-p-uniform', type=float, default=0.50)
    p_native.add_argument('--pipinn-p-emp', type=float, default=0.50)
    p_native.add_argument('--pipinn-p-tau-head', type=float, default=0.50)
    p_native.add_argument('--pipinn-p-tau-near0', type=float, default=0.20)
    p_native.add_argument('--pipinn-tau-head-window', type=int, default=0)
    p_native.add_argument('--pipinn-lr', type=float, default=5.0e-4)
    p_native.add_argument('--pipinn-grad-clip', type=float, default=1.0)
    p_native.add_argument('--pipinn-w-bc', type=float, default=10.0)
    p_native.add_argument('--pipinn-w-bc-dx', type=float, default=3.0)
    p_native.add_argument('--pipinn-scheduler-factor', type=float, default=0.5)
    p_native.add_argument('--pipinn-scheduler-patience', type=int, default=3)
    p_native.add_argument('--pipinn-min-lr', type=float, default=1.0e-5)
    p_native.add_argument('--pipinn-width', type=int, default=128)
    p_native.add_argument('--pipinn-depth', type=int, default=3)
    p_native.add_argument('--pipinn-covariance-train-mode', choices=['dcc_current', 'cross_resid'], default='dcc_current')
    p_native.add_argument('--pipinn-ansatz-mode',choices=['ansatz_log_transform', 'ansatz_normalization', 'ansatz_normalization_log_transform'], default='ansatz_normalization_log_transform')
    p_native.add_argument('--pipinn-policy-output-mode', choices=['projection', 'pure_qp'], default='pure_qp')
    p_native.add_argument('--pipinn-emit-frozen-traincov-strategy', action='store_true')
    p_native.add_argument('--disable-pipinn-save-training-logs', action='store_true')
    p_native.add_argument('--pipinn-show-progress', action='store_true')
    p_native.add_argument('--pipinn-show-epoch-progress', action='store_true')
    p_native.add_argument('--rerank-covariance-models', nargs='+', choices=['const', 'diag', 'dcc', 'adcc', 'regime_dcc'], default=['const', 'dcc', 'adcc', 'regime_dcc'])
    p_native.add_argument('--selection-protocols', nargs='+', default=None, help='Integrated stage1/stage2 protocol candidates. Example: rolling240m_annual. If omitted, the default is a fixed 20-year rolling protocol built from --rolling-oos-window-grid.')
    p_native.add_argument('--rolling-oos-window-grid', nargs='+', type=int, default=None, help='Integrated stage1/stage2 rolling annual window candidates in months. Default: 240. Warm-start semantics: early validation/OOS refits use the available history until the full window is reached.')
    p_native.add_argument('--disable-rolling-oos-window-selection', action='store_true')
    p_native.add_argument('--legacy-stage1-v1-root', default=None, help=argparse.SUPPRESS)
    _add_split_override_args(p_native)
    p_native.set_defaults(func=cmd_select_native_suite)

    p_rank = sub.add_parser('run-rank-sweep')
    p_rank.add_argument('--manifest', required=True)
    p_rank.add_argument('--device', default=None)
    p_rank.add_argument('--ppgdpo-mc-rollouts', type=int, default=None)
    p_rank.add_argument('--ppgdpo-mc-sub-batch', type=int, default=None)
    p_rank.add_argument(
        '--oos-protocols',
        nargs='+',
        default=None,
        help='Requested OOS protocols. Supports fixed, expanding_annual, rolling20y_annual, rolling_selected_annual, selected_protocol, and rolling{N}m_annual.',
    )
    p_rank.add_argument('--emit-legacy-fixed-layout', action='store_true')
    p_rank.set_defaults(func=cmd_run_rank_sweep)

    register_legacy_parsers(sub)
    return p


def main() -> int:
    parser = build_parser()
    argv = normalize_legacy_alias_argv(sys.argv[1:])
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
