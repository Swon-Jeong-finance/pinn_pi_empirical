from __future__ import annotations

import argparse

from .experiment_windows import available_split_profiles
from .legacy_bridge import (
    DEFAULT_MARKET_UNIVERSES,
    build_v1_base_bundle,
    build_v1_base_grid,
    build_v1_lane_bundle,
)
from .legacy_suite import build_v1_lane_suite


LEGACY_ALIAS_COMMANDS: set[str] = {
    'bridge-v1-base-lane',
    'bridge-v1-base-grid',
    'bridge-v1-lane',
    'bridge-v1-suite',
}


def normalize_legacy_alias_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    head = str(argv[0])
    if head in LEGACY_ALIAS_COMMANDS:
        return ['legacy', head, *argv[1:]]
    return argv


def _add_split_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--split-profile', choices=available_split_profiles(), default=None)
    parser.add_argument('--split-train-start', default=None)
    parser.add_argument('--split-train-pool-end', default=None)
    parser.add_argument('--split-test-start', default=None)
    parser.add_argument('--split-end-date', default=None)


def cmd_bridge_v1_lane(args):
    artifacts = build_v1_lane_bundle(
        v1_root=args.v1_root,
        config_stem=args.config_stem,
        out_dir=args.out_dir,
        fred_api_key=args.fred_api_key,
        selected_rank=args.selected_rank,
        spec=args.spec,
        factor_mode=args.factor_mode,
        refresh_fred=bool(args.refresh_fred),
        risk_aversion=args.risk_aversion,
        asset_universe_override=args.asset_universe,
        split_profile_override=args.split_profile,
        split_train_start_override=args.split_train_start,
        split_train_pool_end_override=args.split_train_pool_end,
        split_test_start_override=args.split_test_start,
        split_end_date_override=args.split_end_date,
    )
    print(f'bridge_out_dir: {artifacts.out_dir}')
    print(f'returns_csv: {artifacts.returns_csv}')
    print(f'states_csv: {artifacts.states_csv}')
    print(f'factors_csv: {artifacts.factors_csv}')
    print(f'config_yaml: {artifacts.config_yaml}')
    print(f'metadata_yaml: {artifacts.metadata_yaml}')
    return 0


def cmd_bridge_v1_suite(args):
    artifacts = build_v1_lane_suite(
        v1_root=args.v1_root,
        config_stem=args.config_stem,
        out_dir=args.out_dir,
        fred_api_key=args.fred_api_key,
        top_k=args.top_k,
        factor_mode=args.factor_mode,
        refresh_fred=bool(args.refresh_fred),
        risk_aversion=args.risk_aversion,
        asset_universe_override=args.asset_universe,
        split_profile_override=args.split_profile,
        split_train_start_override=args.split_train_start,
        split_train_pool_end_override=args.split_train_pool_end,
        split_test_start_override=args.split_test_start,
        split_end_date_override=args.split_end_date,
    )
    print(f'suite_out_dir: {artifacts.out_dir}')
    print(f'manifest_yaml: {artifacts.manifest_yaml}')
    print(f'entry_count: {artifacts.entry_count}')
    return 0


def cmd_bridge_v1_base_lane(args):
    artifacts = build_v1_base_bundle(
        v1_root=args.v1_root,
        config_stem=args.config_stem,
        out_dir=args.out_dir,
        fred_api_key=args.fred_api_key,
        refresh_fred=bool(args.refresh_fred),
        asset_universe_override=args.asset_universe,
        split_profile_override=args.split_profile,
        split_train_start_override=args.split_train_start,
        split_train_pool_end_override=args.split_train_pool_end,
        split_test_start_override=args.split_test_start,
        split_end_date_override=args.split_end_date,
    )
    print(f'base_out_dir: {artifacts.out_dir}')
    print(f'returns_csv: {artifacts.returns_csv}')
    print(f'macro_csv: {artifacts.macro_csv}')
    print(f'ff3_csv: {artifacts.ff3_csv}')
    print(f'ff5_csv: {artifacts.ff5_csv}')
    print(f'bond_csv: {artifacts.bond_csv}')
    print(f'manifest_yaml: {artifacts.manifest_yaml}')
    return 0


def cmd_bridge_v1_base_grid(args):
    artifacts = build_v1_base_grid(
        v1_root=args.v1_root,
        config_stem=args.config_stem,
        out_dir=args.out_dir,
        fred_api_key=args.fred_api_key,
        refresh_fred=bool(args.refresh_fred),
        asset_universes=args.asset_universes,
        split_profile_override=args.split_profile,
        split_train_start_override=args.split_train_start,
        split_train_pool_end_override=args.split_train_pool_end,
        split_test_start_override=args.split_test_start,
        split_end_date_override=args.split_end_date,
    )
    print(f'grid_out_dir: {artifacts.out_dir}')
    print(f'manifest_yaml: {artifacts.manifest_yaml}')
    print(f'entry_count: {artifacts.entry_count}')
    return 0


def _configure_bridge_v1_base_lane_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--v1-root', required=True)
    parser.add_argument('--config-stem', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--fred-api-key', default=None)
    parser.add_argument('--refresh-fred', action='store_true')
    parser.add_argument('--asset-universe', choices=sorted(DEFAULT_MARKET_UNIVERSES), default=None)
    _add_split_override_args(parser)
    parser.set_defaults(func=cmd_bridge_v1_base_lane)


def _configure_bridge_v1_base_grid_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--v1-root', required=True)
    parser.add_argument('--config-stem', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--fred-api-key', default=None)
    parser.add_argument('--refresh-fred', action='store_true')
    parser.add_argument('--asset-universes', nargs='+', choices=sorted(DEFAULT_MARKET_UNIVERSES), default=list(DEFAULT_MARKET_UNIVERSES))
    _add_split_override_args(parser)
    parser.set_defaults(func=cmd_bridge_v1_base_grid)


def _configure_bridge_v1_lane_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--v1-root', required=True)
    parser.add_argument('--config-stem', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--fred-api-key', default=None)
    parser.add_argument('--selected-rank', type=int, default=1)
    parser.add_argument('--spec', default=None)
    parser.add_argument('--factor-mode', choices=['ff5_curve_core', 'ff3_curve_core', 'ff5_only'], default='ff5_curve_core')
    parser.add_argument('--refresh-fred', action='store_true')
    parser.add_argument('--risk-aversion', type=float, default=5.0)
    parser.add_argument('--asset-universe', choices=sorted(DEFAULT_MARKET_UNIVERSES), default=None)
    _add_split_override_args(parser)
    parser.set_defaults(func=cmd_bridge_v1_lane)


def _configure_bridge_v1_suite_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--v1-root', required=True)
    parser.add_argument('--config-stem', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--fred-api-key', default=None)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--factor-mode', choices=['ff5_curve_core', 'ff3_curve_core', 'ff5_only'], default='ff5_curve_core')
    parser.add_argument('--refresh-fred', action='store_true')
    parser.add_argument('--risk-aversion', type=float, default=5.0)
    parser.add_argument('--asset-universe', choices=sorted(DEFAULT_MARKET_UNIVERSES), default=None)
    _add_split_override_args(parser)
    parser.set_defaults(func=cmd_bridge_v1_suite)


def register_legacy_parsers(root_subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    legacy = root_subparsers.add_parser(
        'legacy',
        help='legacy dynalloc_v1 bridge and migration utilities',
        description='Legacy dynalloc_v1 bridge and migration utilities.',
    )
    legacy_sub = legacy.add_subparsers(dest='legacy_cmd', required=True)

    _configure_bridge_v1_base_lane_parser(legacy_sub.add_parser('bridge-v1-base-lane'))
    _configure_bridge_v1_base_grid_parser(legacy_sub.add_parser('bridge-v1-base-grid'))
    _configure_bridge_v1_lane_parser(legacy_sub.add_parser('bridge-v1-lane'))
    _configure_bridge_v1_suite_parser(legacy_sub.add_parser('bridge-v1-suite'))
