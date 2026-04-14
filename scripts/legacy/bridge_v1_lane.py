from __future__ import annotations

import argparse
from dynalloc_v2.legacy_bridge import build_v1_lane_bundle


def main() -> int:
    p = argparse.ArgumentParser(description='Build a dynalloc_v2 empirical bundle from a v1 lane.')
    p.add_argument('--v1-root', required=True)
    p.add_argument('--config-stem', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--fred-api-key', default=None)
    p.add_argument('--selected-rank', type=int, default=1)
    p.add_argument('--spec', default=None)
    p.add_argument('--factor-mode', choices=['ff5_curve_core', 'ff3_curve_core', 'ff5_only'], default='ff5_curve_core')
    p.add_argument('--refresh-fred', action='store_true')
    p.add_argument('--risk-aversion', type=float, default=5.0)
    p.add_argument('--asset-universe', choices=['ff1', 'ff6', 'ff25', 'ff38', 'ff49', 'ff100'], default=None)
    args = p.parse_args()
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
    )
    print(artifacts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
