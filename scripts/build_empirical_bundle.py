from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd


def _read(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f'{date_col=} not found in {path}')
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def main() -> int:
    p = argparse.ArgumentParser(description='Build an empirical CSV bundle for dynalloc_v2 from aligned panel files.')
    p.add_argument('--returns-csv', required=True)
    p.add_argument('--states-csv', required=True)
    p.add_argument('--factors-csv', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--date-col', default='date')
    p.add_argument('--state-cols', nargs='+', required=True)
    p.add_argument('--factor-cols', nargs='+', required=True)
    p.add_argument('--train-start', required=True)
    p.add_argument('--test-start', required=True)
    p.add_argument('--end-date', required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rets = _read(Path(args.returns_csv), args.date_col)
    states = _read(Path(args.states_csv), args.date_col)
    factors = _read(Path(args.factors_csv), args.date_col)

    common = rets.index.intersection(states.index).intersection(factors.index)
    rets = rets.loc[common]
    states = states.loc[common, args.state_cols]
    factors = factors.loc[common, args.factor_cols]

    returns_out = out_dir / 'returns_panel.csv'
    states_out = out_dir / 'states_panel.csv'
    factors_out = out_dir / 'factors_panel.csv'
    rets.reset_index().rename(columns={'index': args.date_col}).to_csv(returns_out, index=False)
    states.reset_index().rename(columns={'index': args.date_col}).to_csv(states_out, index=False)
    factors.reset_index().rename(columns={'index': args.date_col}).to_csv(factors_out, index=False)

    cfg = {
        'project': {'name': 'empirical_ppgdpo_apt', 'output_dir': 'outputs/empirical_ppgdpo_apt'},
        'experiment': {'kind': 'ppgdpo'},
        'data': {
            'mode': 'csv',
            'returns_csv': str(returns_out),
            'states_csv': str(states_out),
            'factors_csv': str(factors_out),
            'date_col': args.date_col,
        },
        'split': {
            'train_start': args.train_start,
            'test_start': args.test_start,
            'end_date': args.end_date,
            'refit_every': 12,
            'min_train_months': 72,
        },
        'state': {'columns': list(args.state_cols)},
        'factor_model': {'extractor': 'provided', 'provided_factor_columns': list(args.factor_cols)},
        'mean_model': {'kind': 'factor_apt', 'ridge_lambda': 1.0e-6},
        'covariance_model': {'kind': 'asset_dcc', 'ridge_lambda': 1.0e-6, 'variance_floor': 1.0e-6, 'correlation_shrink': 0.10, 'factor_correlation_mode': 'independent', 'use_persistence': False, 'dcc_alpha': 0.02, 'dcc_beta': 0.97, 'variance_lambda': 0.97, 'asset_covariance_shrink': 0.10},
        'policy': {'risk_aversion': 6.0, 'risky_cap': 1.0, 'pgd_steps': 80, 'step_size': 0.05, 'turnover_penalty': 0.05},
        'ppgdpo': {'device': 'cpu', 'hidden_dim': 32, 'hidden_layers': 2, 'epochs': 40, 'lr': 1.0e-3, 'utility': 'log', 'covariance_mode': 'full', 'cross_strength': 1.0, 'train_seed': 17},
        'comparison': {'cross_modes': ['estimated', 'zero'], 'transaction_cost_bps': 0.0},
    }
    (out_dir / 'config_empirical_ppgdpo_apt.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f'wrote bundle to: {out_dir}')
    print(f'config: {out_dir / "config_empirical_ppgdpo_apt.yaml"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
