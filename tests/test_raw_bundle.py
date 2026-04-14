from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yaml

from dynalloc_v2.raw_bundle import (
    build_ff49_curve_core_bundle,
    build_macro7_panel,
    discover_ff49_curve_core_sources,
)


def _write_kf_zip(path: Path, header: list[str], dates: pd.DatetimeIndex, values: np.ndarray) -> None:
    lines = []
    lines.append('This is a synthetic Ken French style file')
    lines.append(','.join(['Date'] + header))
    for i, dt in enumerate(dates):
        yyyymm = dt.strftime('%Y%m')
        row = [yyyymm] + [f'{100.0 * float(x):.6f}' for x in values[i]]
        lines.append(','.join(row))
    lines.append('')
    lines.append('Annual Factors: January-December')
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem.replace('.zip', '') + '.CSV', '\n'.join(lines))


def test_build_macro7_panel_from_csv_cache(tmp_path: Path):
    idx = pd.date_range('1963-07-31', periods=36, freq='ME')
    cache_dir = tmp_path / 'fred_cache'
    cache_dir.mkdir()
    series_defs = {
        'CPIAUCSL': np.linspace(100.0, 120.0, len(idx)),
        'TB3MS': np.linspace(4.0, 5.0, len(idx)),
        'GS5': np.linspace(4.5, 5.5, len(idx)),
        'GS10': np.linspace(5.0, 6.0, len(idx)),
        'BAA': np.linspace(6.0, 7.0, len(idx)),
        'AAA': np.linspace(5.0, 6.0, len(idx)),
        'INDPRO': np.linspace(90.0, 110.0, len(idx)),
        'UNRATE': np.linspace(5.0, 4.0, len(idx)),
    }
    for series_id, values in series_defs.items():
        pd.DataFrame({'date': idx, 'value': values}).to_csv(cache_dir / f'fred_{series_id}.csv', index=False)

    macro = build_macro7_panel(fred_cache_dir=cache_dir)
    assert list(macro.columns) == [
        'infl_yoy', 'term_spread', 'default_spread', 'indpro_yoy', 'unrate_chg', 'short_rate', 'gs5', 'gs10'
    ]
    assert macro.index.min() > idx.min()
    assert macro.index.max() == idx.max()


def test_discover_ff49_curve_core_sources_search_root(tmp_path: Path):
    root = tmp_path
    french_dir = root / 'some_stage' / '_cache_french'
    french_dir.mkdir(parents=True)
    for name in [
        '49_Industry_Portfolios_CSV.zip',
        'F-F_Research_Data_Factors_CSV.zip',
        'F-F_Research_Data_5_Factors_2x3_CSV.zip',
    ]:
        (french_dir / name).write_bytes(b'PK\x05\x06' + b'\x00' * 18)  # empty zip shell is enough for discovery
    data_dir = root / 'other_stage' / 'data'
    data_dir.mkdir(parents=True)
    for name in ['bond2y.csv', 'bond5y.csv', 'bond10y.csv']:
        (data_dir / name).write_text('date,value\n', encoding='utf-8')

    out = discover_ff49_curve_core_sources(search_root=root)
    assert out.ff49_zip.name == '49_Industry_Portfolios_CSV.zip'
    assert out.ff3_zip.name == 'F-F_Research_Data_Factors_CSV.zip'
    assert out.ff5_zip.name == 'F-F_Research_Data_5_Factors_2x3_CSV.zip'
    assert out.bond2y_csv.name == 'bond2y.csv'
    assert out.bond5y_csv.name == 'bond5y.csv'
    assert out.bond10y_csv.name == 'bond10y.csv'


def test_build_ff49_curve_core_bundle_with_explicit_sources(tmp_path: Path):
    dates = pd.date_range('1963-07-31', periods=30, freq='ME')
    n = len(dates)
    ff49_vals = np.full((n, 49), 0.02, dtype=float)
    ff3_vals = np.column_stack([
        np.full(n, 0.01),
        np.full(n, 0.005),
        np.full(n, 0.004),
        np.full(n, 0.001),
    ])
    ff5_vals = np.column_stack([
        np.full(n, 0.01),
        np.full(n, 0.005),
        np.full(n, 0.004),
        np.full(n, 0.003),
        np.full(n, 0.002),
        np.full(n, 0.001),
    ])
    ff49_zip = tmp_path / '49_Industry_Portfolios_CSV.zip'
    ff3_zip = tmp_path / 'F-F_Research_Data_Factors_CSV.zip'
    ff5_zip = tmp_path / 'F-F_Research_Data_5_Factors_2x3_CSV.zip'
    _write_kf_zip(ff49_zip, [f'ind_{i+1:02d}' for i in range(49)], dates, ff49_vals)
    _write_kf_zip(ff3_zip, ['Mkt-RF', 'SMB', 'HML', 'RF'], dates, ff3_vals)
    _write_kf_zip(ff5_zip, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'], dates, ff5_vals)

    for name, col, val in [
        ('bond2y.csv', 'ret2y', 0.002),
        ('bond5y.csv', 'ret5y', 0.003),
        ('bond10y.csv', 'bond10y_ret', 0.004),
    ]:
        pd.DataFrame({'date': dates, col: np.full(n, val)}).to_csv(tmp_path / name, index=False)

    macro = pd.DataFrame({
        'date': dates,
        'infl_yoy': np.linspace(0.01, 0.02, n),
        'term_spread': np.linspace(0.01, 0.02, n),
        'default_spread': np.linspace(0.02, 0.03, n),
        'indpro_yoy': np.linspace(0.01, 0.015, n),
        'unrate_chg': np.linspace(0.0, 0.005, n),
        'short_rate': np.linspace(0.03, 0.04, n),
        'gs5': np.linspace(0.04, 0.05, n),
        'gs10': np.linspace(0.05, 0.06, n),
    })
    macro_csv = tmp_path / 'macro_panel.csv'
    macro.to_csv(macro_csv, index=False)

    out_dir = tmp_path / 'bundle'
    artifacts = build_ff49_curve_core_bundle(
        out_dir=out_dir,
        ff49_zip=ff49_zip,
        ff3_zip=ff3_zip,
        ff5_zip=ff5_zip,
        bond2y_csv=tmp_path / 'bond2y.csv',
        bond5y_csv=tmp_path / 'bond5y.csv',
        bond10y_csv=tmp_path / 'bond10y.csv',
        macro_panel_csv=macro_csv,
        panel_end_date='1965-06-30',
        manifest_split_profile='cv2000_final20y',
    )

    manifest = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    returns = pd.read_csv(artifacts.returns_csv)
    bond = pd.read_csv(artifacts.bond_csv)
    ff5 = pd.read_csv(artifacts.ff5_csv)
    macro_out = pd.read_csv(artifacts.macro_csv)

    assert manifest['bundle_kind'] == 'ff49_curve_core_canonical_raw'
    assert manifest['split_profile'] == 'cv2000_final20y'
    assert manifest['data_end'] == '1965-06-30'
    assert returns['date'].max() == '1965-06-30'
    assert bond.columns.tolist() == ['date', 'UST2Y', 'UST5Y', 'UST10Y']
    # returns_panel matches legacy schema: 49 industry excess returns + 3 bond excess return columns
    assert returns.columns[-3:].tolist() == ['UST2Y', 'UST5Y', 'UST10Y']
    assert ff5.columns.tolist() == ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    ff3 = pd.read_csv(artifacts.ff3_csv)
    assert ff3.columns.tolist() == ['date', 'Mkt-RF', 'SMB', 'HML']
    assert macro_out.columns.tolist()[0] == 'date'
    # 2.0% total industry return less 0.1% RF
    first_asset = returns.columns[1]
    assert abs(float(returns.iloc[0][first_asset]) - 0.019) < 1e-10
    # 0.2% total bond return less 0.1% RF = 0.1% excess return
    assert abs(float(returns.iloc[0]['UST2Y']) - 0.001) < 1e-10
