from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .experiment_windows import DEFAULT_SPLIT_PROFILE, available_split_profiles, profile_payload
from .bridge_common import BaseBundleArtifacts, _read_monthly_panel_csv
from .workspace import default_fred_cache_dir


_FF49_ZIP_CANDIDATES = (
    '49_Industry_Portfolios_CSV.zip',
    '49_Industry_Portfolios.zip',
    '49_Industry_Portfolios.txt',
)
_FF3_ZIP_CANDIDATES = (
    'F-F_Research_Data_Factors_CSV.zip',
    'F-F_Research_Data_Factors.zip',
    'F-F_Research_Data_Factors.txt',
)
_FF5_ZIP_CANDIDATES = (
    'F-F_Research_Data_5_Factors_2x3_CSV.zip',
    'F-F_Research_Data_5_Factors_2x3.zip',
    'F-F_Research_Data_5_Factors_2x3.txt',
)
_REQUIRED_FRED_SERIES = {
    'cpi': 'CPIAUCSL',
    'tb3ms': 'TB3MS',
    'gs5': 'GS5',
    'gs10': 'GS10',
    'baa': 'BAA',
    'aaa': 'AAA',
    'indpro': 'INDPRO',
    'unrate': 'UNRATE',
}
_DEFAULT_MACRO_FEATURES = [
    'infl_yoy',
    'term_spread',
    'default_spread',
    'indpro_yoy',
    'unrate_chg',
    'short_rate',
    'gs5',
    'gs10',
]
_DEFAULT_MACRO3_COLUMNS = ['infl_yoy', 'term_spread', 'default_spread']
_FRED_API_BASE = 'https://api.stlouisfed.org/fred/series/observations'


@dataclass(frozen=True)
class CanonicalRawSources:
    ff49_zip: Path
    ff3_zip: Path
    ff5_zip: Path
    bond2y_csv: Path
    bond5y_csv: Path
    bond10y_csv: Path
    fred_cache_dir: Path | None
    macro_panel_csv: Path | None


class RawBundleSourceError(RuntimeError):
    pass



def _normalize_month_end(ts: Any) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period('M').to_timestamp('M')



def _extract_single_text_file(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        cand = [n for n in names if n.lower().endswith(('.csv', '.txt'))] or names
        scored: list[tuple[int, int, str, str]] = []
        for name in cand:
            raw = zf.read(name)
            txt_local = None
            for enc in ('utf-8', 'latin-1', 'cp1252'):
                try:
                    txt_local = raw.decode(enc)
                    break
                except Exception:
                    pass
            if txt_local is None:
                txt_local = raw.decode('latin-1', errors='ignore')
            has_monthly = 1 if re.search(r'^\s*\d{6}\s*[, \t]+', txt_local, flags=re.M) else 0
            scored.append((has_monthly, len(raw), name, txt_local))
        if not scored:
            raise RawBundleSourceError('No readable text file found inside Ken French archive.')
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][3]



def _yyyymm_to_month_end(yyyymm: str) -> pd.Timestamp:
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    return pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)



def _parse_monthly_table(text: str, *, expected_n_assets: int | None = None) -> pd.DataFrame:
    lines = text.splitlines()

    def split_tokens(line: str) -> list[str]:
        return line.replace(',', ' ').strip().split()

    data_re = re.compile(r'^\s*(\d{6})\s*[, \t]+')
    annual_re = re.compile(r'^\s*(\d{4})\s*[, \t]+')
    first_idx = None
    for i, line in enumerate(lines):
        if data_re.match(line):
            first_idx = i
            break
    if first_idx is None:
        raise RawBundleSourceError('Could not locate monthly data rows (YYYYMM ...) in Ken French file.')

    header_tokens = None
    for j in range(max(0, first_idx - 40), first_idx)[::-1]:
        toks = split_tokens(lines[j])
        if not toks:
            continue
        if all(all(ch.isalpha() or ch in '-_' for ch in tok) for tok in toks) and len(toks) >= 3:
            header_tokens = toks
            break

    rows: list[str] = []
    last_yyyymm: int | None = None
    for line in lines[first_idx:]:
        if annual_re.match(line) and not data_re.match(line):
            break
        if 'END' in line.upper():
            break
        m = data_re.match(line)
        if not m:
            continue
        yyyymm_int = int(m.group(1))
        if last_yyyymm is not None and yyyymm_int <= last_yyyymm:
            break
        last_yyyymm = yyyymm_int
        rows.append(line.replace(',', ' '))
    if not rows:
        raise RawBundleSourceError('No monthly rows collected from Ken French file.')

    df = pd.read_csv(StringIO('\n'.join(rows)), sep=r'\s+', header=None)
    yyyymm = df.iloc[:, 0].astype(str).str.zfill(6)
    dates = yyyymm.apply(_yyyymm_to_month_end)
    n = df.shape[1] - 1
    if header_tokens is not None and len(header_tokens) == n + 1:
        if header_tokens[0].strip().lower() in ('date', 'yyyymm', 'month'):
            header_tokens = header_tokens[1:]
    if header_tokens is None or len(header_tokens) != n:
        if expected_n_assets is not None:
            n = min(n, expected_n_assets)
        cols = [f'asset_{i+1}' for i in range(n)]
    else:
        cols = header_tokens[:n]

    out = df.iloc[:, 1:1+n].copy()
    out.columns = cols
    out.index = pd.DatetimeIndex(dates)
    out = out.apply(pd.to_numeric, errors='coerce') / 100.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(how='all')
    return out



def _load_french_zip_panel(path: Path, *, expected_n_assets: int | None = None) -> pd.DataFrame:
    text = _extract_single_text_file(path.read_bytes())
    return _parse_monthly_table(text, expected_n_assets=expected_n_assets)



def _load_ff3_panel(path: Path) -> pd.DataFrame:
    df = _load_french_zip_panel(path)
    canon = ['Mkt-RF', 'SMB', 'HML', 'RF']
    if set(canon).issubset(df.columns):
        out = df[canon].copy()
    elif df.shape[1] >= 4:
        out = df.iloc[:, :4].copy()
        out.columns = canon
    else:
        raise RawBundleSourceError('Unexpected FF3 archive format; expected at least 4 monthly columns.')
    return out



def _load_ff5_panel(path: Path) -> pd.DataFrame:
    df = _load_french_zip_panel(path)
    canon = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    if set(canon).issubset(df.columns):
        out = df[canon].copy()
    elif df.shape[1] >= 6:
        out = df.iloc[:, :6].copy()
        out.columns = canon
    else:
        raise RawBundleSourceError('Unexpected FF5 archive format; expected at least 6 monthly columns.')
    return out



def _discover_named_file(search_root: Path, candidates: tuple[str, ...]) -> Path:
    hits: list[Path] = []
    for cand in candidates:
        hits.extend(sorted(search_root.rglob(cand)))
    if not hits:
        raise RawBundleSourceError(f'Could not find any of {candidates} under {search_root}')
    return hits[0].resolve()



def _maybe_discover_fred_cache_dir(search_root: Path) -> Path | None:
    candidates = sorted(p for p in search_root.rglob('_cache_fred') if p.is_dir())
    if not candidates:
        return None
    scored: list[tuple[int, Path]] = []
    for path in candidates:
        count = 0
        for series_id in _REQUIRED_FRED_SERIES.values():
            if list(path.glob(f'fred_{series_id}.csv')) or list(path.glob(f'fred_{series_id}_*.csv')) or list(path.glob(f'fred_{series_id}.parquet')) or list(path.glob(f'fred_{series_id}_*.parquet')):
                count += 1
        scored.append((count, path))
    scored.sort(key=lambda x: (x[0], len(str(x[1]))), reverse=True)
    best_count, best = scored[0]
    return best.resolve() if best_count > 0 else None



def discover_ff49_curve_core_sources(
    *,
    search_root: str | Path | None = None,
    ff49_zip: str | Path | None = None,
    ff3_zip: str | Path | None = None,
    ff5_zip: str | Path | None = None,
    bond2y_csv: str | Path | None = None,
    bond5y_csv: str | Path | None = None,
    bond10y_csv: str | Path | None = None,
    fred_cache_dir: str | Path | None = None,
    macro_panel_csv: str | Path | None = None,
) -> CanonicalRawSources:
    root = Path(search_root).expanduser().resolve() if search_root is not None else None

    def resolve_file(value: str | Path | None, candidates: tuple[str, ...]) -> Path:
        if value is not None:
            path = Path(value).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        if root is None:
            raise RawBundleSourceError(f'Missing required source path for {candidates[0]} and no --search-root was supplied.')
        return _discover_named_file(root, candidates)

    macro_path = None
    if macro_panel_csv is not None:
        macro_path = Path(macro_panel_csv).expanduser().resolve()
        if not macro_path.exists():
            raise FileNotFoundError(macro_path)

    fred_cache = None
    if fred_cache_dir is not None:
        fred_cache = Path(fred_cache_dir).expanduser().resolve()
        if not fred_cache.exists():
            raise FileNotFoundError(fred_cache)
    else:
        shared_cache = default_fred_cache_dir()
        if shared_cache.exists() and any(shared_cache.iterdir()):
            fred_cache = shared_cache
        elif root is not None:
            fred_cache = _maybe_discover_fred_cache_dir(root)

    return CanonicalRawSources(
        ff49_zip=resolve_file(ff49_zip, _FF49_ZIP_CANDIDATES),
        ff3_zip=resolve_file(ff3_zip, _FF3_ZIP_CANDIDATES),
        ff5_zip=resolve_file(ff5_zip, _FF5_ZIP_CANDIDATES),
        bond2y_csv=resolve_file(bond2y_csv, ('bond2y.csv',)),
        bond5y_csv=resolve_file(bond5y_csv, ('bond5y.csv',)),
        bond10y_csv=resolve_file(bond10y_csv, ('bond10y.csv',)),
        fred_cache_dir=fred_cache,
        macro_panel_csv=macro_path,
    )



def _fred_cache_candidates(cache_dir: Path, series_id: str) -> list[Path]:
    return (
        sorted(cache_dir.glob(f'fred_{series_id}.csv'))
        + sorted(cache_dir.glob(f'fred_{series_id}_*.csv'))
        + sorted(cache_dir.glob(f'fred_{series_id}.parquet'))
        + sorted(cache_dir.glob(f'fred_{series_id}_*.parquet'))
    )



def _read_fred_cache_file(path: Path) -> pd.Series:
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    elif path.suffix.lower() == '.parquet':
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise RawBundleSourceError(
                f'Could not read parquet FRED cache {path}. Install pyarrow/fastparquet or provide --fred-api-key.'
            ) from exc
    else:
        raise RawBundleSourceError(f'Unsupported FRED cache format: {path}')
    if 'date' not in df.columns or 'value' not in df.columns:
        raise RawBundleSourceError(f'FRED cache file must contain date and value columns: {path}')
    idx = pd.to_datetime(df['date'])
    s = pd.Series(pd.to_numeric(df['value'], errors='coerce').to_numpy(dtype=float), index=idx, name=path.stem)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index).to_period('M').to_timestamp('M'))
    s = s.sort_index().groupby(level=0).last()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s



def _download_fred_series_monthly(*, series_id: str, api_key: str, cache_dir: Path | None = None, refresh: bool = False) -> pd.Series:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_csv = cache_dir / f'fred_{series_id}.csv'
        if local_csv.exists() and (not refresh):
            return _read_fred_cache_file(local_csv)

    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
    }
    qs = urllib.parse.urlencode(params)
    url = f'{_FRED_API_BASE}?{qs}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; dynalloc-v2 canonical builder)'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        js = json.loads(resp.read().decode('utf-8'))
    obs = js.get('observations', [])
    if not obs:
        raise RawBundleSourceError(f'FRED returned no observations for series_id={series_id}')
    dates: list[pd.Timestamp] = []
    vals: list[float] = []
    for item in obs:
        d = item.get('date')
        v = item.get('value')
        if d is None:
            continue
        dates.append(pd.Timestamp(d) + pd.offsets.MonthEnd(0))
        if v in (None, '.'):
            vals.append(np.nan)
        else:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=series_id).astype(float)
    s = s.sort_index().groupby(level=0).last().replace([np.inf, -np.inf], np.nan).dropna()
    if cache_dir is not None:
        pd.DataFrame({'date': s.index, 'value': s.values}).to_csv(cache_dir / f'fred_{series_id}.csv', index=False)
    return s



def _load_or_download_fred_series(*, series_id: str, fred_cache_dir: Path | None, fred_api_key: str | None, refresh_fred: bool) -> pd.Series:
    if fred_cache_dir is not None:
        hits = _fred_cache_candidates(fred_cache_dir, series_id)
        for hit in hits:
            try:
                return _read_fred_cache_file(hit)
            except RawBundleSourceError:
                pass
    if fred_api_key:
        return _download_fred_series_monthly(series_id=series_id, api_key=fred_api_key, cache_dir=fred_cache_dir, refresh=refresh_fred)
    raise RawBundleSourceError(
        f'Could not load FRED series {series_id}. Supply --fred-api-key or a readable --fred-cache-dir.'
    )



def build_macro7_panel(
    *,
    fred_cache_dir: str | Path | None = None,
    fred_api_key: str | None = None,
    refresh_fred: bool = False,
    feature_ids: list[str] | None = None,
    macro3_columns: list[str] | None = None,
) -> pd.DataFrame:
    features = list(feature_ids or _DEFAULT_MACRO_FEATURES)
    macro3_cols = list(macro3_columns or _DEFAULT_MACRO3_COLUMNS)
    cache_dir = Path(fred_cache_dir).expanduser().resolve() if fred_cache_dir is not None else None
    api_key = str(fred_api_key or '').strip() or None
    series = {
        key: _load_or_download_fred_series(series_id=series_id, fred_cache_dir=cache_dir, fred_api_key=api_key, refresh_fred=refresh_fred)
        for key, series_id in _REQUIRED_FRED_SERIES.items()
    }
    common_start = max(s.index.min() for s in series.values())
    common_end = min(s.index.max() for s in series.values())
    idx = pd.date_range(start=common_start, end=common_end, freq='ME')
    ffilled: dict[str, pd.Series] = {}
    for key, s in series.items():
        s2 = s.reindex(idx).astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        ffilled[key] = s2
    cpi = ffilled['cpi']
    tb3 = ffilled['tb3ms']
    gs5 = ffilled['gs5']
    gs10 = ffilled['gs10']
    baa = ffilled['baa']
    aaa = ffilled['aaa']
    indpro = ffilled['indpro']
    unrate = ffilled['unrate']
    computed: dict[str, pd.Series] = {
        'infl_yoy': np.log(cpi).diff(12).shift(1).rename('infl_yoy'),
        'term_spread': ((gs10 - tb3) / 100.0).rename('term_spread'),
        'default_spread': ((baa - aaa) / 100.0).rename('default_spread'),
        'indpro_yoy': np.log(indpro).diff(12).shift(1).rename('indpro_yoy'),
        'unrate_chg': (unrate / 100.0).diff().shift(1).rename('unrate_chg'),
        'short_rate': (tb3 / 100.0).rename('short_rate'),
        'gs5': (gs5 / 100.0).rename('gs5'),
        'gs10': (gs10 / 100.0).rename('gs10'),
    }
    missing = [feat for feat in features if feat not in computed]
    if missing:
        raise RawBundleSourceError(f'Unknown macro features requested: {missing}')
    columns = list(features)
    for col in macro3_cols:
        if col not in columns:
            columns.append(col)
    out = pd.concat([computed[c] for c in columns], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out



def _coerce_optional_date(value: str | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == '':
        return None
    return _normalize_month_end(value)



def build_ff49_curve_core_bundle(
    *,
    out_dir: str | Path,
    search_root: str | Path | None = None,
    ff49_zip: str | Path | None = None,
    ff3_zip: str | Path | None = None,
    ff5_zip: str | Path | None = None,
    bond2y_csv: str | Path | None = None,
    bond5y_csv: str | Path | None = None,
    bond10y_csv: str | Path | None = None,
    fred_cache_dir: str | Path | None = None,
    fred_api_key: str | None = None,
    refresh_fred: bool = False,
    macro_panel_csv: str | Path | None = None,
    panel_start_date: str | None = None,
    panel_end_date: str | None = None,
    manifest_split_profile: str = DEFAULT_SPLIT_PROFILE,
) -> BaseBundleArtifacts:
    if manifest_split_profile not in available_split_profiles():
        supported = ', '.join(available_split_profiles())
        raise ValueError(f'Unsupported manifest_split_profile={manifest_split_profile!r}. Supported: {supported}')
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_ff49_curve_core_sources(
        search_root=search_root,
        ff49_zip=ff49_zip,
        ff3_zip=ff3_zip,
        ff5_zip=ff5_zip,
        bond2y_csv=bond2y_csv,
        bond5y_csv=bond5y_csv,
        bond10y_csv=bond10y_csv,
        fred_cache_dir=fred_cache_dir,
        macro_panel_csv=macro_panel_csv,
    )

    returns_total = _load_french_zip_panel(sources.ff49_zip, expected_n_assets=49)
    if returns_total.shape[1] > 49:
        returns_total = returns_total.iloc[:, :49]
    ff3_all = _load_ff3_panel(sources.ff3_zip)
    ff5_all = _load_ff5_panel(sources.ff5_zip)
    rf = ff5_all['RF'].astype(float).rename('RF')
    common_ff = ff3_all.index.intersection(ff5_all.index)
    ff3_all = ff3_all.loc[common_ff].copy()
    ff5_all = ff5_all.loc[common_ff].copy()
    rf = rf.loc[common_ff].copy()

    common_eq = returns_total.index.intersection(rf.index)
    returns_total = returns_total.loc[common_eq].copy()
    rf_eq = rf.loc[common_eq].copy()
    returns_excess = returns_total.sub(rf_eq, axis=0)

    bond2 = _read_monthly_panel_csv(sources.bond2y_csv)[['ret2y']].rename(columns={'ret2y': 'UST2Y'})
    bond5 = _read_monthly_panel_csv(sources.bond5y_csv)[['ret5y']].rename(columns={'ret5y': 'UST5Y'})
    bond10 = _read_monthly_panel_csv(sources.bond10y_csv)[['bond10y_ret']].rename(columns={'bond10y_ret': 'UST10Y'})
    bond_total = pd.concat([bond2, bond5, bond10], axis=1)
    bond_common = bond_total.index.intersection(rf.index)
    bond_excess = bond_total.loc[bond_common].sub(rf.loc[bond_common], axis=0)

    if sources.macro_panel_csv is not None:
        macro_full = _read_monthly_panel_csv(sources.macro_panel_csv)
    else:
        macro_full = build_macro7_panel(
            fred_cache_dir=sources.fred_cache_dir,
            fred_api_key=fred_api_key,
            refresh_fred=refresh_fred,
        )

    start_ts = _coerce_optional_date(panel_start_date)
    end_ts = _coerce_optional_date(panel_end_date)
    if end_ts is not None:
        returns_excess = returns_excess.loc[returns_excess.index <= end_ts]
        ff3_all = ff3_all.loc[ff3_all.index <= end_ts]
        ff5_all = ff5_all.loc[ff5_all.index <= end_ts]
        bond_excess = bond_excess.loc[bond_excess.index <= end_ts]
        macro_full = macro_full.loc[macro_full.index <= end_ts]
    if start_ts is not None:
        returns_excess = returns_excess.loc[returns_excess.index >= start_ts]
        ff3_all = ff3_all.loc[ff3_all.index >= start_ts]
        ff5_all = ff5_all.loc[ff5_all.index >= start_ts]
        bond_excess = bond_excess.loc[bond_excess.index >= start_ts]
        macro_full = macro_full.loc[macro_full.index >= start_ts]

    common = returns_excess.index.intersection(ff3_all.index).intersection(ff5_all.index).intersection(bond_excess.index).intersection(macro_full.index)
    if len(common) == 0:
        raise RawBundleSourceError('No common monthly dates across returns, macro, factors, and bond panels.')
    returns_excess = returns_excess.loc[common].sort_index()
    macro_full = macro_full.loc[common].sort_index()
    ff3_all = ff3_all.loc[common].sort_index()
    ff5_all = ff5_all.loc[common].sort_index()
    bond_excess = bond_excess.loc[common].sort_index()

    # Match the historical bundle schema used by earlier stage bundles:
    # - returns_panel contains industry excess returns PLUS the three bond excess return columns
    # - ff3_panel/ff5_panel exclude the RF column (RF is only an internal helper for excess conversion)
    returns_panel = pd.concat([returns_excess, bond_excess], axis=1).loc[common].sort_index()
    ff3_panel = ff3_all[[c for c in ff3_all.columns if c != 'RF']].copy()
    ff5_panel = ff5_all[[c for c in ff5_all.columns if c != 'RF']].copy()

    returns_out = out_dir / 'returns_panel.csv'
    macro_out = out_dir / 'macro_panel.csv'
    ff3_out = out_dir / 'ff3_panel.csv'
    ff5_out = out_dir / 'ff5_panel.csv'
    bond_out = out_dir / 'bond_panel.csv'
    returns_panel.reset_index(names='date').to_csv(returns_out, index=False)
    macro_full.reset_index(names='date').to_csv(macro_out, index=False)
    ff3_panel.reset_index(names='date').to_csv(ff3_out, index=False)
    ff5_panel.reset_index(names='date').to_csv(ff5_out, index=False)
    bond_excess.reset_index(names='date').to_csv(bond_out, index=False)

    split_payload = dict(profile_payload(manifest_split_profile))
    manifest = {
        'bundle_kind': 'ff49_curve_core_canonical_raw',
        'asset_universe': 'ff49',
        'universe_profile': 'ff49ind_curve_core',
        'bond_hook': 'curve_core',
        'macro_profile': 'bond_curve_core_cv',
        'split_profile': manifest_split_profile,
        'split_profile_source': 'bundle_default',
        'split_description': None,
        'split_overrides': {},
        'split': split_payload,
        'data_start': str(returns_excess.index.min().date()),
        'data_end': str(returns_excess.index.max().date()),
        'macro_columns': list(macro_full.columns),
        'ff3_columns': list(ff3_all.columns),
        'ff5_columns': list(ff5_all.columns),
        'bond_columns': list(bond_excess.columns),
        'source_paths': {
            'ff49_zip': str(sources.ff49_zip),
            'ff3_zip': str(sources.ff3_zip),
            'ff5_zip': str(sources.ff5_zip),
            'bond2y_csv': str(sources.bond2y_csv),
            'bond5y_csv': str(sources.bond5y_csv),
            'bond10y_csv': str(sources.bond10y_csv),
            'fred_cache_dir': str(sources.fred_cache_dir) if sources.fred_cache_dir is not None else None,
            'macro_panel_csv': str(sources.macro_panel_csv) if sources.macro_panel_csv is not None else None,
            'search_root': str(Path(search_root).expanduser().resolve()) if search_root is not None else None,
        },
        'notes': 'Panels are monthly excess returns/factors rebuilt from canonical raw caches. Use split overrides downstream for alternate OOS windows.',
    }
    manifest_yaml = out_dir / 'base_bundle_manifest.yaml'
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')
    return BaseBundleArtifacts(
        out_dir=out_dir,
        returns_csv=returns_out,
        macro_csv=macro_out,
        ff3_csv=ff3_out,
        ff5_csv=ff5_out,
        bond_csv=bond_out,
        manifest_yaml=manifest_yaml,
    )
