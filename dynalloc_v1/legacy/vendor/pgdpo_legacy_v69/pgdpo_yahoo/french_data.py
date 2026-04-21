"""
Ken French Data Library loader (monthly).

This module is intentionally dependency-light (stdlib + pandas/numpy).
It downloads zip archives from the Ken French FTP directory and parses the
"Monthly" tables (returns are in percent in the raw files).

We treat these return series as tradable proxies (research backtest assumption).
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import zipfile
import urllib.request


KEN_FRENCH_FTP_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


@dataclass
class FrenchDownloadConfig:
    cache_dir: Path = Path("./_cache_french")
    refresh: bool = False


def _download_bytes(url: str, refresh: bool) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PGDPO empirical loader)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def download_ken_french_zip(
    filenames: List[str],
    cfg: FrenchDownloadConfig,
) -> Tuple[bytes, str]:
    """
    Try multiple candidate filenames in the Ken French FTP directory.
    Returns: (zip_bytes, chosen_filename)
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    last_err: Optional[Exception] = None
    for fn in filenames:
        cache_path = cfg.cache_dir / fn
        if cache_path.exists() and (not cfg.refresh):
            return cache_path.read_bytes(), fn

        url = KEN_FRENCH_FTP_BASE + fn
        try:
            b = _download_bytes(url, refresh=cfg.refresh)
            cache_path.write_bytes(b)
            return b, fn
        except Exception as e:
            last_err = e
            # continue to next candidate
            continue

    raise RuntimeError(
        f"Failed to download Ken French archive. Tried: {filenames}. "
        f"Last error: {last_err}"
    )


def _extract_single_text_file(zip_bytes: bytes) -> str:
    """
    Extract the most likely data file inside a Ken French zip.

    Many archives contain both .CSV and .TXT. We:
    - prefer files whose decoded content contains YYYYMM-like rows
    - otherwise fall back to the largest .csv/.txt
    """
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = z.namelist()
        # consider only text-like entries
        cand = [n for n in names if n.lower().endswith((".csv", ".txt"))]
        if not cand:
            cand = names
        if not cand:
            raise RuntimeError("Zip archive is empty.")

        # read a small portion to score candidates
        scored = []
        for n in cand:
            try:
                raw = z.read(n)
            except Exception:
                continue
            # decode robustly
            txt_local = None
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    txt_local = raw.decode(enc)
                    break
                except Exception:
                    pass
            if txt_local is None:
                txt_local = raw.decode("latin-1", errors="ignore")

            # score: presence of YYYYMM rows
            import re
            has_monthly = 1 if re.search(r"^\s*\d{6}\s*[, \t]+", txt_local, flags=re.M) else 0
            # size score
            scored.append((has_monthly, len(raw), n, txt_local))

        if not scored:
            raise RuntimeError("No readable files inside Ken French zip.")

        # sort by (has_monthly desc, size desc)
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][3]


def _yyyymm_to_month_end(yyyymm: str) -> pd.Timestamp:
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    return (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0))


def _parse_monthly_table(
    text: str,
    expected_n_assets: Optional[int] = None,
) -> pd.DataFrame:
    """
    Parse the first Monthly table in a Ken French-style text/CSV file.

    Robust to both formats:
    - whitespace-delimited tables (common in .txt)
    - comma-delimited tables (common in *_CSV.zip)

    Heuristics:
    - Find the first line that starts with YYYYMM (6 digits), followed by comma or whitespace.
    - Collect consecutive YYYYMM rows until a line starting with YYYY (annual) or END.
    - Infer column names by scanning backward for a header line (alpha-ish tokens),
      allowing both whitespace and comma separation.
    """
    lines = text.splitlines()

    def split_tokens(ln: str) -> List[str]:
        # normalize commas to spaces and split
        return ln.replace(",", " ").strip().split()

    # Find first data line of the monthly table (YYYYMM)
    import re
    data_re = re.compile(r"^\s*(\d{6})\s*[, \t]+")
    first_idx = None
    for i, ln in enumerate(lines):
        if data_re.match(ln):
            first_idx = i
            break
    if first_idx is None:
        raise RuntimeError("Could not locate monthly data rows (YYYYMM ...) in French file.")

    # infer column names by scanning backwards
    header_tokens = None
    for j in range(max(0, first_idx - 40), first_idx)[::-1]:
        toks = split_tokens(lines[j])
        if not toks:
            continue
        # accept header with mostly alpha-ish tokens (allow hyphen/underscore)
        if all(all(ch.isalpha() or ch in "-_" for ch in t) for t in toks) and len(toks) >= 3:
            header_tokens = toks
            break

    # collect data rows
    annual_re = re.compile(r"^\s*(\d{4})\s*[, \t]+")
    rows = []
    last_yyyymm: Optional[int] = None
    started = False
    for ln in lines[first_idx:]:
        # Stop at annual table
        if annual_re.match(ln) and not data_re.match(ln):
            break
        if "END" in ln.upper():
            break
        m = data_re.match(ln)
        if m:
            yyyymm_int = int(m.group(1))
            # Some Ken French files contain multiple monthly tables back-to-back (e.g., VW and EW).
            # Detect a reset in the date index and stop after the first table.
            if last_yyyymm is not None and yyyymm_int <= last_yyyymm:
                break
            last_yyyymm = yyyymm_int
            started = True
            # normalize commas -> spaces so pandas can parse with whitespace delimiter
            rows.append(ln.replace(",", " "))
        else:
            # if we've started collecting rows and encounter a long separator, keep scanning
            continue
    if not rows:
        raise RuntimeError("No monthly rows collected from French file.")

    # Build dataframe from whitespace-delimited rows
    df = pd.read_csv(StringIO("\n".join(rows)), sep=r"\s+", header=None)

    # first col is date (yyyymm)
    yyyymm = df.iloc[:, 0].astype(str).str.zfill(6)
    dates = yyyymm.apply(_yyyymm_to_month_end)

    n = df.shape[1] - 1

    # If header includes a date column label, drop it.
    if header_tokens is not None and len(header_tokens) == n + 1:
        if header_tokens[0].strip().lower() in ("date", "yyyymm", "month"):
            header_tokens = header_tokens[1:]

    if header_tokens is None or len(header_tokens) != n:
        # fallback names
        if expected_n_assets is not None:
            n = min(n, expected_n_assets)
        cols = [f"asset_{i+1}" for i in range(n)]
    else:
        cols = header_tokens[:n]

    out = df.iloc[:, 1:1 + n].copy()
    out.columns = cols
    out.index = pd.DatetimeIndex(dates)

    # Convert to numeric, percent -> decimal
    out = out.apply(pd.to_numeric, errors="coerce") / 100.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return out




def _yyyymmdd_to_timestamp(yyyymmdd: str) -> pd.Timestamp:
    y = int(yyyymmdd[:4])
    m = int(yyyymmdd[4:6])
    d = int(yyyymmdd[6:8])
    return pd.Timestamp(year=y, month=m, day=d)


def _parse_daily_table(
    text: str,
    expected_n_assets: Optional[int] = None,
) -> pd.DataFrame:
    """
    Parse the first Daily table in a Ken French-style text/CSV file.

    Daily archives typically use YYYYMMDD as the first column. We apply the same
    heuristics as the monthly parser and stop when non-daily footer text starts.
    """
    lines = text.splitlines()

    def split_tokens(ln: str) -> List[str]:
        return ln.replace(",", " ").strip().split()

    import re
    data_re = re.compile(r"^\s*(\d{8})\s*[, 	]+")
    first_idx = None
    for i, ln in enumerate(lines):
        if data_re.match(ln):
            first_idx = i
            break
    if first_idx is None:
        raise RuntimeError("Could not locate daily data rows (YYYYMMDD ...) in French file.")

    header_tokens = None
    for j in range(max(0, first_idx - 40), first_idx)[::-1]:
        toks = split_tokens(lines[j])
        if not toks:
            continue
        if all(all(ch.isalpha() or ch in "-_" for ch in t) for t in toks) and len(toks) >= 3:
            header_tokens = toks
            break

    rows = []
    last_yyyymmdd: Optional[int] = None
    for ln in lines[first_idx:]:
        if "END" in ln.upper():
            break
        m = data_re.match(ln)
        if m:
            yyyymmdd_int = int(m.group(1))
            if last_yyyymmdd is not None and yyyymmdd_int <= last_yyyymmdd:
                break
            last_yyyymmdd = yyyymmdd_int
            rows.append(ln.replace(",", " "))
        elif rows:
            # Once the first daily table has started, stop at the first clearly non-data row.
            break
    if not rows:
        raise RuntimeError("No daily rows collected from French file.")

    df = pd.read_csv(StringIO("\n".join(rows)), sep=r"\s+", header=None)
    yyyymmdd = df.iloc[:, 0].astype(str).str.zfill(8)
    dates = yyyymmdd.apply(_yyyymmdd_to_timestamp)

    n = df.shape[1] - 1
    if header_tokens is not None and len(header_tokens) == n + 1:
        if header_tokens[0].strip().lower() in ("date", "yyyymmdd", "day"):
            header_tokens = header_tokens[1:]

    if header_tokens is None or len(header_tokens) != n:
        if expected_n_assets is not None:
            n = min(n, expected_n_assets)
        cols = [f"asset_{i+1}" for i in range(n)]
    else:
        cols = header_tokens[:n]

    out = df.iloc[:, 1:1 + n].copy()
    out.columns = cols
    out.index = pd.DatetimeIndex(dates)
    out = out.apply(pd.to_numeric, errors="coerce") / 100.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return out


def load_10_industry_portfolios_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly simple returns (decimal) for the 10 industry portfolios.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    # Candidates in case naming differs (rare but safer)
    candidates = [
        "10_Industry_Portfolios.zip",
        "10_Industry_Portfolios_CSV.zip",
        "10_Industry_Portfolios.txt",
    ]
    b, chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text, expected_n_assets=10)
    # If the file includes more than 10 columns (rare), keep first 10
    if df.shape[1] > 10:
        df = df.iloc[:, :10]
    return df


def load_6_size_bm_portfolios_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly simple returns (decimal) for the 6 portfolios formed on
    Size and Book-to-Market (2x3).
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "6_Portfolios_2x3_CSV.zip",
        "6_Portfolios_2x3.zip",
        "6_Portfolios_2x3.txt",
    ]
    b, _chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text, expected_n_assets=6)
    if df.shape[1] > 6:
        df = df.iloc[:, :6]
    return df


def load_25_size_bm_portfolios_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly simple returns (decimal) for the 25 portfolios formed on
    Size and Book-to-Market (5x5).

    Ken French FTP filenames are historically stable, but we still try a few
    plausible candidates for robustness.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "25_Portfolios_5x5_CSV.zip",
        "25_Portfolios_5x5.zip",
        "25_Portfolios_5x5.txt",
    ]
    b, _chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text, expected_n_assets=25)
    if df.shape[1] > 25:
        df = df.iloc[:, :25]
    return df

def load_17_industry_portfolios_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly simple returns (decimal) for the 17 industry portfolios.

    Note: Some archives contain multiple monthly tables (VW and EW) back-to-back.
    Our parser stops after the first monthly table by detecting date resets.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "17_Industry_Portfolios_CSV.zip",
        "17_Industry_Portfolios.zip",
        "17_Industry_Portfolios_TXT.zip",
        "17_Industry_Portfolios.txt",
    ]
    b, _chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text, expected_n_assets=17)
    if df.shape[1] > 17:
        df = df.iloc[:, :17]
    return df


def load_30_industry_portfolios_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly simple returns (decimal) for the 30 industry portfolios.

    Note: Some archives contain multiple monthly tables (VW and EW) back-to-back.
    Our parser stops after the first monthly table by detecting date resets.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "30_Industry_Portfolios_CSV.zip",
        "30_Industry_Portfolios.zip",
        "30_Industry_Portfolios_TXT.zip",
        "30_Industry_Portfolios.txt",
    ]
    b, _chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text, expected_n_assets=30)
    if df.shape[1] > 30:
        df = df.iloc[:, :30]
    return df



def load_38_industry_portfolios_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly simple returns (decimal) for the 38 industry portfolios.

    Note: Some archives contain multiple monthly tables (VW and EW) back-to-back.
    Our parser stops after the first monthly table by detecting date resets.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "38_Industry_Portfolios_CSV.zip",
        "38_Industry_Portfolios.zip",
        "38_Industry_Portfolios_TXT.zip",
        "38_Industry_Portfolios.txt",
    ]
    b, _chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text, expected_n_assets=38)
    if df.shape[1] > 38:
        df = df.iloc[:, :38]
    return df


def load_49_industry_portfolios_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly simple returns (decimal) for the 49 industry portfolios.

    Note: Some archives contain multiple monthly tables (VW and EW) back-to-back.
    Our parser stops after the first monthly table by detecting date resets.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "49_Industry_Portfolios.zip",
        "49_Industry_Portfolios_CSV.zip",
        "49_Industry_Portfolios.txt",
    ]
    b, chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text, expected_n_assets=49)
    if df.shape[1] > 49:
        df = df.iloc[:, :49]
    return df


def load_100_size_bm_portfolios_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly simple returns (decimal) for the 100 portfolios formed on
    Size and Book-to-Market (10x10).
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "100_Portfolios_10x10_CSV.zip",
        "100_Portfolios_10x10.zip",
        "100_Portfolios_10x10.txt",
    ]
    b, _chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text, expected_n_assets=100)
    if df.shape[1] > 100:
        df = df.iloc[:, :100]
    return df


def load_ff_factors_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: monthly factors in decimal.
    Columns typically include: Mkt-RF, SMB, HML, RF.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "F-F_Research_Data_Factors.zip",
        "F-F_Research_Data_Factors_CSV.zip",
        "F-F_Research_Data_Factors.txt",
    ]
    b, chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text)
    # Try to map columns to canonical names if missing
    # Often the header is: "Mkt-RF SMB HML RF"
    # If not, we assume first 4 columns in that order.
    if df.shape[1] >= 4:
        if set(["Mkt-RF", "SMB", "HML", "RF"]).issubset(set(df.columns)):
            return df[["Mkt-RF", "SMB", "HML", "RF"]].copy()
        # fallback: rename first 4
        cols = list(df.columns[:4])
        out = df.iloc[:, :4].copy()
        out.columns = ["Mkt-RF", "SMB", "HML", "RF"]
        return out
    raise RuntimeError("Unexpected factors file format: need at least 4 columns (Mkt-RF, SMB, HML, RF).")


def load_ff5_factors_monthly(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """Return Fama–French 5-factor data (2x3, monthly) in decimal.

    The Ken French 5-factor archive typically contains columns:
      - Mkt-RF, SMB, HML, RMW, CMA, RF

    Notes
    -----
    - Raw values are in percent; our parser converts to decimal.
    - The monthly series typically begins in July 1963.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "F-F_Research_Data_5_Factors_2x3.zip",
        "F-F_Research_Data_5_Factors_2x3_CSV.zip",
        "F-F_Research_Data_5_Factors_2x3.txt",
    ]
    b, _chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_monthly_table(text)

    # Preferred canonical ordering
    canon = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    if df.shape[1] >= 6:
        if set(canon).issubset(set(df.columns)):
            return df[canon].copy()
        # fallback: rename first 6
        out = df.iloc[:, :6].copy()
        out.columns = canon
        return out

    raise RuntimeError(
        "Unexpected 5-factor file format: need at least 6 columns (Mkt-RF, SMB, HML, RMW, CMA, RF)."
    )


def load_ff_factors_daily(cfg: Optional[FrenchDownloadConfig] = None) -> pd.DataFrame:
    """
    Returns: daily factors in decimal.
    Columns typically include: Mkt-RF, SMB, HML, RF.
    """
    if cfg is None:
        cfg = FrenchDownloadConfig()

    candidates = [
        "F-F_Research_Data_Factors_daily_CSV.zip",
        "F-F_Research_Data_Factors_daily.zip",
        "F-F_Research_Data_Factors_Daily_CSV.zip",
        "F-F_Research_Data_Factors_Daily.zip",
        "F-F_Research_Data_Factors_daily.txt",
    ]
    b, _chosen = download_ken_french_zip(candidates, cfg)
    text = _extract_single_text_file(b)
    df = _parse_daily_table(text)
    if df.shape[1] >= 4:
        if set(["Mkt-RF", "SMB", "HML", "RF"]).issubset(set(df.columns)):
            return df[["Mkt-RF", "SMB", "HML", "RF"]].copy()
        out = df.iloc[:, :4].copy()
        out.columns = ["Mkt-RF", "SMB", "HML", "RF"]
        return out
    raise RuntimeError("Unexpected daily factors file format: need at least 4 columns (Mkt-RF, SMB, HML, RF).")
