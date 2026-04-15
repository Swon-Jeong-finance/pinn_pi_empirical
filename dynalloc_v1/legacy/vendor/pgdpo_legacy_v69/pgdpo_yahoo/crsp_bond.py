"""CRSP bond helpers.

This empirical codebase works with **monthly** time series indexed by
calendar month-end dates (``...-MM-30/31``).

CRSP monthly Treasury index extracts are commonly stored with month-end
**trading-day** dates (e.g., 2024-11-29). For safe and deterministic merging
with Ken French and FRED series, this module normalizes those dates to the
corresponding **calendar month-end**.

The loader is intentionally lightweight: it assumes the CSV already contains
monthly returns and only performs:
  - date parsing + month-end normalization
  - unit sanity check (decimal vs percent)
  - de-duplication / sorting

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_named_item_specs(spec_text: str | None) -> list[tuple[str, str]]:
    """Parse comma-separated NAME=VALUE entries.

    Examples
    --------
    UST2Y=data/bond2y.csv@ret_2y,UST5Y=data/bond5y.csv@ret_5y
    UST2Y=SERIES_ID_1,UST10Y=SERIES_ID_2
    """
    raw = str(spec_text or "").strip()
    if raw == "":
        return []

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        item = str(chunk).strip()
        if item == "":
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid named spec '{item}'. Expected comma-separated NAME=VALUE entries."
            )
        name, value = item.split("=", 1)
        name = str(name).strip()
        value = str(value).strip()
        if name == "" or value == "":
            raise ValueError(
                f"Invalid named spec '{item}'. Both NAME and VALUE must be non-empty."
            )
        if name in seen:
            raise ValueError(f"Duplicate named spec '{name}' in '{raw}'.")
        seen.add(name)
        out.append((name, value))
    return out


def load_crsp_bond_panel_from_spec_text(
    spec_text: str | None,
    *,
    default_date_col: str = "date",
    default_ret_col: str = "bond10y_ret",
) -> pd.DataFrame:
    """Load multiple bond return series from NAME=PATH[@RET_COL] specs."""
    frames: list[pd.Series] = []
    for name, payload in parse_named_item_specs(spec_text):
        if "@" in payload:
            csv_path, ret_col = payload.rsplit("@", 1)
            csv_path = csv_path.strip()
            ret_col = ret_col.strip() or default_ret_col
        else:
            csv_path = payload
            ret_col = default_ret_col
        s = load_crsp_bond_returns_csv(
            csv_path,
            date_col=default_date_col,
            ret_col=ret_col,
            name=name,
        )
        frames.append(s)
    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def load_crsp_bond_returns_csv(
    path: str | Path,
    *,
    date_col: str = "date",
    ret_col: str = "bond10y_ret",
    name: Optional[str] = None,
) -> pd.Series:
    """Load a monthly bond return series from a CSV.

    Parameters
    ----------
    path:
        CSV path.
    date_col:
        Column containing dates.
    ret_col:
        Column containing **monthly simple returns**.
    name:
        Optional Series name.

    Returns
    -------
    pd.Series
        Monthly simple returns (decimal), indexed by **calendar month-end**.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bond CSV not found: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"CSV missing date column '{date_col}'. Columns: {list(df.columns)}")
    if ret_col not in df.columns:
        raise ValueError(f"CSV missing return column '{ret_col}'. Columns: {list(df.columns)}")

    s = pd.Series(df[ret_col].astype(float).values, index=pd.to_datetime(df[date_col]), name=ret_col)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    # Normalize to calendar month-end.
    s.index = (s.index + pd.offsets.MonthEnd(0))
    s = s.sort_index().groupby(s.index).last()

    # Unit sanity check: if it looks like percent, convert.
    q = float(s.abs().quantile(0.999)) if len(s) > 0 else 0.0
    if q > 0.5:
        s = s / 100.0

    if name is not None:
        s.name = str(name)
    return s
