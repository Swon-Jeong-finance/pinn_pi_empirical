"""
Utilities: seeding, caching, linear algebra helpers.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def seed_everything(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def annualize_from_period(x_per_period: float, periods_per_year: int) -> float:
    """
    Convert a simple per-period rate to an annualized *simple* rate approximation.
    For small returns, this is close. Prefer log/continuous conversions if you need precision.
    """
    return x_per_period * periods_per_year


def period_rate_from_annual(r_annual: float, dt_years: float, mode: str = "simple") -> float:
    """
    Convert an annual rate to per-period (dt) rate.
    - mode="simple": r_dt = r_annual * dt
    - mode="cc":     r_dt = exp(r_annual * dt) - 1   (continuous compounding)
    """
    if mode == "cc":
        return math.exp(r_annual * dt_years) - 1.0
    return r_annual * dt_years


def safe_cholesky(a: np.ndarray, jitter: float = 1e-10, max_tries: int = 10) -> np.ndarray:
    """
    Cholesky with increasing diagonal jitter for near-PSD matrices.
    Returns lower-triangular L such that L L^T ~ a.
    """
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    diag = np.eye(a.shape[0], dtype=a.dtype)
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(a + (jitter * (10**i)) * diag)
        except np.linalg.LinAlgError:
            continue
    # last resort: eigenvalue floor
    w, v = np.linalg.eigh(a)
    w = np.maximum(w, jitter)
    a_psd = (v * w) @ v.T
    return np.linalg.cholesky(a_psd + jitter * diag)


def make_cache_path(cache_dir: Path, key: str, ext: str) -> Path:
    return cache_dir / f"{key}.{ext}"


def has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False
