#!/usr/bin/env python3
"""v69 methods runner.

Defaults the equity universe to FF 25 Size–Book-to-Market, uses test-only
20-year evaluation by default, and exposes the v69 cross diagnostics +
flexible portfolio constraints with a fast path for the common long-only
(no per-asset-cap) case.
"""
from __future__ import annotations

import sys

from run_french49_10y_model_based_latent_varx_fred import main as _main


def _ensure_default_asset_universe() -> None:
    if "--asset_universe" not in sys.argv:
        sys.argv.extend(["--asset_universe", "ff25_szbm"])


def _ensure_default_test_only_methods() -> None:
    eval_flags = {"--eval_mode", "--eval_3x10y", "--eval_horizons", "--eval_30y"}
    if any(flag in sys.argv for flag in eval_flags):
        return
    sys.argv.extend(["--eval_mode", "legacy", "--eval_horizons", "20"])


if __name__ == "__main__":
    _ensure_default_asset_universe()
    _ensure_default_test_only_methods()
    _main()
