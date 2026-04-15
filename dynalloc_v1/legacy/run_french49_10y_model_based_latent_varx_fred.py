#!/usr/bin/env python3
"""Compatibility wrapper that forwards to the vendored legacy v69 runner.

This keeps the old sample-config path stable while stage-1 migration still runs
through the legacy engine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "pgdpo_legacy_v69"
ENTRYPOINT = VENDOR_ROOT / "run_french49_10y_model_based_latent_varx_fred.py"

if not ENTRYPOINT.exists():
    raise SystemExit(
        "Vendored legacy runner is missing. Expected: "
        f"{ENTRYPOINT}"
    )

os.chdir(VENDOR_ROOT)
os.execv(sys.executable, [sys.executable, str(ENTRYPOINT), *sys.argv[1:]])
