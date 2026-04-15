#!/usr/bin/env python3
"""Convenience wrapper for the v69 cross-decomposition / fast-path-constraints study.

Runs several method configurations in one go:
1) estimated world + optional local zero-hedge P-PGDPO ablation
2) zero-cross full dynamic world
3) optional zero-cross myopic proxy

All non-wrapper CLI arguments are forwarded to run_v69_methods.py.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable


WRAPPER_FLAGS_WITH_VALUE = {
    "--suite_out_dir",
}
WRAPPER_FLAGS_BOOL = {
    "--include_zero_proxy",
    "--no_include_zero_proxy",
    "--local_zero_hedge",
    "--no_local_zero_hedge",
}
CONFLICT_FLAGS_WITH_VALUE = {
    "--out_dir",
    "--cross_mode",
    "--zero_cross_policy_proxy",
}
CONFLICT_FLAGS_BOOL = {
    "--ppgdpo_local_zero_hedge",
}


def _strip_conflicts(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in WRAPPER_FLAGS_BOOL or tok in CONFLICT_FLAGS_BOOL:
            i += 1
            continue
        if tok in WRAPPER_FLAGS_WITH_VALUE or tok in CONFLICT_FLAGS_WITH_VALUE:
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def _run(cmd: list[str]) -> None:
    print("\n[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)




def _detect_walk_forward_mode(argv: list[str]) -> str:
    mode = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--walk_forward_mode" and i + 1 < len(argv):
            mode = str(argv[i + 1]).strip().lower()
            i += 2
            continue
        if tok == "--expanding_window":
            mode = "expanding"
        i += 1
    if mode not in {"fixed", "expanding", "rolling"}:
        mode = "fixed"
    return mode

def _collect_summary_rows(summary_csv: Path, suite_variant: str) -> list[dict[str, str]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["suite_variant"] = suite_variant
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_out_dir", type=str, default="out_v69_cross_suite")
    parser.add_argument("--include_zero_proxy", dest="include_zero_proxy", action="store_true", default=False)
    parser.add_argument("--no_include_zero_proxy", dest="include_zero_proxy", action="store_false")
    parser.add_argument("--local_zero_hedge", dest="local_zero_hedge", action="store_true", default=True)
    parser.add_argument("--no_local_zero_hedge", dest="local_zero_hedge", action="store_false")
    args, rest = parser.parse_known_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parent
    runner = root / "run_v69_methods.py"
    suite_out = Path(args.suite_out_dir).resolve()
    suite_out.mkdir(parents=True, exist_ok=True)

    base_args = _strip_conflicts(rest)

    jobs: list[tuple[str, list[str]]] = []
    est_args = list(base_args) + [
        "--out_dir", str(suite_out / "estimated_full"),
        "--cross_mode", "estimated",
    ]
    if args.local_zero_hedge:
        est_args.append("--ppgdpo_local_zero_hedge")
    jobs.append(("estimated_full", est_args))

    zero_full_args = list(base_args) + [
        "--out_dir", str(suite_out / "zero_full"),
        "--cross_mode", "zero",
        "--zero_cross_policy_proxy", "full",
    ]
    jobs.append(("zero_full", zero_full_args))

    if args.include_zero_proxy:
        zero_proxy_args = list(base_args) + [
            "--out_dir", str(suite_out / "zero_proxy"),
            "--cross_mode", "zero",
            "--zero_cross_policy_proxy", "myopic",
        ]
        jobs.append(("zero_proxy", zero_proxy_args))

    for _name, job_args in jobs:
        _run([sys.executable, str(runner), *job_args])

    mode = _detect_walk_forward_mode(base_args)
    all_rows: list[dict[str, str]] = []
    for name, _ in jobs:
        summary_csv = suite_out / name / f"summary_blocks_{mode}.csv"
        all_rows.extend(_collect_summary_rows(summary_csv, name))

    if all_rows:
        out_csv = suite_out / f"suite_summary_{mode}.csv"
        fieldnames = sorted({k for row in all_rows for k in row.keys()})
        with out_csv.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(all_rows)
        print(f"\nSaved: {out_csv}")
    else:
        print("\n[WARN] No summary CSVs were found; suite summary not written.")


if __name__ == "__main__":
    main()
