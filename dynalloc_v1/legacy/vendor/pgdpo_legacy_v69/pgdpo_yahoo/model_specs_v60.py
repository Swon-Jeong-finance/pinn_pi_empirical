"""Model-spec registry helpers for v60 selection.

This v60 code path is now **selection-only** and focuses on an expanded menu of
handcrafted / supervised state specifications.  The learned `vrnn_*` experiments
were intentionally retired after failing to deliver competitive selection metrics
on the French49 empirical skeleton.

The main addition here is a broader set of supervised PLS candidates, including
macro-augmented and block-balanced variants.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import re

from .state_specs import (
    STATE_SPECS,
    STATE_SPECS_V42,
    spec_requires_bond_panel,
    spec_requires_ff3,
    spec_requires_ff5,
    spec_requires_macro3,
    spec_requires_macro7,
)


_PLS_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^pls_H\d+_k\d+$", "legacy:pls"),
    (r"^pls_macro7_H\d+_k\d+$", "legacy:pls_macro7"),
    (r"^pls_ff5_macro7_H\d+_k\d+$", "legacy:pls_ff5_macro7"),
    (r"^pls_ret_macro7_H\d+_k\d+$", "legacy:pls_ret_macro7"),
    (r"^pls_ret_ff5_macro7_H\d+_k\d+$", "legacy:pls_ret_ff5_macro7"),
    (r"^pls_bal_ret_macro7_H\d+_k\d+$", "legacy:pls_bal_ret_macro7"),
    (r"^pls_bal_ret_ff5_macro7_H\d+_k\d+$", "legacy:pls_bal_ret_ff5_macro7"),
)


def infer_spec_family(spec: str) -> str:
    s = str(spec)
    for pat, fam in _PLS_FAMILY_PATTERNS:
        if re.fullmatch(pat, s):
            return fam
    if s == "macro3_only":
        return "legacy:macro3"
    if s == "macro7_only":
        return "legacy:macro7"
    if s == "ff3_only":
        return "legacy:ff3"
    if s == "ff5_only":
        return "legacy:ff5"
    if s.startswith("pca_only"):
        return "legacy:pca_only"
    if s.startswith("macro7_pca"):
        return "legacy:macro7_pca"
    if s.startswith("ff5_macro7_pca"):
        return "legacy:ff5_macro7_pca"
    if s in {"ff_mkt", "ff_smb", "ff_hml", "ff_mkt_smb", "ff_mkt_hml", "ff_smb_hml"}:
        return "legacy:ff3_subset"
    if s in {"macro3_eqbond_block", "macro7_eqbond_block"}:
        return "block:eqbond"
    if s in STATE_SPECS:
        return "legacy:other"
    return "unknown"


def build_v60_candidate_specs(*, legacy_specs: Sequence[str] | None = None) -> list[str]:
    specs: list[str] = []
    legacy = list(STATE_SPECS_V42 if legacy_specs is None else legacy_specs)
    for s in legacy:
        if s not in STATE_SPECS:
            raise ValueError(f"Unknown spec '{s}'. Choose from STATE_SPECS.")
        specs.append(str(s))

    out: list[str] = []
    seen: set[str] = set()
    for s in specs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def required_optional_blocks(specs: Iterable[str]) -> dict[str, bool]:
    """Infer which optional data blocks are needed for the requested candidate set."""
    flags = {"macro3": False, "macro7": False, "ff3": False, "ff5": False, "bond": False}

    for spec in specs:
        s = str(spec)
        if spec_requires_macro3(s):
            flags["macro3"] = True
        if spec_requires_macro7(s):
            flags["macro7"] = True
        if spec_requires_ff3(s):
            flags["ff3"] = True
        if spec_requires_ff5(s):
            flags["ff5"] = True
        if spec_requires_bond_panel(s):
            flags["bond"] = True

    if flags["macro7"]:
        flags["macro3"] = True
    if flags["ff5"]:
        flags["ff3"] = True
    return flags
