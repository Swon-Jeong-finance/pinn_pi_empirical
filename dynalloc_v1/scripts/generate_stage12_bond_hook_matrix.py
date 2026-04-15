from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

MATRICES = {
    "cv1985": {
        "split_profile": "cv1985_final15y",
        "selection_profile": "cv1985_3fold_fullgrid_core_top2",
        "suffix": "cv1985",
        "note_prefix": "CV-era",
    },
    "cv2000": {
        "split_profile": "cv2000_final20y",
        "selection_profile": "cv2000_3fold_fullgrid_core_top2",
        "suffix": "cv2000",
        "note_prefix": "2000-2019 test",
    },
}

WALK_FORWARD_VARIANTS = {
    "fixed": {
        "filename_suffix": "",
        "name_suffix": "",
        "evaluation_patch": {
            "expanding_window": False,
        },
        "note_suffix": "fixed-window evaluation",
    },
    "rolling": {
        "filename_suffix": "_rolling",
        "name_suffix": "_rolling",
        "evaluation_patch": {
            "walk_forward_mode": "rolling",
            "rolling_train_months": 432,
        },
        "note_suffix": "rolling walk-forward evaluation",
    },
}

COMMON = {
    "macro": {"profile": "bond_curve_core_cv"},
    "model": {"profile": "base_estimated_pls3"},
    "constraints": {"profile": "long_only_cash"},
    "method": {"profile": "suite"},
    "comparison": {"profile": "selected_rank1_cross_pair"},
    "rank_sweep": {
        "enabled": True,
        "start_rank": 1,
        "end_rank": 2,
        "resume": True,
        "stop_on_error": False,
    },
    "training": {
        "iters": 800,
        "batch_size": 512,
        "lr": 3e-4,
        "gamma": 5.0,
        "residual_policy": True,
        "residual_ridge": 1.0e-6,
        "ppgdpo_mc": 256,
        "ppgdpo_subbatch": 64,
        "ppgdpo_seed": 123,
    },
    "evaluation": {
        "expanding_window": False,
        "retrain_every": 12,
        "refit_iters": 200,
        "ppgdpo_allow_long_horizon": True,
    },
    "runtime": {
        "backend": "native_stage7",
        "legacy_entrypoint": "legacy/vendor/pgdpo_legacy_v69/run_french49_10y_model_based_latent_varx_fred.py",
        "legacy_workdir": "legacy/vendor/pgdpo_legacy_v69",
        "seed": 0,
        "device": "cuda",
        "dtype": "float64",
        "fred_api_key_env": "FRED_API_KEY",
    },
}

UNIVERSES = [
    ("ff25", "ff25_szbm_base", "FF25"),
    ("ff49", "ff49ind_base", "FF49"),
    ("fama_market", "fama_market_base", "Fama-market"),
]

BOND_HOOKS = [
    ("no_bond", "none", "no bond risky menu"),
    ("curve_core", "curve_core", "core curve risky menu with UST2Y/UST5Y/UST10Y"),
    ("ust10y", "ust10y", "single long-bond risky menu with UST10Y only"),
]


def build_payload(
    *,
    matrix_key: str,
    universe_slug: str,
    universe_profile: str,
    label: str,
    variant_slug: str,
    bond_hook: str,
    note_tail: str,
    walk_forward_variant: str = "fixed",
) -> dict:
    matrix = MATRICES[matrix_key]
    wf = WALK_FORWARD_VARIANTS[walk_forward_variant]
    base_name = f"{universe_slug}_stage12_rank_sweep_{matrix['suffix']}_{variant_slug}"
    name = f"{base_name}{wf['name_suffix']}"
    payload = {
        "experiment": {
            "name": name,
            "output_dir": f"outputs/{name}",
            "notes": f"stage-12 {label} {matrix['note_prefix']} bond-hook matrix: {note_tail} | {wf['note_suffix']}",
        },
        "universe": {
            "profile": universe_profile,
            "bond_hook": bond_hook,
        },
        "split": {"profile": matrix["split_profile"]},
        "selection": {"profile": matrix["selection_profile"]},
        **COMMON,
    }
    payload["evaluation"] = {
        **COMMON["evaluation"],
        **wf["evaluation_patch"],
    }
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate stage-12 bond-hook matrix configs.")
    ap.add_argument(
        "--matrix",
        choices=("cv1985", "cv2000", "cv2000_rolling", "both", "all"),
        default="cv1985",
        help=(
            "Which config matrix to generate. "
            "'both' writes the original 18 fixed configs (cv1985 + cv2000); "
            "'all' also adds the 9 cv2000 rolling configs."
        ),
    )
    return ap.parse_args()


def _matrix_plan(key: str) -> list[tuple[str, str]]:
    if key == "cv1985":
        return [("cv1985", "fixed")]
    if key == "cv2000":
        return [("cv2000", "fixed")]
    if key == "cv2000_rolling":
        return [("cv2000", "rolling")]
    if key == "both":
        return [("cv1985", "fixed"), ("cv2000", "fixed")]
    if key == "all":
        return [("cv1985", "fixed"), ("cv2000", "fixed"), ("cv2000", "rolling")]
    raise ValueError(f"Unknown matrix selection: {key}")


def main() -> None:
    args = parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    plan = _matrix_plan(args.matrix)
    written: list[str] = []
    for matrix_key, walk_forward_variant in plan:
        wf = WALK_FORWARD_VARIANTS[walk_forward_variant]
        for universe_slug, universe_profile, label in UNIVERSES:
            for variant_slug, bond_hook, note_tail in BOND_HOOKS:
                payload = build_payload(
                    matrix_key=matrix_key,
                    universe_slug=universe_slug,
                    universe_profile=universe_profile,
                    label=label,
                    variant_slug=variant_slug,
                    bond_hook=bond_hook,
                    note_tail=note_tail,
                    walk_forward_variant=walk_forward_variant,
                )
                filename = f"{payload['experiment']['name']}.yaml"
                (CONFIG_DIR / filename).write_text(yaml.safe_dump(payload, sort_keys=False))
                written.append(filename)
    print("Wrote:")
    for filename in written:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
