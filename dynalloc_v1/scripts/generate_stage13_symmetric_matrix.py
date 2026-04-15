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

EVAL_VARIANTS = {
    "fixed": {
        "filename_suffix": "_fixed",
        "name_suffix": "_fixed",
        "evaluation_patch": {
            "walk_forward_mode": "fixed",
            "expanding_window": False,
        },
        "note_suffix": "fixed walk-forward evaluation",
    },
    "expanding": {
        "filename_suffix": "_expanding",
        "name_suffix": "_expanding",
        "evaluation_patch": {
            "walk_forward_mode": "expanding",
            "expanding_window": True,
        },
        "note_suffix": "expanding walk-forward evaluation",
    },
    "rolling": {
        "filename_suffix": "_rolling",
        "name_suffix": "_rolling",
        "evaluation_patch": {
            "walk_forward_mode": "rolling",
            "expanding_window": False,
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
        "walk_forward_mode": "fixed",
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

UNIVERSES = {
    "ff1": ("ff1_base", "FF1"),
    "ff6": ("ff6_szbm_base", "FF6"),
    "ff25": ("ff25_szbm_base", "FF25"),
    "ff49": ("ff49ind_base", "FF49"),
    "ff100": ("ff100_szbm_base", "FF100"),
}

BOND_HOOKS = [
    ("no_bond", "none", "no bond risky menu"),
    ("curve_core", "curve_core", "core curve risky menu with UST2Y/UST5Y/UST10Y"),
    ("ust10y", "ust10y", "single long-bond risky menu with UST10Y only"),
]


def _parse_universes(values: list[str]) -> list[str]:
    if not values:
        return ["ff1", "ff6", "ff100"]
    out: list[str] = []
    for value in values:
        token = str(value).strip().lower()
        if token == "all":
            out.extend(["ff1", "ff6", "ff25", "ff49", "ff100"])
            continue
        if token not in UNIVERSES:
            raise ValueError(f"Unknown universe slug: {value}")
        out.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in out:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def build_payload(*, matrix_key: str, universe_slug: str, variant_slug: str, bond_hook: str, note_tail: str, eval_variant: str) -> dict:
    matrix = MATRICES[matrix_key]
    wf = EVAL_VARIANTS[eval_variant]
    universe_profile, label = UNIVERSES[universe_slug]
    base_name = f"{universe_slug}_stage13_rank_sweep_{matrix['suffix']}_{variant_slug}{wf['name_suffix']}"
    payload = {
        "experiment": {
            "name": base_name,
            "output_dir": f"outputs/{base_name}",
            "notes": f"stage-13 {label} {matrix['note_prefix']} symmetric universe matrix: {note_tail} | {wf['note_suffix']}",
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
    ap = argparse.ArgumentParser(description="Generate stage-13 symmetric config matrix.")
    ap.add_argument(
        "--universes",
        nargs="*",
        default=["ff1", "ff6", "ff100"],
        help=(
            "Universe slugs to generate (default: ff1 ff6 ff100). "
            "Use --universes all for the full symmetric matrix including ff25 and ff49."
        ),
    )
    ap.add_argument(
        "--matrix",
        choices=("cv1985", "cv2000", "both"),
        default="both",
        help="Which split matrix to generate.",
    )
    ap.add_argument(
        "--config-dir",
        default=str(CONFIG_DIR),
        help="Destination directory for generated YAML files.",
    )
    return ap.parse_args()


def _matrix_plan(key: str) -> list[str]:
    if key == "cv1985":
        return ["cv1985"]
    if key == "cv2000":
        return ["cv2000"]
    if key == "both":
        return ["cv1985", "cv2000"]
    raise ValueError(f"Unknown matrix selection: {key}")


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    universes = _parse_universes(list(args.universes))
    written: list[str] = []
    for matrix_key in _matrix_plan(args.matrix):
        for universe_slug in universes:
            for variant_slug, bond_hook, note_tail in BOND_HOOKS:
                for eval_variant in ("fixed", "expanding", "rolling"):
                    payload = build_payload(
                        matrix_key=matrix_key,
                        universe_slug=universe_slug,
                        variant_slug=variant_slug,
                        bond_hook=bond_hook,
                        note_tail=note_tail,
                        eval_variant=eval_variant,
                    )
                    filename = f"{payload['experiment']['name']}.yaml"
                    (config_dir / filename).write_text(yaml.safe_dump(payload, sort_keys=False))
                    written.append(filename)
    print("Wrote:")
    for filename in written:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
