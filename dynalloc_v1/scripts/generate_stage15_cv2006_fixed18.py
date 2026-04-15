from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

UNIVERSES = {
    "ff1": ("ff1_base", "FF1"),
    "ff6": ("ff6_szbm_base", "FF6"),
    "ff25": ("ff25_szbm_base", "FF25"),
    "ff38": ("ff38ind_base", "FF38"),
    "ff49": ("ff49ind_base", "FF49"),
    "ff100": ("ff100_szbm_base", "FF100"),
}

BOND_HOOKS = [
    ("no_bond", "none", "no bond risky menu"),
    ("curve_core", "curve_core", "core curve risky menu with UST2Y/UST5Y/UST10Y"),
    ("ust10y", "ust10y", "single long-bond risky menu with UST10Y only"),
]

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


def build_payload(*, universe_slug: str, universe_profile: str, universe_label: str, variant_slug: str, bond_hook: str, note_tail: str) -> dict:
    base_name = f"{universe_slug}_stage15_rank_sweep_cv2006_{variant_slug}_fixed"
    payload = {
        "experiment": {
            "name": base_name,
            "output_dir": f"outputs/{base_name}",
            "notes": f"stage-15 {universe_label} 2006-2025 fixed matrix: {note_tail} | fixed walk-forward evaluation",
        },
        "universe": {
            "profile": universe_profile,
            "bond_hook": bond_hook,
        },
        "split": {"profile": "cv2006_final20y"},
        "selection": {"profile": "cv2006_3fold_fullgrid_core_top2"},
        **COMMON,
    }
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate stage-15 cv2006 fixed-18 configs.")
    ap.add_argument(
        "--config-dir",
        default=str(CONFIG_DIR),
        help="Destination directory for generated YAML files.",
    )
    ap.add_argument(
        "--universes",
        nargs="*",
        default=list(UNIVERSES.keys()),
        help="Universe slugs to generate. Defaults to all stage-15 universes.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for universe_slug in args.universes:
        if universe_slug not in UNIVERSES:
            raise SystemExit(f"Unknown universe slug: {universe_slug}")
        universe_profile, universe_label = UNIVERSES[universe_slug]
        for variant_slug, bond_hook, note_tail in BOND_HOOKS:
            payload = build_payload(
                universe_slug=universe_slug,
                universe_profile=universe_profile,
                universe_label=universe_label,
                variant_slug=variant_slug,
                bond_hook=bond_hook,
                note_tail=note_tail,
            )
            filename = f"{payload['experiment']['name']}.yaml"
            (config_dir / filename).write_text(yaml.safe_dump(payload, sort_keys=False))
            written.append(filename)
    print("Wrote:")
    for filename in written:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
