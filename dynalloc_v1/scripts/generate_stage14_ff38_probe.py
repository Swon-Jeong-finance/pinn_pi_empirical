from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

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


def build_payload(*, variant_slug: str, bond_hook: str, note_tail: str) -> dict:
    base_name = f"ff38_stage14_rank_sweep_cv2000_{variant_slug}_fixed"
    payload = {
        "experiment": {
            "name": base_name,
            "output_dir": f"outputs/{base_name}",
            "notes": f"stage-14 FF38 near-FF49 robustness probe: {note_tail} | fixed walk-forward evaluation",
        },
        "universe": {
            "profile": "ff38ind_base",
            "bond_hook": bond_hook,
        },
        "split": {"profile": "cv2000_final20y"},
        "selection": {"profile": "cv2000_3fold_fullgrid_core_top2"},
        **COMMON,
    }
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate stage-14 ff38 probe configs.")
    ap.add_argument(
        "--config-dir",
        default=str(CONFIG_DIR),
        help="Destination directory for generated YAML files.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for variant_slug, bond_hook, note_tail in BOND_HOOKS:
        payload = build_payload(variant_slug=variant_slug, bond_hook=bond_hook, note_tail=note_tail)
        filename = f"{payload['experiment']['name']}.yaml"
        (config_dir / filename).write_text(yaml.safe_dump(payload, sort_keys=False))
        written.append(filename)
    print("Wrote:")
    for filename in written:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
