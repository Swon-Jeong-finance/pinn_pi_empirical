from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

UNIVERSES = {
    "ff6": ("ff6_szbm_base", "FF6"),
    "ff25": ("ff25_szbm_base", "FF25"),
    "ff38": ("ff38ind_base", "FF38"),
    "ff49": ("ff49ind_base", "FF49"),
    "ff100": ("ff100_szbm_base", "FF100"),
}
SPLITS = {
    "cv2000": ("cv2000_final20y", "cv2000_3fold_pls_curve_core_top2", "2000-2019"),
    "cv2006": ("cv2006_final20y", "cv2006_3fold_pls_curve_core_top2", "2006-2025"),
}
TEMPLATE = {
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
        "lr": 0.0003,
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


def build_payload(universe_slug: str, split_slug: str) -> dict:
    universe_profile, universe_label = UNIVERSES[universe_slug]
    split_profile, selection_profile, split_label = SPLITS[split_slug]
    name = f"{universe_slug}_stage17_rank_sweep_{split_slug}_curve_core_pls_fixed"
    payload = {
        "experiment": {
            "name": name,
            "output_dir": f"outputs/{name}",
            "notes": f"stage-17 {universe_label} {split_label} curve-core only PLS-family suite | ff1 excluded | fixed walk-forward evaluation",
        },
        "universe": {"profile": universe_profile, "bond_hook": "curve_core"},
        "split": {"profile": split_profile},
        "selection": {"profile": selection_profile},
    }
    payload.update(TEMPLATE)
    return payload


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for universe_slug in UNIVERSES:
        for split_slug in SPLITS:
            payload = build_payload(universe_slug, split_slug)
            path = CONFIG_DIR / f"{payload['experiment']['name']}.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            generated.append(path.name)
    out = ROOT / "stage17_generated_configs.txt"
    out.write_text("\n".join(sorted(generated)) + "\n")
    print(f"generated {len(generated)} configs -> {out}")


if __name__ == "__main__":
    main()
