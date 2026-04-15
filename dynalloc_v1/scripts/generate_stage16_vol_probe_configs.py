from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

UNIVERSES = {
    "ff38": ("ff38ind_base", "FF38"),
    "ff49": ("ff49ind_base", "FF49"),
    "ff100": ("ff100_szbm_base", "FF100"),
}
SPLITS = {
    "cv2000": ("cv2000_final20y", "2000-2019"),
    "cv2006": ("cv2006_final20y", "2006-2025"),
}
MACROS = {
    "macroonly": ("bond_curve_core_cv", "current curve-credit macro only"),
    "macroplusrv": ("bond_curve_core_cv_plus_rv1m", "current curve-credit macro plus log_rv_mkt_1m"),
    "volonly": ("rv_mkt_1m_only", "vol-only macro with log_rv_mkt_1m"),
}

TEMPLATE = {
    "selection": {"profile": "disabled"},
    "model": {"profile": "base_estimated_pls3"},
    "constraints": {"profile": "long_only_cash"},
    "method": {"profile": "suite"},
    "comparison": {
        "profile": "fixed_base_model",
        "fixed_spec": "macro7_only",
        "cross_modes": ["estimated", "zero"],
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
        "backend": "native_stage6",
        "legacy_entrypoint": "legacy/vendor/pgdpo_legacy_v69/run_french49_10y_model_based_latent_varx_fred.py",
        "legacy_workdir": "legacy/vendor/pgdpo_legacy_v69",
        "seed": 0,
        "device": "cuda",
        "dtype": "float64",
        "fred_api_key_env": "FRED_API_KEY",
    },
}


def build_payload(universe_slug: str, split_slug: str, macro_slug: str) -> dict:
    universe_profile, universe_label = UNIVERSES[universe_slug]
    split_profile, split_label = SPLITS[split_slug]
    macro_profile, macro_note = MACROS[macro_slug]
    name = f"{universe_slug}_stage16_probe_{split_slug}_curve_core_{macro_slug}_fixedpair"
    payload = {
        "experiment": {
            "name": name,
            "output_dir": f"outputs/{name}",
            "notes": f"stage-16 {universe_label} {split_label} curve-core direct probe | {macro_note} | fixed macro7_only comparison without selection",
        },
        "universe": {"profile": universe_profile, "bond_hook": "curve_core"},
        "split": {"profile": split_profile},
        "macro": {"profile": macro_profile},
    }
    payload.update(TEMPLATE)
    return payload


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for universe_slug in UNIVERSES:
        for split_slug in SPLITS:
            for macro_slug in MACROS:
                payload = build_payload(universe_slug, split_slug, macro_slug)
                path = CONFIG_DIR / f"{payload['experiment']['name']}.yaml"
                path.write_text(yaml.safe_dump(payload, sort_keys=False))
                generated.append(path.name)
    out = ROOT / "stage16_generated_configs.txt"
    out.write_text("\n".join(sorted(generated)) + "\n")
    print(f"generated {len(generated)} configs -> {out}")


if __name__ == "__main__":
    main()
