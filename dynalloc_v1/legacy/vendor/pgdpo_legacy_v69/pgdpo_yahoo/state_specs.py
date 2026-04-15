"""State-spec registry and builders for the French49 empirical pipeline.

This module extracts the spec-specific logic out of the large experiment runner
so that:

1. v58 core training / evaluation stays stable, and
2. spec selection can be run independently of portfolio-learning methods.

It intentionally keeps the numerical behavior of the original v58 builders.
"""
from __future__ import annotations

from typing import Optional, Tuple
import re

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from .discrete_latent_model import (
    LatentPCAConfig,
    DiscreteLatentMarketModel,
    build_discrete_latent_market_model,
    fit_latent_pca,
    fit_apt_mu_sigma,
    fit_varx_transition,
    estimate_cross_cov,
)

# -----------------------------
# State-spec registry (extracted from v58)
# -----------------------------
PLS_HORIZONS = (6, 12, 24, 36, 48)
PLS_COMPONENTS = (1, 2, 3, 4, 5)


def _make_plain_pls_grid() -> tuple[str, ...]:
    return tuple(f"pls_H{h}_k{k}" for h in PLS_HORIZONS for k in PLS_COMPONENTS)


def _make_prefixed_pls_grid(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_H{h}_k{k}" for h in PLS_HORIZONS for k in PLS_COMPONENTS)


EXTENDED_PLS_PREFIXES = (
    "pls_macro7",
    "pls_ff5_macro7",
    "pls_ret_macro7",
    "pls_ret_ff5_macro7",
    "pls_bal_ret_macro7",
    "pls_bal_ret_ff5_macro7",
)


STATE_SPECS_V42 = (
    # A) fixed
    "macro3_only",
    "macro7_only",
    "ff3_only",
    "ff5_only",
    # B) PCA k-sweep (k ∈ {1,2,3,4,5})
    "pca_only_k1",
    "pca_only_k2",
    "pca_only_k3",
    "pca_only_k4",
    "pca_only_k5",
    "macro7_pca_k1",
    "macro7_pca_k2",
    "macro7_pca_k3",
    "macro7_pca_k4",
    "macro7_pca_k5",
    "ff5_macro7_pca_k1",
    "ff5_macro7_pca_k2",
    "ff5_macro7_pca_k3",
    "ff5_macro7_pca_k4",
    "ff5_macro7_pca_k5",
    # C) return-supervised PLS grids
    *_make_plain_pls_grid(),
    *_make_prefixed_pls_grid("pls_macro7"),
    *_make_prefixed_pls_grid("pls_ff5_macro7"),
    *_make_prefixed_pls_grid("pls_ret_macro7"),
    *_make_prefixed_pls_grid("pls_ret_ff5_macro7"),
    *_make_prefixed_pls_grid("pls_bal_ret_macro7"),
    *_make_prefixed_pls_grid("pls_bal_ret_ff5_macro7"),
    # D) FF3 split
    "ff_mkt",
    "ff_smb",
    "ff_hml",
    "ff_mkt_smb",
    "ff_mkt_hml",
    "ff_smb_hml",
)

STATE_SPECS_V41 = (
    "pca_only",
    "macro7_pca",
    "ff5_macro7_pca",
    "pls_only",
)

BLOCK_STATE_SPECS = (
    "macro3_eqbond_block",
    "macro7_eqbond_block",
)

STATE_SPECS = tuple(sorted(set(STATE_SPECS_V42 + STATE_SPECS_V41 + BLOCK_STATE_SPECS)))


def is_eqbond_block_spec(spec: str) -> bool:
    return str(spec) in BLOCK_STATE_SPECS


def spec_requires_macro3(spec: str) -> bool:
    return str(spec) in {"macro3_only", "macro3_eqbond_block"}


def spec_requires_macro7(spec: str) -> bool:
    s = str(spec)
    return (
        s == "macro7_only"
        or s == "macro7_eqbond_block"
        or s.startswith("macro7_pca")
        or s.startswith("ff5_macro7_pca")
        or s.startswith("pls_macro7_H")
        or s.startswith("pls_ff5_macro7_H")
        or s.startswith("pls_ret_macro7_H")
        or s.startswith("pls_ret_ff5_macro7_H")
        or s.startswith("pls_bal_ret_macro7_H")
        or s.startswith("pls_bal_ret_ff5_macro7_H")
    )


def spec_requires_ff3(spec: str) -> bool:
    s = str(spec)
    return (s == "ff3_only") or (s.startswith("ff_") and s not in {"ff3_only", "ff5_only"})


def spec_requires_ff5(spec: str) -> bool:
    s = str(spec)
    return (
        s == "ff5_only"
        or s.startswith("ff5_macro7_pca")
        or s.startswith("pls_ff5_macro7_H")
        or s.startswith("pls_ret_ff5_macro7_H")
        or s.startswith("pls_bal_ret_ff5_macro7_H")
    )


def spec_requires_bond_panel(spec: str) -> bool:
    return is_eqbond_block_spec(spec)


def fit_latent_pls_returns_to_future_avg_returns(
    r_ex: pd.DataFrame,
    *,
    train_end: int,
    n_components: int,
    horizon: int = 12,
    scale: bool = True,
    smooth_span: int = 0,
) -> tuple[PLSRegression, pd.DataFrame]:
    """Fit predictive PLS scores on TRAIN and project all dates."""
    assert horizon >= 1
    assert train_end >= 10

    X_all = r_ex.values.astype(float)
    n_assets = int(X_all.shape[1])

    max_t = int(train_end) - int(horizon)
    if max_t < 5:
        raise ValueError(
            f"Not enough TRAIN points for PLS horizon={horizon}. "
            f"Need train_end-horizon >= 5, got train_end={train_end}."
        )

    N = max_t + 1
    X = X_all[0:N, :]

    Y = np.zeros((N, n_assets), dtype=float)
    for k in range(1, int(horizon) + 1):
        Y += X_all[k : k + N, :]
    Y /= float(horizon)

    pls = PLSRegression(n_components=int(n_components), scale=bool(scale))
    pls.fit(X, Y)

    scores = pls.transform(X_all)
    cols = [f"pls{i+1}" for i in range(int(n_components))]
    y_df = pd.DataFrame(scores, index=r_ex.index, columns=cols)

    if int(smooth_span) > 0:
        y_df = y_df.ewm(span=int(smooth_span), adjust=False).mean()

    return pls, y_df


def _standardize_df_on_train(df: pd.DataFrame, train_end: int) -> pd.DataFrame:
    mu = df.iloc[: train_end + 1].mean(axis=0)
    sd = df.iloc[: train_end + 1].std(axis=0, ddof=0)
    sd = sd.replace(0.0, 1.0)
    return (df - mu) / sd

def _build_pls_predictor_frame(
    blocks: list[tuple[str, pd.DataFrame]],
    *,
    train_end: int,
    balance_blocks: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for prefix, df in blocks:
        z = _standardize_df_on_train(df.copy(), train_end=train_end)
        if bool(balance_blocks) and int(z.shape[1]) > 0:
            z = z / float(np.sqrt(int(z.shape[1])))
        frames.append(z.add_prefix(f"{prefix}__"))
    if len(frames) == 0:
        raise RuntimeError("No predictor blocks were supplied for supervised PLS state construction.")
    return pd.concat(frames, axis=1)


def fit_latent_pls_predictors_to_future_avg_returns(
    x_df: pd.DataFrame,
    *,
    r_ex: pd.DataFrame,
    train_end: int,
    n_components: int,
    horizon: int = 12,
    smooth_span: int = 0,
    standardize_targets: bool = True,
    col_prefix: str = "pls",
) -> tuple[PLSRegression, pd.DataFrame]:
    """Fit predictive PLS scores using generic predictor blocks on TRAIN and project all dates.

    The predictor frame ``x_df`` is expected to already be aligned to ``r_ex.index`` and,
    when desired, standardized / block-balanced on the TRAIN window.
    """
    assert horizon >= 1
    assert train_end >= 10

    x_df = x_df.loc[r_ex.index].copy()
    X_all = x_df.values.astype(float)
    Y_all_src = r_ex.values.astype(float)
    n_assets = int(Y_all_src.shape[1])

    max_t = int(train_end) - int(horizon)
    if max_t < 5:
        raise ValueError(
            f"Not enough TRAIN points for PLS horizon={horizon}. "
            f"Need train_end-horizon >= 5, got train_end={train_end}."
        )

    N = max_t + 1
    X = X_all[0:N, :]

    Y = np.zeros((N, n_assets), dtype=float)
    for k in range(1, int(horizon) + 1):
        Y += Y_all_src[k : k + N, :]
    Y /= float(horizon)

    if bool(standardize_targets):
        y_mean = np.nanmean(Y, axis=0)
        y_std = np.nanstd(Y, axis=0, ddof=0)
        y_std = np.where(y_std <= 1e-12, 1.0, y_std)
        Y_fit = (Y - y_mean[None, :]) / y_std[None, :]
    else:
        Y_fit = Y

    pls = PLSRegression(n_components=int(n_components), scale=False)
    pls.fit(X, Y_fit)

    scores = pls.transform(X_all)
    cols = [f"{col_prefix}{i+1}" for i in range(int(n_components))]
    y_df = pd.DataFrame(scores, index=r_ex.index, columns=cols)

    if int(smooth_span) > 0:
        y_df = y_df.ewm(span=int(smooth_span), adjust=False).mean()

    return pls, y_df


def _build_state_model_from_y_df(
    y_df: pd.DataFrame,
    *,
    r_ex: pd.DataFrame,
    train_end: int,
    pca_cfg: LatentPCAConfig,
    z_df: Optional[pd.DataFrame] = None,
) -> DiscreteLatentMarketModel:
    apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
    varx = fit_varx_transition(y_df, z_df, train_end=train_end)
    Cross = estimate_cross_cov(apt, varx)

    pca_dummy, mean, std, _ = fit_latent_pca(r_ex, train_end=train_end, cfg=pca_cfg)

    return DiscreteLatentMarketModel(
        asset_names=list(r_ex.columns),
        state_names=list(y_df.columns),
        exog_names=[] if z_df is None else list(z_df.columns),
        a=apt.a,
        B=apt.B,
        Sigma=apt.Sigma,
        c=varx.c,
        A=varx.A,
        G=varx.G,
        Q=varx.Q,
        Cross=Cross,
        pca=pca_dummy,
        scaler_mean=mean,
        scaler_std=std,
        trans_beta=varx.beta,
        trans_feature_names=varx.feature_names,
        trans_gvarx_cfg=varx.gvarx_cfg,
        trans_y_mean=varx.y_mean,
        trans_y_std=varx.y_std,
        trans_z_mean=varx.z_mean,
        trans_z_std=varx.z_std,
    )


def _fit_pca_on_train_df(
    df: pd.DataFrame,
    *,
    train_end: int,
    n_components: int,
    standardize: bool,
    col_prefix: str,
) -> Tuple[PCA, np.ndarray, np.ndarray, pd.DataFrame]:
    if train_end < 10:
        raise ValueError("Need at least 10 TRAIN observations for predictor PCA.")

    X_train = df.iloc[: train_end + 1].values.astype(float)
    p = int(X_train.shape[1])
    n_comp = int(min(max(1, int(n_components)), p, X_train.shape[0]))

    if standardize:
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0, ddof=0)
        std = np.where(std <= 1e-12, 1.0, std)
    else:
        mean = np.zeros(p, dtype=float)
        std = np.ones(p, dtype=float)

    X_train_z = (X_train - mean[None, :]) / std[None, :]
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X_train_z)

    X_all = df.values.astype(float)
    X_all_z = (X_all - mean[None, :]) / std[None, :]
    scores = pca.transform(X_all_z)

    cols = [f"{col_prefix}{i+1}" for i in range(int(scores.shape[1]))]
    y_df = pd.DataFrame(scores, index=df.index, columns=cols)
    return pca, mean, std, y_df


def build_model_for_state_spec(
    spec: str,
    *,
    r_ex: pd.DataFrame,
    macro3: Optional[pd.DataFrame],
    macro7: Optional[pd.DataFrame],
    ff3: Optional[pd.DataFrame],
    ff5: Optional[pd.DataFrame],
    train_end: int,
    pca_cfg: LatentPCAConfig,
    pls_horizon: int = 12,
    pls_smooth_span: int = 0,
    bond_asset_names: Optional[list[str]] = None,
    block_eq_k: int = 1,
    block_bond_k: int = 1,
) -> Tuple[DiscreteLatentMarketModel, pd.DataFrame, Optional[pd.DataFrame]]:
    """Build (model, y_df, z_df_aligned) for a given state spec.

    Numerical behavior is preserved from the v58 monolithic runner.
    """
    if spec not in STATE_SPECS:
        raise ValueError(f"Unknown spec={spec}. Choose from {STATE_SPECS}.")

    def _parse_k_from_suffix(s: str) -> Optional[int]:
        m = re.search(r"_k(\d+)$", s)
        return int(m.group(1)) if m else None

    def _override_pca_cfg(cfg: LatentPCAConfig, k: Optional[int]) -> LatentPCAConfig:
        if k is None or int(k) == int(cfg.n_components):
            return cfg
        return LatentPCAConfig(n_components=int(k), standardize=bool(cfg.standardize), random_state=int(cfg.random_state))

    base_spec = spec
    tuned_k: Optional[int] = None
    tuned_pls_h: Optional[int] = None

    m_plain_pls = re.fullmatch(r"pls_H(\d+)_k(\d+)", spec)
    if m_plain_pls:
        base_spec = "pls_only"
        tuned_pls_h = int(m_plain_pls.group(1))
        tuned_k = int(m_plain_pls.group(2))

    if base_spec == spec:
        for prefix in EXTENDED_PLS_PREFIXES:
            m_prefixed_pls = re.fullmatch(rf"{prefix}_H(\d+)_k(\d+)", spec)
            if m_prefixed_pls:
                base_spec = prefix
                tuned_pls_h = int(m_prefixed_pls.group(1))
                tuned_k = int(m_prefixed_pls.group(2))
                break

    if base_spec == spec:
        if spec.startswith("pca_only_k"):
            base_spec = "pca_only"
            tuned_k = _parse_k_from_suffix(spec)
        elif spec.startswith("macro7_pca_k"):
            base_spec = "macro7_pca"
            tuned_k = _parse_k_from_suffix(spec)
        elif spec.startswith("ff5_macro7_pca_k"):
            base_spec = "ff5_macro7_pca"
            tuned_k = _parse_k_from_suffix(spec)

    ff3_map = {"mkt": "Mkt-RF", "smb": "SMB", "hml": "HML"}
    ff3_subset_cols: Optional[list[str]] = None
    if spec.startswith("ff_") and (spec not in ("ff3_only", "ff5_only")):
        toks = spec.replace("ff_", "").split("_")
        if not toks or any(t not in ff3_map for t in toks) or len(toks) > 2:
            raise ValueError(
                f"Invalid FF3-subset spec='{spec}'. Expected one of: "
                f"ff_mkt ff_smb ff_hml ff_mkt_smb ff_mkt_hml ff_smb_hml."
            )
        base_spec = "ff3_subset"
        ff3_subset_cols = [ff3_map[t] for t in toks]

    pca_cfg_local = _override_pca_cfg(pca_cfg, tuned_k)
    pls_h_local = int(tuned_pls_h) if tuned_pls_h is not None else int(pls_horizon)

    if base_spec in BLOCK_STATE_SPECS:
        macro_block_src: Optional[pd.DataFrame]
        if base_spec == "macro3_eqbond_block":
            macro_block_src = macro3
        else:
            macro_block_src = macro7
        if macro_block_src is None:
            raise RuntimeError(f"{base_spec} requires its macro block to be loaded.")

        bond_cols = [str(c) for c in (bond_asset_names or []) if str(c) in r_ex.columns]
        if len(bond_cols) == 0:
            raise RuntimeError(
                f"{base_spec} requires at least one bond asset in r_ex. "
                "Enable bonds and provide a single bond or a bond panel."
            )
        bond_col_set = set(bond_cols)
        eq_cols = [str(c) for c in r_ex.columns if str(c) not in bond_col_set]
        if len(eq_cols) == 0:
            raise RuntimeError(f"{base_spec} requires at least one non-bond asset for the equity block.")

        macro_block = _standardize_df_on_train(macro_block_src.loc[r_ex.index].copy(), train_end=train_end)
        _eq_pca, _eq_mean, _eq_std, eq_block = _fit_pca_on_train_df(
            r_ex[eq_cols].copy(),
            train_end=train_end,
            n_components=max(1, int(block_eq_k)),
            standardize=True,
            col_prefix="eq_pca",
        )
        _bond_pca, _bond_mean, _bond_std, bond_block = _fit_pca_on_train_df(
            r_ex[bond_cols].copy(),
            train_end=train_end,
            n_components=max(1, int(block_bond_k)),
            standardize=True,
            col_prefix="bond_pca",
        )
        y_df = pd.concat([macro_block, eq_block, bond_block], axis=1)
        z_df = None
        model = _build_state_model_from_y_df(
            y_df,
            r_ex=r_ex,
            train_end=train_end,
            pca_cfg=pca_cfg_local,
            z_df=z_df,
        )
        return model, y_df, z_df

    if base_spec == "pca_only":
        model, y_df, z_df = build_discrete_latent_market_model(
            r_ex=r_ex,
            z=None,
            train_end=train_end,
            pca_cfg=pca_cfg_local,
        )
        return model, y_df, z_df

    if base_spec in ("ff3_only", "ff3_subset"):
        if ff3 is None:
            raise RuntimeError(f"ff3 is required for spec={spec}")
        ff3_use = ff3.copy()
        if base_spec == "ff3_subset":
            assert ff3_subset_cols is not None
            ff3_use = ff3_use[ff3_subset_cols].copy()
        y_df = _standardize_df_on_train(ff3_use, train_end=train_end)
        z_df = None

        apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
        varx = fit_varx_transition(y_df, z_df, train_end=train_end)
        Cross = estimate_cross_cov(apt, varx)

        pca_dummy, mean, std, _ = fit_latent_pca(r_ex, train_end=train_end, cfg=pca_cfg_local)

        model = DiscreteLatentMarketModel(
            asset_names=list(r_ex.columns),
            state_names=list(y_df.columns),
            exog_names=[],
            a=apt.a,
            B=apt.B,
            Sigma=apt.Sigma,
            c=varx.c,
            A=varx.A,
            G=varx.G,
            Q=varx.Q,
            Cross=Cross,
            pca=pca_dummy,
            scaler_mean=mean,
            scaler_std=std,
            trans_beta=varx.beta,
            trans_feature_names=varx.feature_names,
            trans_gvarx_cfg=varx.gvarx_cfg,
            trans_y_mean=varx.y_mean,
            trans_y_std=varx.y_std,
            trans_z_mean=None,
            trans_z_std=None,
        )
        return model, y_df, z_df

    if base_spec == "ff5_only":
        if ff5 is None:
            raise RuntimeError("ff5 is required for spec=ff5_only")
        y_df = _standardize_df_on_train(ff5.copy(), train_end=train_end)
        z_df = None

        apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
        varx = fit_varx_transition(y_df, z_df, train_end=train_end)
        Cross = estimate_cross_cov(apt, varx)

        pca_dummy, mean, std, _ = fit_latent_pca(r_ex, train_end=train_end, cfg=pca_cfg_local)

        model = DiscreteLatentMarketModel(
            asset_names=list(r_ex.columns),
            state_names=list(y_df.columns),
            exog_names=[],
            a=apt.a,
            B=apt.B,
            Sigma=apt.Sigma,
            c=varx.c,
            A=varx.A,
            G=varx.G,
            Q=varx.Q,
            Cross=Cross,
            pca=pca_dummy,
            scaler_mean=mean,
            scaler_std=std,
            trans_beta=varx.beta,
            trans_feature_names=varx.feature_names,
            trans_gvarx_cfg=varx.gvarx_cfg,
            trans_y_mean=varx.y_mean,
            trans_y_std=varx.y_std,
            trans_z_mean=None,
            trans_z_std=None,
        )
        return model, y_df, z_df

    if base_spec in (
        "pls_only",
        "pls_macro7",
        "pls_ff5_macro7",
        "pls_ret_macro7",
        "pls_ret_ff5_macro7",
        "pls_bal_ret_macro7",
        "pls_bal_ret_ff5_macro7",
    ):
        if base_spec == "pls_only":
            _pls, y_df = fit_latent_pls_returns_to_future_avg_returns(
                r_ex,
                train_end=train_end,
                n_components=pca_cfg_local.n_components,
                horizon=int(pls_h_local),
                scale=True,
                smooth_span=int(pls_smooth_span),
            )
        else:
            blocks: list[tuple[str, pd.DataFrame]] = []
            balance_blocks = base_spec.startswith("pls_bal_")

            if base_spec == "pls_macro7":
                if macro7 is None:
                    raise RuntimeError("macro7 is required for spec=pls_macro7_H*_k*")
                blocks = [("macro7", macro7.loc[r_ex.index].copy())]
            elif base_spec == "pls_ff5_macro7":
                if ff5 is None:
                    raise RuntimeError("ff5 is required for spec=pls_ff5_macro7_H*_k*")
                if macro7 is None:
                    raise RuntimeError("macro7 is required for spec=pls_ff5_macro7_H*_k*")
                blocks = [
                    ("ff5", ff5.loc[r_ex.index].copy()),
                    ("macro7", macro7.loc[r_ex.index].copy()),
                ]
            elif base_spec in ("pls_ret_macro7", "pls_bal_ret_macro7"):
                if macro7 is None:
                    raise RuntimeError(f"macro7 is required for spec={spec}")
                blocks = [
                    ("ret", r_ex.copy()),
                    ("macro7", macro7.loc[r_ex.index].copy()),
                ]
            elif base_spec in ("pls_ret_ff5_macro7", "pls_bal_ret_ff5_macro7"):
                if ff5 is None:
                    raise RuntimeError(f"ff5 is required for spec={spec}")
                if macro7 is None:
                    raise RuntimeError(f"macro7 is required for spec={spec}")
                blocks = [
                    ("ret", r_ex.copy()),
                    ("ff5", ff5.loc[r_ex.index].copy()),
                    ("macro7", macro7.loc[r_ex.index].copy()),
                ]
            else:
                raise RuntimeError(f"Unhandled supervised PLS spec '{spec}'")

            x_df = _build_pls_predictor_frame(
                blocks,
                train_end=train_end,
                balance_blocks=bool(balance_blocks),
            )
            _pls, y_df = fit_latent_pls_predictors_to_future_avg_returns(
                x_df,
                r_ex=r_ex,
                train_end=train_end,
                n_components=pca_cfg_local.n_components,
                horizon=int(pls_h_local),
                smooth_span=int(pls_smooth_span),
                standardize_targets=True,
                col_prefix="pls",
            )

        z_df = None
        model = _build_state_model_from_y_df(
            y_df,
            r_ex=r_ex,
            train_end=train_end,
            pca_cfg=pca_cfg_local,
            z_df=z_df,
        )
        return model, y_df, z_df

    if base_spec == "macro3_only":
        if macro3 is None:
            raise RuntimeError("macro3 is required for spec=macro3_only")
        y_df = _standardize_df_on_train(macro3.copy(), train_end=train_end)
        z_df = None

        apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
        varx = fit_varx_transition(y_df, z_df, train_end=train_end)
        Cross = estimate_cross_cov(apt, varx)

        pca_dummy, mean, std, _ = fit_latent_pca(r_ex, train_end=train_end, cfg=pca_cfg_local)

        model = DiscreteLatentMarketModel(
            asset_names=list(r_ex.columns),
            state_names=list(y_df.columns),
            exog_names=[],
            a=apt.a,
            B=apt.B,
            Sigma=apt.Sigma,
            c=varx.c,
            A=varx.A,
            G=varx.G,
            Q=varx.Q,
            Cross=Cross,
            pca=pca_dummy,
            scaler_mean=mean,
            scaler_std=std,
            trans_beta=varx.beta,
            trans_feature_names=varx.feature_names,
            trans_gvarx_cfg=varx.gvarx_cfg,
            trans_y_mean=varx.y_mean,
            trans_y_std=varx.y_std,
            trans_z_mean=None,
            trans_z_std=None,
        )
        return model, y_df, z_df

    if base_spec == "macro7_only":
        if macro7 is None:
            raise RuntimeError("macro7 is required for spec=macro7_only")
        y_df = _standardize_df_on_train(macro7.copy(), train_end=train_end)
        z_df = None

        apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
        varx = fit_varx_transition(y_df, z_df, train_end=train_end)
        Cross = estimate_cross_cov(apt, varx)

        pca_dummy, mean, std, _ = fit_latent_pca(r_ex, train_end=train_end, cfg=pca_cfg_local)

        model = DiscreteLatentMarketModel(
            asset_names=list(r_ex.columns),
            state_names=list(y_df.columns),
            exog_names=[],
            a=apt.a,
            B=apt.B,
            Sigma=apt.Sigma,
            c=varx.c,
            A=varx.A,
            G=varx.G,
            Q=varx.Q,
            Cross=Cross,
            pca=pca_dummy,
            scaler_mean=mean,
            scaler_std=std,
            trans_beta=varx.beta,
            trans_feature_names=varx.feature_names,
            trans_gvarx_cfg=varx.gvarx_cfg,
            trans_y_mean=varx.y_mean,
            trans_y_std=varx.y_std,
            trans_z_mean=None,
            trans_z_std=None,
        )
        return model, y_df, z_df

    if base_spec == "macro7_pca":
        if macro7 is None:
            raise RuntimeError("macro7 is required for spec=macro7_pca")
        _pca, _mean, _std, y_df = _fit_pca_on_train_df(
            macro7.copy(),
            train_end=train_end,
            n_components=pca_cfg_local.n_components,
            standardize=True,
            col_prefix="macro7_pca",
        )
        z_df = None

        apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
        varx = fit_varx_transition(y_df, z_df, train_end=train_end)
        Cross = estimate_cross_cov(apt, varx)

        pca_dummy, mean, std, _ = fit_latent_pca(r_ex, train_end=train_end, cfg=pca_cfg_local)

        model = DiscreteLatentMarketModel(
            asset_names=list(r_ex.columns),
            state_names=list(y_df.columns),
            exog_names=[],
            a=apt.a,
            B=apt.B,
            Sigma=apt.Sigma,
            c=varx.c,
            A=varx.A,
            G=varx.G,
            Q=varx.Q,
            Cross=Cross,
            pca=pca_dummy,
            scaler_mean=mean,
            scaler_std=std,
            trans_beta=varx.beta,
            trans_feature_names=varx.feature_names,
            trans_gvarx_cfg=varx.gvarx_cfg,
            trans_y_mean=varx.y_mean,
            trans_y_std=varx.y_std,
            trans_z_mean=None,
            trans_z_std=None,
        )
        return model, y_df, z_df

    if base_spec == "ff5_macro7_pca":
        if macro7 is None:
            raise RuntimeError("macro7 is required for spec=ff5_macro7_pca")
        if ff5 is None:
            raise RuntimeError("ff5 is required for spec=ff5_macro7_pca")
        mix_df = pd.concat([ff5.copy(), macro7.copy()], axis=1)
        mix_df = mix_df.loc[r_ex.index].copy()
        _pca, _mean, _std, y_df = _fit_pca_on_train_df(
            mix_df,
            train_end=train_end,
            n_components=pca_cfg_local.n_components,
            standardize=False,
            col_prefix="mix_pca",
        )
        z_df = None

        apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
        varx = fit_varx_transition(y_df, z_df, train_end=train_end)
        Cross = estimate_cross_cov(apt, varx)

        pca_dummy, mean, std, _ = fit_latent_pca(r_ex, train_end=train_end, cfg=pca_cfg_local)

        model = DiscreteLatentMarketModel(
            asset_names=list(r_ex.columns),
            state_names=list(y_df.columns),
            exog_names=[],
            a=apt.a,
            B=apt.B,
            Sigma=apt.Sigma,
            c=varx.c,
            A=varx.A,
            G=varx.G,
            Q=varx.Q,
            Cross=Cross,
            pca=pca_dummy,
            scaler_mean=mean,
            scaler_std=std,
            trans_beta=varx.beta,
            trans_feature_names=varx.feature_names,
            trans_gvarx_cfg=varx.gvarx_cfg,
            trans_y_mean=varx.y_mean,
            trans_y_std=varx.y_std,
            trans_z_mean=None,
            trans_z_std=None,
        )
        return model, y_df, z_df

    raise RuntimeError(f"Unhandled spec={spec} (STATE_SPECS={STATE_SPECS})")
