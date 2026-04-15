"""
Discrete-time latent-factor market model with exogenous drivers for factor dynamics.

Goal (paper-aligned):
- Returns (APT-style) depend ONLY on latent factor y_t:
    r^{ex}_{t+1} = a + B y_t + eps_{t+1},   eps ~ N(0, Sigma)
- Factor dynamics are flexible (VARX):
    y_{t+1} = c + A y_t + G z_t + u_{t+1},  u ~ N(0, Q)
- Innovations (eps, u) can be correlated:
    Cov(eps, u) = Cross

We treat y_t as a low-dimensional latent factor extracted from returns by PCA
(fast & stable "filter" / factor extraction). You can later replace this
with a learned filter or (neural) state-space model.

This module is used by the model-based empirical script:
  run_french49_10y_model_based_latent_varx_fred.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf


@dataclass
class LatentPCAConfig:
    n_components: int = 2
    standardize: bool = True  # standardize each asset series before PCA (train stats)
    random_state: int = 0


@dataclass
class GVarXConfig:
    """Generalized VARX configuration.

    We keep innovations Gaussian but generalize the conditional mean of
    the factor transition using a *fixed* nonlinear feature map phi:

        y_{t+1} = Phi(y_t, z_t) @ Beta + u_{t+1}

    The default choice (poly2) is intentionally lightweight for short
    financial samples.
    """

    kind: str = "poly2"          # "linear" | "poly2"
    ridge: float = 1e-3          # ridge penalty for numerical stability
    standardize_inputs: bool = True

    include_squares: bool = True
    include_yz_cross: bool = True
    include_pairwise_y: bool = False
    include_pairwise_z: bool = False

    # Stability: clip standardized inputs before building polynomial features.
    # This is important when simulating long horizons (e.g. 120 months),
    # because polynomial autoregressions can explode.
    clip_std_abs: Optional[float] = 8.0

    shrink_Q: bool = True        # LedoitWolf shrinkage for Q


@dataclass
class VarXFit:
    c: np.ndarray      # (k,)
    A: np.ndarray      # (k,k)
    G: np.ndarray      # (k,m) (m=0 allowed)
    Q: np.ndarray      # (k,k)
    resid: np.ndarray  # (T-1,k)

    # Generalized VARX (optional): y_{t+1} = Phi(y_t,z_t) @ beta + u
    beta: Optional[np.ndarray] = None                 # (p,k)
    feature_names: Optional[List[str]] = None         # (p,)
    gvarx_cfg: Optional[GVarXConfig] = None
    y_mean: Optional[np.ndarray] = None               # (k,)
    y_std: Optional[np.ndarray] = None                # (k,)
    z_mean: Optional[np.ndarray] = None               # (m,) or None
    z_std: Optional[np.ndarray] = None                # (m,) or None


@dataclass
class AptFit:
    a: np.ndarray      # (n,)
    B: np.ndarray      # (n,k)
    Sigma: np.ndarray  # (n,n)
    resid: np.ndarray  # (T-1,n)


@dataclass
class DiscreteLatentMarketModel:
    asset_names: list[str]
    state_names: list[str]
    exog_names: list[str]

    # APT / return model (monthly excess returns)
    a: np.ndarray          # (n,)
    B: np.ndarray          # (n,k)
    Sigma: np.ndarray      # (n,n)

    # Factor dynamics (monthly)
    c: np.ndarray          # (k,)
    A: np.ndarray          # (k,k)
    G: np.ndarray          # (k,m)
    Q: np.ndarray          # (k,k)

    # Innovation cross-cov (monthly): Cov(eps, u)
    Cross: np.ndarray      # (n,k)

    # Helpers
    pca: PCA
    scaler_mean: np.ndarray      # (n,)
    scaler_std: np.ndarray       # (n,)

    # Optional generalized VARX transition (if provided, overrides (c,A,G) in simulation)
    trans_beta: Optional[np.ndarray] = None          # (p,k)
    trans_feature_names: Optional[List[str]] = None  # (p,)
    trans_gvarx_cfg: Optional[GVarXConfig] = None
    trans_y_mean: Optional[np.ndarray] = None        # (k,)
    trans_y_std: Optional[np.ndarray] = None         # (k,)
    trans_z_mean: Optional[np.ndarray] = None        # (m,) or None
    trans_z_std: Optional[np.ndarray] = None         # (m,) or None

    def factor_scores(self, r_ex: pd.DataFrame) -> pd.DataFrame:
        """
        Compute y_t from returns using the fitted PCA pipeline.
        """
        X = r_ex.values.astype(float)
        if self.scaler_mean is not None and self.scaler_std is not None:
            X = (X - self.scaler_mean[None, :]) / np.clip(self.scaler_std[None, :], 1e-12, None)
        Y = self.pca.transform(X)
        cols = [f"y{i+1}" for i in range(Y.shape[1])]
        return pd.DataFrame(Y, index=r_ex.index, columns=cols)


def fit_latent_pca(r_ex: pd.DataFrame, train_end: int, cfg: LatentPCAConfig) -> Tuple[PCA, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Fit PCA on TRAIN r_ex[0:train_end+1] and return:
      pca, mean, std, y_df for ALL dates (train+test) by projecting.
    """
    assert train_end >= 10
    X_train = r_ex.iloc[: train_end + 1].values.astype(float)

    if cfg.standardize:
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0, ddof=0)
        std = np.where(std <= 1e-12, 1.0, std)
    else:
        mean = np.zeros(X_train.shape[1], dtype=float)
        std = np.ones(X_train.shape[1], dtype=float)

    X_train_z = (X_train - mean[None, :]) / std[None, :]
    pca = PCA(n_components=cfg.n_components, random_state=cfg.random_state)
    pca.fit(X_train_z)

    # project all
    X_all = r_ex.values.astype(float)
    X_all_z = (X_all - mean[None, :]) / std[None, :]
    Y_all = pca.transform(X_all_z)

    y_cols = [f"y{i+1}" for i in range(cfg.n_components)]
    y_df = pd.DataFrame(Y_all, index=r_ex.index, columns=y_cols)
    return pca, mean, std, y_df


def _ols_fit(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """
    Return beta in Y ~= X @ beta. Uses ridge for numerical stability.
    """
    XtX = X.T @ X
    XtX = XtX + ridge * np.eye(XtX.shape[0])
    XtY = X.T @ Y
    beta = np.linalg.solve(XtX, XtY)
    return beta


def fit_apt_mu_sigma(y_df: pd.DataFrame, r_ex: pd.DataFrame, train_end: int, shrink_cov: bool = True) -> AptFit:
    """
    Fit r^{ex}_{t+1} = a + B y_t + eps_{t+1} on TRAIN only.

    Alignment:
      y_t uses dates [0 .. train_end-1]
      r_ex_{t+1} uses dates [1 .. train_end]
    so we use T_train = train_end observations for regression.
    """
    assert train_end >= 5
    y = y_df.values.astype(float)
    rx = r_ex.values.astype(float)
    n = rx.shape[1]
    k = y.shape[1]

    Y = rx[1 : train_end + 1, :]  # (T_train, n)
    X = y[0 : train_end, :]       # (T_train, k)

    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  # (T_train, 1+k)
    beta = _ols_fit(X1, Y)  # (1+k, n)
    a = beta[0, :]          # (n,)
    B = beta[1:, :].T       # (n,k)

    resid = Y - X1 @ beta   # (T_train, n)
    if shrink_cov:
        lw = LedoitWolf().fit(resid)
        Sigma = lw.covariance_
    else:
        Sigma = np.cov(resid, rowvar=False, ddof=0)

    return AptFit(a=a, B=B, Sigma=Sigma, resid=resid)


def _compute_mean_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=0)
    std = np.where(std <= 1e-12, 1.0, std)
    return mean.astype(float), std.astype(float)


def _gvarx_phi_numpy(
    Xy: np.ndarray,
    Z: Optional[np.ndarray],
    cfg: GVarXConfig,
    *,
    y_mean: Optional[np.ndarray] = None,
    y_std: Optional[np.ndarray] = None,
    z_mean: Optional[np.ndarray] = None,
    z_std: Optional[np.ndarray] = None,
    fit_stats: bool = False,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Build a design matrix Phi(y_t,z_t) for generalized VARX.

    Returns:
      Phi: (T, p)
      names: list of length p
      y_mean,y_std,z_mean,z_std: standardization stats (train)
    """
    T, k = Xy.shape
    m = 0 if Z is None else int(Z.shape[1])

    if fit_stats:
        if cfg.standardize_inputs:
            y_mean, y_std = _compute_mean_std(Xy)
            if Z is not None:
                z_mean, z_std = _compute_mean_std(Z)
            else:
                z_mean, z_std = None, None
        else:
            y_mean = np.zeros(k, dtype=float)
            y_std = np.ones(k, dtype=float)
            if Z is not None:
                z_mean = np.zeros(m, dtype=float)
                z_std = np.ones(m, dtype=float)
            else:
                z_mean, z_std = None, None

    # apply standardization (or identity)
    y_mean = np.zeros(k, dtype=float) if y_mean is None else y_mean
    y_std = np.ones(k, dtype=float) if y_std is None else y_std
    Yz = (Xy - y_mean[None, :]) / y_std[None, :]

    if Z is not None:
        z_mean = np.zeros(m, dtype=float) if z_mean is None else z_mean
        z_std = np.ones(m, dtype=float) if z_std is None else z_std
        Zz = (Z - z_mean[None, :]) / z_std[None, :]
    else:
        Zz = None

    # Optional clipping in standardized space for stability (helps prevent
    # polynomial feature explosions in long-horizon simulation).
    clip = getattr(cfg, "clip_std_abs", None)
    if clip is not None:
        Yz = np.clip(Yz, -float(clip), float(clip))
        if Zz is not None:
            Zz = np.clip(Zz, -float(clip), float(clip))

    feats: list[np.ndarray] = []
    names: list[str] = []

    # intercept
    feats.append(np.ones((T, 1), dtype=float))
    names.append("1")

    # linear y
    feats.append(Yz)
    names.extend([f"y{i+1}" for i in range(k)])

    # linear z
    if Zz is not None:
        feats.append(Zz)
        names.extend([f"z{j+1}" for j in range(m)])

    if cfg.include_squares:
        feats.append(Yz ** 2)
        names.extend([f"y{i+1}^2" for i in range(k)])
        if Zz is not None:
            feats.append(Zz ** 2)
            names.extend([f"z{j+1}^2" for j in range(m)])

    if cfg.include_yz_cross and (Zz is not None):
        yz = (Yz[:, :, None] * Zz[:, None, :]).reshape(T, k * m)
        feats.append(yz)
        names.extend([f"y{i+1}*z{j+1}" for i in range(k) for j in range(m)])

    if cfg.include_pairwise_y:
        cols = []
        cn = []
        for i in range(k):
            for j in range(i + 1, k):
                cols.append((Yz[:, i] * Yz[:, j])[:, None])
                cn.append(f"y{i+1}*y{j+1}")
        if cols:
            feats.append(np.concatenate(cols, axis=1))
            names.extend(cn)

    if cfg.include_pairwise_z and (Zz is not None):
        cols = []
        cn = []
        for i in range(m):
            for j in range(i + 1, m):
                cols.append((Zz[:, i] * Zz[:, j])[:, None])
                cn.append(f"z{i+1}*z{j+1}")
        if cols:
            feats.append(np.concatenate(cols, axis=1))
            names.extend(cn)

    Phi = np.concatenate(feats, axis=1)
    return Phi, names, y_mean, y_std, z_mean, z_std


def fit_varx_transition(
    y_df: pd.DataFrame,
    z_df: Optional[pd.DataFrame],
    train_end: int,
    gvarx_cfg: Optional[GVarXConfig] = None,
) -> VarXFit:
    """Fit the factor transition on TRAIN.

    Linear baseline:
      y_{t+1} = c + A y_t + G z_t + u_{t+1}

    Optional generalized VARX:
      y_{t+1} = Phi(y_t, z_t) @ Beta + u_{t+1}
    where Phi includes an intercept + nonlinear features (poly2).

    Alignment:
      y_t uses dates [0 .. train_end-1]
      y_{t+1} uses dates [1 .. train_end]
      z_t uses dates [0 .. train_end-1] (if provided)
    """
    y = y_df.values.astype(float)
    k = y.shape[1]

    Y = y[1 : train_end + 1, :]   # (T_train, k)
    Xy = y[0 : train_end, :]      # (T_train, k)

    Z = None
    m = 0
    if z_df is not None:
        z = z_df.values.astype(float)
        Z = z[0 : train_end, :]
        m = int(Z.shape[1])

    # --- Always compute the linear baseline params (useful diagnostics) ---
    if Z is not None:
        X_lin = np.concatenate([np.ones((Xy.shape[0], 1)), Xy, Z], axis=1)
    else:
        X_lin = np.concatenate([np.ones((Xy.shape[0], 1)), Xy], axis=1)

    beta_lin = _ols_fit(X_lin, Y, ridge=1e-8)  # (1+k+m, k)
    c = beta_lin[0, :]
    A = beta_lin[1 : 1 + k, :].T
    G = beta_lin[1 + k :, :].T if m > 0 else np.zeros((k, 0), dtype=float)

    # --- If no generalized transition requested, return the baseline fit ---
    if (gvarx_cfg is None) or (gvarx_cfg.kind == "linear"):
        resid = Y - X_lin @ beta_lin
        Q = np.cov(resid, rowvar=False, ddof=0)
        Q = np.atleast_2d(Q)
        return VarXFit(c=c, A=A, G=G, Q=Q, resid=resid)

    if gvarx_cfg.kind != "poly2":
        raise ValueError(f"Unknown gvarx_cfg.kind: {gvarx_cfg.kind}")

    Phi, names, y_mean, y_std, z_mean, z_std = _gvarx_phi_numpy(Xy, Z, gvarx_cfg, fit_stats=True)
    beta = _ols_fit(Phi, Y, ridge=float(gvarx_cfg.ridge))
    resid = Y - Phi @ beta

    if gvarx_cfg.shrink_Q:
        lw = LedoitWolf().fit(resid)
        Q = lw.covariance_
    else:
        Q = np.cov(resid, rowvar=False, ddof=0)
        Q = np.atleast_2d(Q)
    return VarXFit(
        c=c,
        A=A,
        G=G,
        Q=Q,
        resid=resid,
        beta=beta,
        feature_names=names,
        gvarx_cfg=gvarx_cfg,
        y_mean=y_mean,
        y_std=y_std,
        z_mean=z_mean,
        z_std=z_std,
    )


def estimate_cross_cov(apt: AptFit, varx: VarXFit) -> np.ndarray:
    """
    Estimate Cross = Cov(eps, u) on aligned residual samples.
    """
    eps = apt.resid
    u = varx.resid
    T = min(eps.shape[0], u.shape[0])
    eps = eps[-T:, :]
    u = u[-T:, :]
    # center
    eps = eps - eps.mean(axis=0, keepdims=True)
    u = u - u.mean(axis=0, keepdims=True)
    Cross = (eps.T @ u) / float(T)  # (n,k)
    return Cross


def build_discrete_latent_market_model(
    r_ex: pd.DataFrame,
    z: Optional[pd.DataFrame],
    train_end: int,
    pca_cfg: LatentPCAConfig,
    gvarx_cfg: Optional[GVarXConfig] = None,
) -> Tuple[DiscreteLatentMarketModel, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Fit PCA latent factors, APT model, VARX transition, and Cross on TRAIN only.
    Returns (model, y_df_all, z_aligned).
    """
    # align indices
    if z is not None:
        common = r_ex.index.intersection(z.index)
        r_ex = r_ex.loc[common].copy()
        z = z.loc[common].copy()
    # fit PCA on train
    pca, mean, std, y_df = fit_latent_pca(r_ex, train_end=train_end, cfg=pca_cfg)

    # fit APT + VARX
    apt = fit_apt_mu_sigma(y_df, r_ex, train_end=train_end, shrink_cov=True)
    varx = fit_varx_transition(y_df, z, train_end=train_end, gvarx_cfg=gvarx_cfg)
    Cross = estimate_cross_cov(apt, varx)

    model = DiscreteLatentMarketModel(
        asset_names=list(r_ex.columns),
        state_names=list(y_df.columns),
        exog_names=[] if z is None else list(z.columns),
        a=apt.a,
        B=apt.B,
        Sigma=apt.Sigma,
        c=varx.c,
        A=varx.A,
        G=varx.G,
        Q=varx.Q,
        Cross=Cross,
        pca=pca,
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
    return model, y_df, z
