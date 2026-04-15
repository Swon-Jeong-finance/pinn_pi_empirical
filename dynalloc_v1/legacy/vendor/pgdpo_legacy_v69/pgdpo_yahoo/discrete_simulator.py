"""
Discrete-time simulator and PG-DPO warm-up training for the discrete latent market model.

We simulate monthly simple returns directly:
  R_p,t+1 = rf_{t+1} + pi_t^T r^{ex}_{t+1}
  X_{t+1} = X_t * (1 + R_p,t+1)

Factor dynamics:
  baseline: y_{t+1} = c + A y_t + G z_t + u_{t+1}
  optional generalized VARX: y_{t+1} = Phi(y_t,z_t) @ Beta + u_{t+1}

Return model:
  r^{ex}_{t+1} = a + B y_t + eps_{t+1}

with joint normal innovations:
  [eps_{t+1}; u_{t+1}] ~ N(0, [[Sigma, Cross],[Cross^T, Q]])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .simulator import crra_utility
from .discrete_latent_model import DiscreteLatentMarketModel


@dataclass
class DiscreteSimConfig:
    horizon_steps: int = 120   # 10 years monthly
    gamma: float = 5.0
    kappa: float = 1.0         # terminal utility weight
    # Stability
    clamp_wealth_min: float = 1e-12
    # Clamp factor states in standardized units (relative to TRAIN mean/std).
    # Set to None to disable. Helps prevent gvarx explosions in long-horizon simulation.
    clamp_state_std_abs: Optional[float] = 12.0
    # Clamp monthly simple portfolio return for numerical stability (optional).
    clamp_port_ret_min: float = -0.99
    clamp_port_ret_max: float = 5.0


@dataclass
class TrainConfig:
    iters: int = 800
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0
    print_every: int = 200


class EpisodeSampler:
    """
    Sample training episodes by picking a start index and taking the corresponding exogenous path.
    """
    def __init__(self, y_all: np.ndarray, z_all: Optional[np.ndarray], start_max: int, horizon: int):
        """
        y_all: (T,k) states (observed/estimated) for all dates (aligned)
        z_all: (T,m) exogenous drivers for all dates, or None
        start_max: maximum start index allowed (inclusive)
        horizon: episode length in steps (months)
        """
        self.y_all = y_all
        self.z_all = z_all
        self.start_max = int(start_max)
        self.horizon = int(horizon)
        self.k = y_all.shape[1]
        self.m = 0 if z_all is None else z_all.shape[1]

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        idx = np.random.randint(0, self.start_max + 1, size=batch_size)
        y0 = torch.tensor(self.y_all[idx], device=device, dtype=dtype)
        x0 = torch.ones(batch_size, 1, device=device, dtype=dtype)

        z_path = None
        if self.z_all is not None:
            # gather z[t : t+horizon] for each path
            # shape: (batch, horizon, m)
            paths = []
            for s in idx:
                paths.append(self.z_all[s : s + self.horizon, :])
            z_path = torch.tensor(np.stack(paths, axis=0), device=device, dtype=dtype)

        return x0, y0, z_path


class TorchDiscreteLatentModel:
    def __init__(self, model: DiscreteLatentMarketModel, device: torch.device, dtype: torch.dtype = torch.float64):
        self.device = device
        self.dtype = dtype

        self.n = len(model.asset_names)
        self.k = len(model.state_names)
        self.m = len(model.exog_names)

        self.a = torch.tensor(model.a, device=device, dtype=dtype)           # (n,)
        self.B = torch.tensor(model.B, device=device, dtype=dtype)           # (n,k)
        self.c = torch.tensor(model.c, device=device, dtype=dtype)           # (k,)
        self.A = torch.tensor(model.A, device=device, dtype=dtype)           # (k,k)
        self.G = torch.tensor(model.G, device=device, dtype=dtype)           # (k,m) or (k,0)

        # Optional generalized VARX transition
        self.use_gvarx = model.trans_beta is not None
        self.gvarx_cfg = model.trans_gvarx_cfg

        self.trans_beta = None
        self.trans_y_mean = None
        self.trans_y_std = None
        self.trans_z_mean = None
        self.trans_z_std = None

        if self.use_gvarx:
            self.trans_beta = torch.tensor(model.trans_beta, device=device, dtype=dtype)  # (p,k)

            # standardization stats (train)
            ym = model.trans_y_mean if model.trans_y_mean is not None else np.zeros(self.k, dtype=float)
            ys = model.trans_y_std if model.trans_y_std is not None else np.ones(self.k, dtype=float)
            self.trans_y_mean = torch.tensor(ym, device=device, dtype=dtype)
            self.trans_y_std = torch.tensor(np.clip(ys, 1e-12, None), device=device, dtype=dtype)

            if self.m > 0:
                zm = model.trans_z_mean if model.trans_z_mean is not None else np.zeros(self.m, dtype=float)
                zs = model.trans_z_std if model.trans_z_std is not None else np.ones(self.m, dtype=float)
                self.trans_z_mean = torch.tensor(zm, device=device, dtype=dtype)
                self.trans_z_std = torch.tensor(np.clip(zs, 1e-12, None), device=device, dtype=dtype)

        Sigma = torch.tensor(model.Sigma, device=device, dtype=dtype)
        Q = torch.tensor(model.Q, device=device, dtype=dtype)
        Cross = torch.tensor(model.Cross, device=device, dtype=dtype)

        # Store raw moment blocks (useful for diagnostics / P-PGDPO projection)
        self.Sigma = Sigma
        self.Q = Q
        self.Cross = Cross

        # Sanity check: fail fast if the fitted moments contain NaNs/Infs
        for name, T in [("Sigma", Sigma), ("Q", Q), ("Cross", Cross)]:
            if not torch.isfinite(T).all():
                raise ValueError(f"Non-finite entries detected in {name}. Check data alignment / NaNs.")

        # Build joint covariance and its Cholesky for reparameterized sampling
        joint = torch.zeros(self.n + self.k, self.n + self.k, device=device, dtype=dtype)
        joint[: self.n, : self.n] = Sigma
        joint[self.n :, self.n :] = Q
        joint[: self.n, self.n :] = Cross
        joint[self.n :, : self.n] = Cross.T

        # Ensure PD by adding jitter / shrinking cross if needed
        L = None
        jitter = 1e-8
        shrink = 1.0
        for _ in range(40):
            try:
                J = joint.clone()
                # shrink off-diagonal blocks
                J[: self.n, self.n :] *= shrink
                J[self.n :, : self.n] *= shrink
                # jitter
                J = J + jitter * torch.eye(self.n + self.k, device=device, dtype=dtype)
                L = torch.linalg.cholesky(J)
                joint = J
                break
            except RuntimeError:
                shrink *= 0.9
                jitter *= 2.0
        if L is None:
            # fallback: block-diagonal
            J = torch.zeros(self.n + self.k, self.n + self.k, device=device, dtype=dtype)
            J[: self.n, : self.n] = Sigma + 1e-6 * torch.eye(self.n, device=device, dtype=dtype)
            J[self.n :, self.n :] = Q + 1e-6 * torch.eye(self.k, device=device, dtype=dtype)
            L = torch.linalg.cholesky(J)
            joint = J

        self.joint = joint
        self.L_joint = L

    def sample_innov(self, batch: int, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (eps_x, u_y) with shapes (batch,n), (batch,k).
        """
        z = torch.randn(batch, self.n + self.k, device=self.device, dtype=self.dtype, generator=generator)
        innov = z @ self.L_joint.T
        eps = innov[:, : self.n]
        u = innov[:, self.n :]
        return eps, u

    def mu_excess(self, y: torch.Tensor) -> torch.Tensor:
        # (batch,n) = a + y @ B^T
        return self.a[None, :] + y @ self.B.T

    def _phi_gvarx(self, y: torch.Tensor, z_t: Optional[torch.Tensor]) -> torch.Tensor:
        """Torch version of the Phi(y_t, z_t) feature map.

        Must match the ordering used in discrete_latent_model._gvarx_phi_numpy.
        """
        cfg = self.gvarx_cfg
        if cfg is None:
            # Fallback: match defaults in GVarXConfig
            class _Tmp:
                kind = "poly2"
                standardize_inputs = True
                include_squares = True
                include_yz_cross = True
                include_pairwise_y = False
                include_pairwise_z = False
                clip_std_abs = 8.0
            cfg = _Tmp()

        batch = y.shape[0]

        # standardize (train stats) if requested
        if getattr(cfg, "standardize_inputs", True) and (self.trans_y_mean is not None) and (self.trans_y_std is not None):
            Yz = (y - self.trans_y_mean[None, :]) / self.trans_y_std[None, :]
        else:
            Yz = y

        Zz = None
        if (z_t is not None) and (self.m > 0):
            if getattr(cfg, "standardize_inputs", True) and (self.trans_z_mean is not None) and (self.trans_z_std is not None):
                Zz = (z_t - self.trans_z_mean[None, :]) / self.trans_z_std[None, :]
            else:
                Zz = z_t

        # Optional clipping in standardized space for stability (mirrors
        # discrete_latent_model._gvarx_phi_numpy).
        clip = getattr(cfg, "clip_std_abs", None)
        if clip is not None:
            c = float(clip)
            Yz = torch.clamp(Yz, min=-c, max=c)
            if Zz is not None:
                Zz = torch.clamp(Zz, min=-c, max=c)

        feats = []
        # intercept
        feats.append(torch.ones(batch, 1, device=y.device, dtype=y.dtype))
        # linear y
        feats.append(Yz)
        # linear z
        if Zz is not None:
            feats.append(Zz)

        if getattr(cfg, "include_squares", True):
            feats.append(Yz ** 2)
            if Zz is not None:
                feats.append(Zz ** 2)

        if getattr(cfg, "include_yz_cross", True) and (Zz is not None):
            yz = (Yz.unsqueeze(2) * Zz.unsqueeze(1)).reshape(batch, self.k * self.m)
            feats.append(yz)

        if getattr(cfg, "include_pairwise_y", False):
            cols = []
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    cols.append((Yz[:, i] * Yz[:, j]).unsqueeze(1))
            if cols:
                feats.append(torch.cat(cols, dim=1))

        if getattr(cfg, "include_pairwise_z", False) and (Zz is not None):
            cols = []
            for i in range(self.m):
                for j in range(i + 1, self.m):
                    cols.append((Zz[:, i] * Zz[:, j]).unsqueeze(1))
            if cols:
                feats.append(torch.cat(cols, dim=1))

        return torch.cat(feats, dim=1)

    def step_factor(self, y: torch.Tensor, z_t: Optional[torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        # y_{t+1} = E[y_{t+1}|y_t,z_t] + u
        if self.use_gvarx and (self.trans_beta is not None):
            Phi = self._phi_gvarx(y, z_t)
            # (batch,k) = (batch,p) @ (p,k)
            y_next = Phi @ self.trans_beta + u

            # Stabilize: clamp next-state in standardized units using TRAIN stats.
            cfg = self.gvarx_cfg
            clip = getattr(cfg, "clip_std_abs", None) if cfg is not None else None
            if (clip is not None) and (self.trans_y_mean is not None) and (self.trans_y_std is not None):
                c = float(clip)
                y_std = (y_next - self.trans_y_mean[None, :]) / self.trans_y_std[None, :]
                y_std = torch.clamp(y_std, min=-c, max=c)
                y_next = self.trans_y_mean[None, :] + y_std * self.trans_y_std[None, :]

            return y_next

        # baseline linear VARX
        y_next = self.c[None, :] + y @ self.A.T
        if (z_t is not None) and (self.m > 0):
            y_next = y_next + z_t @ self.G.T
        return y_next + u

    def clamp_state(self, y: torch.Tensor, cfg: DiscreteSimConfig) -> torch.Tensor:
        # Clamp factor state in standardized units (TRAIN mean/std) for stability.
        clip = cfg.clamp_state_std_abs
        if clip is None:
            return y
        c = float(clip)
        if (self.trans_y_mean is not None) and (self.trans_y_std is not None):
            y_std = (y - self.trans_y_mean[None, :]) / self.trans_y_std[None, :]
            y_std = torch.clamp(y_std, min=-c, max=c)
            return self.trans_y_mean[None, :] + y_std * self.trans_y_std[None, :]
        # fallback: raw clamp (should rarely be used in our configs)
        return torch.clamp(y, min=-c, max=c)


def rollout_paths_discrete(
    model: TorchDiscreteLatentModel,
    policy: nn.Module,
    X0: torch.Tensor,         # (batch,1)
    Y0: torch.Tensor,         # (batch,k)
    Z_path: Optional[torch.Tensor],  # (batch,H,m) or None
    rf_month: float,
    cfg: DiscreteSimConfig,
    *,
    generator: Optional[torch.Generator] = None,
    tau0: Optional[float] = None,
    tau_step: Optional[float] = None,
) -> torch.Tensor:
    """
    Vectorized discrete simulation for a batch of paths.
    Returns objective per path: J (batch,)
    """
    device = X0.device
    dtype = X0.dtype
    batch = X0.shape[0]
    H = int(cfg.horizon_steps)

    X = X0
    Y = Y0
    for t in range(H):
        if tau0 is None:
            tau_val = 1.0 - t / H
        else:
            step = (1.0 / H) if (tau_step is None) else float(tau_step)
            tau_val = float(tau0) - t * step
        # keep in [0,1] for numerical stability
        tau_val = max(0.0, min(1.0, tau_val))
        tau = torch.full((batch, 1), tau_val, device=device, dtype=dtype)
        pi = policy(tau, X, Y)  # (batch,n)

        eps, u = model.sample_innov(batch=batch, generator=generator)
        r_ex = model.mu_excess(Y) + eps  # (batch,n) monthly excess return
        port_ret = rf_month + torch.sum(pi * r_ex, dim=1)  # (batch,)
        # Clamp portfolio monthly return for numerical stability (avoid -100% and huge jumps)
        port_ret = torch.clamp(port_ret, min=cfg.clamp_port_ret_min, max=cfg.clamp_port_ret_max)
        X = X * (1.0 + port_ret).unsqueeze(1)
        X = torch.clamp(X, min=cfg.clamp_wealth_min)

        z_t = None
        if Z_path is not None:
            z_t = Z_path[:, t, :]  # (batch,m)
        Y = model.step_factor(Y, z_t, u)
        # Optional: clamp factor state to a reasonable range to prevent...
        Y = model.clamp_state(Y, cfg)

    J = cfg.kappa * crra_utility(X.squeeze(1), cfg.gamma)
    # Hard fail-fast: NaNs here usually mean the fitted model contains NaNs (data issues)
    # or the simulated dynamics exploded. Better to stop than silently train on NaNs.
    if not torch.isfinite(J).all():
        raise RuntimeError(
            "Non-finite objective in rollout_paths_discrete(). "
            "Check: (1) data scaling (percent vs decimal), (2) NaNs in inputs (r_ex/rf/z), "
            "(3) stability of the fitted transition (especially under gvarx)."
        )
    return J


def train_pgdpo_discrete(
    model: TorchDiscreteLatentModel,
    policy: nn.Module,
    sampler: EpisodeSampler,
    rf_month: float,
    sim_cfg: DiscreteSimConfig,
    train_cfg: TrainConfig,
) -> None:
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    for it in range(train_cfg.iters):
        X0, Y0, Z = sampler.sample(train_cfg.batch_size, device=model.device, dtype=model.dtype)
        J = rollout_paths_discrete(model, policy, X0, Y0, Z, rf_month=rf_month, cfg=sim_cfg)
        loss = -J.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.clip_grad_norm is not None and train_cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=train_cfg.clip_grad_norm)
        opt.step()

        if (it % train_cfg.print_every) == 0 or it == train_cfg.iters - 1:
            print(f"[PG-DPO] iter={it:4d}  J_mean={float(J.mean().detach().cpu().item()): .6f}")
