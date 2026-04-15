"""
Differentiable simulator for (X_t, Y_t) with correlated Brownian increments.

We implement a simplified version consistent with the paper's Euler scheme, and also provide
a positive-wealth exact update for the no-consumption case.

SDE (annualized parameters):
  dX = X(r + pi^T mu_excess) dt - C dt + X * (pi^T sigma) dW^X
  dY = K(θ - Y) dt + diag(sigmaY) dW^Y

We represent Brownian increments with:
  [dW^X; dW^Y] ~ N(0, Omega dt), Omega = [[I, rho],[rho^T, I]]

Inputs:
- MarketModel already provides:
    Sigma (cov), sigma (chol), cross = sigma*rho*sigmaY
- We do NOT need rho explicitly inside the simulator if we only simulate independent W^X and W^Y,
  but for P-PGDPO projection you may want consistency. So we allow correlated sampling via Omega.

Speed note:
- The simulator is vectorized across paths (batch dimension).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class TorchMarketModel:
    n_assets: int
    n_states: int
    periods_per_year: int

    r_annual: torch.Tensor            # scalar
    a: torch.Tensor                   # (n,)
    B: torch.Tensor                   # (n,k)
    Sigma: torch.Tensor               # (n,n)
    sigma: torch.Tensor               # (n,n) chol
    K: torch.Tensor                   # (k,k)
    theta: torch.Tensor               # (k,)
    sigmaY_diag: torch.Tensor         # (k,)
    cross: torch.Tensor               # (n,k) = sigma*rho*sigmaY

    L_omega: Optional[torch.Tensor] = None  # (n+k, n+k), cholesky of Omega

    @staticmethod
    def from_numpy(model, device: torch.device, dtype: torch.dtype = torch.float64) -> "TorchMarketModel":
        n = len(model.asset_tickers)
        k = len(model.state_names)
        # Build Omega cholesky from cross term: cross = sigma * rho * sigmaY
        # Derive rho = sigma^{-1} cross diag(sigmaY)^{-1}
        sigma = torch.tensor(model.sigma, device=device, dtype=dtype)
        cross = torch.tensor(model.cross, device=device, dtype=dtype)
        sigmaY = torch.tensor(model.sigmaY_diag, device=device, dtype=dtype)
        # solve sigma * M = cross
        M = torch.linalg.solve(sigma, cross)
        rho = M / torch.clamp(sigmaY, min=1e-12)[None, :]
        rho = torch.clamp(rho, -0.95, 0.95)
        Omega = torch.eye(n + k, device=device, dtype=dtype)
        Omega[:n, n:] = rho
        Omega[n:, :n] = rho.T
        # Try cholesky; if fails, shrink rho until works
        L_omega = None
        shrink = 1.0
        for _ in range(20):
            try:
                Om = torch.eye(n + k, device=device, dtype=dtype)
                Om[:n, n:] = shrink * rho
                Om[n:, :n] = (shrink * rho).T
                L_omega = torch.linalg.cholesky(Om)
                break
            except RuntimeError:
                shrink *= 0.9
        if L_omega is None:
            # last resort: identity (no cross-corr)
            L_omega = torch.eye(n + k, device=device, dtype=dtype)

        return TorchMarketModel(
            n_assets=n,
            n_states=k,
            periods_per_year=model.periods_per_year,
            r_annual=torch.tensor(float(model.r_annual), device=device, dtype=dtype),
            a=torch.tensor(model.a, device=device, dtype=dtype),
            B=torch.tensor(model.B, device=device, dtype=dtype),
            Sigma=torch.tensor(model.Sigma, device=device, dtype=dtype),
            sigma=sigma,
            K=torch.tensor(model.K, device=device, dtype=dtype),
            theta=torch.tensor(model.theta, device=device, dtype=dtype),
            sigmaY_diag=torch.tensor(model.sigmaY_diag, device=device, dtype=dtype),
            cross=cross,
            L_omega=L_omega,
        )

    def mu_excess(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (batch,k)
        returns: (batch,n) annualized excess drift
        """
        return self.a[None, :] + y @ self.B.T

    def muY(self, y: torch.Tensor) -> torch.Tensor:
        """
        OU drift: K(θ - y)
        """
        return (self.theta[None, :] - y) @ self.K.T

    def sample_dW(self, batch: int, dt: float, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample correlated Brownian increments (dW^X, dW^Y).
        """
        n, k = self.n_assets, self.n_states
        z = torch.randn(batch, n + k, device=device, dtype=dtype)
        if self.L_omega is None:
            dW = z * (dt ** 0.5)
        else:
            dW = (z @ self.L_omega.T) * (dt ** 0.5)
        return dW[:, :n], dW[:, n:]


def crra_utility(x: torch.Tensor, gamma: float) -> torch.Tensor:
    x = torch.clamp(x, min=1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return torch.log(x)
    return (x ** (1.0 - gamma)) / (1.0 - gamma)


def inv_marginal_utility(u: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    (U')^{-1}(u) for CRRA.
    For gamma != 1: U'(c)=c^{-gamma} -> c=u^{-1/gamma}
    """
    u = torch.clamp(u, min=1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / u
    return u ** (-1.0 / gamma)


@dataclass
class SimConfig:
    horizon_years: float = 1.0
    steps_per_year: int = 52
    include_consumption: bool = False
    delta: float = 0.0     # discount rate
    kappa: float = 1.0     # terminal utility weight
    gamma: float = 5.0
    # Stability
    use_exact_log_wealth_if_no_consumption: bool = True


def rollout_paths(
    model: TorchMarketModel,
    policy,
    cons_policy,
    X0: torch.Tensor,   # (batch,1)
    Y0: torch.Tensor,   # (batch,k)
    cfg: SimConfig,
) -> torch.Tensor:
    """
    Simulate forward and compute objective per path (batch,).
    """
    device = X0.device
    dtype = X0.dtype
    batch = X0.shape[0]
    n = model.n_assets
    k = model.n_states

    steps = int(cfg.horizon_years * cfg.steps_per_year)
    dt = 1.0 / cfg.steps_per_year

    X = X0
    Y = Y0
    J = torch.zeros(batch, device=device, dtype=dtype)

    for t_idx in range(steps):
        tau = torch.full((batch, 1), 1.0 - t_idx / steps, device=device, dtype=dtype)
        pi = policy(tau, X, Y)  # (batch,n)

        C = None
        if cfg.include_consumption and cons_policy is not None:
            C = cons_policy(tau, X, Y)  # (batch,1)
        else:
            C = torch.zeros(batch, 1, device=device, dtype=dtype)

        dWx, dWy = model.sample_dW(batch=batch, dt=dt, device=device, dtype=dtype)

        # Factor update (Euler)
        Y = Y + model.muY(Y) * dt + dWy * model.sigmaY_diag[None, :]

        # Wealth update
        mu_ex = model.mu_excess(Y)  # (batch,n)
        r = model.r_annual

        if cfg.use_exact_log_wealth_if_no_consumption and (not cfg.include_consumption):
            # Exact multiplicative update for dX/X
            sig_p = torch.einsum("bi,ij->bj", pi, model.sigma)  # (batch,n)
            drift = (r + (pi * mu_ex).sum(dim=1)) - 0.5 * (sig_p ** 2).sum(dim=1)  # (batch,)
            diff = (sig_p * dWx).sum(dim=1)  # (batch,)
            X = X * torch.exp((drift * dt + diff).unsqueeze(1))
        else:
            # Euler (can go negative if dt large and leverage high)
            sig_p = torch.einsum("bi,ij->bj", pi, model.sigma)  # (batch,n)
            drift = X * (r + (pi * mu_ex).sum(dim=1, keepdim=True)) - C
            diff = X * (sig_p * dWx).sum(dim=1, keepdim=True)
            X = X + drift * dt + diff
            X = torch.clamp(X, min=1e-12)

        # running utility
        if cfg.include_consumption:
            disc = torch.exp(torch.tensor(-cfg.delta * (t_idx * dt), device=device, dtype=dtype))
            J = J + disc * crra_utility(C.squeeze(1), cfg.gamma) * dt

    # terminal utility
    discT = torch.exp(torch.tensor(-cfg.delta * cfg.horizon_years, device=device, dtype=dtype))
    J = J + cfg.kappa * discT * crra_utility(X.squeeze(1), cfg.gamma)
    return J
