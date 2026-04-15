"""
P-PGDPO deployment projection (Algorithm 2 style).

Given:
- warm-up policy nets (pi_theta, C_phi)
- a deployment state (t, X, Y)
we:
1) run Monte Carlo rollouts from that state
2) compute stabilized costates via autograd:
      lambda = d/dX0 E[J]
      d_x_lambda = d^2/dX0^2 E[J]
      d_Y_lambda = d/dY0 (lambda)  (vector)
3) plug into PMP FOCs:
      C = (U')^{-1}(exp(delta*t)*lambda)
      pi = -(1/(X*d_x_lambda)) * Sigma^{-1} [ lambda*(mu-r1) + (sigma*rho*sigmaY)*d_Y_lambda ]

We implement for CRRA and our TorchMarketModel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .simulator import SimConfig, TorchMarketModel, inv_marginal_utility, rollout_paths


@dataclass
class DeployConfig:
    mc_rollouts: int = 512
    eps_dxx: float = 1e-8   # stabilize division by d_x_lambda
    clip_pi: float = 10.0   # optional cap on portfolio fractions


@torch.no_grad()
def _mu_excess_at(model: TorchMarketModel, y: torch.Tensor) -> torch.Tensor:
    return model.mu_excess(y)


def project_p_pgdpo(
    market: TorchMarketModel,
    policy: nn.Module,
    cons_policy: Optional[nn.Module],
    X: float,
    Y: np.ndarray,
    t_years: float,
    sim_cfg: SimConfig,
    deploy_cfg: DeployConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Tuple[np.ndarray, Optional[float], dict]:
    """
    Returns:
      pi_pmp: (n,) numpy
      C_pmp: optional float (consumption rate at deploy state)
      info: dict with lambda, d_x_lambda, d_Y_lambda
    """
    policy.eval()
    if cons_policy is not None:
        cons_policy.eval()

    n = market.n_assets
    k = market.n_states

    # Scalar initial conditions with grad
    x0 = torch.tensor([X], device=device, dtype=dtype, requires_grad=True)  # (1,)
    y0 = torch.tensor(Y, device=device, dtype=dtype, requires_grad=True)    # (k,)

    # Broadcast to mc_rollouts
    X0 = x0.view(1, 1).repeat(deploy_cfg.mc_rollouts, 1)
    Y0 = y0.view(1, k).repeat(deploy_cfg.mc_rollouts, 1)

    # Use a local sim config with same horizon; "t" only matters through consumption FOC exp(delta*t)
    J_paths = rollout_paths(
        model=market,
        policy=policy,
        cons_policy=cons_policy,
        X0=X0,
        Y0=Y0,
        cfg=sim_cfg,
    )
    J = J_paths.mean()

    # lambda = dJ/dx0
    (lam,) = torch.autograd.grad(J, x0, create_graph=True)
    # d_x_lambda
    (dxx,) = torch.autograd.grad(lam, x0, create_graph=True)
    # d_Y_lambda (vector)
    (dY,) = torch.autograd.grad(lam, y0, create_graph=True)

    # Consumption via PMP FOC (optional)
    C_pmp = None
    if sim_cfg.include_consumption and cons_policy is not None:
        # e^{-delta t} U'(C*) = lambda  => C* = (U')^{-1}(exp(delta*t)*lambda)
        u_arg = torch.exp(torch.tensor(sim_cfg.delta * t_years, device=device, dtype=dtype)) * lam
        C = inv_marginal_utility(u_arg, gamma=sim_cfg.gamma)
        C_pmp = float(C.detach().cpu().item())

    # Portfolio via PMP
    y_tensor = y0.view(1, k)  # (1,k)
    mu_ex = market.mu_excess(y_tensor).view(n)  # (n,)
    # vec = lambda*(mu-r1) + cross * dY
    vec = lam * mu_ex + (market.cross @ dY)  # (n,)
    # Solve Sigma * z = vec
    z = torch.linalg.solve(market.Sigma, vec)

    denom = (x0 * (dxx + deploy_cfg.eps_dxx))  # (1,)
    pi = -(1.0 / denom) * z  # (n,)
    pi = torch.clamp(pi, -deploy_cfg.clip_pi, deploy_cfg.clip_pi)

    pi_np = pi.detach().cpu().numpy().astype(float)

    info = {
        "lambda": float(lam.detach().cpu().item()),
        "d_x_lambda": float(dxx.detach().cpu().item()),
        "d_Y_lambda": dY.detach().cpu().numpy().astype(float),
        "J": float(J.detach().cpu().item()),
    }
    return pi_np, C_pmp, info
