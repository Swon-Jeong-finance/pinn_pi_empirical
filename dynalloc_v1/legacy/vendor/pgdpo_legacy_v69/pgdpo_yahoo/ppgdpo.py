
"""pgdpo_yahoo.ppgdpo

P-PGDPO "Pontryagin projection" for the discrete-time latent factor market used in
run_french49_10y_model_based_latent_varx_fred.py.

Goal:
- Train policy as usual (baseline PG-DPO / "DPO warm-up").
- At deployment / OOS evaluation time, optionally replace the policy action
  with a stagewise PMP/KKT-aligned action computed from *estimated costates*
  and a *barrier-regularized* quadratic micro-problem.

We focus on the common empirical constraint family:
- risky weights with optional box bounds and a total risky cap
- residual cash is always cash = 1 - sum(u)
- long-only + no-borrowing is recovered by u >= 0 and 1^T u <= 1

Micro-problem (concave with log-barrier):
  maximize_u  a^T u - 0.5 u^T Q u
              + eps_bar * (sum_i log(u_i-lower_i) + sum_i log(upper_i-u_i) + log slack)
  where slack = L_cap - 1^T u > 0
  (the upper-barrier term is skipped when no explicit upper bound is active)

Coefficients from discrete-time PMP-style second-order expansion:
  a = X * ( JX * mu_ex(y) + JXY @ Cross^T )          (B,n)
  Q = X^2 * (-JXX) * Sigma                           (B,n,n)

Costates are estimated by Monte Carlo rollouts + autograd, similar to PGDPO_TORCH:
  JX  = d/dX0 E[J]
  JXX = d^2/dX0^2 E[J]
  JXY = d/dY0 (JX)

This module is intentionally self-contained so it can be plugged into any state spec
(PCA / PLS / macro exog) without changing the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from .constraints import PortfolioConstraints, project_box_sum_torch
from .discrete_simulator import TorchDiscreteLatentModel, DiscreteSimConfig, rollout_paths_discrete


@dataclass
class PPGDPOConfig:
    # MC for costates
    mc_rollouts: int = 256
    sub_batch: int = 64
    seed: Optional[int] = 123

    # Barrier / projection solver
    L_cap: float = 1.0
    eps_bar: float = 1e-6
    ridge: float = 1e-10
    tau: float = 0.95               # fraction-to-the-boundary
    armijo: float = 1e-4            # Armijo (maximize)
    backtrack: float = 0.5
    max_newton: int = 30
    tol_grad: float = 1e-8
    max_ls: int = 20
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)

    # Numerical safety
    clamp_neg_jxx_min: float = 1e-12
    clamp_u_min: float = 1e-12
    interior_margin: float = 1e-8   # keep away from boundary to avoid log blow-ups


@torch.no_grad()
def _proj_simplex_batch(v: torch.Tensor, L: float) -> torch.Tensor:
    """Backward-compatible alias for nonnegative box+sum projection."""
    return project_box_sum_torch(v, lower=0.0, upper=None, cap=float(L))


@torch.no_grad()
def project_u_long_only_cash_barrier(
    a_vec: torch.Tensor,   # (B,n)
    Q_b: torch.Tensor,     # (B,n,n)
    cfg: PPGDPOConfig,
) -> torch.Tensor:
    """Barrier-regularized Newton solve for box-constrained risky weights + residual cash.

    Feasible set
    ------------
      lower <= u_i <= upper
      sum_i u_i <= L_cap
      cash = 1 - sum_i u_i

    with ``lower = short_floor`` (or 0 if shorting disabled) and optional
    ``upper = per_asset_cap`` from ``cfg.constraints``.

    Returns
    -------
      u : (B,n) projected risky weights.
    """
    dev, dt = a_vec.device, a_vec.dtype
    B, n = a_vec.shape
    constraints = cfg.constraints
    constraints.validate(n)

    L_cap = float(cfg.L_cap)
    lower = float(constraints.lower_bound())
    upper = constraints.upper_bound()
    eps_bar = float(cfg.eps_bar)
    ridge = float(cfg.ridge)
    tau = float(cfg.tau)
    armijo = float(cfg.armijo)
    back = float(cfg.backtrack)
    max_newton = int(cfg.max_newton)
    tol_grad = float(cfg.tol_grad)
    max_ls = int(cfg.max_ls)

    lo = torch.full((B, n), lower, device=dev, dtype=dt)
    has_upper = upper is not None and np.isfinite(float(upper))
    if has_upper:
        hi = torch.full((B, n), float(upper), device=dev, dtype=dt)
    else:
        hi = None

    I_b = torch.eye(n, device=dev, dtype=dt).expand(B, n, n)
    one = torch.ones(B, n, device=dev, dtype=dt)
    Jmat = torch.ones(B, n, n, device=dev, dtype=dt)

    # Initial: unconstrained normal equation + box+sum projection
    Qreg = Q_b + ridge * I_b
    rhs = a_vec.unsqueeze(-1)
    try:
        u0 = torch.linalg.solve(Qreg, rhs).squeeze(-1)
    except RuntimeError:
        u0 = (torch.linalg.pinv(Qreg) @ rhs).squeeze(-1)

    u = project_box_sum_torch(u0, lower=lower, upper=upper, cap=L_cap)

    # Pull slightly to strict interior for log-barrier stability.
    margin = max(float(cfg.interior_margin), 1e-12)
    slack = (L_cap - u.sum(dim=1, keepdim=True))
    if has_upper:
        u = torch.minimum(u, hi - margin)
    u = torch.maximum(u, lo + margin)
    # If we're too close to the sum constraint, shrink risky weights towards the lower bound.
    near_sum = slack <= margin
    if near_sum.any():
        span = (u - lo).clamp_min(1e-12)
        room = (u.sum(dim=1, keepdim=True) - lo.sum(dim=1, keepdim=True)).clamp_min(1e-12)
        target_total = L_cap - margin
        frac = ((u.sum(dim=1, keepdim=True) - target_total).clamp_min(0.0) / room).clamp(0.0, 1.0)
        u = torch.where(near_sum, u - frac * span, u)
    if has_upper:
        u = torch.minimum(u, hi - margin)
    u = torch.maximum(u, lo + margin)

    def Hbar_batch(uu: torch.Tensor) -> torch.Tensor:
        Quu = torch.bmm(Q_b, uu.unsqueeze(-1)).squeeze(-1)
        val = (a_vec * uu).sum(dim=1, keepdim=True) - 0.5 * (uu * Quu).sum(dim=1, keepdim=True)
        slk = (L_cap - uu.sum(dim=1, keepdim=True))
        feas = (uu > (lo + 0.0)).all(dim=1, keepdim=True) & (slk > 0)
        val = val + eps_bar * torch.log((uu - lo).clamp_min(1e-24)).sum(dim=1, keepdim=True)
        if has_upper:
            feas = feas & (uu < hi).all(dim=1, keepdim=True)
            val = val + eps_bar * torch.log((hi - uu).clamp_min(1e-24)).sum(dim=1, keepdim=True)
        val = val + eps_bar * torch.log(slk.clamp_min(1e-24))
        val = torch.where(feas, val, torch.full_like(val, -float("inf")))
        return val

    for _ in range(max_newton):
        slack = (L_cap - u.sum(dim=1, keepdim=True)).clamp_min(1e-12)

        Qu = torch.bmm(Q_b, u.unsqueeze(-1)).squeeze(-1)
        g = a_vec - Qu
        g = g + eps_bar * (1.0 / (u - lo).clamp_min(1e-12))
        if has_upper:
            g = g - eps_bar * (1.0 / (hi - u).clamp_min(1e-12))
        g = g - (eps_bar / slack) * one

        if g.abs().amax(dim=1).max().item() < tol_grad:
            break

        diag_term = eps_bar / ((u - lo).clamp_min(1e-12) ** 2)
        if has_upper:
            diag_term = diag_term + eps_bar / ((hi - u).clamp_min(1e-12) ** 2)

        H = -Q_b
        H = H - torch.diag_embed(diag_term)
        H = H - (eps_bar / (slack ** 2)).view(B, 1, 1) * Jmat
        H = H + ridge * I_b

        rhs = (-g).unsqueeze(-1)
        try:
            dlt = torch.linalg.solve(H, rhs).squeeze(-1)
        except RuntimeError:
            dlt = (torch.linalg.pinv(H) @ rhs).squeeze(-1)

        # fraction-to-the-boundary
        alpha = torch.ones(B, 1, device=dev, dtype=dt)

        neg_low = dlt < 0
        if neg_low.any():
            denom = (-dlt).clamp_min(1e-24)
            step_low = ((u - lo) / denom) * tau
            step_low = torch.where(neg_low, step_low, torch.full_like(step_low, float("inf")))
            alpha = torch.minimum(alpha, step_low.amin(dim=1, keepdim=True))

        if has_upper:
            pos_high = dlt > 0
            if pos_high.any():
                denom = dlt.clamp_min(1e-24)
                step_high = ((hi - u) / denom) * tau
                step_high = torch.where(pos_high, step_high, torch.full_like(step_high, float("inf")))
                alpha = torch.minimum(alpha, step_high.amin(dim=1, keepdim=True))

        dsum = dlt.sum(dim=1, keepdim=True)
        needs_sum_cap = dsum > 0
        if needs_sum_cap.any():
            alpha_sum = tau * slack / (dsum + 1e-24)
            alpha = torch.minimum(alpha, torch.where(needs_sum_cap, alpha_sum, alpha))

        # Armijo backtracking
        f0 = Hbar_batch(u)
        gTd = (g * dlt).sum(dim=1, keepdim=True)

        u_new = u.clone()
        accepted = torch.zeros(B, 1, dtype=torch.bool, device=dev)

        for _ls in range(max_ls):
            uc = u + alpha * dlt
            f1 = Hbar_batch(uc)
            ok = f1 >= (f0 + armijo * alpha * gTd)

            if ok.any():
                mask = ok & (~accepted)
                u_new = torch.where(mask, uc, u_new)
                f0 = torch.where(mask, f1, f0)
                accepted = accepted | ok

            if (~accepted).any():
                alpha = torch.where(accepted, alpha, alpha * back)
            else:
                break

        u = torch.where(accepted, u_new, u)

        # interior pull
        slack = (L_cap - u.sum(dim=1, keepdim=True))
        near_sum = slack <= margin
        if near_sum.any():
            span = (u - lo).clamp_min(1e-12)
            room = (u.sum(dim=1, keepdim=True) - lo.sum(dim=1, keepdim=True)).clamp_min(1e-12)
            target_total = L_cap - margin
            frac = ((u.sum(dim=1, keepdim=True) - target_total).clamp_min(0.0) / room).clamp(0.0, 1.0)
            u = torch.where(near_sum, u - frac * span, u)
        if has_upper:
            u = torch.minimum(u, hi - margin)
        u = torch.maximum(u, lo + margin)

    # final safety
    u = project_box_sum_torch(u, lower=lower, upper=upper, cap=L_cap)
    return u


def estimate_costates_discrete(
    model: TorchDiscreteLatentModel,
    policy: nn.Module,
    X: torch.Tensor,               # (B,1)
    Y: torch.Tensor,               # (B,k)
    *,
    z_path: Optional[torch.Tensor],  # (B,H,m) or None
    rf_month: float,
    sim_cfg: DiscreteSimConfig,
    ppgdpo_cfg: PPGDPOConfig,
    tau0: Optional[float] = None,
    tau_step: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Monte Carlo + autograd costate estimation.

    Returns dict with keys: JX (B,1), JXX (B,1), JXY (B,k).
    """
    device = X.device
    dtype = X.dtype
    B = X.shape[0]
    k = Y.shape[1]

    # Freeze policy params: we only want derivatives wrt state variables
    params = list(policy.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    gen = None
    if ppgdpo_cfg.seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(ppgdpo_cfg.seed))

    mc = int(ppgdpo_cfg.mc_rollouts)
    sub = int(max(1, ppgdpo_cfg.sub_batch))

    sum_JX = torch.zeros(B, 1, device=device, dtype=dtype)
    sum_JXX = torch.zeros(B, 1, device=device, dtype=dtype)
    sum_JXY = torch.zeros(B, k, device=device, dtype=dtype)

    done = 0
    try:
        while done < mc:
            r = min(sub, mc - done)

            x0 = X.detach().clone().requires_grad_(True)  # (B,1)
            y0 = Y.detach().clone().requires_grad_(True)  # (B,k)

            X_rep = x0.repeat_interleave(r, dim=0)        # (B*r,1)
            Y_rep = y0.repeat_interleave(r, dim=0)        # (B*r,k)
            Z_rep = None
            if z_path is not None:
                Z_rep = z_path.repeat_interleave(r, dim=0)

            J = rollout_paths_discrete(
                model=model,
                policy=policy,
                X0=X_rep,
                Y0=Y_rep,
                Z_path=Z_rep,
                rf_month=rf_month,
                cfg=sim_cfg,
                generator=gen,
                tau0=tau0,
                tau_step=tau_step,
            )  # (B*r,)

            J_sum = J.view(B, r).sum(dim=1)  # (B,)
            total = J_sum.sum()

            # Need create_graph to take second derivatives
            JX, _JY = torch.autograd.grad(total, [x0, y0], create_graph=True, allow_unused=True)
            if JX is None:
                JX = torch.zeros_like(x0)
            sum_JX += JX.detach()

            # NOTE: We need both second derivatives from the SAME graph.
            # Calling autograd.grad twice would free the graph after the first call
            # unless retain_graph=True, so compute them together in one call.
            JXX, JXY = torch.autograd.grad(JX.sum(), [x0, y0], allow_unused=True)
            if JXX is None:
                JXX = torch.zeros_like(x0)
            if JXY is None:
                JXY = torch.zeros_like(y0)
            sum_JXX += JXX.detach()
            sum_JXY += JXY.detach()

            done += r

        out = {
            "JX": sum_JX / float(mc),
            "JXX": sum_JXX / float(mc),
            "JXY": sum_JXY / float(mc),
        }
        return out
    finally:
        for p, r in zip(params, req_bak):
            p.requires_grad_(r)



def _projection_blocks_for_gamma(
    model: TorchDiscreteLatentModel,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Sigma_eff, Cross_eff) used by the stagewise P-PGDPO micro-problem."""
    if abs(float(gamma) - 1.0) < 1e-8:
        # Closed-form gamma=1 branch: use raw fitted blocks for interpretability.
        return model.Sigma, model.Cross
    # Otherwise match the rollout sampling law (post shrink/jitter) used in MC costates.
    Sigma_eff = model.joint[: model.n, : model.n]
    Cross_eff = model.joint[: model.n, model.n :]
    return Sigma_eff, Cross_eff


def ppgdpo_action(
    model: TorchDiscreteLatentModel,
    policy: nn.Module,
    X: torch.Tensor,   # (B,1)
    Y: torch.Tensor,   # (B,k)
    *,
    z_path: Optional[torch.Tensor],
    rf_month: float,
    sim_cfg: DiscreteSimConfig,
    ppgdpo_cfg: PPGDPOConfig,
    tau0: Optional[float] = None,
    tau_step: Optional[float] = None,
    costates_override: Optional[Dict[str, torch.Tensor]] = None,
    cross_scale: float = 1.0,
    cross_override: Optional[torch.Tensor] = None,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """Compute a P-PGDPO projected risky-weight action u at a deployment state.

    Parameters
    ----------
    costates_override:
        If provided, skip MC estimation / closed-form recomputation and reuse the
        supplied costates. Useful for local ablations that keep policy/costates
        fixed and only change the projection-stage Cross term.
    cross_scale:
        Scalar multiplier applied to the projection-stage Cross block. Setting
        ``cross_scale=0`` gives a *local zero-hedge ablation* while keeping the
        estimated-world policy / costates unchanged.
    cross_override:
        Optional explicit Cross block to use in the projection-stage hedge term.
        If provided it is used after dtype/device conversion and then scaled by
        ``cross_scale``.
    return_debug:
        When True, include decomposition terms (mu term / hedge term / a-vector)
        that help diagnose whether Cross is actually moving the projected action.

    Returns dict:
      - u_pp: (B,n)
      - cash_pp: (B,1)
      - costates: dict
      - cross_scale: float
      - debug: optional dict with decomposition tensors
    """
    g = float(sim_cfg.gamma)

    if costates_override is not None:
        costates = dict(costates_override)
        JX = costates["JX"]
        JXX = costates["JXX"]
        JXY = costates["JXY"]
    else:
        # --------------------------------------------------------------
        # Log-utility special case (gamma=1)
        # --------------------------------------------------------------
        # For terminal-log utility with multiplicative wealth dynamics, the
        # value function is homothetic:
        #   V(X,y,t) = kappa * log(X) + v(y,t)
        # hence
        #   JX  = kappa / X,  JXX = -kappa / X^2,  JXY = 0,
        # so the intertemporal hedging term vanishes exactly.
        if abs(g - 1.0) < 1e-8:
            kappa = float(sim_cfg.kappa)
            JX = (kappa / X).detach()
            JXX = (-kappa / (X * X)).detach()
            JXY = torch.zeros_like(Y).detach()
            costates = {"JX": JX, "JXX": JXX, "JXY": JXY, "closed_form": True}
        else:
            with torch.enable_grad():
                costates = estimate_costates_discrete(
                    model=model,
                    policy=policy,
                    X=X,
                    Y=Y,
                    z_path=z_path,
                    rf_month=rf_month,
                    sim_cfg=sim_cfg,
                    ppgdpo_cfg=ppgdpo_cfg,
                    tau0=tau0,
                    tau_step=tau_step,
                )
            JX = costates["JX"]
            JXX = costates["JXX"]
            JXY = costates["JXY"]

    mu = model.mu_excess(Y)  # (B,n)
    Sigma_eff, Cross_eff = _projection_blocks_for_gamma(model, g)

    if cross_override is not None:
        Cross_proj = cross_override.to(device=mu.device, dtype=mu.dtype)
    else:
        Cross_proj = Cross_eff
    Cross_proj = float(cross_scale) * Cross_proj

    mu_term = X * (JX * mu)
    hedge = torch.zeros_like(mu)
    if (JXY is not None) and (model.k > 0):
        hedge = JXY @ Cross_proj.T
    hedge_term = X * hedge

    a_vec = mu_term + hedge_term
    negJXX = (-JXX).clamp_min(ppgdpo_cfg.clamp_neg_jxx_min)
    Q_b = (X * X * negJXX).view(-1, 1, 1) * Sigma_eff.unsqueeze(0)

    u_pp = project_u_long_only_cash_barrier(a_vec, Q_b, cfg=ppgdpo_cfg)
    cash_pp = 1.0 - u_pp.sum(dim=1, keepdim=True)

    out: Dict[str, Any] = {
        "u_pp": u_pp,
        "cash_pp": cash_pp,
        "costates": costates,
        "cross_scale": float(cross_scale),
    }
    if return_debug:
        out["debug"] = {
            "mu_term": mu_term,
            "hedge_term": hedge_term,
            "a_vec": a_vec,
            "negJXX": negJXX,
            "Sigma_eff": Sigma_eff,
            "Cross_proj": Cross_proj,
        }
    return out
