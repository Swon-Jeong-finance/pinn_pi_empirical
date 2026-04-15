"""
Policy networks for portfolio weights and (optional) consumption.

We keep them small by default for speed.

Notes:
- In Merton-style continuous-time, π is *fraction of wealth* in each risky asset.
  Sum(π) need not be 1 (leverage/borrowing allowed).
- For empirical safety, we provide a leverage cap via tanh.
- For long-only, we provide a softmax transform.

You can switch transforms depending on your experimental design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .constraints import PortfolioConstraints, merged_per_asset_cap, project_box_sum_torch


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.Tanh())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PolicyConfig:
    n_assets: int
    n_states: int
    hidden: int = 64
    depth: int = 2
    weight_transform: Literal["none", "tanh_leverage", "long_only", "long_only_cash", "gross_leverage"] = "tanh_leverage"
    max_leverage: float = 2.0          # per-asset cap under tanh_leverage
    max_gross_leverage: float = 1.5    # cap on sum(|pi|) under gross_leverage
    softmax_temperature: float = 1.0  # for long_only / long_only_cash
    softmax_epsilon: float = 0.0      # mix with uniform to reduce extreme weights
    include_time: bool = True          # include time-to-maturity tau in input
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)


class PortfolioPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.cfg.constraints.validate(self.cfg.n_assets)
        in_dim = cfg.n_states + (1 if cfg.include_time else 0) + 1  # +1 for log-wealth or wealth scalar
        out_dim = cfg.n_assets + 1 if cfg.weight_transform == "long_only_cash" else cfg.n_assets
        self.mlp = MLP(in_dim=in_dim, out_dim=out_dim, hidden=cfg.hidden, depth=cfg.depth)

    def forward(self, tau: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        tau: (batch,1) in [0,1], time-to-maturity fraction
        x:   (batch,1) wealth
        y:   (batch,k) states
        returns:
          pi: (batch,n) risky fractions
        """
        pi, _ = self.weights_and_cash(tau, x, y)
        return pi

    def weights_and_cash(self, tau: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          pi:   (batch,n) risky fractions
          cash: (batch,1) cash weight if weight_transform == "long_only_cash", else None

        Note: If cash weight is provided, the portfolio return can be computed as:
          R_p = rf + sum_i pi_i (r_i - rf)
        because cash weight is (1 - sum_i pi_i) by construction.
        """
        # Use log-wealth as a more scale-stable input
        x_in = torch.log(torch.clamp(x, min=1e-12))
        if self.cfg.include_time:
            inp = torch.cat([tau, x_in, y], dim=1)
        else:
            inp = torch.cat([x_in, y], dim=1)
        raw = self.mlp(inp)

        if self.cfg.weight_transform == "long_only_cash":
            constraints = self.cfg.constraints

            temp = float(self.cfg.softmax_temperature) if self.cfg.softmax_temperature is not None else 1.0
            temp = max(temp, 1e-6)

            if not bool(constraints.allow_short):
                # Parameterize risky weights on a simplex budget of size L so that
                # cash = 1 - sum(pi) may become negative when L>1 (borrowing allowed).
                z = torch.softmax(raw / temp, dim=1)  # (batch, n+1)
                eps = float(self.cfg.softmax_epsilon) if self.cfg.softmax_epsilon is not None else 0.0
                if eps > 0.0:
                    z = (1.0 - eps) * z + eps * (1.0 / z.shape[1])

                L_cap = float(constraints.effective_risky_cap())
                pi = L_cap * z[:, : self.cfg.n_assets]
                asset_cap = constraints.upper_bound()
                if asset_cap is not None:
                    pi = project_box_sum_torch(
                        pi,
                        lower=0.0,
                        upper=float(asset_cap),
                        cap=L_cap,
                    )
                cash = 1.0 - pi.sum(dim=1, keepdim=True)
                return pi, cash

            # Short-allowing fallback: map to a box, then project onto the box+sum set.
            lower = float(constraints.lower_bound())
            upper = constraints.upper_bound()
            if upper is None:
                upper = float(constraints.effective_risky_cap())
            center = 0.5 * (float(upper) + lower)
            half = 0.5 * (float(upper) - lower)
            pi_raw = center + half * torch.tanh(raw[:, : self.cfg.n_assets] / temp)
            pi = project_box_sum_torch(
                pi_raw,
                lower=lower,
                upper=float(upper),
                cap=float(constraints.effective_risky_cap()),
            )
            cash = 1.0 - pi.sum(dim=1, keepdim=True)
            return pi, cash

        pi = self._transform(raw)
        return pi, None

    def _transform(self, raw: torch.Tensor) -> torch.Tensor:
        if self.cfg.weight_transform == "none":
            return raw
        if self.cfg.weight_transform == "tanh_leverage":
            return self.cfg.max_leverage * torch.tanh(raw)
        if self.cfg.weight_transform == "long_only":
            # risky weights sum to 1 (no cash). If you want cash too, use long_only_cash.
            temp = float(self.cfg.softmax_temperature) if self.cfg.softmax_temperature is not None else 1.0
            temp = max(temp, 1e-6)
            w = torch.softmax(raw / temp, dim=1)
            eps = float(self.cfg.softmax_epsilon) if self.cfg.softmax_epsilon is not None else 0.0
            if eps > 0.0:
                w = (1.0 - eps) * w + eps * (1.0 / w.shape[1])
            return w
        if self.cfg.weight_transform == "gross_leverage":
            # Allow long/short but cap gross exposure sum(|pi|) <= max_gross_leverage.
            pi = torch.tanh(raw)  # in [-1,1]
            gross = torch.sum(torch.abs(pi), dim=1, keepdim=True)
            scale = torch.minimum(torch.ones_like(gross), self.cfg.max_gross_leverage / (gross + 1e-12))
            return pi * scale
        raise ValueError(f"Unknown transform: {self.cfg.weight_transform}")


# -----------------------------------------------------------------------------
# Residual policy: learn "delta" on top of a constrained myopic baseline
# -----------------------------------------------------------------------------

@dataclass
class ResidualPolicyConfig:
    """Configuration for residual long-only+cash policy.

    The learned network outputs *residual logits* (delta) on top of a
    model-implied constrained myopic baseline:

      u0(y) = argmax_{u>=0, 1'u<=1}  mu(y)'u - (gamma/2) u' Sigma u

    and the final portfolio (including cash) is:

      w = softmax( log(w0) + delta / T )

    so that delta=0 implies w=w0 exactly.
    """

    n_assets: int
    n_states: int

    hidden: int = 64
    depth: int = 2
    include_time: bool = True

    # Residual-to-weights mapping
    softmax_temperature: float = 2.0
    softmax_epsilon: float = 0.01

    # Baseline (myopic) parameters
    gamma: float = 5.0
    ridge: float = 1e-6
    w_max: float | None = None

    # Numerical guards
    eps_base: float = 1e-12

    # Constraint set for risky weights / residual cash.
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)

    # If True, treat baseline weights as constants w.r.t. (x,y) for autograd.
    # (Keeps policy gradients intact; mainly affects costate estimation for P-PGDPO.)
    detach_baseline: bool = False


class ResidualLongOnlyCashPolicy(nn.Module):
    """Residual long-only+cash policy on top of a constrained-myopic baseline.

    Forward returns *risky weights* pi (sum<=1). Use weights_and_cash() to also
    get the cash weight.

    Notes
    -----
    * Baseline is computed from (a,B,Sigma) and current y via a fast
      unconstrained solve followed by an exact simplex projection when the
      common long-only / no per-asset-cap constraint set is active.
    * The residual is applied in log-simplex space so feasibility is automatic.
    """

    def __init__(
        self,
        cfg: ResidualPolicyConfig,
        *,
        a: torch.Tensor,     # (n,)
        B: torch.Tensor,     # (n,k)
        Sigma: torch.Tensor, # (n,n)
    ):
        super().__init__()
        self.cfg = cfg
        self.cfg.constraints.validate(self.cfg.n_assets)

        # Register model moments as buffers (non-trainable, device-aware).
        self.register_buffer('a', a.detach().clone())
        self.register_buffer('B', B.detach().clone())

        n = int(cfg.n_assets)
        k = int(cfg.n_states)
        if self.a.numel() != n:
            raise ValueError(f"Residual policy: a has shape {tuple(self.a.shape)} but n_assets={n}")
        if self.B.shape != (n, k):
            raise ValueError(f"Residual policy: B has shape {tuple(self.B.shape)} but expected (n,k)=({n},{k})")

        # Precompute an inverse for the baseline solve: (Sigma + ridge I)^{-1}
        S = Sigma.detach().clone()
        if S.shape != (n, n):
            raise ValueError(f"Residual policy: Sigma has shape {tuple(S.shape)} but expected ({n},{n})")
        self.register_buffer('Sigma_inv', self._compute_sigma_inv(S))

        in_dim = k + 1  # log-wealth
        if cfg.include_time:
            in_dim += 1

        # Residual logits include cash as the (n+1)-th component.
        self.mlp = MLP(in_dim=in_dim, out_dim=n + 1, hidden=cfg.hidden, depth=cfg.depth)

    def _compute_sigma_inv(self, Sigma: torch.Tensor) -> torch.Tensor:
        n = int(self.cfg.n_assets)
        if Sigma.shape != (n, n):
            raise ValueError(f"Residual policy: Sigma has shape {tuple(Sigma.shape)} but expected ({n},{n})")
        ridge = float(self.cfg.ridge)
        if ridge < 0:
            raise ValueError("residual_ridge must be >=0")
        S = Sigma.detach().clone()
        eye = torch.eye(n, device=S.device, dtype=S.dtype)
        S_reg = S + ridge * eye

        Sigma_inv = None
        jitter = 0.0
        for _ in range(6):
            try:
                if jitter > 0:
                    Sinv = torch.linalg.inv(S_reg + jitter * eye)
                else:
                    Sinv = torch.linalg.inv(S_reg)
                if torch.isfinite(Sinv).all():
                    Sigma_inv = Sinv
                    break
            except Exception:
                pass
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0

        if Sigma_inv is None:
            evals, evecs = torch.linalg.eigh(S_reg)
            floor = torch.clamp(evals, min=1e-10)
            Sigma_inv = (evecs * (1.0 / floor).unsqueeze(0)) @ evecs.T
        return Sigma_inv

    def refresh_baseline(self, *, a: torch.Tensor, B: torch.Tensor, Sigma: torch.Tensor) -> None:
        n = int(self.cfg.n_assets)
        k = int(self.cfg.n_states)
        if a.numel() != n:
            raise ValueError(f"Residual policy refresh: a has shape {tuple(a.shape)} but n_assets={n}")
        if B.shape != (n, k):
            raise ValueError(f"Residual policy refresh: B has shape {tuple(B.shape)} but expected ({n},{k})")
        if Sigma.shape != (n, n):
            raise ValueError(f"Residual policy refresh: Sigma has shape {tuple(Sigma.shape)} but expected ({n},{n})")
        with torch.no_grad():
            self.a.copy_(a.detach().to(device=self.a.device, dtype=self.a.dtype))
            self.B.copy_(B.detach().to(device=self.B.device, dtype=self.B.dtype))
            Sigma_inv = self._compute_sigma_inv(Sigma.detach().to(device=self.Sigma_inv.device, dtype=self.Sigma_inv.dtype))
            self.Sigma_inv.copy_(Sigma_inv)

    # -------------------------
    # Baseline: constrained-myopic
    # -------------------------
    def _baseline_weights_and_cash(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute baseline (risky, cash) weights for each row of y.

        Uses a fast heuristic:
          w_unc = (Sigma + ridge I)^{-1} mu(y) / gamma
          clip to long-only, optional cap, then renormalize if sum>1.

        Returns
        -------
        (pi0, cash0)
          pi0: (B,n)
          cash0: (B,1)
        """
        mu = self.a.view(1, -1) + y @ self.B.T  # (B,n)

        ra = max(float(self.cfg.gamma), 1e-8)
        w_unc = (mu @ self.Sigma_inv) / ra

        constraints = self.cfg.constraints
        asset_cap = merged_per_asset_cap(constraints, self.cfg.w_max)
        w = project_box_sum_torch(
            w_unc,
            lower=float(constraints.lower_bound()),
            upper=asset_cap,
            cap=float(constraints.effective_risky_cap()),
        )
        cash = 1.0 - w.sum(dim=1, keepdim=True)

        if bool(self.cfg.detach_baseline):
            w = w.detach()
            cash = cash.detach()

        return w, cash

    # -------------------------
    # Residual policy interface
    # -------------------------
    def weights_and_cash(self, tau: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # baseline
        pi0, cash0 = self._baseline_weights_and_cash(y)
        constraints = self.cfg.constraints
        L_cap = float(constraints.effective_risky_cap())

        # residual logits / perturbations
        x_in = torch.log(torch.clamp(x, min=1e-12))
        if self.cfg.include_time:
            inp = torch.cat([tau, x_in, y], dim=1)
        else:
            inp = torch.cat([x_in, y], dim=1)
        delta = self.mlp(inp)

        if not bool(constraints.allow_short):
            slack0 = (L_cap - pi0.sum(dim=1, keepdim=True)).clamp_min(float(self.cfg.eps_base))
            z0_full = torch.cat([pi0, slack0], dim=1) / max(L_cap, float(self.cfg.eps_base))
            z0_full = torch.clamp(z0_full, min=float(self.cfg.eps_base))
            base_logits = torch.log(z0_full)

            T = max(float(self.cfg.softmax_temperature), 1e-6)
            logits = base_logits + (delta / T)
            z_full = torch.softmax(logits, dim=1)

            eps = float(self.cfg.softmax_epsilon)
            if eps > 0.0:
                z_full = (1.0 - eps) * z_full + eps * (1.0 / z_full.shape[1])

            pi = L_cap * z_full[:, : self.cfg.n_assets]
            asset_cap = merged_per_asset_cap(constraints, self.cfg.w_max)
            if asset_cap is not None:
                pi = project_box_sum_torch(pi, lower=0.0, upper=asset_cap, cap=L_cap)
            cash = 1.0 - pi.sum(dim=1, keepdim=True)
            return pi, cash

        # Short-allowing fallback: additive residual in risky space + projection.
        lower = float(constraints.lower_bound())
        asset_cap = merged_per_asset_cap(constraints, self.cfg.w_max)
        upper = asset_cap if asset_cap is not None else float(L_cap)
        step_scale = max(abs(lower), abs(float(upper)), 1.0)
        pi_raw = pi0 + 0.25 * step_scale * torch.tanh(delta[:, : self.cfg.n_assets])
        pi = project_box_sum_torch(pi_raw, lower=lower, upper=float(upper), cap=L_cap)
        cash = 1.0 - pi.sum(dim=1, keepdim=True)
        return pi, cash

    def forward(self, tau: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pi, _cash = self.weights_and_cash(tau, x, y)
        return pi

    def weights_and_cash_with_base(
        self, tau: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: return (pi, cash, pi0, cash0)."""
        pi0, cash0 = self._baseline_weights_and_cash(y)
        pi, cash = self.weights_and_cash(tau, x, y)
        return pi, cash, pi0, cash0


@dataclass
class ConsumptionConfig:
    hidden: int = 32
    depth: int = 2
    max_consume_frac_per_year: float = 0.5   # C_t <= frac * X_t (per year)
    include_time: bool = True


class ConsumptionPolicy(nn.Module):
    """
    Output is a *consumption rate* C_t (per year), so that in Euler:
      dX = ... - C_t dt
    """
    def __init__(self, n_states: int, cfg: ConsumptionConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = n_states + (1 if cfg.include_time else 0) + 1
        self.mlp = MLP(in_dim=in_dim, out_dim=1, hidden=cfg.hidden, depth=cfg.depth)
        self.softplus = nn.Softplus()

    def forward(self, tau: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_in = torch.log(torch.clamp(x, min=1e-12))
        if self.cfg.include_time:
            inp = torch.cat([tau, x_in, y], dim=1)
        else:
            inp = torch.cat([x_in, y], dim=1)
        raw = self.mlp(inp)
        frac = self.softplus(raw)
        frac = torch.clamp(frac, max=self.cfg.max_consume_frac_per_year)
        # convert to rate: C = frac * X
        return frac * x
