from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class PortfolioConstraints:
    """Simple portfolio-constraint family used across policy / myopic / P-PGDPO.

    The current implementation is designed around a risky-weight vector ``pi`` and
    residual cash ``w_cash = 1 - sum(pi)``.

    Constraints
    -----------
    - componentwise lower bound:  pi_i >= short_floor  (or 0 if shorting disabled)
    - optional componentwise upper bound: pi_i <= per_asset_cap
    - total risky cap:           sum(pi) <= risky_cap
    - optional cash floor:       w_cash >= cash_floor, equivalent to sum(pi) <= 1-cash_floor

    The effective total-risky cap is therefore
        L_eff = min(risky_cap, 1 - cash_floor).

    Notes
    -----
    * ``allow_short=False`` forces the lower bound to 0 regardless of ``short_floor``.
    * ``per_asset_cap=None`` means no explicit componentwise upper bound beyond what is
      already implied by the total risky cap.
    * This module does **not** enforce a cash ceiling; cash is always the residual.
    """

    risky_cap: float = 1.0
    cash_floor: float = 0.0
    allow_short: bool = False
    short_floor: float = 0.0
    per_asset_cap: Optional[float] = None

    def lower_bound(self) -> float:
        return float(self.short_floor) if bool(self.allow_short) else 0.0

    def upper_bound(self) -> Optional[float]:
        return None if self.per_asset_cap is None else float(self.per_asset_cap)

    def risky_cap_from_cash_floor(self) -> float:
        return 1.0 - float(self.cash_floor)

    def effective_risky_cap(self) -> float:
        return min(float(self.risky_cap), self.risky_cap_from_cash_floor())

    def validate(self, n_assets: Optional[int] = None) -> None:
        L = float(self.effective_risky_cap())
        lo = float(self.lower_bound())
        hi = self.upper_bound()

        if not np.isfinite(L):
            raise ValueError(f"effective risky cap must be finite, got {L}")
        if L <= 0.0:
            raise ValueError(f"effective risky cap must be >0, got {L}")
        if hi is not None and hi < lo - 1e-12:
            raise ValueError(f"per_asset_cap ({hi}) must be >= lower bound ({lo})")
        if hi is not None and hi <= 0.0 and not bool(self.allow_short):
            raise ValueError(f"per_asset_cap must be >0 for long-only constraints, got {hi}")
        if n_assets is not None:
            if n_assets <= 0:
                raise ValueError(f"n_assets must be positive, got {n_assets}")
            if n_assets * lo > L + 1e-12:
                raise ValueError(
                    f"Infeasible constraint set: n_assets*lower = {n_assets*lo:.6f} exceeds effective risky cap {L:.6f}"
                )
            if hi is not None and n_assets * hi < min(L, max(0.0, L)) - 1e-12:
                # Not strictly infeasible because sum(pi) may be < L, but warn-worthy.
                pass

    def summary(self) -> str:
        hi = self.upper_bound()
        return (
            f"allow_short={bool(self.allow_short)} "
            f"short_floor={self.lower_bound():.4f} "
            f"per_asset_cap={('None' if hi is None else f'{hi:.4f}')} "
            f"risky_cap={float(self.risky_cap):.4f} "
            f"cash_floor={float(self.cash_floor):.4f} "
            f"effective_risky_cap={self.effective_risky_cap():.4f}"
        )


def _as_numpy_bounds(
    n: int,
    *,
    lower: float,
    upper: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(n, float(lower), dtype=float)
    if upper is None:
        hi = np.full(n, np.inf, dtype=float)
    else:
        hi = np.full(n, float(upper), dtype=float)
    return lo, hi


def _as_torch_bounds(
    n: int,
    *,
    lower: float,
    upper: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.full((1, n), float(lower), device=device, dtype=dtype)
    if upper is None:
        hi = torch.full((1, n), float('inf'), device=device, dtype=dtype)
    else:
        hi = torch.full((1, n), float(upper), device=device, dtype=dtype)
    return lo, hi


def can_use_simplex_fast_path(*, lower: float, upper: Optional[float], atol: float = 1e-15) -> bool:
    """Fast path for the common long-only + sum-cap set {x >= 0, sum(x) <= cap}."""
    return (upper is None) and (abs(float(lower)) <= float(atol))


def project_simplex_sum_numpy(v: np.ndarray, *, cap: float) -> np.ndarray:
    """Exact Euclidean projection onto {x >= 0, sum(x) <= cap}."""
    x = np.asarray(v, dtype=float).reshape(-1)
    if float(cap) <= 0.0:
        return np.zeros_like(x)
    n = x.shape[0]
    if n == 0:
        return x.copy()

    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - float(cap)
    ind = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / ind > 0.0
    rho = int(np.count_nonzero(cond))
    rho = max(rho, 1)
    theta = float(cssv[rho - 1] / rho)
    theta = max(theta, 0.0)
    return np.maximum(x - theta, 0.0)


@torch.no_grad()
def project_simplex_sum_torch(v: torch.Tensor, *, cap: float) -> torch.Tensor:
    """Batch exact Euclidean projection onto {x >= 0, sum(x) <= cap}."""
    if v.ndim != 2:
        raise ValueError(f"project_simplex_sum_torch expects 2D tensor, got shape {tuple(v.shape)}")
    if float(cap) <= 0.0:
        return torch.zeros_like(v)

    B, n = v.shape
    if n == 0:
        return v.clone()

    u, _ = torch.sort(v, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - float(cap)
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype).view(1, n)
    cond = u - cssv / ind > 0.0
    rho = cond.sum(dim=1, keepdim=True).clamp_min(1)
    theta = cssv.gather(1, (rho - 1).long()) / rho.to(dtype=v.dtype)
    theta = theta.clamp_min(0.0)
    return (v - theta).clamp_min(0.0)


def project_box_sum_numpy(
    v: np.ndarray,
    *,
    lower: float,
    upper: Optional[float],
    cap: float,
    max_iter: int = 80,
    tol: float = 1e-12,
) -> np.ndarray:
    """Euclidean projection onto {x: lower<=x<=upper, sum(x)<=cap}."""
    x = np.asarray(v, dtype=float).reshape(-1)
    if can_use_simplex_fast_path(lower=lower, upper=upper, atol=tol):
        return project_simplex_sum_numpy(x, cap=float(cap))

    n = x.shape[0]
    lo, hi = _as_numpy_bounds(n, lower=lower, upper=upper)

    clipped = np.minimum(np.maximum(x, lo), hi)
    if float(np.sum(clipped)) <= float(cap) + tol:
        return clipped

    if n * float(lower) > float(cap) + tol:
        raise ValueError(
            f"Infeasible projection set: n*lower={n*float(lower):.6f} exceeds cap={float(cap):.6f}"
        )

    def f(lmbd: float) -> float:
        y = np.minimum(np.maximum(x - lmbd, lo), hi)
        return float(np.sum(y) - cap)

    lam_lo = 0.0
    val_lo = f(lam_lo)
    if val_lo <= 0.0:
        return np.minimum(np.maximum(x, lo), hi)

    lam_hi = 1.0
    val_hi = f(lam_hi)
    it = 0
    while val_hi > 0.0 and it < 120:
        lam_hi *= 2.0
        val_hi = f(lam_hi)
        it += 1

    for _ in range(int(max_iter)):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        val_mid = f(lam_mid)
        if abs(val_mid) <= tol:
            lam_lo = lam_hi = lam_mid
            break
        if val_mid > 0.0:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    lmbd = 0.5 * (lam_lo + lam_hi)
    y = np.minimum(np.maximum(x - lmbd, lo), hi)
    return y


@torch.no_grad()
def project_box_sum_torch(
    v: torch.Tensor,
    *,
    lower: float,
    upper: Optional[float],
    cap: float,
    max_iter: int = 80,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Batch Euclidean projection onto {x: lower<=x<=upper, sum(x)<=cap}."""
    if v.ndim != 2:
        raise ValueError(f"project_box_sum_torch expects 2D tensor, got shape {tuple(v.shape)}")

    if can_use_simplex_fast_path(lower=lower, upper=upper, atol=tol):
        return project_simplex_sum_torch(v, cap=float(cap))

    B, n = v.shape
    device, dtype = v.device, v.dtype
    lo, hi = _as_torch_bounds(n, lower=lower, upper=upper, device=device, dtype=dtype)
    lo = lo.expand(B, n)
    hi = hi.expand(B, n)

    clipped = torch.minimum(torch.maximum(v, lo), hi)
    s = clipped.sum(dim=1, keepdim=True)
    if bool((s <= float(cap) + tol).all()):
        return clipped

    if n * float(lower) > float(cap) + tol:
        raise ValueError(
            f"Infeasible projection set: n*lower={n*float(lower):.6f} exceeds cap={float(cap):.6f}"
        )

    def f(lmbd: torch.Tensor) -> torch.Tensor:
        y = torch.minimum(torch.maximum(v - lmbd, lo), hi)
        return y.sum(dim=1, keepdim=True) - float(cap)

    lam_lo = torch.zeros(B, 1, device=device, dtype=dtype)
    val_lo = f(lam_lo)
    if bool((val_lo <= 0.0).all()):
        return clipped

    lam_hi = torch.ones(B, 1, device=device, dtype=dtype)
    val_hi = f(lam_hi)
    it = 0
    while bool((val_hi > 0.0).any()) and it < 120:
        lam_hi = torch.where(val_hi > 0.0, lam_hi * 2.0, lam_hi)
        val_hi = f(lam_hi)
        it += 1

    for _ in range(int(max_iter)):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        val_mid = f(lam_mid)
        done = val_mid.abs() <= tol
        lam_lo = torch.where(done, lam_mid, torch.where(val_mid > 0.0, lam_mid, lam_lo))
        lam_hi = torch.where(done, lam_mid, torch.where(val_mid > 0.0, lam_hi, lam_mid))

    lmbd = 0.5 * (lam_lo + lam_hi)
    y = torch.minimum(torch.maximum(v - lmbd, lo), hi)
    return y


def merged_per_asset_cap(constraints: PortfolioConstraints, extra_cap: Optional[float]) -> Optional[float]:
    caps = []
    base = constraints.upper_bound()
    if base is not None:
        caps.append(float(base))
    if extra_cap is not None:
        caps.append(float(extra_cap))
    if not caps:
        return None
    return float(min(caps))
