from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .mean_model import MeanModelResult
from .transition import CrossCovarianceEstimate, StateTransitionResult
from .ppgdpo import CostateEstimate, solve_ppgdpo_projection


def _symmetrize_psd(mat: np.ndarray, *, floor: float = 1.0e-10) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    arr = 0.5 * (arr + arr.T)
    eigval, eigvec = np.linalg.eigh(arr)
    eigval = np.clip(eigval, floor, None)
    return eigvec @ np.diag(eigval) @ eigvec.T


def _safe_box_quantiles(states_t: pd.DataFrame, *, q_low: float, q_high: float, buffer: float) -> tuple[np.ndarray, np.ndarray]:
    arr = states_t.to_numpy(dtype=float)
    lo = np.nanquantile(arr, q_low, axis=0)
    hi = np.nanquantile(arr, q_high, axis=0)
    width = hi - lo
    for j in range(len(width)):
        if (not np.isfinite(width[j])) or width[j] <= 1.0e-8:
            col = arr[:, j]
            finite = col[np.isfinite(col)]
            if len(finite) >= 2:
                lo[j] = float(np.min(finite))
                hi[j] = float(np.max(finite))
                width[j] = hi[j] - lo[j]
            else:
                lo[j], hi[j], width[j] = -3.0, 3.0, 6.0
    return (lo - buffer * width).astype(float), (hi + buffer * width).astype(float)


def _proj_simplex_eq(v: torch.Tensor, s: float) -> torch.Tensor:
    s = float(s)
    if s <= 0.0:
        return torch.zeros_like(v)
    u, _ = torch.sort(v, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - s
    ind = torch.arange(1, u.size(1) + 1, device=v.device, dtype=v.dtype).view(1, -1)
    cond = u - cssv / ind > 0
    rho = torch.clamp(cond.sum(dim=1) - 1, min=0)
    theta = cssv.gather(1, rho.view(-1, 1)) / (rho.to(v.dtype).view(-1, 1) + 1.0)
    return torch.clamp(v - theta, min=0.0)


def _proj_nonneg_l1_ball(v: torch.Tensor, cap: float) -> torch.Tensor:
    cap = float(cap)
    if cap <= 0.0:
        return torch.zeros_like(v)
    out = torch.clamp(v, min=0.0)
    sums = out.sum(dim=1, keepdim=True)
    mask = (sums > cap + 1.0e-12).squeeze(1)
    if mask.any():
        out = out.clone()
        out[mask] = _proj_simplex_eq(out[mask], cap)
    return out


@torch.no_grad()
def _solve_qp_long_only_budget_full(
    Sigma: torch.Tensor,
    v: torch.Tensor,
    gamma: float,
    cap: float,
    *,
    iters: int = 300,
    tol: float = 1.0e-10,
    step_scale: float = 1.1,
) -> torch.Tensor:
    gamma = float(gamma)
    cap = float(cap)
    if cap <= 0.0:
        return torch.zeros_like(v)
    if v.ndim == 1:
        v2 = v.view(1, -1)
    else:
        v2 = v
    n_assets = int(v2.shape[1])
    Sigma = Sigma.reshape(n_assets, n_assets)
    Sigma = 0.5 * (Sigma + Sigma.T)
    eig = torch.linalg.eigvalsh(Sigma)
    lam_max = torch.clamp(eig.max(), min=1.0e-12)
    alpha = 1.0 / (float(step_scale) * gamma * lam_max)
    rhs = v2.t().contiguous()
    try:
        sol = torch.linalg.solve(Sigma, rhs).t()
    except RuntimeError:
        sol = torch.linalg.lstsq(Sigma, rhs).solution.t()
    pi = _proj_nonneg_l1_ball((1.0 / gamma) * sol, cap)
    for _ in range(max(int(iters), 1)):
        grad = gamma * torch.matmul(pi, Sigma.T) - v2
        pi_new = _proj_nonneg_l1_ball(pi - alpha * grad, cap)
        diff = torch.max(torch.abs(pi_new - pi)).item()
        pi = pi_new
        if diff < float(tol):
            break
    return pi if v.ndim == 2 else pi.squeeze(0)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.Tanh())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(
        self,
        *,
        tau_max: float,
        x_min: np.ndarray,
        x_max: np.ndarray,
        width: int,
        depth: int,
    ):
        super().__init__()
        x_min = np.asarray(x_min, dtype=float).reshape(1, -1)
        x_max = np.asarray(x_max, dtype=float).reshape(1, -1)
        self.register_buffer('x_min', torch.tensor(x_min, dtype=torch.float64))
        self.register_buffer('x_max', torch.tensor(x_max, dtype=torch.float64))
        self.tau_max = float(max(tau_max, 1.0e-8))
        self.mlp = _MLP(in_dim=1 + x_min.shape[1], out_dim=1, hidden_dim=int(width), hidden_layers=int(depth))

    def forward(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # tau = torch.log1p(tau)  # log(1 + τ), τ=240이면 약 5.48
        inp = torch.cat([tau, x], dim=1)
        return self.mlp(inp)


@dataclass(frozen=True)
class _BatchPolicyCoefficients:
    tau: torch.Tensor
    x: torch.Tensor
    A: torch.Tensor
    Bcoef: torch.Tensor


@dataclass(frozen=True)
class _MeanAffineMap:
    kind: Literal['direct_assets', 'factor_apt', 'factor_apt_regime']
    state_columns: list[str]
    assets: list[str]
    intercept: np.ndarray
    slope: np.ndarray
    intercept_low: np.ndarray | None = None
    slope_low: np.ndarray | None = None
    intercept_high: np.ndarray | None = None
    slope_high: np.ndarray | None = None
    regime_weight_train: float = 0.5

    @classmethod
    def from_mean_model(cls, mean_model: MeanModelResult) -> '_MeanAffineMap':
        coef = np.asarray(mean_model.coef, dtype=float)
        assets = list(mean_model.assets)
        state_columns = list(mean_model.columns)
        if mean_model.kind == 'direct_assets':
            return cls(
                kind='direct_assets',
                state_columns=state_columns,
                assets=assets,
                intercept=coef[0, :].astype(float),
                slope=coef[1:, :].T.astype(float),
            )
        if mean_model.asset_alpha is None or mean_model.loadings is None or mean_model.factor_columns is None:
            raise ValueError('factor mean model is missing loadings/alpha/factor_columns')
        loadings = mean_model.loadings.reindex(index=assets, columns=mean_model.factor_columns).to_numpy(dtype=float)
        alpha = mean_model.asset_alpha.reindex(assets).to_numpy(dtype=float)
        if mean_model.kind == 'factor_apt':
            intercept = alpha + loadings @ coef[0, :]
            slope = loadings @ coef[1:, :].T
            return cls(
                kind='factor_apt',
                state_columns=state_columns,
                assets=assets,
                intercept=intercept.astype(float),
                slope=slope.astype(float),
            )
        if mean_model.regime_beta_low is None or mean_model.regime_beta_high is None:
            intercept = alpha + loadings @ coef[0, :]
            slope = loadings @ coef[1:, :].T
            return cls(
                kind='factor_apt_regime',
                state_columns=state_columns,
                assets=assets,
                intercept=intercept.astype(float),
                slope=slope.astype(float),
                intercept_low=intercept.astype(float),
                slope_low=slope.astype(float),
                intercept_high=intercept.astype(float),
                slope_high=slope.astype(float),
                regime_weight_train=float(mean_model.regime_high_fraction or 0.5),
            )
        beta_low = np.asarray(mean_model.regime_beta_low, dtype=float)
        beta_high = np.asarray(mean_model.regime_beta_high, dtype=float)
        intercept_low = alpha + loadings @ beta_low[0, :]
        slope_low = loadings @ beta_low[1:, :].T
        intercept_high = alpha + loadings @ beta_high[0, :]
        slope_high = loadings @ beta_high[1:, :].T
        intercept = (1.0 - float(mean_model.regime_high_fraction or 0.5)) * intercept_low + float(mean_model.regime_high_fraction or 0.5) * intercept_high
        slope = (1.0 - float(mean_model.regime_high_fraction or 0.5)) * slope_low + float(mean_model.regime_high_fraction or 0.5) * slope_high
        return cls(
            kind='factor_apt_regime',
            state_columns=state_columns,
            assets=assets,
            intercept=intercept.astype(float),
            slope=slope.astype(float),
            intercept_low=intercept_low.astype(float),
            slope_low=slope_low.astype(float),
            intercept_high=intercept_high.astype(float),
            slope_high=slope_high.astype(float),
            regime_weight_train=float(mean_model.regime_high_fraction or 0.5),
        )

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.kind != 'factor_apt_regime' or self.intercept_low is None or self.slope_low is None or self.intercept_high is None or self.slope_high is None:
            return self.intercept[None, :] + x @ self.slope.T
        w = float(np.clip(self.regime_weight_train, 0.0, 1.0))
        mu_low = self.intercept_low[None, :] + x @ self.slope_low.T
        mu_high = self.intercept_high[None, :] + x @ self.slope_high.T
        return (1.0 - w) * mu_low + w * mu_high


class PIPINNEnvFromPPGDPO:
    def __init__(
        self,
        *,
        mean_model: MeanModelResult,
        transition: StateTransitionResult,
        cross_est: CrossCovarianceEstimate,
        states_t: pd.DataFrame,
        sigma_train: np.ndarray,
        cfg: Any,
        tau_max: float,
        device: str,
        dtype: torch.dtype,
    ):
        if states_t.shape[1] <= 0:
            raise ValueError('PI-PINN backend requires at least one state column')
        self.device = device
        self.dtype = dtype
        self.gamma = float(cfg.policy.risk_aversion)
        self.risky_cap = float(min(float(cfg.policy.risky_cap), 1.0 - float(getattr(cfg.policy, 'cash_floor', 0.0) or 0.0)))
        self.tau_max = float(max(tau_max, 1.0e-8))
        self.state_columns = list(states_t.columns)
        self.asset_columns = list(mean_model.assets)
        self.n_states = len(self.state_columns)
        self.n_assets = len(self.asset_columns)
        self.mean_map = _MeanAffineMap.from_mean_model(mean_model)
        coef = np.asarray(transition.coef, dtype=float)
        self.transition_intercept = coef[0, :].astype(float)
        self.transition_matrix = coef[1:, :].T.astype(float)
        self.drift_intercept = self.transition_intercept.copy()
        self.drift_matrix = self.transition_matrix - np.eye(self.n_states, dtype=float)
        self.Q = _symmetrize_psd(np.asarray(cross_est.state_innov_cov, dtype=float), floor=1.0e-10)
        cross_df = cross_est.cross.reindex(index=self.asset_columns, columns=self.state_columns)
        self.C_train = cross_df.to_numpy(dtype=float)
        self.Sigma_train = _symmetrize_psd(np.asarray(sigma_train, dtype=float), floor=1.0e-10)
        self.Sigma_train_t = torch.tensor(self.Sigma_train, device=device, dtype=dtype)
        self.Q_t = torch.tensor(self.Q, device=device, dtype=dtype)
        self.C_train_t = torch.tensor(self.C_train, device=device, dtype=dtype)
        x_min, x_max = _safe_box_quantiles(
            states_t,
            q_low=float(cfg.pipinn.x_domain_quantile_low),
            q_high=float(cfg.pipinn.x_domain_quantile_high),
            buffer=float(cfg.pipinn.x_domain_buffer),
        )
        self.x_min = x_min
        self.x_max = x_max
        self.x_min_t = torch.tensor(x_min.reshape(1, -1), device=device, dtype=dtype)
        self.x_max_t = torch.tensor(x_max.reshape(1, -1), device=device, dtype=dtype)
        self.x_empirical = states_t.to_numpy(dtype=float)
        self.state_mean = torch.tensor(states_t.mean(axis=0).to_numpy(dtype=float), device=device, dtype=dtype)
        self.state_std = torch.tensor(states_t.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=float), device=device, dtype=dtype)
        self.covariance_train_mode = str(cfg.pipinn.covariance_train_mode)

    def mu_batch(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mean_map.predict_batch(x.detach().cpu().numpy())
        return torch.tensor(mu, device=self.device, dtype=self.dtype)

    def drift_batch(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        drift = self.drift_intercept[None, :] + x_np @ self.drift_matrix.T
        return torch.tensor(drift, device=self.device, dtype=self.dtype)


class TrainedPIPINN:
    def __init__(
        self,
        *,
        model_u: ValueNet,
        env: PIPINNEnvFromPPGDPO,
        train_objective: float,
        train_seed: int,
        train_history: list[dict[str, Any]] | None = None,
        best_validation_loss: float | None = None,
    ):
        self.model_u = model_u
        self.env = env
        self.train_objective = float(train_objective)
        self.train_seed = int(train_seed)
        self.state_columns = list(env.state_columns)
        self.asset_columns = list(env.asset_columns)
        self.train_history = list(train_history or [])
        self.best_validation_loss = float(best_validation_loss) if best_validation_loss is not None else float('nan')

    def _state_tensor(self, state_row: pd.Series | np.ndarray) -> torch.Tensor:
        if isinstance(state_row, pd.Series):
            arr = state_row[self.state_columns].to_numpy(dtype=float).reshape(1, -1)
        else:
            arr = np.asarray(state_row, dtype=float).reshape(1, -1)
        return torch.tensor(arr, device=self.env.device, dtype=self.env.dtype)

    def grad_u(self, state_row: pd.Series | np.ndarray, *, tau: float | None = None) -> np.ndarray:
        tau_val = float(self.env.tau_max if tau is None else tau)
        tau_t = torch.tensor([[tau_val]], device=self.env.device, dtype=self.env.dtype, requires_grad=True)
        x_t = self._state_tensor(state_row).detach().clone().requires_grad_(True)
        with torch.enable_grad():
            u = self.model_u(tau_t, x_t)
            u_x = torch.autograd.grad(u, x_t, grad_outputs=torch.ones_like(u), create_graph=False)[0]
        return u_x.squeeze(0).detach().cpu().numpy().astype(float)

    def estimate_costates(self, state_row: pd.Series | np.ndarray, *, wealth: float = 1.0, tau0: float | None = None) -> CostateEstimate:
        wealth = float(max(wealth, 1.0e-12))
        grad = self.grad_u(state_row, tau=tau0)
        return CostateEstimate(
            JX=1.0 / wealth,
            JXX=-float(self.env.gamma) / (wealth * wealth),
            JXY=grad / wealth,
            closed_form=True,
        )

    def policy_weights(
        self,
        state_row: pd.Series | np.ndarray,
        *,
        covariance: np.ndarray | None = None,
        cross_mat: np.ndarray | None = None,
        tau: float | None = None,
    ) -> np.ndarray:
        cov = self.env.Sigma_train if covariance is None else _symmetrize_psd(np.asarray(covariance, dtype=float), floor=1.0e-10)
        cross = self.env.C_train if cross_mat is None else np.asarray(cross_mat, dtype=float)
        if cross.ndim == 1:
            cross = cross.reshape(-1, 1)
        costates = self.estimate_costates(state_row, wealth=1.0, tau0=tau)
        if isinstance(state_row, pd.Series):
            x = state_row[self.state_columns].to_numpy(dtype=float).reshape(1, -1)
        else:
            x = np.asarray(state_row, dtype=float).reshape(1, -1)
        mu = self.env.mean_map.predict_batch(x).reshape(-1)
        w, _ = solve_ppgdpo_projection(
            mu=mu,
            cov=cov,
            cross_mat=cross,
            costates=costates,
            risky_cap=self.env.risky_cap,
            cash_floor=0.0,
            wealth=1.0,
            cross_scale=1.0,
        )
        return np.asarray(w, dtype=float)


def _sample_collocation(
    env: PIPINNEnvFromPPGDPO,
    *,
    n_int: int,
    n_bc: int,
    p_uniform: float,
    p_emp: float,
    p_tau_head: float,
    p_tau_near0: float,
    tau_head_window: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p_uniform = float(np.clip(p_uniform, 0.0, 1.0))
    p_emp = float(np.clip(p_emp, 0.0, 1.0))
    if p_uniform + p_emp <= 0.0:
        p_emp = 1.0
        p_uniform = 0.0
    else:
        s = p_uniform + p_emp
        p_uniform /= s
        p_emp /= s

    u = rng.uniform(size=n_int)
    tau = np.empty(n_int, dtype=float)
    p_tau_head = float(np.clip(p_tau_head, 0.0, 1.0))
    p_tau_near0 = float(np.clip(p_tau_near0, 0.0, 1.0))
    p_tau_uniform = float(max(1.0 - p_tau_head - p_tau_near0, 0.0))
    s_tau = p_tau_head + p_tau_near0 + p_tau_uniform
    if s_tau <= 0.0:
        p_tau_head, p_tau_near0, p_tau_uniform = 0.0, 0.0, 1.0
    else:
        p_tau_head /= s_tau
        p_tau_near0 /= s_tau
        p_tau_uniform /= s_tau

    head_window = float(max(int(tau_head_window), 1))
    tau_head_lo = float(max(env.tau_max - head_window, 0.0))
    mask_head_tau = u < p_tau_head
    mask_near0_tau = (u >= p_tau_head) & (u < (p_tau_head + p_tau_near0))
    mask_uniform_tau = ~(mask_head_tau | mask_near0_tau)
    tau[mask_head_tau] = rng.uniform(tau_head_lo, env.tau_max, size=int(mask_head_tau.sum()))
    tau[mask_near0_tau] = rng.beta(0.5, 1.0, size=int(mask_near0_tau.sum())) * env.tau_max
    tau[mask_uniform_tau] = rng.uniform(0.0, env.tau_max, size=int(mask_uniform_tau.sum()))

    x = np.empty((n_int, env.n_states), dtype=float)
    mix = rng.uniform(size=n_int)
    use_unif = mix < p_uniform
    use_emp = ~use_unif
    if use_unif.any():
        x[use_unif] = rng.uniform(low=env.x_min, high=env.x_max, size=(int(use_unif.sum()), env.n_states))
    if use_emp.any():
        idx = rng.integers(low=0, high=env.x_empirical.shape[0], size=int(use_emp.sum()))
        x[use_emp] = env.x_empirical[idx]
    x = np.minimum(np.maximum(x, env.x_min), env.x_max)

    xb = rng.uniform(low=env.x_min, high=env.x_max, size=(n_bc, env.n_states))
    if n_bc > 0:
        idxb = rng.integers(low=0, high=env.x_empirical.shape[0], size=n_bc)
        mask_emp_bc = rng.uniform(size=n_bc) < 0.5
        xb[mask_emp_bc] = env.x_empirical[idxb[mask_emp_bc]]
    xb = np.minimum(np.maximum(xb, env.x_min), env.x_max)
    taub = np.zeros((n_bc, 1), dtype=float)
    return tau.reshape(-1, 1), x, taub, xb


def _g_and_derivs(model_u: ValueNet, tau: torch.Tensor, x: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u = model_u(tau, x)
    g = torch.exp(torch.clamp(u, min=-20.0, max=20.0))
    ones = torch.ones_like(g)
    g_tau = torch.autograd.grad(g, tau, grad_outputs=ones, create_graph=create_graph, retain_graph=True)[0]
    g_x = torch.autograd.grad(g, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return u, g, g_tau, g_x


def _drift_term(env: PIPINNEnvFromPPGDPO, x: torch.Tensor, g_x: torch.Tensor) -> torch.Tensor:
    drift = env.drift_batch(x)
    return torch.sum(drift * g_x, dim=1, keepdim=True)


def _diffusion_term(env: PIPINNEnvFromPPGDPO, x: torch.Tensor, g_x: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    bsz, m = x.shape
    if np.allclose(env.Q, np.diag(np.diag(env.Q)), atol=1.0e-12):
        q_diag = torch.diag(env.Q_t).view(1, -1)
        g_xx_diag: list[torch.Tensor] = []
        for i in range(m):
            gi = g_x[:, i:i + 1]
            dgi_dx = torch.autograd.grad(gi, x, grad_outputs=torch.ones_like(gi), create_graph=create_graph, retain_graph=True)[0]
            g_xx_diag.append(dgi_dx[:, i:i + 1])
        g_xx = torch.cat(g_xx_diag, dim=1)
        return 0.5 * torch.sum(q_diag * g_xx, dim=1, keepdim=True)
    hess_parts: list[torch.Tensor] = []
    for i in range(m):
        gi = g_x[:, i:i + 1]
        dgi_dx = torch.autograd.grad(gi, x, grad_outputs=torch.ones_like(gi), create_graph=create_graph, retain_graph=True)[0]
        hess_parts.append(dgi_dx.unsqueeze(1))
    hess = torch.cat(hess_parts, dim=1)
    tr = torch.einsum('ij,bij->b', env.Q_t, hess).unsqueeze(-1)
    return 0.5 * tr


def _precompute_policy_coeffs(
    env: PIPINNEnvFromPPGDPO,
    policy_u_net: ValueNet | None,
    tau_np: np.ndarray,
    x_np: np.ndarray,
) -> _BatchPolicyCoefficients:
    tau = torch.tensor(tau_np, device=env.device, dtype=env.dtype)
    x = torch.tensor(x_np, device=env.device, dtype=env.dtype)
    mu = env.mu_batch(x)
    if policy_u_net is None:
        u_x = torch.zeros((x.shape[0], env.n_states), device=env.device, dtype=env.dtype)
    else:
        tau_g = tau.detach().clone().requires_grad_(True)
        x_g = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            u = policy_u_net(tau_g, x_g)
            u_x = torch.autograd.grad(u, x_g, grad_outputs=torch.ones_like(u), create_graph=False)[0]
        u_x = u_x.detach()
    hedging = torch.matmul(u_x, env.C_train_t.T)
    v = mu + hedging
    pi = _solve_qp_long_only_budget_full(
        env.Sigma_train_t,
        v,
        gamma=env.gamma,
        cap=env.risky_cap,
        iters=300,
        tol=1.0e-10,
        step_scale=1.1,
    )
    pi_sigma_pi = torch.sum(pi * torch.matmul(pi, env.Sigma_train_t.T), dim=1, keepdim=True)
    A = (1.0 - env.gamma) * (torch.sum(pi * mu, dim=1, keepdim=True) - 0.5 * env.gamma * pi_sigma_pi)
    Bcoef = (1.0 - env.gamma) * torch.matmul(pi, env.C_train_t)
    return _BatchPolicyCoefficients(tau=tau.detach(), x=x.detach(), A=A.detach(), Bcoef=Bcoef.detach())

def _compute_gradient_diagnostics(
    model_u: ValueNet,
    x: torch.Tensor,
    tau: torch.Tensor,
) -> dict[str, float]:
    """
    State 방향 gradient ∇_x u의 분포 특성을 계산.

    - grad_l2_mean/std/cv: 샘플마다 ||∇_x u||의 크기 분포.
      cv(variation coefficient)가 작으면 gradient 크기가 state에 거의 무관 → spectral bias 증거.
    - grad_dir_cos_mean: 서로 다른 state 샘플 간 gradient 방향의 cosine similarity.
      1에 가까우면 gradient 방향이 거의 상수 → policy가 state에 반응 안 함 → spectral bias의 결정적 지표.

    evaluation 시점이라 create_graph=False.
    """
    with torch.enable_grad():
        x_req = x.detach().clone().requires_grad_(True)
        tau_req = tau.detach().clone()
        u_pred = model_u(tau_req, x_req)
        grad = torch.autograd.grad(
            u_pred, x_req, grad_outputs=torch.ones_like(u_pred),
            create_graph=False, retain_graph=False,
        )[0]
    # 크기 분포
    grad_l2 = grad.norm(dim=1)  # [B]
    l2_mean = float(grad_l2.mean().item())
    l2_std = float(grad_l2.std().item())
    l2_cv = l2_std / (l2_mean + 1e-9)
    # 방향 분포: 단위 벡터들 간 평균 pairwise cosine
    g_unit = grad / (grad.norm(dim=1, keepdim=True) + 1e-9)
    cos_mat = g_unit @ g_unit.T  # [B, B]
    # 상삼각만 (대각 제외): 평균 pairwise cosine
    B = cos_mat.shape[0]
    if B > 1:
        mask = torch.triu(torch.ones(B, B, dtype=torch.bool, device=cos_mat.device), diagonal=1)
        cos_mean = float(cos_mat[mask].mean().item())
    else:
        cos_mean = float('nan')
    return {
        'grad_l2_mean': l2_mean,
        'grad_l2_std': l2_std,
        'grad_l2_cv': float(l2_cv),
        'grad_dir_cos_mean': cos_mean,
    }


def _policy_evaluation(
    env: PIPINNEnvFromPPGDPO,
    model_u: ValueNet,
    optimizer: torch.optim.Optimizer,
    train_coeff: _BatchPolicyCoefficients,
    train_taub: torch.Tensor,
    train_xb: torch.Tensor,
    val_coeff: _BatchPolicyCoefficients,
    val_taub: torch.Tensor,
    val_xb: torch.Tensor,
    *,
    epochs: int,
    w_bc: float,
    w_bc_dx: float,
    grad_clip: float,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[list[dict[str, float]], float]:
    hist: list[dict[str, float]] = []
    best_val = float('inf')
    best_state = None
    epoch_iter = range(1, max(int(epochs), 1) + 1)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc=str(progress_desc or 'PI-PINN epochs'), leave=False, unit='epoch')
    for epoch_in_outer in epoch_iter:
        model_u.train()
        optimizer.zero_grad()
        tau = train_coeff.tau.detach().clone().requires_grad_(True)
        x = train_coeff.x.detach().clone().requires_grad_(True)
        _, g, g_tau, g_x = _g_and_derivs(model_u, tau, x, create_graph=True)
        drift = _drift_term(env, x, g_x)
        diff = _diffusion_term(env, x, g_x, create_graph=True)
        bterm = torch.sum(train_coeff.Bcoef * g_x, dim=1, keepdim=True)
        rhs = drift + diff + train_coeff.A * g + bterm
        res = g_tau - rhs
        loss_pde = torch.mean(res * res)
        xb_g = train_xb.detach().clone().requires_grad_(True)
        u_bc = model_u(train_taub, xb_g)
        loss_bc = torch.mean(u_bc * u_bc)
        u_x_bc = torch.autograd.grad(u_bc, xb_g, grad_outputs=torch.ones_like(u_bc), create_graph=True)[0]
        loss_bc_dx = torch.mean(torch.sum(u_x_bc * u_x_bc, dim=1, keepdim=True))
        loss = loss_pde + float(w_bc) * loss_bc + float(w_bc_dx) * loss_bc_dx
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_u.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        model_u.eval()
        with torch.enable_grad():
            tau_v = val_coeff.tau.detach().clone().requires_grad_(True)
            x_v = val_coeff.x.detach().clone().requires_grad_(True)
            _, g_v, g_tau_v, g_x_v = _g_and_derivs(model_u, tau_v, x_v, create_graph=False)
            drift_v = _drift_term(env, x_v, g_x_v)
            diff_v = _diffusion_term(env, x_v, g_x_v, create_graph=False)
            bterm_v = torch.sum(val_coeff.Bcoef * g_x_v, dim=1, keepdim=True)
            rhs_v = drift_v + diff_v + val_coeff.A * g_v + bterm_v
            res_v = g_tau_v - rhs_v
            val_pde = torch.mean(res_v * res_v)
            xb_v = val_xb.detach().clone().requires_grad_(True)
            u_bc_v = model_u(val_taub, xb_v)
            val_bc = torch.mean(u_bc_v * u_bc_v)
            u_x_bc_v = torch.autograd.grad(u_bc_v, xb_v, grad_outputs=torch.ones_like(u_bc_v), create_graph=False)[0]
            val_bc_dx = torch.mean(torch.sum(u_x_bc_v * u_x_bc_v, dim=1, keepdim=True))
            val_total = val_pde + float(w_bc) * val_bc + float(w_bc_dx) * val_bc_dx
            # --- 진단 지표 (spectral bias detector) ---
            diag = _compute_gradient_diagnostics(model_u, val_coeff.x, val_coeff.tau)
        cur_val = float(val_total.item())
        hist.append({
            'epoch_in_outer': int(epoch_in_outer),
            'lr': float(optimizer.param_groups[0]['lr']) if optimizer.param_groups else float('nan'),
            'train_total': float(loss.item()),
            'train_pde': float(loss_pde.item()),
            'train_bc': float(loss_bc.item()),
            'train_bc_dx': float(loss_bc_dx.item()),
            'val_total': cur_val,
            'val_pde': float(val_pde.item()),
            'val_bc': float(val_bc.item()),
            'val_bc_dx': float(val_bc_dx.item()),
            # --- 진단 지표 ---
            'val_grad_l2_mean':     diag['grad_l2_mean'],
            'val_grad_l2_std':      diag['grad_l2_std'],
            'val_grad_l2_cv':       diag['grad_l2_cv'],
            'val_grad_dir_cos_mean': diag['grad_dir_cos_mean'],
        })
        if show_progress and hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix({'val': f'{cur_val:.3e}'})
        if cur_val < best_val:
            best_val = cur_val
            best_state = {k: v.detach().cpu().clone() for k, v in model_u.state_dict().items()}
    if best_state is not None:
        model_u.load_state_dict(best_state)
    return hist, best_val


def _select_training_covariance(
    *,
    cfg: Any,
    cov_model: Any,
    cross_est: CrossCovarianceEstimate,
    state_train: pd.DataFrame,
    factor_train: pd.DataFrame,
    loadings: pd.DataFrame,
    residual_var: pd.Series,
) -> np.ndarray:
    mode = str(cfg.pipinn.covariance_train_mode)
    if mode == 'cross_resid':
        return _symmetrize_psd(np.asarray(cross_est.return_resid_cov, dtype=float), floor=1.0e-10)
    if len(state_train) <= 0:
        return _symmetrize_psd(np.asarray(cross_est.return_resid_cov, dtype=float), floor=1.0e-10)
    state_row = state_train.iloc[-1]
    latest_factor_return = factor_train.iloc[-1] if len(factor_train) > 0 else pd.Series(dtype=float)
    try:
        cov_fc = cov_model.predict(state_row, latest_factor_return, loadings, residual_var)
        return _symmetrize_psd(np.asarray(cov_fc.asset_cov, dtype=float), floor=1.0e-10)
    except Exception:
        return _symmetrize_psd(np.asarray(cross_est.return_resid_cov, dtype=float), floor=1.0e-10)


def train_pipinn_policy(
    states_t: pd.DataFrame,
    returns_tp1: pd.DataFrame,
    cfg: Any,
    transaction_cost: float,
    *,
    mean_model: MeanModelResult,
    transition: StateTransitionResult,
    cross_est: CrossCovarianceEstimate,
    cov_model: Any,
    factor_repr: Any,
    progress_label: str | None = None,
    tau_max: float | None = None,
) -> TrainedPIPINN:
    del transaction_cost
    if states_t.shape[1] <= 0:
        raise ValueError('PI-PINN backend requires at least one state variable')
    train_seed = int(cfg.ppgdpo.train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if str(cfg.pipinn.device).lower() == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = str(cfg.pipinn.device or cfg.ppgdpo.device)
    dtype = torch.float64 if str(cfg.pipinn.dtype).lower() == 'float64' else torch.float32
    tau_head_window_cfg = int(getattr(cfg.pipinn, 'tau_head_window', 0) or 0)
    if tau_head_window_cfg <= 0:
        tau_head_window_eff = int(max(int(getattr(cfg.split, 'refit_every', 1) or 1), 1))
    else:
        tau_head_window_eff = tau_head_window_cfg
    sigma_train = _select_training_covariance(
        cfg=cfg,
        cov_model=cov_model,
        cross_est=cross_est,
        state_train=states_t,
        factor_train=factor_repr.factors if hasattr(factor_repr, 'factors') else pd.DataFrame(index=states_t.index),
        loadings=factor_repr.loadings,
        residual_var=factor_repr.residual_var,
    )
    # tau_max = float(max(int(cfg.ppgdpo.horizon_steps), 1))
    tau_max_cfg = int(cfg.ppgdpo.horizon_steps)
    tau_cap = tau_max_cfg if tau_max is None else int(np.ceil(float(tau_max)))
    tau_max = float(max(tau_cap, 1))
    env = PIPINNEnvFromPPGDPO(
        mean_model=mean_model,
        transition=transition,
        cross_est=cross_est,
        states_t=states_t,
        sigma_train=sigma_train,
        cfg=cfg,
        tau_max=tau_max,
        device=device,
        dtype=dtype,
    )
    model_u = ValueNet(
        tau_max=tau_max,
        x_min=env.x_min,
        x_max=env.x_max,
        width=int(cfg.pipinn.width),
        depth=int(cfg.pipinn.depth),
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model_u.parameters(), lr=float(cfg.pipinn.lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(cfg.pipinn.scheduler_factor),
        patience=int(cfg.pipinn.scheduler_patience),
        min_lr=float(cfg.pipinn.min_lr),
    )
    tau_val_np, x_val_np, taub_val_np, xb_val_np = _sample_collocation(
        env,
        n_int=int(cfg.pipinn.n_val_int),
        n_bc=int(cfg.pipinn.n_val_bc),
        p_uniform=float(cfg.pipinn.p_uniform),
        p_emp=float(cfg.pipinn.p_emp),
        p_tau_head=float(cfg.pipinn.p_tau_head),
        p_tau_near0=float(cfg.pipinn.p_tau_near0),
        tau_head_window=tau_head_window_eff,
        seed=train_seed + 999,
    )
    val_taub = torch.tensor(taub_val_np, device=device, dtype=dtype)
    val_xb = torch.tensor(xb_val_np, device=device, dtype=dtype)

    best_overall = float('inf')
    best_state = None
    all_hist: list[dict[str, float]] = []
    policy_u_net: ValueNet | None = None
    global_epoch = 0
    show_progress = bool(getattr(cfg.pipinn, 'show_progress', False))
    show_epoch_progress = bool(getattr(cfg.pipinn, 'show_epoch_progress', False))
    outer_iter = range(1, max(int(cfg.pipinn.outer_iters), 1) + 1)
    if show_progress:
        outer_iter = tqdm(outer_iter, desc=str(progress_label or 'PI-PINN training'), unit='outer')
    for it in outer_iter:
        tau_tr_np, x_tr_np, taub_tr_np, xb_tr_np = _sample_collocation(
            env,
            n_int=int(cfg.pipinn.n_train_int),
            n_bc=int(cfg.pipinn.n_train_bc),
            p_uniform=float(cfg.pipinn.p_uniform),
            p_emp=float(cfg.pipinn.p_emp),
            p_tau_head=float(cfg.pipinn.p_tau_head),
            p_tau_near0=float(cfg.pipinn.p_tau_near0),
            tau_head_window=tau_head_window_eff,
            seed=train_seed + it,
        )
        train_taub = torch.tensor(taub_tr_np, device=device, dtype=dtype)
        train_xb = torch.tensor(xb_tr_np, device=device, dtype=dtype)
        train_coeff = _precompute_policy_coeffs(env, policy_u_net, tau_tr_np, x_tr_np)
        val_coeff = _precompute_policy_coeffs(env, policy_u_net, tau_val_np, x_val_np)
        hist, best_val = _policy_evaluation(
            env,
            model_u,
            optimizer,
            train_coeff,
            train_taub,
            train_xb,
            val_coeff,
            val_taub,
            val_xb,
            epochs=int(cfg.pipinn.eval_epochs),
            w_bc=float(cfg.pipinn.w_bc),
            w_bc_dx=float(cfg.pipinn.w_bc_dx),
            grad_clip=float(cfg.pipinn.grad_clip),
            show_progress=show_epoch_progress,
            progress_desc=f"{progress_label or 'PI-PINN'} outer {int(it)}",
        )
        for row in hist:
            global_epoch += 1
            enriched = dict(row)
            enriched['outer_iter'] = int(it)
            enriched['global_epoch'] = int(global_epoch)
            all_hist.append(enriched)
        scheduler.step(best_val)
        if show_progress and hasattr(outer_iter, 'set_postfix'):
            outer_iter.set_postfix({
                'best_val': f'{float(best_val):.3e}',
                'lr': f"{float(optimizer.param_groups[0]['lr']) if optimizer.param_groups else float('nan'):.2e}",
            })
        if best_val < best_overall:
            best_overall = best_val
            best_state = {k: v.detach().cpu().clone() for k, v in model_u.state_dict().items()}
        policy_u_net = copy.deepcopy(model_u).eval()
    if best_state is not None:
        model_u.load_state_dict(best_state)
    train_objective = -float(best_overall) if np.isfinite(best_overall) else float('nan')
    return TrainedPIPINN(
        model_u=model_u,
        env=env,
        train_objective=train_objective,
        train_seed=train_seed,
        train_history=all_hist,
        best_validation_loss=best_overall if np.isfinite(best_overall) else None,
    )
