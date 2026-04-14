from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .mean_model import MeanModelResult
from .transition import CrossCovarianceEstimate, StateTransitionResult
from .utils import project_capped_simplex


def _effective_gamma(*, utility: str, risk_aversion: float) -> float:
    return 1.0 if str(utility).lower() == 'log' else float(risk_aversion)


def _crra_utility(x: torch.Tensor, gamma: float) -> torch.Tensor:
    x = torch.clamp(x, min=1.0e-12)
    if abs(float(gamma) - 1.0) < 1.0e-12:
        return torch.log(x)
    return x.pow(1.0 - float(gamma)) / (1.0 - float(gamma))


def _symmetrize_psd(mat: np.ndarray, *, floor: float = 1.0e-10) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    arr = 0.5 * (arr + arr.T)
    eigval, eigvec = np.linalg.eigh(arr)
    eigval = np.clip(eigval, floor, None)
    return eigvec @ np.diag(eigval) @ eigvec.T


def _as_numpy_state(state_row: pd.Series, columns: list[str]) -> np.ndarray:
    return state_row[columns].to_numpy(dtype=np.float64)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WarmupPolicyNet(nn.Module):
    """Long-only risky weights with residual cash on a simplex budget."""

    def __init__(
        self,
        *,
        state_dim: int,
        asset_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        risky_cap: float,
        include_time: bool = True,
    ):
        super().__init__()
        self.asset_dim = int(asset_dim)
        self.risky_cap = float(risky_cap)
        self.include_time = bool(include_time)
        in_dim = int(state_dim) + 1 + (1 if include_time else 0)
        self.body = _MLP(in_dim=in_dim, out_dim=self.asset_dim + 1, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

    def forward(self, tau: torch.Tensor, wealth: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        log_wealth = torch.log(torch.clamp(wealth, min=1.0e-12))
        inp = torch.cat([tau, log_wealth, states], dim=1) if self.include_time else torch.cat([log_wealth, states], dim=1)
        logits = self.body(inp)
        z = torch.softmax(logits, dim=-1)
        risky = z[:, : self.asset_dim] * self.risky_cap
        return risky


@dataclass
class LinearGaussianDynamicsModel:
    a: torch.Tensor
    B: torch.Tensor
    c: torch.Tensor
    A: torch.Tensor
    Sigma: torch.Tensor
    Q: torch.Tensor
    Cross: torch.Tensor
    joint: torch.Tensor
    chol_joint: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor

    @property
    def n_assets(self) -> int:
        return int(self.a.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.c.shape[0])

    def mu_excess(self, y: torch.Tensor) -> torch.Tensor:
        return self.a.unsqueeze(0) + y @ self.B.T

    def sample_innov(self, batch: int, *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(batch, self.n_assets + self.n_states, dtype=self.a.dtype, device=self.a.device, generator=generator)
        innov = z @ self.chol_joint.T
        return innov[:, : self.n_assets], innov[:, self.n_assets :]

    def step_state(self, y: torch.Tensor, innov: torch.Tensor) -> torch.Tensor:
        return self.c.unsqueeze(0) + y @ self.A.T + innov

    def clamp_state(self, y: torch.Tensor, *, clip_std_abs: float | None) -> torch.Tensor:
        if clip_std_abs is None:
            return y
        c = float(clip_std_abs)
        z = (y - self.state_mean.unsqueeze(0)) / self.state_std.unsqueeze(0)
        z = torch.clamp(z, min=-c, max=c)
        return self.state_mean.unsqueeze(0) + z * self.state_std.unsqueeze(0)


def _asset_mean_blocks_from_model(mean_model: MeanModelResult) -> tuple[np.ndarray, np.ndarray, list[str]]:
    coef = np.asarray(mean_model.coef, dtype=float)
    assets = list(mean_model.assets)
    if mean_model.kind == 'direct_assets':
        return coef[0, :], coef[1:, :].T, assets
    if mean_model.asset_alpha is None or mean_model.loadings is None or mean_model.factor_columns is None:
        raise ValueError('factor_apt mean model is missing loadings/alpha/factor_columns')
    loadings = mean_model.loadings.reindex(index=assets, columns=mean_model.factor_columns).to_numpy(dtype=float)
    alpha = mean_model.asset_alpha.reindex(assets).to_numpy(dtype=float)
    intercept = alpha + loadings @ coef[0, :]
    state_beta = loadings @ coef[1:, :].T
    return intercept, state_beta, assets


def build_linear_gaussian_dynamics(
    *,
    mean_model: MeanModelResult,
    transition: StateTransitionResult,
    cross_est: CrossCovarianceEstimate,
    states_t: pd.DataFrame,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float64,
    covariance_mode: str = 'full',
) -> LinearGaussianDynamicsModel:
    a_np, B_np, asset_order = _asset_mean_blocks_from_model(mean_model)
    trans_coef = np.asarray(transition.coef, dtype=float)
    c_np = trans_coef[0, :]
    A_np = trans_coef[1:, :].T

    ret_source_order = list(cross_est.cross.index)
    state_source_order = list(cross_est.cross.columns)
    asset_perm = [ret_source_order.index(asset) for asset in asset_order]
    state_perm = [state_source_order.index(state) for state in list(transition.targets)]

    sigma_np = np.asarray(cross_est.return_resid_cov, dtype=float)[np.ix_(asset_perm, asset_perm)]
    if str(covariance_mode).lower() == 'diag':
        sigma_np = np.diag(np.diag(sigma_np))
    sigma_np = _symmetrize_psd(sigma_np, floor=1.0e-10)
    q_np = _symmetrize_psd(np.asarray(cross_est.state_innov_cov, dtype=float)[np.ix_(state_perm, state_perm)], floor=1.0e-10)
    cross_np = cross_est.cross.reindex(index=asset_order, columns=list(transition.targets)).to_numpy(dtype=float)

    n = len(asset_order)
    k = len(transition.targets)
    joint_np = np.zeros((n + k, n + k), dtype=float)
    joint_np[:n, :n] = sigma_np
    joint_np[n:, n:] = q_np
    joint_np[:n, n:] = cross_np
    joint_np[n:, :n] = cross_np.T

    chol_np = None
    shrink = 1.0
    jitter = 1.0e-10
    for _ in range(48):
        try:
            cand = joint_np.copy()
            cand[:n, n:] *= shrink
            cand[n:, :n] *= shrink
            cand = 0.5 * (cand + cand.T) + jitter * np.eye(n + k)
            chol_np = np.linalg.cholesky(cand)
            joint_np = cand
            break
        except np.linalg.LinAlgError:
            shrink *= 0.9
            jitter *= 2.0
    if chol_np is None:
        fallback = np.zeros_like(joint_np)
        fallback[:n, :n] = sigma_np + 1.0e-6 * np.eye(n)
        fallback[n:, n:] = q_np + 1.0e-6 * np.eye(k)
        joint_np = fallback
        chol_np = np.linalg.cholesky(joint_np)
        cross_np = np.zeros_like(cross_np)

    state_mean = states_t.reindex(columns=list(transition.targets)).mean().to_numpy(dtype=float)
    state_std = states_t.reindex(columns=list(transition.targets)).std(ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=float)

    return LinearGaussianDynamicsModel(
        a=torch.tensor(a_np, dtype=dtype, device=device),
        B=torch.tensor(B_np, dtype=dtype, device=device),
        c=torch.tensor(c_np, dtype=dtype, device=device),
        A=torch.tensor(A_np, dtype=dtype, device=device),
        Sigma=torch.tensor(sigma_np, dtype=dtype, device=device),
        Q=torch.tensor(q_np, dtype=dtype, device=device),
        Cross=torch.tensor(cross_np, dtype=dtype, device=device),
        joint=torch.tensor(joint_np, dtype=dtype, device=device),
        chol_joint=torch.tensor(chol_np, dtype=dtype, device=device),
        state_mean=torch.tensor(state_mean, dtype=dtype, device=device),
        state_std=torch.tensor(np.clip(state_std, 1.0e-8, None), dtype=dtype, device=device),
    )


class EpisodeSampler:
    def __init__(self, states_t: pd.DataFrame):
        self._states = states_t.to_numpy(dtype=np.float64)
        if self._states.size == 0:
            raise ValueError('states_t must not be empty for PG-DPO warm-up sampling')

    def sample(self, batch_size: int, *, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, len(self._states), size=int(batch_size))
        y0 = torch.tensor(self._states[idx], dtype=dtype, device=device)
        x0 = torch.ones((int(batch_size), 1), dtype=dtype, device=device)
        return x0, y0


def rollout_paths_linear(
    *,
    model: LinearGaussianDynamicsModel,
    policy: WarmupPolicyNet,
    X0: torch.Tensor,
    Y0: torch.Tensor,
    horizon_steps: int,
    gamma: float,
    kappa: float,
    clamp_wealth_min: float,
    clamp_port_ret_min: float,
    clamp_port_ret_max: float,
    clamp_state_std_abs: float | None,
    generator: torch.Generator | None = None,
    tau0: float = 1.0,
) -> torch.Tensor:
    X = X0
    Y = Y0
    batch = int(X.shape[0])
    H = max(int(horizon_steps), 1)
    for t in range(H):
        tau_val = max(0.0, min(1.0, float(tau0) - (t / H)))
        tau = torch.full((batch, 1), tau_val, dtype=X.dtype, device=X.device)
        pi = policy(tau, X, Y)                                                     # tau는 policy network 입력으로만 사용
        eps, u = model.sample_innov(batch=batch, generator=generator)              # 월간 공분산에서 샘플
        r = model.mu_excess(Y) + eps                                               # 월간 기대수익 + 월간 충격
        port_ret = torch.sum(pi * r, dim=1)
        port_ret = torch.clamp(port_ret, min=float(clamp_port_ret_min), max=float(clamp_port_ret_max))
        X = torch.clamp(X * (1.0 + port_ret).unsqueeze(1), min=float(clamp_wealth_min))
        Y = model.clamp_state(model.step_state(Y, u), clip_std_abs=clamp_state_std_abs)
    J = float(kappa) * _crra_utility(X.squeeze(1), gamma)
    if not torch.isfinite(J).all():
        raise RuntimeError('Non-finite PG-DPO objective detected during rollout_paths_linear')
    return J


@dataclass
class CostateEstimate:
    JX: float
    JXX: float
    JXY: np.ndarray
    closed_form: bool = False


@dataclass
class TrainedPPGDPO:
    policy_net: WarmupPolicyNet
    dynamics: LinearGaussianDynamicsModel
    train_objective: float
    state_columns: list[str]
    asset_columns: list[str]
    device: str
    gamma: float
    utility: str
    horizon_steps: int
    kappa: float
    clamp_wealth_min: float
    clamp_port_ret_min: float
    clamp_port_ret_max: float
    clamp_state_std_abs: float | None
    mc_rollouts: int
    mc_sub_batch: int
    clamp_neg_jxx_min: float
    train_seed: int

    def policy_weights(self, state_row: pd.Series, *, wealth: float = 1.0, tau: float = 1.0) -> np.ndarray:
        y = _as_numpy_state(state_row, self.state_columns)
        dtype = self.policy_net.body.net[0].weight.dtype
        y_t = torch.tensor(y, dtype=dtype, device=self.device).unsqueeze(0)
        x_t = torch.tensor([[float(wealth)]], dtype=dtype, device=self.device)
        tau_t = torch.tensor([[float(tau)]], dtype=dtype, device=self.device)
        with torch.no_grad():
            return self.policy_net(tau_t, x_t, y_t).squeeze(0).detach().cpu().numpy().astype(float)

    def estimate_costates(self, state_row: pd.Series, *, wealth: float = 1.0, tau0: float = 1.0) -> CostateEstimate:
        if abs(float(self.gamma) - 1.0) < 1.0e-12 or str(self.utility).lower() == 'log':
            x = max(float(wealth), 1.0e-12)
            return CostateEstimate(JX=float(self.kappa / x), JXX=float(-self.kappa / (x * x)), JXY=np.zeros(len(self.state_columns), dtype=float), closed_form=True)

        params = list(self.policy_net.parameters())
        req_bak = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(self.train_seed))
        dtype = self.policy_net.body.net[0].weight.dtype
        x0 = torch.tensor([[float(wealth)]], dtype=dtype, device=self.device, requires_grad=True)
        y0_np = _as_numpy_state(state_row, self.state_columns).reshape(1, -1)
        y0 = torch.tensor(y0_np, dtype=dtype, device=self.device, requires_grad=True)
        sum_JX = torch.zeros_like(x0)
        sum_JXX = torch.zeros_like(x0)
        sum_JXY = torch.zeros_like(y0)
        done = 0
        mc = max(int(self.mc_rollouts), 1)
        sub = max(int(self.mc_sub_batch), 1)
        try:
            while done < mc:
                r = min(sub, mc - done)
                x_rep = x0.repeat_interleave(r, dim=0)
                y_rep = y0.repeat_interleave(r, dim=0)
                J = rollout_paths_linear(
                    model=self.dynamics,
                    policy=self.policy_net,
                    X0=x_rep,
                    Y0=y_rep,
                    horizon_steps=self.horizon_steps,
                    gamma=self.gamma,
                    kappa=self.kappa,
                    clamp_wealth_min=self.clamp_wealth_min,
                    clamp_port_ret_min=self.clamp_port_ret_min,
                    clamp_port_ret_max=self.clamp_port_ret_max,
                    clamp_state_std_abs=self.clamp_state_std_abs,
                    generator=gen,
                    tau0=float(tau0),
                )
                total = J.view(1, r).sum(dim=1).sum()
                JX, _ = torch.autograd.grad(total, [x0, y0], create_graph=True, allow_unused=True)
                if JX is None:
                    JX = torch.zeros_like(x0)
                JXX, JXY = torch.autograd.grad(JX.sum(), [x0, y0], allow_unused=True)
                if JXX is None:
                    JXX = torch.zeros_like(x0)
                if JXY is None:
                    JXY = torch.zeros_like(y0)
                sum_JX += JX.detach()
                sum_JXX += JXX.detach()
                sum_JXY += JXY.detach()
                done += r
        finally:
            for p, req in zip(params, req_bak):
                p.requires_grad_(req)
        return CostateEstimate(
            JX=float((sum_JX / float(mc)).item()),
            JXX=float((sum_JXX / float(mc)).item()),
            JXY=(sum_JXY / float(mc)).squeeze(0).detach().cpu().numpy().astype(float),
            closed_form=False,
        )


def train_warmup_policy(
    states_t: pd.DataFrame,
    returns_tp1: pd.DataFrame,
    cfg: Any,
    transaction_cost: float,
    *,
    mean_model: MeanModelResult,
    transition: StateTransitionResult,
    cross_est: CrossCovarianceEstimate,
) -> TrainedPPGDPO:
    _ = transaction_cost
    train_seed = int(cfg.ppgdpo.train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    device = str(cfg.ppgdpo.device)
    dtype = torch.float64
    state_cols = list(states_t.columns)
    asset_cols = list(returns_tp1.columns)
    effective_risky_cap = min(float(cfg.policy.risky_cap), 1.0 - float(getattr(cfg.policy, 'cash_floor', 0.0)))

    dynamics = build_linear_gaussian_dynamics(
        mean_model=mean_model,
        transition=transition,
        cross_est=cross_est,
        states_t=states_t,
        device=device,
        dtype=dtype,
        covariance_mode=str(cfg.ppgdpo.covariance_mode),
    )
    policy = WarmupPolicyNet(
        state_dim=len(state_cols),
        asset_dim=len(asset_cols),
        hidden_dim=int(cfg.ppgdpo.hidden_dim),
        hidden_layers=int(cfg.ppgdpo.hidden_layers),
        risky_cap=effective_risky_cap,
        include_time=True,
    ).to(device=device, dtype=dtype)
    sampler = EpisodeSampler(states_t)
    opt = torch.optim.Adam(policy.parameters(), lr=float(cfg.ppgdpo.lr))
    gamma = _effective_gamma(utility=str(cfg.ppgdpo.utility), risk_aversion=float(cfg.policy.risk_aversion))

    train_objective = float('nan')
    for _ in range(max(int(cfg.ppgdpo.epochs), 1)):
        X0, Y0 = sampler.sample(int(cfg.ppgdpo.batch_size), device=device, dtype=dtype)
        J = rollout_paths_linear(
            model=dynamics,
            policy=policy,
            X0=X0,
            Y0=Y0,
            horizon_steps=int(cfg.ppgdpo.horizon_steps),
            gamma=gamma,
            kappa=float(cfg.ppgdpo.kappa),
            clamp_wealth_min=float(cfg.ppgdpo.clamp_wealth_min),
            clamp_port_ret_min=float(cfg.ppgdpo.clamp_min_return),
            clamp_port_ret_max=float(cfg.ppgdpo.clamp_port_ret_max),
            clamp_state_std_abs=cfg.ppgdpo.clamp_state_std_abs,
        )
        loss = -J.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
        train_objective = float(J.mean().detach().cpu().item())

    return TrainedPPGDPO(
        policy_net=policy,
        dynamics=dynamics,
        train_objective=train_objective,
        state_columns=state_cols,
        asset_columns=asset_cols,
        device=device,
        gamma=gamma,
        utility=str(cfg.ppgdpo.utility),
        horizon_steps=int(cfg.ppgdpo.horizon_steps),
        kappa=float(cfg.ppgdpo.kappa),
        clamp_wealth_min=float(cfg.ppgdpo.clamp_wealth_min),
        clamp_port_ret_min=float(cfg.ppgdpo.clamp_min_return),
        clamp_port_ret_max=float(cfg.ppgdpo.clamp_port_ret_max),
        clamp_state_std_abs=cfg.ppgdpo.clamp_state_std_abs,
        mc_rollouts=int(cfg.ppgdpo.mc_rollouts),
        mc_sub_batch=int(cfg.ppgdpo.mc_sub_batch),
        clamp_neg_jxx_min=float(cfg.ppgdpo.clamp_neg_jxx_min),
        train_seed=train_seed,
    )


def _interior_pull_long_only(u: np.ndarray, *, cap: float, margin: float) -> np.ndarray:
    out = np.maximum(np.asarray(u, dtype=float), margin)
    total = float(out.sum())
    if total >= cap - margin:
        scale = max((cap - margin) / max(total, 1.0e-12), 0.0)
        out = np.maximum(out * scale, margin)
        if float(out.sum()) >= cap - margin:
            out = project_capped_simplex(out, z=max(cap - margin, margin * len(out)))
            out = np.maximum(out, margin)
    return out


def _project_u_long_only_cash_barrier_numpy(
    *,
    a_vec: np.ndarray,
    q_mat: np.ndarray,
    cap: float,
    eps_bar: float,
    ridge: float,
    tau: float,
    armijo: float,
    backtrack: float,
    max_newton: int,
    tol_grad: float,
    max_ls: int,
    interior_margin: float,
) -> np.ndarray:
    a = np.asarray(a_vec, dtype=float).reshape(-1)
    q = _symmetrize_psd(np.asarray(q_mat, dtype=float), floor=max(float(ridge), 1.0e-12))
    n = len(a)
    I = np.eye(n)
    one = np.ones(n)
    try:
        u = np.linalg.solve(q + float(ridge) * I, a)
    except np.linalg.LinAlgError:
        u = np.linalg.pinv(q + float(ridge) * I) @ a
    u = _interior_pull_long_only(project_capped_simplex(u, z=float(cap)), cap=float(cap), margin=max(float(interior_margin), 1.0e-12))

    def hbar(x: np.ndarray) -> float:
        slack = float(cap - x.sum())
        if np.any(x <= 0.0) or slack <= 0.0:
            return -np.inf
        return float(a @ x - 0.5 * x @ q @ x + float(eps_bar) * np.log(x).sum() + float(eps_bar) * np.log(slack))

    for _ in range(max(int(max_newton), 1)):
        slack = max(float(cap - u.sum()), 1.0e-12)
        g = a - q @ u + float(eps_bar) / np.clip(u, 1.0e-12, None) - (float(eps_bar) / slack) * one
        if float(np.max(np.abs(g))) < float(tol_grad):
            break
        H = -q - np.diag(float(eps_bar) / np.clip(u, 1.0e-12, None) ** 2) - (float(eps_bar) / (slack ** 2)) * np.outer(one, one) + float(ridge) * I
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            d = np.linalg.pinv(H) @ (-g)
        alpha = 1.0
        neg = d < 0.0
        if np.any(neg):
            alpha = min(alpha, float(np.min(float(tau) * u[neg] / np.clip(-d[neg], 1.0e-24, None))))
        dsum = float(d.sum())
        if dsum > 0.0:
            alpha = min(alpha, float(tau) * slack / max(dsum, 1.0e-24))
        alpha = max(alpha, 1.0e-12)
        f0 = hbar(u)
        gTd = float(g @ d)
        accepted = False
        a_cur = alpha
        for _ls in range(max(int(max_ls), 1)):
            cand = u + a_cur * d
            f1 = hbar(cand)
            if np.isfinite(f1) and f1 >= f0 + float(armijo) * a_cur * gTd:
                u = cand
                accepted = True
                break
            a_cur *= float(backtrack)
        if not accepted:
            break
        u = _interior_pull_long_only(u, cap=float(cap), margin=max(float(interior_margin), 1.0e-12))
    return project_capped_simplex(u, z=float(cap))


def solve_ppgdpo_projection(
    *,
    mu: np.ndarray,
    cov: np.ndarray,
    cross_mat: np.ndarray,
    costates: CostateEstimate,
    risky_cap: float,
    cash_floor: float = 0.0,
    wealth: float = 1.0,
    cross_scale: float = 1.0,
    eps_bar: float = 1.0e-6,
    ridge: float = 1.0e-10,
    tau: float = 0.95,
    armijo: float = 1.0e-4,
    backtrack: float = 0.5,
    max_newton: int = 30,
    tol_grad: float = 1.0e-8,
    max_ls: int = 20,
    interior_margin: float = 1.0e-8,
    clamp_neg_jxx_min: float = 1.0e-12,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | bool]]:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = _symmetrize_psd(np.asarray(cov, dtype=float), floor=max(float(ridge), 1.0e-12))
    cross = np.asarray(cross_mat, dtype=float)
    jxy = np.asarray(costates.JXY, dtype=float).reshape(-1)
    wealth = float(wealth)
    hedge_signal = float(cross_scale) * (cross @ jxy)
    mu_term = wealth * float(costates.JX) * mu
    hedge_term = wealth * hedge_signal
    a_vec = mu_term + hedge_term
    neg_jxx = max(-float(costates.JXX), float(clamp_neg_jxx_min))
    q_mat = (wealth * wealth * neg_jxx) * cov
    cap = min(float(risky_cap), 1.0 - float(cash_floor))
    u = _project_u_long_only_cash_barrier_numpy(
        a_vec=a_vec,
        q_mat=q_mat,
        cap=cap,
        eps_bar=float(eps_bar),
        ridge=float(ridge),
        tau=float(tau),
        armijo=float(armijo),
        backtrack=float(backtrack),
        max_newton=int(max_newton),
        tol_grad=float(tol_grad),
        max_ls=int(max_ls),
        interior_margin=float(interior_margin),
    )
    return u, {
        'hedge_signal': hedge_signal,
        'mu_term': mu_term,
        'hedge_term': hedge_term,
        'a_vec': a_vec,
        'neg_jxx': neg_jxx,
        'closed_form_costates': bool(costates.closed_form),
    }
