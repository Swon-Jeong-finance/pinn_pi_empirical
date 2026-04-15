"""
Offline / historical-scenario training for a finite-horizon (e.g., 30-year) problem.

This matches an empirical protocol:
- Split data into non-overlapping TRAIN and TEST.
- TRAIN: generate many episodes by selecting start dates (moving windows) and running a 30-year horizon
         using the realized return/state sequences from the TRAIN period.
- TEST: evaluate one (or several) fixed-horizon runs on the most recent 30-year segment.

Key point:
- This does NOT require fitting a stochastic SDE simulator. Returns/states are treated as exogenous
  sequences, so the backtest is fully differentiable w.r.t. policy parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .simulator import crra_utility


@dataclass
class OfflineEpisodeConfig:
    horizon_years: int = 30
    steps_per_year: int = 12  # monthly
    include_rf: bool = True
    gamma: float = 5.0
    tc_bps: float = 0.0       # simple linear cost on turnover
    clip_pi: float = 5.0      # safety clamp

    # Regularization knobs (to improve OOS stability)
    l2_weight_penalty: float = 0.0   # penalize concentration / leverage via mean(sum(pi^2))
    turnover_penalty: float = 0.0    # additional penalty on mean turnover (beyond tc_bps)


class OfflineEpisodeSampler:
    """
    Samples episodes from historical arrays.

    Convention:
    - states Y are indexed by date t
    - returns R are realized from t-1 -> t, stored at date t
    - When allocating at date t, we use Y_t and apply to next return R_{t+1}.
      So for an episode starting at index s (decision time):
        decisions at indices s ... s+H-1
        realized returns indices (s+1) ... (s+H)
    """
    def __init__(
        self,
        y: np.ndarray,          # (T, k)
        r: np.ndarray,          # (T, n) simple returns for the period ending at t
        rf_step: Optional[np.ndarray],  # (T,) per-step simple rf, aligned with r
        horizon_steps: int,
    ):
        assert y.ndim == 2 and r.ndim == 2
        self.y = y
        self.r = r
        self.rf = rf_step
        self.H = horizon_steps
        self.T = y.shape[0]
        self.n = r.shape[1]
        self.k = y.shape[1]

        # valid start indices s such that s+H < T-1 (need r at s+H)
        self.valid = np.arange(0, self.T - self.H - 1, dtype=int)
        if len(self.valid) == 0:
            raise ValueError("Not enough data to form even one episode with this horizon.")

    def sample_batch(self, batch_size: int) -> np.ndarray:
        idx = np.random.randint(0, len(self.valid), size=batch_size)
        return self.valid[idx]

    def batch_tensors(
        self,
        starts: np.ndarray,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          Y_seq: (B, H, k)  states at decision times
          R_seq: (B, H, n)  realized simple returns for next periods
          rf_seq:(B, H)     per-step rf for next periods (0 if None)
        """
        B = len(starts)
        H = self.H
        k = self.k
        n = self.n

        Y_seq = np.zeros((B, H, k), dtype=float)
        R_seq = np.zeros((B, H, n), dtype=float)
        rf_seq = np.zeros((B, H), dtype=float)

        for i, s in enumerate(starts):
            Y_seq[i] = self.y[s : s + H]
            R_seq[i] = self.r[s + 1 : s + H + 1]
            if self.rf is not None:
                rf_seq[i] = self.rf[s + 1 : s + H + 1]

        Y_t = torch.tensor(Y_seq, device=device, dtype=dtype)
        R_t = torch.tensor(R_seq, device=device, dtype=dtype)
        rf_t = torch.tensor(rf_seq, device=device, dtype=dtype)
        return Y_t, R_t, rf_t


def rollout_offline_episode(
    policy: nn.Module,
    Y_seq: torch.Tensor,   # (B, H, k)
    R_seq: torch.Tensor,   # (B, H, n) simple returns
    rf_seq: torch.Tensor,  # (B, H) per-step rf (simple)
    cfg: OfflineEpisodeConfig,
) -> torch.Tensor:
    """
    Differentiable backtest over historical sequences.
    Returns per-path objective J (B,).
    """
    device = Y_seq.device
    dtype = Y_seq.dtype
    B, H, k = Y_seq.shape
    n = R_seq.shape[2]

    X = torch.ones(B, 1, device=device, dtype=dtype)
    pi_prev = torch.zeros(B, n, device=device, dtype=dtype)

    pen_l2 = torch.zeros(B, 1, device=device, dtype=dtype)
    pen_turn = torch.zeros(B, 1, device=device, dtype=dtype)

    for j in range(H):
        # time-to-go fraction in [0,1]
        tau = torch.full((B, 1), (H - j) / H, device=device, dtype=dtype)
        y = Y_seq[:, j, :]  # (B,k)
        pi = policy(tau, X, y)  # (B,n)
        pi = torch.clamp(pi, -cfg.clip_pi, cfg.clip_pi)

        if cfg.l2_weight_penalty and cfg.l2_weight_penalty > 0:
            pen_l2 = pen_l2 + torch.sum(pi * pi, dim=1, keepdim=True)

        r_next = R_seq[:, j, :]  # (B,n)
        rf_next = rf_seq[:, j].view(B, 1)  # (B,1)

        # portfolio gross return for the period: 1 + rf + pi*(r - rf)
        port_ret = rf_next + torch.sum(pi * (r_next - rf_next), dim=1, keepdim=True)  # (B,1)
        gross = 1.0 + port_ret
        gross = torch.clamp(gross, min=1e-12)

        # turnover penalty (optional, for OOS stability)
        if cfg.turnover_penalty and cfg.turnover_penalty > 0:
            pen_turn = pen_turn + torch.sum(torch.abs(pi - pi_prev), dim=1, keepdim=True)

        # transaction cost (linear in turnover)
        if cfg.tc_bps > 0:
            turnover = torch.sum(torch.abs(pi - pi_prev), dim=1, keepdim=True)
            tc = (cfg.tc_bps * 1e-4) * turnover
            gross = gross * (1.0 - tc)

        X = X * gross
        pi_prev = pi

    # terminal utility
    J = crra_utility(X.squeeze(1), gamma=cfg.gamma)

    # subtract regularization penalties (use averages per step)
    if cfg.l2_weight_penalty and cfg.l2_weight_penalty > 0:
        J = J - cfg.l2_weight_penalty * (pen_l2.squeeze(1) / max(H, 1))
    if cfg.turnover_penalty and cfg.turnover_penalty > 0:
        J = J - cfg.turnover_penalty * (pen_turn.squeeze(1) / max(H, 1))

    return J


@dataclass
class OfflineTrainConfig:
    iters: int = 2000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0
    print_every: int = 100


def train_offline_policy(
    policy: nn.Module,
    sampler: OfflineEpisodeSampler,
    ep_cfg: OfflineEpisodeConfig,
    train_cfg: OfflineTrainConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> None:
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    for it in range(train_cfg.iters):
        starts = sampler.sample_batch(train_cfg.batch_size)
        Y_seq, R_seq, rf_seq = sampler.batch_tensors(starts, device=device, dtype=dtype)

        J = rollout_offline_episode(policy, Y_seq, R_seq, rf_seq, ep_cfg)
        loss = -J.mean()  # maximize J

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.clip_grad_norm and train_cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=train_cfg.clip_grad_norm)
        opt.step()

        if (it % train_cfg.print_every) == 0 or it == train_cfg.iters - 1:
            with torch.no_grad():
                print(f"[OFFLINE-TRAIN] iter={it:5d}  J_mean={float(J.mean().cpu()): .6f}")


@torch.no_grad()
def evaluate_fixed_horizon_detailed(
    policy: nn.Module,
    y: np.ndarray,         # (T,k)
    r: np.ndarray,         # (T,n) simple returns
    rf_step: Optional[np.ndarray],  # (T,)
    start_idx: int,
    horizon_steps: int,
    tc_bps: float = 0.0,
    clip_pi: float = 5.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> dict:
    """
    Like evaluate_fixed_horizon, but returns detailed per-step logs.

    Returns a dict with:
      wealth:     (H+1,)
      port_ret:   (H,)
      rf:         (H,)
      pi:         (H, n)
      cash:       (H,) or None
      gross:      (H,) gross exposure sum(|pi|)
      net:        (H,) net exposure sum(pi)
      turnover:   (H,)
      tc:         (H,)
    """
    policy.eval()
    T = y.shape[0]
    H = horizon_steps
    if start_idx + H >= T:
        raise ValueError("Not enough data after start_idx for the requested horizon.")

    X = 1.0
    wealth = [X]
    pi_prev = None

    port_rets = []
    rf_list = []
    pi_list = []
    cash_list = []
    gross_list = []
    net_list = []
    turnover_list = []
    tc_list = []

    has_cash = hasattr(policy, "weights_and_cash")

    for j in range(H):
        tau = torch.tensor([[ (H - j) / H ]], device=device, dtype=dtype)
        x_t = torch.tensor([[X]], device=device, dtype=dtype)
        y_t = torch.tensor(y[start_idx + j : start_idx + j + 1], device=device, dtype=dtype)  # (1,k)

        if has_cash:
            pi_t, cash_t = policy.weights_and_cash(tau, x_t, y_t)
            pi = pi_t.cpu().numpy().reshape(-1)
            cash = float(cash_t.cpu().numpy().reshape(-1)[0]) if cash_t is not None else None
        else:
            pi = policy(tau, x_t, y_t).cpu().numpy().reshape(-1)
            cash = None

        pi = np.clip(pi, -clip_pi, clip_pi)

        r_next = r[start_idx + j + 1]  # (n,)
        rf_next = float(rf_step[start_idx + j + 1]) if rf_step is not None else 0.0

        pr = rf_next + float(np.dot(pi, (r_next - rf_next)))
        gross = float(np.sum(np.abs(pi)))
        net = float(np.sum(pi))

        # transaction cost
        if pi_prev is None:
            turnover = gross
        else:
            turnover = float(np.sum(np.abs(pi - pi_prev)))
        tc = (tc_bps * 1e-4) * turnover if tc_bps > 0.0 else 0.0

        # update wealth
        gross_ret = 1.0 + pr
        gross_ret = max(gross_ret, 1e-12)
        gross_ret *= (1.0 - tc)  # linear cost
        X = X * gross_ret

        wealth.append(X)
        port_rets.append(pr)
        rf_list.append(rf_next)
        pi_list.append(pi)
        cash_list.append(cash)
        gross_list.append(gross)
        net_list.append(net)
        turnover_list.append(turnover)
        tc_list.append(tc)

        pi_prev = pi

    out = {
        "wealth": np.asarray(wealth, dtype=float),
        "port_ret": np.asarray(port_rets, dtype=float),
        "rf": np.asarray(rf_list, dtype=float),
        "pi": np.asarray(pi_list, dtype=float),
        "cash": None if all(c is None for c in cash_list) else np.asarray([0.0 if c is None else c for c in cash_list], dtype=float),
        "gross": np.asarray(gross_list, dtype=float),
        "net": np.asarray(net_list, dtype=float),
        "turnover": np.asarray(turnover_list, dtype=float),
        "tc": np.asarray(tc_list, dtype=float),
    }
    return out

@torch.no_grad()
def evaluate_fixed_horizon(
    policy: nn.Module,
    y: np.ndarray,         # (T,k)
    r: np.ndarray,         # (T,n) simple returns
    rf_step: Optional[np.ndarray],  # (T,)
    start_idx: int,
    horizon_steps: int,
    tc_bps: float = 0.0,
    clip_pi: float = 5.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one fixed-horizon backtest from start_idx for horizon_steps.

    Returns:
      wealth_path: (H+1,) including initial wealth=1
      port_ret:    (H,) per-step portfolio returns (simple)
    """
    policy.eval()
    T = y.shape[0]
    H = horizon_steps
    # Need r at indices start_idx+1 ... start_idx+H (inclusive), so require start_idx+H < T
    if start_idx + H >= T:
        raise ValueError("Not enough data after start_idx for the requested horizon.")

    X = 1.0
    wealth = [X]
    pi_prev = None
    port_rets = []

    for j in range(H):
        tau = torch.tensor([[ (H - j) / H ]], device=device, dtype=dtype)
        x_t = torch.tensor([[X]], device=device, dtype=dtype)
        y_t = torch.tensor(y[start_idx + j : start_idx + j + 1], device=device, dtype=dtype)  # (1,k)

        pi = policy(tau, x_t, y_t).cpu().numpy().reshape(-1)
        pi = np.clip(pi, -clip_pi, clip_pi)

        r_next = r[start_idx + j + 1]  # (n,)
        rf_next = float(rf_step[start_idx + j + 1]) if rf_step is not None else 0.0

        pr = rf_next + float(np.dot(pi, (r_next - rf_next)))
        gross = 1.0 + pr

        if tc_bps > 0.0:
            if pi_prev is None:
                turnover = np.sum(np.abs(pi))
            else:
                turnover = np.sum(np.abs(pi - pi_prev))
            tc = (tc_bps * 1e-4) * turnover
            gross *= (1.0 - tc)

        X = X * gross
        wealth.append(X)
        port_rets.append(pr)
        pi_prev = pi

    return np.asarray(wealth, dtype=float), np.asarray(port_rets, dtype=float)
