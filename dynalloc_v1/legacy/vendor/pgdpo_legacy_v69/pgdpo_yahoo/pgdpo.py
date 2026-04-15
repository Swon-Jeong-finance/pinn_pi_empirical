"""
Baseline PG-DPO warm-up training (Algorithm 1 style).

We implement a **vectorized** version:
- sample a batch of initial states (X0, Y0)
- simulate M paths in parallel
- maximize mean objective via Adam (gradient ascent on J == minimize -J)

For empirical walk-forward, you can warm-start the network between windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .simulator import SimConfig, TorchMarketModel, rollout_paths


@dataclass
class TrainConfig:
    iters: int = 300
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0
    print_every: int = 50


class InitialStateSampler:
    """
    Empirical default:
    - sample Y0 from historical states within a window
    - X0 fixed to 1
    """
    def __init__(self, states_window: np.ndarray):
        assert states_window.ndim == 2
        self.Y = states_window
        self.k = self.Y.shape[1]

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self.Y.shape[0], size=batch_size)
        y0 = torch.tensor(self.Y[idx], device=device, dtype=dtype)
        x0 = torch.ones(batch_size, 1, device=device, dtype=dtype)
        return x0, y0


def train_pgdpo(
    market: TorchMarketModel,
    policy: nn.Module,
    cons_policy: Optional[nn.Module],
    init_sampler: InitialStateSampler,
    sim_cfg: SimConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> None:
    policy.train()
    if cons_policy is not None:
        cons_policy.train()

    params = list(policy.parameters()) + ([] if cons_policy is None else list(cons_policy.parameters()))
    opt = torch.optim.Adam(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    for it in range(train_cfg.iters):
        X0, Y0 = init_sampler.sample(train_cfg.batch_size, device=device, dtype=dtype)

        J = rollout_paths(
            model=market,
            policy=policy,
            cons_policy=cons_policy,
            X0=X0,
            Y0=Y0,
            cfg=sim_cfg,
        )
        loss = -J.mean()  # maximize J

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if train_cfg.clip_grad_norm is not None and train_cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=train_cfg.clip_grad_norm)

        opt.step()

        if (it % train_cfg.print_every) == 0 or it == train_cfg.iters - 1:
            with torch.no_grad():
                j_mean = float(J.mean().detach().cpu().item())
                print(f"[PG-DPO] iter={it:4d}  J_mean={j_mean: .6f}")
