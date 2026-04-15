"""
A minimal walk-forward / receding-horizon backtest.

Design:
- At each rebalance date:
  1) fit market model on rolling window
  2) warm-up train PG-DPO policy (optional)
  3) compute action by either:
      - myopic Merton
      - warm-up policy output
      - P-PGDPO projected policy
  4) apply action to realized returns for the next period to update wealth

This is intentionally minimal; you can extend with transaction costs, constraints, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .estimators import fit_market_model
from .pgdpo import InitialStateSampler, TrainConfig, train_pgdpo
from .policy import ConsumptionPolicy, ConsumptionConfig, PolicyConfig, PortfolioPolicy
from .projection import DeployConfig, project_p_pgdpo
from .simulator import SimConfig, TorchMarketModel


@dataclass
class BacktestConfig:
    window_years: float = 5.0
    eval_start_date: Optional[str] = None   # report only from this date (wealth resets to 1 at start)
    rebalance_every: int = 1       # in dataset steps (weekly default)
    strategy: Literal["myopic_merton", "pgdpo_policy", "p_pgdpo_projected"] = "p_pgdpo_projected"
    gamma: float = 5.0
    # training/projection
    warmup_iters: int = 200
    mc_rollouts: int = 256
    horizon_years: float = 1.0
    steps_per_year: int = 52
    # constraints
    clip_pi: float = 5.0
    # transaction cost
    tc_bps: float = 5.0   # simple linear cost on turnover (bps of notional traded)


def myopic_merton_pi(model_np, y: np.ndarray, gamma: float) -> np.ndarray:
    """
    Myopic term only:
      pi = (1/gamma) Sigma^{-1} mu_excess
    (Note: Under CRRA and diffusion model, the exact myopic term is -Vx/(X Vxx) Sigma^{-1} mu_excess
     and for CRRA that factor equals 1/gamma.)
    """
    mu_ex = model_np.mu_excess_annual(y)  # (n,)
    # solve Sigma z = mu_ex
    z = np.linalg.solve(model_np.Sigma, mu_ex)
    return (1.0 / gamma) * z


def run_backtest(
    dataset,
    bt_cfg: BacktestConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> pd.DataFrame:
    dates = dataset.dates
    r_log = dataset.log_returns
    r_simple = dataset.simple_returns
    rf_annual = dataset.rf_annual
    Y_df = dataset.states

    periods_per_year = dataset.periods_per_year
    dt_years = 1.0 / periods_per_year

    window_len = int(bt_cfg.window_years * periods_per_year)
    if window_len < 30:
        raise ValueError("window_years too small; need enough data to estimate covariances.")

    n = r_log.shape[1]
    k = Y_df.shape[1]

    # init policy nets (warm-start across windows)
    pol_cfg = PolicyConfig(n_assets=n, n_states=k, hidden=64, depth=2, weight_transform="tanh_leverage", max_leverage=2.0)
    policy = PortfolioPolicy(pol_cfg).to(device=device, dtype=dtype)
    cons_policy = None

    sim_cfg = SimConfig(
        horizon_years=bt_cfg.horizon_years,
        steps_per_year=bt_cfg.steps_per_year,
        include_consumption=False,
        delta=0.0,
        kappa=1.0,
        gamma=bt_cfg.gamma,
    )

    train_cfg = TrainConfig(
        iters=bt_cfg.warmup_iters,
        batch_size=256,
        lr=3e-4,
        print_every=max(50, bt_cfg.warmup_iters // 4),
    )
    deploy_cfg = DeployConfig(
        mc_rollouts=bt_cfg.mc_rollouts,
        clip_pi=bt_cfg.clip_pi,
    )

    wealth = 1.0
    pi_prev = np.zeros(n)
    records = []
    eval_start = pd.Timestamp(bt_cfg.eval_start_date) if bt_cfg.eval_start_date else None
    started = (eval_start is None)


    for t_idx in range(window_len, len(dates) - 1, bt_cfg.rebalance_every):
        window_slice = slice(t_idx - window_len, t_idx)
        # Fit model on window
        model_np = fit_market_model(
            log_returns=r_log.iloc[window_slice],
            rf_annual=rf_annual.iloc[window_slice],
            states=Y_df.iloc[window_slice],
            ridge_alpha=1e-4,
        )
        market = TorchMarketModel.from_numpy(model_np, device=device, dtype=dtype)

        # Warm-up training (optional for myopic)
        if bt_cfg.strategy in ["pgdpo_policy", "p_pgdpo_projected"]:
            init_sampler = InitialStateSampler(states_window=Y_df.iloc[window_slice].values)
            train_pgdpo(
                market=market,
                policy=policy,
                cons_policy=cons_policy,
                init_sampler=init_sampler,
                sim_cfg=sim_cfg,
                train_cfg=train_cfg,
                device=device,
                dtype=dtype,
            )

        # Current state
        y0 = Y_df.iloc[t_idx].values.astype(float)
        # If evaluation start is set, reset wealth at the first evaluation step.
        next_date = dates[t_idx + 1]
        if (not started) and (eval_start is not None) and (next_date >= eval_start):
            wealth = 1.0
            pi_prev = np.zeros(n)
            started = True

        if bt_cfg.strategy == "myopic_merton":
            pi = myopic_merton_pi(model_np, y=y0, gamma=bt_cfg.gamma)
        elif bt_cfg.strategy == "pgdpo_policy":
            with torch.no_grad():
                tau = torch.tensor([[1.0]], device=device, dtype=dtype)
                x_tensor = torch.tensor([[wealth]], device=device, dtype=dtype)
                y_tensor = torch.tensor(y0[None, :], device=device, dtype=dtype)
                pi = policy(tau, x_tensor, y_tensor).cpu().numpy().reshape(-1)
        elif bt_cfg.strategy == "p_pgdpo_projected":
            pi, _, info = project_p_pgdpo(
                market=market,
                policy=policy,
                cons_policy=cons_policy,
                X=wealth,
                Y=y0,
                t_years=0.0,
                sim_cfg=sim_cfg,
                deploy_cfg=deploy_cfg,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown strategy: {bt_cfg.strategy}")

        pi = np.clip(pi, -bt_cfg.clip_pi, bt_cfg.clip_pi)

        # Apply to realized returns next period
        r_next = r_simple.iloc[t_idx + 1].values.astype(float)  # (n,)
        rf_next = float(rf_annual.iloc[t_idx + 1] * dt_years)
        # portfolio simple return
        port_ret = rf_next + np.dot(pi, (r_next - rf_next))
        wealth_next = wealth * (1.0 + port_ret)

        # transaction cost (linear in turnover)
        turnover = np.sum(np.abs(pi - pi_prev))
        tc = (bt_cfg.tc_bps * 1e-4) * turnover  # bps -> decimal
        wealth_next *= (1.0 - tc)

        if started:
            records.append({
                "date": dates[t_idx + 1],
                "wealth": wealth_next,
                "port_ret": port_ret,
                "rf": rf_next,
                "turnover": turnover,
                "tc": tc,
            })
        wealth = wealth_next
        pi_prev = pi

    out = pd.DataFrame(records).set_index("date")
    return out
