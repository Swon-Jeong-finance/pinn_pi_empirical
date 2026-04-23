from __future__ import annotations

from pathlib import Path

from dynalloc_v2.cli import load_config
from dynalloc_v2.experiments import _resolve_eval_tau

ROOT = Path(__file__).resolve().parents[1]


def _base_cfg():
    return load_config(ROOT / 'configs' / 'demo_synthetic_ppgdpo.yaml')


def test_eval_tau_mode_test_remaining():
    cfg = _base_cfg()
    cfg.pipinn.eval_tau_mode = 'test_remaining'
    assert _resolve_eval_tau(cfg, step_idx=0, eval_horizon_months=240, last_refit_step=0) == 240.0
    assert _resolve_eval_tau(cfg, step_idx=1, eval_horizon_months=240, last_refit_step=0) == 239.0


def test_eval_tau_mode_maturity_declining():
    cfg = _base_cfg()
    cfg.pipinn.eval_tau_mode = 'maturity_declining'
    cfg.pipinn.eval_tau_maturity_years = 2
    assert _resolve_eval_tau(cfg, step_idx=0, eval_horizon_months=240, last_refit_step=0) == 24.0
    assert _resolve_eval_tau(cfg, step_idx=1, eval_horizon_months=240, last_refit_step=0) == 23.0


def test_eval_tau_mode_maturity_constant():
    cfg = _base_cfg()
    cfg.pipinn.eval_tau_mode = 'maturity_constant'
    cfg.pipinn.eval_tau_maturity_years = 2
    assert _resolve_eval_tau(cfg, step_idx=0, eval_horizon_months=240, last_refit_step=0) == 24.0
    assert _resolve_eval_tau(cfg, step_idx=5, eval_horizon_months=240, last_refit_step=0) == 24.0


def test_eval_tau_mode_reset_on_refit():
    cfg = _base_cfg()
    cfg.pipinn.eval_tau_mode = 'maturity_declining'
    cfg.pipinn.eval_tau_maturity_years = 2
    cfg.pipinn.eval_tau_reset_on_refit = True
    assert _resolve_eval_tau(cfg, step_idx=15, eval_horizon_months=240, last_refit_step=12) == 21.0


def test_eval_tau_mode_reset_on_refit_starts_from_maturity_on_refit_month():
    cfg = _base_cfg()
    cfg.pipinn.eval_tau_mode = 'maturity_declining'
    cfg.pipinn.eval_tau_maturity_years = 3
    cfg.pipinn.eval_tau_reset_on_refit = True

    seq = []
    last_refit_step = 0
    refit_every = 12
    for i in range(13):
        refit_now = (i == 0) or (i % refit_every == 0)
        tau_refit_anchor = i if refit_now else last_refit_step
        seq.append(_resolve_eval_tau(cfg, step_idx=i, eval_horizon_months=240, last_refit_step=tau_refit_anchor))
        if refit_now:
            last_refit_step = i

    assert seq[:12] == [36.0, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0]
    assert seq[12] == 36.0
