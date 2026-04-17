import numpy as np
import pandas as pd

from dynalloc_v2.transition import estimate_return_state_cross, fit_state_transition


def _toy_panels(n: int = 80, m_assets: int = 4, m_states: int = 2):
    rng = np.random.default_rng(7)
    idx = pd.date_range('2010-01-31', periods=n, freq='M')
    states_t = pd.DataFrame(rng.normal(size=(n, m_states)), index=idx, columns=[f's{i+1}' for i in range(m_states)])
    innov = rng.normal(scale=0.15, size=(n, m_states))
    states_tp1 = pd.DataFrame(0.8 * states_t.to_numpy() + innov, index=idx, columns=states_t.columns)
    ret_noise = rng.normal(scale=0.05, size=(n, m_assets))
    returns_tp1 = pd.DataFrame(
        0.1 * states_t.to_numpy() @ rng.normal(size=(m_states, m_assets)) + ret_noise,
        index=idx,
        columns=[f'a{i+1}' for i in range(m_assets)],
    )
    return states_t, states_tp1, returns_tp1


def test_dynamic_cross_dcc_updates_and_shapes():
    states_t, states_tp1, returns_tp1 = _toy_panels()
    transition = fit_state_transition(states_t, states_tp1)
    mean_pred = np.zeros((len(states_t), returns_tp1.shape[1]), dtype=float)
    est = estimate_return_state_cross(
        returns_tp1=returns_tp1,
        returns_mean_pred=mean_pred,
        states_t=states_t,
        states_tp1=states_tp1,
        transition=transition,
        dynamic_cross_kind='dcc',
    )
    assert est.dynamic_model is not None
    cross0 = est.dynamic_model.current_cross_covariance()
    assert cross0.shape == (returns_tp1.shape[1], states_t.shape[1])
    est.dynamic_model.update_with_realized(
        realized_return=returns_tp1.iloc[-1].to_numpy(dtype=float),
        predicted_return_mean=np.zeros(returns_tp1.shape[1], dtype=float),
        realized_state=states_tp1.iloc[-1].to_numpy(dtype=float),
        predicted_state=transition.predict(states_t.iloc[-1]).to_numpy(dtype=float),
    )
    cross1 = est.dynamic_model.current_cross_covariance()
    assert cross1.shape == cross0.shape
    assert not np.allclose(cross0, cross1)


def test_dynamic_cross_variants_supported():
    states_t, states_tp1, returns_tp1 = _toy_panels()
    transition = fit_state_transition(states_t, states_tp1)
    mean_pred = np.zeros((len(states_t), returns_tp1.shape[1]), dtype=float)
    for kind in ['adcc', 'regime_dcc']:
        est = estimate_return_state_cross(
            returns_tp1=returns_tp1,
            returns_mean_pred=mean_pred,
            states_t=states_t,
            states_tp1=states_tp1,
            transition=transition,
            dynamic_cross_kind=kind,
        )
        assert est.dynamic_model is not None
        cross = est.dynamic_model.current_cross_covariance()
        assert cross.shape == (returns_tp1.shape[1], states_t.shape[1])