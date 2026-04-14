from dynalloc_v2.native_selection import _normalize_rolling_oos_window_grid, _parse_rerank_covariance_models, _expand_stage2_model_specs
from dynalloc_v2.oos_protocols import protocol_spec


def test_default_rolling_grid_is_20y_only():
    assert _normalize_rolling_oos_window_grid(None) == [240]


def test_dynamic_rolling_protocol_descriptions_use_warm_start_language():
    spec = protocol_spec("rolling240m_annual")
    assert spec.rolling_train_months == 240
    assert "warm-start" in spec.description


def test_stage2_model_variants_limit_gating_to_regime_dcc():
    specs = _expand_stage2_model_specs(_parse_rerank_covariance_models(['dcc', 'regime_dcc']))
    labels = [spec.label for spec in specs]
    assert labels == [
        'dcc',
        'dcc__zero_cross',
        'regime_dcc',
        'regime_dcc__zero_cross',
        'regime_dcc__gated_cross',
    ]
