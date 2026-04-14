from __future__ import annotations

from pathlib import Path

import yaml

from dynalloc_v2.legacy_suite import build_v1_lane_suite


def test_build_v1_lane_suite_manifest(tmp_path: Path, monkeypatch):
    v1_root = tmp_path / 'v1'
    selected_dir = v1_root / 'outputs' / 'lane'
    selected_dir.mkdir(parents=True)
    (selected_dir / 'selected_spec.yaml').write_text(
        yaml.safe_dump({'selected_specs': ['spec_a', 'spec_b', 'spec_c']}, sort_keys=False),
        encoding='utf-8',
    )

    called = []

    def fake_bridge(**kwargs):
        rank = kwargs['selected_rank']
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        config_yaml = out_dir / 'config_empirical_ppgdpo_apt.yaml'
        metadata_yaml = out_dir / 'bridge_metadata.yaml'
        returns_csv = out_dir / 'returns_panel.csv'
        states_csv = out_dir / 'states_panel.csv'
        factors_csv = out_dir / 'factors_panel.csv'
        for p in [returns_csv, states_csv, factors_csv]:
            p.write_text('date,x\n', encoding='utf-8')
        config_yaml.write_text(
            'project:\n'
            '  name: demo\n'
            '  output_dir: outputs/demo\n'
            'experiment:\n'
            '  kind: ppgdpo\n'
            'data:\n'
            '  mode: synthetic\n'
            '  synthetic:\n'
            '    periods: 20\n'
            '    assets: 4\n'
            '    factors: 2\n'
            '    seed: 17\n'
            'split:\n'
            '  train_start: 2000-01-31\n'
            '  test_start: 2001-01-31\n'
            '  end_date: 2001-12-31\n'
            '  refit_every: 12\n'
            '  min_train_months: 12\n'
            'state:\n'
            '  columns: [slow_value, fast_vol]\n'
            'factor_model:\n'
            '  extractor: provided\n'
            '  provided_factor_columns: [MKT, VALUE]\n'
            'mean_model:\n'
            '  kind: factor_apt\n'
            'comparison:\n'
            '  cross_modes: [estimated, zero]\n',
            encoding='utf-8',
        )
        metadata_yaml.write_text(yaml.safe_dump({'spec': kwargs['spec']}, sort_keys=False), encoding='utf-8')
        called.append(rank)
        from types import SimpleNamespace
        return SimpleNamespace(
            out_dir=out_dir,
            returns_csv=returns_csv,
            states_csv=states_csv,
            factors_csv=factors_csv,
            config_yaml=config_yaml,
            metadata_yaml=metadata_yaml,
        )

    monkeypatch.setattr('dynalloc_v2.legacy_suite.build_v1_lane_bundle', fake_bridge)
    artifacts = build_v1_lane_suite(
        v1_root=v1_root,
        config_stem='lane',
        out_dir=tmp_path / 'suite',
        top_k=2,
        split_profile_override='cv2006_final20y',
        split_test_start_override='2007-01-01',
        split_end_date_override='2024-12-31',
    )
    manifest = yaml.safe_load(artifacts.manifest_yaml.read_text(encoding='utf-8'))
    assert artifacts.entry_count == 2
    assert called == [1, 2]
    assert [e['spec'] for e in manifest['entries']] == ['spec_a', 'spec_b']
    assert manifest['split_profile_override'] == 'cv2006_final20y'
    assert manifest['split_overrides']['test_start'] == '2007-01-01'
