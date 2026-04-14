from pathlib import Path

from dynalloc_v2.workspace import init_workspace


def test_init_workspace_creates_expected_subdirs(tmp_path: Path):
    payload = init_workspace(root=tmp_path / 'shared')
    assert payload['root'].exists()
    assert payload['cache_french'].exists()
    assert payload['cache_fred'].exists()
    assert payload['base_bundles'].exists()
    assert payload['experiments'].exists()
