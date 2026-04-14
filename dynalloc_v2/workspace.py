from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_SHARED_ROOT_TEXT = os.environ.get("DYNALLOC_SHARED_ROOT", "~/empricial_test/shared")


def shared_root(explicit_root: str | Path | None = None) -> Path:
    root = Path(explicit_root if explicit_root is not None else _DEFAULT_SHARED_ROOT_TEXT)
    return root.expanduser().resolve()


def shared_subdir(name: str, *, root: str | Path | None = None) -> Path:
    path = shared_root(root) / str(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_french_cache_dir(*, root: str | Path | None = None) -> Path:
    return shared_subdir('cache_french', root=root)


def default_fred_cache_dir(*, root: str | Path | None = None) -> Path:
    return shared_subdir('cache_fred', root=root)


def default_base_bundle_root(*, root: str | Path | None = None) -> Path:
    return shared_subdir('base_bundles', root=root)


def default_experiments_root(*, root: str | Path | None = None) -> Path:
    return shared_subdir('experiments', root=root)


def init_workspace(*, root: str | Path | None = None) -> dict[str, Path]:
    return {
        'root': shared_root(root),
        'cache_french': default_french_cache_dir(root=root),
        'cache_fred': default_fred_cache_dir(root=root),
        'base_bundles': default_base_bundle_root(root=root),
        'experiments': default_experiments_root(root=root),
    }
