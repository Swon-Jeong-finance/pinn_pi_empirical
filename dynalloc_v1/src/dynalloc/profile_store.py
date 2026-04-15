from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

PROFILE_SECTIONS = (
    "universe",
    "macro",
    "model",
    "constraints",
    "method",
    "split",
    "selection",
    "comparison",
)


@dataclass(frozen=True)
class ProfileStore:
    root: Path
    source: str

    def section_dir(self, section: str) -> Path:
        if section not in PROFILE_SECTIONS:
            raise KeyError(f"Unknown profile section '{section}'.")
        path = self.root / section
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"Profile section '{section}' does not exist under {self.root}."
            )
        return path

    def names(self, section: str) -> list[str]:
        return sorted(path.stem for path in self.section_dir(section).glob("*.yaml"))

    def load(self, section: str, name: str) -> dict[str, Any]:
        path = self.section_dir(section) / f"{name}.yaml"
        if not path.exists():
            available = ", ".join(self.names(section))
            raise KeyError(
                f"Unknown {section} profile '{name}'. Available: {available}"
            )
        payload = yaml.safe_load(path.read_text()) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Profile file must contain a mapping: {path}")
        return payload


def _candidate_profile_roots(start: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    env_root = os.environ.get("DYNALLOC_PROFILE_ROOT")
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path not in seen:
            roots.append(path)
            seen.add(path)

    def add_ancestors(base: Path) -> None:
        base = base.resolve()
        candidates = [base] if base.is_dir() else [base.parent]
        for current in candidates:
            for parent in [current, *current.parents]:
                candidate = parent / "profiles"
                if candidate not in seen:
                    roots.append(candidate)
                    seen.add(candidate)

    if start is not None:
        add_ancestors(start)
    add_ancestors(Path.cwd())

    try:
        pkg_root = resources.files("dynalloc").joinpath("profile_data")
        pkg_path = Path(str(pkg_root))
        if pkg_path not in seen:
            roots.append(pkg_path)
            seen.add(pkg_path)
    except Exception:
        pass

    return roots


def discover_profile_store(start: Path | None = None) -> ProfileStore:
    for candidate in _candidate_profile_roots(start):
        if candidate.exists() and candidate.is_dir():
            sections_present = all((candidate / section).exists() for section in PROFILE_SECTIONS)
            if sections_present:
                source = "packaged-defaults"
                if candidate.name == "profiles":
                    source = "workspace-profiles"
                if os.environ.get("DYNALLOC_PROFILE_ROOT") and candidate == Path(os.environ["DYNALLOC_PROFILE_ROOT"]).expanduser().resolve():
                    source = "env-profile-root"
                return ProfileStore(root=candidate, source=source)
    searched = "\n".join(str(path) for path in _candidate_profile_roots(start))
    raise FileNotFoundError(
        "Could not locate a dynalloc profile store. Searched:\n" + searched
    )
