
from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml

from .profile_store import discover_profile_store
from .schema import (
    BondAssetConfig,
    ComparisonConfig,
    ConstraintConfig,
    RankSweepConfig,
    EvaluationConfig,
    ExperimentMeta,
    MacroConfig,
    MethodConfig,
    ModelConfig,
    RawExperimentConfig,
    ResolvedExperimentConfig,
    RuntimeConfig,
    SelectionConfig,
    SplitConfig,
    TrainingConfig,
    UniverseConfig,
)

T = TypeVar("T")


def _clean_override_values(override_model: Any) -> dict[str, Any]:
    return override_model.model_dump(exclude_none=True, exclude={"profile"})


def _merge_profile(
    *,
    config_cls: type[T],
    override_model: Any,
    profile_store: Any | None = None,
    section: str | None = None,
) -> T:
    base_values: dict[str, Any] = {}
    if profile_store is not None and section is not None:
        profile_name = getattr(override_model, "profile", None)
        if profile_name:
            base_values = profile_store.load(section, profile_name)
    merged = {**base_values, **_clean_override_values(override_model)}
    return config_cls(**merged)


def load_raw_config(path: str | Path) -> RawExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise TypeError("Config file must contain a mapping at the top level.")
    return RawExperimentConfig.model_validate(payload)


def resolve_config(raw: RawExperimentConfig, *, profile_store: Any) -> ResolvedExperimentConfig:
    universe = _merge_profile(
        config_cls=UniverseConfig,
        override_model=raw.universe,
        profile_store=profile_store,
        section="universe",
    )
    macro = _merge_profile(
        config_cls=MacroConfig,
        override_model=raw.macro,
        profile_store=profile_store,
        section="macro",
    )
    model = _merge_profile(
        config_cls=ModelConfig,
        override_model=raw.model,
        profile_store=profile_store,
        section="model",
    )
    constraints = _merge_profile(
        config_cls=ConstraintConfig,
        override_model=raw.constraints,
        profile_store=profile_store,
        section="constraints",
    )
    method = _merge_profile(
        config_cls=MethodConfig,
        override_model=raw.method,
        profile_store=profile_store,
        section="method",
    )
    split = _merge_profile(
        config_cls=SplitConfig,
        override_model=raw.split,
        profile_store=profile_store,
        section="split",
    )
    selection = _merge_profile(
        config_cls=SelectionConfig,
        override_model=raw.selection,
        profile_store=profile_store,
        section="selection",
    )
    comparison = _merge_profile(
        config_cls=ComparisonConfig,
        override_model=raw.comparison,
        profile_store=profile_store,
        section="comparison",
    )
    rank_sweep = _merge_profile(
        config_cls=RankSweepConfig,
        override_model=raw.rank_sweep,
    )
    training = _merge_profile(
        config_cls=TrainingConfig,
        override_model=raw.training,
    )
    evaluation = _merge_profile(
        config_cls=EvaluationConfig,
        override_model=raw.evaluation,
    )
    runtime = _merge_profile(
        config_cls=RuntimeConfig,
        override_model=raw.runtime,
    )

    if not constraints.enabled and runtime.backend in {"legacy_bridge", "native_stage3a", "native_stage3b", "native_stage4", "native_stage5", "native_stage6", "native_stage7"}:
        raise ValueError(
            "constraints.enabled=False is reserved for a future unconstrained backend. "
            "The current empirical runners still expect the constrained runner."
        )

    if runtime.backend in {"legacy_bridge", "native_stage3a", "native_stage3b", "native_stage4", "native_stage5", "native_stage6", "native_stage7"} and runtime.legacy_entrypoint is None:
        raise ValueError(
            f"runtime.backend='{runtime.backend}' requires runtime.legacy_entrypoint."
        )

    return ResolvedExperimentConfig(
        experiment=raw.experiment,
        universe=universe,
        macro=macro,
        model=model,
        constraints=constraints,
        method=method,
        split=split,
        selection=selection,
        comparison=comparison,
        rank_sweep=rank_sweep,
        training=training,
        evaluation=evaluation,
        runtime=runtime,
        profile_source=profile_store.source,
        profile_root=str(profile_store.root),
    )


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def load_and_resolve(path: str | Path) -> ResolvedExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = load_raw_config(config_path)
    profile_store = discover_profile_store(config_path.parent)
    resolved = resolve_config(raw, profile_store=profile_store)

    profile_root = Path(resolved.profile_root)
    if profile_root.name == "profiles":
        project_root = profile_root.parent
    else:
        project_root = config_path.parent

    resolved.experiment = ExperimentMeta(
        name=resolved.experiment.name,
        output_dir=_resolve_path(project_root, resolved.experiment.output_dir)
        or resolved.experiment.output_dir,
        notes=resolved.experiment.notes,
    )
    resolved.runtime.legacy_entrypoint = _resolve_path(
        project_root, resolved.runtime.legacy_entrypoint
    )
    resolved.runtime.legacy_workdir = _resolve_path(
        project_root, resolved.runtime.legacy_workdir
    )

    if resolved.runtime.backend == "legacy_bridge" and resolved.runtime.legacy_workdir is None:
        entrypoint = resolved.runtime.legacy_entrypoint_path
        if entrypoint is not None:
            resolved.runtime.legacy_workdir = str(entrypoint.parent)

    if resolved.universe.include_bond and resolved.universe.bond_assets:
        rebased: list[BondAssetConfig] = []
        for asset in resolved.universe.bond_assets:
            rebased.append(
                BondAssetConfig(
                    name=asset.name,
                    source=asset.source,
                    csv=_resolve_path(project_root, asset.csv) if asset.csv else None,
                    ret_col=asset.ret_col,
                    date_col=asset.date_col,
                    series_id=asset.series_id,
                )
            )
        resolved.universe.bond_assets = rebased
    elif resolved.universe.include_bond and resolved.universe.bond_source == "crsp_csv":
        resolved.universe.bond_csv = _resolve_path(project_root, resolved.universe.bond_csv) or resolved.universe.bond_csv

    return resolved


def dump_yaml(data: dict[str, object]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
