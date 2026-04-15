from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import ResolvedExperimentConfig
from .selection import choose_selected_spec, choose_selected_specs, resolve_candidate_specs, write_selected_spec_artifact

_SECRET_FLAGS = {"--fred_api_key"}


@dataclass(frozen=True)
class LegacyRunSpec:
    command: list[str]
    env: dict[str, str]
    cwd: str | None = None
    label: str | None = None

    def shell_command(self, *, redact_secrets: bool = True) -> str:
        cmd = list(self.command)
        if redact_secrets:
            for idx, token in enumerate(cmd[:-1]):
                if token in _SECRET_FLAGS:
                    cmd[idx + 1] = "***REDACTED***"
        rendered = " ".join(shlex.quote(token) for token in cmd)
        if self.cwd:
            return f"cd {shlex.quote(self.cwd)} && {rendered}"
        return rendered


@dataclass(frozen=True)
class LegacyPreflightReport:
    entrypoint: str
    cwd: str
    fred_api_key_env: str | None
    fred_api_key_present: bool
    help_ok: bool
    help_returncode: int
    stderr_tail: str | None = None


@dataclass(frozen=True)
class Stage2RunResult:
    selected_spec: str | None
    selected_specs: list[str]
    completed_phases: list[str]


class PhaseError(RuntimeError):
    pass


def _resolve_python(config: ResolvedExperimentConfig) -> str:
    if config.runtime.legacy_python:
        return str(config.runtime.legacy_python)
    return sys.executable


def _append_bool_flag(
    cmd: list[str], *, enabled: bool, true_flag: str, false_flag: str | None = None
) -> None:
    if enabled:
        cmd.append(true_flag)
    elif false_flag is not None:
        cmd.append(false_flag)


def _ensure_fred_key(config: ResolvedExperimentConfig) -> tuple[dict[str, str], str | None]:
    env = os.environ.copy()
    fred_value = None
    if config.runtime.fred_api_key_env:
        fred_value = env.get(config.runtime.fred_api_key_env)
    return env, fred_value


def _append_bond_args(cmd: list[str], config: ResolvedExperimentConfig) -> None:
    bond_assets = config.universe.normalized_bond_assets
    bond_source = config.universe.bond_source_kind or str(config.universe.bond_source)
    cmd.extend(["--bond_source", str(bond_source)])
    if bond_assets:
        first = bond_assets[0]
        cmd.extend(["--bond_name", str(first.name)])
        if first.source == "crsp_csv":
            cmd.extend(["--bond_csv", str(first.csv)])
            cmd.extend(["--bond_ret_col", str(first.ret_col)])
            cmd.extend([
                "--bond_csv_specs",
                ",".join(f"{asset.name}={asset.csv}@{asset.ret_col}" for asset in bond_assets),
            ])
        else:
            cmd.extend(["--bond_series_id", str(first.series_id)])
            cmd.extend([
                "--bond_fred_specs",
                ",".join(f"{asset.name}={asset.series_id}" for asset in bond_assets),
            ])
    else:
        cmd.extend(["--bond_csv", str(config.universe.bond_csv)])
        cmd.extend(["--bond_name", str(config.universe.bond_name)])
        cmd.extend(["--bond_series_id", str(config.universe.bond_series_id)])
        cmd.extend(["--bond_ret_col", str(config.universe.bond_ret_col)])



def _base_command(
    config: ResolvedExperimentConfig,
    *,
    out_dir: Path,
    state_spec: str | None = None,
) -> LegacyRunSpec:
    entrypoint = config.runtime.legacy_entrypoint_path
    if entrypoint is None:
        raise ValueError("runtime.legacy_entrypoint must be set for legacy_bridge.")

    cwd = config.runtime.legacy_workdir_path or entrypoint.parent
    env, fred_value = _ensure_fred_key(config)

    cmd = [_resolve_python(config), str(entrypoint)]
    if fred_value:
        cmd.extend(["--fred_api_key", fred_value])
    cmd.extend(["--out_dir", str(out_dir.resolve())])
    cmd.extend(["--seed", str(config.runtime.seed)])
    cmd.extend(["--device", str(config.runtime.device)])
    cmd.extend(["--dtype", str(config.runtime.dtype)])

    cmd.extend(["--asset_universe", config.universe.asset_universe])
    if config.universe.custom_risky_assets:
        cmd.extend([
            "--custom_risky_assets_json",
            json.dumps([asset.model_dump() for asset in config.universe.custom_risky_assets], separators=(",", ":")),
        ])
    _append_bool_flag(
        cmd,
        enabled=config.universe.include_bond,
        true_flag="--include_bond",
        false_flag="--no_bond",
    )
    _append_bond_args(cmd, config)

    effective_state_spec = state_spec or config.model.state_spec
    cmd.extend(["--state_spec", effective_state_spec])
    cmd.extend(["--latent_k", str(config.model.latent_k)])
    cmd.extend(["--pls_horizon", str(config.model.pls_horizon)])
    cmd.extend(["--pls_smooth_span", str(config.model.pls_smooth_span)])
    cmd.extend(["--cross_mode", str(config.model.cross_mode)])
    cmd.extend(["--zero_cross_policy_proxy", str(config.model.zero_cross_policy_proxy)])

    cmd.extend(["--risky_cap", str(config.constraints.risky_cap)])
    cmd.extend(["--cash_floor", str(config.constraints.cash_floor)])
    if config.constraints.per_asset_cap is not None:
        cmd.extend(["--per_asset_cap", str(config.constraints.per_asset_cap)])
    if config.constraints.allow_short:
        cmd.append("--allow_short")
        cmd.extend(["--short_floor", str(config.constraints.short_floor)])

    cmd.extend(["--end_date", str(config.split.end_date)])
    if config.split.train_pool_start:
        cmd.extend(["--train_pool_start", str(config.split.train_pool_start)])
    _append_bool_flag(
        cmd,
        enabled=config.split.force_common_calendar,
        true_flag="--force_common_calendar",
        false_flag="--no_force_common_calendar",
    )
    cmd.extend(["--common_start_mode", str(config.split.common_start_mode)])

    if config.runtime.pass_through_args:
        cmd.extend(config.runtime.pass_through_args)

    return LegacyRunSpec(command=cmd, env=env, cwd=str(cwd))


def build_selection_run_spec(config: ResolvedExperimentConfig) -> LegacyRunSpec:
    if not config.selection.enabled:
        raise PhaseError("Selection phase is disabled in this config.")
    candidate_specs = resolve_candidate_specs(config)
    out_dir = config.selection_output_dir
    spec = _base_command(config, out_dir=out_dir, state_spec=candidate_specs[0])
    cmd = list(spec.command)
    cmd.extend(["--eval_mode", "v57"])
    cmd.extend(["--cv_folds", str(config.split.selection_cv_folds)])
    cmd.extend(["--cv_min_train_months", str(config.split.selection_min_train_months)])
    if config.split.selection_val_months > 0:
        cmd.extend(["--cv_val_months", str(config.split.selection_val_months)])
    cmd.append("--selection_only")
    cmd.append("--compare_specs")
    cmd.append("--specs")
    cmd.extend(candidate_specs)
    cmd.extend(["--selection_top_k", str(config.selection.top_k)])
    cmd.extend(["--selection_window_mode", str(config.selection.window_mode)])
    cmd.extend(["--selection_rolling_window", str(config.selection.rolling_window)])
    cmd.extend(["--selection_return_baseline", str(config.selection.return_baseline)])
    cmd.extend(["--selection_state_baseline", str(config.selection.state_baseline)])
    cmd.extend(["--selection_return_floor", str(config.selection.return_floor)])
    cmd.extend(["--selection_state_q10_floor", str(config.selection.state_q10_floor)])
    cmd.extend(["--selection_cross_warn", str(config.selection.cross_warn)])
    cmd.extend(["--selection_cross_fail", str(config.selection.cross_fail)])
    cmd.extend(["--selection_score_mode", str(config.selection.score_mode)])
    cmd.extend([
        "--selection_score_ret_mean_weight",
        str(config.selection.score_ret_mean_weight),
    ])
    cmd.extend([
        "--selection_score_ret_q10_weight",
        str(config.selection.score_ret_q10_weight),
    ])
    cmd.extend([
        "--selection_score_state_mean_weight",
        str(config.selection.score_state_mean_weight),
    ])
    cmd.extend([
        "--selection_score_state_q10_weight",
        str(config.selection.score_state_q10_weight),
    ])
    if config.selection.guarded_only:
        cmd.append("--selection_guarded_only")
    return LegacyRunSpec(command=cmd, env=spec.env, cwd=spec.cwd, label="selection")


def build_comparison_run_spec(
    config: ResolvedExperimentConfig, *, selected_spec: str | None = None
) -> LegacyRunSpec:
    if not config.comparison.enabled:
        raise PhaseError("Comparison phase is disabled in this config.")
    final_spec = selected_spec
    if final_spec is None:
        final_spec = config.comparison.fixed_spec or config.model.state_spec
    out_dir = config.comparison_output_dir
    spec = _base_command(config, out_dir=out_dir, state_spec=final_spec)
    cmd = list(spec.command)
    cmd.extend(["--eval_mode", "legacy"])
    cmd.extend(["--eval_horizons", str(config.split.final_test_years)])

    policy_iters = config.training.iters if config.method.train_policy else 0
    cmd.extend(["--iters", str(policy_iters)])
    cmd.extend(["--batch_size", str(config.training.batch_size)])
    cmd.extend(["--lr", str(config.training.lr)])
    cmd.extend(["--gamma", str(config.training.gamma)])
    _append_bool_flag(
        cmd,
        enabled=config.training.residual_policy,
        true_flag="--residual_policy",
        false_flag="--no_residual_policy",
    )
    cmd.extend(["--residual_ridge", str(config.training.residual_ridge)])
    if config.training.residual_w_max is not None:
        cmd.extend(["--residual_w_max", str(config.training.residual_w_max)])
    if config.training.residual_detach_baseline:
        cmd.append("--residual_detach_baseline")

    _append_bool_flag(
        cmd,
        enabled=config.method.tc_sweep,
        true_flag="--tc_sweep",
        false_flag="--no_tc_sweep",
    )
    if config.method.tc_sweep and config.method.tc_bps:
        cmd.append("--tc_bps")
        cmd.extend(str(x) for x in config.method.tc_bps)

    _append_bool_flag(
        cmd,
        enabled=config.method.evaluate_ppgdpo,
        true_flag="--ppgdpo",
        false_flag="--no_ppgdpo",
    )
    cmd.extend(["--ppgdpo_mc", str(config.training.ppgdpo_mc)])
    cmd.extend(["--ppgdpo_subbatch", str(config.training.ppgdpo_subbatch)])
    cmd.extend(["--ppgdpo_seed", str(config.training.ppgdpo_seed)])
    if config.training.ppgdpo_L is not None:
        cmd.extend(["--ppgdpo_L", str(config.training.ppgdpo_L)])
    cmd.extend(["--ppgdpo_eps", str(config.training.ppgdpo_eps)])
    cmd.extend(["--ppgdpo_ridge", str(config.training.ppgdpo_ridge)])
    cmd.extend(["--ppgdpo_tau", str(config.training.ppgdpo_tau)])
    cmd.extend(["--ppgdpo_armijo", str(config.training.ppgdpo_armijo)])
    cmd.extend(["--ppgdpo_backtrack", str(config.training.ppgdpo_backtrack)])
    cmd.extend(["--ppgdpo_newton", str(config.training.ppgdpo_newton)])
    cmd.extend(["--ppgdpo_tol_grad", str(config.training.ppgdpo_tol_grad)])
    cmd.extend(["--ppgdpo_ls", str(config.training.ppgdpo_ls)])

    if config.evaluation.ppgdpo_local_zero_hedge:
        cmd.append("--ppgdpo_local_zero_hedge")
    if config.evaluation.ppgdpo_cross_scales:
        cmd.append("--ppgdpo_cross_scales")
        cmd.extend(str(x) for x in config.evaluation.ppgdpo_cross_scales)
    _append_bool_flag(
        cmd,
        enabled=config.evaluation.ppgdpo_allow_long_horizon,
        true_flag="--ppgdpo_allow_long_horizon",
        false_flag="--ppgdpo_disallow_long_horizon",
    )
    cmd.extend(["--walk_forward_mode", str(config.evaluation.walk_forward_mode)])
    if config.evaluation.is_expanding:
        cmd.append("--expanding_window")
    if config.effective_rolling_train_months is not None:
        cmd.extend(["--rolling_train_months", str(config.effective_rolling_train_months)])
    cmd.extend(["--retrain_every", str(config.evaluation.retrain_every)])
    cmd.extend(["--refit_iters", str(config.evaluation.refit_iters)])
    return LegacyRunSpec(command=cmd, env=spec.env, cwd=spec.cwd, label="comparison")


def preflight_legacy(config: ResolvedExperimentConfig) -> LegacyPreflightReport:
    if config.runtime.legacy_entrypoint is None:
        raise ValueError("runtime.legacy_entrypoint must be configured for preflight.")
    entrypoint = str(config.runtime.legacy_entrypoint_path)
    cwd = str(config.runtime.legacy_workdir_path or Path(entrypoint).parent)
    env, _fred_value = _ensure_fred_key(config)
    help_result = subprocess.run(
        [_resolve_python(config), entrypoint, "--help"],
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    stderr_tail = None
    if help_result.returncode != 0 and help_result.stderr:
        stderr_tail = "\n".join(help_result.stderr.strip().splitlines()[-10:])
    fred_key_present = False
    if config.runtime.fred_api_key_env:
        fred_key_present = bool(os.environ.get(config.runtime.fred_api_key_env))
    return LegacyPreflightReport(
        entrypoint=entrypoint,
        cwd=cwd,
        fred_api_key_env=config.runtime.fred_api_key_env,
        fred_api_key_present=fred_key_present,
        help_ok=help_result.returncode == 0,
        help_returncode=help_result.returncode,
        stderr_tail=stderr_tail,
    )


def render_preflight_report(report: LegacyPreflightReport) -> str:
    lines = [
        f"entrypoint: {report.entrypoint}",
        f"workdir: {report.cwd}",
        f"fred_api_key_env: {report.fred_api_key_env}",
        f"fred_api_key_present: {str(report.fred_api_key_present).lower()}",
        f"help_ok: {str(report.help_ok).lower()}",
        f"help_returncode: {report.help_returncode}",
    ]
    if report.stderr_tail:
        lines.append("stderr_tail:")
        lines.extend(f"  {line}" for line in report.stderr_tail.splitlines())
    return "\n".join(lines)


def _phase_specs_for_render(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    selected_spec: str | None,
) -> list[LegacyRunSpec]:
    if phase == "selection":
        return [build_selection_run_spec(config)]
    if phase == "comparison":
        final_spec = resolve_comparison_spec(config, selected_spec=selected_spec)
        return [build_comparison_run_spec(config, selected_spec=final_spec)]
    if phase == "all":
        specs: list[LegacyRunSpec] = []
        if config.selection.enabled:
            specs.append(build_selection_run_spec(config))
            if selected_spec is None:
                return specs
        if config.comparison.enabled:
            final_spec = resolve_comparison_spec(config, selected_spec=selected_spec)
            specs.append(build_comparison_run_spec(config, selected_spec=final_spec))
        return specs
    raise PhaseError(f"Unknown phase '{phase}'. Choose from selection|comparison|all.")


def render_phase_commands(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    selected_spec: str | None = None,
    redact_secrets: bool = True,
) -> str:
    specs = _phase_specs_for_render(config, phase=phase, selected_spec=selected_spec)
    lines = [spec.shell_command(redact_secrets=redact_secrets) for spec in specs]
    if phase == "all" and config.selection.enabled and selected_spec is None:
        lines.append(
            "# comparison command omitted in dry rendering because the selected spec is only known after the selection phase completes."
        )
    return "\n".join(lines)


def _run_subprocess(spec: LegacyRunSpec) -> None:
    subprocess.run(
        spec.command,
        cwd=spec.cwd,
        env=spec.env,
        check=True,
        text=True,
        capture_output=False,
    )


def write_stage2_manifest(
    config: ResolvedExperimentConfig,
    *,
    selected_specs: list[str],
    completed_phases: list[str],
    commands: dict[str, str],
) -> None:
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "experiment": config.experiment.name,
        "completed_phases": completed_phases,
        "selected_specs": selected_specs,
        "primary_selected_spec": selected_specs[0] if selected_specs else None,
        "selection_output_dir": str(config.selection_output_dir),
        "comparison_output_dir": str(config.comparison_output_dir),
        "commands": commands,
    }
    config.manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def resolve_comparison_spec(
    config: ResolvedExperimentConfig,
    *,
    selected_spec: str | None = None,
    selected_rank: int | None = None,
) -> str:
    if selected_spec is not None:
        return selected_spec
    rank = config.comparison.selected_rank if selected_rank is None else int(selected_rank)
    if config.comparison.use_selected_spec:
        return choose_selected_spec(config, rank=rank)
    return config.comparison.fixed_spec or config.model.state_spec


def run_phase(
    config: ResolvedExperimentConfig,
    *,
    phase: str,
    dry_run: bool = False,
    selected_spec: str | None = None,
) -> Stage2RunResult | list[LegacyRunSpec]:
    if phase == "selection":
        spec = build_selection_run_spec(config)
        if dry_run:
            return [spec]
        _run_subprocess(spec)
        selected_specs = choose_selected_specs(config)
        write_selected_spec_artifact(config, selected_specs)
        write_stage2_manifest(
            config,
            selected_specs=selected_specs,
            completed_phases=["selection"],
            commands={"selection": spec.shell_command(redact_secrets=True)},
        )
        return Stage2RunResult(
            selected_spec=selected_specs[0],
            selected_specs=selected_specs,
            completed_phases=["selection"],
        )

    if phase == "comparison":
        final_spec = resolve_comparison_spec(config, selected_spec=selected_spec)
        spec = build_comparison_run_spec(config, selected_spec=final_spec)
        if dry_run:
            return [spec]
        _run_subprocess(spec)
        selected_specs = [final_spec]
        write_stage2_manifest(
            config,
            selected_specs=selected_specs,
            completed_phases=["comparison"],
            commands={"comparison": spec.shell_command(redact_secrets=True)},
        )
        return Stage2RunResult(
            selected_spec=final_spec,
            selected_specs=selected_specs,
            completed_phases=["comparison"],
        )

    if phase == "all":
        if dry_run:
            # When selection is enabled, the comparison spec is not yet known.
            return _phase_specs_for_render(config, phase="all", selected_spec=selected_spec)

        selected_specs: list[str] = []
        completed_phases: list[str] = []
        commands: dict[str, str] = {}

        if config.selection.enabled:
            sel_spec = build_selection_run_spec(config)
            _run_subprocess(sel_spec)
            completed_phases.append("selection")
            commands["selection"] = sel_spec.shell_command(redact_secrets=True)
            selected_specs = choose_selected_specs(config)
            write_selected_spec_artifact(config, selected_specs)
            selected_spec = selected_specs[0]

        if config.comparison.enabled:
            final_spec = resolve_comparison_spec(config, selected_spec=selected_spec)
            cmp_spec = build_comparison_run_spec(config, selected_spec=final_spec)
            _run_subprocess(cmp_spec)
            completed_phases.append("comparison")
            commands["comparison"] = cmp_spec.shell_command(redact_secrets=True)
            if not selected_specs:
                selected_specs = [final_spec]

        write_stage2_manifest(
            config,
            selected_specs=selected_specs,
            completed_phases=completed_phases,
            commands=commands,
        )
        return Stage2RunResult(
            selected_spec=selected_specs[0] if selected_specs else None,
            selected_specs=selected_specs,
            completed_phases=completed_phases,
        )

    raise PhaseError(f"Unknown phase '{phase}'. Choose from selection|comparison|all.")
