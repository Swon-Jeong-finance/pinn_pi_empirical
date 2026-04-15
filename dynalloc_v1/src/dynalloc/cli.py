from __future__ import annotations

import argparse
import json
import sys

from .legacy_bridge import preflight_legacy, render_phase_commands, render_preflight_report, run_phase
from .planner import render_resolved_config
from .profile_store import PROFILE_SECTIONS, discover_profile_store
from .resolver import load_and_resolve
from .selection import choose_selected_spec
from .stage3a_backend import artifact_snapshot as artifact_snapshot_stage3a, run_stage3a
from .stage3b_backend import artifact_snapshot as artifact_snapshot_stage3b, run_stage3b
from .stage4_backend import artifact_snapshot as artifact_snapshot_stage4, run_stage4
from .stage5_backend import artifact_snapshot as artifact_snapshot_stage5, run_stage5
from .stage6_backend import artifact_snapshot as artifact_snapshot_stage6, run_stage6
from .stage7_backend import artifact_snapshot as artifact_snapshot_stage7, run_stage7


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")


def _add_show_secrets_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="Show secret CLI arguments such as --fred_api_key in clear text.",
    )


def _add_phase_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phase",
        choices=["selection", "comparison", "all"],
        default="all",
        help="Which phase to render or run.",
    )
    parser.add_argument(
        "--selected-spec",
        help="Optional explicit selected spec override for the comparison phase.",
    )
    parser.add_argument(
        "--selected-rank",
        type=int,
        help="Optional selected rank override (1=best, 2=runner-up, ...).",
    )
    parser.add_argument(
        "--worker-id",
        help="Optional worker id for coordinated rank-sweep runs.",
    )
    parser.add_argument(
        "--device",
        help="Optional runtime.device override for this invocation (for example cuda:2).",
    )



def _cmd_list_profiles(args: argparse.Namespace) -> int:
    store = discover_profile_store()
    sections = [args.section] if args.section else list(PROFILE_SECTIONS)
    print(f"profile_root: {store.root} ({store.source})")
    for name in sections:
        print(f"[{name}]")
        for profile_name in store.names(name):
            print(f"  - {profile_name}")
    return 0



def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_and_resolve(args.config)
    print(f"OK: {config.experiment.name}")
    return 0



def _cmd_plan(args: argparse.Namespace) -> int:
    config = load_and_resolve(args.config)
    if args.format == "json":
        print(json.dumps(config.model_dump(mode="json"), indent=2))
    else:
        print(render_resolved_config(config))
    return 0



def _cmd_preflight(args: argparse.Namespace) -> int:
    config = load_and_resolve(args.config)
    report = preflight_legacy(config)
    print(render_preflight_report(report))
    return 0 if report.help_ok else 1



def _cmd_legacy_cmd(args: argparse.Namespace) -> int:
    config = load_and_resolve(args.config)
    rendered = render_phase_commands(
        config,
        phase=args.phase,
        selected_spec=args.selected_spec,
        redact_secrets=not args.show_secrets,
    )
    print(rendered)
    return 0



def _cmd_selected_spec(args: argparse.Namespace) -> int:
    config = load_and_resolve(args.config)
    rank = args.rank if args.rank is not None else 1
    print(choose_selected_spec(config, rank=rank))
    return 0



def _cmd_artifacts(args: argparse.Namespace) -> int:
    config = load_and_resolve(args.config)
    if config.runtime.backend == "native_stage7":
        print(artifact_snapshot_stage7(config, phase=args.phase, selected_spec=args.selected_spec, selected_rank=args.selected_rank))
    elif config.runtime.backend == "native_stage6":
        print(artifact_snapshot_stage6(config, phase=args.phase, selected_spec=args.selected_spec, selected_rank=args.selected_rank))
    elif config.runtime.backend == "native_stage5":
        print(artifact_snapshot_stage5(config, phase=args.phase, selected_spec=args.selected_spec))
    elif config.runtime.backend == "native_stage4":
        print(artifact_snapshot_stage4(config, phase=args.phase, selected_spec=args.selected_spec))
    elif config.runtime.backend == "native_stage3b":
        print(artifact_snapshot_stage3b(config, phase=args.phase, selected_spec=args.selected_spec))
    else:
        print(artifact_snapshot_stage3a(config, phase=args.phase, selected_spec=args.selected_spec))
    return 0



def _cmd_run(args: argparse.Namespace) -> int:
    config = load_and_resolve(args.config)
    if config.runtime.backend == "plan":
        print(render_resolved_config(config))
        return 0
    if config.runtime.backend == "legacy_bridge":
        result = run_phase(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            selected_spec=args.selected_spec,
        )
    elif config.runtime.backend == "native_stage3a":
        result = run_stage3a(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            selected_spec=args.selected_spec,
        )
    elif config.runtime.backend == "native_stage3b":
        result = run_stage3b(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            selected_spec=args.selected_spec,
        )
    elif config.runtime.backend == "native_stage4":
        result = run_stage4(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            selected_spec=args.selected_spec,
        )
    elif config.runtime.backend == "native_stage5":
        result = run_stage5(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            selected_spec=args.selected_spec,
        )
    elif config.runtime.backend == "native_stage7":
        result = run_stage7(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            selected_spec=args.selected_spec,
            selected_rank=args.selected_rank,
            worker_id=args.worker_id,
            runtime_device=args.device,
        )
    elif config.runtime.backend == "native_stage6":
        result = run_stage6(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            selected_spec=args.selected_spec,
            selected_rank=args.selected_rank,
        )
    else:
        raise ValueError(f"Unknown backend: {config.runtime.backend}")

    if args.dry_run:
        for spec in result:
            print(spec.shell_command(redact_secrets=not args.show_secrets))
        if args.phase == "all" and config.selection.enabled and args.selected_spec is None:
            print(
                "# comparison command omitted in dry run because the selected spec is only known after the selection phase completes."
            )
        return 0

    if result.selected_spec:
        print(f"selected_spec: {result.selected_spec}")
    processed_ranks = getattr(result, "processed_ranks", None)
    if processed_ranks:
        print("processed_ranks:")
        for rank in processed_ranks:
            print(f"  - {rank}")
    failed_ranks = getattr(result, "failed_ranks", None)
    if failed_ranks:
        print("failed_ranks:")
        for rank in failed_ranks:
            print(f"  - {rank}")
    print("completed_phases:")
    for phase_name in result.completed_phases:
        print(f"  - {phase_name}")
    warnings = getattr(result, "warnings", [])
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dynalloc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_profiles = subparsers.add_parser("list-profiles", help="List available profile names.")
    list_profiles.add_argument(
        "--section",
        choices=list(PROFILE_SECTIONS),
        help="Optional single section to list.",
    )
    list_profiles.set_defaults(func=_cmd_list_profiles)

    validate = subparsers.add_parser("validate", help="Validate and resolve a config.")
    _add_config_argument(validate)
    validate.set_defaults(func=_cmd_validate)

    plan = subparsers.add_parser("plan", help="Print the resolved experiment plan.")
    _add_config_argument(plan)
    plan.add_argument("--format", choices=["yaml", "json"], default="yaml")
    plan.set_defaults(func=_cmd_plan)

    preflight = subparsers.add_parser(
        "preflight",
        help="Check that the legacy entrypoint imports cleanly and the workdir exists.",
    )
    _add_config_argument(preflight)
    preflight.set_defaults(func=_cmd_preflight)

    legacy_cmd = subparsers.add_parser(
        "legacy-cmd",
        help="Render the translated legacy command(s) for the requested phase.",
    )
    _add_config_argument(legacy_cmd)
    _add_phase_argument(legacy_cmd)
    _add_show_secrets_argument(legacy_cmd)
    legacy_cmd.set_defaults(func=_cmd_legacy_cmd)

    selected_spec = subparsers.add_parser(
        "selected-spec",
        help="Read the chosen spec from a completed selection phase.",
    )
    _add_config_argument(selected_spec)
    selected_spec.add_argument("--rank", type=int, default=1, help="Selection rank to read (1=best).")
    selected_spec.set_defaults(func=_cmd_selected_spec)

    artifacts = subparsers.add_parser(
        "artifacts",
        help="Inspect discovered output artifacts for the requested phase.",
    )
    _add_config_argument(artifacts)
    _add_phase_argument(artifacts)
    artifacts.set_defaults(func=_cmd_artifacts)

    run = subparsers.add_parser("run", help="Run through the selected backend.")
    _add_config_argument(run)
    _add_phase_argument(run)
    run.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    _add_show_secrets_argument(run)
    run.set_defaults(func=_cmd_run)

    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
