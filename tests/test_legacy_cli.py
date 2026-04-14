from __future__ import annotations

from dynalloc_v2.cli import build_parser
from dynalloc_v2.legacy_cli import normalize_legacy_alias_argv


def test_cli_help_hides_top_level_bridge_aliases():
    help_text = build_parser().format_help()
    assert (' legacy ' in help_text) or ('\n  legacy' in help_text)
    assert 'bridge-v1-lane' not in help_text
    assert 'bridge-v1-suite' not in help_text


def test_cli_parses_nested_legacy_bridge_command():
    parser = build_parser()
    args = parser.parse_args([
        'legacy',
        'bridge-v1-lane',
        '--v1-root', 'v1',
        '--config-stem', 'lane',
        '--out-dir', 'out',
    ])
    assert args.cmd == 'legacy'
    assert args.legacy_cmd == 'bridge-v1-lane'
    assert callable(args.func)


def test_cli_normalizes_legacy_alias_to_nested_command():
    assert normalize_legacy_alias_argv(['bridge-v1-lane', '--v1-root', 'v1']) == [
        'legacy',
        'bridge-v1-lane',
        '--v1-root',
        'v1',
    ]
