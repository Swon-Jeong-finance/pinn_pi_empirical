# Stage46: Ported stage-1 becomes authoritative, external v1 becomes audit-only

This stage tightens the remaining legacy boundary in native selection without changing the selection semantics.

## What changed

- `select-native-suite` now always uses the **ported v2 stage-1 evaluator** as the selection path.
- The optional external v1 loader (`legacy_stage1_v1_root`) no longer overrides stage-1 metrics.
- When a legacy v1 root is supplied, v2 computes an **external audit comparison** instead:
  - `spec_selection_stage1_external_audit.csv`
  - per-block ported vs external metrics
  - absolute and signed deltas for the key predictive diagnostics

## Why this matters

Before this stage, the production stage-1 path could silently switch engines depending on whether a v1 checkout happened to be available. That made it too easy to blur

1. the ported stage-1 semantics that now live in v2, and
2. the external v1 implementation used only for verification.

The new behavior makes the ownership clear:

- **v2 owns the stage-1 logic**
- **v1 is only a parity/audit reference**

## Outputs

`selected_spec.yaml` and `suite_manifest.yaml` now record:

- `stage1_engine: ported_legacy_v2`
- `stage1_external_audit_enabled`
- `stage1_external_audit_v1_root`
- `selection_stage1_audit_csv`

When no external v1 root is supplied, the audit fields remain `false` / `null`.

## Validation

Two regression checks were added:

1. Supplying a fake external v1 root with deliberately absurd metrics does **not** change the selected specs.
2. The external audit CSV is emitted only when the audit root is provided.
