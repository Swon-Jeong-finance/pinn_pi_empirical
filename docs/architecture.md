# Architecture

## Package layout

- `dynalloc_v2.schema` – Pydantic configuration models.
- `dynalloc_v2.data` – CSV and synthetic data loaders.
- `dynalloc_v2.factors` – factor extraction and loading estimation.
- `dynalloc_v2.covariance` – factor covariance estimators.
- `dynalloc_v2.policies` – myopic and projected policies.
- `dynalloc_v2.experiments` – walk-forward experiment runner and reporting.
- `dynalloc_v2.cli` – user-facing command line interface.

## Separation of concerns

The old code mixed state construction, asset-universe specifics, covariance assumptions,
policy logic, and experiment orchestration in one line of execution.

The new code draws a hard line between:

1. **observables** (`R_t`, provided factor returns, state variables),
2. **representations** (`Λ`, factor blocks, derived covariance states),
3. **decision rules** (myopic, projected), and
4. **experiment logistics** (walk-forward, logging, summaries).

This is designed so that future work can swap in:

- a richer state model,
- a different factor extractor,
- a different covariance update,
- or a different projection engine,

without rewriting the entire runner.
