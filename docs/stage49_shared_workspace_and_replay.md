# Stage 49: shared workspace + in-sample replay

This stage adds two workflow features aimed at making empirical runs easier to reproduce across code versions.

## 1) Shared workspace root

The package now understands a version-independent shared root via `DYNALLOC_SHARED_ROOT`.
When unset, it defaults to `~/empricial_test/shared`.

The helper command

```bash
dynalloc-v2 init-workspace
```

creates the standard layout:

- `cache_french/`
- `cache_fred/`
- `base_bundles/`
- `experiments/`

Legacy bridge helpers now seed and use the shared French/FRED caches instead of relying on a relative `./_cache_french` in the current working directory.

## 2) Manifest-driven replay of train / validation / full in-sample

A new command

```bash
dynalloc-v2 replay-sample --manifest <suite_manifest.yaml> --rank 1 --sample insample_full
```

replays a selected rank on windows derived from `selection_split` in the suite manifest.

Supported samples:

- `insample_full`
- `selection_train`
- `selection_validation`

The replay fits once on the full pre-test pool (`train_pool_start` to `final_oos_train_end`) and then replays the chosen sample window, which is useful for diagnosing whether crisis-period hedge failures are already visible inside the in-sample region.
