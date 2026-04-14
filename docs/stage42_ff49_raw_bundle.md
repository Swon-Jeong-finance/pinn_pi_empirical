# Stage42: canonical FF49 raw bundle refresh

This stage adds a **raw-driven** FF49 bundle builder so newer OOS windows can be used without
reaching back into legacy stage output bundles.

## New command

```bash
dynalloc-v2 build-ff49-base-bundle \
  --out-dir ~/empricial_test/base_bundles/ff49_full_curve_core \
  --search-root ~/empricial_test \
  --fred-api-key "$FRED_API_KEY" \
  --manifest-split-profile cv2000_final20y
```

The command can also take explicit raw paths:

- `--ff49-zip`
- `--ff3-zip`
- `--ff5-zip`
- `--bond2y-csv`
- `--bond5y-csv`
- `--bond10y-csv`
- `--fred-cache-dir`
- `--macro-panel-csv`

## What it builds

The output bundle matches the v2 native selection format:

- `returns_panel.csv`
- `macro_panel.csv`
- `ff3_panel.csv`
- `ff5_panel.csv`
- `bond_panel.csv`
- `base_bundle_manifest.yaml`

## Design goal

- keep **raw discovery / bundle generation** separate from selection and OOS
- avoid depending on old stage output bundles just to move the calendar window
- preserve downstream compatibility with `select-native-suite` and `run-rank-sweep`
