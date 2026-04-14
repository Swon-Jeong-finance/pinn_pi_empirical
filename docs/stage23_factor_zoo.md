
# Stage23: native factor zoo screening

This stage extends `select-native-suite` from a PLS-only selection universe to a broader factor zoo.

Candidate zoo `factor_zoo_v1` includes:
- FF interpretable factors: FF1, FF3, FF5, FF3+curve_core, FF5+curve_core
- PCA on asset returns: k=1..4
- FF + PCA residual hybrids: FF3/FF5 plus residual PCA k=1,2
- PLS latent factors with H in {6,12,24}, k in {2,3}
  - returns-only
  - returns + macro7
  - returns + FF5 + macro7
  - returns + FF5 + curve_core + macro7

The stage keeps the conditional law fixed so that screening focuses on factor-builder performance first.
