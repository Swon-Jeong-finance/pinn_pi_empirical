# legacy

Stage 1 uses a vendored snapshot of the old empirical engine under:

```text
legacy/vendor/pgdpo_legacy_v69/
```

The compatibility wrapper below forwards into that vendored repo:

```text
legacy/run_french49_10y_model_based_latent_varx_fred.py
```

You can use either style in configs:

- explicit vendored entrypoint + `runtime.legacy_workdir`
- the wrapper path above, which `chdir`s into the vendored repo for you

The vendored subset includes the real monolithic runner, the `pgdpo_yahoo/`
package, and the bond CSV needed by the current bridge-first stage.
