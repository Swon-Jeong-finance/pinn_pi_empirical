# Factor-covariance model

## Legacy-style mean-only model

A mean-only dynamic allocation model can be written as

\[
R_{t+1} = \mu(S_t) + \eta_{t+1},
\qquad
\eta_{t+1} \sim (0, \Sigma)
\]

In practice this means expected returns move with the state, but the conditional covariance is effectively fixed.

## Factor-covariance extension

`dynalloc_v2` instead uses a low-rank factor representation:

\[
R_{t+1} = \mu(S_t) + \Lambda F_{t+1} + U_{t+1}
\]

with conditional covariance

\[
\operatorname{Var}(R_{t+1} \mid \mathcal{F}_t)
= \Lambda \Omega_t \Lambda^\top + D
\]

where

- `\Lambda` is the asset-by-factor loading matrix,
- `\Omega_t` is the factor covariance matrix,
- `D` is diagonal idiosyncratic variance.

The minimal state-dependent covariance model in this package is:

\[
\Omega_t = \operatorname{diag}(\sqrt{h_t}) \, C \, \operatorname{diag}(\sqrt{h_t})
\]

where `C` is a fixed factor correlation matrix estimated on the training sample and the factor variances follow

\[
\log h_{j,t+1} = \omega_j + \phi_j \log(f_{j,t}^2 + \epsilon) + \gamma_j^\top S_t.
\]

This is deliberately simple:

- factor **means** can move with the state,
- factor **variances** can move with the state,
- factor **correlations** are fixed in the first version,
- asset covariance moves because factor covariance moves.

That is the smallest clean step from a mean-only world toward a covariance-aware world.
