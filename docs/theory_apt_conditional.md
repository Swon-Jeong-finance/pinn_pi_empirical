# APT conditional mean + conditional factor covariance

Stage19 moves `dynalloc_v2` one step closer to the intended theory world:

- **Asset returns** are generated from a low-rank factor structure

  \[
  r_{t+1} = \alpha + \Lambda f_{t+1} + \eta_{t+1}
  \]

- **Factor means** are state-dependent

  \[
  \mathbb E[f_{t+1}\mid s_t] = B_0 + B s_t
  \]

- **Factor covariance** is also state-dependent, but **not** GARCH/DCC:

  \[
  \Omega_t = \mathrm{diag}(h_{1,t},\dots,h_{m,t}),
  \qquad
  \log h_{j,t} = \omega_j + \gamma_j^\top s_t
  \]

- This implies an asset covariance matrix

  \[
  \Sigma_t = \Lambda \Omega_t \Lambda^\top + D
  \]

The key modeling choice is deliberate:

- **no multivariate GARCH**
- **no DCC**
- **yes to APT / factor covariance transmission**

The economic idea is that conditional heteroskedasticity should enter through the covariance of common factors, and then propagate to assets through the loading matrix.
