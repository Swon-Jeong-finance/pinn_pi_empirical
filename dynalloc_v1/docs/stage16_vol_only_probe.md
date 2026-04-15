# Stage 16 vol-only hedge probe

This stage adds a daily-to-monthly realized-volatility state so we can test whether
cross-hedging in the strong ff49+cv2000+curve_core lane is really a volatility
channel.

## New macro features
- `log_rv_mkt_1m`: log monthly realized variance of the Ken French daily market total return (`Mkt-RF + RF`).
- `log_rv_mkt_3m`: log 3-month rolling realized variance of the same daily market series.

## New macro profiles
- `rv_mkt_1m_only`: pure vol-only state.
- `bond_curve_core_cv_plus_rv1m`: current curve-credit macro plus one fast vol state.

## Probe configs
The generated stage-16 configs intentionally bypass broad spec selection:
- `selection.profile: disabled`
- `comparison.profile: fixed_base_model`
- `comparison.fixed_spec: macro7_only`

This ensures the new state does not get rejected early by the existing return/state-fit selection screen.

## Suggested interpretation
- If `volonly` works: volatility channel alone may drive the hedge effect.
- If `macroplusrv` works but `volonly` does not: fast volatility is a useful add-on, but not a sufficient state by itself.
- If neither works: the original ff49+curve_core effect likely depends on slower curve/credit channels rather than pure volatility mean reversion.
