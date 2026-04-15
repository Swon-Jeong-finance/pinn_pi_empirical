# v69 fast-path constraints

This version keeps the flexible v68 constraint interface, but restores a fast
training/evaluation path for the common case:

- `allow_short = False`
- `per_asset_cap = None`
- risky-only constraint handled as `sum(risky) <= effective_risky_cap`
- cash remains residual: `cash = 1 - sum(risky)`

What changed versus v68:

- the common long-only + sum-cap set now uses an exact vectorized simplex projection
  instead of the slower generic box+sum bisection path;
- policy constraint validation is done once at module init instead of every forward pass;
- generic box+sum projection is still available when `per_asset_cap` is activated or
  when shorting is allowed.

Practical implication:

- `risky_cap=2`, `cash_floor=-1`, `allow_short=False`, `per_asset_cap=None` should be
  much closer in speed to the pre-v68 code, while keeping the leverage experiment support.
