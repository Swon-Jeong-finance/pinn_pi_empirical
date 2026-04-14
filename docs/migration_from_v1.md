# Migration notes from dynalloc_v1

`dynalloc_v2` is not a drop-in replacement for the old stage-based code.
It is a new project that keeps the *research problem* but changes the internal architecture.

## What stays conceptually similar?

- walk-forward estimation,
- train/test split logic,
- comparison across covariance / cross modes,
- projected policy as the economically interesting object.

## What changes?

- no stage3/4/5/6/7 layering,
- no dependency on the legacy `run_french49_10y_...py` runner,
- no monolithic output format tied to old rank-sweep internals,
- explicit distinction between state, factor, covariance, and policy layers.

## Recommended workflow

- Keep `dynalloc_v1` frozen for reproduction and paper tables.
- Use `dynalloc_v2` as the sandbox for the next paper / engine.
