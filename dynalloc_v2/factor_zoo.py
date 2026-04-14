from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


_V1_PLS_SMOOTH_SPAN = 6


@dataclass(frozen=True)
class FactorZooCandidate:
    name: str
    kind: str
    horizon: int | None = None
    n_components: int | None = None
    feature_blocks: tuple[str, ...] = ()
    provided_source: str | None = None
    residual_base: str | None = None
    residual_k: int | None = None


def default_pls_candidate_registry() -> list[FactorZooCandidate]:
    """Restore the v1-style pure returns PLS sweep used for stage1 debugging.

    `pls_only` should contain *only* the plain returns-driven predictive PLS
    candidates from the legacy code path. Extended supervised PLS families such as
    returns+macro7 or returns+ff5+macro7 belong to the broader factor zoo, not to
    the pure PLS regression-sanity subset.
    """
    return [
        FactorZooCandidate(
            name=f'pls_H{horizon}_k{k}',
            kind='pls',
            horizon=horizon,
            n_components=k,
            feature_blocks=('returns',),
        )
        for horizon in (6, 12, 24)
        for k in (2, 3)
    ]


def default_factor_zoo_registry() -> list[FactorZooCandidate]:
    specs: list[FactorZooCandidate] = []
    specs.extend([
        FactorZooCandidate(name='ff1', kind='provided', provided_source='ff1'),
        FactorZooCandidate(name='ff3', kind='provided', provided_source='ff3'),
        FactorZooCandidate(name='ff5', kind='provided', provided_source='ff5'),
        FactorZooCandidate(name='ff3_curve_core', kind='provided', provided_source='ff3_curve_core'),
        FactorZooCandidate(name='ff5_curve_core', kind='provided', provided_source='ff5_curve_core'),
    ])
    for k in (1, 2, 3, 4):
        specs.append(FactorZooCandidate(name=f'pca_k{k}', kind='pca', n_components=k))
    for base in ('ff3', 'ff5'):
        for k in (1, 2):
            specs.append(FactorZooCandidate(name=f'{base}_pcares_k{k}', kind='resid_pca', residual_base=base, residual_k=k))
    families: list[tuple[str, tuple[str, ...]]] = [
        ('pls', ('returns',)),
        ('pls_ret_macro7', ('returns', 'macro7')),
        ('pls_ret_ff5_macro7', ('returns', 'ff5', 'macro7')),
        ('pls_ret_ff5_curve_macro7', ('returns', 'ff5', 'curve_core', 'macro7')),
    ]
    for family, blocks in families:
        for horizon in (6, 12, 24):
            for k in (2, 3):
                specs.append(FactorZooCandidate(name=f'{family}_H{horizon}_k{k}', kind='pls', horizon=horizon, n_components=k, feature_blocks=blocks))
    return specs


def build_candidate_registry(zoo: str) -> list[FactorZooCandidate]:
    if zoo == 'pls_only':
        return default_pls_candidate_registry()
    if zoo == 'factor_zoo_v1':
        return default_factor_zoo_registry()
    raise KeyError(f'Unknown candidate zoo {zoo!r}')


def _future_avg_panel(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    horizon = int(max(1, horizon))
    for t in range(len(df)):
        end = t + 1 + horizon
        if end <= len(df):
            out[t] = values[t + 1 : end].mean(axis=0)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _next_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df.shift(-1)


def _provided_panel(source: str, *, ff3: pd.DataFrame, ff5: pd.DataFrame, bond: pd.DataFrame) -> pd.DataFrame:
    if source == 'ff1':
        return ff3[['Mkt-RF']].copy()
    if source == 'ff3':
        return ff3[['Mkt-RF', 'SMB', 'HML']].copy()
    if source == 'ff5':
        return ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()
    if source == 'ff3_curve_core':
        return pd.concat([ff3[['Mkt-RF', 'SMB', 'HML']], bond], axis=1)
    if source == 'ff5_curve_core':
        return pd.concat([ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], bond], axis=1)
    raise KeyError(f'Unknown provided factor source {source!r}')


def _feature_block_df(block: str, *, returns: pd.DataFrame, macro: pd.DataFrame, ff3: pd.DataFrame, ff5: pd.DataFrame, bond: pd.DataFrame) -> pd.DataFrame:
    if block == 'returns':
        return returns
    if block == 'macro7':
        return macro
    if block == 'ff3':
        return ff3
    if block == 'ff5':
        return ff5
    if block == 'curve_core':
        return bond
    raise KeyError(f'Unsupported feature block {block!r}')


def _build_candidate_features(candidate: FactorZooCandidate, *, returns: pd.DataFrame, macro: pd.DataFrame, ff3: pd.DataFrame, ff5: pd.DataFrame, bond: pd.DataFrame) -> pd.DataFrame:
    parts = [_feature_block_df(block, returns=returns, macro=macro, ff3=ff3, ff5=ff5, bond=bond) for block in candidate.feature_blocks]
    x = pd.concat(parts, axis=1)
    x = x.loc[:, ~x.columns.duplicated()].copy()
    return x


def _fit_standardizer(df: pd.DataFrame, train_dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    train = df.loc[df.index.intersection(train_dates)].dropna()
    if train.empty:
        train = df.dropna()
    mu = train.mean(axis=0)
    sig = train.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return mu, sig


def _standardize(df: pd.DataFrame, train_dates: pd.DatetimeIndex) -> pd.DataFrame:
    mu, sig = _fit_standardizer(df, train_dates)
    return (df - mu) / sig


def _fit_pca_factors(returns: pd.DataFrame, train_dates: pd.DatetimeIndex, n_components: int) -> pd.DataFrame:
    train = returns.loc[returns.index.intersection(train_dates)].dropna()
    if len(train) < 12:
        raise ValueError('Not enough training rows for PCA fit')
    ncomp = int(max(1, min(n_components, train.shape[1], len(train) - 1)))
    pca = PCA(n_components=ncomp)
    pca.fit(train.to_numpy(dtype=float))
    x_full = returns.dropna().copy()
    scores = pca.transform(x_full.to_numpy(dtype=float))
    cols = [f'PCA{j+1}' for j in range(scores.shape[1])]
    return pd.DataFrame(scores, index=x_full.index, columns=cols)


def _fit_residual_pca_factors(returns: pd.DataFrame, base_factors: pd.DataFrame, train_dates: pd.DatetimeIndex, residual_k: int) -> pd.DataFrame:
    common = returns.index.intersection(base_factors.index)
    Y = returns.loc[common].copy()
    F = base_factors.loc[common].copy()
    train_common = common.intersection(train_dates)
    train_Y = Y.loc[train_common].dropna()
    train_F = F.loc[train_Y.index].dropna()
    aligned_train = train_Y.index.intersection(train_F.index)
    train_Y = train_Y.loc[aligned_train]
    train_F = train_F.loc[aligned_train]
    if len(aligned_train) < 12:
        raise ValueError('Not enough training rows for residual PCA fit')
    X_train = np.column_stack([np.ones(len(aligned_train)), train_F.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(X_train, train_Y.to_numpy(dtype=float), rcond=None)
    full_idx = Y.dropna().index.intersection(F.dropna().index)
    X_full = np.column_stack([np.ones(len(full_idx)), F.loc[full_idx].to_numpy(dtype=float)])
    resid_full = Y.loc[full_idx].to_numpy(dtype=float) - X_full @ beta
    resid_train = pd.DataFrame(resid_full, index=full_idx).loc[full_idx.intersection(aligned_train)].to_numpy(dtype=float)
    ncomp = int(max(1, min(residual_k, resid_train.shape[1], len(resid_train) - 1)))
    pca = PCA(n_components=ncomp)
    pca.fit(resid_train)
    resid_scores = pca.transform(resid_full)
    resid_cols = [f'RESPC{j+1}' for j in range(resid_scores.shape[1])]
    resid_df = pd.DataFrame(resid_scores, index=full_idx, columns=resid_cols)
    return pd.concat([F.loc[full_idx], resid_df], axis=1)


def _train_end_position(index: pd.DatetimeIndex, train_dates: pd.DatetimeIndex) -> int:
    """Return the last in-sample row position for legacy PLS fitting.

    The legacy v1 code fit predictive PLS on all training observations up to the
    final training date *T*, then internally truncated usable target rows to
    `T - horizon`. An earlier v2 migration mistakenly first shrank the training
    dates to `T - horizon` and then subtracted `horizon` a second time, which
    effectively reduced the usable sample to `T - 2*horizon`.
    """
    train_dates = pd.DatetimeIndex(train_dates).sort_values().unique()
    if train_dates.empty:
        return -1
    pos_lookup = {dt: i for i, dt in enumerate(index)}
    train_pos = [pos_lookup[dt] for dt in train_dates if dt in pos_lookup]
    if not train_pos:
        return -1
    return int(max(train_pos))


def _smooth_scores(df: pd.DataFrame, span: int = _V1_PLS_SMOOTH_SPAN) -> pd.DataFrame:
    if int(span) <= 0:
        return df
    return df.ewm(span=int(span), adjust=False).mean()


def _fit_pls_returns_to_future_avg_returns(returns: pd.DataFrame, *, train_dates: pd.DatetimeIndex, n_components: int, horizon: int, smooth_span: int = _V1_PLS_SMOOTH_SPAN) -> pd.DataFrame:
    train_end_pos = _train_end_position(returns.index, train_dates)
    if train_end_pos < 10:
        raise ValueError('Not enough training rows for PLS fit')
    X_all = returns.to_numpy(dtype=float)
    n_assets = int(X_all.shape[1])
    max_t = int(train_end_pos) - int(horizon)
    if max_t < 5:
        raise ValueError('Not enough effective training rows for PLS fit')
    N = max_t + 1
    X = X_all[0:N, :]
    Y = np.zeros((N, n_assets), dtype=float)
    for k in range(1, int(horizon) + 1):
        Y += X_all[k : k + N, :]
    Y /= float(horizon)
    ncomp = int(max(1, min(int(n_components), X.shape[1], Y.shape[1], len(X) - 1)))
    pls = PLSRegression(n_components=ncomp, scale=True)
    pls.fit(X, Y)
    scores = pls.transform(X_all)
    cols = [f'PLS{j+1}' for j in range(scores.shape[1])]
    return _smooth_scores(pd.DataFrame(scores, index=returns.index, columns=cols), span=smooth_span)


def _build_pls_predictor_frame(candidate: FactorZooCandidate, *, returns: pd.DataFrame, macro: pd.DataFrame, ff3: pd.DataFrame, ff5: pd.DataFrame, bond: pd.DataFrame, train_dates: pd.DatetimeIndex) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block in candidate.feature_blocks:
        df = _feature_block_df(block, returns=returns, macro=macro, ff3=ff3, ff5=ff5, bond=bond).copy()
        z = _standardize(df, train_dates)
        prefix = 'ret' if block == 'returns' else block
        frames.append(z.add_prefix(f'{prefix}__'))
    if not frames:
        raise RuntimeError('No predictor blocks were supplied for supervised PLS state construction.')
    x_df = pd.concat(frames, axis=1)
    x_df = x_df.loc[:, ~x_df.columns.duplicated()].copy()
    return x_df


def _fit_pls_predictors_to_future_avg_returns(candidate: FactorZooCandidate, *, returns: pd.DataFrame, macro: pd.DataFrame, ff3: pd.DataFrame, ff5: pd.DataFrame, bond: pd.DataFrame, train_dates: pd.DatetimeIndex, smooth_span: int = _V1_PLS_SMOOTH_SPAN) -> pd.DataFrame:
    assert candidate.horizon is not None
    x_df = _build_pls_predictor_frame(candidate, returns=returns, macro=macro, ff3=ff3, ff5=ff5, bond=bond, train_dates=train_dates)
    train_end_pos = _train_end_position(returns.index, train_dates)
    if train_end_pos < 10:
        raise ValueError('Not enough training rows for PLS fit')
    x_df = x_df.loc[returns.index].copy()
    X_all = x_df.to_numpy(dtype=float)
    Y_all_src = returns.to_numpy(dtype=float)
    n_assets = int(Y_all_src.shape[1])
    max_t = int(train_end_pos) - int(candidate.horizon)
    if max_t < 5:
        raise ValueError('Not enough effective training rows for PLS fit')
    N = max_t + 1
    X = X_all[0:N, :]
    Y = np.zeros((N, n_assets), dtype=float)
    for k in range(1, int(candidate.horizon) + 1):
        Y += Y_all_src[k : k + N, :]
    Y /= float(candidate.horizon)
    y_mean = np.nanmean(Y, axis=0)
    y_std = np.nanstd(Y, axis=0, ddof=0)
    y_std = np.where(y_std <= 1e-12, 1.0, y_std)
    Y_fit = (Y - y_mean[None, :]) / y_std[None, :]
    ncomp = int(max(1, min(int(candidate.n_components or 1), X.shape[1], Y_fit.shape[1], len(X) - 1)))
    pls = PLSRegression(n_components=ncomp, scale=False)
    pls.fit(X, Y_fit)
    scores = pls.transform(X_all)
    cols = [f'PLS{j+1}' for j in range(scores.shape[1])]
    return _smooth_scores(pd.DataFrame(scores, index=returns.index, columns=cols), span=smooth_span)


def _fit_pls_factors(candidate: FactorZooCandidate, *, returns: pd.DataFrame, macro: pd.DataFrame, ff3: pd.DataFrame, ff5: pd.DataFrame, bond: pd.DataFrame, train_dates: pd.DatetimeIndex) -> pd.DataFrame:
    assert candidate.horizon is not None
    if tuple(candidate.feature_blocks) == ('returns',):
        return _fit_pls_returns_to_future_avg_returns(
            returns,
            train_dates=train_dates,
            n_components=int(candidate.n_components or 1),
            horizon=int(candidate.horizon),
        )
    return _fit_pls_predictors_to_future_avg_returns(
        candidate,
        returns=returns,
        macro=macro,
        ff3=ff3,
        ff5=ff5,
        bond=bond,
        train_dates=train_dates,
    )


def build_candidate_panels(candidate: FactorZooCandidate, *, returns: pd.DataFrame, macro: pd.DataFrame, ff3: pd.DataFrame, ff5: pd.DataFrame, bond: pd.DataFrame, train_dates: pd.DatetimeIndex) -> dict[str, Any]:
    if candidate.kind == 'provided':
        factors = _provided_panel(str(candidate.provided_source), ff3=ff3, ff5=ff5, bond=bond)
    elif candidate.kind == 'pca':
        factors = _fit_pca_factors(returns, train_dates, int(candidate.n_components or 1))
    elif candidate.kind == 'resid_pca':
        base = _provided_panel(str(candidate.residual_base), ff3=ff3, ff5=ff5, bond=bond)
        factors = _fit_residual_pca_factors(returns, base, train_dates, int(candidate.residual_k or 1))
    elif candidate.kind == 'pls':
        factors = _fit_pls_factors(candidate, returns=returns, macro=macro, ff3=ff3, ff5=ff5, bond=bond, train_dates=train_dates)
    else:
        raise KeyError(f'Unknown candidate kind {candidate.kind!r}')
    factors = factors.sort_index().dropna(how='any')
    states = _standardize(factors.copy(), train_dates)
    states.columns = [f'state_{c}' for c in states.columns]
    return {
        'factors': factors,
        'states': states,
        'meta': {
            'kind': candidate.kind,
            'horizon': candidate.horizon,
            'n_components': candidate.n_components,
            'feature_blocks': list(candidate.feature_blocks),
            'provided_source': candidate.provided_source,
            'residual_base': candidate.residual_base,
            'residual_k': candidate.residual_k,
            'pls_recipe': 'v1_compatible_future_avg_targets',
            'pls_smooth_span': _V1_PLS_SMOOTH_SPAN if candidate.kind == 'pls' else 0,
        },
    }
