from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class FactorRepresentation:
    factors: pd.DataFrame
    loadings: pd.DataFrame
    residual_var: pd.Series
    asset_alpha: pd.Series


class FactorExtractor:
    def fit(self, returns: pd.DataFrame, provided_factors: pd.DataFrame | None = None) -> FactorRepresentation:
        raise NotImplementedError


class ProvidedFactorExtractor(FactorExtractor):
    def __init__(self, factor_columns: list[str]):
        self.factor_columns = factor_columns

    def fit(self, returns: pd.DataFrame, provided_factors: pd.DataFrame | None = None) -> FactorRepresentation:
        if provided_factors is None:
            raise ValueError('provided_factors are required for extractor="provided"')
        factors = provided_factors[self.factor_columns].copy()
        common = returns.index.intersection(factors.index)
        Y = returns.loc[common].to_numpy(dtype=float)
        F = factors.loc[common].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(common)), F])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        alpha = beta[0]
        loadings = beta[1:]
        resid = Y - X @ beta
        resid_var = resid.var(axis=0, ddof=max(1, X.shape[1]))
        load_df = pd.DataFrame(loadings, index=factors.columns, columns=returns.columns).T
        alpha_s = pd.Series(alpha, index=returns.columns)
        return FactorRepresentation(
            factors=factors.loc[common],
            loadings=load_df,
            residual_var=pd.Series(np.maximum(resid_var, 1.0e-8), index=returns.columns),
            asset_alpha=alpha_s,
        )


class PCAFactorExtractor(FactorExtractor):
    def __init__(self, n_factors: int):
        self.n_factors = n_factors

    def fit(self, returns: pd.DataFrame, provided_factors: pd.DataFrame | None = None) -> FactorRepresentation:
        X = returns.to_numpy(dtype=float)
        pca = PCA(n_components=self.n_factors)
        scores = pca.fit_transform(X)
        factors = pd.DataFrame(scores, index=returns.index, columns=[f'PC{i+1}' for i in range(self.n_factors)])
        Xreg = np.column_stack([np.ones(len(factors)), factors.to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(Xreg, X, rcond=None)
        alpha = beta[0]
        loadings = beta[1:]
        recon = Xreg @ beta
        resid = X - recon
        load_df = pd.DataFrame(loadings, index=factors.columns, columns=returns.columns).T
        alpha_s = pd.Series(alpha, index=returns.columns)
        resid_var = pd.Series(np.maximum(resid.var(axis=0, ddof=max(1, Xreg.shape[1])), 1.0e-8), index=returns.columns)
        return FactorRepresentation(factors=factors, loadings=load_df, residual_var=resid_var, asset_alpha=alpha_s)
