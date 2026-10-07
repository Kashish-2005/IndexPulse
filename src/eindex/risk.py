"""Risk model module: Ledoit-Wolf covariance estimation and factor decomposition."""

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

ANNUALIZATION_FACTOR = 252
EIGENVALUE_FLOOR = 1e-8


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized Ledoit-Wolf shrinkage covariance matrix guaranteed to be PSD.

    Args:
        returns: Daily asset returns DataFrame with shape (T, N), where index is date
            and columns are ticker symbols.

    Returns:
        pd.DataFrame: Annualized (x252), symmetrized, eigenvalue-floored covariance
            matrix with shape (N, N), matching input columns on both axes.

    Raises:
        ValueError: If returns contain NaNs or input DataFrame is empty.
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")

    if returns.isna().any().any():
        nan_cols = returns.columns[returns.isna().any()].tolist()
        raise ValueError(f"Returns DataFrame contains NaNs in columns: {nan_cols}")

    tickers = returns.columns
    n_assets = len(tickers)

    # 1. Fit Ledoit-Wolf shrinkage model on daily returns
    lw = LedoitWolf()
    lw.fit(returns.to_numpy())
    cov_daily = lw.covariance_

    # 2. Annualize (x252)
    cov_ann = cov_daily * ANNUALIZATION_FACTOR

    # 3. Symmetrize to counter numerical roundoff
    cov_sym = 0.5 * (cov_ann + cov_ann.T)

    # 4. Eigenvalue floor at 1e-8 to ensure strict positive semi-definiteness
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    eigvals_floored = np.maximum(eigvals, EIGENVALUE_FLOOR)
    cov_psd = eigvecs @ np.diag(eigvals_floored) @ eigvecs.T

    # 5. Final symmetrization post-reconstruction
    cov_psd = 0.5 * (cov_psd + cov_psd.T)

    logger.debug(
        "Estimated Ledoit-Wolf covariance for %d assets across %d observations.",
        n_assets,
        len(returns),
    )

    return pd.DataFrame(cov_psd, index=tickers, columns=tickers)


def psd_factor(cov: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Compute lower/left factor L such that cov ~= L @ L.T using eigendecomposition.

    Suitable for quadratic risk formulation in cvxpy:
    x.T @ cov @ x = ||L.T @ x||_2^2 = sum_squares(L.T @ x)

    Args:
        cov: Symmetric positive semi-definite covariance matrix (N x N).

    Returns:
        np.ndarray: Factor matrix L of shape (N, N) such that L @ L.T ~= cov.
    """
    matrix = cov.to_numpy() if isinstance(cov, pd.DataFrame) else np.asarray(cov)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {matrix.shape}")

    # Symmetrize input
    matrix_sym = 0.5 * (matrix + matrix.T)

    # Eigendecomposition: cov = V @ diag(w) @ V.T = (V @ diag(sqrt(w))) @ (V @ diag(sqrt(w))).T
    eigvals, eigvecs = np.linalg.eigh(matrix_sym)
    eigvals_clipped = np.maximum(eigvals, 0.0)

    # L = V @ diag(sqrt(eigvals))
    factor_l = eigvecs @ np.diag(np.sqrt(eigvals_clipped))
    return factor_l
