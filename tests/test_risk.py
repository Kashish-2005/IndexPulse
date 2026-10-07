"""Unit tests for the risk module (Ledoit-Wolf covariance and PSD factor)."""

import numpy as np
import pandas as pd
import pytest

from eindex.risk import ledoit_wolf_cov, psd_factor


@pytest.fixture
def synthetic_returns():
    """Generate 300 business days of synthetic Gaussian returns for 4 assets."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    tickers = ["TICK_A", "TICK_B", "TICK_C", "TICK_D"]
    rets = rng.normal(0.0005, 0.015, size=(300, 4))
    return pd.DataFrame(rets, index=dates, columns=tickers)


def test_raises_on_nan(synthetic_returns):
    """Ensure ledoit_wolf_cov fails loudly if NaNs exist in returns."""
    corrupted_returns = synthetic_returns.copy()
    corrupted_returns.iloc[5, 1] = np.nan

    with pytest.raises(ValueError, match="contains NaNs"):
        ledoit_wolf_cov(corrupted_returns)


def test_raises_on_empty_frame():
    """Ensure ledoit_wolf_cov raises on an empty input DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Returns DataFrame is empty"):
        ledoit_wolf_cov(empty_df)


def test_shape_and_axes_match(synthetic_returns):
    """Covariance shape, index, and columns must match ticker universe."""
    cov = ledoit_wolf_cov(synthetic_returns)
    assert cov.shape == (4, 4)
    pd.testing.assert_index_equal(cov.index, synthetic_returns.columns)
    pd.testing.assert_index_equal(cov.columns, synthetic_returns.columns)


def test_matrix_is_symmetric(synthetic_returns):
    """Covariance matrix must be strictly symmetric."""
    cov = ledoit_wolf_cov(synthetic_returns)
    cov_arr = cov.to_numpy()
    np.testing.assert_allclose(cov_arr, cov_arr.T, atol=1e-10)


def test_eigenvalue_floor_guarantees_psd(synthetic_returns):
    """Minimum eigenvalue must be at least 1e-8."""
    cov = ledoit_wolf_cov(synthetic_returns)
    eigvals = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigvals >= 1e-8).all()
    assert np.min(eigvals) >= 1e-8


def test_annualization_scaling():
    """Check annualization scales daily variance by ~252."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2023-01-01", periods=1000, freq="B")
    daily_sigma = 0.01
    # Independent returns with daily variance = 0.01^2 = 0.0001
    rets = rng.normal(0.0, daily_sigma, size=(1000, 2))
    df = pd.DataFrame(rets, index=dates, columns=["S1", "S2"])

    cov = ledoit_wolf_cov(df)
    expected_annual_var = (daily_sigma**2) * 252  # ~0.0252

    # Allow tolerance due to shrinkage toward identity
    assert np.isclose(cov.loc["S1", "S1"], expected_annual_var, rtol=0.15)
    assert np.isclose(cov.loc["S2", "S2"], expected_annual_var, rtol=0.15)


def test_psd_factor_reconstruction(synthetic_returns):
    """Verify L @ L.T reconstructs the original covariance matrix."""
    cov = ledoit_wolf_cov(synthetic_returns)
    factor_l = psd_factor(cov)

    assert factor_l.shape == cov.shape
    reconstructed = factor_l @ factor_l.T

    np.testing.assert_allclose(reconstructed, cov.to_numpy(), atol=1e-8)
