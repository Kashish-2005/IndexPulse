"""Tests for the alpha signal generation module."""

import numpy as np
import pandas as pd
import pytest

from eindex.signals import compute_alpha


class MockSignalWeights:
    momentum: float = 0.5
    low_vol: float = 0.3
    reversal: float = 0.2


class MockConfig:
    signal_weights = MockSignalWeights()
    ic: float = 0.05


@pytest.fixture
def synthetic_data():
    """Create 300 days of price data for 3 tickers."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    # Random walk prices
    data = np.exp(rng.normal(0.0001, 0.01, (300, 3)).cumsum(axis=0)) * 100
    return pd.DataFrame(data, index=dates, columns=tickers)


def test_no_look_ahead(synthetic_data):
    """Corrupting future data should not change current alpha."""
    cfg = MockConfig()
    t_date = synthetic_data.index[260]

    alpha_clean = compute_alpha(synthetic_data, t_date, cfg)

    # Corrupt data after t_date
    corrupted_data = synthetic_data.copy()
    corrupted_data.loc[corrupted_data.index > t_date, :] = 9999.9

    alpha_corrupted = compute_alpha(corrupted_data, t_date, cfg)

    pd.testing.assert_series_equal(alpha_clean, alpha_corrupted)


def test_insufficient_history(synthetic_data):
    """Fails if history < 252 days."""
    cfg = MockConfig()
    short_data = synthetic_data.iloc[:100]
    t_date = short_data.index[-1]

    with pytest.raises(ValueError, match="Insufficient history"):
        compute_alpha(short_data, t_date, cfg)


def test_alpha_distribution(synthetic_data):
    """Alpha should have no NaNs and reasonable variation."""
    cfg = MockConfig()
    t_date = synthetic_data.index[-1]
    alpha = compute_alpha(synthetic_data, t_date, cfg)

    assert isinstance(alpha, pd.Series)
    assert not alpha.isna().any()
    assert alpha.name == "alpha"
    # Given IC=0.05 and vol ~15-30%, alpha magnitude should be modest
    assert (alpha.abs() < 0.1).all()


def test_momentum_ranking(synthetic_data):
    """Stock with superior price trend should have higher alpha."""
    cfg = MockConfig()
    data = synthetic_data.copy()
    t_idx = 280
    t_date = data.index[t_idx]

    # Historical prices: AAPL trending sharply up, MSFT flat
    data.loc[data.index[t_idx - 252 : t_idx - 20], "AAPL"] = np.linspace(100, 200, 232)
    data.loc[data.index[t_idx - 252 : t_idx - 20], "MSFT"] = 100.0

    # Set t-21 to t flat to isolate momentum from reversal
    data.loc[data.index[t_idx - 21 : t_idx + 1], :] = 150.0

    alpha = compute_alpha(data, t_date, cfg)

    assert alpha["AAPL"] > alpha["MSFT"]
