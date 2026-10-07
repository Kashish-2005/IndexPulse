"""Tests for the data layer module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from eindex.data import (
    build_benchmark_weights,
    load_market_caps,
    load_prices,
    load_sectors,
)


@pytest.fixture
def sample_prices():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "AAPL": np.linspace(150, 160, 10),
            "MSFT": np.linspace(250, 260, 10),
        },
        index=dates,
    )


def test_load_prices_validation_drops_bad_ticker(tmp_path, monkeypatch):
    """Test that tickers with >5% NaNs are dropped."""
    monkeypatch.setattr("eindex.data.CACHE_FILE", tmp_path / "cache1.parquet")

    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    # AAPL has 2 NaNs (10% - should drop), MSFT has 0 (should keep)
    df = pd.DataFrame(
        {
            "AAPL": [np.nan, np.nan] + [150.0] * 18,
            "MSFT": [250.0] * 20,
        },
        index=dates,
    )

    with patch("yfinance.download", return_value={"Adj Close": df}):
        result = load_prices(["AAPL", "MSFT"], "2023-01-01")
        assert "AAPL" not in result.columns
        assert "MSFT" in result.columns


def test_load_prices_ffill_logic(tmp_path, monkeypatch):
    """Test that gaps of <= 3 days are filled, while larger gaps fail."""
    # 50 days so 2 NaNs = 4% (< 5% threshold), testing ffill independently of drop logic
    dates = pd.date_range("2023-01-01", periods=50, freq="D")

    # Gap of 2 days (allowed: <=3 days and <=5% missing)
    monkeypatch.setattr("eindex.data.CACHE_FILE", tmp_path / "cache_pass.parquet")
    pass_vals = [100.0, np.nan, np.nan] + [103.0 + i for i in range(47)]
    df_pass = pd.DataFrame({"T1": pass_vals}, index=dates)

    with patch("yfinance.download", return_value={"Adj Close": df_pass}):
        res = load_prices(["T1"], "2023-01-01")
        assert res.isnull().sum().sum() == 0
        assert res.loc[dates[1], "T1"] == 100.0

    # Gap of 4 consecutive days (fails: > 3 days ffill limit)
    # Using 100 days so 4 NaNs = 4% (< 5%), ensuring it fails on ffill, not on drop threshold
    dates_100 = pd.date_range("2023-01-01", periods=100, freq="D")
    monkeypatch.setattr("eindex.data.CACHE_FILE", tmp_path / "cache_fail.parquet")
    fail_vals = [100.0, np.nan, np.nan, np.nan, np.nan] + [105.0 + i for i in range(95)]
    df_fail = pd.DataFrame({"T1": fail_vals}, index=dates_100)

    with patch("yfinance.download", return_value={"Adj Close": df_fail}):
        with pytest.raises(ValueError, match="Data contains NaNs after cleaning"):
            load_prices(["T1"], "2023-01-01")


def test_load_prices_volatility_warning(tmp_path, monkeypatch, caplog):
    """Test that a 40%+ move triggers a log warning."""
    monkeypatch.setattr("eindex.data.CACHE_FILE", tmp_path / "cache_vol.parquet")
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    df = pd.DataFrame({"T1": [100.0, 141.0]}, index=dates)  # 41% move

    with patch("yfinance.download", return_value={"Adj Close": df}):
        load_prices(["T1"], "2023-01-01")
        assert "Extreme price movement" in caplog.text


def test_load_market_caps_fallback():
    """Test market cap fetching and fallback to default."""
    with patch("yfinance.Ticker") as mock_ticker:

        def side_effect(symbol):
            m = MagicMock()
            if symbol == "A":
                m.info = {"sharesOutstanding": 2000}
            else:
                m.info = {}
            return m

        mock_ticker.side_effect = side_effect
        res = load_market_caps(["A", "B"])
        assert res["A"] == 2000
        assert res["B"] == 1e9


def test_load_sectors_fallback(tmp_path, monkeypatch):
    """Test yfinance -> static CSV -> 'Other' fallback chain."""
    static_csv = tmp_path / "static_sectors.csv"
    pd.DataFrame({"sector": ["Tech"]}, index=pd.Index(["S2"], name="ticker")).to_csv(static_csv)
    monkeypatch.setattr("eindex.data.STATIC_SECTORS_FILE", static_csv)

    with patch("yfinance.Ticker") as mock_ticker:

        def side_effect(symbol):
            m = MagicMock()
            m.info = {"sector": "Energy"} if symbol == "S1" else {}
            return m

        mock_ticker.side_effect = side_effect

        res = load_sectors(["S1", "S2", "S3"])
        assert res["S1"] == "Energy"
        assert res["S2"] == "Tech"
        assert res["S3"] == "Other"


def test_build_benchmark_weights(sample_prices):
    """Test weight calculation logic and sum-to-one constraint."""
    shares = pd.Series({"AAPL": 100, "MSFT": 100})
    dates = [sample_prices.index[0], sample_prices.index[-1]]

    weights = build_benchmark_weights(sample_prices, shares, dates)

    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), 1.0)
    # At start, AAPL=150, MSFT=250. AAPL weight = 15000 / (15000 + 25000) = 0.375
    assert weights.iloc[0]["AAPL"] == pytest.approx(0.375)
