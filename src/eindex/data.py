"""Data ingestion, caching, validation, and benchmark weight construction."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "prices.parquet"
STATIC_SECTORS_FILE = Path("data/static_sectors.csv")
MAX_MISSING_PCT = 0.05
MAX_FFILL_DAYS = 3
VOLATILITY_THRESHOLD = 0.40
CACHE_EXPIRY_HOURS = 24
DEFAULT_SHARES = 1e9
WEIGHT_TOLERANCE = 1e-6


def validate_and_clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Validate price data: drop >5% missing, forward-fill <=3 days, check NaNs and anomalies."""
    if df.empty:
        raise ValueError("Input price DataFrame is empty.")

    # 1. Validation: Drop tickers with > 5% missing data
    missing_pct = df.isnull().mean()
    bad_tickers = missing_pct[missing_pct > MAX_MISSING_PCT].index.tolist()
    if bad_tickers:
        logger.warning(
            "Dropping tickers with >%d%% missing: %s",
            int(MAX_MISSING_PCT * 100),
            bad_tickers,
        )
        df = df.drop(columns=bad_tickers)

    if df.shape[1] == 0:
        raise ValueError("All tickers dropped due to excessive missing data.")

    # 2. Forward-fill short gaps
    df = df.ffill(limit=MAX_FFILL_DAYS)

    # 3. Fail loudly on remaining NaNs
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Data contains NaNs after cleaning in columns: {nan_cols}")

    # 4. Volatility check
    returns = df.pct_change().abs()
    if (returns > VOLATILITY_THRESHOLD).any().any():
        logger.warning(
            "Extreme price movement (>%d%%) detected in returns.",
            int(VOLATILITY_THRESHOLD * 100),
        )

    return df


def load_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Fetch adjusted close prices with caching and strict validation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    use_cache = False
    if CACHE_FILE.exists():
        mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - mtime < timedelta(hours=CACHE_EXPIRY_HOURS):
            use_cache = True

    if use_cache:
        logger.info("Loading prices from cache: %s", CACHE_FILE)
        df = pd.read_parquet(CACHE_FILE)
        available = [t for t in tickers if t in df.columns]
        if available:
            df = df.loc[start:, available]
            return validate_and_clean_prices(df)

    logger.info("Fetching prices from yfinance for %d tickers", len(tickers))
    raw_df = yf.download(tickers, start=start, progress=False, group_by="column", auto_adjust=True)

    if raw_df is None or (isinstance(raw_df, pd.DataFrame) and raw_df.empty):
        raise ValueError(f"No data returned from yfinance for start date {start}")

    # Extract Close / Adj Close cleanly from DataFrame or MultiIndex
    if isinstance(raw_df, pd.DataFrame):
        if isinstance(raw_df.columns, pd.MultiIndex):
            if "Close" in raw_df.columns.levels[0]:
                df = raw_df["Close"]
            elif "Adj Close" in raw_df.columns.levels[0]:
                df = raw_df["Adj Close"]
            else:
                df = raw_df.xs(raw_df.columns.levels[0][0], level=0, axis=1)
        else:
            col = "Close" if "Close" in raw_df.columns else "Adj Close"
            df = raw_df[[col]] if col in raw_df.columns else raw_df
    elif isinstance(raw_df, dict):  # Support mock dict returns
        df = raw_df.get("Adj Close", raw_df.get("Close"))
    else:
        df = pd.DataFrame(raw_df)

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    df = validate_and_clean_prices(df)
    df.to_parquet(CACHE_FILE)
    return df


def load_market_caps(tickers: list[str]) -> pd.Series:
    """Fetch shares outstanding to approximate market cap weights."""
    shares_map: dict[str, float] = {}
    for ticker in tickers:
        try:
            t_info = yf.Ticker(ticker).info or {}
            shares = t_info.get("sharesOutstanding")
            shares_map[ticker] = float(shares) if shares else DEFAULT_SHARES
        except Exception as e:
            logger.error("Failed to fetch shares for %s: %s. Using default.", ticker, e)
            shares_map[ticker] = DEFAULT_SHARES

    return pd.Series(shares_map, index=tickers, name="shares_outstanding")


def load_sectors(tickers: list[str]) -> pd.Series:
    """Map tickers to sectors using yfinance with a local CSV fallback."""
    sector_map: dict[str, str] = {}
    static_df = pd.DataFrame()

    if STATIC_SECTORS_FILE.exists():
        static_df = pd.read_csv(STATIC_SECTORS_FILE, index_col=0)

    for ticker in tickers:
        sec = None
        try:
            t_info = yf.Ticker(ticker).info or {}
            sec = t_info.get("sector")
        except Exception:
            pass

        if sec:
            sector_map[ticker] = sec
            continue

        if not static_df.empty and ticker in static_df.index:
            sector_map[ticker] = str(static_df.loc[ticker, "sector"])
        else:
            sector_map[ticker] = "Other"

    return pd.Series(sector_map, index=tickers, name="sector")


def build_benchmark_weights(
    prices: pd.DataFrame, shares: pd.Series, dates: list[pd.Timestamp]
) -> pd.DataFrame:
    """Calculate normalized market-cap weights for specific rebalance dates."""
    missing_dates = [d for d in dates if d not in prices.index]
    if missing_dates:
        raise ValueError(f"Rebalance dates {missing_dates} missing from price index")

    common_tickers = prices.columns.intersection(shares.index)
    if common_tickers.empty:
        raise ValueError("No common tickers between prices and shares series")

    p_subset = prices.loc[dates, common_tickers]
    s_subset = shares.loc[common_tickers]

    mcap = p_subset.mul(s_subset, axis=1)
    weights = mcap.div(mcap.sum(axis=1), axis=0)

    if (weights < 0).any().any():
        raise ValueError("Negative prices or shares led to negative weights")

    sums = weights.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=WEIGHT_TOLERANCE):
        raise ValueError("Benchmark weights do not sum to 1.0")

    return weights
