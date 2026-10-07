"""Alpha signal generation and cross-sectional factor combination."""

import logging

import numpy as np
import pandas as pd
from scipy.stats import mstats

from eindex.config import Config

logger = logging.getLogger(__name__)

MOM_START = 252  # 12 months
MOM_END = 21  # 1 month
VOL_WINDOW = 60
REV_WINDOW = 21
TRADING_DAYS_YEAR = 252


def _process_cross_section(series: pd.Series) -> pd.Series:
    """Winsorize at 1st/99th percentiles and cross-sectional z-score."""
    if series.empty:
        return series

    # mstats.winsorize returns a masked array; np.asarray extracts clean values
    winsorized = np.asarray(mstats.winsorize(series.to_numpy(), limits=[0.01, 0.01]))
    series_win = pd.Series(winsorized, index=series.index, dtype=float)

    std = series_win.std(ddof=0)
    if std < 1e-12 or np.isnan(std):
        logger.warning("Signal has zero standard deviation; returning zeros.")
        return pd.Series(0.0, index=series.index)

    return (series_win - series_win.mean()) / std


def compute_alpha(prices: pd.DataFrame, date: pd.Timestamp, cfg: Config) -> pd.Series:
    """Compute expected excess returns (alpha) using momentum, low-vol, and reversal signals.

    Args:
        prices: Adjusted close prices (T x N).
        date: Rebalance date (t).
        cfg: Pydantic Config object containing weights and IC.

    Returns:
        pd.Series: Alpha values indexed by ticker, name='alpha'.

    Raises:
        ValueError: If history < 252 rows or NaNs found in output.
    """
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    # 1. Strict point-in-time filter (no look-ahead)
    hist = prices.loc[prices.index <= date].copy()

    # 2. History check
    if len(hist) < TRADING_DAYS_YEAR:
        raise ValueError(
            f"Insufficient history at {date}: required {TRADING_DAYS_YEAR}, found {len(hist)}"
        )

    # 3. Calculate Signals
    # 12-1 Momentum: Return from t-252 to t-21
    p_t_minus_21 = hist.iloc[-MOM_END]
    p_t_minus_252 = hist.iloc[-MOM_START]
    mom_raw = (p_t_minus_21 / p_t_minus_252) - 1.0

    # Low Volatility: -std(returns[-60:])
    returns = hist.pct_change()
    returns_60d = returns.iloc[-VOL_WINDOW:]
    std_60d = returns_60d.std(ddof=1)
    low_vol_raw = -std_60d

    # 1-month reversal: -(P_t / P_{t-21} - 1)
    p_t = hist.iloc[-1]
    rev_raw = -(p_t / p_t_minus_21 - 1.0)

    # 4. Cross-sectional processing
    z_mom = _process_cross_section(mom_raw)
    z_vol = _process_cross_section(low_vol_raw)
    z_rev = _process_cross_section(rev_raw)

    # Combine signals with weights
    combined_scores = (
        cfg.signal_weights.momentum * z_mom
        + cfg.signal_weights.low_vol * z_vol
        + cfg.signal_weights.reversal * z_rev
    )

    # Re-z-score the combined score
    z_combined = _process_cross_section(combined_scores)

    # 5. Expected excess return (alpha)
    # alpha_i = IC * annualized_vol_i * z_combined_i
    ann_vol = std_60d * np.sqrt(TRADING_DAYS_YEAR)
    alpha = cfg.ic * ann_vol * z_combined
    alpha.name = "alpha"

    if alpha.isna().any():
        nan_tickers = alpha.index[alpha.isna()].tolist()
        raise ValueError(f"Alpha calculation resulted in NaNs for: {nan_tickers}")

    return alpha
