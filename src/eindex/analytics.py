"""Performance analytics for backtest results."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def compute_metrics(returns_df: pd.DataFrame) -> dict:
    """Compute key performance metrics from backtest returns.

    Args:
        returns_df: DataFrame with columns (date, portfolio, benchmark).

    Returns:
        dict with CAGR, active_return, tracking_error, information_ratio,
        sharpe, beta, max_drawdown, turnover.
    """
    port = returns_df["portfolio"].values
    bench = returns_df["benchmark"].values
    n_days = len(port)

    if n_days < 2:
        raise ValueError("Need at least 2 days of returns.")

    cum_port = np.prod(1.0 + port)
    cum_bench = np.prod(1.0 + bench)
    years = n_days / TRADING_DAYS

    cagr_port = cum_port ** (1.0 / years) - 1.0 if years > 0 else 0.0
    cagr_bench = cum_bench ** (1.0 / years) - 1.0 if years > 0 else 0.0

    active = port - bench
    active_return = np.mean(active) * TRADING_DAYS
    tracking_error = np.std(active, ddof=1) * np.sqrt(TRADING_DAYS) if n_days > 1 else 0.0
    information_ratio = active_return / tracking_error if tracking_error > 1e-10 else 0.0

    sharpe = (
        (np.mean(port) * TRADING_DAYS) / (np.std(port, ddof=1) * np.sqrt(TRADING_DAYS))
        if np.std(port, ddof=1) > 1e-10
        else 0.0
    )

    bench_var = np.var(bench, ddof=1)
    beta = np.cov(port, bench, ddof=1)[0, 1] / bench_var if bench_var > 1e-10 else 1.0

    cum_returns = np.cumprod(1.0 + port)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / running_max - 1.0
    max_drawdown = float(np.min(drawdowns))

    metrics = {
        "cagr": float(cagr_port),
        "cagr_benchmark": float(cagr_bench),
        "active_return": float(active_return),
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio),
        "sharpe": float(sharpe),
        "beta": float(beta),
        "max_drawdown": float(max_drawdown),
    }

    if abs(information_ratio) > 1.5:
        logger.warning("Information ratio %.2f exceeds 1.5 — verify results.", information_ratio)
    if 0 < tracking_error < 0.005:
        logger.warning(
            "Tracking error %.4f is below 0.5%% — portfolio may be too passive.", tracking_error
        )

    return metrics
