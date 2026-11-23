"""Walk-forward backtesting engine."""

import logging

import numpy as np
import pandas as pd

from eindex.optimizer import optimize_portfolio
from eindex.risk import ledoit_wolf_cov
from eindex.signals import compute_alpha

logger = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 504


_PERIOD_FREQ_MAP = {"ME": "M", "QE": "Q", "YE": "Y"}


def _rebalance_dates(prices: pd.DataFrame, freq: str) -> list[pd.Timestamp]:
    """Month-end dates from the price index with enough history."""
    all_dates = prices.index.sort_values()
    period_freq = _PERIOD_FREQ_MAP.get(freq, freq)
    month_ends = all_dates.to_series().groupby(all_dates.to_period(period_freq)).last()
    cutoff = all_dates[MIN_HISTORY_DAYS - 1]
    return [d for d in month_ends if d >= cutoff]


def _drift_weights(weights: np.ndarray, returns_1d: np.ndarray) -> np.ndarray:
    """Drift weights forward by one day's returns."""
    grown = weights * (1.0 + returns_1d)
    total = grown.sum()
    if total <= 0:
        return weights
    return grown / total


def run_backtest(
    prices: pd.DataFrame,
    benchmark_weights: pd.Series,
    sectors: pd.Series,
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Walk-forward backtest over month-end rebalance dates.

    Args:
        prices: Adjusted close prices (T x N), DatetimeIndex.
        benchmark_weights: Equal or cap-weighted benchmark (Series indexed by ticker).
        sectors: Sector mapping (Series indexed by ticker).
        cfg: Config object.

    Returns:
        (returns_df, weights_df, diagnostics_list)
        returns_df columns: date, portfolio, benchmark
        weights_df columns: date, ticker, sector, weight, bench_weight
    """
    tickers = sorted(prices.columns.intersection(benchmark_weights.index).intersection(sectors.index))
    prices = prices[tickers].copy()
    benchmark_weights = benchmark_weights.loc[tickers]
    benchmark_weights = benchmark_weights / benchmark_weights.sum()
    sectors = sectors.loc[tickers]

    daily_returns = prices.pct_change().iloc[1:]
    dates = daily_returns.index
    rebal_dates = _rebalance_dates(prices, cfg.rebalance_freq)

    if not rebal_dates:
        raise ValueError("No rebalance dates with sufficient history.")

    w_port = benchmark_weights.values.copy()
    w_bench = benchmark_weights.values.copy()
    w_prev = benchmark_weights.values.copy()

    port_returns = []
    bench_returns = []
    return_dates = []
    all_weights = []
    all_diagnostics = []
    rebal_set = set(rebal_dates)
    first_rebal_date = rebal_dates[0]
    # Only record returns from the trading day after the first rebalance
    recording = False

    for date in dates:
        ret = daily_returns.loc[date].values

        if date in rebal_set:
            hist_prices = prices.loc[prices.index <= date]
            if len(hist_prices) < MIN_HISTORY_DAYS:
                continue

            lookback = min(cfg.lookback_cov_days, len(hist_prices) - 1)
            ret_window = hist_prices.pct_change().iloc[-lookback:]
            ret_window = ret_window.dropna(how="all")

            try:
                alpha = compute_alpha(prices, date, cfg)
                alpha = alpha.reindex(tickers).fillna(0.0)
                cov = ledoit_wolf_cov(ret_window)

                w_bench_s = pd.Series(w_bench, index=tickers)
                w_prev_s = pd.Series(w_prev, index=tickers)

                new_w, diag = optimize_portfolio(alpha, cov, w_bench_s, w_prev_s, sectors, cfg)
                diag["date"] = date
                all_diagnostics.append(diag)

                turnover_oneway = 0.5 * np.sum(np.abs(new_w.values - w_port))
                cost = (cfg.cost_bps / 1e4) * turnover_oneway
                diag["cost_deducted"] = cost

                w_prev = w_port.copy()
                w_port = new_w.reindex(tickers).values.copy()
            except Exception as e:
                logger.warning("Rebalance failed at %s: %s. Holding previous weights.", date, e)
                cost = 0.0

            for t_i, ticker in enumerate(tickers):
                all_weights.append({
                    "date": date,
                    "ticker": ticker,
                    "sector": sectors[ticker],
                    "weight": w_port[t_i],
                    "bench_weight": w_bench[t_i],
                })
        else:
            cost = 0.0

        # Start recording returns on the first day after the first rebalance
        if not recording and date > first_rebal_date:
            recording = True

        port_ret = float(np.dot(w_port, ret)) - cost
        bench_ret = float(np.dot(w_bench, ret))

        if recording:
            port_returns.append(port_ret)
            bench_returns.append(bench_ret)
            return_dates.append(date)

        w_port = _drift_weights(w_port, ret)
        w_bench = _drift_weights(w_bench, ret)

    returns_df = pd.DataFrame({
        "date": return_dates,
        "portfolio": port_returns,
        "benchmark": bench_returns,
    })

    weights_df = pd.DataFrame(all_weights)

    return returns_df, weights_df, all_diagnostics
