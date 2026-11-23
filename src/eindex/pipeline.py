"""End-to-end pipeline: data -> signals -> risk -> optimize -> backtest -> analytics."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from eindex.analytics import compute_metrics
from eindex.backtest import run_backtest
from eindex.config import load_config
from eindex.data import build_benchmark_weights, load_market_caps, load_prices, load_sectors

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def run_all(config_path: str = "config.yaml") -> dict:
    """Execute full pipeline and write result files."""
    cfg = load_config(config_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reviewed_moves = {(m.ticker, m.date) for m in cfg.reviewed_extreme_moves}
    logger.info("Loading data for %d tickers from %s", len(cfg.tickers), cfg.start_date)
    prices = load_prices(cfg.tickers, cfg.start_date, reviewed_moves=reviewed_moves)
    tickers = list(prices.columns)

    shares = load_market_caps(tickers)
    sectors = load_sectors(tickers)

    rebal_dates = prices.index.to_series().resample(cfg.rebalance_freq).last().dropna().values
    rebal_dates_in_prices = [d for d in rebal_dates if d in prices.index]
    bench_weights_df = build_benchmark_weights(prices, shares, rebal_dates_in_prices)
    w_bench = bench_weights_df.iloc[0]
    w_bench = w_bench / w_bench.sum()

    logger.info("Running backtest...")
    returns_df, weights_df, diagnostics = run_backtest(prices, w_bench, sectors, cfg)

    logger.info("Computing analytics...")
    metrics = compute_metrics(returns_df)

    if diagnostics:
        total_turnover = sum(d.get("turnover", 0.0) for d in diagnostics)
        metrics["total_turnover"] = total_turnover
        metrics["avg_turnover"] = total_turnover / len(diagnostics)

    returns_df.to_csv(RESULTS_DIR / "returns.csv", index=False)
    weights_df.to_csv(RESULTS_DIR / "weights.csv", index=False)

    if weights_df.empty:
        latest_trades = pd.DataFrame(columns=["ticker", "current_weight", "target_weight", "trade"])
    else:
        last_date = weights_df["date"].max()
        latest = weights_df[weights_df["date"] == last_date].set_index("ticker")
        latest_trades = pd.DataFrame(
            {
                "ticker": latest.index,
                "current_weight": latest["weight"].values,
                "target_weight": latest["bench_weight"].values,
                "trade": latest["bench_weight"].values - latest["weight"].values,
            }
        )
    latest_trades.to_csv(RESULTS_DIR / "latest_trades.csv", index=False)

    summary = {
        **metrics,
        "n_rebalances": len(diagnostics),
        "n_days": len(returns_df),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_start": returns_df["date"].min().strftime("%Y-%m-%d")
        if not returns_df.empty
        else None,
        "data_end": returns_df["date"].max().strftime("%Y-%m-%d") if not returns_df.empty else None,
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results written to %s/", RESULTS_DIR)
    logger.info(
        "Summary: CAGR=%.2f%%, TE=%.2f%%, IR=%.2f",
        metrics["cagr"] * 100,
        metrics["tracking_error"] * 100,
        metrics["information_ratio"],
    )

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_all(path)
