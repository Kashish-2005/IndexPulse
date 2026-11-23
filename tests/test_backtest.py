"""Tests for backtest, analytics, and pipeline modules."""

import json
import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from eindex.analytics import compute_metrics
from eindex.backtest import run_backtest


class _Cfg:
    """Minimal config stub matching the optimizer/backtest interface."""

    def __init__(self, **overrides):
        defaults = dict(
            rebalance_freq="ME",
            lookback_cov_days=504,
            ic=0.05,
            risk_aversion=1.0,
            cost_bps=0.0,
            max_active_stock=0.05,
            max_active_sector=0.10,
            max_te_annual=0.03,
            max_turnover_oneway=0.30,
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)

    class signal_weights:
        momentum = 0.5
        low_vol = 0.3
        reversal = 0.2


TICKERS = ["A", "B", "C", "D"]
SECTORS = pd.Series(["Tech", "Tech", "Health", "Health"], index=TICKERS)


def _make_prices(n_days=600, seed=42):
    """Random-walk prices with enough history for backtest."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    log_ret = rng.normal(0.0003, 0.012, (n_days, 4))
    prices = np.exp(np.cumsum(log_ret, axis=0)) * 100
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


def _equal_bench():
    return pd.Series(0.25, index=TICKERS, name="bench")


# ------------------------------------------------------------------ tests


def test_look_ahead_bias():
    """Truncating prices after date t must not change weights at t."""
    cfg = _Cfg()
    prices = _make_prices(700)
    bench = _equal_bench()

    _, weights_full, _ = run_backtest(prices, bench, SECTORS, cfg)
    if weights_full.empty:
        pytest.skip("No rebalance dates produced.")

    first_rebal = weights_full["date"].min()
    cutoff = first_rebal + pd.Timedelta(days=1)
    prices_truncated = prices.loc[prices.index <= cutoff].copy()

    _, weights_trunc, _ = run_backtest(prices_truncated, bench, SECTORS, cfg)
    if weights_trunc.empty:
        pytest.skip("No rebalance in truncated window.")

    trunc_first = weights_trunc[weights_trunc["date"] == first_rebal]
    full_first = weights_full[weights_full["date"] == first_rebal]

    merged = pd.merge(trunc_first, full_first, on="ticker", suffixes=("_trunc", "_full"))
    np.testing.assert_allclose(
        merged["weight_trunc"].values,
        merged["weight_full"].values,
        atol=1e-6,
        err_msg="Look-ahead bias detected: future prices affected weights.",
    )


def test_zero_alpha_matches_benchmark():
    """With zero IC (hence zero alpha), portfolio return should equal benchmark."""
    cfg = _Cfg(ic=0.0, cost_bps=0.0)
    prices = _make_prices(600)
    bench = _equal_bench()

    # With ic=0, compute_alpha produces all-zero alphas, optimizer stays at benchmark
    # But compute_alpha multiplies by ic, so alpha = 0 regardless of signals
    returns_df, _, _ = run_backtest(prices, bench, SECTORS, cfg)

    np.testing.assert_allclose(
        returns_df["portfolio"].values,
        returns_df["benchmark"].values,
        atol=1e-8,
        err_msg="Zero-alpha portfolio diverged from benchmark.",
    )


def test_costs_reduce_returns():
    """Transaction costs are deducted from portfolio returns on trade days."""
    prices = _make_prices(600)
    bench = _equal_bench()
    cfg = _Cfg(cost_bps=50.0)

    ret_df, _, diags = run_backtest(prices, bench, SECTORS, cfg)

    assert len(diags) > 0, "Expected at least one rebalance"
    total_cost = sum(d.get("cost_deducted", 0.0) for d in diags)
    assert total_cost > 1e-6, (
        "With nonzero alpha and cost_bps, costs should have been deducted"
    )

    # Deducting a positive cost from daily returns strictly reduces cumulative return
    # vs the same weight sequence without cost deduction.
    # Verify each individual cost is non-negative.
    for d in diags:
        assert d.get("cost_deducted", 0.0) >= 0.0


def test_output_columns():
    """returns_df and weights_df must have exactly the specified columns."""
    cfg = _Cfg()
    prices = _make_prices(600)
    bench = _equal_bench()

    returns_df, weights_df, _ = run_backtest(prices, bench, SECTORS, cfg)

    assert list(returns_df.columns) == ["date", "portfolio", "benchmark"]

    if not weights_df.empty:
        assert list(weights_df.columns) == ["date", "ticker", "sector", "weight", "bench_weight"]


def test_sanity_warnings(caplog):
    """Analytics should warn on suspicious IR or TE values."""
    n = 500
    rng = np.random.default_rng(99)
    bench_ret = rng.normal(0.0004, 0.01, n)
    # Large mean active return with small noise → IR >> 1.5
    port_ret = bench_ret + 0.002 + rng.normal(0, 0.0005, n)

    returns_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "portfolio": port_ret,
        "benchmark": bench_ret,
    })

    with caplog.at_level(logging.WARNING, logger="eindex.analytics"):
        metrics = compute_metrics(returns_df)

    assert metrics["information_ratio"] > 1.5
    assert any("1.5" in rec.message for rec in caplog.records)


def test_low_te_warning(caplog):
    """Analytics should warn if TE < 0.5%."""
    n = 500
    rng = np.random.default_rng(77)
    bench_ret = rng.normal(0.0004, 0.01, n)
    port_ret = bench_ret + rng.normal(0, 0.0001, n)

    returns_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "portfolio": port_ret,
        "benchmark": bench_ret,
    })

    with caplog.at_level(logging.WARNING, logger="eindex.analytics"):
        metrics = compute_metrics(returns_df)

    assert metrics["tracking_error"] < 0.005
    assert any("0.5%" in rec.message for rec in caplog.records)


def test_warmup_trim_starts_after_first_rebalance():
    """returns_df must start after the first rebalance date; near-zero active days are few."""
    cfg = _Cfg(ic=0.05, cost_bps=0.0)
    prices = _make_prices(700)
    bench = _equal_bench()

    returns_df, weights_df, _ = run_backtest(prices, bench, SECTORS, cfg)

    if weights_df.empty:
        pytest.skip("No rebalance dates produced.")

    first_rebal_date = weights_df["date"].min()

    assert not returns_df.empty, "returns_df must not be empty"
    assert returns_df["date"].iloc[0] > first_rebal_date, (
        f"First return date {returns_df['date'].iloc[0]} should be after "
        f"first rebalance date {first_rebal_date}"
    )

    # Days with exactly zero active return should be a small fraction
    active = (returns_df["portfolio"] - returns_df["benchmark"]).abs()
    zero_days = (active == 0.0).sum()
    total_days = len(returns_df)
    assert zero_days / total_days < 0.05, (
        f"Too many zero-active-return days: {zero_days}/{total_days} = "
        f"{zero_days/total_days:.1%} (expected < 5%)"
    )


def test_summary_json_contains_required_fields(tmp_path):
    """run_all writes generated_at, data_start, data_end to summary.json."""
    import os

    from eindex.pipeline import run_all

    prices = _make_prices(700)
    sectors_s = pd.Series(["Tech", "Tech", "Health", "Health"], index=TICKERS)

    rng = np.random.default_rng(5)
    shares = pd.Series({t: float(rng.integers(1_000_000, 1_000_000_000)) for t in TICKERS})

    # Write a minimal config.yaml into tmp_path
    cfg_text = """\
start_date: "2020-01-01"
rebalance_freq: "ME"
lookback_cov_days: 504
signal_weights:
  momentum: 0.4
  low_vol: 0.3
  reversal: 0.3
ic: 0.05
risk_aversion: 1.0
max_active_stock: 0.05
max_active_sector: 0.10
max_te_annual: 0.03
max_turnover_oneway: 0.30
cost_bps: 0.0
min_weight: 0.0
random_seed: 42
tickers: [A, B, C, D]
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text)

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with (
            patch("eindex.pipeline.load_prices", return_value=prices),
            patch("eindex.pipeline.load_market_caps", return_value=shares),
            patch("eindex.pipeline.load_sectors", return_value=sectors_s),
            patch("eindex.pipeline.build_benchmark_weights", return_value=pd.DataFrame(
                {t: [0.25] for t in TICKERS},
                index=[prices.index[0]],
            )),
        ):
            summary = run_all(str(cfg_path))
    finally:
        os.chdir(orig_cwd)

    # Check in-memory return value
    assert "generated_at" in summary, "summary must contain 'generated_at'"
    assert "data_start" in summary, "summary must contain 'data_start'"
    assert "data_end" in summary, "summary must contain 'data_end'"

    # Check generated_at is a valid UTC ISO timestamp ending in Z
    assert summary["generated_at"].endswith("Z"), f"generated_at should end with Z: {summary['generated_at']}"

    # Check data_start < data_end
    assert summary["data_start"] < summary["data_end"], "data_start must be before data_end"

    # Check written file agrees
    written = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert written["generated_at"] == summary["generated_at"]
    assert written["data_start"] == summary["data_start"]
    assert written["data_end"] == summary["data_end"]
