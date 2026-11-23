"""Tests for the portfolio optimizer."""

import numpy as np
import pandas as pd
import pytest

from eindex.optimizer import optimize_portfolio


class _Cfg:
    """Minimal config stub for optimizer tests."""

    def __init__(self, **overrides):
        defaults = dict(
            risk_aversion=1.0,
            cost_bps=5.0,
            max_active_stock=0.05,
            max_active_sector=0.10,
            max_te_annual=0.03,
            max_turnover_oneway=0.20,
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)


TICKERS = ["A", "B", "C", "D"]
SECTORS = pd.Series(["Tech", "Tech", "Health", "Health"], index=TICKERS)


def _equal_bench():
    return pd.Series(0.25, index=TICKERS, name="bench")


def _cov_diagonal(var: float = 0.04):
    """Diagonal annualized covariance (vol = 20% each)."""
    arr = np.eye(4) * var
    return pd.DataFrame(arr, index=TICKERS, columns=TICKERS)


# ------------------------------------------------------------------ tests


def test_zero_alpha_returns_benchmark():
    """With zero alpha, optimizer should stay at benchmark."""
    cfg = _Cfg()
    alpha = pd.Series(0.0, index=TICKERS)
    w_bench = _equal_bench()
    w_prev = _equal_bench()
    cov = _cov_diagonal()

    w, diag = optimize_portfolio(alpha, cov, w_bench, w_prev, SECTORS, cfg)

    np.testing.assert_allclose(w.values, w_bench.values, atol=1e-4)
    assert diag["step"] == "optimal"


def test_weights_sum_to_one_and_nonneg():
    cfg = _Cfg()
    alpha = pd.Series([0.02, -0.01, 0.01, -0.005], index=TICKERS)
    w, _ = optimize_portfolio(alpha, _cov_diagonal(), _equal_bench(), _equal_bench(), SECTORS, cfg)

    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_realized_te_within_limit():
    cfg = _Cfg(max_te_annual=0.03)
    alpha = pd.Series([0.03, -0.02, 0.01, -0.01], index=TICKERS)
    w, diag = optimize_portfolio(alpha, _cov_diagonal(), _equal_bench(), _equal_bench(), SECTORS, cfg)

    assert diag["realized_te"] <= cfg.max_te_annual + 1e-6


def test_active_weight_within_limit():
    cfg = _Cfg(max_active_stock=0.05)
    alpha = pd.Series([0.05, -0.03, 0.02, -0.01], index=TICKERS)
    w, _ = optimize_portfolio(alpha, _cov_diagonal(), _equal_bench(), _equal_bench(), SECTORS, cfg)

    active = (w - _equal_bench()).abs()
    assert (active <= cfg.max_active_stock + 1e-6).all()


def test_turnover_within_limit():
    cfg = _Cfg(max_turnover_oneway=0.20)
    alpha = pd.Series([0.04, -0.02, 0.01, -0.01], index=TICKERS)
    w, diag = optimize_portfolio(alpha, _cov_diagonal(), _equal_bench(), _equal_bench(), SECTORS, cfg)

    assert diag["turnover"] <= cfg.max_turnover_oneway + 1e-6


def test_fallback_on_infeasible():
    """Impossible constraints trigger fallback and still return valid weights."""
    cfg = _Cfg(
        max_te_annual=1e-10,
        max_active_stock=1e-10,
        max_active_sector=1e-10,
        max_turnover_oneway=1e-10,
    )
    alpha = pd.Series([0.1, -0.1, 0.05, -0.05], index=TICKERS)
    # w_prev far from benchmark so even benchmark requires turnover exceeding the tiny limit
    w_prev = pd.Series([0.7, 0.1, 0.1, 0.1], index=TICKERS, name="bench")
    w, diag = optimize_portfolio(alpha, _cov_diagonal(), _equal_bench(), w_prev, SECTORS, cfg)

    assert diag["step"] == "benchmark_fallback"
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_higher_alpha_increases_weight():
    """A stock with higher alpha should get overweighted vs benchmark."""
    cfg = _Cfg()
    alpha = pd.Series([0.05, 0.0, 0.0, 0.0], index=TICKERS)
    w, _ = optimize_portfolio(alpha, _cov_diagonal(), _equal_bench(), _equal_bench(), SECTORS, cfg)

    assert w["A"] > _equal_bench()["A"] + 1e-6


def test_mismatched_indexes_raises():
    cfg = _Cfg()
    alpha = pd.Series([0.01, 0.02], index=["A", "B"])
    with pytest.raises(ValueError, match="Index mismatch"):
        optimize_portfolio(alpha, _cov_diagonal(), _equal_bench(), _equal_bench(), SECTORS, cfg)
