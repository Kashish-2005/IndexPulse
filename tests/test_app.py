"""Streamlit AppTest suite for the Enhanced Index dashboard."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def _clear_st_cache():
    """Clear Streamlit cache between tests to prevent cross-contamination."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _make_fixture(tmp: Path) -> None:
    """Write minimal valid results files into tmp/results/."""
    res = tmp / "results"
    res.mkdir()

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=300)
    b = rng.normal(0.0003, 0.01, 300)
    p = b + rng.normal(0.0001, 0.002, 300)
    ret = pd.DataFrame({"date": dates, "portfolio": p, "benchmark": b})
    ret.to_csv(res / "returns.csv", index=False)

    tickers = [f"T{i}" for i in range(10)]
    sectors = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]
    rows = []
    for d in pd.date_range("2021-01-29", periods=12, freq="ME"):
        ws = rng.dirichlet(np.ones(10))
        bw = np.full(10, 0.1)
        for t, s, w, bwi in zip(tickers, sectors, ws, bw, strict=False):
            rows.append({"date": d, "ticker": t, "sector": s, "weight": w, "bench_weight": bwi})
    pd.DataFrame(rows).to_csv(res / "weights.csv", index=False)

    trades = pd.DataFrame({
        "ticker": tickers[:4],
        "current_weight": [0.12, 0.08, 0.11, 0.09],
        "target_weight": [0.10, 0.10, 0.10, 0.10],
        "trade": [-0.02, 0.02, -0.01, 0.01],
    })
    trades.to_csv(res / "latest_trades.csv", index=False)

    summary = {"cagr": 0.08, "tracking_error": 0.02, "n_days": 300}
    with open(res / "summary.json", "w") as f:
        json.dump(summary, f)


def test_demo_mode_no_exceptions():
    """App runs in demo mode (no results/) without any exceptions."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            assert not at.exception, [str(e) for e in at.exception]
        finally:
            os.chdir(orig_cwd)


def test_with_fixture_no_exceptions_no_demo_banner():
    """App runs with fixture data, no exceptions, and demo banner is absent."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _make_fixture(tmp_path)
        orig_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            assert not at.exception, [str(e) for e in at.exception]
            warnings = [w.value for w in at.warning]
            demo_warnings = [w for w in warnings if "demo" in w.lower() or "Demo" in w]
            assert not demo_warnings, (
                f"Demo banner should not appear with real data, got: {demo_warnings}"
            )
        finally:
            os.chdir(orig_cwd)


def test_missing_trades_and_summary_no_exceptions():
    """App runs when latest_trades.csv and summary.json are absent."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _make_fixture(tmp_path)
        (tmp_path / "results" / "latest_trades.csv").unlink()
        (tmp_path / "results" / "summary.json").unlink()
        orig_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            assert not at.exception, [str(e) for e in at.exception]
        finally:
            os.chdir(orig_cwd)
