"""Enhanced Indexing dashboard.  Run:  streamlit run app/streamlit_app.py

Reads backtest outputs from ./results/ (see CONTRACT below). If missing, shows demo data.

CONTRACT (written by backtest/engine.py):
  results/returns.csv  -> date, portfolio, benchmark   (daily NET returns, decimals)
  results/weights.csv  -> date, ticker, sector, weight, bench_weight  (rebalance dates)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path("results")
TD = 252

st.set_page_config(page_title="Enhanced Index Dashboard", layout="wide")


# ---------------------------------------------------------------- data
def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", "2025-12-31")
    b = rng.normal(0.0004, 0.01, len(dates))
    p = b + rng.normal(0.00008, 0.0015, len(dates))
    returns = pd.DataFrame({"portfolio": p, "benchmark": b}, index=dates)

    tickers = [f"STK{i:02d}" for i in range(50)]
    sectors = rng.choice(["IT", "Banks", "Energy", "FMCG", "Auto", "Pharma"], 50)
    bw = rng.dirichlet(np.ones(50) * 2)
    rows = []
    for d in pd.date_range("2019-01-31", "2025-12-31", freq="ME"):
        w = np.clip(bw + rng.normal(0, 0.004, 50), 0, None)
        w /= w.sum()
        rows += [(d, t, s, wi, bi) for t, s, wi, bi in zip(tickers, sectors, w, bw, strict=False)]
    weights = pd.DataFrame(rows, columns=["date", "ticker", "sector", "weight", "bench_weight"])
    return returns, weights


@st.cache_data(show_spinner=False)
def load() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    rp, wp = RESULTS / "returns.csv", RESULTS / "weights.csv"
    if rp.exists() and wp.exists():
        r = pd.read_csv(rp, parse_dates=["date"]).set_index("date").sort_index()
        w = pd.read_csv(wp, parse_dates=["date"])
        need_r, need_w = {"portfolio", "benchmark"}, {"date", "ticker", "sector", "weight", "bench_weight"}
        if not need_r <= set(r.columns) or not need_w <= set(w.columns):
            st.error(f"CSV columns must include {need_r} and {need_w}")
            st.stop()
        return r.dropna(), w, False
    r, w = _synthetic()
    return r, w, True


# ---------------------------------------------------------------- analytics
def cagr(x: pd.Series) -> float:
    return float((1 + x).prod() ** (TD / len(x)) - 1)


def max_dd(x: pd.Series) -> float:
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())


def compute_metrics(r: pd.DataFrame) -> dict[str, float]:
    p, b = r["portfolio"], r["benchmark"]
    a = p - b
    te = float(a.std() * np.sqrt(TD))
    active = cagr(p) - cagr(b)
    return {
        "Portfolio CAGR": cagr(p),
        "Benchmark CAGR": cagr(b),
        "Active return": active,
        "Tracking error": te,
        "Information ratio": active / te if te > 0 else float("nan"),
        "Sharpe (rf=0)": float(p.mean() / p.std() * np.sqrt(TD)),
        "Beta": float(p.cov(b) / b.var()),
        "Max drawdown": max_dd(p),
        "Benchmark max DD": max_dd(b),
        "Hit rate (monthly)": float(
            (((1 + a).resample("ME").prod() - 1) > 0).mean()
        ),
    }


def turnover_series(w: pd.DataFrame) -> pd.Series:
    piv = w.pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum").fillna(0)
    return (piv.diff().abs().sum(axis=1) / 2).iloc[1:]


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.2%}"


# ---------------------------------------------------------------- UI
returns_all, weights_all, demo = load()

st.title("Enhanced Indexing Portfolio Dashboard")
if demo:
    st.warning("Demo mode: synthetic data. Put returns.csv and weights.csv in ./results/ for real output.")

with st.sidebar:
    st.header("Controls")
    lo, hi = returns_all.index.min().date(), returns_all.index.max().date()
    start, end = st.date_input("Period", (lo, hi), min_value=lo, max_value=hi)
    roll = st.slider("Rolling window (days)", 30, 252, 126, step=6)
    top_n = st.slider("Top N active positions", 5, 25, 10)
    st.header("Constraint limits")
    te_lim = st.number_input("Max tracking error (%)", 0.5, 10.0, 3.0, 0.5) / 100
    stock_lim = st.number_input("Max active weight / stock (%)", 0.5, 10.0, 2.0, 0.5) / 100
    sector_lim = st.number_input("Max active weight / sector (%)", 0.5, 10.0, 3.0, 0.5) / 100
    to_lim = st.number_input("Max turnover / rebalance (%)", 1.0, 100.0, 20.0, 1.0) / 100

r = returns_all.loc[str(start):str(end)]
if len(r) < 60:
    st.error("Select at least ~3 months of data.")
    st.stop()
w_all = weights_all[(weights_all["date"] >= pd.Timestamp(start)) & (weights_all["date"] <= pd.Timestamp(end))]
if w_all.empty:
    st.error("No rebalance dates in selected period.")
    st.stop()

m = compute_metrics(r)
tabs = st.tabs(["Overview", "Risk", "Active positions", "Turnover", "Compliance", "Data"])

# ---- Overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Portfolio CAGR", pct(m["Portfolio CAGR"]))
    c[1].metric("Benchmark CAGR", pct(m["Benchmark CAGR"]))
    c[2].metric("Active return", pct(m["Active return"]))
    c[3].metric("Tracking error", pct(m["Tracking error"]))
    c[4].metric("Information ratio", f"{m['Information ratio']:.2f}")
    c = st.columns(5)
    c[0].metric("Sharpe", f"{m['Sharpe (rf=0)']:.2f}")
    c[1].metric("Beta", f"{m['Beta']:.2f}")
    c[2].metric("Max drawdown", pct(m["Max drawdown"]))
    c[3].metric("Benchmark max DD", pct(m["Benchmark max DD"]))
    c[4].metric("Monthly hit rate", pct(m["Hit rate (monthly)"]))

    if m["Information ratio"] > 1.5 or m["Tracking error"] < 0.005:
        st.info("IR above 1.5 or near-zero TE is unusual. Check for look-ahead bias or missing costs.")

    cum = (1 + r).cumprod()
    fig = go.Figure()
    fig.add_scatter(x=cum.index, y=cum["portfolio"], name="Portfolio")
    fig.add_scatter(x=cum.index, y=cum["benchmark"], name="Benchmark")
    fig.update_layout(title="Growth of 1", height=380, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    act_cum = (1 + (r["portfolio"] - r["benchmark"])).cumprod() - 1
    f2 = go.Figure(go.Scatter(x=act_cum.index, y=act_cum, fill="tozeroy", name="Cumulative active"))
    f2.update_layout(title="Cumulative active return", height=320, yaxis_tickformat=".1%")
    col1.plotly_chart(f2, width="stretch")

    dd = pd.DataFrame({k: (cum[k] / cum[k].cummax() - 1) for k in cum})
    f3 = go.Figure()
    f3.add_scatter(x=dd.index, y=dd["portfolio"], name="Portfolio")
    f3.add_scatter(x=dd.index, y=dd["benchmark"], name="Benchmark")
    f3.update_layout(title="Drawdown", height=320, yaxis_tickformat=".0%")
    col2.plotly_chart(f3, width="stretch")

    mo = (1 + r).resample("ME").prod() - 1
    act_m = (mo["portfolio"] - mo["benchmark"]).to_frame("a")
    act_m["y"], act_m["m"] = act_m.index.year, act_m.index.month
    hm = act_m.pivot(index="y", columns="m", values="a")
    f4 = go.Figure(go.Heatmap(z=hm.values * 100, x=hm.columns, y=hm.index.astype(str),
                              colorscale="RdYlGn", zmid=0, colorbar_title="%"))
    f4.update_layout(title="Monthly active return (%)", height=320)
    st.plotly_chart(f4, width="stretch")

# ---- Risk
with tabs[1]:
    a = r["portfolio"] - r["benchmark"]
    rte = a.rolling(roll).std() * np.sqrt(TD)
    rir = a.rolling(roll).mean() * TD / rte
    rbeta = r["portfolio"].rolling(roll).cov(r["benchmark"]) / r["benchmark"].rolling(roll).var()
    f = go.Figure(go.Scatter(x=rte.index, y=rte, name="Rolling TE"))
    f.add_hline(y=te_lim, line_dash="dash", annotation_text="TE limit")
    f.update_layout(title=f"Rolling tracking error ({roll}d)", yaxis_tickformat=".1%", height=330)
    st.plotly_chart(f, width="stretch")
    c1, c2 = st.columns(2)
    f = go.Figure(go.Scatter(x=rir.index, y=rir))
    f.update_layout(title="Rolling information ratio", height=300)
    c1.plotly_chart(f, width="stretch")
    f = go.Figure(go.Scatter(x=rbeta.index, y=rbeta))
    f.add_hline(y=1, line_dash="dot")
    f.update_layout(title="Rolling beta", height=300)
    c2.plotly_chart(f, width="stretch")
    f = go.Figure(go.Histogram(x=a * 1e4, nbinsx=60))
    f.update_layout(title="Daily active return distribution (bps)", height=300)
    st.plotly_chart(f, width="stretch")

# ---- Active positions
with tabs[2]:
    dates = sorted(w_all["date"].unique())
    sel = st.select_slider("Rebalance date", options=dates, value=dates[-1],
                           format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    cur = w_all[w_all["date"] == sel].copy()
    cur["active"] = cur["weight"] - cur["bench_weight"]
    top = pd.concat([cur.nlargest(top_n, "active"), cur.nsmallest(top_n, "active")]).drop_duplicates("ticker")
    top = top.sort_values("active")
    f = go.Figure(go.Bar(x=top["active"], y=top["ticker"], orientation="h",
                         marker_color=np.where(top["active"] >= 0, "#2e7d32", "#c62828")))
    f.update_layout(title=f"Top {top_n} overweights / underweights", xaxis_tickformat=".2%",
                    height=max(350, 22 * len(top)))
    st.plotly_chart(f, width="stretch")

    sec = cur.groupby("sector")[["weight", "bench_weight"]].sum()
    sec["active"] = sec["weight"] - sec["bench_weight"]
    c1, c2 = st.columns(2)
    f = go.Figure()
    f.add_bar(x=sec.index, y=sec["weight"], name="Portfolio")
    f.add_bar(x=sec.index, y=sec["bench_weight"], name="Benchmark")
    f.update_layout(title="Sector weights", barmode="group", yaxis_tickformat=".0%", height=330)
    c1.plotly_chart(f, width="stretch")
    f = go.Figure(go.Bar(x=sec.index, y=sec["active"],
                         marker_color=np.where(sec["active"] >= 0, "#2e7d32", "#c62828")))
    f.update_layout(title="Sector active weights", yaxis_tickformat=".2%", height=330)
    c2.plotly_chart(f, width="stretch")

    st.dataframe(cur.sort_values("weight", ascending=False)
                 .style.format({"weight": "{:.2%}", "bench_weight": "{:.2%}", "active": "{:+.2%}"}),
                 width="stretch", hide_index=True)

# ---- Turnover
with tabs[3]:
    to = turnover_series(w_all)
    if to.empty:
        st.info("Need at least two rebalance dates.")
    else:
        f = go.Figure(go.Bar(x=to.index, y=to))
        f.add_hline(y=to_lim, line_dash="dash", annotation_text="Turnover limit")
        f.update_layout(title="One-way turnover per rebalance", yaxis_tickformat=".1%", height=350)
        st.plotly_chart(f, width="stretch")
        c = st.columns(3)
        c[0].metric("Average turnover", pct(to.mean()))
        c[1].metric("Max turnover", pct(to.max()))
        years = max((to.index[-1] - to.index[0]).days / 365.25, 1e-9)
        c[2].metric("Annualized turnover", pct(to.sum() / years))

# ---- Compliance
with tabs[4]:
    rows = []
    for d, g in w_all.groupby("date"):
        act = g["weight"] - g["bench_weight"]
        sact = g.groupby("sector")[["weight", "bench_weight"]].sum().diff(axis=1).iloc[:, 1]
        rows.append({"date": d, "sum_w": g["weight"].sum(), "min_w": g["weight"].min(),
                     "max_stock_active": act.abs().max(), "max_sector_active": sact.abs().max()})
    comp = pd.DataFrame(rows).set_index("date")
    to = turnover_series(w_all)
    comp["turnover"] = to.reindex(comp.index)
    tol = 1e-6
    checks = pd.DataFrame({
        "Fully invested (sum = 1)": (comp["sum_w"] - 1).abs() <= 1e-4,
        "Long-only": comp["min_w"] >= -tol,
        "Stock active limit": comp["max_stock_active"] <= stock_lim + tol,
        "Sector active limit": comp["max_sector_active"] <= sector_lim + tol,
        "Turnover limit": comp["turnover"].fillna(0) <= to_lim + tol,
    })
    summary = pd.DataFrame({
        "Pass rate": checks.mean().map("{:.0%}".format),
        "Breaches": (~checks).sum().astype(str),
    })
    summary.loc["Realized TE vs limit"] = [
        "OK" if m["Tracking error"] <= te_lim else "BREACH", f"{m['Tracking error']:.2%}"]
    st.dataframe(summary, width="stretch")
    bad = comp[~checks.all(axis=1)]
    if bad.empty:
        st.success("No constraint breaches on any rebalance date.")
    else:
        st.error(f"{len(bad)} rebalance date(s) with breaches")
        st.dataframe(bad.style.format("{:.4f}"), width="stretch")

# ---- Data
with tabs[5]:
    st.download_button("Download returns (CSV)", r.to_csv().encode(), "returns.csv")
    st.download_button("Download weights (CSV)", w_all.to_csv(index=False).encode(), "weights.csv")
    st.dataframe(pd.Series(m, name="value").to_frame(), width="stretch")
