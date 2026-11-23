"""Enhanced Indexing dashboard.  Run:  streamlit run app/streamlit_app.py

Reads backtest outputs from ./results/ (see CONTRACT below). If missing, shows demo data.

CONTRACT (written by backtest/engine.py):
  results/returns.csv  -> date, portfolio, benchmark   (daily NET returns, decimals)
  results/weights.csv  -> date, ticker, sector, weight, bench_weight  (rebalance dates)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path("results")
TD = 252

st.set_page_config(
    page_title="IndexPulse | Enhanced Index Dashboard",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- palette & theme
PORT_COLOR = "#0d9488"
BENCH_COLOR = "#94a3b8"
GREEN = "#059669"
RED = "#dc2626"
AMBER = "#d97706"
PLOTLY_COLORWAY = [PORT_COLOR, BENCH_COLOR, "#6366f1", "#f59e0b", "#ef4444", "#8b5cf6"]


def _apply_plotly_theme(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
        colorway=PLOTLY_COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#e2e8f0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=48, r=16, t=48, b=40),
        xaxis=dict(gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(gridcolor="#f1f5f9", zeroline=False),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    return fig


# ---------------------------------------------------------------- CSS
_CSS = """
<style>
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

:root {
  --accent: #0d9488; --accent-light: #ccfbf1; --accent-dark: #0f766e;
  --green: #059669; --red: #dc2626; --amber: #d97706;
  --surface: #ffffff; --surface-alt: #f8fafc; --border: #e2e8f0;
  --text-primary: #0f172a; --text-secondary: #64748b;
  --shadow-sm: 0 1px 2px rgba(0,0,0,.05); --shadow-md: 0 4px 12px rgba(0,0,0,.08);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --accent: #2dd4bf; --accent-light: #042f2e; --accent-dark: #5eead4;
    --green: #34d399; --red: #f87171; --amber: #fbbf24;
    --surface: #1e293b; --surface-alt: #0f172a; --border: #334155;
    --text-primary: #f1f5f9; --text-secondary: #94a3b8;
    --shadow-sm: 0 1px 2px rgba(0,0,0,.2); --shadow-md: 0 4px 12px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"] {
  --accent: #2dd4bf; --accent-light: #042f2e; --accent-dark: #5eead4;
  --green: #34d399; --red: #f87171; --amber: #fbbf24;
  --surface: #1e293b; --surface-alt: #0f172a; --border: #334155;
  --text-primary: #f1f5f9; --text-secondary: #94a3b8;
  --shadow-sm: 0 1px 2px rgba(0,0,0,.2); --shadow-md: 0 4px 12px rgba(0,0,0,.3);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px; }

@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.kpi-row { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
.kpi-card {
  flex: 1 1 160px; min-width: 150px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; box-shadow: var(--shadow-sm);
  animation: fadeSlideUp .45s ease-out both;
  transition: box-shadow .2s, transform .2s;
}
.kpi-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.kpi-card:nth-child(1) { animation-delay: .00s; }
.kpi-card:nth-child(2) { animation-delay: .06s; }
.kpi-card:nth-child(3) { animation-delay: .12s; }
.kpi-card:nth-child(4) { animation-delay: .18s; }
.kpi-card:nth-child(5) { animation-delay: .24s; }
.kpi-label { font-size: .72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .04em; color: var(--text-secondary); margin-bottom: 4px; }
.kpi-value { font-size: 1.5rem; font-weight: 700; color: var(--text-primary);
  font-variant-numeric: tabular-nums; }
.kpi-hint { font-size: .7rem; margin-top: 2px; }
.kpi-hint.pos { color: var(--green); }
.kpi-hint.neg { color: var(--red); }
.kpi-hint.neutral { color: var(--text-secondary); }

.chip {
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: .72rem; font-weight: 600; letter-spacing: .02em;
}
.chip-green { background: #d1fae5; color: #065f46; }
.chip-red   { background: #fee2e2; color: #991b1b; }
.chip-gray  { background: #f1f5f9; color: #475569; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .chip-green { background: #064e3b; color: #6ee7b7; }
  :root:not([data-theme="light"]) .chip-red   { background: #450a0a; color: #fca5a5; }
  :root:not([data-theme="light"]) .chip-gray  { background: #334155; color: #94a3b8; }
}
:root[data-theme="dark"] .chip-green { background: #064e3b; color: #6ee7b7; }
:root[data-theme="dark"] .chip-red   { background: #450a0a; color: #fca5a5; }
:root[data-theme="dark"] .chip-gray  { background: #334155; color: #94a3b8; }

.trade-badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: .72rem; font-weight: 600;
}
.trade-buy  { background: #d1fae5; color: #065f46; }
.trade-sell { background: #fee2e2; color: #991b1b; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .trade-buy  { background: #064e3b; color: #6ee7b7; }
  :root:not([data-theme="light"]) .trade-sell { background: #450a0a; color: #fca5a5; }
}
:root[data-theme="dark"] .trade-buy  { background: #064e3b; color: #6ee7b7; }
:root[data-theme="dark"] .trade-sell { background: #450a0a; color: #fca5a5; }

.section-header { font-size: .85rem; font-weight: 700; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: .06em; margin: 20px 0 8px; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def _kpi_card(label: str, value: str, hint: str = "", hint_cls: str = "neutral") -> str:
    h = f'<div class="kpi-hint {hint_cls}">{hint}</div>' if hint else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{h}</div>'
    )


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.2%}"


def _fmt_bps(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 1e4:+.0f} bps"


def _hint_cls(x: float, invert: bool = False) -> str:
    if pd.isna(x):
        return "neutral"
    if invert:
        return "pos" if x <= 0 else "neg"
    return "pos" if x >= 0 else "neg"


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


@st.cache_data(show_spinner=False)
def load_latest_trades() -> pd.DataFrame | None:
    fp = RESULTS / "latest_trades.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    need = {"ticker", "current_weight", "target_weight", "trade"}
    if not need <= set(df.columns):
        return None
    return df


@st.cache_data(show_spinner=False)
def load_summary() -> dict | None:
    fp = RESULTS / "summary.json"
    if not fp.exists():
        return None
    try:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------- analytics (unchanged logic)
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


# ---------------------------------------------------------------- load data
with st.spinner("Loading data..."):
    returns_all, weights_all, demo = load()
    latest_trades = load_latest_trades()
    summary_json = load_summary()


# ---------------------------------------------------------------- header
def _build_header(m_dict: dict[str, float] | None, is_demo: bool, summary: dict | None) -> None:
    hdr = '<div style="margin-bottom:20px;">'
    hdr += (
        '<div style="font-size:1.8rem;font-weight:800;line-height:1.2;'
        'color:#1e3a8a;color:var(--accent-title, #1e3a8a);">'
    )
    hdr += "IndexPulse</div>"
    hdr += '<div style="font-size:.95rem;color:var(--text-secondary);margin-bottom:8px;">'
    hdr += "Enhanced Index vs Benchmark</div>"
    chips = ""
    if summary:
        generated_at = summary.get("generated_at", "N/A")
        data_end = summary.get("data_end")
        label = f"Last updated: {generated_at}"
        if data_end:
            label += f" | data through {data_end}"
        chips += f'<span class="chip chip-gray">{label}</span> '
    if m_dict is not None:
        compliant = not (m_dict.get("Tracking error", 0) > 0.03 or m_dict.get("Information ratio", 0) > 1.5)
        if compliant:
            chips += '<span class="chip chip-green">Compliant</span>'
        else:
            chips += '<span class="chip chip-red">Breach</span>'
    if chips:
        hdr += f'<div style="margin-top:4px;">{chips}</div>'
    hdr += "</div>"
    st.markdown(hdr, unsafe_allow_html=True)

    if is_demo:
        st.warning(
            "Demo mode — showing synthetic data. "
            "Run the pipeline (`python -m eindex.pipeline`) to generate real results in `./results/`."
        )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div style="font-size:1.1rem;font-weight:700;margin-bottom:12px;">⚙️ Controls</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">\U0001f4c5 Date range</div>', unsafe_allow_html=True)
    lo, hi = returns_all.index.min().date(), returns_all.index.max().date()
    start, end = st.date_input(
        "Period", (lo, hi), min_value=lo, max_value=hi,
        help="Select the backtest evaluation window.",
    )

    st.markdown('<div class="section-header">\U0001f4ca Analysis</div>', unsafe_allow_html=True)
    roll = st.slider("Rolling window (days)", 30, 252, 126, step=6,
                     help="Window for rolling risk metrics (TE, IR, beta).")
    top_n = st.slider("Top N active positions", 5, 25, 10,
                      help="Number of overweights / underweights to display.")

    st.markdown('<div class="section-header">\U0001f6e1️ Constraint limits</div>', unsafe_allow_html=True)
    te_lim = st.number_input("Max tracking error (%)", 0.5, 10.0, 3.0, 0.5,
                             help="Annualized TE ceiling for compliance check.") / 100
    stock_lim = st.number_input("Max active weight / stock (%)", 0.5, 10.0, 2.0, 0.5,
                                help="Per-stock active weight limit.") / 100
    sector_lim = st.number_input("Max active weight / sector (%)", 0.5, 10.0, 3.0, 0.5,
                                 help="Per-sector active weight limit.") / 100
    to_lim = st.number_input("Max turnover / rebalance (%)", 1.0, 100.0, 20.0, 1.0,
                             help="One-way turnover cap per rebalance.") / 100

    st.divider()
    if st.button("↻ Reset filters", width="stretch"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------- filter data
r = returns_all.loc[str(start):str(end)]
if len(r) < 60:
    st.error("Select at least ~3 months of data.")
    st.stop()
w_all = weights_all[(weights_all["date"] >= pd.Timestamp(start)) & (weights_all["date"] <= pd.Timestamp(end))]
if w_all.empty:
    st.error("No rebalance dates in selected period.")
    st.stop()

m = compute_metrics(r)

# ---------------------------------------------------------------- header (needs m)
_build_header(m, demo, summary_json)

# ---------------------------------------------------------------- tabs
tabs = st.tabs([
    "\U0001f4c8 Overview", "\U0001f6e1️ Risk", "⚖️ Active Positions",
    "\U0001f504 Turnover", "✅ Compliance", "\U0001f4cb Latest Trades",
    "\U0001f4be Data", "\U0001f4bc My Portfolio", "ℹ️ About",
])

# ---- Overview
with tabs[0]:
    row1 = "".join([
        _kpi_card("Portfolio CAGR", pct(m["Portfolio CAGR"]),
                  _fmt_bps(m["Portfolio CAGR"] - m["Benchmark CAGR"]) + " vs bench",
                  _hint_cls(m["Portfolio CAGR"] - m["Benchmark CAGR"])),
        _kpi_card("Benchmark CAGR", pct(m["Benchmark CAGR"])),
        _kpi_card("Active Return", pct(m["Active return"]), "",
                  _hint_cls(m["Active return"])),
        _kpi_card("Tracking Error", pct(m["Tracking error"]),
                  "annualized", "neutral"),
        _kpi_card("Information Ratio",
                  f"{m['Information ratio']:.2f}" if not pd.isna(m["Information ratio"]) else "n/a",
                  "active / TE", _hint_cls(m["Information ratio"])),
    ])
    st.markdown(f'<div class="kpi-row">{row1}</div>', unsafe_allow_html=True)

    row2 = "".join([
        _kpi_card("Sharpe (rf=0)", f"{m['Sharpe (rf=0)']:.2f}",
                  "", _hint_cls(m["Sharpe (rf=0)"])),
        _kpi_card("Beta", f"{m['Beta']:.2f}"),
        _kpi_card("Max Drawdown", pct(m["Max drawdown"]),
                  "", _hint_cls(m["Max drawdown"], invert=True)),
        _kpi_card("Bench Max DD", pct(m["Benchmark max DD"])),
        _kpi_card("Monthly Hit Rate", pct(m["Hit rate (monthly)"]),
                  "months outperforming", "neutral"),
    ])
    st.markdown(f'<div class="kpi-row">{row2}</div>', unsafe_allow_html=True)

    if not pd.isna(m["Information ratio"]) and (m["Information ratio"] > 1.5 or m["Tracking error"] < 0.005):
        st.info("IR above 1.5 or near-zero TE is unusual. Check for look-ahead bias or missing costs.")

    cum = (1 + r).cumprod()

    # Animated growth chart with frames
    n_pts = len(cum)
    n_frames = min(30, n_pts)
    step = max(1, n_pts // n_frames)
    frames = []
    for i in range(step, n_pts + 1, step):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=cum.index[:i], y=cum["portfolio"].values[:i],
                           mode="lines", line=dict(color=PORT_COLOR, width=2.5), name="Portfolio"),
                go.Scatter(x=cum.index[:i], y=cum["benchmark"].values[:i],
                           mode="lines", line=dict(color=BENCH_COLOR, width=2), name="Benchmark"),
            ],
            name=str(i),
        ))
    if n_pts % step != 0:
        frames.append(go.Frame(
            data=[
                go.Scatter(x=cum.index, y=cum["portfolio"], mode="lines",
                           line=dict(color=PORT_COLOR, width=2.5), name="Portfolio"),
                go.Scatter(x=cum.index, y=cum["benchmark"], mode="lines",
                           line=dict(color=BENCH_COLOR, width=2), name="Benchmark"),
            ],
            name=str(n_pts),
        ))

    fig = go.Figure(
        data=[
            go.Scatter(x=[cum.index[0]], y=[cum["portfolio"].values[0]],
                       mode="lines", line=dict(color=PORT_COLOR, width=2.5), name="Portfolio"),
            go.Scatter(x=[cum.index[0]], y=[cum["benchmark"].values[0]],
                       mode="lines", line=dict(color=BENCH_COLOR, width=2), name="Benchmark"),
        ],
        frames=frames,
    )
    _apply_plotly_theme(fig, 390)
    fig.update_layout(
        title="Growth of ₹1",
        yaxis_title="Cumulative value",
        margin=dict(l=48, r=16, t=64, b=40),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=1.0, y=-0.12, xanchor="right", yanchor="top",
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=40, redraw=True),
                                           fromcurrent=True, mode="immediate")])],
        )],
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    act_cum = (1 + (r["portfolio"] - r["benchmark"])).cumprod() - 1
    f2 = go.Figure(go.Scatter(x=act_cum.index, y=act_cum, fill="tozeroy",
                              fillcolor="rgba(13,148,136,.12)", line=dict(color=PORT_COLOR, width=2),
                              name="Cumulative active"))
    _apply_plotly_theme(f2, 320)
    f2.update_layout(title="Cumulative active return", yaxis_tickformat=".1%")
    col1.plotly_chart(f2, width="stretch")

    dd = pd.DataFrame({k: (cum[k] / cum[k].cummax() - 1) for k in cum})
    f3 = go.Figure()
    f3.add_scatter(x=dd.index, y=dd["portfolio"], name="Portfolio",
                   fill="tozeroy", fillcolor="rgba(13,148,136,.08)",
                   line=dict(color=PORT_COLOR, width=1.5))
    f3.add_scatter(x=dd.index, y=dd["benchmark"], name="Benchmark",
                   line=dict(color=BENCH_COLOR, width=1.5))
    _apply_plotly_theme(f3, 320)
    f3.update_layout(title="Drawdown", yaxis_tickformat=".0%")
    col2.plotly_chart(f3, width="stretch")

    mo = (1 + r).resample("ME").prod() - 1
    act_m = (mo["portfolio"] - mo["benchmark"]).to_frame("a")
    act_m["y"], act_m["m"] = act_m.index.year, act_m.index.month
    hm = act_m.pivot(index="y", columns="m", values="a")
    f4 = go.Figure(go.Heatmap(
        z=hm.values * 100,
        x=[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1] for m in hm.columns],
        y=hm.index.astype(str),
        colorscale=[[0, "#ef4444"], [0.5, "#fafafa"], [1, "#059669"]],
        zmid=0, colorbar_title="%",
    ))
    _apply_plotly_theme(f4, 320)
    f4.update_layout(title="Monthly active return (%)")
    st.plotly_chart(f4, width="stretch")

# ---- Risk
with tabs[1]:
    a = r["portfolio"] - r["benchmark"]
    rte = a.rolling(roll).std() * np.sqrt(TD)
    rir = a.rolling(roll).mean() * TD / rte
    rbeta = r["portfolio"].rolling(roll).cov(r["benchmark"]) / r["benchmark"].rolling(roll).var()

    f = go.Figure()
    f.add_scatter(x=rte.index, y=rte, name="Rolling TE", line=dict(color=PORT_COLOR, width=2))
    f.add_hline(y=te_lim, line_dash="dash", line_color=RED,
                annotation_text=f"TE limit ({te_lim:.1%})", annotation_font_color=RED)
    _apply_plotly_theme(f, 340)
    f.update_layout(title=f"Rolling tracking error ({roll}d)", yaxis_tickformat=".1%")
    st.plotly_chart(f, width="stretch")

    c1, c2 = st.columns(2)
    f = go.Figure(go.Scatter(x=rir.index, y=rir, line=dict(color=PORT_COLOR, width=1.5)))
    _apply_plotly_theme(f, 300)
    f.update_layout(title="Rolling information ratio")
    c1.plotly_chart(f, width="stretch")

    f = go.Figure(go.Scatter(x=rbeta.index, y=rbeta, line=dict(color=PORT_COLOR, width=1.5)))
    f.add_hline(y=1, line_dash="dot", line_color=BENCH_COLOR)
    _apply_plotly_theme(f, 300)
    f.update_layout(title="Rolling beta")
    c2.plotly_chart(f, width="stretch")

    f = go.Figure(go.Histogram(x=a * 1e4, nbinsx=60, marker_color=PORT_COLOR,
                               marker_line_color="white", marker_line_width=0.5))
    _apply_plotly_theme(f, 300)
    f.update_layout(title="Daily active return distribution (bps)", xaxis_title="bps")
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

    f = go.Figure(go.Bar(
        x=top["active"], y=top["ticker"], orientation="h",
        marker_color=[GREEN if v >= 0 else RED for v in top["active"]],
        marker_line_width=0,
    ))
    _apply_plotly_theme(f, max(350, 22 * len(top)))
    f.update_layout(title=f"Top {top_n} overweights / underweights", xaxis_tickformat=".2%")
    st.plotly_chart(f, width="stretch")

    sec = cur.groupby("sector")[["weight", "bench_weight"]].sum()
    sec["active"] = sec["weight"] - sec["bench_weight"]
    c1, c2 = st.columns(2)
    f = go.Figure()
    f.add_bar(x=sec.index, y=sec["weight"], name="Portfolio", marker_color=PORT_COLOR)
    f.add_bar(x=sec.index, y=sec["bench_weight"], name="Benchmark", marker_color=BENCH_COLOR)
    _apply_plotly_theme(f, 330)
    f.update_layout(title="Sector weights", barmode="group", yaxis_tickformat=".0%")
    c1.plotly_chart(f, width="stretch")

    f = go.Figure(go.Bar(
        x=sec.index, y=sec["active"],
        marker_color=[GREEN if v >= 0 else RED for v in sec["active"]],
    ))
    _apply_plotly_theme(f, 330)
    f.update_layout(title="Sector active weights", yaxis_tickformat=".2%")
    c2.plotly_chart(f, width="stretch")

    st.dataframe(
        cur.sort_values("weight", ascending=False)
        .style.format({"weight": "{:.2%}", "bench_weight": "{:.2%}", "active": "{:+.2%}"}),
        width="stretch", hide_index=True,
    )

# ---- Turnover
with tabs[3]:
    to = turnover_series(w_all)
    if to.empty:
        st.info("Need at least two rebalance dates to compute turnover.")
    else:
        f = go.Figure(go.Bar(x=to.index, y=to, marker_color=PORT_COLOR,
                             marker_line_width=0))
        f.add_hline(y=to_lim, line_dash="dash", line_color=RED,
                    annotation_text=f"Limit ({to_lim:.0%})", annotation_font_color=RED)
        _apply_plotly_theme(f, 350)
        f.update_layout(title="One-way turnover per rebalance", yaxis_tickformat=".1%")
        st.plotly_chart(f, width="stretch")

        to_cards = "".join([
            _kpi_card("Avg Turnover", pct(to.mean())),
            _kpi_card("Max Turnover", pct(to.max())),
            _kpi_card("Annualized Turnover",
                      pct(to.sum() / max((to.index[-1] - to.index[0]).days / 365.25, 1e-9))),
        ])
        st.markdown(f'<div class="kpi-row">{to_cards}</div>', unsafe_allow_html=True)

# ---- Compliance
with tabs[4]:
    rows = []
    for d, g in w_all.groupby("date"):
        act = g["weight"] - g["bench_weight"]
        sact = g.groupby("sector")[["weight", "bench_weight"]].sum().diff(axis=1).iloc[:, 1]
        rows.append({"date": d, "sum_w": g["weight"].sum(), "min_w": g["weight"].min(),
                     "max_stock_active": act.abs().max(), "max_sector_active": sact.abs().max()})
    comp = pd.DataFrame(rows).set_index("date")
    to_comp = turnover_series(w_all)
    comp["turnover"] = to_comp.reindex(comp.index)
    tol = 1e-6
    checks = pd.DataFrame({
        "Fully invested (sum = 1)": (comp["sum_w"] - 1).abs() <= 1e-4,
        "Long-only": comp["min_w"] >= -tol,
        "Stock active limit": comp["max_stock_active"] <= stock_lim + tol,
        "Sector active limit": comp["max_sector_active"] <= sector_lim + tol,
        "Turnover limit": comp["turnover"].fillna(0) <= to_lim + tol,
    })
    comp_summary = pd.DataFrame({
        "Pass rate": checks.mean().map("{:.0%}".format),
        "Breaches": (~checks).sum().astype(str),
    })
    comp_summary.loc["Realized TE vs limit"] = [
        "OK" if m["Tracking error"] <= te_lim else "BREACH", f"{m['Tracking error']:.2%}"]
    st.dataframe(comp_summary, width="stretch")
    bad = comp[~checks.all(axis=1)]
    if bad.empty:
        st.success("No constraint breaches on any rebalance date.")
    else:
        st.error(f"{len(bad)} rebalance date(s) with breaches")
        st.dataframe(bad.style.format("{:.4f}"), width="stretch")

# ---- Latest Trades
with tabs[5]:
    if latest_trades is None or latest_trades.empty:
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;color:var(--text-secondary);">'
            '<div style="font-size:2rem;margin-bottom:8px;">\U0001f4ed</div>'
            '<div style="font-size:1rem;font-weight:600;">No trade data available</div>'
            '<div style="font-size:.85rem;margin-top:4px;">Run the pipeline to generate '
            '<code>results/latest_trades.csv</code></div></div>',
            unsafe_allow_html=True,
        )
    else:
        buys = latest_trades[latest_trades["trade"] > 1e-6].sort_values("trade", ascending=False)
        sells = latest_trades[latest_trades["trade"] < -1e-6].sort_values("trade")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Buys**")
            if buys.empty:
                st.info("No buy trades.")
            else:
                buy_html = ""
                for _, row in buys.iterrows():
                    buy_html += (
                        f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                        f'<span class="trade-badge trade-buy">BUY</span>'
                        f'<strong>{row["ticker"]}</strong>'
                        f'<span style="color:var(--text-secondary);font-size:.85rem;">'
                        f'{row["trade"]:+.2%}</span></div>'
                    )
                st.markdown(buy_html, unsafe_allow_html=True)

        with c2:
            st.markdown("**Sells**")
            if sells.empty:
                st.info("No sell trades.")
            else:
                sell_html = ""
                for _, row in sells.iterrows():
                    sell_html += (
                        f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                        f'<span class="trade-badge trade-sell">SELL</span>'
                        f'<strong>{row["ticker"]}</strong>'
                        f'<span style="color:var(--text-secondary);font-size:.85rem;">'
                        f'{row["trade"]:+.2%}</span></div>'
                    )
                st.markdown(sell_html, unsafe_allow_html=True)

        trades_sorted = latest_trades.copy()
        trades_sorted = trades_sorted[trades_sorted["trade"].abs() > 1e-6].sort_values("trade")
        if not trades_sorted.empty:
            f = go.Figure(go.Bar(
                x=trades_sorted["trade"], y=trades_sorted["ticker"], orientation="h",
                marker_color=[GREEN if v > 0 else RED for v in trades_sorted["trade"]],
            ))
            _apply_plotly_theme(f, max(300, 20 * len(trades_sorted)))
            f.update_layout(title="Trade sizes", xaxis_tickformat=".2%")
            st.plotly_chart(f, width="stretch")

        st.download_button(
            "⬇ Download trades (CSV)",
            latest_trades.to_csv(index=False).encode(), "latest_trades.csv",
            width="stretch",
        )

# ---- Data
with tabs[6]:
    st.download_button("⬇ Download returns (CSV)", r.to_csv().encode(), "returns.csv",
                       width="stretch")
    st.download_button("⬇ Download weights (CSV)", w_all.to_csv(index=False).encode(), "weights.csv",
                       width="stretch")
    st.markdown('<div class="section-header">Performance summary</div>', unsafe_allow_html=True)
    st.dataframe(pd.Series(m, name="value").to_frame(), width="stretch")

# ---- My Portfolio
with tabs[7]:
    st.warning(
        "**Disclaimer:** This is an educational analysis tool, not investment advice. "
        "Past performance does not predict future results. "
        "Consult a qualified financial advisor before making investment decisions."
    )

    _port_latest_dt = weights_all["date"].max()
    _port_latest_w = weights_all[weights_all["date"] == _port_latest_dt].copy()
    _nifty_tickers = set(_port_latest_w["ticker"].unique())

    _sample_csv = (
        "ticker,value_inr\n"
        "RELIANCE,100000\n"
        "TCS,80000\n"
        "HDFCBANK,75000\n"
        "INFY,60000\n"
        "ICICIBANK,50000\n"
    )
    st.download_button(
        "⬇ Download sample CSV template", _sample_csv.encode(),
        "portfolio_template.csv", mime="text/csv",
    )

    _uploaded = st.file_uploader(
        "Upload your portfolio CSV", type=["csv"],
        help="CSV with columns: **ticker** and **value_inr** (market value in ₹)",
        key="portfolio_csv_upload",
    )

    if _uploaded is not None:
        _parse_ok = True
        try:
            _udf = pd.read_csv(_uploaded)
        except Exception:
            st.error("Could not read the file. Please upload a valid CSV.")
            _parse_ok = False
            _udf = pd.DataFrame()

        if _parse_ok:
            _col_map = {c.strip().lower(): c for c in _udf.columns}
            if "ticker" not in _col_map or "value_inr" not in _col_map:
                st.error("CSV must contain columns **ticker** and **value_inr**.")
            else:
                _udf = _udf.rename(columns={
                    _col_map["ticker"]: "ticker",
                    _col_map["value_inr"]: "value_inr",
                })
                _udf["value_inr"] = pd.to_numeric(_udf["value_inr"], errors="coerce")
                _bad = _udf["value_inr"].isna() | (_udf["value_inr"] <= 0)
                if _bad.any():
                    st.warning(f"Dropped {_bad.sum()} row(s) with missing or non-positive values.")
                    _udf = _udf[~_bad]

                if _udf.empty:
                    st.error("No valid rows remain after validation.")
                else:
                    _udf["ticker"] = _udf["ticker"].astype(str).str.strip().str.upper()
                    _udf.loc[~_udf["ticker"].str.endswith(".NS"), "ticker"] += ".NS"

                    _matched_mask = _udf["ticker"].isin(_nifty_tickers)
                    _unmatched = _udf[~_matched_mask]
                    _udf = _udf[_matched_mask].copy()

                    if not _unmatched.empty:
                        st.warning(
                            f"**{len(_unmatched)} ticker(s) not in Nifty 50 universe** (excluded): "
                            + ", ".join(_unmatched["ticker"].tolist())
                        )

                    if _udf.empty:
                        st.error("No tickers matched the Nifty 50 universe.")
                    else:
                        _udf = _udf.groupby("ticker", as_index=False)["value_inr"].sum()
                        _total_val = _udf["value_inr"].sum()
                        _udf["my_weight"] = _udf["value_inr"] / _total_val

                        _cmp = _port_latest_w[["ticker", "sector", "bench_weight"]].copy()
                        _cmp = _cmp.merge(
                            _port_latest_w[["ticker", "weight"]].rename(
                                columns={"weight": "model_weight"}
                            ),
                            on="ticker",
                        )
                        _cmp = _cmp.merge(
                            _udf[["ticker", "my_weight"]], on="ticker", how="left",
                        )
                        _cmp["my_weight"] = _cmp["my_weight"].fillna(0.0)
                        _cmp["active_vs_bench"] = _cmp["my_weight"] - _cmp["bench_weight"]
                        _cmp["diff_vs_model"] = _cmp["my_weight"] - _cmp["model_weight"]

                        st.markdown(
                            '<div class="section-header">Weight Comparison</div>',
                            unsafe_allow_html=True,
                        )
                        _disp = _cmp[[
                            "ticker", "sector", "my_weight", "bench_weight",
                            "model_weight", "active_vs_bench", "diff_vs_model",
                        ]].copy()
                        _disp.columns = [
                            "Ticker", "Sector", "My Weight", "Bench Weight",
                            "Model Weight", "Active vs Bench", "Diff vs Model",
                        ]
                        _disp = _disp.sort_values("My Weight", ascending=False)
                        st.dataframe(
                            _disp.style.format({
                                "My Weight": "{:.2%}",
                                "Bench Weight": "{:.2%}",
                                "Model Weight": "{:.2%}",
                                "Active vs Bench": "{:+.2%}",
                                "Diff vs Model": "{:+.2%}",
                            }),
                            width="stretch",
                            hide_index=True,
                        )

                        _c1, _c2 = st.columns(2)

                        _top_act = pd.concat([
                            _cmp.nlargest(top_n, "active_vs_bench"),
                            _cmp.nsmallest(top_n, "active_vs_bench"),
                        ]).drop_duplicates("ticker").sort_values("active_vs_bench")

                        _fig = go.Figure(go.Bar(
                            x=_top_act["active_vs_bench"],
                            y=_top_act["ticker"],
                            orientation="h",
                            marker_color=[
                                GREEN if v >= 0 else RED
                                for v in _top_act["active_vs_bench"]
                            ],
                            marker_line_width=0,
                        ))
                        _apply_plotly_theme(
                            _fig, max(350, 22 * len(_top_act)),
                        )
                        _fig.update_layout(
                            title=f"Top {top_n} over/underweights vs benchmark",
                            xaxis_tickformat=".2%",
                        )
                        _c1.plotly_chart(_fig, width="stretch")

                        _sec = _cmp.groupby("sector")[
                            ["my_weight", "bench_weight"]
                        ].sum()
                        _fig2 = go.Figure()
                        _fig2.add_bar(
                            x=_sec.index,
                            y=_sec["my_weight"],
                            name="My Portfolio",
                            marker_color=PORT_COLOR,
                        )
                        _fig2.add_bar(
                            x=_sec.index,
                            y=_sec["bench_weight"],
                            name="Benchmark",
                            marker_color=BENCH_COLOR,
                        )
                        _apply_plotly_theme(_fig2, 350)
                        _fig2.update_layout(
                            title="Sector exposure vs benchmark",
                            barmode="group",
                            yaxis_tickformat=".0%",
                        )
                        _c2.plotly_chart(_fig2, width="stretch")

                        st.markdown(
                            '<div class="section-header">Suggested Rebalance</div>',
                            unsafe_allow_html=True,
                        )
                        _reb = _cmp[["ticker", "my_weight", "model_weight"]].copy()
                        _reb["raw_trade"] = _reb["model_weight"] - _reb["my_weight"]
                        _raw_to = _reb["raw_trade"].abs().sum() / 2

                        if _raw_to > to_lim and _raw_to > 0:
                            _scale = to_lim / _raw_to
                            _reb["trade_pct"] = _reb["raw_trade"] * _scale
                            st.info(
                                f"Trades scaled to respect {to_lim:.0%} turnover limit "
                                f"(raw one-way turnover: {_raw_to:.1%})."
                            )
                        else:
                            _reb["trade_pct"] = _reb["raw_trade"]

                        _reb["trade_inr"] = _reb["trade_pct"] * _total_val
                        _trades = _reb[
                            _reb["trade_pct"].abs() > 1e-6
                        ].sort_values("trade_pct", ascending=False)

                        if _trades.empty:
                            st.success(
                                "Your portfolio already matches the model portfolio!"
                            )
                        else:
                            _trades_disp = _trades[
                                ["ticker", "trade_pct", "trade_inr"]
                            ].copy()
                            _trades_disp.columns = [
                                "Ticker", "Trade (%)", "Trade (₹)",
                            ]
                            st.dataframe(
                                _trades_disp.style.format({
                                    "Trade (%)": "{:+.2%}",
                                    "Trade (₹)": "₹{:+,.0f}",
                                }),
                                width="stretch",
                                hide_index=True,
                            )

                        st.info(
                            "\U0001f512 **Privacy:** Your file is not stored. "
                            "All processing happens in memory only."
                        )

# ---- About
with tabs[8]:
    st.markdown("""
### Methodology

This dashboard presents results from an **enhanced indexing** strategy that
aims to slightly outperform a benchmark while keeping tracking error low.

1. **Alpha signals** — 12‑1 momentum, 60-day low volatility, and
   1-month reversal are combined with configurable weights and an assumed
   information coefficient (IC).
2. **Risk model** — A Ledoit-Wolf shrinkage covariance matrix is estimated
   from rolling daily returns and annualized.
3. **Optimization** — A convex optimizer (CLARABEL via cvxpy) maximizes
   expected active return minus risk and transaction costs, subject to
   per-stock, per-sector, tracking-error, and turnover constraints.
4. **Backtest** — Walk-forward monthly rebalancing with drift between
   rebalance dates and transaction-cost deduction.

### Limitations

| Limitation | Detail |
|:--|:--|
| **Survivorship bias** | The universe uses current Nifty 50 constituents, not historical membership. Stocks that were removed (often after poor performance) are excluded, which likely overstates returns. |
| **Proxy benchmark** | Benchmark weights are approximated from market-cap data at each rebalance, not from the official Nifty 50 free-float methodology. |
| **Price-only signals** | Signals use adjusted close prices only — no fundamental, sentiment, or alternative data. |
| **Backtest ≠ live** | Past simulated performance does not predict future results. Slippage, market impact, and liquidity constraints are not modeled. |
| **Not financial advice** | This tool is for educational and research purposes only. It does not constitute investment advice. |
""")
