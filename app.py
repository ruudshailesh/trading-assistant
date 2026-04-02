"""
app.py — Trading Assistant UI (Redesigned)
Modern dark financial terminal aesthetic.
5 pages: Signals | Watchlist | History | Performance | Insights
"""

import io
import traceback
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import database as db
import universe as uni
import data_fetch as df_mod
import scoring as sc
import risk as rk
import learning as lrn
import backtest as bt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeSignal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: #080b12 !important;
    color: #c9d1d9;
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] .stMarkdown p { font-size: 0.82rem; color: #8b949e; }

/* ── Typography ── */
h1 { font-size: 1.6rem !important; font-weight: 700 !important; color: #f0f6fc !important; letter-spacing: -0.5px; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #f0f6fc !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #58a6ff !important; margin: 0 !important; }
p, li { color: #8b949e; font-size: 0.88rem; }

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover { border-color: #388bfd; }
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #f0f6fc !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    box-shadow: 0 0 20px rgba(56,139,253,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 30px rgba(56,139,253,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #30363d !important;
    border-color: #8b949e !important;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,0.1) !important;
}

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 1.5rem 0 !important; }

/* ── Tables / Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: #161b22 !important;
    color: #8b949e !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
[data-testid="stDataFrame"] td {
    color: #c9d1d9 !important;
    font-size: 0.83rem !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #8b949e !important; font-size: 0.85rem !important; }

/* ── Custom components ── */
.trade-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #3fb950;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.trade-card:hover { box-shadow: 0 4px 24px rgba(0,0,0,0.4); border-color: #388bfd; }
.trade-card-b { border-left-color: #f78166; }

.ticker-badge {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    color: #f0f6fc;
    letter-spacing: 1px;
}
.score-badge {
    display: inline-block;
    background: rgba(56,139,253,0.12);
    border: 1px solid rgba(56,139,253,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 700;
    color: #388bfd;
}
.strategy-badge-a {
    display: inline-block;
    background: rgba(63,185,80,0.12);
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #3fb950;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.strategy-badge-b {
    display: inline-block;
    background: rgba(247,129,102,0.12);
    border: 1px solid rgba(247,129,102,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #f78166;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.price-block {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    text-align: center;
}
.price-label {
    font-size: 0.68rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 4px;
}
.price-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f0f6fc;
}
.price-pct { font-size: 0.75rem; margin-top: 2px; }
.pct-up { color: #3fb950; }
.pct-down { color: #f85149; }

.stat-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.stat-chip {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.76rem;
    color: #8b949e;
}
.stat-chip span { color: #c9d1d9; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8b949e;
}

.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.s-pending  { background:#1f2d4a; color:#58a6ff; }
.s-executed { background:#1a3a2a; color:#3fb950; }
.s-success  { background:#1a3a1a; color:#56d364; }
.s-failed   { background:#3a1a1a; color:#f85149; }
.s-expired  { background:#1e1e1e; color:#6e7681; }

.market-banner {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.mb-open  { background:#0a2419; border:1px solid #1a4731; color:#3fb950; }
.mb-closed { background:#1a1208; border:1px solid #3a2e0a; color:#d29922; }
.mb-error  { background:#1a0a0a; border:1px solid #3a1515; color:#f85149; }

.expl-box {
    background: #0a0e14;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    white-space: pre-wrap;
    line-height: 1.7;
}
.expl-box strong { color: #c9d1d9; }

.disc {
    margin-top: 2rem;
    padding: 0.8rem 1.2rem;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    font-size: 0.72rem;
    color: #6e7681;
    line-height: 1.6;
}

/* ── Spy banner ── */
.spy-neutral { background:#0a1929; border:1px solid #1c3a5e; color:#58a6ff; border-radius:8px; padding:0.5rem 1rem; font-size:0.82rem; }
.spy-bearish { background:#1a1208; border:1px solid #3a2e0a; color:#d29922; border-radius:8px; padding:0.5rem 1rem; font-size:0.82rem; }
.spy-highvol { background:#1a0a0a; border:1px solid #3a1515; color:#f85149; border-radius:8px; padding:0.5rem 1rem; font-size:0.82rem; }

/* ── Watchlist card ── */
.wl-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s;
}
.wl-card:hover { border-color: #30363d; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid #21262d;
}
.logo-text {
    font-size: 1.2rem;
    font-weight: 800;
    color: #f0f6fc;
    letter-spacing: -0.5px;
}
.logo-dot { color: #388bfd; }

/* ── Sidebar nav ── */
[data-testid="stRadio"] label {
    font-size: 0.85rem !important;
    color: #8b949e !important;
    padding: 0.3rem 0 !important;
}
[data-testid="stRadio"] label:hover { color: #c9d1d9 !important; }

/* ── Progress/spinner ── */
.stSpinner > div { border-top-color: #388bfd !important; }

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; font-size: 0.85rem !important; }

/* ── Checkbox ── */
.stCheckbox label { font-size: 0.85rem !important; color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in {
    "ohlcv": None, "spy_ctx": None, "data_timestamp": None,
    "last_fetch_date": None, "daily_trades": [], "pipeline_log": {},
    "backtest_result": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Startup ───────────────────────────────────────────────────────────────────
@st.cache_resource
def _startup():
    db.init_db()
    db.expire_old_pending_trades()

_startup()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _price(v):
    return f"${v:,.2f}" if v is not None else "—"

def _pct(v, decimals=2):
    if v is None: return "—"
    sign = "+" if v >= 0 else ""
    cls  = "pct-up" if v >= 0 else "pct-down"
    return f'<span class="{cls}">{sign}{v*100:.{decimals}f}%</span>'

def _pill(status):
    s = status.lower()
    return f'<span class="status-pill s-{s}">{status}</span>'

def _load_data(force=False):
    today = date.today().isoformat()
    if not force and st.session_state["ohlcv"] and st.session_state["last_fetch_date"] == today:
        return False
    tickers = uni.get_tickers_by_strategy()
    if not tickers:
        return False
    with st.spinner("Fetching market data..."):
        ohlcv, spy_ctx, ts = df_mod.load_all_data(tickers)
        st.session_state.update({
            "ohlcv": ohlcv, "spy_ctx": spy_ctx,
            "data_timestamp": ts, "last_fetch_date": today,
        })
        gaps = df_mod.check_overnight_gaps(db.get_active_trades())
        for g in gaps:
            db.update_trade_status(g["trade_id"], "Failed", exit_price=g["exit_price"], gap_exit=True)
            lrn.run_all_learning_updates()
        if gaps:
            st.toast(f"⚡ {len(gaps)} overnight gap exit(s) processed", icon="🌙")
    return True


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown('<div class="logo-text">Trade<span class="logo-dot">Signal</span></div>', unsafe_allow_html=True)
        st.markdown(f'<p style="margin-top:-4px;font-size:0.72rem;color:#6e7681">Capital: ${config.CAPITAL:,.0f} · Risk: {config.RISK_PER_TRADE*100:.0f}%/trade</p>', unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio("", [
            "⚡  Signals",
            "👁  Watchlist",
            "📋  History",
            "📊  Performance",
            "🧠  Insights",
        ], label_visibility="collapsed")

        st.markdown("---")

        # Universe
        u = uni.get_universe_status()
        if u["exists"] and not u["stale"]:
            st.markdown(f'<p style="color:#3fb950;font-size:0.78rem">✓ Universe · {u["total_count"]} tickers · {u["age_days"]}d old</p>', unsafe_allow_html=True)
        elif u["exists"]:
            st.markdown(f'<p style="color:#d29922;font-size:0.78rem">⚠ Universe stale ({u["age_days"]}d)</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#f85149;font-size:0.78rem">✗ No universe</p>', unsafe_allow_html=True)

        if st.button("↺ Refresh Universe", use_container_width=True):
            ph = st.empty()
            def prog(m): ph.markdown(f'<p style="font-size:0.78rem;color:#8b949e">{m}</p>', unsafe_allow_html=True)
            with st.spinner("Refreshing..."):
                ok, msg = uni.refresh_universe(progress_callback=prog)
            ph.empty()
            if ok:
                st.success(msg)
                _load_data(force=True)
            else:
                st.error(msg)

        st.markdown("---")

        # DB health
        h = db.db_health_check()
        if h["ok"]:
            st.markdown(f'<p style="color:#3fb950;font-size:0.78rem">✓ Supabase · {h["trade_count"]} trades</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:#f85149;font-size:0.78rem">✗ DB error</p>', unsafe_allow_html=True)

        # Market status
        mkt = df_mod.get_market_status()
        color = "#3fb950" if mkt["is_open"] else "#d29922"
        st.markdown(f'<p style="color:{color};font-size:0.78rem">{mkt["message"]}</p>', unsafe_allow_html=True)

    return page


# ── Header ────────────────────────────────────────────────────────────────────
def _header(title, subtitle=None):
    ts  = st.session_state.get("data_timestamp", "")
    mkt = df_mod.get_market_status()
    spy = st.session_state.get("spy_ctx") or db.get_spy_context()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"# {title}")
        if subtitle:
            st.markdown(f'<p style="color:#8b949e;margin-top:-0.8rem">{subtitle}</p>', unsafe_allow_html=True)
    with c2:
        flags = []
        if spy.get("bearish_flag"):  flags.append("📉 SPY Bearish")
        if spy.get("high_vol_flag"): flags.append("⚠ High Vol")
        flag_str = " · ".join(flags) if flags else "✓ Market Neutral"
        color = "#f85149" if spy.get("high_vol_flag") else "#d29922" if spy.get("bearish_flag") else "#3fb950"
        st.markdown(f'''
        <div style="text-align:right;padding-top:0.5rem">
            <div style="font-size:0.75rem;color:#6e7681">{ts}</div>
            <div style="font-size:0.82rem;font-weight:600;color:{color};margin-top:2px">{flag_str}</div>
        </div>''', unsafe_allow_html=True)
    st.markdown('<hr style="margin:0.8rem 0 1.5rem 0">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
def page_signals():
    _header("⚡ Trade Signals", "AI-free · Deterministic · Explainable")

    # Portfolio stats row
    s = rk.get_portfolio_risk_summary()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active",        f"{s['active_count']}/{config.MAX_ACTIVE_TRADES}")
    c2.metric("Slots Free",    s["slots_free"])
    c3.metric("Capital",       f"${config.CAPITAL:,.0f}")
    c4.metric("At Risk",       f"${s['total_risk_$']:,.0f}")
    c5.metric("Risk Used",     f"{s['pct_used']}%")

    st.markdown("")

    # Action bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        go_btn = st.button("⚡ Generate Signals", type="primary",
                           disabled=s["at_capacity"],
                           use_container_width=True)
    with col2:
        if st.button("↺ Refresh Data", use_container_width=True):
            _load_data(force=True)
            st.rerun()
    with col3:
        if st.button("▶ Run Backtest", use_container_width=True):
            if st.session_state.get("ohlcv"):
                with st.spinner("Running backtest..."):
                    r = bt.run_backtest(st.session_state["ohlcv"])
                    st.session_state["backtest_result"] = r
                st.toast("Backtest complete", icon="📊")
            else:
                st.warning("Load data first")

    if s["at_capacity"]:
        st.warning(f"Portfolio at capacity — {s['active_count']} active trades. Close existing trades first.")
        _disclaimer()
        return

    # Load data if needed
    if not st.session_state.get("ohlcv"):
        if not uni.get_universe_status()["exists"]:
            st.info("👈 Click **Refresh Universe** in the sidebar first (~30 seconds)")
            _disclaimer()
            return
        _load_data()

    if go_btn:
        ohlcv   = st.session_state.get("ohlcv", {})
        spy_ctx = st.session_state.get("spy_ctx", {"bearish_flag": 0, "high_vol_flag": 0})
        if not ohlcv:
            st.error("No market data. Click Refresh Data.")
            _disclaimer()
            return

        with st.spinner("Running signal pipeline..."):
            try:
                candidates, log = sc.run_selection_pipeline(ohlcv, spy_ctx)
                approved, rejections = rk.attach_risk_levels(candidates, ohlcv)
                st.session_state["daily_trades"] = approved
                st.session_state["pipeline_log"] = log

                today = date.today().isoformat()
                for t in approved:
                    db.insert_trade({
                        "date": today, "ticker": t["ticker"],
                        "strategy": t["strategy"], "score": t["score"],
                        "entry": t["entry"], "stop": t["stop"], "target": t["target"],
                        "position_size": t["position_size"],
                        "momentum": t["momentum"], "volume_spike": t["volume_spike"],
                        "volatility": t["volatility"], "atr": t["atr"],
                        "sector": t.get("sector", "Unknown"),
                        "explanation": t.get("explanation", ""),
                    })
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                _disclaimer()
                return

    trades = st.session_state.get("daily_trades", [])
    log    = st.session_state.get("pipeline_log", {})

    if trades:
        # SPY context banner
        spy = st.session_state.get("spy_ctx", {})
        _spy_banner(spy)
        st.markdown("")

        # Pipeline stats inline
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Universe", log.get("universe_size", "—"))
        col_b.metric("Scored",   log.get("features_computed", "—"))
        col_c.metric("Filtered", log.get("after_spy", "—"))
        col_d.metric("Selected", log.get("selected", "—"))
        st.markdown("")

        for i, trade in enumerate(trades):
            _trade_card(trade, i + 1)

    elif log:
        st.info("No trades generated today. See pipeline stats above.")
        if log.get("skipped_reasons"):
            with st.expander("Why were tickers skipped?"):
                for r in log["skipped_reasons"][:10]:
                    st.markdown(f'<p style="font-size:0.82rem;color:#8b949e">• {r}</p>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align:center;padding:3rem;color:#6e7681">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">⚡</div>
            <div style="font-size:1rem;font-weight:600;color:#8b949e">Ready to generate signals</div>
            <div style="font-size:0.82rem;margin-top:0.3rem">Click Generate Signals to run the pipeline</div>
        </div>''', unsafe_allow_html=True)

    _disclaimer()


def _spy_banner(spy):
    bearish  = spy.get("bearish_flag", 0)
    high_vol = spy.get("high_vol_flag", 0)
    price    = spy.get("spy_price", "—")
    dma      = spy.get("spy_20dma")
    dma_str  = f" vs 20DMA ${dma:.2f}" if dma else ""

    if high_vol:
        cls, icon, msg = "spy-highvol", "🔴", f"HIGH VOLATILITY REGIME — Filter B disabled · SPY ${price}{dma_str}"
    elif bearish:
        cls, icon, msg = "spy-bearish", "🟡", f"SPY BEARISH — All scores reduced 20% · SPY ${price}{dma_str}"
    else:
        cls, icon, msg = "spy-neutral", "🟢", f"SPY NEUTRAL · SPY ${price}{dma_str}"

    st.markdown(f'<div class="{cls}">{icon} {msg}</div>', unsafe_allow_html=True)


def _trade_card(trade, rank):
    is_b     = trade.get("strategy") == "filter_b"
    card_cls = "trade-card trade-card-b" if is_b else "trade-card"
    strat_badge = f'<span class="strategy-badge-b">RISKY</span>' if is_b else f'<span class="strategy-badge-a">STABLE</span>'

    entry  = trade.get("entry", 0)
    stop   = trade.get("stop",  0)
    target = trade.get("target", 0)
    size   = trade.get("position_size", 0)
    risk   = trade.get("dollar_risk", 0)
    rr     = trade.get("rr_ratio", 0)
    score  = trade.get("score", 0)
    sector = trade.get("sector", "—")

    stop_pct   = (entry - stop)   / entry * 100 if entry else 0
    target_pct = (target - entry) / entry * 100 if entry else 0

    st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)

    # Header row
    st.markdown(f'''
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
        <div style="display:flex;align-items:center;gap:0.8rem">
            <span style="color:#6e7681;font-size:0.78rem;font-weight:700">#{rank}</span>
            <span class="ticker-badge">{trade["ticker"]}</span>
            {strat_badge}
            <span class="score-badge">{score:.1f} / 100</span>
        </div>
        <div style="font-size:0.78rem;color:#6e7681">{sector}</div>
    </div>''', unsafe_allow_html=True)

    # Price blocks
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    with pc1:
        st.markdown(f'''<div class="price-block">
            <div class="price-label">Entry</div>
            <div class="price-value">{_price(entry)}</div>
        </div>''', unsafe_allow_html=True)
    with pc2:
        st.markdown(f'''<div class="price-block">
            <div class="price-label">Stop Loss</div>
            <div class="price-value">{_price(stop)}</div>
            <div class="price-pct pct-down">-{stop_pct:.1f}%</div>
        </div>''', unsafe_allow_html=True)
    with pc3:
        st.markdown(f'''<div class="price-block">
            <div class="price-label">Target</div>
            <div class="price-value">{_price(target)}</div>
            <div class="price-pct pct-up">+{target_pct:.1f}%</div>
        </div>''', unsafe_allow_html=True)
    with pc4:
        st.markdown(f'''<div class="price-block">
            <div class="price-label">R:R Ratio</div>
            <div class="price-value">{rr:.1f}:1</div>
        </div>''', unsafe_allow_html=True)
    with pc5:
        st.markdown(f'''<div class="price-block">
            <div class="price-label">{size} shares · ${risk:,.0f} at risk</div>
            <div class="price-value" style="font-size:0.85rem">
                mom {trade.get("momentum",0)*100:+.1f}% · vol {trade.get("volume_spike",0):.1f}×
            </div>
        </div>''', unsafe_allow_html=True)

    # Explanation expander
    with st.expander("Signal breakdown"):
        st.markdown(f'<div class="expl-box">{trade.get("explanation","")}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: WATCHLIST
# ─────────────────────────────────────────────────────────────────────────────
def page_watchlist():
    _header("👁 Watchlist", "Score any ticker instantly")

    # Search bar
    st.markdown('<div class="section-header"><span class="section-title">Score Any Ticker</span></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        ticker_input = st.text_input("", placeholder="Enter ticker e.g. AAPL", label_visibility="collapsed").upper().strip()
    with col2:
        strat = st.selectbox("", ["filter_a", "filter_b"], label_visibility="collapsed")
    with col3:
        search = st.button("Score →", type="primary", use_container_width=True)

    if search and ticker_input:
        with st.spinner(f"Fetching {ticker_input}..."):
            df = df_mod.fetch_single_ticker(ticker_input)
        if df is None:
            st.error(f"Could not fetch **{ticker_input}** — check the symbol")
        else:
            spy = st.session_state.get("spy_ctx") or db.get_spy_context()
            res = sc.score_single_ticker(ticker_input, df, spy, strategy=strat)
            if res is None:
                st.warning(f"Insufficient data for **{ticker_input}**")
            else:
                stop, target, dollar_risk, rr = rk.calculate_trade_risk(res["entry"], res["atr"])
                size = rk.calculate_position_size(res["entry"], stop, dollar_risk)

                r1, r2, r3, r4, r5, r6 = st.columns(6)
                r1.metric("Score",    f"{res['score']:.1f}/100")
                r2.metric("Price",    _price(res["entry"]))
                r3.metric("Stop",     _price(stop))
                r4.metric("Target",   _price(target))
                r5.metric("R:R",      f"{rr:.1f}:1")
                r6.metric("Shares",   str(size))

                with st.expander("Signal breakdown"):
                    st.markdown(f'<div class="expl-box">{res.get("explanation","")}</div>', unsafe_allow_html=True)

                ca, cb = st.columns([1, 4])
                with ca:
                    if st.button("➕ Add to Watchlist"):
                        db.add_to_watchlist(ticker_input)
                        st.success(f"{ticker_input} added")
                        st.rerun()

    st.markdown("")
    st.markdown('<div class="section-header"><span class="section-title">My Watchlist</span></div>', unsafe_allow_html=True)

    watchlist = db.get_watchlist(active_only=True)
    if not watchlist:
        st.markdown('<p style="color:#6e7681;text-align:center;padding:2rem">Your watchlist is empty. Search for a ticker above.</p>', unsafe_allow_html=True)
    else:
        spy  = st.session_state.get("spy_ctx") or db.get_spy_context()
        ohlcv = st.session_state.get("ohlcv") or {}
        rows  = []
        for w in watchlist:
            t  = w["ticker"]
            df = ohlcv.get(t)
            if df is not None:
                s = sc.score_single_ticker(t, df, spy)
                if s:
                    stop, target, _, rr = rk.calculate_trade_risk(s["entry"], s["atr"])
                    rows.append({
                        "Ticker":    t,
                        "Price":     _price(s["entry"]),
                        "Score":     f"{s['score']:.1f}",
                        "Momentum":  f"{s['momentum']*100:+.1f}%",
                        "Vol Spike": f"{s['volume_spike']:.1f}×",
                        "Stop":      _price(stop),
                        "Target":    _price(target),
                        "R:R":       f"{rr:.1f}:1",
                        "Added":     w["date_added"],
                    })
                    continue
            rows.append({"Ticker": t, "Price": "—", "Score": "—",
                         "Momentum": "—", "Vol Spike": "—",
                         "Stop": "—", "Target": "—", "R:R": "—",
                         "Added": w["date_added"]})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        rem = st.selectbox("Remove ticker", ["—"] + [w["ticker"] for w in watchlist])
        if st.button("Remove") and rem != "—":
            db.remove_from_watchlist(rem)
            st.rerun()

    _disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def page_history():
    _header("📋 Trade History", "Track and update your trades")

    c1, c2, c3 = st.columns(3)
    with c1: sf = st.selectbox("Status", ["All"] + list(config.TRADE_STATUSES))
    with c2: stf = st.selectbox("Strategy", ["All", "filter_a", "filter_b"])
    with c3: lim = st.number_input("Limit", 10, 500, 100)

    trades = db.get_all_trades(
        status_filter=None if sf == "All" else sf,
        strategy_filter=None if stf == "All" else stf,
        limit=int(lim),
    )

    if not trades:
        st.info("No trades yet. Generate signals on the Signals page.")
        _disclaimer()
        return

    total  = len(trades)
    active = sum(1 for t in trades if t["status"] in ("Pending","Executed"))
    wins   = sum(1 for t in trades if t["status"] == "Success")
    losses = sum(1 for t in trades if t["status"] == "Failed")
    gaps   = sum(1 for t in trades if t.get("gap_exit"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total",   total)
    c2.metric("Active",  active)
    c3.metric("Wins",    wins)
    c4.metric("Losses",  losses)
    c5.metric("Gap Exits", gaps)

    st.markdown("")

    rows = []
    for t in trades:
        op = t.get("outcome_pct")
        rows.append({
            "ID":       t["id"],
            "Date":     t["date"],
            "Ticker":   t["ticker"],
            "Strategy": t["strategy"].replace("filter_","F"),
            "Score":    f"{t.get('score',0):.0f}",
            "Entry":    f"${t.get('entry',0):.2f}",
            "Stop":     f"${t.get('stop',0):.2f}",
            "Target":   f"${t.get('target',0):.2f}",
            "Size":     t.get("position_size","—"),
            "Status":   t["status"],
            "P&L":      f"{op*100:+.2f}%" if op is not None else "—",
            "Gap":      "⚡" if t.get("gap_exit") else "",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    st.download_button("⬇ Export CSV", buf.getvalue(),
                       f"trades_{date.today()}.csv", "text/csv")

    st.markdown("")
    st.markdown('<div class="section-header"><span class="section-title">Update Trade Status</span></div>', unsafe_allow_html=True)

    active_trades = [t for t in trades if t["status"] in ("Pending","Executed")]
    if not active_trades:
        st.info("No active trades to update.")
    else:
        labels = {t["id"]: f"#{t['id']} {t['ticker']} ({t['status']})" for t in active_trades}
        sel_id = st.selectbox("Trade", list(labels.keys()), format_func=lambda x: labels[x])
        new_st = st.selectbox("New Status", ["Executed","Success","Failed","Expired"])

        ep = None
        if new_st in ("Success","Failed"):
            ep = st.number_input("Exit Price ($)", min_value=0.01, value=0.01, format="%.4f")

        if st.button("💾 Update", type="primary"):
            ok = db.update_trade_status(sel_id, new_st,
                                        exit_price=ep if ep and ep > 0.01 else None)
            if ok:
                if new_st in ("Success","Failed","Expired"):
                    lrn.run_all_learning_updates()
                    st.toast("Learning system updated 🧠")
                st.success(f"Trade #{sel_id} → {new_st}")
                st.rerun()
            else:
                st.error("Update failed")

    # Gap exit alerts
    gap_trades = [t for t in trades if t.get("gap_exit")]
    if gap_trades:
        st.markdown("")
        st.markdown('<div class="section-header"><span class="section-title">⚡ Gap Exits</span></div>', unsafe_allow_html=True)
        for t in gap_trades:
            st.warning(f"**{t['ticker']}** gapped below stop on {t.get('exit_date','?')} · Exit: {_price(t.get('exit_price'))} · P&L: {t.get('outcome_pct',0)*100:.2f}%")

    _disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def page_performance():
    _header("📊 Performance", "Live metrics · Equity curve · Backtest")

    closed = db.get_closed_trades(limit=500)
    n = len(closed)

    if n < 20:
        st.warning(f"Only {n} closed trades — metrics are directional only. Need 20+ for reliability.")

    # Per-strategy metrics
    for strat in config.BASE_WEIGHTS.keys():
        st_trades = [t for t in closed if t.get("strategy") == strat]
        if not st_trades:
            continue

        wins   = sum(1 for t in st_trades if t["status"] == "Success")
        total  = len(st_trades)
        wr     = wins / total
        rets   = [t["outcome_pct"] for t in st_trades if t.get("outcome_pct") is not None]
        avg_r  = np.mean(rets) if rets else None

        max_dd = None
        if rets:
            eq   = np.cumprod(1 + np.array(rets))
            pk   = np.maximum.accumulate(eq)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd = np.where(pk > 0, (eq - pk) / pk, 0)
            max_dd = float(np.min(dd))

        st.markdown(f'<div class="section-header"><span class="section-title">{strat.upper().replace("_"," ")}</span></div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades",     total)
        c2.metric("Win Rate",   f"{wr*100:.1f}%")
        c3.metric("Avg Return", f"{avg_r*100:+.2f}%" if avg_r is not None else "—")
        c4.metric("Max DD",     f"{max_dd*100:.1f}%" if max_dd is not None else "—")
        c5.metric("W/L",        f"{wins}/{total-wins}")

    # Equity curve
    closed_sorted = sorted([t for t in closed if t.get("outcome_pct") is not None],
                           key=lambda x: x.get("exit_date") or "")
    if closed_sorted:
        st.markdown("")
        st.markdown('<div class="section-header"><span class="section-title">Equity Curve</span></div>', unsafe_allow_html=True)
        rets_series = [t["outcome_pct"] for t in closed_sorted]
        equity      = np.cumprod(1 + np.array(rets_series))
        dates       = [t.get("exit_date","") for t in closed_sorted]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=equity,
            mode="lines",
            line=dict(color="#388bfd", width=2),
            fill="tozeroy",
            fillcolor="rgba(56,139,253,0.06)",
            name="Portfolio",
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#30363d", opacity=0.8)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#080b12",
            plot_bgcolor="#0d1117",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Multiplier", gridcolor="#161b22"),
            xaxis=dict(gridcolor="#161b22"),
            font=dict(family="Inter", color="#8b949e", size=11),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Backtest results
    st.markdown("")
    st.markdown('<div class="section-header"><span class="section-title">Backtesting-Lite · Last 30 Days</span></div>', unsafe_allow_html=True)
    st.caption("Entry = day's open · Outcome = next-day close vs stop/target · No lookahead")

    if st.button("▶ Run Backtest", type="secondary"):
        ohlcv = st.session_state.get("ohlcv")
        if not ohlcv:
            st.error("Load market data first (click Refresh Data on Signals page)")
        else:
            with st.spinner("Simulating 30 days..."):
                r = bt.run_backtest(ohlcv)
                st.session_state["backtest_result"] = r
            if r.get("error"):
                st.error(r["error"])

    bt_res = st.session_state.get("backtest_result") or {}
    if not bt_res:
        latest = bt.get_latest_backtest_results()
        if latest:
            bt_res = {"strategies": latest}

    if bt_res.get("strategies"):
        for strat, m in bt_res["strategies"].items():
            if not m or not m.get("trades_sampled"):
                continue
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{strat.upper()} Trades", m.get("trades_sampled","—"))
            c2.metric("Win Rate",   f"{m['win_rate']*100:.1f}%" if m.get("win_rate") is not None else "—")
            c3.metric("Avg Return", f"{m['avg_return']*100:+.3f}%" if m.get("avg_return") is not None else "—")
            c4.metric("Max DD",     f"{m['max_drawdown']*100:.2f}%" if m.get("max_drawdown") is not None else "—")
            if m.get("note"):
                st.caption(m["note"])

    _disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def page_insights():
    _header("🧠 Strategy Insights", "Adaptive weights · Bucket win rates · Audit trail")

    summary = lrn.get_learning_summary()

    # Current weights
    st.markdown('<div class="section-header"><span class="section-title">Current Feature Weights</span></div>', unsafe_allow_html=True)

    for strat in config.BASE_WEIGHTS.keys():
        w    = summary["weights"].get(strat, config.BASE_WEIGHTS[strat])
        base = config.BASE_WEIGHTS[strat]

        st.markdown(f"**{strat.upper().replace('_',' ')}**")
        c1, c2, c3 = st.columns(3)
        for col, feat in zip([c1, c2, c3], ["momentum", "volume", "volatility"]):
            cur   = w.get(feat, 0)
            delta = cur - base.get(feat, 0)
            col.metric(feat.capitalize(), f"{cur:.3f}",
                       delta=f"{delta:+.3f}" if abs(delta) > 0.001 else None)

        # Weight bar chart
        feats   = ["momentum", "volume", "volatility"]
        cur_vals = [w.get(f, 0) for f in feats]
        base_vals = [base.get(f, 0) for f in feats]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current", x=feats, y=cur_vals,
                             marker_color="#388bfd", opacity=0.9))
        fig.add_trace(go.Bar(name="Base",    x=feats, y=base_vals,
                             marker_color="#30363d", opacity=0.7))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080b12",
            plot_bgcolor="#0d1117", height=200, barmode="group",
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0,1], gridcolor="#161b22"),
            font=dict(family="Inter", color="#8b949e", size=10),
            legend=dict(orientation="h", y=1.15, font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bucket heatmap
    st.markdown('<div class="section-header"><span class="section-title">Bucket Win Rates</span></div>', unsafe_allow_html=True)
    st.caption(f"4 buckets (momentum × volume) · Min {config.MIN_TRADES_BUCKET} trades to activate")

    for strat in config.BASE_WEIGHTS.keys():
        bstats = summary["bucket_stats"].get(strat, {})
        if not bstats: continue
        st.markdown(f"**{strat.upper().replace('_',' ')}**")
        cols = st.columns(4)
        for col, key in zip(cols, lrn.get_all_bucket_keys()):
            s     = bstats.get(key, {})
            wr    = s.get("win_rate", 0.5)
            count = s.get("trade_count", 0)
            ready = s.get("ready", False)
            label = key.replace("_", " ").replace("mom", "Mom").replace("vol", "Vol")
            color = "#3fb950" if wr > 0.55 else "#f85149" if wr < 0.45 else "#d29922"
            status_str = "✓ Active" if ready else f"⏳ {count}/{config.MIN_TRADES_BUCKET}"
            col.markdown(f'''
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                        padding:0.8rem;text-align:center;margin-bottom:0.5rem">
                <div style="font-size:0.68rem;color:#6e7681;margin-bottom:4px">{label}</div>
                <div style="font-size:1.6rem;font-weight:800;color:{color};font-family:JetBrains Mono">{wr*100:.0f}%</div>
                <div style="font-size:0.68rem;color:#6e7681;margin-top:2px">{count} trades · {status_str}</div>
            </div>''', unsafe_allow_html=True)

    # Weight history
    st.markdown('<div class="section-header"><span class="section-title">Weight History</span></div>', unsafe_allow_html=True)
    for strat in config.BASE_WEIGHTS.keys():
        hist = summary["weights_history"].get(strat, [])
        if len(hist) < 2: continue
        hdf = pd.DataFrame(hist).sort_values("timestamp")
        fig = go.Figure()
        for feat, color in [("momentum_w","#388bfd"),("volume_w","#3fb950"),("volatility_w","#f78166")]:
            if feat in hdf.columns:
                fig.add_trace(go.Scatter(
                    x=hdf["timestamp"], y=hdf[feat],
                    name=feat.replace("_w",""),
                    line=dict(color=color, width=2),
                    mode="lines+markers", marker=dict(size=4),
                ))
        fig.update_layout(
            title=dict(text=strat.upper().replace("_"," "), font=dict(size=11, color="#8b949e")),
            template="plotly_dark", paper_bgcolor="#080b12", plot_bgcolor="#0d1117",
            height=220, margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(range=[0,1], gridcolor="#161b22"),
            xaxis=dict(gridcolor="#161b22"),
            font=dict(family="Inter", color="#8b949e", size=10),
            legend=dict(orientation="h", y=1.2, font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Audit log
    st.markdown('<div class="section-header"><span class="section-title">Recent Weight Changes</span></div>', unsafe_allow_html=True)
    hist_all = db.get_weights_history(limit=10)
    if hist_all:
        rows = [{"Time": h["timestamp"][:16], "Strategy": h["strategy"],
                 "Mom": f"{h['momentum_w']:.3f}", "Vol": f"{h['volume_w']:.3f}",
                 "Vlt": f"{h['volatility_w']:.3f}",
                 "Trigger": h["trigger_event"], "Trades": h.get("trades_count","—")}
                for h in hist_all]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Reset
    st.markdown("")
    st.markdown('<div class="section-header"><span class="section-title">Reset Weights</span></div>', unsafe_allow_html=True)
    confirm = st.checkbox("I understand this discards all learned weight adaptations")
    if confirm:
        if st.button("↺ Reset to Base Weights", type="primary"):
            if db.reset_weights_to_base():
                st.success("Weights reset to base values")
                st.rerun()

    st.markdown(f'<p style="font-size:0.75rem;color:#6e7681;margin-top:1rem">Lookback: {config.ROLLING_LOOKBACK} trades · Decay: {config.DECAY_FACTOR} · α: {config.LEARNING_ALPHA} · Min bucket: {config.MIN_TRADES_BUCKET}</p>', unsafe_allow_html=True)
    _disclaimer()


# ── Disclaimer ────────────────────────────────────────────────────────────────
def _disclaimer():
    st.markdown(f'<div class="disc">{config.DISCLAIMER}</div>', unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────
def main():
    page = _sidebar()
    if   "Signal"  in page: page_signals()
    elif "Watch"   in page: page_watchlist()
    elif "History" in page: page_history()
    elif "Perform" in page: page_performance()
    elif "Insight" in page: page_insights()

if __name__ == "__main__":
    main()
