"""
app.py — Streamlit UI for the Trading Assistant.

5 Pages:
  1. Daily Top Trades     — generate signals, show full explanation + risk
  2. Live Watchlist       — search any ticker, track custom list
  3. Trade History        — view/update all trades, export CSV
  4. Performance Metrics  — live stats, backtest results, equity curve
  5. Strategy Insights    — weights, learning curves, reset button

Global header on every page:
  - Data timestamp + delay notice
  - Market open/closed banner
  - Universe staleness warning
  - DB health indicator

Run: streamlit run app.py
"""

import io
import traceback
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import config
import database as db
import universe as uni
import data_fetch as df_mod
import scoring as sc
import risk as rk
import learning as lrn
import backtest as bt

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — clean dark financial aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  .stApp { background-color: #0e1117; color: #e0e0e0; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2d3446;
    border-radius: 8px;
    padding: 0.8rem 1rem;
  }

  /* Trade cards */
  .trade-card {
    background: #1a1f2e;
    border-left: 4px solid #00d4aa;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
  }
  .trade-card-risky {
    border-left-color: #ff6b35;
  }
  .trade-card-expired {
    border-left-color: #666;
    opacity: 0.7;
  }

  /* Status pills */
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .pill-pending  { background:#2d3a5e; color:#7ba7ff; }
  .pill-executed { background:#2d4a3e; color:#00d4aa; }
  .pill-success  { background:#1e4a2e; color:#4cff88; }
  .pill-failed   { background:#4a1e1e; color:#ff6b6b; }
  .pill-expired  { background:#2a2a2a; color:#888;    }

  /* Banner */
  .banner-green  { background:#1a3a2a; border:1px solid #00d4aa; border-radius:6px; padding:0.5rem 1rem; color:#00d4aa; }
  .banner-yellow { background:#3a3a1a; border:1px solid #ffd700; border-radius:6px; padding:0.5rem 1rem; color:#ffd700; }
  .banner-red    { background:#3a1a1a; border:1px solid #ff6b6b; border-radius:6px; padding:0.5rem 1rem; color:#ff6b6b; }
  .banner-gray   { background:#1a1a1a; border:1px solid #444;    border-radius:6px; padding:0.5rem 1rem; color:#aaa; }

  /* Explanation block */
  .explanation {
    background:#111827;
    border-radius:6px;
    padding:0.8rem;
    font-family: monospace;
    font-size: 0.82rem;
    color: #b0bec5;
    white-space: pre-wrap;
  }

  /* Section headers */
  h3 { color: #00d4aa; border-bottom: 1px solid #2d3446; padding-bottom: 4px; }

  /* Disclaimer */
  .disclaimer {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    color: #888;
    font-size: 0.75rem;
    margin-top: 2rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "ohlcv":            None,
        "spy_ctx":          None,
        "data_timestamp":   None,
        "last_fetch_date":  None,
        "daily_trades":     [],
        "pipeline_log":     {},
        "backtest_result":  None,
        "refresh_progress": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP TASKS (run once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _startup():
    """DB init + auto-expire old pending trades. Runs once per Streamlit process."""
    db.init_db()
    expired = db.expire_old_pending_trades()
    return expired

_expired_count = _startup()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _status_pill(status: str) -> str:
    cls = f"pill-{status.lower()}"
    return f'<span class="pill {cls}">{status}</span>'


def _fmt_pct(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    color = "#4cff88" if v >= 0 else "#ff6b6b"
    return f'<span style="color:{color}">{sign}{v*100:.{decimals}f}%</span>'


def _fmt_price(v: Optional[float]) -> str:
    return f"${v:,.2f}" if v is not None else "—"


def _load_market_data(force: bool = False) -> bool:
    """
    Load OHLCV + SPY context into session state.
    Caches within the session for the current trading day.
    Returns True if data was freshly fetched, False if using cache.
    """
    today = date.today().isoformat()
    if (
        not force
        and st.session_state["ohlcv"] is not None
        and st.session_state["last_fetch_date"] == today
    ):
        return False  # Already loaded today

    tickers = uni.get_tickers_by_strategy()
    if not tickers:
        return False

    with st.spinner("⏳ Fetching market data..."):
        ohlcv, spy_ctx, timestamp = df_mod.load_all_data(tickers)
        st.session_state["ohlcv"]           = ohlcv
        st.session_state["spy_ctx"]         = spy_ctx
        st.session_state["data_timestamp"]  = timestamp
        st.session_state["last_fetch_date"] = today

        # Check overnight gaps on startup
        active = db.get_active_trades()
        gaps   = df_mod.check_overnight_gaps(active)
        for gap in gaps:
            db.update_trade_status(
                gap["trade_id"], "Failed",
                exit_price=gap["exit_price"],
                gap_exit=True,
            )
            lrn.run_all_learning_updates()
        if gaps:
            st.toast(f"⚠ {len(gaps)} overnight gap exit(s) processed", icon="🌙")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL HEADER
# ─────────────────────────────────────────────────────────────────────────────
def _render_global_header():
    """Renders status banners shown at top of every page."""
    col1, col2, col3 = st.columns([2, 2, 1])

    # Market status
    with col1:
        mkt = df_mod.get_market_status()
        if mkt["is_open"]:
            st.markdown(f'<div class="banner-green">🟢 {mkt["message"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="banner-yellow">{mkt["message"]}</div>',
                        unsafe_allow_html=True)

    # Data freshness
    with col2:
        ts = st.session_state.get("data_timestamp")
        if ts:
            st.markdown(
                f'<div class="banner-gray">📡 Data: {ts}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="banner-gray">📡 No data loaded yet</div>',
                unsafe_allow_html=True,
            )

    # Universe status
    with col3:
        u_status = uni.get_universe_status()
        if not u_status["exists"]:
            st.markdown('<div class="banner-red">⚠ No universe</div>',
                        unsafe_allow_html=True)
        elif u_status["stale"]:
            st.markdown(
                f'<div class="banner-yellow">⚠ Universe {u_status["age_days"]}d old</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="banner-green">✅ {u_status["total_count"]} tickers</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: DAILY TOP TRADES
# ─────────────────────────────────────────────────────────────────────────────
def page_daily_trades():
    st.title("📈 Daily Trade Signals")
    _render_global_header()

    # Portfolio capacity summary
    summary = rk.get_portfolio_risk_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Trades",  f"{summary['active_count']}/{config.MAX_ACTIVE_TRADES}")
    c2.metric("Slots Available", summary["slots_free"])
    c3.metric("Capital at Risk", f"${summary['total_risk_$']:,.0f}")
    c4.metric("Risk Used",       f"{summary['pct_used']}%")

    st.divider()

    # Generate button
    col_btn, col_refresh = st.columns([2, 1])
    with col_btn:
        generate = st.button(
            "🚀 Generate Today's Trades",
            type="primary",
            disabled=summary["at_capacity"],
            help="Runs the full selection pipeline and generates up to 3 trade signals.",
        )
    with col_refresh:
        if st.button("🔄 Refresh Data", help="Force re-fetch market data"):
            _load_market_data(force=True)
            st.rerun()

    if summary["at_capacity"]:
        st.warning(
            f"Portfolio at capacity — {summary['active_count']} active trades. "
            "Close or expire existing trades before generating new signals."
        )
        _show_active_trades_mini(summary)
        _render_disclaimer()
        return

    # Load data if needed
    if st.session_state["ohlcv"] is None:
        u_status = uni.get_universe_status()
        if not u_status["exists"]:
            st.error("Universe not found. Go to the sidebar and click 'Refresh Universe' first.")
            _render_disclaimer()
            return
        _load_market_data()

    if generate:
        ohlcv   = st.session_state.get("ohlcv", {})
        spy_ctx = st.session_state.get("spy_ctx", {"bearish_flag": 0, "high_vol_flag": 0})

        if not ohlcv:
            st.error("No market data available. Check universe setup and internet connection.")
            _render_disclaimer()
            return

        with st.spinner("🔍 Running selection pipeline..."):
            try:
                # Selection pipeline
                candidates, pipeline_log = sc.run_selection_pipeline(ohlcv, spy_ctx)

                # Attach risk levels
                approved, rejections = rk.attach_risk_levels(candidates, ohlcv)

                st.session_state["daily_trades"]  = approved
                st.session_state["pipeline_log"]  = pipeline_log

                # Insert into DB
                today_str = date.today().isoformat()
                for trade in approved:
                    db.insert_trade({
                        "date":         today_str,
                        "ticker":       trade["ticker"],
                        "strategy":     trade["strategy"],
                        "score":        trade["score"],
                        "entry":        trade["entry"],
                        "stop":         trade["stop"],
                        "target":       trade["target"],
                        "position_size": trade["position_size"],
                        "momentum":     trade["momentum"],
                        "volume_spike": trade["volume_spike"],
                        "volatility":   trade["volatility"],
                        "atr":          trade["atr"],
                        "sector":       trade.get("sector", "Unknown"),
                        "explanation":  trade.get("explanation", ""),
                    })
            except Exception as e:
                st.error(f"Pipeline error: {e}\nCheck logs/errors.log for details.")
                st.code(traceback.format_exc())
                _render_disclaimer()
                return

    # Show trades
    trades = st.session_state.get("daily_trades", [])
    log    = st.session_state.get("pipeline_log", {})

    if trades:
        st.success(f"✅ {len(trades)} trade signal(s) generated")

        # Pipeline stats expander
        with st.expander("📊 Pipeline Stats", expanded=False):
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Universe Size",    log.get("universe_size", "—"))
            pc2.metric("Features Computed", log.get("features_computed", "—"))
            pc3.metric("After SPY Filter",  log.get("after_spy", "—"))
            pc4.metric("Selected",          log.get("selected", "—"))
            if log.get("skipped_reasons"):
                st.caption("Skipped reasons: " + " | ".join(log["skipped_reasons"][:5]))

        # SPY context
        spy = st.session_state.get("spy_ctx", {})
        _render_spy_banner(spy)
        st.markdown("<br>", unsafe_allow_html=True)

        # Trade cards
        for i, trade in enumerate(trades):
            _render_trade_card(trade, i + 1)

    elif log:
        st.info("No trades generated. Check pipeline stats for reasons.")
        with st.expander("📊 Pipeline Stats"):
            st.json(log)
    else:
        st.info("Click **Generate Today's Trades** to run the signal pipeline.")

    _render_disclaimer()


def _render_spy_banner(spy: Dict):
    bearish  = spy.get("bearish_flag", 0)
    high_vol = spy.get("high_vol_flag", 0)
    price    = spy.get("spy_price", "—")
    dma      = spy.get("spy_20dma", "—")

    if high_vol:
        st.markdown(
            f'<div class="banner-red">⚠ SPY HIGH-VOL REGIME — Filter B disabled | '
            f'SPY ${price} vs 20DMA ${dma:.2f}</div>' if isinstance(dma, float) else
            f'<div class="banner-red">⚠ SPY HIGH-VOL REGIME — Filter B disabled</div>',
            unsafe_allow_html=True,
        )
    elif bearish:
        st.markdown(
            f'<div class="banner-yellow">📉 SPY BEARISH — All scores reduced 20% | '
            f'SPY ${price} < 20DMA ${dma:.2f}</div>' if isinstance(dma, float) else
            f'<div class="banner-yellow">📉 SPY BEARISH — All scores reduced 20%</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner-green">✅ SPY NEUTRAL | '
            f'SPY ${price} > 20DMA ${dma:.2f}</div>' if isinstance(dma, float) else
            f'<div class="banner-green">✅ SPY NEUTRAL</div>',
            unsafe_allow_html=True,
        )


def _render_trade_card(trade: Dict, rank: int):
    is_risky = trade.get("strategy") == "filter_b"
    card_cls = "trade-card trade-card-risky" if is_risky else "trade-card"

    ticker   = trade["ticker"]
    score    = trade.get("score", 0)
    strategy = trade.get("strategy", "—").upper()
    sector   = trade.get("sector", "Unknown")
    entry    = trade.get("entry", 0)
    stop     = trade.get("stop", 0)
    target   = trade.get("target", 0)
    size     = trade.get("position_size", 0)
    risk_usd = trade.get("dollar_risk", 0)
    rr       = trade.get("rr_ratio", 0)

    stop_pct   = (entry - stop)   / entry * 100 if entry else 0
    target_pct = (target - entry) / entry * 100 if entry else 0

    st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

    with col1:
        st.markdown(f"### #{rank} {ticker}")
        st.caption(f"{strategy} | {sector} | Score: **{score:.1f}/100**")

    with col2:
        st.metric("Entry",  _fmt_price(entry))
        st.metric("Stop",   f"{_fmt_price(stop)} (-{stop_pct:.1f}%)")

    with col3:
        st.metric("Target", f"{_fmt_price(target)} (+{target_pct:.1f}%)")
        st.metric("R:R",    f"{rr:.1f}:1")

    with col4:
        st.metric("Shares",   str(size))
        st.metric("$ at Risk", f"${risk_usd:,.0f}")

    with st.expander(f"📋 Signal Explanation — {ticker}"):
        explanation = trade.get("explanation", "No explanation available.")
        st.markdown(
            f'<div class="explanation">{explanation}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")


def _show_active_trades_mini(summary: Dict):
    """Show compact active trade list when portfolio is at capacity."""
    st.markdown("### Active Trades")
    active = db.get_active_trades()
    if active:
        rows = []
        for t in active:
            rows.append({
                "Ticker":   t["ticker"],
                "Strategy": t["strategy"],
                "Entry":    _fmt_price(t.get("entry")),
                "Stop":     _fmt_price(t.get("stop")),
                "Target":   _fmt_price(t.get("target")),
                "Status":   t["status"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: LIVE WATCHLIST
# ─────────────────────────────────────────────────────────────────────────────
def page_watchlist():
    st.title("👁 Live Watchlist")
    _render_global_header()

    # Search bar
    st.markdown("### 🔍 Score Any Ticker")
    col_search, col_strat, col_go = st.columns([3, 2, 1])

    with col_search:
        search_ticker = st.text_input(
            "Ticker symbol",
            placeholder="e.g. AAPL",
            label_visibility="collapsed",
        ).upper().strip()
    with col_strat:
        search_strategy = st.selectbox(
            "Strategy",
            ["filter_a", "filter_b"],
            label_visibility="collapsed",
        )
    with col_go:
        search_go = st.button("Score", type="primary")

    if search_go and search_ticker:
        with st.spinner(f"Fetching {search_ticker}..."):
            ticker_df = df_mod.fetch_single_ticker(search_ticker)

        if ticker_df is None or ticker_df.empty:
            st.error(f"Could not fetch data for **{search_ticker}**. Check ticker symbol.")
        else:
            spy_ctx = st.session_state.get("spy_ctx") or db.get_spy_context()
            result  = sc.score_single_ticker(
                search_ticker, ticker_df, spy_ctx, strategy=search_strategy
            )

            if result is None:
                st.warning(f"Could not compute features for **{search_ticker}** — insufficient data.")
            else:
                stop, target, dollar_risk, rr = rk.calculate_trade_risk(
                    result["entry"], result["atr"]
                )
                size = rk.calculate_position_size(result["entry"], stop, dollar_risk)

                st.success(f"**{search_ticker}** scored successfully")

                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Score",    f"{result['score']:.1f}/100")
                r2.metric("Entry",    _fmt_price(result["entry"]))
                r3.metric("Stop",     _fmt_price(stop))
                r4.metric("Target",   _fmt_price(target))
                r5.metric("Shares",   str(size))

                with st.expander("📋 Full Signal Explanation"):
                    st.markdown(
                        f'<div class="explanation">{result["explanation"]}</div>',
                        unsafe_allow_html=True,
                    )

                # Add to watchlist button
                col_add, col_note = st.columns([1, 3])
                with col_add:
                    if st.button(f"➕ Add to Watchlist"):
                        db.add_to_watchlist(search_ticker)
                        st.success(f"{search_ticker} added to watchlist")
                        st.rerun()
                with col_note:
                    note = st.text_input("Note (optional)", key="wl_note")

    st.divider()

    # Watchlist table
    st.markdown("### 📋 My Watchlist")
    watchlist = db.get_watchlist(active_only=True)

    if not watchlist:
        st.info("Your watchlist is empty. Search for a ticker above and add it.")
    else:
        # Rescore all watchlist tickers
        spy_ctx = st.session_state.get("spy_ctx") or db.get_spy_context()
        rows = []
        for wl in watchlist:
            ticker = wl["ticker"]
            ticker_df = (st.session_state.get("ohlcv") or {}).get(ticker)

            if ticker_df is not None:
                scored = sc.score_single_ticker(ticker, ticker_df, spy_ctx)
                if scored:
                    stop, target, _, rr = rk.calculate_trade_risk(
                        scored["entry"], scored["atr"]
                    )
                    rows.append({
                        "Ticker":      ticker,
                        "Price":       _fmt_price(scored["entry"]),
                        "Score":       f"{scored['score']:.1f}",
                        "Momentum":    f"{scored['momentum']*100:+.2f}%",
                        "Vol Spike":   f"{scored['volume_spike']:.2f}×",
                        "Stop":        _fmt_price(stop),
                        "Target":      _fmt_price(target),
                        "R:R":         f"{rr:.1f}:1",
                        "Added":       wl["date_added"],
                        "Notes":       wl.get("notes", ""),
                    })
                    continue

            rows.append({
                "Ticker": ticker, "Price": "—", "Score": "—",
                "Momentum": "—", "Vol Spike": "—",
                "Stop": "—", "Target": "—", "R:R": "—",
                "Added": wl["date_added"], "Notes": wl.get("notes", ""),
            })

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        # Remove ticker
        st.markdown("#### Remove from Watchlist")
        tickers_in_wl = [w["ticker"] for w in watchlist]
        remove_ticker = st.selectbox("Select ticker to remove", ["—"] + tickers_in_wl)
        if st.button("🗑 Remove") and remove_ticker != "—":
            db.remove_from_watchlist(remove_ticker)
            st.success(f"{remove_ticker} removed")
            st.rerun()

    _render_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def page_trade_history():
    st.title("📚 Trade History & Tracking")
    _render_global_header()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All"] + list(config.TRADE_STATUSES),
        )
    with col_f2:
        strategy_filter = st.selectbox(
            "Filter by Strategy",
            ["All", "filter_a", "filter_b"],
        )
    with col_f3:
        limit = st.number_input("Max rows", min_value=10, max_value=500, value=100)

    trades = db.get_all_trades(
        status_filter=None if status_filter == "All" else status_filter,
        strategy_filter=None if strategy_filter == "All" else strategy_filter,
        limit=int(limit),
    )

    if not trades:
        st.info("No trades found. Generate signals on Page 1 first.")
        _render_disclaimer()
        return

    # Summary stats
    total     = len(trades)
    active    = sum(1 for t in trades if t["status"] in ("Pending", "Executed"))
    closed    = sum(1 for t in trades if t["status"] in ("Success", "Failed", "Expired"))
    gap_exits = sum(1 for t in trades if t.get("gap_exit"))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Trades",  total)
    s2.metric("Active",        active)
    s3.metric("Closed",        closed)
    s4.metric("Gap Exits",     gap_exits)

    st.divider()

    # Trades table
    display_rows = []
    for t in trades:
        outcome = t.get("outcome_pct")
        display_rows.append({
            "ID":       t["id"],
            "Date":     t["date"],
            "Ticker":   t["ticker"],
            "Strategy": t["strategy"],
            "Score":    f"{t.get('score',0):.1f}",
            "Entry":    _fmt_price(t.get("entry")),
            "Stop":     _fmt_price(t.get("stop")),
            "Target":   _fmt_price(t.get("target")),
            "Shares":   t.get("position_size", "—"),
            "Status":   t["status"],
            "Outcome":  f"{outcome*100:+.2f}%" if outcome is not None else "—",
            "Gap":      "⚡" if t.get("gap_exit") else "",
            "Sector":   t.get("sector", "—"),
        })

    df_display = pd.DataFrame(display_rows)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # CSV Export
    csv_buf = io.StringIO()
    df_display.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇ Export to CSV",
        data=csv_buf.getvalue(),
        file_name=f"trades_{date.today().isoformat()}.csv",
        mime="text/csv",
    )

    st.divider()

    # Manual status update
    st.markdown("### ✏️ Update Trade Status")
    trade_ids   = [t["id"] for t in trades if t["status"] in ("Pending", "Executed")]
    trade_labels = {
        t["id"]: f"#{t['id']} {t['ticker']} ({t['status']})"
        for t in trades if t["status"] in ("Pending", "Executed")
    }

    if not trade_ids:
        st.info("No active trades to update.")
    else:
        selected_id = st.selectbox(
            "Select Trade",
            trade_ids,
            format_func=lambda x: trade_labels.get(x, str(x)),
        )
        new_status = st.selectbox(
            "New Status",
            ["Executed", "Success", "Failed", "Expired"],
        )

        exit_price = None
        if new_status in ("Success", "Failed"):
            exit_price = st.number_input(
                "Exit Price ($)",
                min_value=0.01,
                value=0.01,
                format="%.4f",
            )

        if st.button("💾 Update Status", type="primary"):
            ok = db.update_trade_status(
                selected_id,
                new_status,
                exit_price=exit_price if exit_price and exit_price > 0.01 else None,
            )
            if ok:
                st.success(f"Trade #{selected_id} updated to {new_status}")
                # Trigger learning update on close
                if new_status in ("Success", "Failed", "Expired"):
                    lrn.run_all_learning_updates()
                    st.toast("📚 Learning system updated", icon="🧠")
                st.rerun()
            else:
                st.error("Update failed — check logs/errors.log")

    # Overnight gap alerts
    gap_trades = [t for t in trades if t.get("gap_exit")]
    if gap_trades:
        st.divider()
        st.markdown("### ⚡ Overnight Gap Exits")
        for t in gap_trades:
            st.warning(
                f"**{t['ticker']}** — Gap exit on {t.get('exit_date','?')} | "
                f"Entry: {_fmt_price(t.get('entry'))} | "
                f"Exit (open): {_fmt_price(t.get('exit_price'))} | "
                f"Loss: {t.get('outcome_pct',0)*100:.2f}%"
            )

    _render_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def page_performance():
    st.title("📊 Performance Metrics & Backtest")
    _render_global_header()

    closed_trades = db.get_closed_trades(limit=500)
    total_closed  = len(closed_trades)

    if total_closed < 20:
        st.warning(
            f"⚠ Only {total_closed} closed trades. "
            "Metrics are unreliable with fewer than 20 closed trades — "
            "treat these numbers as directional only."
        )

    # ── Live performance metrics ───────────────────────────────────────────────
    st.markdown("### 📈 Live Performance")

    for strategy in config.BASE_WEIGHTS.keys():
        strat_trades = [t for t in closed_trades if t.get("strategy") == strategy]
        if not strat_trades:
            st.caption(f"No closed trades for **{strategy}** yet.")
            continue

        wins     = [t for t in strat_trades if t["status"] == "Success"]
        losses   = [t for t in strat_trades if t["status"] in ("Failed", "Expired")]
        win_rate = len(wins) / len(strat_trades) if strat_trades else 0

        returns = [t["outcome_pct"] for t in strat_trades if t.get("outcome_pct") is not None]
        avg_ret = np.mean(returns) if returns else None

        # Max drawdown from equity curve
        max_dd = None
        if returns:
            equity = np.cumprod(1 + np.array(returns))
            peak   = np.maximum.accumulate(equity)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd = np.where(peak > 0, (equity - peak) / peak, 0)
            max_dd = float(np.min(dd))

        st.markdown(f"#### {strategy.upper()}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades",       len(strat_trades))
        m2.metric("Win Rate",     f"{win_rate*100:.1f}%")
        m3.metric("Avg Return",   f"{avg_ret*100:+.2f}%" if avg_ret else "—")
        m4.metric("Max Drawdown", f"{max_dd*100:.2f}%"   if max_dd else "—")
        m5.metric("Wins/Losses",  f"{len(wins)}/{len(losses)}")

        # Save live metrics to DB
        if strat_trades:
            db.save_metrics({
                "date":           date.today().isoformat(),
                "strategy":       strategy,
                "win_rate":       win_rate,
                "avg_return":     avg_ret,
                "max_drawdown":   max_dd,
                "trades_sampled": len(strat_trades),
                "source":         "live",
            })

    # ── Equity curve ──────────────────────────────────────────────────────────
    if closed_trades:
        st.markdown("### 📉 Equity Curve (All Closed Trades)")
        sorted_trades = sorted(
            [t for t in closed_trades if t.get("outcome_pct") is not None],
            key=lambda x: x.get("exit_date") or "",
        )
        if sorted_trades:
            returns_series = [t["outcome_pct"] for t in sorted_trades]
            equity_curve   = np.cumprod(1 + np.array(returns_series))
            dates          = [t.get("exit_date", "") for t in sorted_trades]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=equity_curve,
                mode="lines+markers",
                line=dict(color="#00d4aa", width=2),
                marker=dict(size=5),
                name="Equity",
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.08)",
            ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="#666", opacity=0.5)
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Portfolio Multiplier",
                xaxis_title="Trade Exit Date",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Backtest ───────────────────────────────────────────────────────────────
    st.markdown("### 🔬 Backtesting-Lite (Last 30 Days)")

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_bt = st.button("▶ Run Backtest", type="secondary")
    with col_info:
        st.caption(
            "Simulates the last 30 days using current weights. "
            "Entry = day's open. Outcome = next-day close vs stop/target. "
            "Strict no-lookahead: all features computed from past data only."
        )

    if run_bt:
        ohlcv = st.session_state.get("ohlcv")
        if not ohlcv:
            st.error("Load market data first (Page 1 → Refresh Data)")
        else:
            with st.spinner("Running backtest simulation..."):
                result = bt.run_backtest(ohlcv)
                st.session_state["backtest_result"] = result
            if result.get("error"):
                st.error(f"Backtest error: {result['error']}")
            else:
                st.success(
                    f"Backtest complete — "
                    f"{len(result.get('all_sim_trades',[]))} simulated trades"
                )

    # Show latest backtest results
    bt_result = st.session_state.get("backtest_result") or {}
    if not bt_result:
        latest = bt.get_latest_backtest_results()
        if latest:
            st.caption("Showing last saved backtest results:")
            bt_result = {"strategies": {k: v for k, v in latest.items()}}

    if bt_result.get("strategies"):
        for strategy, metrics in bt_result["strategies"].items():
            if not metrics or metrics.get("trades_sampled", 0) == 0:
                st.caption(f"No backtest data for {strategy}")
                continue

            st.markdown(f"#### {strategy.upper()} — Backtest")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Trades Simulated", metrics.get("trades_sampled", "—"))
            b2.metric("Win Rate",
                      f"{metrics['win_rate']*100:.1f}%" if metrics.get("win_rate") else "—")
            b3.metric("Avg Return",
                      f"{metrics['avg_return']*100:+.3f}%" if metrics.get("avg_return") else "—")
            b4.metric("Max Drawdown",
                      f"{metrics['max_drawdown']*100:.2f}%" if metrics.get("max_drawdown") else "—")
            if metrics.get("note"):
                st.caption(f"ℹ {metrics['note']}")

    _render_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: STRATEGY INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def page_strategy_insights():
    st.title("🧠 Strategy Insights")
    _render_global_header()

    summary = lrn.get_learning_summary()

    # ── Current weights ────────────────────────────────────────────────────────
    st.markdown("### ⚖️ Current Feature Weights")

    for strategy in config.BASE_WEIGHTS.keys():
        w    = summary["weights"].get(strategy, config.BASE_WEIGHTS[strategy])
        base = config.BASE_WEIGHTS[strategy]

        st.markdown(f"#### {strategy.upper()}")
        wc1, wc2, wc3 = st.columns(3)

        for col, feature in zip([wc1, wc2, wc3], ["momentum", "volume", "volatility"]):
            current = w.get(feature, 0)
            base_v  = base.get(feature, 0)
            delta   = current - base_v
            col.metric(
                feature.capitalize(),
                f"{current:.3f}",
                delta=f"{delta:+.3f}" if abs(delta) > 0.001 else None,
                help=f"Base: {base_v:.3f}",
            )

        # Weight bar chart
        features = ["momentum", "volume", "volatility"]
        values   = [w.get(f, 0) for f in features]
        base_vals = [base.get(f, 0) for f in features]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current", x=features, y=values,
            marker_color="#00d4aa", opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="Base", x=features, y=base_vals,
            marker_color="#555", opacity=0.6,
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=220,
            barmode="group",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Weight history ─────────────────────────────────────────────────────────
    st.markdown("### 📈 Weight History (Drift Over Time)")

    for strategy in config.BASE_WEIGHTS.keys():
        history = summary["weights_history"].get(strategy, [])
        if len(history) < 2:
            st.caption(f"Not enough history for {strategy} yet.")
            continue

        hist_df = pd.DataFrame(history).sort_values("timestamp")

        fig = go.Figure()
        for feature, color in [
            ("momentum_w", "#00d4aa"),
            ("volume_w",   "#7ba7ff"),
            ("volatility_w", "#ff6b35"),
        ]:
            if feature in hist_df.columns:
                fig.add_trace(go.Scatter(
                    x=hist_df["timestamp"],
                    y=hist_df[feature],
                    name=feature.replace("_w", ""),
                    line=dict(color=color, width=2),
                    mode="lines+markers",
                ))
        fig.update_layout(
            title=strategy.upper(),
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=260,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(range=[0, 1], title="Weight"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Bucket win rate heatmap ────────────────────────────────────────────────
    st.markdown("### 🔥 Bucket Win Rates")

    for strategy in config.BASE_WEIGHTS.keys():
        bucket_stats = summary["bucket_stats"].get(strategy, {})
        if not bucket_stats:
            continue

        st.markdown(f"#### {strategy.upper()}")
        bucket_keys = lrn.get_all_bucket_keys()

        bc_cols = st.columns(len(bucket_keys))
        for col, key in zip(bc_cols, bucket_keys):
            stats = bucket_stats.get(key, {})
            wr    = stats.get("win_rate", 0.5)
            count = stats.get("trade_count", 0)
            ready = stats.get("ready", False)

            label  = key.replace("_", " ").title()
            color  = "#4cff88" if wr > 0.55 else "#ff6b6b" if wr < 0.45 else "#ffd700"
            status = "✅" if ready else f"⏳ {count}/{config.MIN_TRADES_BUCKET}"

            col.markdown(
                f"""<div style="background:#1a1f2e;border-radius:8px;padding:12px;text-align:center">
                <div style="font-size:0.75rem;color:#888">{label}</div>
                <div style="font-size:1.4rem;font-weight:700;color:{color}">{wr*100:.0f}%</div>
                <div style="font-size:0.72rem;color:#666">{count} trades {status}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Learning log ───────────────────────────────────────────────────────────
    st.markdown("### 📋 Recent Weight Changes")
    history_all = db.get_weights_history(limit=15)
    if history_all:
        log_rows = []
        for h in history_all:
            log_rows.append({
                "Timestamp":  h["timestamp"][:19],
                "Strategy":   h["strategy"],
                "Momentum":   f"{h['momentum_w']:.3f}",
                "Volume":     f"{h['volume_w']:.3f}",
                "Volatility": f"{h['volatility_w']:.3f}",
                "Trigger":    h["trigger_event"],
                "# Trades":   h.get("trades_count", "—"),
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No weight change history yet.")

    # ── Reset weights button ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚠️ Reset Weights")
    st.warning(
        "Resetting weights discards all learned adaptations and "
        "returns to base weights. This cannot be undone (though history is preserved)."
    )
    confirm = st.checkbox("I understand — reset weights to base values")
    if confirm:
        if st.button("🔄 Reset All Weights to Base", type="primary"):
            ok = db.reset_weights_to_base()
            if ok:
                st.success("✅ Weights reset to base values.")
                st.rerun()
            else:
                st.error("Reset failed — check logs/errors.log")

    # Summary stats
    st.divider()
    st.caption(
        f"Total closed trades: {summary['total_closed']} | "
        f"Rolling lookback: {config.ROLLING_LOOKBACK} | "
        f"Decay factor: {config.DECAY_FACTOR} | "
        f"Min bucket size: {config.MIN_TRADES_BUCKET} | "
        f"Learning α: {config.LEARNING_ALPHA}"
    )

    _render_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Trading Assistant")
        st.caption(f"Capital: ${config.CAPITAL:,.0f}")
        st.caption(f"Risk/trade: {config.RISK_PER_TRADE*100:.1f}%")

        st.divider()

        # Universe management
        st.markdown("### 🌍 Universe")
        u_status = uni.get_universe_status()

        if u_status["exists"]:
            st.caption(
                f"Filter A: {u_status['filter_a_count']} | "
                f"Filter B: {u_status['filter_b_count']}\n"
                f"Updated: {u_status['last_updated']}"
            )
            if u_status["stale"]:
                st.warning(f"⚠ {u_status['age_days']}d old — refresh needed")
        else:
            st.error("No universe file")

        if st.button("🔄 Refresh Universe", help="~2-5 minutes. Fetches S&P 500 + filters."):
            progress_msgs = []
            progress_placeholder = st.empty()

            def on_progress(msg):
                progress_msgs.append(msg)
                progress_placeholder.info("\n".join(progress_msgs[-3:]))

            with st.spinner("Refreshing universe..."):
                ok, msg = uni.refresh_universe(progress_callback=on_progress)

            if ok:
                st.success(msg)
                _load_market_data(force=True)
            else:
                st.error(msg)

        st.divider()

        # DB health
        st.markdown("### 🗄 Database")
        health = db.db_health_check()
        if health["ok"]:
            st.success(
                f"✅ OK | {health['trade_count']} trades | "
                f"{health['active_trades']} active"
            )
        else:
            st.error(f"❌ DB error: {health['error']}")

        st.divider()
        st.caption("v1.0 | All signals educational only")


# ─────────────────────────────────────────────────────────────────────────────
# DISCLAIMER FOOTER
# ─────────────────────────────────────────────────────────────────────────────
def _render_disclaimer():
    st.markdown(
        f'<div class="disclaimer">{config.DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — PAGE ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _render_sidebar()

    pages = {
        "📈 Daily Trades":       page_daily_trades,
        "👁 Watchlist":          page_watchlist,
        "📚 Trade History":      page_trade_history,
        "📊 Performance":        page_performance,
        "🧠 Strategy Insights":  page_strategy_insights,
    }

    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            list(pages.keys()),
            label_visibility="collapsed",
        )

    pages[page]()


if __name__ == "__main__":
    main()
