"""
data_fetch.py — All yfinance interactions and market context.

Responsibilities:
- Single batch yf.download() for entire universe (never per-ticker loops)
- Market hours guard — surfaces staleness to UI without blocking
- SPY context: compute bearish/high-vol flags, persist to DB
- Overnight gap detection: checks if today's open < stop on active trades
- Returns clean DataFrames — no raw yfinance objects leave this module

Import chain: config -> database -> universe -> data_fetch
"""

import logging
import traceback
from datetime import date, datetime, time as dtime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

import config
import database as db

logger     = logging.getLogger("data_fetch")
app_logger = logging.getLogger("app")


# ── Market hours ──────────────────────────────────────────────────────────────
def get_market_status() -> Dict[str, Any]:
    """
    Returns NYSE market status based on current ET time.
    Simple weekday + hour check — no external calendar dependency.
    Does NOT block app functionality — only informs the UI banner.

    Returns dict:
        is_open  : bool
        message  : str  (display in UI banner)
        now_et   : datetime (ET) or None on error
    """
    try:
        tz     = pytz.timezone(config.MARKET_TIMEZONE)
        now_et = datetime.now(tz)
        wday   = now_et.weekday()           # 0=Mon … 6=Sun

        open_t  = dtime(config.MARKET_OPEN_HOUR,  config.MARKET_OPEN_MINUTE)
        close_t = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
        cur_t   = now_et.time()

        if wday >= 5:
            return {
                "is_open": False,
                "message": "⚠ Weekend — NYSE closed. Showing last available data.",
                "now_et":  now_et,
            }
        if cur_t < open_t:
            return {
                "is_open": False,
                "message": f"⚠ Pre-market ({now_et.strftime('%H:%M')} ET) — data from prior session.",
                "now_et":  now_et,
            }
        if cur_t >= close_t:
            return {
                "is_open": False,
                "message": f"⚠ After-hours ({now_et.strftime('%H:%M')} ET) — data from today's session.",
                "now_et":  now_et,
            }
        return {
            "is_open": True,
            "message": f"🟢 Market Open — {now_et.strftime('%H:%M')} ET",
            "now_et":  now_et,
        }
    except Exception as e:
        logger.error("get_market_status: %s", e)
        return {"is_open": False, "message": "⚠ Market status unavailable.", "now_et": None}


# ── OHLCV batch download ──────────────────────────────────────────────────────
def fetch_ohlcv_batch(
    tickers: List[str],
    period:  str = config.FETCH_PERIOD,
    interval: str = config.FETCH_INTERVAL,
) -> Dict[str, pd.DataFrame]:
    """
    Downloads OHLCV data for all tickers in one yf.download() call.
    Splits into 200-ticker chunks to avoid yfinance instability on large lists.

    Returns: {ticker: DataFrame(Open, High, Low, Close, Volume)}
    Tickers with insufficient/all-NaN data are silently excluded with a log entry.
    """
    if not tickers:
        return {}

    results: Dict[str, pd.DataFrame] = {}
    chunk_size = 200

    for chunk_start in range(0, len(tickers), chunk_size):
        chunk = tickers[chunk_start : chunk_start + chunk_size]
        try:
            raw = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,   # adjusts for splits/dividends automatically
                threads=True,
                progress=False,
                show_errors=False,
            )

            if raw.empty:
                logger.warning("fetch_ohlcv_batch: empty response for chunk starting %d", chunk_start)
                continue

            # yfinance >= 0.2.x always returns MultiIndex (field, ticker)
            # even for single-ticker downloads
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in chunk:
                    try:
                        # level=1 is the ticker name, level=0 is field name
                        df = raw.xs(ticker, axis=1, level=1, drop_level=True).copy()
                        df.columns = [str(c) for c in df.columns]
                        # Remove 'Adj Close' if present — we use auto_adjust=True
                        df = df[[c for c in df.columns if c in
                                  ("Open","High","Low","Close","Volume")]]
                        if _is_valid(df, ticker):
                            results[ticker] = df
                    except KeyError:
                        pass  # ticker not returned by yfinance
                    except Exception as e:
                        logger.warning("fetch_ohlcv_batch: parse error %s: %s", ticker, e)
            else:
                # Flat columns — single ticker fallback
                ticker = chunk[0]
                df = raw.copy()
                df = df[[c for c in df.columns if c in
                          ("Open","High","Low","Close","Volume")]]
                if _is_valid(df, ticker):
                    results[ticker] = df

        except Exception as e:
            logger.error(
                "fetch_ohlcv_batch: download failed chunk@%d: %s\n%s",
                chunk_start, e, traceback.format_exc(),
            )

    app_logger.info(
        "fetch_ohlcv_batch: requested %d, received %d tickers",
        len(tickers), len(results),
    )
    return results


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to single level if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _is_valid(df: pd.DataFrame, ticker: str) -> bool:
    """
    Returns True if DataFrame has sufficient valid rows to compute all features.
    Minimum rows = largest window + generous buffer to avoid edge cases.
    """
    min_rows = max(
        config.ATR_WINDOW,
        config.VOL_AVG_WINDOW,
        config.MOMENTUM_WINDOW,
        config.SPY_MA_WINDOW,
    ) + 10  # buffer

    required = {"Open", "High", "Low", "Close", "Volume"}
    if df is None or df.empty:
        return False
    if not required.issubset(set(df.columns)):
        logger.debug("%s: missing columns %s", ticker, required - set(df.columns))
        return False
    valid_rows = df["Close"].notna().sum()
    if valid_rows < min_rows:
        logger.debug("%s: only %d valid rows (need %d)", ticker, valid_rows, min_rows)
        return False
    return True


# ── SPY context ───────────────────────────────────────────────────────────────
def compute_spy_context(spy_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes market regime flags from SPY OHLCV DataFrame.
    All values computed from PAST data only (no lookahead).

    bearish_flag  = 1 if SPY last close < 20-day moving average
    high_vol_flag = 1 if SPY 20-day return std-dev > 1.5% OR > 80th pct of
                    rolling 20-day std-dev over the past 252 trading days.

    Returns dict ready to pass to db.save_spy_context().
    """
    close = spy_df["Close"].dropna()

    spy_price = float(close.iloc[-1])
    spy_20dma = float(close.rolling(config.SPY_MA_WINDOW).mean().iloc[-1])

    daily_ret = close.pct_change().dropna()
    spy_vol_20d = float(daily_ret.rolling(config.SPY_VOL_WINDOW).std().iloc[-1])

    # 80th percentile of rolling 20-day vol over last 252 days
    rolling_vol = daily_ret.rolling(config.SPY_VOL_WINDOW).std().dropna()
    lookback    = min(len(rolling_vol), config.SPY_HIST_WINDOW)
    spy_vol_80pct = float(rolling_vol.iloc[-lookback:].quantile(0.80))

    bearish_flag  = int(spy_price < spy_20dma)
    high_vol_flag = int(
        spy_vol_20d > config.SPY_HIGH_VOL_FIXED or
        spy_vol_20d > spy_vol_80pct
    )

    return {
        "date":          date.today().isoformat(),
        "spy_price":     round(spy_price,  2),
        "spy_20dma":     round(spy_20dma,  2),
        "spy_vol_20d":   round(spy_vol_20d,   6),
        "spy_vol_80pct": round(spy_vol_80pct, 6),
        "bearish_flag":  bearish_flag,
        "high_vol_flag": high_vol_flag,
    }


def fetch_and_store_spy_context(spy_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute SPY flags and persist to DB.
    If spy_df not provided, fetches SPY separately (400-day window for 252-day pct).
    On any failure: loads last known state from DB (graceful degradation).
    """
    try:
        if spy_df is None:
            raw = yf.download(
                "SPY", period="400d", interval="1d",
                auto_adjust=True, progress=False, show_errors=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if raw.empty:
                raise ValueError("SPY download returned empty DataFrame")
            spy_df = raw

        ctx = compute_spy_context(spy_df)
        db.save_spy_context(ctx)
        app_logger.info(
            "SPY context updated: price=%.2f 20dma=%.2f vol_20d=%.4f "
            "bearish=%d high_vol=%d",
            ctx["spy_price"], ctx["spy_20dma"], ctx["spy_vol_20d"],
            ctx["bearish_flag"], ctx["high_vol_flag"],
        )
        return ctx

    except Exception as e:
        logger.error("fetch_and_store_spy_context failed: %s\n%s", e, traceback.format_exc())
        ctx = db.get_spy_context()
        app_logger.warning(
            "SPY fetch failed — using cached context. bearish=%d high_vol=%d",
            ctx.get("bearish_flag", 0), ctx.get("high_vol_flag", 0),
        )
        return ctx


# ── Master data loader ────────────────────────────────────────────────────────
def load_all_data(
    tickers: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], str]:
    """
    Master function called by scoring.py.
    1. Fetches OHLCV for all universe tickers + SPY in one batch
    2. Extracts SPY to compute market context
    3. Persists SPY context to DB

    Returns:
        ohlcv      : {ticker: DataFrame}  (SPY excluded — handled separately)
        spy_ctx    : market context dict
        timestamp  : human-readable freshness string for UI
    """
    all_tickers = list(set(tickers + ["SPY"]))
    app_logger.info("load_all_data: fetching %d tickers...", len(all_tickers))

    ohlcv = fetch_ohlcv_batch(all_tickers)

    # Extract SPY for context — don't pass to scoring
    spy_df = ohlcv.pop("SPY", None)
    spy_ctx = fetch_and_store_spy_context(spy_df)

    timestamp = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"(~{config.DATA_DELAY_MINUTES} min delay)"
    )
    app_logger.info(
        "load_all_data: complete. %d tickers with valid data.", len(ohlcv)
    )
    return ohlcv, spy_ctx, timestamp


# ── Overnight gap detection ───────────────────────────────────────────────────
def check_overnight_gaps(
    active_trades: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Checks Executed trades for overnight gap below stop loss.
    Called at app startup each trading day.

    Gap rule: if today's open < trade stop_loss ->
      close trade at today's open and log true loss.

    Uses period='2d' fetch to get today's open price efficiently.

    Returns list of dicts: {trade_id, exit_price, true_loss_pct}
    """
    executed = [t for t in active_trades if t.get("status") == "Executed"]
    if not executed:
        return []

    tickers      = list({t["ticker"] for t in executed})
    gap_closures = []

    try:
        raw = yf.download(
            tickers=tickers,
            period="2d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            show_errors=False,
        )
        if raw.empty:
            return []

        for trade in executed:
            ticker = trade["ticker"]
            stop   = float(trade["stop"])
            entry  = float(trade["entry"])

            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    try:
                        df = raw.xs(ticker, axis=1, level=1, drop_level=True).copy()
                    except KeyError:
                        continue
                else:
                    df = raw.copy()

                if df.empty or len(df) < 1:
                    continue

                # Today's open is the last row's Open value
                today_open = float(df["Open"].iloc[-1])

                if pd.isna(today_open):
                    continue

                if today_open < stop:
                    # Gap triggered: exit at actual open, compute true loss
                    true_loss_pct = (today_open - entry) / entry
                    gap_closures.append({
                        "trade_id":      trade["id"],
                        "exit_price":    round(today_open, 4),
                        "true_loss_pct": round(true_loss_pct, 6),
                        "ticker":        ticker,
                    })
                    app_logger.warning(
                        "GAP EXIT: %s trade_id=%d open=%.2f stop=%.2f loss=%.2f%%",
                        ticker, trade["id"], today_open, stop, true_loss_pct * 100,
                    )

            except Exception as e:
                logger.error("check_overnight_gaps: error for %s: %s", ticker, e)

    except Exception as e:
        logger.error("check_overnight_gaps: download failed: %s\n%s", e, traceback.format_exc())

    return gap_closures


# ── Single ticker fetch (for watchlist) ──────────────────────────────────────
def fetch_single_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV for one ticker. Used by watchlist search on Page 2.
    Returns None on failure or insufficient data.
    """
    try:
        result = fetch_ohlcv_batch([ticker.upper()])
        return result.get(ticker.upper())
    except Exception as e:
        logger.error("fetch_single_ticker failed for %s: %s", ticker, e)
        return None


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("data_fetch.py self-test")
    print("=" * 50)

    # 1. Market status (no network needed)
    status = get_market_status()
    print(f"\n[1] Market status: {status['message']}")
    print(f"    is_open = {status['is_open']}")

    # 2. Fetch a small batch of real tickers
    print("\n[2] Fetching OHLCV for ['AAPL', 'MSFT', 'SPY'] ...")
    data = fetch_ohlcv_batch(["AAPL", "MSFT", "SPY"], period="60d")
    for t, df in data.items():
        print(f"    {t}: {len(df)} rows, last close = {df['Close'].iloc[-1]:.2f}")

    # 3. SPY context
    print("\n[3] Computing SPY context...")
    spy_df = data.get("SPY")
    if spy_df is not None:
        ctx = compute_spy_context(spy_df)
        print(f"    SPY price   : {ctx['spy_price']}")
        print(f"    SPY 20DMA   : {ctx['spy_20dma']}")
        print(f"    Vol 20d     : {ctx['spy_vol_20d']:.4f}")
        print(f"    Vol 80pct   : {ctx['spy_vol_80pct']:.4f}")
        print(f"    Bearish flag: {ctx['bearish_flag']}")
        print(f"    High-vol    : {ctx['high_vol_flag']}")
    else:
        print("    SPY not in fetch results")

    # 4. load_all_data (integration)
    print("\n[4] load_all_data(['AAPL','MSFT','AMZN']) ...")
    import database as db
    db.init_db()
    ohlcv, spy_ctx, ts = load_all_data(["AAPL", "MSFT", "AMZN"])
    print(f"    Loaded {len(ohlcv)} tickers | timestamp: {ts}")
    print(f"    SPY bearish={spy_ctx['bearish_flag']} high_vol={spy_ctx['high_vol_flag']}")

    # 5. Gap detection with mock data (no open trades in fresh DB)
    print("\n[5] Gap detection (no active trades in fresh DB — expected empty list):")
    gaps = check_overnight_gaps([])
    print(f"    Gap closures: {gaps}  (expected: [])")

    print("\ndata_fetch.py self-test complete.")
