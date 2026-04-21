"""
data_fetch.py  —  Trading Assistant
Data layer: Polygon.io replaces yfinance.
All public function signatures are identical to the original so no other
module needs changing.

Polygon.io free tier limits:
  - 5 API calls / minute
  - Previous-day close only (no real-time on free tier)
  - Historical OHLCV going back years — fully sufficient for signals

Set POLYGON_API_KEY in your .env / HF Space secrets.

Functions (same signatures as original):
  fetch_ohlcv_batch(tickers, days=30)  -> dict[str, pd.DataFrame]
  load_all_data(tickers)               -> dict[str, pd.DataFrame]
  fetch_and_store_spy_context(db)      -> dict
  check_overnight_gaps(db, data)       -> list[str]
  get_market_status()                  -> dict
"""

import os
import time
import logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE    = "https://api.polygon.io"

# Rate limiter: free tier = 5 req/min → 1 req per 12s to be safe
_RATE_LIMIT_DELAY = 13   # seconds between calls on free tier
_last_call_time   = 0.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _rate_limit():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _RATE_LIMIT_DELAY:
        time.sleep(_RATE_LIMIT_DELAY - elapsed)
    _last_call_time = time.time()


def _get(endpoint: str, params: dict) -> dict | None:
    """Single authenticated GET with basic error handling."""
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set")
        return None
    params["apiKey"] = POLYGON_API_KEY
    _rate_limit()
    try:
        r = requests.get(f"{POLYGON_BASE}{endpoint}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.warning(f"Polygon request failed for {endpoint}: {e}")
        return None


def _date_range(days: int) -> tuple[str, str]:
    """Return (from_date, to_date) strings covering `days` calendar days."""
    today = date.today()
    # Add buffer for weekends / holidays — fetch extra days, pandas will have gaps
    from_dt = today - timedelta(days=days + 14)
    return from_dt.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def _bars_to_df(results: list) -> pd.DataFrame:
    """Convert Polygon aggregate bars to a standard OHLCV DataFrame."""
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    # Polygon uses 't' = unix ms timestamp
    df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date
    df = df.rename(columns={
        "o": "Open", "h": "High", "l": "Low",
        "c": "Close", "v": "Volume"
    })
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ── single ticker fetch ───────────────────────────────────────────────────────

def _fetch_one(ticker: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for one ticker via Polygon /v2/aggs.
    Returns empty DataFrame on failure.
    """
    from_date, to_date = _date_range(days)
    data = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
        params={"adjusted": "true", "sort": "asc", "limit": 120}
    )
    if not data or data.get("resultsCount", 0) == 0:
        logger.warning(f"No data returned for {ticker}")
        return pd.DataFrame()
    return _bars_to_df(data.get("results", []))


# ── public API ────────────────────────────────────────────────────────────────

def fetch_ohlcv_batch(tickers: list[str], days: int = 30) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for a list of tickers.
    Returns dict: { ticker: DataFrame }

    Polygon free tier has no true batch endpoint — we iterate with rate limiting.
    For the free tier this will be slow on large universes; see load_all_data()
    for the tiered retry strategy.
    """
    result: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        logger.info(f"Fetching {ticker} ({i+1}/{len(tickers)})")
        df = _fetch_one(ticker, days=days)
        if not df.empty:
            result[ticker] = df
        else:
            logger.warning(f"Skipping {ticker} — empty response")
    return result


def load_all_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    3-level retry strategy (mirrors original yfinance logic):

      Level 1 — full universe, all tickers
      Level 2 — if >20% failed, retry failed tickers individually
      Level 3 — if still failing, return top 10 most liquid (hardcoded fallback)

    Returns dict: { ticker: DataFrame }
    """
    # ── Level 1: full batch ──
    logger.info(f"load_all_data: Level 1 — fetching {len(tickers)} tickers")
    data = fetch_ohlcv_batch(tickers, days=30)
    failed = [t for t in tickers if t not in data]

    # ── Level 2: retry failed ──
    if failed and len(failed) / len(tickers) > 0.20:
        logger.warning(f"Level 2 retry for {len(failed)} failed tickers")
        retry_data = fetch_ohlcv_batch(failed, days=30)
        data.update(retry_data)
        failed = [t for t in tickers if t not in data]

    # ── Level 3: minimal fallback ──
    if len(data) == 0:
        logger.error("Level 3 fallback — fetching top 10 liquid tickers only")
        FALLBACK = ["SPY","AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","JPM","V"]
        data = fetch_ohlcv_batch(FALLBACK, days=30)

    logger.info(f"load_all_data: got data for {len(data)}/{len(tickers)} tickers")
    return data


def fetch_and_store_spy_context(db) -> dict:
    """
    Fetch SPY OHLCV, compute regime flags, store in DB.
    Falls back to cached DB value if Polygon call fails.

    Returns:
        {
          "spy_price":    float,
          "spy_20dma":    float,
          "spy_vol_20d":  float,
          "spy_vol_80pct":float,
          "bearish_flag": int,   # 1 if price < 20dMA
          "high_vol_flag":int    # 1 if 20d vol > 1.5% OR > 80th pct
        }
    """
    df = _fetch_one("SPY", days=90)

    if df.empty:
        logger.warning("SPY fetch failed — loading from DB cache")
        cached = db.get_spy_context()
        return cached if cached else {
            "spy_price": 0, "spy_20dma": 0, "spy_vol_20d": 0,
            "spy_vol_80pct": 0, "bearish_flag": 0, "high_vol_flag": 0
        }

    df["return"] = df["Close"].pct_change()
    spy_price    = float(df["Close"].iloc[-1])
    spy_20dma    = float(df["Close"].tail(20).mean())
    spy_vol_20d  = float(df["return"].tail(20).std())

    # 80th percentile of rolling 20d vol over the last 60 days
    rolling_vols  = [df["return"].iloc[max(0,i-20):i].std()
                     for i in range(20, len(df))]
    spy_vol_80pct = float(np.percentile(rolling_vols, 80)) if rolling_vols else spy_vol_20d

    bearish_flag  = 1 if spy_price < spy_20dma else 0
    high_vol_flag = 1 if (spy_vol_20d > 0.015 or spy_vol_20d > spy_vol_80pct) else 0

    context = {
        "date":          date.today().isoformat(),
        "spy_price":     round(spy_price, 2),
        "spy_20dma":     round(spy_20dma, 2),
        "spy_vol_20d":   round(spy_vol_20d, 6),
        "spy_vol_80pct": round(spy_vol_80pct, 6),
        "bearish_flag":  bearish_flag,
        "high_vol_flag": high_vol_flag,
    }

    try:
        db.save_spy_context(context)
        logger.info(f"SPY context saved: price={spy_price:.2f}, "
                    f"20dma={spy_20dma:.2f}, bearish={bearish_flag}, hi_vol={high_vol_flag}")
    except Exception as e:
        logger.warning(f"Failed to save SPY context to DB: {e}")

    return context


def check_overnight_gaps(db, data: dict[str, pd.DataFrame]) -> list[str]:
    """
    For each Executed trade, check if latest open gapped below the stop price.
    Returns list of tickers that gapped below stop (need to be closed).
    """
    active_trades = db.get_active_trades()
    gapped = []

    for trade in active_trades:
        if trade.get("status") != "Executed":
            continue
        ticker = trade["ticker"]
        stop   = float(trade["stop"])

        if ticker not in data or data[ticker].empty:
            continue

        latest_open = float(data[ticker]["Open"].iloc[-1])
        if latest_open < stop:
            logger.warning(f"Overnight gap detected: {ticker} opened at "
                           f"{latest_open:.2f} below stop {stop:.2f}")
            gapped.append(ticker)

    return gapped


def get_market_status() -> dict:
    """
    Returns NYSE market status using ET timezone only — no external dependency.

    Returns:
        {
          "is_open":       bool,
          "current_et":    str,   # HH:MM
          "session":       str,   # "pre", "open", "after", "closed"
          "next_open_et":  str
        }
    """
    ET   = ZoneInfo("America/New_York")
    now  = datetime.now(ET)
    dow  = now.weekday()   # 0=Mon, 6=Sun
    t    = now.time()

    OPEN_TIME  = datetime.strptime("09:30", "%H:%M").time()
    CLOSE_TIME = datetime.strptime("16:00", "%H:%M").time()
    PRE_TIME   = datetime.strptime("04:00", "%H:%M").time()
    AFTER_TIME = datetime.strptime("20:00", "%H:%M").time()

    is_weekday = dow < 5   # Mon–Fri only (no holiday check)

    if not is_weekday:
        session = "closed"
        is_open = False
    elif t < PRE_TIME:
        session = "closed"
        is_open = False
    elif t < OPEN_TIME:
        session = "pre"
        is_open = False
    elif t <= CLOSE_TIME:
        session = "open"
        is_open = True
    elif t <= AFTER_TIME:
        session = "after"
        is_open = False
    else:
        session = "closed"
        is_open = False

    # Next open: if today is open-eligible and before open → today
    # else next weekday
    if is_weekday and t < OPEN_TIME:
        next_open = now.strftime("%Y-%m-%d") + " 09:30 ET"
    else:
        days_ahead = 1
        while True:
            candidate = now + timedelta(days=days_ahead)
            if candidate.weekday() < 5:
                next_open = candidate.strftime("%Y-%m-%d") + " 09:30 ET"
                break
            days_ahead += 1

    return {
        "is_open":      is_open,
        "current_et":   now.strftime("%H:%M"),
        "session":      session,
        "next_open_et": next_open,
    }


# ── convenience: get latest close for a single ticker (used by watchlist) ──

def get_latest_price(ticker: str) -> float | None:
    """
    Return the most recent closing price for a single ticker.
    Useful for the Watchlist page live price display.
    """
    df = _fetch_one(ticker, days=5)
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])
