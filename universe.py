"""
universe.py — Ticker universe management.

Responsibilities:
- Generate and cache the stock universe as data/universe.csv
- Apply Filter A (reputable: mcap>$2B, vol>1M, price>$5)
- Apply Filter B (risky: price<$5, volume spike)
- Cache sector data in CSV — fetched ONCE per weekly refresh, never live
- Staleness check + UI warning support
- Never loops yf.Ticker() during daily scoring — batch only

Flow:
  refresh_universe()  <- called by UI button (expensive, ~2-5 min)
  load_universe()     <- called by scoring.py (instant CSV read)
"""

import os
import time
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger     = logging.getLogger("universe")
app_logger = logging.getLogger("app")

# ── Filter B seed list ────────────────────────────────────────────────────────
# Known high-volume tickers typically under $5. Price/volume filters are
# re-applied on every weekly refresh — this is just the candidate pool.
FILTER_B_SEED = [
    "SIRI","VALE","NOK","ITUB","BBD","NIO","XPEV","SNAP","PLTR","F",
    "AAL","CCL","NCLH","AMC","GME","SOFI","OPEN","UWMC","RKT","MVIS",
    "SNDL","TLRY","CGC","ACB","HEXO","CRON","ZNGA","MARA","RIOT","HUT",
    "BITF","CLSK","CIFR","BTBT","OCGN","NVAX","SRNE","IDEX","GNUS",
    "TTOO","NURO","MNMD","CMPS","ATAI","NRXP","SAVA","ACER","VISL",
    "PRTS","AMRX","ARBK","BTCM","GRWG","VFF","OGI","BYND","PTON",
    "SPCE","SKLZ","DKNG","HIMS","FSR","GOEV","RIDE","WKHS","NKLA",
]


# ── S&P 500 fetch ─────────────────────────────────────────────────────────────
def get_sp500_tickers() -> List[str]:
    """
    Fetch S&P 500 constituents from Wikipedia (free, no API key).
    Cleans ticker symbols (Wikipedia uses '.' where markets use '-').
    Returns empty list on failure — caller handles.
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        tickers = tables[0]["Symbol"].tolist()
        tickers = [str(t).replace(".", "-").strip().upper() for t in tickers]
        app_logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as e:
        logger.error("get_sp500_tickers failed: %s\n%s", e, traceback.format_exc())
        return []


# ── Info batch fetch ──────────────────────────────────────────────────────────
def _fetch_info_batch(
    tickers: List[str],
    batch_size: int = 40,
    sleep_sec: float = 1.2,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict]:
    """
    Fetches market_cap, avg_volume, price, sector for each ticker.
    Uses yf.Ticker.fast_info (lightweight) + .info only for sector.
    Batches with sleep to avoid yfinance rate limiting.

    Returns: {ticker: {market_cap, avg_volume, price, sector}}
    Missing/errored tickers get None values — filtered out downstream.
    """
    results: Dict[str, Dict] = {}
    total   = len(tickers)
    batches = (total - 1) // batch_size + 1

    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        if progress:
            progress(f"  Fetching info batch {batch_num}/{batches} ({len(batch)} tickers)...")

        for ticker in batch:
            entry: Dict[str, Any] = {
                "market_cap": None,
                "avg_volume":  None,
                "price":       None,
                "sector":      "Unknown",
            }
            try:
                t  = yf.Ticker(ticker)
                fi = t.fast_info

                entry["market_cap"] = getattr(fi, "market_cap", None)
                entry["avg_volume"] = getattr(fi, "three_month_average_volume", None)
                entry["price"]      = getattr(fi, "last_price", None)

                # Sector requires heavier .info call — wrap separately
                try:
                    info = t.info
                    entry["sector"] = info.get("sector") or "Unknown"
                except Exception:
                    pass  # sector unavailable — leave as Unknown

            except Exception as e:
                logger.warning("_fetch_info_batch: error for %s: %s", ticker, e)

            results[ticker] = entry

        if i + batch_size < total:
            time.sleep(sleep_sec)

    return results


# ── Filter application ────────────────────────────────────────────────────────
def _apply_filter_a(info: Dict[str, Dict]) -> List[Dict]:
    """
    Filter A — Reputable stocks.
    Criteria: market_cap > $2B, avg_volume > 1M, price > $5.
    Any None/NaN field -> skip ticker (never propagate bad data).
    """
    out = []
    for ticker, d in info.items():
        mc    = d.get("market_cap")
        vol   = d.get("avg_volume")
        price = d.get("price")

        if any(v is None for v in (mc, vol, price)):
            continue
        if any(isinstance(v, float) and np.isnan(v) for v in (mc, vol, price)):
            continue

        if (mc    >= config.FILTER_A_MIN_MCAP  and
            vol   >= config.FILTER_A_MIN_VOL   and
            price >= config.FILTER_A_MIN_PRICE):
            out.append({
                "ticker":     ticker,
                "strategy":   "filter_a",
                "sector":     d.get("sector", "Unknown"),
                "market_cap": mc,
                "avg_volume": vol,
                "price":      price,
            })
    return out


def _apply_filter_b(info: Dict[str, Dict], fa_tickers: set) -> List[Dict]:
    """
    Filter B — Speculative / momentum tickers.
    Criteria: price < $5. Volume spike verified at scoring time (dynamic).
    Tickers already in Filter A are excluded.
    """
    out = []
    for ticker, d in info.items():
        if ticker in fa_tickers:
            continue

        price = d.get("price")
        if price is None or (isinstance(price, float) and np.isnan(price)):
            continue
        if price >= config.FILTER_B_MAX_PRICE:
            continue

        out.append({
            "ticker":     ticker,
            "strategy":   "filter_b",
            "sector":     d.get("sector", "Unknown"),
            "market_cap": d.get("market_cap") or 0,
            "avg_volume": d.get("avg_volume") or 0,
            "price":      price,
        })
    return out


# ── Main refresh ──────────────────────────────────────────────────────────────
def refresh_universe(
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Full universe refresh. Expected runtime: 2-5 minutes.
    Called by UI 'Refresh Universe' button — never called during daily scoring.

    Returns (success: bool, message: str) — never raises.
    """
    def prog(msg: str):
        app_logger.info("refresh_universe: %s", msg)
        if progress_callback:
            progress_callback(msg)

    try:
        prog("Step 1/5 — Fetching S&P 500 ticker list from Wikipedia...")
        sp500 = get_sp500_tickers()
        if not sp500:
            return False, "Failed to fetch S&P 500 tickers. Check internet connection."

        candidates = list(set(sp500 + FILTER_B_SEED))
        prog(f"Step 2/5 — {len(candidates)} candidates. Fetching market data (slow step)...")

        info = _fetch_info_batch(candidates, progress=prog)
        prog(f"Step 3/5 — Data fetched for {len(info)} tickers. Applying filters...")

        fa_rows = _apply_filter_a(info)
        fa_set  = {r["ticker"] for r in fa_rows}
        fb_rows = _apply_filter_b(info, fa_set)
        all_rows = fa_rows + fb_rows

        prog(
            f"Step 4/5 — Filter A: {len(fa_rows)}, "
            f"Filter B: {len(fb_rows)} = {len(all_rows)} total. Saving..."
        )

        if not all_rows:
            return False, (
                "No tickers passed filters. "
                "Check FILTER_A_MIN_MCAP / FILTER_A_MIN_PRICE in config.py."
            )

        df = pd.DataFrame(all_rows)
        df["last_updated"] = datetime.now().isoformat(timespec="seconds")
        df.to_csv(config.UNIVERSE_CSV, index=False)

        msg = (
            f"Universe refreshed: {len(fa_rows)} Filter A + "
            f"{len(fb_rows)} Filter B = {len(all_rows)} total tickers."
        )
        prog(f"Step 5/5 — {msg}")
        return True, msg

    except Exception as e:
        msg = f"Universe refresh failed: {e}"
        logger.error("%s\n%s", msg, traceback.format_exc())
        return False, msg


# ── Read-only helpers (called daily) ─────────────────────────────────────────
def load_universe() -> pd.DataFrame:
    """
    Load universe CSV. Returns empty DataFrame if file missing.
    Columns: ticker, strategy, sector, market_cap, avg_volume, price, last_updated
    """
    try:
        if not os.path.exists(config.UNIVERSE_CSV):
            return pd.DataFrame()
        df = pd.read_csv(config.UNIVERSE_CSV, dtype={"ticker": str})
        df["ticker"] = df["ticker"].str.upper().str.strip()
        df = df.dropna(subset=["ticker"])
        return df
    except Exception as e:
        logger.error("load_universe failed: %s", e)
        return pd.DataFrame()


def get_universe_status() -> Dict[str, Any]:
    """
    Returns metadata about the universe file for UI status banner.
    Never raises.
    """
    status: Dict[str, Any] = {
        "exists": False, "age_days": None, "stale": True,
        "filter_a_count": 0, "filter_b_count": 0,
        "total_count": 0, "last_updated": None, "message": "",
    }
    try:
        if not os.path.exists(config.UNIVERSE_CSV):
            status["message"] = (
                "universe.csv not found. "
                "Click 'Refresh Universe' to generate it (~2-5 min)."
            )
            return status

        mtime    = os.path.getmtime(config.UNIVERSE_CSV)
        age_days = (datetime.now().timestamp() - mtime) / 86_400

        status.update({
            "exists":       True,
            "age_days":     round(age_days, 1),
            "stale":        age_days > config.UNIVERSE_STALE_DAYS,
            "last_updated": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
        })

        df = load_universe()
        if not df.empty and "strategy" in df.columns:
            status["filter_a_count"] = int((df["strategy"] == "filter_a").sum())
            status["filter_b_count"] = int((df["strategy"] == "filter_b").sum())
            status["total_count"]    = len(df)

        if status["stale"]:
            status["message"] = (
                f"Universe is {age_days:.1f} days old "
                f"(threshold: {config.UNIVERSE_STALE_DAYS}d). Refresh recommended."
            )
        else:
            status["message"] = (
                f"Universe fresh ({age_days:.1f}d old) — "
                f"{status['total_count']} tickers."
            )
    except Exception as e:
        status["message"] = f"Error checking universe: {e}"
        logger.error("get_universe_status: %s", e)

    return status


def get_tickers_by_strategy(strategy: Optional[str] = None) -> List[str]:
    """Returns ticker list from universe, optionally filtered by strategy."""
    df = load_universe()
    if df.empty:
        return []
    if strategy:
        df = df[df["strategy"] == strategy]
    return df["ticker"].tolist()


def get_sector(ticker: str) -> str:
    """Cached sector lookup from universe CSV. Falls back to 'Unknown'."""
    df = load_universe()
    if df.empty:
        return "Unknown"
    row = df[df["ticker"] == ticker.upper()]
    if row.empty:
        return "Unknown"
    val = row.iloc[0].get("sector", "Unknown")
    return str(val) if val and str(val) != "nan" else "Unknown"


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("universe.py self-test (no network)")

    status = get_universe_status()
    print(f"Status : {status['message']}")

    df = load_universe()
    if not df.empty:
        print(f"Loaded : {len(df)} tickers | cols: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
    else:
        print("No universe.csv — run refresh_universe() to generate.")

    # Mock filter tests
    mock_info = {
        "AAPL":  {"market_cap": 3e12, "avg_volume": 80e6, "price": 190.0, "sector": "Technology"},
        "MID":   {"market_cap": 5e9,  "avg_volume": 2e6,  "price": 45.0,  "sector": "Financials"},
        "PENNY": {"market_cap": 300e6,"avg_volume": 2e6,  "price": 2.50,  "sector": "Healthcare"},
        "CHEAP": {"market_cap": 100e6,"avg_volume": 500e3,"price": 1.20,  "sector": "Unknown"},
        "JUNK":  {"market_cap": None, "avg_volume": None, "price": None,  "sector": "Unknown"},
    }
    fa = _apply_filter_a(mock_info)
    fb = _apply_filter_b(mock_info, {r["ticker"] for r in fa})
    print(f"\nFilter A: {[r['ticker'] for r in fa]}  (expect: AAPL, MID)")
    print(f"Filter B: {[r['ticker'] for r in fb]}  (expect: PENNY, CHEAP)")
    assert "JUNK" not in [r["ticker"] for r in fa + fb], "JUNK should be skipped"
    print("All mock filter assertions passed.")
    print("\nuniverse.py self-test complete.")
