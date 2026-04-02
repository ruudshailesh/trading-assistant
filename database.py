"""
database.py — Supabase Python client version.

Uses SUPABASE_URL + SUPABASE_KEY (secret key) from environment variables.
No direct database connection needed — all operations go through Supabase REST API.

All public function signatures are IDENTICAL to the SQLite version.
No other module needs to change.

Required environment variables:
    SUPABASE_URL  — https://xxxxxxxxxxxx.supabase.co
    SUPABASE_KEY  — sb_secret_... (your secret key, NOT publishable)

Set locally:  .env file (never commit)
Set on Railway: Variables tab in your service
"""

import logging
import os
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

import config

# ── Load .env locally (no-op on Railway) ─────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=config.ERROR_LOG,
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger     = logging.getLogger("database")
app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)
_fh = logging.FileHandler(config.APP_LOG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
app_logger.addHandler(_fh)


# ── Supabase client (created once, reused) ────────────────────────────────────
_client: Optional[Client] = None

def _get_client() -> Client:
    """
    Returns the Supabase client. Creates it on first call.
    Raises EnvironmentError if credentials are missing.
    """
    global _client
    if _client is not None:
        return _client

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()

    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set.\n"
            "Local: add them to your .env file.\n"
            "Railway: add them in the Variables tab."
        )

    _client = create_client(url, key)
    return _client


def _sb() -> Client:
    """Shorthand for _get_client()."""
    return _get_client()


# ── Schema creation via SQL ───────────────────────────────────────────────────
# Supabase Python client doesn't run raw DDL directly.
# Tables must be created via Supabase SQL Editor (one-time setup).
# init_db() checks if tables exist and prints instructions if not.

CREATE_TABLES_SQL = """
-- Run this ONCE in Supabase SQL Editor (Database → SQL Editor → New query)

CREATE TABLE IF NOT EXISTS trades (
    id            BIGSERIAL PRIMARY KEY,
    date          TEXT        NOT NULL,
    ticker        TEXT        NOT NULL,
    strategy      TEXT        NOT NULL,
    score         REAL,
    entry         REAL        NOT NULL,
    stop          REAL        NOT NULL,
    target        REAL        NOT NULL,
    position_size INTEGER     NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'Pending',
    outcome_pct   REAL,
    momentum      REAL,
    volume_spike  REAL,
    volatility    REAL,
    atr           REAL,
    sector        TEXT,
    exit_price    REAL,
    exit_date     TEXT,
    gap_exit      INTEGER     NOT NULL DEFAULT 0,
    explanation   TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id             BIGSERIAL PRIMARY KEY,
    date           TEXT    NOT NULL,
    strategy       TEXT    NOT NULL,
    win_rate       REAL,
    avg_return     REAL,
    max_drawdown   REAL,
    trades_sampled INTEGER,
    source         TEXT    DEFAULT 'live'
);

CREATE TABLE IF NOT EXISTS weights (
    strategy     TEXT PRIMARY KEY,
    momentum_w   REAL NOT NULL,
    volume_w     REAL NOT NULL,
    volatility_w REAL NOT NULL,
    last_updated TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weights_history (
    id            BIGSERIAL PRIMARY KEY,
    timestamp     TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    momentum_w    REAL NOT NULL,
    volume_w      REAL NOT NULL,
    volatility_w  REAL NOT NULL,
    trigger_event TEXT NOT NULL,
    trades_count  INTEGER
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker        TEXT PRIMARY KEY,
    date_added    TEXT    NOT NULL,
    active_status INTEGER NOT NULL DEFAULT 1,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS spy_context (
    id            INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    date          TEXT,
    spy_price     REAL,
    spy_20dma     REAL,
    spy_vol_20d   REAL,
    spy_vol_80pct REAL,
    bearish_flag  INTEGER NOT NULL DEFAULT 0,
    high_vol_flag INTEGER NOT NULL DEFAULT 0
);

-- Seed spy_context row
INSERT INTO spy_context (id, bearish_flag, high_vol_flag)
VALUES (1, 0, 0) ON CONFLICT (id) DO NOTHING;

-- Disable Row Level Security (since only your server accesses this)
ALTER TABLE trades          DISABLE ROW LEVEL SECURITY;
ALTER TABLE metrics         DISABLE ROW LEVEL SECURITY;
ALTER TABLE weights         DISABLE ROW LEVEL SECURITY;
ALTER TABLE weights_history DISABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist       DISABLE ROW LEVEL SECURITY;
ALTER TABLE spy_context     DISABLE ROW LEVEL SECURITY;
"""


# ── Init ──────────────────────────────────────────────────────────────────────
def init_db() -> bool:
    """
    Checks DB connectivity and seeds default weights.
    Tables must already exist (created via SQL Editor — see CREATE_TABLES_SQL above).
    Prints helpful instructions if tables are missing.
    """
    try:
        sb = _sb()

        # Check connectivity by querying weights table
        try:
            sb.table("weights").select("strategy").limit(1).execute()
        except Exception:
            print("\n" + "="*60)
            print("FIRST TIME SETUP REQUIRED")
            print("="*60)
            print("Tables not found in Supabase.")
            print("Please run the SQL below in:")
            print("Supabase → SQL Editor → New query → paste → Run\n")
            print(CREATE_TABLES_SQL)
            print("="*60 + "\n")
            return False

        _seed_default_weights()
        app_logger.info("Supabase DB ready")
        return True

    except Exception as e:
        logger.error("init_db failed: %s\n%s", e, traceback.format_exc())
        return False


def _seed_default_weights():
    """Insert BASE_WEIGHTS for each strategy if not already present."""
    sb  = _sb()
    now = datetime.now(timezone.utc).isoformat()

    for strategy, w in config.BASE_WEIGHTS.items():
        existing = (
            sb.table("weights")
            .select("strategy")
            .eq("strategy", strategy)
            .execute()
        )
        if not existing.data:
            sb.table("weights").insert({
                "strategy":    strategy,
                "momentum_w":  w["momentum"],
                "volume_w":    w["volume"],
                "volatility_w": w["volatility"],
                "last_updated": now,
            }).execute()
            sb.table("weights_history").insert({
                "timestamp":    now,
                "strategy":     strategy,
                "momentum_w":   w["momentum"],
                "volume_w":     w["volume"],
                "volatility_w": w["volatility"],
                "trigger_event": "init",
                "trades_count": 0,
            }).execute()


# ── Trades ────────────────────────────────────────────────────────────────────
def insert_trade(trade: Dict[str, Any]) -> Optional[int]:
    """Insert new trade. Returns row id or None on failure."""
    try:
        res = _sb().table("trades").insert({
            "date":          trade["date"],
            "ticker":        trade["ticker"],
            "strategy":      trade["strategy"],
            "score":         trade.get("score"),
            "entry":         trade["entry"],
            "stop":          trade["stop"],
            "target":        trade["target"],
            "position_size": trade["position_size"],
            "status":        "Pending",
            "momentum":      trade.get("momentum"),
            "volume_spike":  trade.get("volume_spike"),
            "volatility":    trade.get("volatility"),
            "atr":           trade.get("atr"),
            "sector":        trade.get("sector"),
            "explanation":   trade.get("explanation"),
        }).execute()
        if res.data:
            return res.data[0]["id"]
        return None
    except Exception as e:
        logger.error("insert_trade failed for %s: %s", trade.get("ticker"), e)
        return None


def update_trade_status(
    trade_id:   int,
    status:     str,
    exit_price: Optional[float] = None,
    exit_date:  Optional[str]   = None,
    gap_exit:   bool            = False,
) -> bool:
    """Update trade status and optionally record exit details."""
    if status not in config.TRADE_STATUSES:
        logger.error("Invalid status: %s", status)
        return False
    try:
        sb = _sb()

        # Fetch entry price to compute outcome_pct
        row = sb.table("trades").select("entry").eq("id", trade_id).execute()
        if not row.data:
            logger.error("Trade id=%s not found", trade_id)
            return False

        entry       = row.data[0].get("entry")
        outcome_pct = None
        if exit_price and entry:
            outcome_pct = (exit_price - entry) / entry

        sb.table("trades").update({
            "status":      status,
            "exit_price":  exit_price,
            "exit_date":   exit_date or date.today().isoformat(),
            "outcome_pct": outcome_pct,
            "gap_exit":    1 if gap_exit else 0,
        }).eq("id", trade_id).execute()

        app_logger.info(
            "Trade id=%s → %s exit=%.4f outcome=%s gap=%s",
            trade_id, status, exit_price or 0, outcome_pct, gap_exit,
        )
        return True
    except Exception as e:
        logger.error("update_trade_status failed id=%s: %s", trade_id, e)
        return False


def get_active_trades() -> List[Dict[str, Any]]:
    try:
        res = (
            _sb().table("trades")
            .select("*")
            .in_("status", ["Pending", "Executed"])
            .order("date", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error("get_active_trades failed: %s", e)
        return []


def get_all_trades(
    status_filter:   Optional[str] = None,
    strategy_filter: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    try:
        q = _sb().table("trades").select("*")
        if status_filter:
            q = q.eq("status", status_filter)
        if strategy_filter:
            q = q.eq("strategy", strategy_filter)
        res = q.order("date", desc=True).limit(limit).execute()
        return res.data or []
    except Exception as e:
        logger.error("get_all_trades failed: %s", e)
        return []


def get_closed_trades(strategy: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    try:
        q = (
            _sb().table("trades")
            .select("*")
            .in_("status", ["Success", "Failed", "Expired"])
        )
        if strategy:
            q = q.eq("strategy", strategy)
        res = q.order("exit_date", desc=True).limit(limit).execute()
        return res.data or []
    except Exception as e:
        logger.error("get_closed_trades failed: %s", e)
        return []


def check_wash_sale(ticker: str) -> bool:
    try:
        cutoff = (date.today() - timedelta(days=config.WASH_SALE_DAYS)).isoformat()
        res = (
            _sb().table("trades")
            .select("id, outcome_pct, status")
            .eq("ticker", ticker.upper())
            .in_("status", ["Failed", "Expired"])
            .gte("exit_date", cutoff)
            .execute()
        )
        if not res.data:
            return False
        # Check for losses or unconfirmed expired trades
        for row in res.data:
            op = row.get("outcome_pct")
            if op is not None and op < 0:
                return True
            if row.get("status") == "Expired" and op is None:
                return True
        return False
    except Exception as e:
        logger.error("check_wash_sale failed for %s: %s", ticker, e)
        return False


def get_tickers_in_active_trades() -> List[str]:
    try:
        res = (
            _sb().table("trades")
            .select("ticker")
            .in_("status", ["Pending", "Executed"])
            .execute()
        )
        return list({r["ticker"] for r in (res.data or [])})
    except Exception as e:
        logger.error("get_tickers_in_active_trades failed: %s", e)
        return []


def expire_old_pending_trades() -> int:
    try:
        cutoff = (
            date.today() - timedelta(days=config.SIGNAL_EXPIRY_DAYS * 1.5)
        ).isoformat()
        res = (
            _sb().table("trades")
            .update({
                "status":      "Expired",
                "exit_date":   date.today().isoformat(),
                "outcome_pct": -1.0,
            })
            .eq("status", "Pending")
            .lte("date", cutoff)
            .execute()
        )
        count = len(res.data) if res.data else 0
        if count:
            app_logger.info("Auto-expired %d old Pending trade(s)", count)
        return count
    except Exception as e:
        logger.error("expire_old_pending_trades failed: %s", e)
        return 0


# ── Weights ───────────────────────────────────────────────────────────────────
def get_weights(strategy: str) -> Dict[str, float]:
    try:
        res = (
            _sb().table("weights")
            .select("momentum_w, volume_w, volatility_w")
            .eq("strategy", strategy)
            .execute()
        )
        if res.data:
            row = res.data[0]
            return {
                "momentum":   row["momentum_w"],
                "volume":     row["volume_w"],
                "volatility": row["volatility_w"],
            }
    except Exception as e:
        logger.error("get_weights failed for %s: %s", strategy, e)
    return config.BASE_WEIGHTS[strategy].copy()


def update_weights(
    strategy:      str,
    new_weights:   Dict[str, float],
    trigger_event: str = "learning_update",
    trades_count:  int = 0,
) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    try:
        sb = _sb()
        sb.table("weights").update({
            "momentum_w":   new_weights["momentum"],
            "volume_w":     new_weights["volume"],
            "volatility_w": new_weights["volatility"],
            "last_updated": now,
        }).eq("strategy", strategy).execute()

        sb.table("weights_history").insert({
            "timestamp":     now,
            "strategy":      strategy,
            "momentum_w":    new_weights["momentum"],
            "volume_w":      new_weights["volume"],
            "volatility_w":  new_weights["volatility"],
            "trigger_event": trigger_event,
            "trades_count":  trades_count,
        }).execute()
        return True
    except Exception as e:
        logger.error("update_weights failed for %s: %s", strategy, e)
        return False


def reset_weights_to_base() -> bool:
    return all(
        update_weights(s, w, trigger_event="manual_reset", trades_count=0)
        for s, w in config.BASE_WEIGHTS.items()
    )


def get_weights_history(strategy: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    try:
        q = _sb().table("weights_history").select("*")
        if strategy:
            q = q.eq("strategy", strategy)
        res = q.order("timestamp", desc=True).limit(limit).execute()
        return res.data or []
    except Exception as e:
        logger.error("get_weights_history failed: %s", e)
        return []


# ── SPY context ───────────────────────────────────────────────────────────────
def save_spy_context(ctx: Dict[str, Any]) -> bool:
    try:
        _sb().table("spy_context").upsert({
            "id":            1,
            "date":          ctx.get("date"),
            "spy_price":     ctx.get("spy_price"),
            "spy_20dma":     ctx.get("spy_20dma"),
            "spy_vol_20d":   ctx.get("spy_vol_20d"),
            "spy_vol_80pct": ctx.get("spy_vol_80pct"),
            "bearish_flag":  ctx.get("bearish_flag", 0),
            "high_vol_flag": ctx.get("high_vol_flag", 0),
        }).execute()
        return True
    except Exception as e:
        logger.error("save_spy_context failed: %s", e)
        return False


def get_spy_context() -> Dict[str, Any]:
    try:
        res = _sb().table("spy_context").select("*").eq("id", 1).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        logger.error("get_spy_context failed: %s", e)
    return {"bearish_flag": 0, "high_vol_flag": 0, "date": None}


# ── Metrics ───────────────────────────────────────────────────────────────────
def save_metrics(m: Dict[str, Any]) -> bool:
    try:
        _sb().table("metrics").insert({
            "date":           m["date"],
            "strategy":       m["strategy"],
            "win_rate":       m.get("win_rate"),
            "avg_return":     m.get("avg_return"),
            "max_drawdown":   m.get("max_drawdown"),
            "trades_sampled": m.get("trades_sampled"),
            "source":         m.get("source", "live"),
        }).execute()
        return True
    except Exception as e:
        logger.error("save_metrics failed: %s", e)
        return False


def get_latest_metrics(strategy: str, source: str = "live") -> Dict[str, Any]:
    try:
        res = (
            _sb().table("metrics")
            .select("*")
            .eq("strategy", strategy)
            .eq("source", source)
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else {}
    except Exception as e:
        logger.error("get_latest_metrics failed: %s", e)
        return {}


# ── Watchlist ─────────────────────────────────────────────────────────────────
def add_to_watchlist(ticker: str, notes: str = "") -> bool:
    try:
        _sb().table("watchlist").upsert({
            "ticker":        ticker.upper(),
            "date_added":    date.today().isoformat(),
            "active_status": 1,
            "notes":         notes,
        }).execute()
        return True
    except Exception as e:
        logger.error("add_to_watchlist failed for %s: %s", ticker, e)
        return False


def remove_from_watchlist(ticker: str) -> bool:
    try:
        _sb().table("watchlist").update(
            {"active_status": 0}
        ).eq("ticker", ticker.upper()).execute()
        return True
    except Exception as e:
        logger.error("remove_from_watchlist failed for %s: %s", ticker, e)
        return False


def get_watchlist(active_only: bool = True) -> List[Dict[str, Any]]:
    try:
        q = _sb().table("watchlist").select("*")
        if active_only:
            q = q.eq("active_status", 1)
        res = q.order("date_added", desc=True).execute()
        return res.data or []
    except Exception as e:
        logger.error("get_watchlist failed: %s", e)
        return []


# ── Health check ──────────────────────────────────────────────────────────────
def db_health_check() -> Dict[str, Any]:
    result = {
        "ok": False, "path": "Supabase",
        "trade_count": 0, "active_trades": 0, "error": None,
    }
    try:
        r1 = _sb().table("trades").select("id", count="exact").execute()
        r2 = (
            _sb().table("trades")
            .select("id", count="exact")
            .in_("status", ["Pending", "Executed"])
            .execute()
        )
        result.update({
            "ok":            True,
            "trade_count":   r1.count or 0,
            "active_trades": r2.count or 0,
        })
    except Exception as e:
        result["error"] = str(e)
        logger.error("db_health_check failed: %s", e)
    return result


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("database.py (Supabase client) — self-test")
    print("Needs SUPABASE_URL + SUPABASE_KEY in .env")
    print("=" * 50)

    print(f"init_db():         {'✅' if init_db() else '❌'}")

    tid = insert_trade({
        "date": date.today().isoformat(), "ticker": "TEST",
        "strategy": "filter_a", "score": 72.5,
        "entry": 100.0, "stop": 98.0, "target": 104.0,
        "position_size": 50, "momentum": 0.03,
        "volume_spike": 1.8, "volatility": 0.012,
        "atr": 2.0, "sector": "Technology",
        "explanation": "Supabase client self-test",
    })
    print(f"insert_trade():    {'✅' if tid else '❌'}  id={tid}")
    print(f"update_status():   {'✅' if update_trade_status(tid, 'Success', 104.5) else '❌'}")
    print(f"get_weights():     {get_weights('filter_a')}")
    print(f"spy_context():     {get_spy_context()}")
    print(f"health_check():    {db_health_check()}")

    # Cleanup
    _sb().table("trades").delete().eq("ticker", "TEST").execute()
    print("Cleaned up. Self-test complete.")
