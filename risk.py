"""
risk.py — Trade risk calculation and position sizing.

Responsibilities:
- Compute stop loss, target, position size for each trade candidate
- Enforce 2:1 reward/risk ratio via ATR multipliers (guaranteed by config)
- Liquidity check: reject if position too large vs average daily volume
- Portfolio risk check: reject if adding trade would breach max total risk
- Attach all risk levels to trade candidates for DB insertion

Import chain: config -> database -> universe -> data_fetch -> scoring -> risk
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
import database as db

logger     = logging.getLogger("risk")
app_logger = logging.getLogger("app")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRADE RISK CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_trade_risk(
    entry: float,
    atr:   float,
) -> Tuple[float, float, float, float]:
    """
    Compute stop, target, dollar_risk, and R:R ratio for a trade.

    stop         = entry - ATR_STOP_MULT × atr
    target       = entry + ATR_TARGET_MULT × atr
    dollar_risk  = CAPITAL × RISK_PER_TRADE  (fixed 1% of portfolio)
    rr_ratio     = (target - entry) / (entry - stop)  →  always >= 2.0

    ATR floor is already applied in scoring.py — atr will never be 0.
    Returns (stop, target, dollar_risk, rr_ratio).
    """
    # ATR floor guard — defensive, should already be applied in compute_features
    atr_safe = max(atr, entry * config.ATR_MIN_FLOOR_PCT)

    stop        = entry - config.ATR_STOP_MULT   * atr_safe
    target      = entry + config.ATR_TARGET_MULT * atr_safe
    dollar_risk = config.CAPITAL * config.RISK_PER_TRADE
    risk_pts    = entry - stop

    # R:R ratio — with default config (stop=1×, target=2×) this is always 2.0
    rr_ratio = (target - entry) / risk_pts if risk_pts > 0 else 0.0

    return (
        round(stop,        4),
        round(target,      4),
        round(dollar_risk, 2),
        round(rr_ratio,    2),
    )


def calculate_position_size(
    entry:       float,
    stop:        float,
    dollar_risk: float,
) -> int:
    """
    Compute whole-share position size.

    position_size = floor(dollar_risk / (entry - stop))

    Whole shares only — no fractional shares (Robinhood supports fractional,
    but whole shares keeps risk math simple and explainable).
    Returns 0 if entry == stop (should never happen due to ATR floor).
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        logger.error(
            "calculate_position_size: risk_per_share=%.4f <= 0 "
            "(entry=%.4f stop=%.4f) — returning 0 shares",
            risk_per_share, entry, stop,
        )
        return 0
    return math.floor(dollar_risk / risk_per_share)


# ─────────────────────────────────────────────────────────────────────────────
# LIQUIDITY CHECK
# Reject if our position would be too large relative to average daily volume
# ─────────────────────────────────────────────────────────────────────────────

def check_liquidity(
    ticker:        str,
    position_size: int,
    entry:         float,
    avg_vol_10d:   float,
) -> Tuple[bool, str]:
    """
    Reject trade if position dollar value > MAX_POSITION_VS_VOL × avg daily $ volume.

    Rationale: entering or exiting a position larger than 1% of average daily
    volume risks significant market impact / slippage. This protects against
    illiquid trades that look good on paper but can't be executed cleanly.

    avg_vol_10d: average daily SHARE volume over past 10 days.
    Returns (passes: bool, reason: str).
    """
    if avg_vol_10d <= 0:
        return True, ""  # Can't check — allow (fail open)

    position_value     = position_size * entry
    avg_dollar_volume  = avg_vol_10d * entry
    max_allowed        = config.MAX_POSITION_VS_VOL * avg_dollar_volume

    if position_value > max_allowed:
        reason = (
            f"Position ${position_value:,.0f} exceeds {config.MAX_POSITION_VS_VOL*100:.0f}% "
            f"of avg daily $ volume (${avg_dollar_volume:,.0f}). "
            f"Max allowed: ${max_allowed:,.0f}"
        )
        return False, reason

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK CHECK
# Reject if adding this trade would breach total portfolio risk cap
# ─────────────────────────────────────────────────────────────────────────────

def check_portfolio_risk(
    new_dollar_risk: float,
    active_trades:   Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str]:
    """
    Reject trade if (existing open risk + new trade risk) > MAX_TOTAL_RISK × CAPITAL.

    Existing risk calculated as:
      sum of (entry - stop) × position_size for all active trades

    new_dollar_risk: dollar amount at risk on the new trade.
    Returns (passes: bool, reason: str).
    """
    if active_trades is None:
        active_trades = db.get_active_trades()

    # Sum dollar risk of all current active positions
    existing_risk = 0.0
    for t in active_trades:
        entry = t.get("entry", 0) or 0
        stop  = t.get("stop",  0) or 0
        size  = t.get("position_size", 0) or 0
        trade_risk = (entry - stop) * size
        if trade_risk > 0:
            existing_risk += trade_risk

    total_risk     = existing_risk + new_dollar_risk
    max_risk       = config.CAPITAL * config.MAX_TOTAL_RISK

    if total_risk > max_risk:
        reason = (
            f"Adding this trade (${new_dollar_risk:,.0f} at risk) would bring "
            f"total portfolio risk to ${total_risk:,.0f}, "
            f"exceeding the {config.MAX_TOTAL_RISK*100:.0f}% cap "
            f"(${max_risk:,.0f})"
        )
        return False, reason

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# ATTACH RISK TO CANDIDATES (called after scoring pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def attach_risk_levels(
    candidates:   List[Dict[str, Any]],
    ohlcv:        Dict,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Compute and attach stop/target/position_size to each candidate.
    Runs liquidity and portfolio risk checks — rejects failing candidates.

    Returns (approved_candidates, rejection_reasons_list).
    Approved candidates are ready for DB insertion.
    """
    approved:   List[Dict[str, Any]] = []
    rejections: List[str]            = []
    active_trades = db.get_active_trades()

    for c in candidates:
        ticker = c["ticker"]
        entry  = c["entry"]
        atr    = c["atr"]

        # ── Core risk math ────────────────────────────────────────────────────
        stop, target, dollar_risk, rr_ratio = calculate_trade_risk(entry, atr)
        position_size = calculate_position_size(entry, stop, dollar_risk)

        if position_size <= 0:
            rejections.append(f"{ticker}: position size = 0 (ATR too large or entry == stop)")
            continue

        # ── Liquidity check ───────────────────────────────────────────────────
        avg_vol_10d = _compute_avg_volume(ticker, ohlcv)
        liq_ok, liq_reason = check_liquidity(ticker, position_size, entry, avg_vol_10d)
        if not liq_ok:
            rejections.append(f"{ticker}: {liq_reason}")
            app_logger.info("Liquidity reject: %s", liq_reason)
            continue

        # ── Portfolio risk check ──────────────────────────────────────────────
        risk_ok, risk_reason = check_portfolio_risk(dollar_risk, active_trades)
        if not risk_ok:
            rejections.append(f"{ticker}: {risk_reason}")
            app_logger.info("Portfolio risk reject: %s", risk_reason)
            continue

        # ── All checks passed — attach risk levels ────────────────────────────
        c = dict(c)
        c.update({
            "stop":          stop,
            "target":        target,
            "position_size": position_size,
            "dollar_risk":   dollar_risk,
            "rr_ratio":      rr_ratio,
            "avg_vol_10d":   avg_vol_10d,
        })

        # Update explanation with risk levels appended
        c["explanation"] = _append_risk_to_explanation(c)

        approved.append(c)
        app_logger.info(
            "Risk approved: %s entry=%.2f stop=%.2f target=%.2f "
            "size=%d risk=$%.0f rr=%.1f",
            ticker, entry, stop, target, position_size, dollar_risk, rr_ratio,
        )

    return approved, rejections


def _compute_avg_volume(ticker: str, ohlcv: Dict) -> float:
    """
    Compute 10-day average volume from OHLCV data for liquidity check.
    Returns 0 if data unavailable (liquidity check will pass — fail open).
    """
    import pandas as pd
    df = ohlcv.get(ticker)
    if df is None or df.empty:
        return 0.0
    try:
        vol_series = df["Volume"].dropna()
        if len(vol_series) < config.VOL_AVG_WINDOW + 2:
            return float(vol_series.mean()) if len(vol_series) > 0 else 0.0
        # Same window as scoring: t-11 to t-1 (excludes current bar)
        return float(vol_series.iloc[-(config.VOL_AVG_WINDOW + 1):-1].mean())
    except Exception as e:
        logger.error("_compute_avg_volume failed for %s: %s", ticker, e)
        return 0.0


def _append_risk_to_explanation(c: Dict[str, Any]) -> str:
    """Append risk/sizing details to the existing explanation string."""
    existing = c.get("explanation", "")
    risk_block = (
        f"\n  --- Risk Levels ---\n"
        f"  Entry:         ${c['entry']:.2f}\n"
        f"  Stop Loss:     ${c['stop']:.2f}  "
        f"(-{((c['entry']-c['stop'])/c['entry']*100):.1f}%)\n"
        f"  Target:        ${c['target']:.2f}  "
        f"(+{((c['target']-c['entry'])/c['entry']*100):.1f}%)\n"
        f"  R:R Ratio:     {c['rr_ratio']:.1f}:1\n"
        f"  Position Size: {c['position_size']} shares\n"
        f"  $ at Risk:     ${c['dollar_risk']:,.0f}  "
        f"({config.RISK_PER_TRADE*100:.1f}% of ${config.CAPITAL:,.0f} capital)"
    )
    return existing + risk_block


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SUMMARY (for UI display)
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio_risk_summary() -> Dict[str, Any]:
    """
    Returns current portfolio risk status for UI display on Page 1 header.
    Never raises — returns safe defaults on error.
    """
    try:
        active = db.get_active_trades()
        total_risk   = 0.0
        trade_risks  = []

        for t in active:
            entry = t.get("entry", 0) or 0
            stop  = t.get("stop",  0) or 0
            size  = t.get("position_size", 0) or 0
            risk  = max(0, (entry - stop) * size)
            total_risk += risk
            trade_risks.append({
                "ticker": t["ticker"],
                "risk":   round(risk, 2),
                "status": t["status"],
            })

        max_risk       = config.CAPITAL * config.MAX_TOTAL_RISK
        pct_used       = (total_risk / max_risk * 100) if max_risk > 0 else 0

        return {
            "active_count":  len(active),
            "slots_free":    config.MAX_ACTIVE_TRADES - len(active),
            "total_risk_$":  round(total_risk, 2),
            "max_risk_$":    round(max_risk, 2),
            "pct_used":      round(pct_used, 1),
            "trade_risks":   trade_risks,
            "at_capacity":   len(active) >= config.MAX_ACTIVE_TRADES,
        }
    except Exception as e:
        logger.error("get_portfolio_risk_summary failed: %s", e)
        return {
            "active_count": 0, "slots_free": config.MAX_ACTIVE_TRADES,
            "total_risk_$": 0, "max_risk_$": config.CAPITAL * config.MAX_TOTAL_RISK,
            "pct_used": 0, "trade_risks": [], "at_capacity": False,
        }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import database as db
    db.init_db()

    print("risk.py self-test")
    print("=" * 55)

    # [1] Core risk math
    print("\n[1] calculate_trade_risk:")
    entry, atr = 100.0, 2.0
    stop, target, dollar_risk, rr = calculate_trade_risk(entry, atr)
    print(f"  entry={entry}  atr={atr}")
    print(f"  stop={stop}   target={target}   dollar_risk={dollar_risk}   rr={rr}")
    assert stop   == 98.0,    f"Expected 98.0, got {stop}"
    assert target == 104.0,   f"Expected 104.0, got {target}"
    assert rr     == 2.0,     f"Expected 2.0, got {rr}"
    assert dollar_risk == config.CAPITAL * config.RISK_PER_TRADE
    print("  ✅ All assertions passed")

    # [2] ATR floor
    print("\n[2] ATR floor (very low ATR):")
    stop2, target2, _, rr2 = calculate_trade_risk(100.0, 0.0001)
    expected_atr = 100.0 * config.ATR_MIN_FLOOR_PCT
    print(f"  ATR floor applied: {expected_atr}  rr={rr2}")
    assert rr2 == 2.0, "R:R must be 2.0 even with floored ATR"
    print("  ✅ Floor applied correctly")

    # [3] Position sizing
    print("\n[3] calculate_position_size:")
    size = calculate_position_size(100.0, 98.0, 1000.0)
    print(f"  dollar_risk=1000  risk_per_share=2.0  -> size={size}  (expect 500)")
    assert size == 500, f"Expected 500, got {size}"
    size_zero = calculate_position_size(100.0, 100.0, 1000.0)
    print(f"  entry==stop -> size={size_zero}  (expect 0)")
    assert size_zero == 0
    print("  ✅ Sizing correct")

    # [4] Liquidity check
    print("\n[4] Liquidity check:")
    ok, reason = check_liquidity("AAPL", 500, 100.0, 5_000_000)
    print(f"  position=$50,000  avg_vol_$=$500M  passes={ok}  (expect True)")
    assert ok

    fail, reason = check_liquidity("TINY", 10_000, 100.0, 50_000)
    print(f"  position=$1M  avg_vol_$=$5M  passes={fail}  reason={reason[:60]}")
    assert not fail
    print("  ✅ Liquidity check correct")

    # [5] Portfolio risk check
    print("\n[5] Portfolio risk check:")
    ok, _ = check_portfolio_risk(500.0, active_trades=[])
    print(f"  new_risk=$500  no active trades  passes={ok}  (expect True)")
    assert ok

    # Mock 4 active trades each risking $1,200 = $4,800 > $5,000 cap when adding $500
    mock_active = [
        {"entry": 100.0, "stop": 98.0, "position_size": 600}
    ] * 4  # 4 × $1,200 = $4,800
    fail2, reason2 = check_portfolio_risk(500.0, active_trades=mock_active)
    print(f"  existing=$4,800 + new=$500 > cap=${config.CAPITAL*config.MAX_TOTAL_RISK:.0f}  passes={fail2}")
    assert not fail2
    print("  ✅ Portfolio risk check correct")

    # [6] Portfolio summary
    print("\n[6] Portfolio risk summary:")
    summary = get_portfolio_risk_summary()
    print(f"  active={summary['active_count']}  slots_free={summary['slots_free']}")
    print(f"  total_risk=${summary['total_risk_$']}  pct_used={summary['pct_used']}%")
    print(f"  at_capacity={summary['at_capacity']}")

    print("\nrisk.py self-test complete.")
