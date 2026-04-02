"""
learning.py — Adaptive weight system.

Responsibilities:
- Bucket closed trades by momentum/volume features (2×2 = 4 buckets per strategy)
- Compute exponentially-decayed win rates per bucket
- Adjust feature weights using explicit formula: w_new = w × (1 + α × signal)
- Clamp weights to [WEIGHT_MIN, WEIGHT_MAX] and normalize to sum=1.0
- Persist updated weights + full audit trail to DB
- Provide bucket stats for UI (Page 5)

Design:
  4 buckets (2×2) per strategy to avoid sparse-bucket problem.
  Minimum MIN_TRADES_BUCKET trades required before any bucket fires.
  Exponential decay: older trades count less (0.95^days_old).
  Weight changes are bounded and reversible via manual reset.

Import chain: config -> database -> learning
"""

import logging
import traceback
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
import database as db

logger     = logging.getLogger("learning")
app_logger = logging.getLogger("app")


# ─────────────────────────────────────────────────────────────────────────────
# BUCKET ASSIGNMENT
# 2×2 bucketing: momentum (low/high) × volume_spike (low/high)
# Coarse granularity ensures enough trades per bucket even with small samples
# ─────────────────────────────────────────────────────────────────────────────

def assign_bucket(momentum: float, volume_spike: float) -> str:
    """
    Assign a trade to one of 4 buckets based on momentum and volume features.

    Thresholds from config:
      MOMENTUM_BUCKET_THRESHOLD  = 0.025  (2.5% 5-day return)
      VOLUME_BUCKET_THRESHOLD    = 1.5    (1.5× average volume)

    Returns bucket key: 'low_mom_low_vol' | 'low_mom_high_vol' |
                        'high_mom_low_vol' | 'high_mom_high_vol'
    """
    mom_label = "high_mom" if momentum  >= config.MOMENTUM_BUCKET_THRESHOLD else "low_mom"
    vol_label = "high_vol" if volume_spike >= config.VOLUME_BUCKET_THRESHOLD  else "low_vol"
    return f"{mom_label}_{vol_label}"


def get_all_bucket_keys() -> List[str]:
    """Returns all possible bucket key strings (for initializing dicts)."""
    return [
        "low_mom_low_vol",
        "low_mom_high_vol",
        "high_mom_low_vol",
        "high_mom_high_vol",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# BUCKET WIN RATE CALCULATION WITH EXPONENTIAL DECAY
# ─────────────────────────────────────────────────────────────────────────────

def compute_bucket_win_rates(
    trades:   List[Dict[str, Any]],
    strategy: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute exponentially-decayed win rates for each bucket from closed trades.

    For each trade:
      - Outcome: 1 if Success, 0 if Failed/Expired
      - Weight:  DECAY_FACTOR ^ days_since_close
        (recent trades count more than older ones)

    Weighted win rate = Σ(weight_i × outcome_i) / Σ(weight_i)

    Returns dict keyed by bucket_key:
      {
        'win_rate':    float (0–1, weighted),
        'trade_count': int   (raw count, not weighted),
        'total_weight': float,
        'ready':       bool  (True if trade_count >= MIN_TRADES_BUCKET)
      }
    """
    # Initialize all buckets
    buckets: Dict[str, Dict] = {
        key: {"wins_weighted": 0.0, "total_weight": 0.0, "trade_count": 0}
        for key in get_all_bucket_keys()
    }

    today = date.today()

    for t in trades:
        if t.get("strategy") != strategy:
            continue

        # Skip trades without outcome (open trades should never be here,
        # but guard anyway)
        outcome_pct = t.get("outcome_pct")
        status      = t.get("status", "")
        if status not in ("Success", "Failed", "Expired"):
            continue

        # Outcome: 1 = win (Success), 0 = loss (Failed or Expired)
        outcome = 1.0 if status == "Success" else 0.0

        # Exponential decay weight based on days since close
        exit_date_str = t.get("exit_date")
        if exit_date_str:
            try:
                exit_dt   = date.fromisoformat(exit_date_str)
                days_old  = max(0, (today - exit_dt).days)
            except (ValueError, TypeError):
                days_old = 0
        else:
            days_old = 0

        decay_weight = config.DECAY_FACTOR ** days_old

        # Feature values — use stored features from trade record
        momentum     = t.get("momentum",     0.0) or 0.0
        volume_spike = t.get("volume_spike", 1.0) or 1.0

        bucket_key = assign_bucket(momentum, volume_spike)

        buckets[bucket_key]["wins_weighted"]  += outcome * decay_weight
        buckets[bucket_key]["total_weight"]   += decay_weight
        buckets[bucket_key]["trade_count"]    += 1

    # Compute win rates
    result = {}
    for key, data in buckets.items():
        tw = data["total_weight"]
        tc = data["trade_count"]
        win_rate = (data["wins_weighted"] / tw) if tw > 0 else 0.5
        result[key] = {
            "win_rate":    round(win_rate, 4),
            "trade_count": tc,
            "total_weight": round(tw, 4),
            "ready":       tc >= config.MIN_TRADES_BUCKET,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT ADJUSTMENT FORMULA
# Explicit formula — no vagueness
# ─────────────────────────────────────────────────────────────────────────────

def adjust_weights(
    current_weights: Dict[str, float],
    bucket_stats:    Dict[str, Dict[str, Any]],
    strategy:        str,
) -> Tuple[Dict[str, float], bool]:
    """
    Adjust feature weights based on bucket win rates.

    Formula (per feature weight w_f):
      signal   = weighted_avg_win_rate(buckets relevant to feature) - 0.50
                 (positive = feature performing above chance, negative = below)

      w_f_new  = w_f × (1 + LEARNING_ALPHA × signal)
      w_f_new  = clip(w_f_new, WEIGHT_MIN, WEIGHT_MAX)

    After all weights updated:
      normalize so sum(weights) = 1.0
      re-clamp once after normalization

    Only fires if at least one bucket has reached MIN_TRADES_BUCKET.

    Returns (new_weights, was_updated).
    was_updated=False means no bucket had enough data — weights unchanged.
    """
    # Check if ANY bucket has enough trades to fire
    any_ready = any(v["ready"] for v in bucket_stats.values())
    if not any_ready:
        app_logger.info(
            "Learning: no bucket has %d trades yet for %s — skipping weight update",
            config.MIN_TRADES_BUCKET, strategy,
        )
        return current_weights.copy(), False

    # ── Compute per-feature signals ───────────────────────────────────────────
    # Each feature maps to buckets where it is the differentiating variable.
    # Momentum signal: compare high_mom buckets vs low_mom buckets
    # Volume signal:   compare high_vol buckets vs low_vol buckets
    # Volatility:      no direct bucket mapping → use overall win rate signal

    def avg_win_rate(keys: List[str]) -> float:
        """Weighted average win rate across specified buckets (only ready buckets)."""
        ready_stats = [bucket_stats[k] for k in keys if bucket_stats[k]["ready"]]
        if not ready_stats:
            return 0.5  # Neutral if no ready buckets
        total_w = sum(s["total_weight"] for s in ready_stats)
        if total_w == 0:
            return 0.5
        return sum(s["win_rate"] * s["total_weight"] for s in ready_stats) / total_w

    # Momentum: high_mom buckets
    mom_rate   = avg_win_rate(["high_mom_low_vol", "high_mom_high_vol"])
    # Volume: high_vol buckets
    vol_rate   = avg_win_rate(["low_mom_high_vol", "high_mom_high_vol"])
    # Volatility: overall (no dedicated bucket — use all ready buckets)
    all_ready_keys = [k for k, v in bucket_stats.items() if v["ready"]]
    vlt_rate   = avg_win_rate(all_ready_keys) if all_ready_keys else 0.5

    signals = {
        "momentum":   mom_rate - 0.50,
        "volume":     vol_rate - 0.50,
        "volatility": vlt_rate - 0.50,
    }

    app_logger.info(
        "Learning signals for %s: momentum=%.3f volume=%.3f volatility=%.3f",
        strategy, signals["momentum"], signals["volume"], signals["volatility"],
    )

    # ── Apply adjustment formula ──────────────────────────────────────────────
    new_weights: Dict[str, float] = {}
    for feature, w_old in current_weights.items():
        signal = signals.get(feature, 0.0)
        w_new  = w_old * (1 + config.LEARNING_ALPHA * signal)
        # Clamp to [WEIGHT_MIN, WEIGHT_MAX]
        # Exception: if original weight was 0.0 (disabled feature like
        # volatility in filter_a), keep it at 0.0 — don't activate it
        if w_old == 0.0:
            w_new = 0.0
        else:
            w_new = max(config.WEIGHT_MIN, min(config.WEIGHT_MAX, w_new))
        new_weights[feature] = w_new

    # ── Normalize so sum = 1.0 ────────────────────────────────────────────────
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {f: w / total for f, w in new_weights.items()}
    else:
        # Degenerate case — fall back to base weights
        app_logger.warning("Learning: normalization total=0, reverting to base weights")
        return config.BASE_WEIGHTS[strategy].copy(), False

    # ── Re-clamp after normalization (normalization can push values out of range)
    # Only clamp non-zero weights
    for feature, w in new_weights.items():
        if w > 0:
            new_weights[feature] = max(config.WEIGHT_MIN, min(config.WEIGHT_MAX, w))

    # ── Final renormalize after re-clamp ─────────────────────────────────────
    total2 = sum(new_weights.values())
    if total2 > 0:
        new_weights = {f: round(w / total2, 6) for f, w in new_weights.items()}

    return new_weights, True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LEARNING TRIGGER
# Called after every trade closes
# ─────────────────────────────────────────────────────────────────────────────

def run_learning_update(strategy: str) -> Dict[str, Any]:
    """
    Full learning cycle for one strategy after a trade closes.

    Steps:
      1. Load last ROLLING_LOOKBACK closed trades for this strategy
      2. Compute bucket win rates with decay
      3. Adjust weights if enough data
      4. Persist to DB (weights table + weights_history audit trail)

    Returns summary dict for logging/UI.
    """
    result = {
        "strategy":     strategy,
        "updated":      False,
        "trades_used":  0,
        "bucket_stats": {},
        "old_weights":  {},
        "new_weights":  {},
        "message":      "",
    }

    try:
        # Step 1: Load closed trades
        trades = db.get_closed_trades(strategy=strategy, limit=config.ROLLING_LOOKBACK)
        result["trades_used"] = len(trades)

        if len(trades) == 0:
            result["message"] = f"No closed trades for {strategy} — nothing to learn from"
            return result

        # Step 2: Compute bucket stats
        bucket_stats = compute_bucket_win_rates(trades, strategy)
        result["bucket_stats"] = bucket_stats

        # Step 3: Adjust weights
        current_weights = db.get_weights(strategy)
        result["old_weights"] = current_weights.copy()

        new_weights, was_updated = adjust_weights(current_weights, bucket_stats, strategy)
        result["new_weights"] = new_weights

        if not was_updated:
            result["message"] = (
                f"Buckets not yet ready for {strategy} — "
                f"need {config.MIN_TRADES_BUCKET} trades per bucket"
            )
            return result

        # Step 4: Persist
        ok = db.update_weights(
            strategy,
            new_weights,
            trigger_event="learning_update",
            trades_count=len(trades),
        )
        result["updated"] = ok
        result["message"] = (
            f"Weights updated for {strategy}. "
            f"Trades used: {len(trades)}. "
            f"Changes: " + ", ".join(
                f"{f}: {result['old_weights'].get(f,0):.3f}→{w:.3f}"
                for f, w in new_weights.items()
            )
        )
        app_logger.info("Learning update: %s", result["message"])

    except Exception as e:
        result["message"] = f"Learning update failed for {strategy}: {e}"
        logger.error("%s\n%s", result["message"], traceback.format_exc())

    return result


def run_all_learning_updates() -> List[Dict[str, Any]]:
    """
    Run learning update for all strategies.
    Called after any trade closes.
    Returns list of result dicts (one per strategy).
    """
    return [run_learning_update(s) for s in config.BASE_WEIGHTS.keys()]


# ─────────────────────────────────────────────────────────────────────────────
# BUCKET STATS FOR UI (Page 5 heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def get_bucket_stats_for_display() -> Dict[str, Dict[str, Dict]]:
    """
    Returns bucket stats for all strategies — formatted for Page 5 heatmap.
    Returns {strategy: {bucket_key: stats_dict}}.
    """
    result = {}
    for strategy in config.BASE_WEIGHTS.keys():
        trades = db.get_closed_trades(strategy=strategy, limit=config.ROLLING_LOOKBACK)
        result[strategy] = compute_bucket_win_rates(trades, strategy)
    return result


def get_learning_summary() -> Dict[str, Any]:
    """
    Returns high-level learning system status for Page 5 display.
    Includes current weights, recent history, and bucket readiness.
    """
    summary: Dict[str, Any] = {
        "weights":         {},
        "weights_history": {},
        "bucket_stats":    {},
        "total_closed":    0,
    }
    try:
        closed = db.get_closed_trades(limit=500)
        summary["total_closed"] = len(closed)

        for strategy in config.BASE_WEIGHTS.keys():
            summary["weights"][strategy]         = db.get_weights(strategy)
            summary["weights_history"][strategy] = db.get_weights_history(strategy, limit=20)
            strat_trades = [t for t in closed if t.get("strategy") == strategy]
            summary["bucket_stats"][strategy]    = compute_bucket_win_rates(strat_trades, strategy)

    except Exception as e:
        logger.error("get_learning_summary failed: %s", e)

    return summary


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import database as db
    db.init_db()

    print("learning.py self-test")
    print("=" * 55)

    # [1] Bucket assignment
    print("\n[1] Bucket assignment:")
    cases = [
        (0.03,  2.0,  "high_mom_high_vol"),
        (0.01,  2.0,  "low_mom_high_vol"),
        (0.03,  1.0,  "high_mom_low_vol"),
        (0.01,  1.0,  "low_mom_low_vol"),
        (0.025, 1.5,  "high_mom_high_vol"),  # boundary: >= threshold
        (0.024, 1.49, "low_mom_low_vol"),    # boundary: just below
    ]
    for mom, vol, expected in cases:
        result = assign_bucket(mom, vol)
        status = "✅" if result == expected else f"❌ got {result}"
        print(f"  mom={mom:.3f} vol={vol:.2f} → {result}  {status}")

    # [2] Compute bucket win rates with mock trades
    print("\n[2] Bucket win rates (mock trades):")
    mock_trades = []
    today_str = date.today().isoformat()

    # 6 wins in high_mom_high_vol, 2 losses → win rate ~0.75
    for i in range(6):
        mock_trades.append({
            "strategy": "filter_a", "status": "Success",
            "momentum": 0.04, "volume_spike": 2.0,
            "outcome_pct": 0.02, "exit_date": today_str,
        })
    for i in range(2):
        mock_trades.append({
            "strategy": "filter_a", "status": "Failed",
            "momentum": 0.04, "volume_spike": 2.0,
            "outcome_pct": -0.01, "exit_date": today_str,
        })
    # 3 wins in low_mom_low_vol, 3 losses → win rate ~0.50
    for i in range(3):
        mock_trades.append({
            "strategy": "filter_a", "status": "Success",
            "momentum": 0.01, "volume_spike": 1.0,
            "outcome_pct": 0.015, "exit_date": today_str,
        })
    for i in range(3):
        mock_trades.append({
            "strategy": "filter_a", "status": "Failed",
            "momentum": 0.01, "volume_spike": 1.0,
            "outcome_pct": -0.01, "exit_date": today_str,
        })

    stats = compute_bucket_win_rates(mock_trades, "filter_a")
    for bucket, s in stats.items():
        print(
            f"  {bucket:25s}: win_rate={s['win_rate']:.2f}  "
            f"count={s['trade_count']}  ready={s['ready']}"
        )
    assert abs(stats["high_mom_high_vol"]["win_rate"] - 0.75) < 0.01, "Expected ~0.75"
    assert abs(stats["low_mom_low_vol"]["win_rate"]   - 0.50) < 0.01, "Expected ~0.50"
    print("  ✅ Win rate assertions passed")

    # [3] Weight adjustment
    print("\n[3] Weight adjustment:")
    base_w  = config.BASE_WEIGHTS["filter_a"].copy()   # {momentum:0.5, volume:0.5, vol:0.0}
    new_w, updated = adjust_weights(base_w, stats, "filter_a")
    print(f"  Base weights:    {base_w}")
    print(f"  Updated weights: {new_w}")
    print(f"  Was updated: {updated}")

    weight_sum = sum(new_w.values())
    assert abs(weight_sum - 1.0) < 1e-4, f"Weights must sum to 1.0, got {weight_sum}"
    for f, w in new_w.items():
        if base_w.get(f, 0) > 0:  # only check non-zero weights
            assert config.WEIGHT_MIN <= w <= config.WEIGHT_MAX, \
                f"Weight {f}={w} out of bounds [{config.WEIGHT_MIN},{config.WEIGHT_MAX}]"
    print(f"  ✅ Weights sum={weight_sum:.6f}, all in bounds")

    # [4] Decay test
    print("\n[4] Exponential decay:")
    old_trade = {
        "strategy": "filter_a", "status": "Success",
        "momentum": 0.04, "volume_spike": 2.0,
        "outcome_pct": 0.02, "exit_date": "2020-01-01",  # very old
    }
    new_trade = {
        "strategy": "filter_a", "status": "Success",
        "momentum": 0.04, "volume_spike": 2.0,
        "outcome_pct": 0.02, "exit_date": today_str,
    }
    old_stats = compute_bucket_win_rates([old_trade], "filter_a")
    new_stats = compute_bucket_win_rates([new_trade], "filter_a")
    old_weight = old_stats["high_mom_high_vol"]["total_weight"]
    new_weight = new_stats["high_mom_high_vol"]["total_weight"]
    print(f"  Old trade (2020) weight: {old_weight:.8f}")
    print(f"  Today's trade weight:    {new_weight:.4f}")
    assert new_weight > old_weight * 1000, "Recent trades should weigh far more than old ones"
    print("  ✅ Decay working correctly (recent >> old)")

    # [5] Run full learning update (inserts mock closed trades into DB)
    print("\n[5] Full learning update cycle:")
    # Insert some closed trades into DB
    for i, t in enumerate(mock_trades[:10]):
        db.insert_trade({
            "date": today_str, "ticker": f"T{i:03d}",
            "strategy": t["strategy"], "score": 60.0,
            "entry": 100.0, "stop": 98.0, "target": 104.0,
            "position_size": 50,
            "momentum": t["momentum"], "volume_spike": t["volume_spike"],
            "volatility": 0.01, "atr": 2.0, "sector": "Technology",
            "explanation": "test",
        })
    # Manually mark some as closed
    all_trades = db.get_all_trades()
    for t in all_trades[:5]:
        db.update_trade_status(t["id"], "Success", exit_price=104.0)
    for t in all_trades[5:8]:
        db.update_trade_status(t["id"], "Failed", exit_price=97.5)

    result = run_learning_update("filter_a")
    print(f"  Result: {result['message']}")
    print(f"  Updated: {result['updated']}")
    print(f"  Trades used: {result['trades_used']}")
    if result["new_weights"]:
        print(f"  New weights: {result['new_weights']}")

    # Cleanup test trades
    import sqlite3, threading
    lock = threading.Lock()
    with lock:
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("DELETE FROM trades WHERE ticker LIKE 'T%'")
        conn.commit()
        conn.close()
    print("  Test data cleaned up.")

    print("\nlearning.py self-test complete.")
