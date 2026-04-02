"""
config.py — Single source of truth for ALL tunables.
No logic lives here. Only constants and base configuration.
Import this module in all other modules. Never hardcode numbers elsewhere.
"""

# ── Portfolio & Risk ──────────────────────────────────────────────────────────
CAPITAL             = 100_000   # Total portfolio capital ($)
RISK_PER_TRADE      = 0.01      # 1% of capital risked per trade
MAX_TOTAL_RISK      = 0.05      # 5% max simultaneous open risk
MAX_ACTIVE_TRADES   = 3         # Hard cap on concurrent system trades

# ── ATR / Stop / Target ───────────────────────────────────────────────────────
ATR_STOP_MULT       = 1.0       # Stop  = Entry - N × ATR
ATR_TARGET_MULT     = 2.0       # Target = Entry + N × ATR  →  2:1 R:R guaranteed
ATR_MIN_FLOOR_PCT   = 0.01      # ATR floor = max(ATR, price × 1%)  →  prevents ÷0

# ── Feature Windows ───────────────────────────────────────────────────────────
MOMENTUM_WINDOW     = 5         # Days for momentum return calculation
ATR_WINDOW          = 14        # Days for ATR rolling average
VOL_AVG_WINDOW      = 10        # Days for volume baseline average
VOLATILITY_WINDOW   = 10        # Days for std-dev volatility calculation
SPY_MA_WINDOW       = 20        # Days for SPY moving average
SPY_VOL_WINDOW      = 20        # Days for SPY return std-dev
SPY_HIST_WINDOW     = 252       # Days for SPY 80th-pct volatility lookback

# ── Data Fetch ────────────────────────────────────────────────────────────────
FETCH_PERIOD        = "400d"    # yfinance download period (covers all windows + buffer)
FETCH_INTERVAL      = "1d"      # Daily bars only
DATA_DELAY_MINUTES  = 15        # Disclosed delay shown to user in UI

# ── Universe Filters ──────────────────────────────────────────────────────────
FILTER_A_MIN_MCAP   = 2e9       # $2B minimum market cap
FILTER_A_MIN_VOL    = 1_000_000 # 1M average daily share volume
FILTER_A_MIN_PRICE  = 5.0       # $5 minimum price (avoids micro-cap noise)
FILTER_B_MAX_PRICE  = 5.0       # Filter B: price strictly < $5
FILTER_B_VOL_MULT   = 2.0       # Filter B: current vol ≥ 2× 10-day avg
UNIVERSE_STALE_DAYS = 7         # Warn if universe CSV older than N days

# ── Selection Constraints ─────────────────────────────────────────────────────
MAX_SAME_SECTOR     = 1         # Max trades per broad GICS sector per session
MAX_RISKY_TRADES    = 1         # Max Filter B trades in active portfolio
CORR_THRESHOLD      = 0.75      # Skip ticker if 10-day return correlation > this
MAX_POSITION_VS_VOL = 0.01      # Max position value as fraction of 10-day avg $ volume
WASH_SALE_DAYS      = 30        # Skip ticker if sold at loss within N days

# ── Scoring Normalization Bounds (fixed — not min-max across universe) ─────────
# These are calibrated to typical equity behavior ranges.
# Fixed bounds ensure scores are comparable across days and universe sizes.
NORM_BOUNDS = {
    "momentum":     (-0.20, 0.20),   # 5-day return: -20% to +20%
    "volume_spike": (0.50,  5.00),   # volume ratio: 0.5× to 5×
    "volatility":   (0.005, 0.08),   # daily std-dev: 0.5% to 8%
}

# ── Learning System ───────────────────────────────────────────────────────────
LEARNING_ALPHA      = 0.15      # Weight adjustment step size per update
WEIGHT_MIN          = 0.05      # Floor for any single feature weight
WEIGHT_MAX          = 0.90      # Ceiling for any single feature weight
DECAY_FACTOR        = 0.95      # Exponential decay: trade weight = 0.95^days_old
MIN_TRADES_BUCKET   = 5         # Min trades in bucket before weight adjustment fires
ROLLING_LOOKBACK    = 30        # Number of recent closed trades used for learning

# ── Bucket thresholds (4 buckets per strategy: momentum×volume, 2×2) ──────────
# Reduces sparse-bucket problem vs. finer granularity
MOMENTUM_BUCKET_THRESHOLD  = 0.025  # < threshold → 'low', >= → 'high'
VOLUME_BUCKET_THRESHOLD    = 1.5    # < threshold → 'low', >= → 'high'

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_DAYS       = 30        # Calendar days of history to simulate

# ── Trade Lifecycle ───────────────────────────────────────────────────────────
SIGNAL_EXPIRY_DAYS  = 2         # Pending trades auto-expire after N trading days
TRADE_STATUSES      = ("Pending", "Executed", "Success", "Failed", "Expired")

# ── Base Weights (used for initialization and manual reset) ───────────────────
# Must sum to 1.0 per strategy. Volatility weight for filter_a is 0 by design
# (stable stocks — volatility is not a signal, just noise).
BASE_WEIGHTS = {
    "filter_a": {"momentum": 0.50, "volume": 0.50, "volatility": 0.00},
    "filter_b": {"momentum": 0.30, "volume": 0.55, "volatility": 0.15},
}

# ── File Paths ────────────────────────────────────────────────────────────────
import os
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
DB_PATH         = os.path.join(DATA_DIR, "trading.db")
UNIVERSE_CSV    = os.path.join(DATA_DIR, "universe.csv")
ERROR_LOG       = os.path.join(LOGS_DIR, "errors.log")
APP_LOG         = os.path.join(LOGS_DIR, "app.log")

# ── Ensure directories exist on import ────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ── NYSE Market Hours (ET) ────────────────────────────────────────────────────
MARKET_OPEN_HOUR    = 9
MARKET_OPEN_MINUTE  = 30
MARKET_CLOSE_HOUR   = 16
MARKET_CLOSE_MINUTE = 0
MARKET_TIMEZONE     = "America/New_York"

# ── SPY Bearish Threshold ─────────────────────────────────────────────────────
SPY_HIGH_VOL_FIXED  = 0.015     # 1.5% daily std-dev triggers high-vol flag
SPY_SCORE_REDUCTION = 0.80      # Multiply all scores by this when SPY is bearish

# ── Disclaimer ────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ DISCLAIMER: This tool is for educational and research purposes only. "
    "It does not constitute financial advice. Past signals and backtested results "
    "do not guarantee future performance. Always consult a licensed financial "
    "advisor before making investment decisions. Never risk money you cannot "
    "afford to lose."
)


# ── Validation (runs once on import to catch config errors early) ─────────────
def _validate():
    for strategy, weights in BASE_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, (
            f"BASE_WEIGHTS['{strategy}'] must sum to 1.0, got {total}"
        )
        for name, w in weights.items():
            assert WEIGHT_MIN <= w <= WEIGHT_MAX or w == 0.0, (
                f"BASE_WEIGHTS['{strategy}']['{name}']={w} outside "
                f"[{WEIGHT_MIN}, {WEIGHT_MAX}] (0.0 allowed as disabled)"
            )
    assert ATR_TARGET_MULT / ATR_STOP_MULT >= 2.0, (
        "ATR_TARGET_MULT / ATR_STOP_MULT must be >= 2.0 to guarantee 2:1 R:R"
    )
    assert RISK_PER_TRADE * MAX_ACTIVE_TRADES <= MAX_TOTAL_RISK + 1e-9, (
        "RISK_PER_TRADE × MAX_ACTIVE_TRADES exceeds MAX_TOTAL_RISK"
    )


_validate()


if __name__ == "__main__":
    print("✅ config.py loaded successfully")
    print(f"   Capital:          ${CAPITAL:,.0f}")
    print(f"   Risk per trade:   {RISK_PER_TRADE*100:.1f}%  (${CAPITAL*RISK_PER_TRADE:,.0f})")
    print(f"   Max active trades:{MAX_ACTIVE_TRADES}")
    print(f"   Max total risk:   {MAX_TOTAL_RISK*100:.1f}%")
    print(f"   ATR multipliers:  stop={ATR_STOP_MULT}×  target={ATR_TARGET_MULT}×  (R:R={ATR_TARGET_MULT/ATR_STOP_MULT:.1f}:1)")
    print(f"   DB path:          {DB_PATH}")
    print(f"   Universe CSV:     {UNIVERSE_CSV}")
    print("\n   Base weights:")
    for strat, w in BASE_WEIGHTS.items():
        print(f"     {strat}: {w}")
    print("\n   Validation: PASSED")
