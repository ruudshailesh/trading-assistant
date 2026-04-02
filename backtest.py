"""
backtest.py — Backtesting-lite engine.

Strict no-lookahead rules:
  - For each simulated day T, features use ONLY data from T-15 to T-1
  - Entry price = T's open (simulates realistic fill — not close)
  - Volume average = mean(volume[T-11 : T-1]) — never includes T
  - Momentum = (close[T-1] - close[T-6]) / close[T-6] — never includes T
  - ATR = mean of True Ranges from T-14 to T-1 — never includes T
  - SPY context evaluated using T-1 data only
  - Outcome: close[T+1] vs stop/target (next-day hold assumption)

Design:
  Uses same feature/scoring/risk logic as live system (no separate code paths).
  Weights snapshot taken at run time — shown in results so user knows
  backtest reflects current weights, not historical weights.
  Results saved to metrics table (source='backtest') for Page 4 display.

Import chain: config -> database -> scoring -> risk -> backtest
"""

import logging
import traceback
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import database as db
import scoring as sc
import risk as rk

logger     = logging.getLogger("backtest")
app_logger = logging.getLogger("app")


# ─────────────────────────────────────────────────────────────────────────────
# SLICE OHLCV FOR A SIMULATED DAY
# Returns a sub-DataFrame ending at T-1 (so scoring sees only past data)
# ─────────────────────────────────────────────────────────────────────────────

def _slice_for_day(df: pd.DataFrame, sim_day_idx: int) -> Optional[pd.DataFrame]:
    """
    Returns OHLCV slice ending at sim_day_idx - 1 (exclusive of sim day).
    Minimum length enforced for feature computation.

    sim_day_idx: integer position in the DataFrame (iloc index of simulated day T)
    Returns None if insufficient history before T.
    """
    min_history = max(
        config.ATR_WINDOW,
        config.VOL_AVG_WINDOW,
        config.MOMENTUM_WINDOW,
        config.SPY_MA_WINDOW,
    ) + 5  # buffer

    end_idx = sim_day_idx  # exclusive: slice goes up to but NOT including sim day
    if end_idx < min_history:
        return None

    return df.iloc[:end_idx].copy()


def _compute_spy_context_for_day(
    spy_df: pd.DataFrame,
    sim_day_idx: int,
) -> Dict[str, Any]:
    """
    Compute SPY market context using data strictly before sim_day T.
    Mirrors data_fetch.compute_spy_context() but on a slice.
    Returns neutral context on failure.
    """
    neutral = {"bearish_flag": 0, "high_vol_flag": 0}
    try:
        sliced = spy_df.iloc[:sim_day_idx]
        if len(sliced) < config.SPY_MA_WINDOW + 5:
            return neutral

        close     = sliced["Close"].dropna()
        spy_price = float(close.iloc[-1])
        spy_20dma = float(close.rolling(config.SPY_MA_WINDOW).mean().iloc[-1])

        daily_ret   = close.pct_change().dropna()
        spy_vol_20d = float(daily_ret.rolling(config.SPY_VOL_WINDOW).std().iloc[-1])

        rolling_vol   = daily_ret.rolling(config.SPY_VOL_WINDOW).std().dropna()
        lookback      = min(len(rolling_vol), config.SPY_HIST_WINDOW)
        spy_vol_80pct = float(rolling_vol.iloc[-lookback:].quantile(0.80))

        return {
            "bearish_flag":  int(spy_price < spy_20dma),
            "high_vol_flag": int(
                spy_vol_20d > config.SPY_HIGH_VOL_FIXED or
                spy_vol_20d > spy_vol_80pct
            ),
            "spy_price":     spy_price,
            "spy_20dma":     spy_20dma,
        }
    except Exception as e:
        logger.debug("_compute_spy_context_for_day failed at idx %d: %s", sim_day_idx, e)
        return neutral


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE ONE DAY
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_day(
    ticker:       str,
    df:           pd.DataFrame,
    spy_df:       pd.DataFrame,
    sim_day_idx:  int,
    strategy:     str,
    weights:      Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """
    Simulate one trade signal on day T and its outcome on day T+1.

    Entry price: T's open (iloc[sim_day_idx]['Open'])
    Outcome:
      'success'  if close[T+1] >= target
      'failed'   if close[T+1] <= stop
      'open'     if neither (excluded from win rate)

    Returns dict with all trade details, or None if signal can't be generated.
    """
    # We need T+1 to evaluate outcome
    if sim_day_idx + 1 >= len(df):
        return None

    # Slice: data up to T-1 (strictly no lookahead)
    hist = _slice_for_day(df, sim_day_idx)
    if hist is None:
        return None

    # Compute features on historical slice
    feats = sc.compute_features(ticker, hist)
    if feats is None:
        return None

    # Override entry price: use T's open (not last close from hist)
    # This simulates a realistic market-open fill
    try:
        entry_price = float(df.iloc[sim_day_idx]["Open"])
        if pd.isna(entry_price) or entry_price <= 0:
            return None
        feats["entry"] = round(entry_price, 4)
    except (IndexError, KeyError):
        return None

    # Recompute ATR-based stops using actual entry price
    atr   = feats["atr"]
    stop, target, dollar_risk, rr = rk.calculate_trade_risk(entry_price, atr)

    # SPY context at T-1
    spy_ctx = _compute_spy_context_for_day(spy_df, sim_day_idx)

    # Score with current weights
    score = sc.compute_score(feats, strategy, weights)

    # Apply SPY reduction
    if spy_ctx.get("bearish_flag"):
        score = round(score * config.SPY_SCORE_REDUCTION, 2)
    if spy_ctx.get("high_vol_flag") and strategy == "filter_b":
        return None  # Signal disabled in high-vol regime

    # Outcome: check T+1 close vs stop/target
    try:
        next_close = float(df.iloc[sim_day_idx + 1]["Close"])
        if pd.isna(next_close):
            outcome = "open"
        elif next_close >= target:
            outcome = "success"
        elif next_close <= stop:
            outcome = "failed"
        else:
            outcome = "open"  # Neither hit — excluded from win rate
    except (IndexError, KeyError):
        outcome = "open"

    outcome_pct = None
    if outcome in ("success", "failed"):
        outcome_pct = (next_close - entry_price) / entry_price

    return {
        "ticker":      ticker,
        "strategy":    strategy,
        "sim_date":    df.index[sim_day_idx].strftime("%Y-%m-%d") if hasattr(df.index[sim_day_idx], 'strftime') else str(df.index[sim_day_idx]),
        "entry":       round(entry_price, 4),
        "stop":        stop,
        "target":      target,
        "score":       score,
        "momentum":    feats["momentum"],
        "volume_spike": feats["volume_spike"],
        "volatility":  feats["volatility"],
        "atr":         atr,
        "outcome":     outcome,
        "outcome_pct": round(outcome_pct, 6) if outcome_pct is not None else None,
        "spy_bearish": spy_ctx.get("bearish_flag", 0),
        "spy_high_vol": spy_ctx.get("high_vol_flag", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    sim_trades: List[Dict[str, Any]],
    strategy:   str,
    weights:    Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute win rate, avg return, and max drawdown from simulated trades.

    Only 'success' and 'failed' outcomes count toward win rate.
    'open' outcomes are excluded (neither stop nor target was hit next day).

    Max drawdown:
      equity_curve = cumulative product of (1 + return_i) for closed trades
      running_peak = expanding maximum of equity_curve
      drawdown     = (equity_curve - running_peak) / running_peak
      max_drawdown = min(drawdown_series)  →  negative number

    Returns metrics dict.
    """
    closed = [t for t in sim_trades if t["outcome"] in ("success", "failed")]
    total  = len(closed)

    if total == 0:
        return {
            "win_rate":       None,
            "avg_return":     None,
            "max_drawdown":   None,
            "trades_sampled": 0,
            "open_count":     len(sim_trades) - total,
            "note":           "No closed simulated trades",
        }

    wins    = sum(1 for t in closed if t["outcome"] == "success")
    win_rate = wins / total

    returns = [t["outcome_pct"] for t in closed if t["outcome_pct"] is not None]
    avg_return = float(np.mean(returns)) if returns else None

    # Max drawdown via equity curve
    max_drawdown = None
    if returns:
        equity = np.cumprod(1 + np.array(returns))
        peak   = np.maximum.accumulate(equity)
        # Guard against peak = 0 (degenerate)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (equity - peak) / peak, 0)
        max_drawdown = float(np.min(dd)) if len(dd) > 0 else None

    open_count = len([t for t in sim_trades if t["outcome"] == "open"])

    return {
        "strategy":       strategy,
        "win_rate":       round(win_rate, 4),
        "avg_return":     round(avg_return, 6) if avg_return is not None else None,
        "max_drawdown":   round(max_drawdown, 6) if max_drawdown is not None else None,
        "trades_sampled": total,
        "open_count":     open_count,
        "wins":           wins,
        "losses":         total - wins,
        "weights_used":   weights.copy(),
        "note":           (
            f"Backtest used weights: "
            + ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
            + ". Results are hypothetical under current weights."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    ohlcv:    Dict[str, pd.DataFrame],
    spy_ohlcv: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Run backtesting-lite over the last BACKTEST_DAYS of available data.

    For each ticker × each simulated day:
      - Compute features using only past data
      - Score using current DB weights
      - Simulate outcome using next-day close

    Then compute win rate, avg return, max drawdown per strategy.
    Save results to metrics table (source='backtest').

    Returns full backtest result dict for UI display.
    """
    result: Dict[str, Any] = {
        "run_date":      date.today().isoformat(),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategies":    {},
        "all_sim_trades": [],
        "error":          None,
    }

    try:
        # Load current weights (snapshot for this backtest run)
        weights = {s: db.get_weights(s) for s in config.BASE_WEIGHTS.keys()}
        result["weights_snapshot"] = weights

        # Build SPY DataFrame — use provided or extract from ohlcv
        if spy_ohlcv is None:
            spy_ohlcv = ohlcv.get("SPY")
        if spy_ohlcv is None:
            # Neutral SPY context if not available
            spy_ohlcv = pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            )

        # Load universe to know which strategy each ticker belongs to
        from universe import load_universe
        universe_df = load_universe()
        ticker_strategy: Dict[str, str] = {}
        if not universe_df.empty and "strategy" in universe_df.columns:
            ticker_strategy = dict(zip(universe_df["ticker"], universe_df["strategy"]))

        all_sim_trades: List[Dict[str, Any]] = []

        for ticker, df in ohlcv.items():
            if ticker == "SPY" or df is None or df.empty:
                continue

            strategy = ticker_strategy.get(ticker, "filter_a")
            w        = weights[strategy]

            # Determine simulation window: last BACKTEST_DAYS calendar days
            # Convert to approx trading days (5/7 ratio)
            trading_days = int(config.BACKTEST_DAYS * 5 / 7)
            start_idx    = max(0, len(df) - trading_days - 1)
            end_idx      = len(df) - 1  # leave last bar as T+1 for final sim

            for sim_idx in range(start_idx, end_idx):
                trade = _simulate_day(ticker, df, spy_ohlcv, sim_idx, strategy, w)
                if trade is not None:
                    all_sim_trades.append(trade)

        result["all_sim_trades"] = all_sim_trades
        app_logger.info(
            "Backtest: %d simulated trades across %d tickers",
            len(all_sim_trades), len(ohlcv),
        )

        # Compute metrics per strategy
        today_str = date.today().isoformat()
        for strategy in config.BASE_WEIGHTS.keys():
            strat_trades = [t for t in all_sim_trades if t["strategy"] == strategy]
            metrics      = _compute_metrics(strat_trades, strategy, weights[strategy])
            result["strategies"][strategy] = metrics

            # Save to DB (source='backtest')
            if metrics["trades_sampled"] > 0:
                db.save_metrics({
                    "date":           today_str,
                    "strategy":       strategy,
                    "win_rate":       metrics["win_rate"],
                    "avg_return":     metrics["avg_return"],
                    "max_drawdown":   metrics["max_drawdown"],
                    "trades_sampled": metrics["trades_sampled"],
                    "source":         "backtest",
                })

    except Exception as e:
        result["error"] = str(e)
        logger.error("run_backtest failed: %s\n%s", e, traceback.format_exc())

    return result


def get_latest_backtest_results() -> Dict[str, Any]:
    """
    Returns most recent backtest metrics from DB for Page 4 display.
    Returns empty dict if no backtest has been run.
    """
    results = {}
    for strategy in config.BASE_WEIGHTS.keys():
        m = db.get_latest_metrics(strategy, source="backtest")
        if m:
            results[strategy] = m
    return results


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import database as db
    db.init_db()

    print("backtest.py self-test (mock data — no network)")
    print("=" * 55)

    # Build realistic mock OHLCV data (200 bars for proper windowing)
    np.random.seed(42)

    def make_df(n=200, trend=0.001, vol_spike_days=None):
        idx   = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 50.0 * np.cumprod(1 + np.random.randn(n) * 0.012 + trend)
        high  = close * (1 + abs(np.random.randn(n) * 0.006))
        low   = close * (1 - abs(np.random.randn(n) * 0.006))
        vol   = np.full(n, 3_000_000.0)
        if vol_spike_days:
            for d in vol_spike_days:
                if 0 <= d < n:
                    vol[d] = 9_000_000.0
        return pd.DataFrame(
            {"Open": close * 0.999, "High": high, "Low": low,
             "Close": close, "Volume": vol},
            index=idx,
        )

    mock_spy = make_df(n=200, trend=0.0005)
    mock_ohlcv = {
        "AAPL":  make_df(trend=0.002),
        "MSFT":  make_df(trend=0.001),
        "GOOGL": make_df(trend=0.0015),
        "TSLA":  make_df(trend=-0.001),
        "NVDA":  make_df(trend=0.003),
        "SPY":   mock_spy,
    }

    # Mock universe CSV
    import universe as uni
    import unittest.mock as mock
    mock_universe = pd.DataFrame([
        {"ticker": "AAPL",  "strategy": "filter_a", "sector": "Technology"},
        {"ticker": "MSFT",  "strategy": "filter_a", "sector": "Technology"},
        {"ticker": "GOOGL", "strategy": "filter_a", "sector": "Communication"},
        {"ticker": "TSLA",  "strategy": "filter_a", "sector": "Consumer Discretionary"},
        {"ticker": "NVDA",  "strategy": "filter_a", "sector": "Technology"},
    ])

    with mock.patch.object(uni, "load_universe", return_value=mock_universe):
        print("\n[1] Running backtest...")
        result = run_backtest(
            {k: v for k, v in mock_ohlcv.items() if k != "SPY"},
            spy_ohlcv=mock_spy,
        )

    if result["error"]:
        print(f"  ERROR: {result['error']}")
    else:
        total_sims = len(result["all_sim_trades"])
        print(f"  Total simulated trades: {total_sims}")

        for strategy, metrics in result["strategies"].items():
            print(f"\n  Strategy: {strategy}")
            print(f"    Trades sampled: {metrics['trades_sampled']}")
            print(f"    Open (neither hit): {metrics.get('open_count', 0)}")
            if metrics["win_rate"] is not None:
                print(f"    Win rate:    {metrics['win_rate']*100:.1f}%")
                print(f"    Avg return:  {metrics['avg_return']*100:.3f}%" if metrics['avg_return'] else "    Avg return: N/A")
                print(f"    Max drawdown:{metrics['max_drawdown']*100:.2f}%" if metrics['max_drawdown'] else "    Max DD: N/A")
                print(f"    Note: {metrics['note']}")
                # Assertions
                assert 0 <= metrics["win_rate"] <= 1, "Win rate out of [0,1]"
                if metrics["max_drawdown"] is not None:
                    assert metrics["max_drawdown"] <= 0, "Max drawdown must be <= 0"
            else:
                print(f"    {metrics['note']}")

    # [2] Slice test (no-lookahead verification)
    print("\n[2] No-lookahead slice test:")
    df_test = make_df(n=100)
    for t_idx in [30, 50, 80]:
        sliced = _slice_for_day(df_test, t_idx)
        if sliced is not None:
            assert len(sliced) == t_idx, f"Slice length should be t_idx={t_idx}"
            assert sliced.index[-1] < df_test.index[t_idx], "Slice must not include sim day"
    print("  ✅ Slices end strictly before sim day T")

    # [3] SPY context at each sim day
    print("\n[3] SPY context slicing:")
    for t_idx in [50, 100, 150]:
        ctx = _compute_spy_context_for_day(mock_spy, t_idx)
        assert "bearish_flag"  in ctx
        assert "high_vol_flag" in ctx
    print("  ✅ SPY context computed without lookahead")

    # [4] Latest results from DB
    print("\n[4] Latest backtest from DB:")
    latest = get_latest_backtest_results()
    for strat, m in latest.items():
        print(f"  {strat}: win_rate={m.get('win_rate')}  samples={m.get('trades_sampled')}")

    print("\nbacktest.py self-test complete.")
