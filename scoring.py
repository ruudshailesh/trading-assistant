"""
scoring.py — Feature calculation, scoring, and trade selection pipeline.

Responsibilities:
- Compute all 4 features per ticker from OHLCV data (strictly past data only)
- Normalize features to 0-1 using fixed bounds (not min-max — scores comparable
  across days and universe sizes)
- Load weights from DB, apply SPY market context adjustments
- Run full selection pipeline with all constraints:
    deduplication, wash sale, sector cap, correlation filter, Filter B cap
- Return fully explained trade candidates

Import chain: config -> database -> universe -> data_fetch -> scoring
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import database as db
import universe as uni

logger     = logging.getLogger("scoring")
app_logger = logging.getLogger("app")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CALCULATION
# All windows use STRICTLY PAST data — current day's close used only for entry
# price, never for feature computation. This eliminates lookahead bias.
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(ticker: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Compute all 4 features for a single ticker from its OHLCV DataFrame.
    Returns None if any required feature cannot be computed (NaN guard).

    Features:
      momentum     = 5-day return: (close[t-1] - close[t-6]) / close[t-6]
      volume_spike = close[t-1] volume / mean(volume[t-11 : t-1])
      volatility   = std of daily returns over close[t-11 : t-1]
      atr          = mean True Range over last ATR_WINDOW days, floored at 1% of price

    Window indexing (all exclusive of current bar):
      iloc[-1]  = most recent completed bar (today if after close, yesterday if intraday)
      iloc[-6]  = 5 bars ago (for 5-day momentum)
      iloc[-(VOL_AVG_WINDOW+1) : -1] = 10 bars before the last completed bar
    """
    try:
        close  = df["Close"].dropna()
        high   = df["High"].dropna()
        low    = df["Low"].dropna()
        volume = df["Volume"].dropna()

        # Minimum data guard
        min_len = max(
            config.MOMENTUM_WINDOW + 2,
            config.VOL_AVG_WINDOW  + 2,
            config.ATR_WINDOW      + 2,
        )
        if len(close) < min_len:
            logger.debug("%s: insufficient rows (%d < %d)", ticker, len(close), min_len)
            return None

        # ── Momentum: 5-day return ────────────────────────────────────────────
        # Uses close[t-6] and close[t-1] — no current bar
        if len(close) < config.MOMENTUM_WINDOW + 2:
            return None
        c_now  = float(close.iloc[-1])
        c_prev = float(close.iloc[-(config.MOMENTUM_WINDOW + 1)])
        if c_prev <= 0:
            return None
        momentum = (c_now - c_prev) / c_prev

        # ── Volume spike: current / 10-day avg ───────────────────────────────
        # Avg computed from [t-11 : t-1] to exclude current bar — no lookahead
        if len(volume) < config.VOL_AVG_WINDOW + 2:
            return None
        vol_current = float(volume.iloc[-1])
        vol_avg     = float(volume.iloc[-(config.VOL_AVG_WINDOW + 1):-1].mean())
        if vol_avg <= 0 or np.isnan(vol_avg):
            return None
        volume_spike = vol_current / vol_avg

        # ── Volatility: std dev of daily returns ──────────────────────────────
        # Computed from [t-11 : t-1] — excludes current bar
        if len(close) < config.VOLATILITY_WINDOW + 2:
            return None
        ret_slice  = close.iloc[-(config.VOLATILITY_WINDOW + 1):-1].pct_change().dropna()
        if len(ret_slice) < 3:
            return None
        volatility = float(ret_slice.std())

        # ── ATR: Average True Range ───────────────────────────────────────────
        # True Range = max(H-L, |H-prev_C|, |L-prev_C|) over last ATR_WINDOW bars
        # All computed from past bars — no current day in the range calculation
        if len(close) < config.ATR_WINDOW + 2:
            return None
        atr_close  = close.iloc[-(config.ATR_WINDOW + 1):-1].values
        atr_high   = high.iloc[-(config.ATR_WINDOW + 1):-1].values
        atr_low    = low.iloc[-(config.ATR_WINDOW + 1):-1].values

        tr_list = []
        for i in range(1, len(atr_close)):
            h, l, pc = atr_high[i], atr_low[i], atr_close[i - 1]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_list.append(tr)

        if not tr_list:
            return None
        atr_raw = float(np.mean(tr_list))

        # ATR floor: prevents division-by-zero in risk.py for very low-vol tickers
        entry_price = c_now
        atr = max(atr_raw, entry_price * config.ATR_MIN_FLOOR_PCT)

        # ── NaN guard — reject if any feature is NaN ──────────────────────────
        features = {
            "ticker":       ticker,
            "entry":        round(entry_price, 4),
            "momentum":     momentum,
            "volume_spike": volume_spike,
            "volatility":   volatility,
            "atr":          atr,
        }
        for key, val in features.items():
            if key in ("ticker",):
                continue
            if val is None or (isinstance(val, float) and np.isnan(val)):
                logger.warning("%s: NaN in feature '%s', skipped", ticker, key)
                return None

        return features

    except Exception as e:
        logger.error("compute_features failed for %s: %s\n%s",
                     ticker, e, traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SCORE NORMALIZATION
# Fixed bounds — not min-max across universe.
# Ensures scores are comparable across days regardless of universe composition.
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(value: float, feature: str) -> float:
    """
    Clip value to fixed bounds then scale to [0, 1].
    Bounds defined in config.NORM_BOUNDS.
    Returns 0.5 as neutral if feature not in bounds dict (safe fallback).
    """
    if feature not in config.NORM_BOUNDS:
        return 0.5
    lo, hi = config.NORM_BOUNDS[feature]
    clipped = max(lo, min(hi, value))
    if hi == lo:
        return 0.5
    return (clipped - lo) / (hi - lo)


def compute_score(
    features: Dict[str, Any],
    strategy: str,
    weights:  Dict[str, float],
) -> float:
    """
    Compute weighted confidence score for a ticker.

    raw_score = w_momentum  × norm(momentum)
              + w_volume    × norm(volume_spike)
              + w_volatility × norm(volatility)

    Final score = raw_score × 100  →  range [0, 100]

    Weights loaded from DB (adapted by learning system over time).
    For filter_a, volatility weight is 0 by default (stable stocks —
    volatility is noise, not signal for this tier).
    """
    w = weights
    raw = (
        w.get("momentum",   0) * _normalize(features["momentum"],     "momentum")     +
        w.get("volume",     0) * _normalize(features["volume_spike"],  "volume_spike") +
        w.get("volatility", 0) * _normalize(features["volatility"],   "volatility")
    )
    return round(raw * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# SPY MARKET CONTEXT ADJUSTMENTS
# Applied AFTER scoring — market context reduces/disables scores, never boosts
# ─────────────────────────────────────────────────────────────────────────────

def apply_spy_adjustments(
    scored: List[Dict[str, Any]],
    spy_ctx: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Apply SPY market context to scored candidates:
      - bearish_flag=1  → multiply ALL scores by SPY_SCORE_REDUCTION (0.80)
      - high_vol_flag=1 → remove ALL filter_b candidates entirely

    Returns adjusted list (may be shorter if filter_b removed).
    Adds 'spy_note' field to each candidate for UI explanation.
    """
    bearish  = bool(spy_ctx.get("bearish_flag",  0))
    high_vol = bool(spy_ctx.get("high_vol_flag", 0))

    adjusted = []
    for c in scored:
        # High volatility regime: disable risky (filter_b) trades entirely
        if high_vol and c["strategy"] == "filter_b":
            logger.debug("SPY high-vol: removing filter_b candidate %s", c["ticker"])
            continue

        # Bearish regime: reduce all scores by 20%
        if bearish:
            c = dict(c)  # copy — don't mutate original
            c["score"]    = round(c["score"] * config.SPY_SCORE_REDUCTION, 2)
            c["spy_note"] = "SPY bearish (price < 20DMA) — score reduced 20%"
        elif high_vol:
            c["spy_note"] = "SPY high-volatility — Filter B disabled"
        else:
            c["spy_note"] = "SPY neutral"

        adjusted.append(c)

    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION FILTER
# Skip tickers too similar to already-selected ones (10-day return correlation)
# ─────────────────────────────────────────────────────────────────────────────

def _get_return_series(
    ticker: str,
    ohlcv:  Dict[str, pd.DataFrame],
    window: int = 10,
) -> Optional[pd.Series]:
    """
    Returns the last `window` daily returns for a ticker.
    Used for pairwise correlation check during selection.
    Returns None if data unavailable.
    """
    df = ohlcv.get(ticker)
    if df is None or len(df) < window + 1:
        return None
    returns = df["Close"].pct_change().dropna().iloc[-window:]
    if len(returns) < window:
        return None
    return returns


def _is_too_correlated(
    ticker:   str,
    selected: List[str],
    ohlcv:    Dict[str, pd.DataFrame],
) -> bool:
    """
    Returns True if `ticker` has 10-day return correlation > CORR_THRESHOLD
    with ANY already-selected ticker.
    Conservatively returns False (allow trade) if data is missing.
    """
    new_ret = _get_return_series(ticker, ohlcv)
    if new_ret is None:
        return False  # Can't check — allow

    for sel_ticker in selected:
        sel_ret = _get_return_series(sel_ticker, ohlcv)
        if sel_ret is None:
            continue
        try:
            # Align on shared index before correlating
            aligned = pd.concat([new_ret, sel_ret], axis=1).dropna()
            if len(aligned) < 5:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if not np.isnan(corr) and corr > config.CORR_THRESHOLD:
                logger.debug(
                    "Correlation filter: %s vs %s = %.2f > threshold %.2f",
                    ticker, sel_ticker, corr, config.CORR_THRESHOLD,
                )
                return True
        except Exception:
            continue

    return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SELECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_selection_pipeline(
    ohlcv:   Dict[str, pd.DataFrame],
    spy_ctx: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Full daily trade selection pipeline. Returns (selected_trades, pipeline_log).

    Steps:
      1.  Portfolio capacity check — if full, return early
      2.  Load universe from CSV
      3.  Compute features for all tickers (NaN-safe)
      4.  Load weights from DB per strategy
      5.  Score all tickers
      6.  Apply SPY market context
      7.  Sort by score descending
      8.  Greedy selection with:
            - Deduplication (skip if already in active trade)
            - Wash sale check
            - Filter B volume spike re-validation (live OHLCV)
            - Sector cap (max 1 per GICS sector)
            - Max 1 Filter B trade
            - Correlation filter
      9.  Attach risk levels (stop/target/size computed by risk.py)
      10. Build full explanation string per trade

    pipeline_log: dict of counts/flags for UI display and debugging.
    """
    pipeline_log: Dict[str, Any] = {
        "universe_size":     0,
        "features_computed": 0,
        "after_spy":         0,
        "after_dedup":       0,
        "selected":          0,
        "capacity_slots":    0,
        "skipped_reasons":   [],
    }

    # ── Step 1: Portfolio capacity ─────────────────────────────────────────────
    active_trades = db.get_active_trades()
    active_count  = len(active_trades)
    slots         = config.MAX_ACTIVE_TRADES - active_count
    pipeline_log["capacity_slots"] = slots

    if slots <= 0:
        app_logger.info("Selection pipeline: portfolio at capacity (%d active)", active_count)
        pipeline_log["skipped_reasons"].append(
            f"Portfolio at capacity ({active_count}/{config.MAX_ACTIVE_TRADES} active trades)"
        )
        return [], pipeline_log

    # ── Step 2: Load universe ─────────────────────────────────────────────────
    universe_df = uni.load_universe()
    if universe_df.empty:
        pipeline_log["skipped_reasons"].append("Universe CSV not found — run Refresh Universe")
        return [], pipeline_log

    # Only score tickers we actually have OHLCV data for
    available_tickers = set(ohlcv.keys())
    universe_df = universe_df[universe_df["ticker"].isin(available_tickers)]
    pipeline_log["universe_size"] = len(universe_df)

    # ── Step 3 & 4: Compute features + load weights ────────────────────────────
    weights = {
        strat: db.get_weights(strat)
        for strat in ("filter_a", "filter_b")
    }

    scored_candidates: List[Dict[str, Any]] = []

    for _, row in universe_df.iterrows():
        ticker   = row["ticker"]
        strategy = row["strategy"]

        feats = compute_features(ticker, ohlcv[ticker])
        if feats is None:
            continue  # NaN or insufficient data — skip with logged warning

        score = compute_score(feats, strategy, weights[strategy])

        scored_candidates.append({
            **feats,
            "strategy": strategy,
            "sector":   uni.get_sector(ticker),
            "score":    score,
            "weights_used": weights[strategy].copy(),
        })

    pipeline_log["features_computed"] = len(scored_candidates)

    if not scored_candidates:
        pipeline_log["skipped_reasons"].append("No tickers had valid feature data")
        return [], pipeline_log

    # ── Step 5: Apply SPY adjustments ─────────────────────────────────────────
    scored_candidates = apply_spy_adjustments(scored_candidates, spy_ctx)
    pipeline_log["after_spy"] = len(scored_candidates)

    # ── Step 6: Sort by score descending ──────────────────────────────────────
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # ── Step 7: Pre-fetch active ticker set + sector counts ────────────────────
    active_tickers  = set(db.get_tickers_in_active_trades())
    active_sectors  = {}
    risky_count     = sum(1 for t in active_trades if t.get("strategy") == "filter_b")

    # Track selections in-flight for this session
    selected:        List[Dict[str, Any]] = []
    selected_tickers: List[str]           = []
    session_sectors:  Dict[str, int]      = dict(active_sectors)

    # ── Step 8: Greedy selection with all constraints ─────────────────────────
    for candidate in scored_candidates:
        if len(selected) >= min(slots, 3):
            break  # Filled available slots

        ticker   = candidate["ticker"]
        strategy = candidate["strategy"]
        sector   = candidate.get("sector", "Unknown")

        # -- Deduplication: skip if already in active trade -------------------
        if ticker in active_tickers:
            pipeline_log["skipped_reasons"].append(f"{ticker}: already in active trade")
            continue

        # -- Wash sale check --------------------------------------------------
        if db.check_wash_sale(ticker):
            pipeline_log["skipped_reasons"].append(f"{ticker}: wash sale (loss within {config.WASH_SALE_DAYS}d)")
            continue

        # -- Filter B: re-validate volume spike with live OHLCV ---------------
        if strategy == "filter_b":
            df = ohlcv.get(ticker)
            if df is not None and len(df) >= config.VOL_AVG_WINDOW + 2:
                vol_series  = df["Volume"].dropna()
                avg_vol     = vol_series.iloc[-(config.VOL_AVG_WINDOW + 1):-1].mean()
                current_vol = vol_series.iloc[-1]
                if avg_vol > 0 and current_vol < config.FILTER_B_VOL_MULT * avg_vol:
                    pipeline_log["skipped_reasons"].append(
                        f"{ticker}: filter_b volume spike too low "
                        f"({current_vol/avg_vol:.1f}x < {config.FILTER_B_VOL_MULT}x)"
                    )
                    continue

        # -- Max 1 Filter B trade in portfolio --------------------------------
        if strategy == "filter_b":
            current_risky = risky_count + sum(
                1 for s in selected if s["strategy"] == "filter_b"
            )
            if current_risky >= config.MAX_RISKY_TRADES:
                pipeline_log["skipped_reasons"].append(
                    f"{ticker}: filter_b cap reached ({config.MAX_RISKY_TRADES} max risky)"
                )
                continue

        # -- Sector cap: max 1 trade per GICS sector --------------------------
        sector_count = session_sectors.get(sector, 0)
        if sector_count >= config.MAX_SAME_SECTOR:
            pipeline_log["skipped_reasons"].append(
                f"{ticker}: sector cap ({sector} already has {sector_count} trade)"
            )
            continue

        # -- Correlation filter -----------------------------------------------
        if _is_too_correlated(ticker, selected_tickers, ohlcv):
            pipeline_log["skipped_reasons"].append(
                f"{ticker}: correlation > {config.CORR_THRESHOLD} with a selected ticker"
            )
            continue

        # ── All checks passed — select this ticker ───────────────────────────
        session_sectors[sector] = sector_count + 1
        selected_tickers.append(ticker)

        candidate = dict(candidate)
        candidate["explanation"] = _build_explanation(candidate, spy_ctx, weights[strategy])
        selected.append(candidate)

    pipeline_log["selected"]      = len(selected)
    pipeline_log["after_dedup"]   = len(scored_candidates)

    app_logger.info(
        "Selection pipeline complete: %d/%d slots filled. "
        "Universe=%d  Features=%d  PostSPY=%d  Selected=%d",
        len(selected), slots,
        pipeline_log["universe_size"],
        pipeline_log["features_computed"],
        pipeline_log["after_spy"],
        pipeline_log["selected"],
    )
    return selected, pipeline_log


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION BUILDER
# Every selected trade gets a plain-English explanation for the UI
# ─────────────────────────────────────────────────────────────────────────────

def _build_explanation(
    c:        Dict[str, Any],
    spy_ctx:  Dict[str, Any],
    weights:  Dict[str, float],
) -> str:
    """
    Builds the full plain-English explanation string for a trade candidate.
    Shown on Page 1 (Daily Trades) and stored in the trades table.
    All values are deterministic — same inputs always produce same explanation.
    """
    bearish  = spy_ctx.get("bearish_flag",  0)
    high_vol = spy_ctx.get("high_vol_flag", 0)

    spy_status = (
        "HIGH-VOL REGIME (Filter B disabled)"  if high_vol else
        "BEARISH (score reduced 20%)"           if bearish  else
        "NEUTRAL"
    )

    momentum_pct   = round(c["momentum"]     * 100, 2)
    vol_spike_x    = round(c["volume_spike"], 2)
    vol_pct        = round(c["volatility"]   * 100, 2)
    w_mom          = round(weights.get("momentum",   0), 2)
    w_vol          = round(weights.get("volume",     0), 2)
    w_vlt          = round(weights.get("volatility", 0), 2)

    return (
        f"Strategy: {c['strategy'].upper()} | "
        f"Sector: {c.get('sector','Unknown')} | "
        f"Score: {c['score']}/100\n"
        f"  Momentum (5d): {momentum_pct:+.2f}%  "
        f"[weight={w_mom}]\n"
        f"  Volume spike:  {vol_spike_x:.2f}×  "
        f"[weight={w_vol}]\n"
        f"  Volatility:    {vol_pct:.2f}% daily std  "
        f"[weight={w_vlt}]\n"
        f"  ATR (14d):     ${c['atr']:.2f}\n"
        f"  SPY context:   {spy_status}\n"
        f"  {c.get('spy_note','')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TICKER SCORING (for watchlist Page 2)
# ─────────────────────────────────────────────────────────────────────────────

def score_single_ticker(
    ticker: str,
    df:     pd.DataFrame,
    spy_ctx: Dict[str, Any],
    strategy: str = "filter_a",
) -> Optional[Dict[str, Any]]:
    """
    Score a single ticker for watchlist display.
    Strategy defaults to filter_a unless caller specifies filter_b.
    Returns full candidate dict or None if features can't be computed.
    """
    feats = compute_features(ticker, df)
    if feats is None:
        return None

    weights = db.get_weights(strategy)
    score   = compute_score(feats, strategy, weights)

    candidate = {
        **feats,
        "strategy":    strategy,
        "sector":      uni.get_sector(ticker),
        "score":       score,
        "weights_used": weights,
        "spy_note":    "SPY neutral",
    }

    # Apply SPY adjustments (single item list)
    adjusted = apply_spy_adjustments([candidate], spy_ctx)
    if not adjusted:
        return None

    candidate = adjusted[0]
    candidate["explanation"] = _build_explanation(candidate, spy_ctx, weights)
    return candidate


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import database as db
    db.init_db()

    print("scoring.py self-test (mock data — no network)")
    print("=" * 55)

    # Build a realistic mock OHLCV DataFrame
    def make_mock_ohlcv(
        n: int = 100,
        trend: float = 0.001,
        vol_spike: bool = False,
    ) -> pd.DataFrame:
        np.random.seed(42)
        idx    = pd.date_range("2024-01-01", periods=n, freq="B")
        close  = 50.0 * np.cumprod(1 + np.random.randn(n) * 0.012 + trend)
        high   = close * (1 + abs(np.random.randn(n) * 0.005))
        low    = close * (1 - abs(np.random.randn(n) * 0.005))
        volume = np.full(n, 2_000_000)
        if vol_spike:
            volume[-1] = 8_000_000  # 4× spike on last bar
        return pd.DataFrame(
            {"Open": close, "High": high, "Low": low,
             "Close": close, "Volume": volume},
            index=idx,
        )

    mock_ohlcv = {
        "AAPL": make_mock_ohlcv(trend=0.002, vol_spike=True),
        "MSFT": make_mock_ohlcv(trend=0.001, vol_spike=False),
        "TSLA": make_mock_ohlcv(trend=-0.003),
        "NVDA": make_mock_ohlcv(trend=0.003, vol_spike=True),
    }

    # Test feature computation
    print("\n[1] Feature computation:")
    for ticker, df in mock_ohlcv.items():
        f = compute_features(ticker, df)
        if f:
            print(
                f"  {ticker}: momentum={f['momentum']*100:+.2f}%  "
                f"vol_spike={f['volume_spike']:.2f}x  "
                f"vol={f['volatility']*100:.2f}%  "
                f"atr={f['atr']:.3f}  entry={f['entry']:.2f}"
            )
        else:
            print(f"  {ticker}: FAILED (None returned)")

    # Test normalization
    print("\n[2] Normalization bounds:")
    for feat, (lo, hi) in config.NORM_BOUNDS.items():
        n_lo  = _normalize(lo,         feat)
        n_mid = _normalize((lo+hi)/2,  feat)
        n_hi  = _normalize(hi,         feat)
        print(f"  {feat}: lo={n_lo:.2f}  mid={n_mid:.2f}  hi={n_hi:.2f}  (expect 0.0/0.5/1.0)")

    # Test scoring
    print("\n[3] Scoring:")
    feats_aapl = compute_features("AAPL", mock_ohlcv["AAPL"])
    weights_a  = db.get_weights("filter_a")
    weights_b  = db.get_weights("filter_b")
    score_a    = compute_score(feats_aapl, "filter_a", weights_a)
    score_b    = compute_score(feats_aapl, "filter_b", weights_b)
    print(f"  AAPL filter_a score: {score_a}/100")
    print(f"  AAPL filter_b score: {score_b}/100")
    assert 0 <= score_a <= 100, "Score out of range"

    # Test SPY adjustments
    print("\n[4] SPY adjustments:")
    mock_scored = [
        {"ticker": "AAPL", "strategy": "filter_a", "score": 72.0, "spy_note": ""},
        {"ticker": "RISKY", "strategy": "filter_b", "score": 65.0, "spy_note": ""},
    ]
    # Neutral context
    spy_neutral  = {"bearish_flag": 0, "high_vol_flag": 0}
    adj_neutral  = apply_spy_adjustments(mock_scored, spy_neutral)
    print(f"  Neutral:   {[(c['ticker'], c['score']) for c in adj_neutral]}")
    assert len(adj_neutral) == 2

    # Bearish: scores reduced 20%
    spy_bearish  = {"bearish_flag": 1, "high_vol_flag": 0}
    adj_bearish  = apply_spy_adjustments(mock_scored, spy_bearish)
    print(f"  Bearish:   {[(c['ticker'], round(c['score'],1)) for c in adj_bearish]}")
    assert abs(adj_bearish[0]["score"] - 72.0 * 0.80) < 0.1

    # High-vol: filter_b removed
    spy_highvol  = {"bearish_flag": 0, "high_vol_flag": 1}
    adj_highvol  = apply_spy_adjustments(mock_scored, spy_highvol)
    print(f"  High-vol:  {[(c['ticker'], c['score']) for c in adj_highvol]}  (RISKY removed)")
    assert len(adj_highvol) == 1
    assert adj_highvol[0]["ticker"] == "AAPL"

    # Test single ticker scoring
    print("\n[5] Single ticker score (watchlist):")
    spy_ctx = {"bearish_flag": 0, "high_vol_flag": 0}
    result  = score_single_ticker("AAPL", mock_ohlcv["AAPL"], spy_ctx, "filter_a")
    if result:
        print(f"  AAPL: score={result['score']}  sector={result['sector']}")
        print(f"  Explanation:\n{result['explanation']}")
    else:
        print("  FAILED — None returned")

    # Test correlation filter
    print("\n[6] Correlation filter:")
    # Identical return series → should be filtered
    same_df    = mock_ohlcv["AAPL"].copy()
    corr_ohlcv = {"AAPL": mock_ohlcv["AAPL"], "SAME": same_df}
    too_corr   = _is_too_correlated("SAME", ["AAPL"], corr_ohlcv)
    print(f"  Identical series correlated: {too_corr}  (expect True)")
    not_corr   = _is_too_correlated("TSLA", ["AAPL"], {"AAPL": mock_ohlcv["AAPL"], "TSLA": mock_ohlcv["TSLA"]})
    print(f"  Different series correlated: {not_corr}  (expect False — different trends)")

    print("\nscoring.py self-test complete.")
