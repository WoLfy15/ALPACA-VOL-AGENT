"""
research/toxicity.py -- an execution-quality gate, adapted from the
rl_execution_extension project's vpin.py (Easley, Lopez de Prado &
O'Hara 2012, "Flow Toxicity and Liquidity in a High-Frequency World").

Scoped deliberately as EXECUTION QUALITY, not a new alpha leg: VPIN's
own literature (and its most-cited follow-up critique, Andersen &
Bondarenko 2014) treats it as an adverse-selection / liquidity-fragility
proxy for HOW you trade, not a directional forecast of WHERE price goes
next -- research/signal.py's job. Used here to throttle position size
and widen execution limit-price buffers when the underlying's recent
order flow looks unusually one-sided, i.e. exactly VPIN's intended use,
not a repurposing of it.

TWO changes versus a literal port of vpin.py, both empirically motivated
rather than stylistic:

  1. Bar-level Bulk Volume Classification instead of tick-level. Alpaca's
     REST bars endpoint (data/market_data.py, alpaca_client.py) gives
     OHLCV bars, not a raw trade tape -- BVC only needs a price and a
     volume per bucket-forming unit (Easley-Lopez de Prado-O'Hara's own
     eq. 2 is defined on bucket close-to-close price changes, not on
     individual trade attributes), so a bar's (close, volume) is a
     legitimate BVC input; this is a standard practical adaptation when
     tick data isn't available, not an ad hoc shortcut.
  2. CAUSAL ROLLING sigma for the BVC standardization, not vpin.py's
     single whole-sample sigma. This is a real, tested fix, not a
     stylistic preference: a single sigma estimated once over a window
     that spans both a quiet period and a genuine toxic burst gets
     inflated by the burst's own outlier price moves, which SUPPRESSES
     the standardized signal for the (majority) quiet buckets and mutes
     the burst's own buckets too -- verified by construction on a
     synthetic quiet-vs-burst series while building this module: the
     whole-sample-sigma version scored the burst LOWER than the quiet
     baseline (backwards), the causal rolling-sigma version correctly
     peaks during the burst. See _bvc_classify_rolling's docstring.

VPIN's own well-documented noise floor (Andersen & Bondarenko 2014):
because BVC's buy-fraction is Phi(standardized price change), under
PURE noise it is already Uniform(0,1)-distributed, which gives
|2*U-1| a baseline EXPECTED value of 0.5 -- i.e. "VPIN ~ 0.5" is not
itself an alarm, it is closer to VPIN's typical resting state. This is
exactly why the gate below reads current VPIN against its OWN recent
percentile history, not a fixed absolute cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm

MIN_BARS = 120
DEFAULT_BUCKET_DIVISOR = 25    # ~25 "trades" (bars) per volume bucket, same scale vpin.py's own self-test uses
DEFAULT_WINDOW_BUCKETS = 20
SIGMA_LOOKBACK_BUCKETS = 30
MIN_SIGMA_FRACTION_OF_GLOBAL = 0.15  # floors the rolling sigma so a too-quiet lookback doesn't make BVC hypersensitive to tiny ticks

# percentile-vs-own-history thresholds (see module docstring on VPIN's noise floor)
ELEVATED_PERCENTILE = 0.90
TOXIC_PERCENTILE = 0.97
REGIME_SIZE_MULTIPLIER = {"Normal": 1.0, "Elevated": 0.85, "Toxic": 0.60, "Unavailable": 1.0}
# ADDITIONAL bps to widen an execution limit price by, on top of whatever
# base buffer execution/order_manager.py already applies for normal
# indicative-feed spread width -- 0 under normal flow (no extra
# widening needed), growing as flow looks more one-sided.
REGIME_EXTRA_LIMIT_BUFFER_BPS = {"Normal": 0.0, "Elevated": 15.0, "Toxic": 40.0, "Unavailable": 0.0}


@dataclass
class FlowToxicity:
    available: bool
    n_bars: int = 0
    n_buckets: int = 0
    current_vpin: Optional[float] = None
    percentile_vs_history: Optional[float] = None
    regime: str = "Unavailable"
    size_multiplier: float = 1.0
    extra_limit_buffer_bps: float = 0.0   # ADDITIONAL bps to widen execution limits by -- see REGIME_EXTRA_LIMIT_BUFFER_BPS
    notes: list[str] = field(default_factory=list)


def _make_volume_buckets(price: np.ndarray, size: np.ndarray, bucket_volume: float) -> list[dict]:
    """Partition a (price, size) series into fixed-total-volume buckets.
    Ported unchanged from rl_execution_extension/src/vpin.py's
    make_volume_buckets -- a trade (here: a bar) that straddles a bucket
    boundary is split across both buckets, preserving exact volume
    alignment, standard VPIN practice."""
    n = len(price)
    buckets: list[dict] = []
    cum_vol, last_close = 0.0, price[0]
    for i in range(n):
        remaining = size[i]
        while remaining > 1e-12:
            room = bucket_volume - cum_vol
            take = min(room, remaining)
            cum_vol += take
            remaining -= take
            last_close = price[i]
            if cum_vol >= bucket_volume - 1e-9:
                buckets.append({"close": last_close, "volume": cum_vol})
                cum_vol = 0.0
    if cum_vol > 0:
        buckets.append({"close": last_close, "volume": cum_vol})
    return buckets


def _bvc_classify_rolling(buckets: list[dict], bucket_volume: float, sigma_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Bulk Volume Classification (Easley-Lopez de Prado-O'Hara 2012 eq. 2),
    with a CAUSAL rolling sigma instead of vpin.py's single whole-sample
    sigma -- see module docstring for why. sigma at bucket i uses only
    buckets [i-sigma_window, i) (no lookahead), floored at a fraction of
    the whole-sample sigma so a too-quiet lookback window doesn't make
    the classification hypersensitive to noise-level ticks."""
    closes = np.array([b["close"] for b in buckets])
    dP = np.diff(closes, prepend=closes[0])
    dP[0] = 0.0
    n = len(dP)
    global_sigma = float(np.std(dP[1:])) if n > 2 else 1.0
    global_sigma = max(global_sigma, 1e-9)

    sigma_roll = np.empty(n)
    for i in range(n):
        lo = max(0, i - sigma_window)
        window = dP[lo:i]
        sigma_roll[i] = np.std(window) if len(window) >= 5 else global_sigma
    sigma_roll = np.maximum(sigma_roll, MIN_SIGMA_FRACTION_OF_GLOBAL * global_sigma)
    sigma_roll = np.maximum(sigma_roll, 1e-9)

    frac_buy = norm.cdf(dP / sigma_roll)
    return frac_buy * bucket_volume, (1 - frac_buy) * bucket_volume


def _compute_vpin_series(price: np.ndarray, size: np.ndarray, bucket_volume: float,
                          window_buckets: int, sigma_window: int) -> tuple[np.ndarray, list[dict]]:
    buckets = _make_volume_buckets(price, size, bucket_volume)
    if len(buckets) < window_buckets + 1:
        return np.array([]), buckets
    buy_vol, sell_vol = _bvc_classify_rolling(buckets, bucket_volume, sigma_window)
    imbalance = np.abs(buy_vol - sell_vol)
    vpin = np.full(len(buckets), np.nan)
    csum = np.cumsum(imbalance)
    for k in range(window_buckets - 1, len(buckets)):
        window_imbalance = csum[k] - (csum[k - window_buckets] if k >= window_buckets else 0.0)
        vpin[k] = window_imbalance / (window_buckets * bucket_volume)
    return vpin, buckets


def assess_flow_toxicity(
    bars: list[dict], bucket_divisor: int = DEFAULT_BUCKET_DIVISOR,
    window_buckets: int = DEFAULT_WINDOW_BUCKETS, sigma_window: int = SIGMA_LOOKBACK_BUCKETS,
) -> FlowToxicity:
    """bars: Alpaca bar dicts (the same raw shape data/market_data.py's
    daily_log_returns_from_bars consumes -- 'c' close, 'v' volume),
    ideally intraday (e.g. 5Min) bars over the last several trading days
    so enough buckets form to have a meaningful trailing distribution to
    read the current percentile against."""
    if len(bars) < MIN_BARS:
        return FlowToxicity(available=False, n_bars=len(bars),
                             notes=[f"only {len(bars)} bars (<{MIN_BARS}); flow-toxicity read skipped."])

    price = np.array([b["c"] for b in bars], dtype=float)
    size = np.array([b.get("v", 0) for b in bars], dtype=float)
    if np.all(size <= 0) or np.median(size[size > 0] if np.any(size > 0) else [0]) <= 0:
        return FlowToxicity(available=False, n_bars=len(bars), notes=["bar volume data missing/zero; flow-toxicity read skipped."])

    bucket_volume = float(np.median(size[size > 0])) * bucket_divisor
    if bucket_volume <= 0:
        return FlowToxicity(available=False, n_bars=len(bars), notes=["degenerate bucket volume; flow-toxicity read skipped."])

    vpin, buckets = _compute_vpin_series(price, size, bucket_volume, window_buckets, sigma_window)
    valid = vpin[~np.isnan(vpin)]
    if len(valid) < window_buckets:
        return FlowToxicity(available=False, n_bars=len(bars), n_buckets=len(buckets),
                             notes=[f"only {len(valid)} completed VPIN windows; not enough trailing history for a percentile read."])

    current = float(valid[-1])
    percentile = float(np.mean(valid <= current))  # current's rank within its OWN trailing distribution

    if percentile >= TOXIC_PERCENTILE:
        regime = "Toxic"
    elif percentile >= ELEVATED_PERCENTILE:
        regime = "Elevated"
    else:
        regime = "Normal"

    return FlowToxicity(
        available=True, n_bars=len(bars), n_buckets=len(buckets),
        current_vpin=current, percentile_vs_history=percentile, regime=regime,
        size_multiplier=REGIME_SIZE_MULTIPLIER[regime],
        extra_limit_buffer_bps=REGIME_EXTRA_LIMIT_BUFFER_BPS[regime],
    )
