"""
research/signal.py -- the "options alpha" layer: turns independent
cheap/rich readings into one composite trade signal.

Up to three legs, deliberately built from unrelated data so they can
cross-check each other (this is the same discipline PLAN.md's Model
Risk section argues for -- an internally-consistent single model can
still be simply wrong, so every trading decision here draws on more
than one method that doesn't share the others' assumptions):

  1. VRP (volatility risk premium): GARCH(1,1)+HAR-RV's forecast of
     REALIZED vol (research/vol_forecast.py) minus the market's live
     ATM IMPLIED vol, read straight off Alpaca's option chain
     (data/market_data.py). Positive => the market is under-pricing
     vol relative to what we expect realized vol to be => options look
     cheap => lean long gamma/vega. TIME-SERIES method.
  2. SAS (Strike-Adjusted Spread, Zou-Derman 1999, pricing_engine/risk/sas.py):
     the SAME market IV minus a "fair" IV built ONLY from the
     underlying's own historical return distribution (no forecasting
     model at all -- a risk-neutralized historical density). Positive
     SAS => this specific strike is priced rich versus its own history.
     HISTORICAL-DENSITY method.
  3. Heston term-structure gap (optional, research/heston_signal.py):
     the forecast vol vs. the LONG-RUN variance level (sqrt(theta))
     implied by fitting Heston to the CURRENT smile shape across
     strikes/expiries -- a CROSS-SECTIONAL method, unlike the other two.
     Deliberately NOT "Heston's fitted ATM IV vs. live ATM IV" -- Heston
     is calibrated to match the smile it's fed, so that comparison would
     be closer to circular than informative. theta is a genuinely
     different read: the market's whole-curve-implied steady-state vol,
     not a re-statement of one point already used to fit the model.

Each leg could be wrong for a different reason (VRP if the GARCH/HAR
forecast is a bad model of future vol; SAS if the recent historical
window isn't representative of the risk-neutral distribution the market
is pricing; Heston if the smile fit is poor or the chain too thin to
calibrate reliably -- see heston_signal.py's data_quality_ok gate).
Requiring them to agree in sign, and DAMPING the composite edge when
they don't, is the whole point of using more than one -- v1 computed
`sas_agrees` but never actually used it to shrink anything, which meant
a full-size trade could fire even while the code's own rationale string
said "SAS disagrees ... size down accordingly." Fixed here: v2 damps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

REGIME_GAMMA_MAP = {
    "Crash": "STRONG_LONG_GAMMA",
    "Vol_Expansion": "LONG_GAMMA",
    "Range": "SHORT_GAMMA",
    "Trend": "DIRECTIONAL_GAMMA",
}

# How much a disagreeing leg shrinks the blended pre-tanh signal. 0.5 is a
# deliberately moderate choice: full-strength damping (0.0, i.e. "any
# disagreement -> flat") would let one noisy leg veto two agreeing ones;
# no damping is v1's bug. Applied once regardless of how many of the
# present legs disagree (multiplying per-pair would over-punish a 3-leg
# read for the same single dissenting leg).
DISAGREEMENT_DAMPING = 0.65


@dataclass
class CompositeSignal:
    vrp_raw: float                 # forecast_vol - live_atm_iv, vol points
    vrp_z: float                   # z-scored against the historical proxy-VRP distribution
    sas_mean: Optional[float]      # mean SAS across near-ATM strikes sampled, vol points
    sas_agrees: bool               # same sign as VRP leg (or SAS unavailable -> True, don't block on missing data)
    composite_edge: float          # in [-1, 1], tanh-squashed blended z; feeds strategy/sizing.py's Kelly step
    regime_label: str
    gamma_regime: str              # meta-controller label, see REGIME_GAMMA_MAP
    rationale: str                 # human-readable one-liner for agent/explain.py
    heston_term_gap: Optional[float] = None   # forecast_vol_now - heston fair long-run vol, vol points
    heston_agrees: bool = True                # same convention as sas_agrees; True when leg unavailable
    damping_applied: bool = False             # transparency flag: did disagreement actually shrink this edge
    legs_used: list[str] = field(default_factory=list)   # which legs fed the blend, for the UI/decision log


def _historical_vrp_zscore(
    forecast_series: "np.ndarray | list[float]",
    realized_vol_series: "np.ndarray | list[float]",
    live_vrp: float,
    proxy_multiplier: float = 1.15,
    lookback: int = 60,
) -> float:
    """Scale `live_vrp` against the historical VRP distribution. We don't
    have a daily history of Alpaca's *live* ATM IV (that would mean
    re-querying the options snapshot endpoint once per historical day,
    which the hackathon's paper environment doesn't make cheap), so the
    distribution's mean/std is built the same way the original notebook
    built its whole VRP series: realized_vol * proxy_multiplier standing
    in for a historical implied-vol level. That proxy is only used to
    calibrate the *scale* of "how big a VRP reading is unusual" -- the
    actual live_vrp value being scored is always the real thing."""
    forecast_series = np.asarray(forecast_series, dtype=float)
    realized_vol_series = np.asarray(realized_vol_series, dtype=float)
    n = min(len(forecast_series), len(realized_vol_series), lookback)
    if n < 10:
        return 0.0
    proxy_iv = realized_vol_series[-n:] * proxy_multiplier
    hist_vrp = forecast_series[-n:] - proxy_iv
    mean, std = float(np.mean(hist_vrp)), float(np.std(hist_vrp))
    if std < 1e-8:
        return 0.0
    return (live_vrp - mean) / std


def build_signal(
    regime_label: str,
    forecast_vol_now: float,
    live_atm_iv: float,
    forecast_vol_history: "np.ndarray | list[float]",
    realized_vol_history: "np.ndarray | list[float]",
    sas_values: Optional[list[float]] = None,
    sas_scale: float = 0.05,
    vrp_weight: float = 0.70,
    heston_term_gap: Optional[float] = None,
    heston_confident: bool = True,
    heston_scale: float = 0.05,
) -> CompositeSignal:
    """
    forecast_vol_now: today's blended GARCH+HAR forecast (VolForecast.blended_next).
    live_atm_iv: today's real ATM IV from Alpaca's chain (data.market_data.atm_implied_vol).
    forecast_vol_history / realized_vol_history: aligned historical series
        used only to calibrate the VRP z-score's scale (see docstring above).
    sas_values: SAS per near-ATM strike from pricing_engine.risk.sas.strike_adjusted_spread,
        already computed by the caller against the live chain + historical returns.
    sas_scale: a SAS reading of this magnitude (vol points) counts as a
        "full unit" of the SAS leg, pre-tanh. 0.05 (5 vol points) is a
        reasonable starting point for liquid single names/ETFs; tune per
        symbol if the chain shows persistently wider/narrower spreads.
    vrp_weight: how much of the composite comes from VRP vs. the other
        legs when more than one is available -- the remaining weight is
        split evenly across whichever of {SAS, Heston} are present.
    heston_term_gap: forecast_vol_now - heston fair long-run vol
        (research.heston_signal.HestonCrossCheck.term_gap), vol points.
        None when the chain didn't have enough usable strikes to
        calibrate, OR when the caller decided not to compute it --
        either way the signal degrades gracefully to the v1 two-leg
        (or one-leg) read, same as sas_values=None always has.
    heston_confident: research.heston_signal's own data_quality_ok flag.
        A poor calibration fit (few strikes, high RMSE) is worse than no
        leg at all -- pass False to exclude it even if a number exists.
    heston_scale: same role as sas_scale, for the Heston leg.
    """
    vrp_raw = forecast_vol_now - live_atm_iv
    vrp_z = _historical_vrp_zscore(forecast_vol_history, realized_vol_history, vrp_raw)

    sas_mean = float(np.mean(sas_values)) if sas_values else None
    sas_component = -sas_mean / sas_scale if sas_mean is not None else 0.0
    # SAS positive => rich => bearish-on-vol (short-vol lean) => negative
    # contribution to a signal where positive = "go long vol".

    use_heston = heston_term_gap is not None and heston_confident
    heston_component = heston_term_gap / heston_scale if use_heston else 0.0
    # heston_term_gap follows the same sign convention as vrp_raw (forecast
    # MINUS a "fair" vol reading), so, unlike SAS, it does NOT get negated.

    sas_agrees = True
    if sas_mean is not None and abs(vrp_z) > 0.25 and abs(sas_mean) > 1e-4:
        sas_agrees = (vrp_z > 0) == (sas_component > 0)

    heston_agrees = True
    if use_heston and abs(vrp_z) > 0.25 and abs(heston_component) > 1e-4:
        heston_agrees = (vrp_z > 0) == (heston_component > 0)

    legs_used = ["vrp"]
    weighted_terms = [(vrp_weight if (sas_mean is not None or use_heston) else 1.0, vrp_z)]
    others = []
    if sas_mean is not None:
        others.append(("sas", sas_component))
    if use_heston:
        others.append(("heston", heston_component))
    if others:
        other_weight_each = (1.0 - vrp_weight) / len(others)
        for name, val in others:
            weighted_terms.append((other_weight_each, val))
            legs_used.append(name)

    blended = sum(w * v for w, v in weighted_terms)

    # DAMPING: shrink the blended read (before the tanh squash) whenever
    # any present leg disagrees with VRP in sign -- see module docstring
    # and DISAGREEMENT_DAMPING's own comment for why a partial, single
    # multiplier rather than a hard veto or a per-leg stack.
    disagreement = (sas_mean is not None and not sas_agrees) or (use_heston and not heston_agrees)
    damping_applied = bool(disagreement)
    if disagreement:
        blended *= DISAGREEMENT_DAMPING

    composite_edge = math.tanh(0.85 * blended)

    gamma_regime = REGIME_GAMMA_MAP.get(regime_label, "NEUTRAL")
    if regime_label == "Vol_Expansion" and vrp_raw <= 0:
        gamma_regime = "NEUTRAL"
    if regime_label == "Range" and vrp_raw >= 0:
        gamma_regime = "NEUTRAL"

    direction = "cheap (long-vol lean)" if composite_edge > 0 else "rich (short-vol lean)"
    leg_notes = []
    if sas_mean is not None:
        leg_notes.append(f"SAS mean={sas_mean:+.1%} ({'agrees' if sas_agrees else 'DISAGREES'})")
    else:
        leg_notes.append("SAS unavailable")
    if use_heston:
        leg_notes.append(f"Heston term gap={heston_term_gap:+.1%} ({'agrees' if heston_agrees else 'DISAGREES'})")
    elif heston_term_gap is not None and not heston_confident:
        leg_notes.append("Heston leg low-confidence, excluded")
    damping_note = " Edge damped for disagreement -- size down accordingly." if damping_applied else ""
    rationale = (
        f"Regime={regime_label} ({gamma_regime}). Forecast vol {forecast_vol_now:.1%} vs "
        f"live ATM IV {live_atm_iv:.1%} -> VRP {vrp_raw:+.1%} (z={vrp_z:+.2f}). "
        f"{'; '.join(leg_notes)}. "
        f"Composite edge {composite_edge:+.2f}: options look {direction}.{damping_note}"
    )

    return CompositeSignal(
        vrp_raw=vrp_raw, vrp_z=vrp_z, sas_mean=sas_mean, sas_agrees=sas_agrees,
        composite_edge=composite_edge, regime_label=regime_label, gamma_regime=gamma_regime,
        rationale=rationale, heston_term_gap=heston_term_gap if use_heston else None,
        heston_agrees=heston_agrees, damping_applied=damping_applied, legs_used=legs_used,
    )
