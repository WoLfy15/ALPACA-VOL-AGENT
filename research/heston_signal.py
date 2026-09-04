"""
research/heston_signal.py -- wires the vendored Heston calibrator
(pricing_engine/calibration/heston_calibration.py + models/heston.py)
into the live agent. Both modules existed in the vendored pricing_engine
from day one but were never called anywhere outside their own tests --
this is genuinely new plumbing, not a cosmetic wrapper.

Deliberately scoped as TWO things, and explicitly NOT a third
independent "is IV cheap or rich" leg on its own:

  1. A DATA-QUALITY cross-check. Calibrating Heston jointly to every
     strike on the live chain (not just the single ATM print
     data/market_data.atm_implied_vol reads) and comparing its fitted
     ATM IV back against that single print catches a stale/crossed
     quote at the specific ATM strike that the rest of the smile
     doesn't corroborate -- exactly the kind of failure an
     indicative-feed paper account can produce. This is why
     `heston_atm_iv` is NOT fed into research/signal.py as a cheap/rich
     read: Heston is CALIBRATED to match the smile it's given, so
     "model ATM IV vs. market ATM IV" is close to circular by
     construction (of course they roughly agree -- that's what fitting
     means) and would silently double-count the same live_atm_iv number
     signal.py already uses in VRP. It is genuinely informative as a
     *consistency* check, though: a big gap means the ATM point is an
     outlier versus the rest of the curve, not that Heston found new
     information the market didn't have.
  2. term_gap, the one number that DOES feed research/signal.py as a
     genuine third leg: forecast_vol_now minus sqrt(theta), Heston's
     fitted long-run/steady-state variance level. Unlike v0 (the ATM
     point), theta is informed by the WHOLE calibrated curve shape
     (skew, term structure if >=2 expiries were available) -- a
     cross-sectional read that doesn't share VRP's time-series
     assumptions or SAS's own-history-only assumption.
  3. A vol-of-vol RISK THROTTLE, not an alpha read: a low Feller ratio
     (2*kappa*theta / xi^2 close to or below 1) or a large xi means the
     calibrated variance process is fragile/gappy -- the CIR variance
     SDE is relying on the discretization floor rather than its own
     drift to stay non-negative (see models/heston.py's HestonParams
     docstring). That's a reason to size DOWN regardless of which
     direction the edge points, not a reason to trade more.

Bounded compute, deliberately: calibration is O(iterations x maturities)
FFT evaluations, cheap per call but not free, and this runs once per
agent cycle. `max_strikes_per_expiry` and `max_expiries` cap the problem
size fed to scipy.optimize.least_squares; see assess_heston_cross_check's
docstring for the defaults and why.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional

from data.market_data import ChainContract, implied_vol_for, to_option_quote
from pricing_engine.calibration.heston_calibration import calibrate_heston
from pricing_engine.market_data.contract import MarketSmilePoint, OptionType

# Feller ratio thresholds for the vol-of-vol regime label. These are a
# reasonable operating heuristic (Feller ratio > 1 keeps the CIR
# variance process technically non-negative without hitting the
# discretization floor; well below 1 means the fit is leaning hard on
# that floor), not a value re-derived or backtested here -- treat them
# as a starting point to tune per-symbol, same spirit as SAS's sas_scale.
FELLER_FRAGILE_BELOW = 1.0
FELLER_ELEVATED_BELOW = 3.0

VOL_OF_VOL_SIZE_MULTIPLIER = {"Stable": 1.0, "Elevated": 0.85, "Fragile": 0.65}

# Data-quality gate: a fitted smile with a worse RMSE than this (vol
# points), or built from too few strikes, is more likely to mislead than
# help -- excluded from the signal leg (heston_confident=False) even
# though a number still comes out the other end for the UI to show.
MAX_ACCEPTABLE_RMSE_VOL_PTS = 3.0
MIN_USABLE_POINTS = 5
ATM_DATA_QUALITY_GAP_VOL_PTS = 0.03  # 3 vol points


@dataclass
class HestonCrossCheck:
    available: bool
    n_points: int = 0
    n_expiries: int = 0
    calibration_success: bool = False
    rmse_vol_pts: Optional[float] = None
    heston_atm_iv: Optional[float] = None
    atm_gap_vol_pts: Optional[float] = None      # heston_atm_iv - live_atm_iv (data-quality read only)
    data_quality_ok: bool = False
    fair_term_vol: Optional[float] = None        # sqrt(theta) -- the signal-feeding number
    term_gap: Optional[float] = None              # forecast_vol_now - fair_term_vol
    feller_ratio: Optional[float] = None
    vol_of_vol_xi: Optional[float] = None
    vol_of_vol_regime: str = "Unavailable"
    size_multiplier: float = 1.0
    notes: list[str] = field(default_factory=list)


def _smile_points_for_heston(
    chain: list[ChainContract], spot: float, risk_free_rate: float,
    expiry: dt.date, as_of: dt.date, max_strikes_per_expiry: int, max_expiries: int,
) -> list[MarketSmilePoint]:
    """Build MarketSmilePoints from the live chain, capped so calibration
    stays cheap: the target expiry plus, if present, up to
    (max_expiries - 1) further expiries closest to it (for a genuine
    term-structure read, not just a single-slice fit); within each
    expiry, up to max_strikes_per_expiry strikes closest to spot (that
    is where quotes are most liquid/reliable and where the ATM
    consistency check actually needs precision)."""
    all_expiries = sorted({c.expiry for c in chain})
    if expiry not in all_expiries:
        return []
    others = sorted((e for e in all_expiries if e != expiry), key=lambda e: abs((e - expiry).days))
    use_expiries = [expiry] + others[: max(max_expiries - 1, 0)]

    points: list[MarketSmilePoint] = []
    for exp in use_expiries:
        same_expiry = [c for c in chain if c.expiry == exp and c.option_type == OptionType.CALL]
        same_expiry = sorted(same_expiry, key=lambda c: abs(c.strike - spot))[:max_strikes_per_expiry]
        for c in same_expiry:
            quote = to_option_quote(c, spot, risk_free_rate)
            iv = implied_vol_for(quote)
            if iv is None or iv <= 0:
                continue
            points.append(MarketSmilePoint(strike=c.strike, expiry_years=quote.expiry_years, implied_vol=iv))
    return points


def assess_heston_cross_check(
    chain: list[ChainContract], expiry: dt.date, spot: float, risk_free_rate: float,
    live_atm_iv: Optional[float], forecast_vol_now: float,
    as_of: Optional[dt.date] = None, max_strikes_per_expiry: int = 7, max_expiries: int = 2,
    max_nfev: int = 400,
) -> HestonCrossCheck:
    as_of = as_of or dt.date.today()
    notes: list[str] = []
    points = _smile_points_for_heston(chain, spot, risk_free_rate, expiry, as_of, max_strikes_per_expiry, max_expiries)
    if len(points) < MIN_USABLE_POINTS:
        return HestonCrossCheck(
            available=False, n_points=len(points),
            notes=[f"only {len(points)} usable smile points (<{MIN_USABLE_POINTS}); Heston cross-check skipped."],
        )

    try:
        result = calibrate_heston(points, spot=spot, r=risk_free_rate, q=0.0, max_nfev=max_nfev)
    except Exception as exc:  # noqa: BLE001 -- calibration failure shouldn't crash the cycle
        return HestonCrossCheck(
            available=False, n_points=len(points),
            notes=[f"Heston calibration raised {type(exc).__name__}: {exc}"],
        )

    n_expiries = len({p.expiry_years for p in points})
    heston_atm_iv = math.sqrt(max(result.params.v0, 0.0))
    fair_term_vol = math.sqrt(max(result.params.theta, 0.0))
    feller = result.feller_ratio
    xi = result.params.xi

    atm_gap = None
    if live_atm_iv is not None:
        atm_gap = heston_atm_iv - live_atm_iv
        if abs(atm_gap) > ATM_DATA_QUALITY_GAP_VOL_PTS:
            notes.append(
                f"Heston-fitted ATM IV ({heston_atm_iv:.1%}) diverges from the raw ATM print "
                f"({live_atm_iv:.1%}) by {atm_gap:+.1%} -- the smile's OTHER strikes disagree with "
                f"that single quote; treat the raw ATM print with extra caution this cycle."
            )

    data_quality_ok = (
        result.rmse_vol_pts <= MAX_ACCEPTABLE_RMSE_VOL_PTS
        and (atm_gap is None or abs(atm_gap) <= 2 * ATM_DATA_QUALITY_GAP_VOL_PTS)
    )
    if not result.success:
        # scipy hit max_nfev without its own convergence criterion firing --
        # informational only, NOT part of the data_quality_ok gate: RMSE is
        # the substantive fit-quality number (how close the calibrated
        # smile actually is to what was quoted), and a fit that lands well
        # under MAX_ACCEPTABLE_RMSE_VOL_PTS despite hitting the iteration
        # cap is still a usable fit, not a bad one -- gating on scipy's
        # internal flag as well as RMSE double-penalizes exactly the
        # common case of "found a good answer, ran out of budget before
        # declaring victory." (Verified empirically while building this:
        # the RMSE ~0.002 vol-pt case that motivated this comment scored
        # success=False here despite being a materially better fit than
        # the MAX_ACCEPTABLE_RMSE_VOL_PTS=3.0-vol-point tolerance requires.)
        notes.append(f"Heston optimizer hit max_nfev before its own convergence check fired ({result.message}); RMSE-based quality gate used instead.")
    if result.rmse_vol_pts > MAX_ACCEPTABLE_RMSE_VOL_PTS:
        notes.append(f"Heston fit RMSE {result.rmse_vol_pts:.1%} exceeds {MAX_ACCEPTABLE_RMSE_VOL_PTS:.0%} tolerance; excluded from signal.")

    if feller < FELLER_FRAGILE_BELOW:
        vol_of_vol_regime = "Fragile"
        notes.append(f"Feller ratio {feller:.2f} < {FELLER_FRAGILE_BELOW:.1f}: calibrated vol-of-vol regime is fragile -- sizing throttled.")
    elif feller < FELLER_ELEVATED_BELOW:
        vol_of_vol_regime = "Elevated"
    else:
        vol_of_vol_regime = "Stable"
    size_multiplier = VOL_OF_VOL_SIZE_MULTIPLIER[vol_of_vol_regime]

    term_gap = forecast_vol_now - fair_term_vol

    return HestonCrossCheck(
        available=True, n_points=len(points), n_expiries=n_expiries,
        calibration_success=result.success, rmse_vol_pts=result.rmse_vol_pts,
        heston_atm_iv=heston_atm_iv, atm_gap_vol_pts=atm_gap, data_quality_ok=data_quality_ok,
        fair_term_vol=fair_term_vol, term_gap=term_gap,
        feller_ratio=feller, vol_of_vol_xi=xi, vol_of_vol_regime=vol_of_vol_regime,
        size_multiplier=size_multiplier, notes=notes,
    )
