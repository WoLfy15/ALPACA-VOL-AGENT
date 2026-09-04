"""
strategy/sizing.py -- turns a research.signal.CompositeSignal into a
dollar risk budget, then into a contract count (long leg) or a maximum
structure loss (short-vol leg), via a fractional-Kelly step.

Ported from IITfinal.ipynb's position-sizing block:

    vol_edge  = sigma_forecast^2 - sigma_implied^2
    kelly     = 0.25 * (vol_edge / vol_variance), clipped [-15%, 15%]
    notional  = capital * kelly
    contracts = notional / (straddle_price * 100) * gamma_weight * regime_scale

generalized in two ways. First, `vol_edge / vol_variance` (a
signal-to-noise ratio that needs a rolling variance estimate) is replaced
by research.signal.CompositeSignal.composite_edge, which is already a
tanh-squashed, roughly [-1, 1] combination of two independent cheap/rich
readings (see research/signal.py's docstring) -- so the Kelly step here
is `kelly_fraction * composite_edge` rather than re-deriving a variance
normalization. Second, because Alpaca's paper options levels don't allow
naked short options (see strategy/legs.py), a negative edge produces a
*risk budget in dollars* for a defined-risk short-vol structure instead
of a negative contract count for a naked short straddle.

v2 additions, both fixing real gaps rather than adding decoration:

  1. INCREMENTAL sizing against existing exposure. v1 re-sized against
     the FULL Kelly budget every cycle with no memory of what was
     already on the book (strategy/portfolio.existing_vol_exposure_notional
     didn't exist yet). Looped at agent/loop.py's default 15-minute
     interval, that meant re-buying a fresh straddle on every single
     cycle for as long as the composite edge stayed positive -- which
     it typically does for many consecutive cycles, since regime/vol
     signals don't flip every 15 minutes. `decide_sizing` now takes the
     book's already-deployed notional and only sizes the REMAINING gap
     to target, which is what actually keeps the book converging to (not
     compounding past) the risk budget. This is a correctness fix, not
     a new alpha source -- it is nonetheless the single highest-impact
     change in this pass: unbounded re-entry both over-leverages the
     book and pays the bid/ask spread repeatedly for no new edge.
  2. Optional `cost_floor_vol_pts` gate: refuse to size a new long-vol
     entry when the live edge (in vol points, e.g. research.signal's
     vrp_raw) doesn't clear the estimated Leland transaction-cost drag
     of actually running the position
     (pricing_engine.risk.friction_hedging.leland_cost_bps_of_vega,
     previously vendored but never called anywhere). A trade that can't
     clear its own hedging cost has negative expected value by
     construction, independent of how confident the edge read is.
  3. Optional `extra_scale`: a further multiplicative throttle applied
     pre-clip, same slot REGIME_SIZE_SCALE already occupies -- used by
     agent/loop.py to fold in the Heston vol-of-vol fragility read and
     the order-flow toxicity read (research/heston_signal.py,
     research/toxicity.py) without this module needing to know anything
     about either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import RiskLimits

REGIME_SIZE_SCALE = {
    "Crash": 0.55, "Vol_Expansion": 0.70, "Trend": 0.85, "Range": 1.0,
}


@dataclass
class SizingDecision:
    kelly_fraction: float       # signed, clipped to +-risk.kelly_clip
    risk_budget_dollars: float  # |kelly_fraction| * capital, before Greek caps
    direction: str              # "long_vol" | "short_vol" | "flat"
    contracts: int              # long_vol: straddle contracts to buy. short_vol/flat: 0
    max_structure_loss: float   # short_vol: target max loss for the credit structure. else: 0.0
    capped_by: list[str] = field(default_factory=list)   # which caps actually bound (for the decision log / UI)
    incremental_budget_dollars: float = 0.0  # target budget net of what's already deployed (0 if nothing new to add)
    already_at_target: bool = False          # existing book already >= target budget this cycle -- no new entry


def kelly_fraction(
    composite_edge: float, regime_label: str, risk: RiskLimits, extra_scale: float = 1.0,
) -> float:
    scale = REGIME_SIZE_SCALE.get(regime_label, 1.0) * max(0.0, extra_scale)
    raw = risk.kelly_fraction * composite_edge * scale
    return max(-risk.kelly_clip, min(risk.kelly_clip, raw))


def size_long_straddle(
    kelly: float,
    capital: float,
    straddle_price: float,
    straddle_gamma: float,
    straddle_vega: float,
    risk: RiskLimits,
    deployed_long_vol_notional: float = 0.0,
) -> SizingDecision:
    """kelly > 0 branch: how many straddles (1 call + 1 put, 1x each) to
    buy. gamma/vega are PER STRADDLE (sum of the call and put Greeks),
    already in "per 1.0 change in underlying / vol" units matching
    pricing_engine.models.bsm's convention.

    `deployed_long_vol_notional`: premium value already committed to
    long-vol positions this cycle (strategy.portfolio.
    existing_vol_exposure_notional). Only the gap between the target
    budget and this is sized as NEW contracts -- see module docstring
    point 1."""
    capped_by: list[str] = []
    if kelly <= 0 or straddle_price <= 0:
        return SizingDecision(kelly, 0.0, "flat", 0, 0.0, capped_by)

    budget = capital * kelly
    incremental_budget = max(budget - max(deployed_long_vol_notional, 0.0), 0.0)
    if incremental_budget <= 0.0:
        return SizingDecision(
            kelly_fraction=kelly, risk_budget_dollars=budget, direction="long_vol",
            contracts=0, max_structure_loss=0.0, capped_by=[],
            incremental_budget_dollars=0.0, already_at_target=True,
        )

    raw_contracts = incremental_budget / (straddle_price * 100)

    gamma_risk = abs(raw_contracts * straddle_gamma * 100)
    max_gamma = risk.max_gamma_risk_pct * capital
    if gamma_risk > max_gamma and gamma_risk > 0:
        raw_contracts *= max_gamma / gamma_risk
        capped_by.append("gamma_limit")

    vega_risk = abs(raw_contracts * straddle_vega * 100)
    max_vega = risk.max_vega_risk_pct * capital
    if vega_risk > max_vega and vega_risk > 0:
        raw_contracts *= max_vega / vega_risk
        capped_by.append("vega_limit")

    contracts = int(max(0, min(raw_contracts, risk.max_contracts_per_leg)))
    if raw_contracts > risk.max_contracts_per_leg:
        capped_by.append("max_contracts_per_leg")

    return SizingDecision(
        kelly_fraction=kelly, risk_budget_dollars=budget, direction="long_vol",
        contracts=contracts, max_structure_loss=0.0, capped_by=capped_by,
        incremental_budget_dollars=incremental_budget, already_at_target=False,
    )


def size_short_vol_structure(
    kelly: float, capital: float, risk: RiskLimits, deployed_short_vol_risk: float = 0.0,
) -> SizingDecision:
    """kelly < 0 branch: |kelly| * capital becomes the target MAXIMUM LOSS
    of a defined-risk structure (iron condor / credit spread, see
    strategy/legs.py), not a premium-collected target -- max loss is the
    number that actually bounds tail risk on Alpaca's covered-legs-only
    multi-leg orders, so it's the number worth budgeting directly.

    `deployed_short_vol_risk`: max-loss-proxy already committed to short
    structures this cycle (see module docstring point 1 -- same
    incremental-budget logic as the long-vol branch)."""
    if kelly >= 0:
        return SizingDecision(kelly, 0.0, "flat", 0, 0.0, [])
    budget = capital * abs(kelly)
    incremental_budget = max(budget - max(deployed_short_vol_risk, 0.0), 0.0)
    if incremental_budget <= 0.0:
        return SizingDecision(
            kelly_fraction=kelly, risk_budget_dollars=budget, direction="short_vol",
            contracts=0, max_structure_loss=0.0, capped_by=[],
            incremental_budget_dollars=0.0, already_at_target=True,
        )
    return SizingDecision(
        kelly_fraction=kelly, risk_budget_dollars=budget, direction="short_vol",
        contracts=0, max_structure_loss=incremental_budget, capped_by=[],
        incremental_budget_dollars=incremental_budget, already_at_target=False,
    )


def decide_sizing(
    composite_edge: float, regime_label: str, capital: float,
    straddle_price: float, straddle_gamma: float, straddle_vega: float, risk: RiskLimits,
    deployed_long_vol_notional: float = 0.0,
    deployed_short_vol_risk: float = 0.0,
    extra_scale: float = 1.0,
    vrp_raw: "float | None" = None,
    cost_floor_vol_pts: "float | None" = None,
) -> SizingDecision:
    """vrp_raw / cost_floor_vol_pts: when BOTH are supplied, a long-vol
    entry is refused (forced flat) unless |vrp_raw| clears the estimated
    transaction-cost floor -- see module docstring point 2. Leaving
    either as None (the default) skips the check entirely, matching v1
    behavior exactly, so existing callers/tests are unaffected."""
    kelly = kelly_fraction(composite_edge, regime_label, risk, extra_scale=extra_scale)
    if math.isclose(kelly, 0.0, abs_tol=1e-4):
        return SizingDecision(kelly, 0.0, "flat", 0, 0.0, [])
    if kelly > 0:
        if vrp_raw is not None and cost_floor_vol_pts is not None and abs(vrp_raw) < cost_floor_vol_pts:
            return SizingDecision(
                kelly_fraction=kelly, risk_budget_dollars=0.0, direction="flat", contracts=0,
                max_structure_loss=0.0, capped_by=["cost_floor"],
            )
        return size_long_straddle(
            kelly, capital, straddle_price, straddle_gamma, straddle_vega, risk,
            deployed_long_vol_notional=deployed_long_vol_notional,
        )
    return size_short_vol_structure(kelly, capital, risk, deployed_short_vol_risk=deployed_short_vol_risk)
