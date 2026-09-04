"""
strategy/hedging.py -- decides whether, and how much, to trade the
underlying to keep the options book close to delta-neutral.

Two pieces ported from IITfinal.ipynb, both now backed by
pricing_engine.risk.friction_hedging instead of being charged at zero
cost:

  1. Min-variance hedge ratio: h* = Cov(option_returns, stock_returns) /
     Var(stock_returns), over a rolling window, falling back to the
     book's own BSM delta when there isn't yet enough return history
     (a fresh position, or the very first cycle after startup).
  2. Whalley-Wilmott (1997) no-trade band: only actually submit the
     hedge trade when the portfolio's net delta has drifted outside a
     transaction-cost-aware band around zero, rather than rehedging to
     exact zero delta every cycle -- see
     pricing_engine/risk/friction_hedging.py's docstring for the
     derivation and the explicit caveat about the leading constant
     being a textbook convention, not re-derived here.

Units note, stated plainly because it matters for correctness: every
Greek in this module is a PORTFOLIO-level Greek already scaled by
contracts x 100 (i.e. "shares of delta exposure per $1 move in the
underlying"), not a per-contract Greek. whalley_wilmott_band is applied
directly to portfolio gamma, so its output band is in the same
"delta-shares" units as portfolio delta -- consistent, but a convention
this module owns, not something friction_hedging.py enforces itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import HedgeParams
from pricing_engine.risk.friction_hedging import whalley_wilmott_band

REGIME_HEDGE_MULTIPLIER = {"Range": 1.0, "Trend": 0.8, "Vol_Expansion": 1.1, "Crash": 1.3}


@dataclass
class HedgeDecision:
    should_trade: bool
    target_shares: float
    current_shares: float
    trade_shares: float
    band_half_width: float
    hedge_ratio_used: float
    reason: str


def min_variance_hedge_ratio(
    option_pnl_returns: "np.ndarray | list[float]",
    stock_returns: "np.ndarray | list[float]",
    fallback_delta: float,
    min_obs: int = 20,
) -> float:
    opt = np.asarray(option_pnl_returns, dtype=float)
    stk = np.asarray(stock_returns, dtype=float)
    n = min(len(opt), len(stk))
    if n < min_obs:
        return fallback_delta
    opt, stk = opt[-n:], stk[-n:]
    var = np.var(stk)
    if var <= 1e-12:
        return fallback_delta
    cov = np.cov(opt, stk)[0, 1]
    return float(cov / var)


def decide_hedge(
    portfolio_option_delta_shares: float,
    portfolio_gamma_shares: float,
    current_hedge_shares: float,
    regime_label: str,
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    params: HedgeParams,
    hedge_ratio_override: Optional[float] = None,
) -> HedgeDecision:
    """portfolio_option_delta_shares: sum over all option positions of
    (contract_delta * contracts * 100) -- i.e. how many shares of the
    underlying the options book is currently equivalent to.
    portfolio_gamma_shares: same convention, for gamma.
    current_hedge_shares: the equity position already held for hedging
    (signed; short is negative)."""
    hedge_ratio = hedge_ratio_override if hedge_ratio_override is not None else 1.0
    regime_mult = REGIME_HEDGE_MULTIPLIER.get(regime_label, 1.0)

    ideal_target = -regime_mult * hedge_ratio * portfolio_option_delta_shares
    k = params.transaction_cost_bps / 10_000.0
    band = whalley_wilmott_band(
        spot=spot, gamma=portfolio_gamma_shares, k=k,
        risk_aversion=params.risk_aversion, r=risk_free_rate, time_to_expiry=max(time_to_expiry, 1 / 365),
    )

    drift = (current_hedge_shares - (-portfolio_option_delta_shares))  # net portfolio delta, hedge included
    should_trade = abs(drift) > band

    if not should_trade:
        return HedgeDecision(
            should_trade=False, target_shares=current_hedge_shares, current_shares=current_hedge_shares,
            trade_shares=0.0, band_half_width=band, hedge_ratio_used=hedge_ratio,
            reason=f"net delta drift {drift:+.1f} shares within no-trade band (+/-{band:.1f}); holding.",
        )

    trade = ideal_target - current_hedge_shares
    return HedgeDecision(
        should_trade=True, target_shares=ideal_target, current_shares=current_hedge_shares,
        trade_shares=trade, band_half_width=band, hedge_ratio_used=hedge_ratio,
        reason=(
            f"net delta drift {drift:+.1f} shares exceeds no-trade band (+/-{band:.1f}) "
            f"under {regime_label} regime (hedge multiplier {regime_mult}x); "
            f"{'buying' if trade > 0 else 'selling'} {abs(trade):.0f} shares to retarget."
        ),
    )
