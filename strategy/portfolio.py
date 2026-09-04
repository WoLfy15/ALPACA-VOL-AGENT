"""
strategy/portfolio.py -- the "portfolio overlay" track: reads whatever
is actually sitting in the Alpaca paper account right now (options AND
equity, however they got there -- this agent's own prior trades, or
positions the user placed by hand) and turns them into one set of
portfolio Greeks, a drawdown check, and a volatility-targeting scalar.

This is deliberately account-state-driven rather than trusting an
in-memory ledger: every cycle re-derives Greeks from Alpaca's own
positions endpoint, so a restart of the agent (or the FastAPI process)
never loses track of real risk.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Optional

from alpaca_client import AlpacaClient
from config import RiskLimits
from pricing_engine.calibration.iv_solver import solve_implied_vol, IVSolverError
from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()

_OCC_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ_symbol(symbol: str) -> Optional[tuple[str, dt.date, OptionType, float]]:
    """Standard OCC option symbol, e.g. AAPL240119C00100000 -> ('AAPL',
    date(2024,1,19), CALL, 100.0). Alpaca's option contract symbols
    follow this format; see docs.alpaca.markets/us/docs/options-trading
    for a worked example."""
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    root = m.group("root")
    expiry = dt.date(2000 + int(m.group("yy")), int(m.group("mm")), int(m.group("dd")))
    otype = OptionType.CALL if m.group("cp") == "C" else OptionType.PUT
    strike = int(m.group("strike")) / 1000.0
    return root, expiry, otype, strike


@dataclass
class PositionGreeks:
    symbol: str
    qty: float               # signed: long positive, short negative (contracts, not shares)
    market_value: float
    delta_shares: float       # qty * 100 * per-contract delta
    gamma_shares: float
    vega_dollars: float       # qty * 100 * per-contract vega (per 1.0 = 100% vol move)
    theta_dollars: float      # per year; divide by 365 for a daily figure in the UI


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    equity_shares_held: float          # underlying shares currently held (the hedge leg)
    option_positions: list[PositionGreeks] = field(default_factory=list)
    net_delta_shares: float = 0.0
    net_gamma_shares: float = 0.0
    net_vega_dollars: float = 0.0
    net_theta_dollars: float = 0.0

    @property
    def net_delta_incl_hedge(self) -> float:
        return self.net_delta_shares + self.equity_shares_held


def _option_position_greeks(pos: dict, spot: float, risk_free_rate: float) -> Optional[PositionGreeks]:
    parsed = parse_occ_symbol(pos["symbol"])
    if parsed is None:
        return None
    _root, expiry, otype, strike = parsed
    expiry_years = max((expiry - dt.date.today()).days, 0) / 365.0
    if expiry_years <= 0:
        return None

    qty = float(pos["qty"])
    side_sign = 1.0 if pos.get("side", "long") == "long" else -1.0
    signed_qty = abs(qty) * side_sign if float(pos.get("qty", 0)) >= 0 else float(pos["qty"])
    # Alpaca returns qty already signed for short equity, but option short
    # positions from a covered-call/spread can come through with a
    # positive qty and side="short" depending on endpoint version -- use
    # whichever sign is present, falling back to side.
    signed_qty = float(pos["qty"]) if float(pos["qty"]) != 0 else signed_qty

    current_price = float(pos.get("current_price") or pos.get("avg_entry_price") or 0.0)
    quote = OptionQuote(
        underlying=pos.get("underlying_symbol", pos["symbol"][:6]),
        spot=spot, strike=strike, expiry_years=expiry_years, option_type=otype,
        risk_free_rate=risk_free_rate, market_price=current_price if current_price > 0 else None,
    )
    if quote.market_price is None:
        return None
    try:
        iv = solve_implied_vol(_bsm, quote, quote.market_price).implied_vol
    except IVSolverError:
        return None

    greeks = _bsm.greeks(quote, iv)
    mult = signed_qty * 100
    return PositionGreeks(
        symbol=pos["symbol"], qty=signed_qty, market_value=float(pos.get("market_value", 0.0)),
        delta_shares=mult * greeks["delta"], gamma_shares=mult * greeks["gamma"],
        vega_dollars=mult * greeks["vega"], theta_dollars=mult * greeks["theta"],
    )


def build_portfolio_snapshot(client: AlpacaClient, underlying_symbol: str, spot: float, risk_free_rate: float) -> PortfolioSnapshot:
    account = client.get_account()
    positions = client.get_positions()

    equity_shares_held = 0.0
    option_greeks: list[PositionGreeks] = []
    for pos in positions:
        if pos.get("asset_class") == "us_equity" and pos.get("symbol") == underlying_symbol:
            equity_shares_held += float(pos.get("qty", 0.0))
        elif pos.get("asset_class") in ("us_option",) or (len(pos.get("symbol", "")) > 10 and parse_occ_symbol(pos["symbol"])):
            parsed = parse_occ_symbol(pos["symbol"])
            if parsed and parsed[0] == underlying_symbol:
                pg = _option_position_greeks(pos, spot, risk_free_rate)
                if pg:
                    option_greeks.append(pg)

    snap = PortfolioSnapshot(
        equity=float(account["equity"]), cash=float(account["cash"]),
        equity_shares_held=equity_shares_held, option_positions=option_greeks,
    )
    snap.net_delta_shares = sum(p.delta_shares for p in option_greeks)
    snap.net_gamma_shares = sum(p.gamma_shares for p in option_greeks)
    snap.net_vega_dollars = sum(p.vega_dollars for p in option_greeks)
    snap.net_theta_dollars = sum(p.theta_dollars for p in option_greeks)
    return snap


SHORT_VOL_RISK_MULTIPLIER = 6.0  # see existing_vol_exposure_notional's docstring


def existing_vol_exposure_notional(option_positions: list[PositionGreeks]) -> tuple[float, float]:
    """(deployed_long_vol_notional, deployed_short_vol_risk_proxy).

    Added so strategy/sizing.py can size an entry against the RISK BUDGET
    STILL REMAINING rather than the full budget every cycle -- see
    decide_sizing's docstring for why this matters. This agent only ever
    puts on long straddles (qty > 0, both legs) or defined-risk short
    iron condors (qty < 0 short strikes, paired long wings -- see
    strategy/legs.py), so a simple sign split on existing positions is a
    reasonable, cheap classifier without needing to reconstruct which
    positions belong to which historical structure:

      long_vol_notional  = sum(market_value) over qty > 0 legs -- the
        premium already paid for straddle-like long exposure. Exact and
        apples-to-apples with strategy/sizing.py's long-vol target
        budget, which is ALSO a premium-dollar figure.

      short_vol_risk_proxy = SHORT_VOL_RISK_MULTIPLIER * sum(|market_value|)
        over qty < 0 legs. NOT apples-to-apples on its own: the target
        short-vol budget is a MAX-LOSS figure (strategy/sizing.py sizes
        a credit structure by its defined risk, not its collected
        premium), while a short leg's own market_value is just the
        premium at risk on THAT leg -- for a real iron condor the max
        loss is (wing width - net credit), typically several times the
        credit alone. Using raw premium unscaled was checked against a
        synthetic existing position while building this and materially
        under-counted deployed risk (a fully-sized 20-lot condor's
        short legs priced at single-digit-dollar premiums against a
        ~$45k max-loss budget) -- meaning the incremental-sizing check
        would rarely bind for the short-vol path in practice, even
        though the book was already fully sized. SHORT_VOL_RISK_MULTIPLIER
        is a fixed heuristic scalar (typical wing-width-to-credit ratio
        for the delta targets this project defaults to,
        config.Settings.short_leg_target_delta=0.16 /
        wing_target_delta=0.05), not a re-derivation of each open
        structure's actual width -- tune it per symbol/strike spacing if
        the chain's typical condors run wider or narrower than that."""
    long_notional = sum(p.market_value for p in option_positions if p.qty > 0)
    short_notional = SHORT_VOL_RISK_MULTIPLIER * sum(abs(p.market_value) for p in option_positions if p.qty < 0)
    return float(long_notional), float(short_notional)


def check_drawdown_kill_switch(equity_now: float, equity_high_water_mark: float, risk: RiskLimits) -> tuple[bool, float]:
    """Mirrors IITfinal.ipynb's hard stop (25% drawdown -> flatten). Returns
    (breached, drawdown_pct). Caller (agent/loop.py) owns persisting the
    high-water mark across cycles and deciding what "flatten" means."""
    if equity_high_water_mark <= 0:
        return False, 0.0
    drawdown = (equity_now - equity_high_water_mark) / equity_high_water_mark
    return drawdown < -risk.max_drawdown_pct, drawdown


def vol_target_scale(realized_portfolio_vol: Optional[float], risk: RiskLimits) -> float:
    """Scales new entries down if the book's recent realized vol is
    already running hot relative to the target -- same role as
    IITfinal.ipynb's `vol_scale = target_vol / realized_vol`. Returns 1.0
    (no scaling) until there's enough portfolio return history to
    estimate realized_portfolio_vol."""
    if not realized_portfolio_vol or realized_portfolio_vol <= 0:
        return 1.0
    return min(risk.target_portfolio_vol / realized_portfolio_vol, 1.2)
