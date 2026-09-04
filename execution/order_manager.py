"""
execution/order_manager.py -- the only place in this codebase that
builds an Alpaca order payload or calls alpaca_client.submit_order.

Every function here returns the payload it WOULD submit; submit_orders()
is the single choke point that actually calls the network, and it is
gated by settings.dry_run. This split matters for the UI this backend
sits behind: the API can call the `build_*` functions and show the user
exactly what the agent is about to do, then let the user (or an
autonomous loop running with dry_run=False) decide whether it fires.

v2: straddle entries moved from plain MARKET orders to marketable LIMIT
orders (mid + a small buffer). A market order on a 2-3-wide indicative-
feed option quote pays the full spread with no ceiling; a limit at
mid + buffer still fills promptly against a reasonable book while
capping the worst-case slippage -- and the buffer widens when
research/toxicity.py's flow-toxicity read is elevated, i.e. exactly
when a resting order is more likely to get picked off. This is a real
cost reduction, not a cosmetic order-type swap: on a $5 straddle with a
$0.10 spread, a market order can pay the full $0.05/side vs. mid; a 2%
limit buffer caps that at $0.10 while still being marketable against a
normal book.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

from alpaca_client import AlpacaClient
from config import Settings
from strategy.legs import IronCondorLegs, StraddleLegs


@dataclass
class PlannedOrder:
    kind: str            # "straddle_buy" | "iron_condor" | "equity_hedge"
    payloads: list[dict]  # one or more Alpaca order payloads (>1 only for the two-leg straddle)
    description: str


def plan_straddle_buy(
    straddle: StraddleLegs, contracts: int, limit_price_buffer_bps: float = 200.0,
) -> PlannedOrder:
    """Two single-leg marketable-LIMIT BUY orders (Level 2 -- "buy a
    call" / "buy a put" are independently compliant; there's no reason
    to bundle them into an mleg order here since neither leg is short).
    limit_price_buffer_bps: how far above each leg's own mid the limit
    is set, in bps of that leg's mid price -- default 200bps (2%) is a
    reasonable resting buffer for a liquid weekly ATM option; agent/loop.py
    widens this when research/toxicity.py reads elevated flow toxicity."""
    payloads = []
    for contract in (straddle.call, straddle.put):
        leg_mid = contract.mid if contract.mid else straddle.price / 2.0
        limit = round(leg_mid * (1.0 + limit_price_buffer_bps / 10_000.0), 2)
        payloads.append({
            "symbol": contract.symbol,
            "qty": str(contracts),
            "side": "buy",
            "type": "limit",
            "limit_price": f"{limit:.2f}",
            "time_in_force": "day",
            "client_order_id": f"volagent-straddle-{uuid.uuid4().hex[:12]}",
        })
    return PlannedOrder(
        kind="straddle_buy", payloads=payloads,
        description=(
            f"BUY {contracts}x straddle on {straddle.call.underlying} {straddle.expiry.isoformat()} "
            f"{straddle.call.strike:g} strike (call {straddle.call.symbol} + put {straddle.put.symbol}), "
            f"~${straddle.price * 100 * contracts:,.0f} notional, "
            f"limit buffer {limit_price_buffer_bps:.0f}bps."
        ),
    )


def plan_iron_condor(
    condor: IronCondorLegs, limit_price: Optional[float] = None, limit_buffer_bps: float = 0.0,
) -> PlannedOrder:
    """One order_class='mleg' order, all four legs covered within the
    order (long wings protect the short strikes) -- compliant with
    Alpaca's Level 3 multi-leg rules, including the "no uncovered legs,
    even mid-fill" restriction, since MLeg orders fill as a single unit.
    limit_buffer_bps: shave this many bps off the theoretical net credit
    to make the resting limit more likely to fill (0 = the original v1
    behavior of resting exactly at net_credit); widened by agent/loop.py
    under elevated flow toxicity, same rationale as the straddle buffer."""
    limit = limit_price if limit_price is not None else round(condor.net_credit * (1.0 - limit_buffer_bps / 10_000.0), 2)
    legs = [
        {"symbol": condor.long_put.symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
        {"symbol": condor.short_put.symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
        {"symbol": condor.short_call.symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
        {"symbol": condor.long_call.symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
    ]
    payload = {
        "order_class": "mleg",
        "qty": str(condor.contracts),
        "type": "limit",
        "limit_price": f"{limit:.2f}",
        "time_in_force": "day",
        "legs": legs,
        "client_order_id": f"volagent-condor-{uuid.uuid4().hex[:12]}",
    }
    return PlannedOrder(
        kind="iron_condor", payloads=[payload],
        description=(
            f"SELL {condor.contracts}x iron condor on {condor.short_call.underlying} "
            f"{condor.expiry.isoformat()}: puts {condor.long_put.strike:g}/{condor.short_put.strike:g}, "
            f"calls {condor.short_call.strike:g}/{condor.long_call.strike:g}, "
            f"credit ~${limit * 100 * condor.contracts:,.0f}, "
            f"max loss ~${condor.max_loss_per_contract * condor.contracts:,.0f}."
        ),
    )


def plan_equity_hedge(symbol: str, trade_shares: float) -> Optional[PlannedOrder]:
    if abs(trade_shares) < 1:
        return None
    qty = int(round(abs(trade_shares)))
    side = "buy" if trade_shares > 0 else "sell"
    payload = {
        "symbol": symbol, "qty": str(qty), "side": side, "type": "market", "time_in_force": "day",
        "client_order_id": f"volagent-hedge-{uuid.uuid4().hex[:12]}",
    }
    return PlannedOrder(
        kind="equity_hedge", payloads=[payload],
        description=f"{side.upper()} {qty} shares {symbol} to rebalance the delta hedge.",
    )


def plan_position_close(
    symbol: str, qty: float, reason: str, pnl_pct: float = 0.0, dte: int = 0,
) -> Optional[PlannedOrder]:
    """Close an existing option position by selling (long) or buying to close (short).
    `reason` is one of 'take_profit', 'stop_loss', 'near_expiry'.
    `qty` is signed: positive for long positions (sell to close),
    negative for short positions (buy to close)."""
    if abs(qty) < 1:
        return None
    abs_qty = int(round(abs(qty)))
    # Long positions: sell to close. Short positions: buy to close.
    side = "sell" if qty > 0 else "buy"
    intent = "sell_to_close" if qty > 0 else "buy_to_close"
    pnl_str = f"{pnl_pct:+.1%}" if pnl_pct else ""
    reason_label = reason.replace("_", " ").upper()
    payload = {
        "symbol": symbol,
        "qty": str(abs_qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "position_intent": intent,
        "client_order_id": f"volagent-exit-{uuid.uuid4().hex[:12]}",
    }
    return PlannedOrder(
        kind=f"exit_{reason}",
        payloads=[payload],
        description=(
            f"[{reason_label}] {side.upper()} {abs_qty}x {symbol} to close "
            f"(P&L {pnl_str}, {dte} DTE)."
        ),
    )


def submit_orders(client: AlpacaClient, plan: PlannedOrder, settings: Settings) -> dict[str, Any]:
    if settings.dry_run:
        return {"submitted": False, "dry_run": True, "would_submit": plan.payloads, "description": plan.description}
    results = [client.submit_order(p) for p in plan.payloads]
    return {"submitted": True, "dry_run": False, "orders": results, "description": plan.description}


def submit_close_position(client: AlpacaClient, symbol: str, settings: Settings, reason: str) -> dict[str, Any]:
    """Alternative close path using Alpaca's DELETE /v2/positions/{symbol}
    instead of building a sell order. Simpler but still gated by dry_run."""
    if settings.dry_run:
        return {"submitted": False, "dry_run": True, "close_symbol": symbol, "reason": reason}
    result = client.close_position(symbol)
    return {"submitted": True, "dry_run": False, "close_result": result, "reason": reason}
