"""
agent/exits.py -- position exit evaluation logic.

Every cycle, agent/loop.py calls `evaluate_exits()` to scan all open
option positions and check three exit triggers:

  1. TAKE PROFIT: the position's unrealized P&L exceeds a configurable
     threshold (default +50%). Locks in gains before mean-reversion
     erodes the edge.

  2. STOP LOSS: the position's unrealized P&L drops below a configurable
     threshold (default -35%). Cuts losses to preserve capital -- the
     drawdown kill switch in strategy/portfolio.py is a portfolio-level
     backstop, but position-level stops fire much earlier and more
     surgically.

  3. NEAR EXPIRY: the position's DTE drops to or below a configurable
     threshold (default 1 day). Avoids pin risk, gamma explosion on
     0-DTE, and potential exercise/assignment on short legs of iron
     condors.

P&L is calculated from Alpaca's own `avg_entry_price` vs `current_price`
on each position, so it survives agent restarts -- no in-memory state
needed.

Exit orders flow through the same execution/order_manager.py choke point
(plan_position_close -> submit_orders) and respect the same dry_run gate
as every other order in this project.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional

from config import RiskLimits
from strategy.portfolio import parse_occ_symbol

log = logging.getLogger("alpaca_vol_agent.exits")


@dataclass
class ExitDecision:
    """One decision to close a single option position."""
    symbol: str           # OCC option symbol
    qty: float            # signed: positive = long, negative = short
    reason: str           # "take_profit" | "stop_loss" | "near_expiry"
    pnl_pct: float        # unrealized P&L as a fraction (e.g. 0.50 = +50%)
    dte: int              # days to expiry
    avg_entry_price: float
    current_price: float
    market_value: float


def evaluate_exits(
    positions: list[dict],
    risk: RiskLimits,
    underlying_symbol: str,
    today: Optional[dt.date] = None,
) -> list[ExitDecision]:
    """Scan all positions from Alpaca's GET /v2/positions and return a
    list of ExitDecision for each option position that triggers an exit.

    `positions`: raw list[dict] from AlpacaClient.get_positions().
    `risk`: the RiskLimits with exit thresholds.
    `underlying_symbol`: only evaluate positions on this underlying.
    `today`: override for testing; defaults to date.today().
    """
    if today is None:
        today = dt.date.today()

    exits: list[ExitDecision] = []

    for pos in positions:
        symbol = pos.get("symbol", "")

        # Only process option positions on the target underlying
        parsed = parse_occ_symbol(symbol)
        if parsed is None:
            continue
        root, expiry, _otype, _strike = parsed
        if root != underlying_symbol:
            continue

        # Extract position data
        qty = float(pos.get("qty", 0))
        if qty == 0:
            continue

        avg_entry = float(pos.get("avg_entry_price", 0))
        current = float(pos.get("current_price", 0))
        market_value = float(pos.get("market_value", 0))

        # Calculate P&L percentage
        if avg_entry > 0 and current > 0:
            pnl_pct = (current - avg_entry) / avg_entry
        else:
            pnl_pct = 0.0

        # For short positions, P&L is inverted: we SOLD at avg_entry,
        # and the position losing value (current < avg_entry) is profit for us.
        if qty < 0:
            pnl_pct = -pnl_pct

        dte = max((expiry - today).days, 0)

        # --- Trigger 1: Near Expiry ---
        if dte <= risk.exit_dte_close_days:
            log.info(
                "EXIT near_expiry: %s qty=%g dte=%d (threshold=%d)",
                symbol, qty, dte, risk.exit_dte_close_days,
            )
            exits.append(ExitDecision(
                symbol=symbol, qty=qty, reason="near_expiry",
                pnl_pct=pnl_pct, dte=dte,
                avg_entry_price=avg_entry, current_price=current,
                market_value=market_value,
            ))
            continue  # Don't double-trigger

        # --- Trigger 2: Take Profit ---
        if pnl_pct >= risk.exit_take_profit_pct:
            log.info(
                "EXIT take_profit: %s qty=%g pnl=%.1f%% (threshold=%.0f%%)",
                symbol, qty, pnl_pct * 100, risk.exit_take_profit_pct * 100,
            )
            exits.append(ExitDecision(
                symbol=symbol, qty=qty, reason="take_profit",
                pnl_pct=pnl_pct, dte=dte,
                avg_entry_price=avg_entry, current_price=current,
                market_value=market_value,
            ))
            continue

        # --- Trigger 3: Stop Loss ---
        if pnl_pct <= -risk.exit_stop_loss_pct:
            log.info(
                "EXIT stop_loss: %s qty=%g pnl=%.1f%% (threshold=-%.0f%%)",
                symbol, qty, pnl_pct * 100, risk.exit_stop_loss_pct * 100,
            )
            exits.append(ExitDecision(
                symbol=symbol, qty=qty, reason="stop_loss",
                pnl_pct=pnl_pct, dte=dte,
                avg_entry_price=avg_entry, current_price=current,
                market_value=market_value,
            ))
            continue

    return exits
