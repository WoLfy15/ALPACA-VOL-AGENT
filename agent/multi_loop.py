"""
agent/multi_loop.py -- runs the full agent cycle across every symbol in
the configured watchlist, collecting results per-symbol and producing an
aggregate summary.

Each symbol gets its own independent pass through agent/loop.run_cycle():
fetch bars -> regime detect -> vol forecast -> signal -> sizing -> exit
evaluation -> order submission. A failure on one symbol (e.g. illiquid
chain, insufficient bar history) is caught and logged without killing
the scan for the remaining symbols.

The account-level state (equity, drawdown HWM, equity curve) is shared
across all symbols since they trade from the same Alpaca paper account.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from agent.loop import AgentRunResult, run_cycle
from alpaca_client import AlpacaClient
from config import Settings

log = logging.getLogger("alpaca_vol_agent.multi_loop")


@dataclass
class MultiCycleResult:
    timestamp_utc: str
    watchlist: list[str]
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _make_symbol_settings(base: Settings, symbol: str) -> Settings:
    """Clone base settings with a different symbol. Since Settings is
    frozen (dataclass(frozen=True)), we rebuild from scratch."""
    return Settings(
        api_key=base.api_key, api_secret=base.api_secret,
        data_feed=base.data_feed, stock_feed=base.stock_feed,
        symbol=symbol, watchlist=base.watchlist,
        target_dte_days=base.target_dte_days, dte_tolerance_days=base.dte_tolerance_days,
        short_leg_target_delta=base.short_leg_target_delta, wing_target_delta=base.wing_target_delta,
        dry_run=base.dry_run,
        i_understand_this_is_live=base.i_understand_this_is_live,
        enable_credit_spreads=base.enable_credit_spreads,
        enable_equity_hedge=base.enable_equity_hedge,
        min_open_interest=base.min_open_interest,
        enable_heston_cross_check=base.enable_heston_cross_check,
        enable_toxicity_gate=base.enable_toxicity_gate,
        enable_cost_floor=base.enable_cost_floor,
        enable_auto_exit=base.enable_auto_exit,
        risk=base.risk, hedge=base.hedge,
    )


def run_multi_cycle(
    settings: Settings,
    client: Optional[AlpacaClient] = None,
    symbols: Optional[list[str]] = None,
) -> MultiCycleResult:
    """Run one agent cycle per symbol in the watchlist.

    `symbols`: override the watchlist from settings (used by the API
    when the caller passes a subset). Defaults to settings.watchlist.
    """
    settings.validate()
    client = client or AlpacaClient(settings=settings)
    watchlist = symbols or settings.watchlist
    timestamp = dt.datetime.utcnow().isoformat()

    multi = MultiCycleResult(timestamp_utc=timestamp, watchlist=watchlist)

    long_count = 0
    short_count = 0
    flat_count = 0
    exit_count = 0
    order_count = 0

    for sym in watchlist:
        log.info("=== Multi-cycle: running %s (%d/%d) ===", sym, watchlist.index(sym) + 1, len(watchlist))
        try:
            sym_settings = _make_symbol_settings(settings, sym)
            result = run_cycle(sym_settings, client=client)
            result_dict = asdict(result)
            multi.results.append(result_dict)

            # Tally for summary
            direction = result.direction
            if direction == "long_vol":
                long_count += 1
            elif direction == "short_vol":
                short_count += 1
            else:
                flat_count += 1

            exit_count += len(result.exit_decisions)
            order_count += len(result.planned_orders)

        except Exception as exc:  # noqa: BLE001
            log.warning("Multi-cycle: %s FAILED: %s", sym, exc, exc_info=True)
            multi.errors.append({
                "symbol": sym,
                "error": f"{type(exc).__name__}: {exc}",
            })

    multi.summary = {
        "total_symbols": len(watchlist),
        "succeeded": len(multi.results),
        "failed": len(multi.errors),
        "long_vol": long_count,
        "short_vol": short_count,
        "flat": flat_count,
        "total_exits": exit_count,
        "total_orders": order_count,
    }

    log.info(
        "Multi-cycle complete: %d/%d succeeded, %d long, %d short, %d flat, %d exits, %d orders",
        len(multi.results), len(watchlist), long_count, short_count, flat_count, exit_count, order_count,
    )

    return multi
