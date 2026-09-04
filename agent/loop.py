"""
agent/loop.py -- the orchestration layer. `run_cycle()` is the one
function that ties every other module together into a single pass:

  fetch (Alpaca account, chain, bars, minute bars)
    -> analyze (regime, vol forecast, SAS, Heston cross-check, flow
       toxicity, composite signal)
    -> risk-check (drawdown kill switch, portfolio Greeks, existing
       exposure, vol-of-vol + toxicity throttles, transaction-cost floor)
    -> decide (Kelly sizing -> concrete legs -> hedge trade)
    -> explain (human-readable rationale for every decision)
    -> execute (only if not dry_run)

It returns a single AgentRunResult dataclass with everything the FastAPI
layer (api/server.py) needs to render a UI: the raw signal numbers, the
regime, every planned order (submitted or not), and the hedge decision.
Nothing here talks to the network except through alpaca_client.py, and
nothing submits an order except through execution/order_manager.py.

v2 wires four things into this pass that were previously vendored/built
but never actually called from here -- see each module's own docstring
for the reasoning, this file just plumbs them together:

  - strategy/portfolio.existing_vol_exposure_notional + the incremental
    sizing logic it feeds (strategy/sizing.py): v1 re-sized a FRESH
    entry against the full Kelly budget every single cycle with no
    memory of what was already on the book, which at the default
    15-minute loop interval meant re-buying more straddles every cycle
    for as long as the edge stayed positive. This is the highest-impact
    fix in this pass.
  - research/heston_signal.py: a data-quality cross-check on the raw
    ATM IV print plus a vol-of-vol/Feller-ratio risk throttle, both
    derived from calibrating the vendored Heston model to the live
    chain -- and, when the fit is good, a genuine third signal leg
    (the smile's implied long-run vol level vs. the GARCH/HAR forecast).
  - research/toxicity.py: a VPIN-style order-flow toxicity read off
    recent minute bars, throttling size and widening execution limit
    buffers under one-sided flow -- an execution-quality control, not
    a new alpha leg (see that module's docstring for why).
  - agent/state.record_equity + compute_risk_metrics: the account
    equity curve is now actually persisted every cycle, so live
    Sharpe/Sortino/drawdown/win-rate are computable at all -- v1 only
    ever kept a running high-water mark, nothing time-series shaped.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from agent.exits import ExitDecision, evaluate_exits
from agent.state import compute_risk_metrics, load_state, record_decision, record_equity, save_state
from alpaca_client import AlpacaClient
from pricing_engine.calibration.iv_solver import IVSolverError
from pricing_engine.risk.friction_hedging import leland_cost_bps_of_vega
from config import Settings
from data.market_data import (
    atm_implied_vol, daily_log_returns_from_bars, fetch_chain, implied_vol_for, to_option_quote,
)
from execution.order_manager import plan_equity_hedge, plan_iron_condor, plan_position_close, plan_straddle_buy, submit_close_position, submit_orders
from pricing_engine.market_data.contract import OptionType
from pricing_engine.risk.sas import strike_adjusted_spread
from research.heston_signal import HestonCrossCheck, assess_heston_cross_check
from research.regime import detect_regime
from research.signal import build_signal
from research.toxicity import FlowToxicity, assess_flow_toxicity
from research.vol_forecast import estimate_hurst, forecast_volatility
from strategy.hedging import decide_hedge, min_variance_hedge_ratio
from strategy.legs import build_iron_condor, build_straddle, select_expiry
from strategy.portfolio import build_portfolio_snapshot, check_drawdown_kill_switch, existing_vol_exposure_notional
from strategy.sizing import REGIME_SIZE_SCALE, decide_sizing, kelly_fraction as _compute_kelly

log = logging.getLogger("alpaca_vol_agent.loop")

RISK_FREE_RATE_FALLBACK = 0.045
BAR_HISTORY_DAYS = 400
TOXICITY_LOOKBACK_DAYS = 7
STRADDLE_BASE_LIMIT_BUFFER_BPS = 200.0   # 2% above mid -- see execution/order_manager.py's docstring


@dataclass
class AgentRunResult:
    timestamp_utc: str
    symbol: str
    spot: float
    regime_label: str
    regime_confidence: float
    gamma_regime: str
    forecast_vol: float
    live_atm_iv: Optional[float]
    vrp_raw: Optional[float]
    vrp_z: Optional[float]
    sas_mean: Optional[float]
    composite_edge: Optional[float]
    signal_rationale: str
    kelly_fraction: Optional[float]
    direction: str
    portfolio_delta_shares: float
    portfolio_gamma_shares: float
    portfolio_vega_dollars: float
    portfolio_theta_dollars: float
    equity: float
    drawdown_pct: float
    kill_switch: bool
    planned_orders: list[dict[str, Any]] = field(default_factory=list)
    hedge_decision: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)
    # -- v2 additions --
    heston_cross_check: Optional[dict[str, Any]] = None
    flow_toxicity: Optional[dict[str, Any]] = None
    vol_path_hurst: Optional[float] = None
    deployed_long_vol_notional: float = 0.0
    deployed_short_vol_risk: float = 0.0
    extra_size_multiplier: float = 1.0
    incremental_budget_dollars: float = 0.0
    already_at_target: bool = False
    risk_metrics: dict[str, Optional[float]] = field(default_factory=dict)
    # -- sizing detail (what the agent WOULD deploy; non-zero even with no open positions) --
    risk_budget_dollars: float = 0.0           # |kelly| * capital before Greek caps
    contracts_planned: int = 0                 # straddle contracts that would be bought (long_vol)
    max_structure_loss: float = 0.0            # max risk budget for short-vol structure
    capped_by: list[str] = field(default_factory=list)  # which risk caps actually fired
    regime_size_scale: float = 1.0             # REGIME_SIZE_SCALE applied this cycle
    # -- auto-exit decisions (v2) --
    exit_decisions: list[dict[str, Any]] = field(default_factory=list)


def _bars_to_frame(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t").sort_index()
    df = df.rename(columns={"c": "close"})
    return df


def run_cycle(settings: Settings, client: Optional[AlpacaClient] = None) -> AgentRunResult:
    settings.validate()
    client = client or AlpacaClient(settings=settings)
    state = load_state()
    notes: list[str] = []

    account = client.get_account()
    equity = float(account["equity"])
    if state.equity_high_water_mark <= 0:
        state.equity_high_water_mark = equity
    state.equity_high_water_mark = max(state.equity_high_water_mark, equity)
    kill_switch, drawdown_pct = check_drawdown_kill_switch(equity, state.equity_high_water_mark, settings.risk)
    if kill_switch:
        notes.append(
            f"DRAWDOWN KILL SWITCH TRIPPED: {drawdown_pct:.1%} below high-water mark "
            f"(limit {settings.risk.max_drawdown_pct:.0%}). No new entries this cycle."
        )

    spot = client.get_mid_price(settings.symbol)

    end = dt.date.today()
    start = end - dt.timedelta(days=BAR_HISTORY_DAYS)
    bars = client.get_stock_bars(settings.symbol, timeframe="1Day", start=start.isoformat(), end=end.isoformat())
    df = _bars_to_frame(bars)
    if len(df) < 120:
        raise RuntimeError(
            f"only {len(df)} daily bars returned for {settings.symbol} over the last "
            f"{BAR_HISTORY_DAYS} days -- not enough history for the HMM/GARCH fit. "
            "Try a more liquid symbol or widen BAR_HISTORY_DAYS."
        )

    regime = detect_regime(df["close"])
    feat_log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    realized_vol = feat_log_returns.rolling(21).std() * np.sqrt(252)
    realized_vol = realized_vol.dropna()
    aligned_returns = feat_log_returns.loc[realized_vol.index]
    vol_fc = forecast_volatility(aligned_returns, realized_vol)

    # Diagnostic only -- see research/vol_forecast.estimate_hurst's
    # docstring for why this deliberately does NOT feed the forecast
    # blend or sizing. Surfaced in the UI as context, nothing more.
    vol_path_hurst: Optional[float] = None
    with np.errstate(divide="ignore"):
        log_rv = np.log(realized_vol.replace(0, np.nan)).dropna()
    if len(log_rv) >= 40:
        vol_path_hurst = estimate_hurst(log_rv.values)

    # fetch the live chain once, in the DTE window we actually care about
    dte_search_end = end + dt.timedelta(days=settings.target_dte_days + settings.dte_tolerance_days + 5)
    chain = fetch_chain(
        client, settings.symbol,
        expiration_date_gte=end.isoformat(),
        expiration_date_lte=dte_search_end.isoformat(),
        min_open_interest=settings.min_open_interest,
    )
    available_expiries = sorted({c.expiry for c in chain})
    log.info(
        "Chain fetch: %d contracts, expiries=%s (searched %s to %s, target_dte=%d, tol=%d)",
        len(chain), [str(e) for e in available_expiries],
        end.isoformat(), dte_search_end.isoformat(),
        settings.target_dte_days, settings.dte_tolerance_days,
    )
    expiry = select_expiry(chain, settings.target_dte_days, settings.dte_tolerance_days, as_of=end)
    if expiry is None:
        if not available_expiries:
            raise RuntimeError(
                f"Chain returned 0 contracts for {settings.symbol} between {end} and {dte_search_end}. "
                "Check that the paper account has options data access and that MIN_OPEN_INTEREST "
                "is not filtering everything out."
            )
        # Fallback: use the nearest available expiry and warn. This is common on
        # the indicative feed where only near-term contracts have live snapshots.
        expiry = min(available_expiries, key=lambda e: abs((e - end).days - settings.target_dte_days))
        actual_dte = (expiry - end).days
        notes.append(
            f"DTE fallback: no expiry within {settings.dte_tolerance_days} days of "
            f"{settings.target_dte_days}-day target. Using nearest available: "
            f"{expiry} ({actual_dte} DTE). Available: {[str(e) for e in available_expiries]}."
        )
        log.warning("DTE fallback: using %s (%d DTE) -- indicative feed may only have near-term snapshots.", expiry, actual_dte)

    risk_free_rate = RISK_FREE_RATE_FALLBACK
    live_iv = atm_implied_vol([c for c in chain if c.expiry == expiry], spot, risk_free_rate)

    sas_values: list[float] = []
    daily_returns = daily_log_returns_from_bars(bars)
    if live_iv and len(daily_returns) > 30:
        same_expiry = [c for c in chain if c.expiry == expiry]
        atm_strike = min({c.strike for c in same_expiry}, key=lambda k: abs(k - spot), default=None)
        near_strikes = sorted({c.strike for c in same_expiry}, key=lambda k: abs(k - spot))[:5]
        for c in same_expiry:
            if c.strike not in near_strikes or c.option_type != OptionType.CALL:
                continue
            quote = to_option_quote(c, spot, risk_free_rate)
            iv = implied_vol_for(quote)
            if iv is None:
                continue
            try:
                sas = strike_adjusted_spread(quote, iv, daily_returns)
                sas_values.append(sas)
            except (ValueError, ZeroDivisionError, IVSolverError, Exception):
                # 0-DTE or deep-ITM quotes often produce stale/crossed prices
                # that can't be bracketed by the IV solver -- skip silently.
                continue
    else:
        notes.append("SAS leg skipped: insufficient live IV or return history this cycle.")

    if live_iv is None:
        notes.append(f"No live ATM IV available for {settings.symbol} {expiry}; signal falls back to VRP-only.")
        live_iv = float(realized_vol.iloc[-1]) * 1.15  # last-resort proxy, clearly flagged

    # --- Heston smile-consistency cross-check + vol-of-vol throttle ---
    heston: Optional[HestonCrossCheck] = None
    if settings.enable_heston_cross_check:
        try:
            heston = assess_heston_cross_check(
                chain, expiry, spot, risk_free_rate,
                live_atm_iv=live_iv, forecast_vol_now=vol_fc.blended_next,
            )
            if not heston.available:
                notes.extend(f"Heston: {n}" for n in heston.notes)
            elif heston.notes:
                notes.extend(f"Heston: {n}" for n in heston.notes)
        except Exception as exc:  # noqa: BLE001 -- a bad calibration cycle shouldn't kill the whole run
            notes.append(f"Heston cross-check failed unexpectedly: {type(exc).__name__}: {exc}")
            heston = None

    # --- flow-toxicity read off recent minute bars ---
    toxicity: Optional[FlowToxicity] = None
    if settings.enable_toxicity_gate:
        try:
            tox_start = end - dt.timedelta(days=TOXICITY_LOOKBACK_DAYS)
            minute_bars = client.get_stock_bars(
                settings.symbol, timeframe="1Min", start=tox_start.isoformat(), end=end.isoformat(), limit=3000,
            )
            toxicity = assess_flow_toxicity(minute_bars)
            if not toxicity.available:
                notes.extend(f"Flow toxicity: {n}" for n in toxicity.notes)
            elif toxicity.regime in ("Elevated", "Toxic"):
                notes.append(
                    f"Flow toxicity: {toxicity.regime} (VPIN percentile {toxicity.percentile_vs_history:.0%} "
                    f"of trailing history) -- sizing throttled x{toxicity.size_multiplier:.2f}, "
                    f"execution limits widened +{toxicity.extra_limit_buffer_bps:.0f}bps."
                )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Flow-toxicity read failed unexpectedly: {type(exc).__name__}: {exc}")
            toxicity = None

    signal = build_signal(
        regime_label=regime.current_label,
        forecast_vol_now=vol_fc.blended_next,
        live_atm_iv=live_iv,
        forecast_vol_history=vol_fc.blended.values,
        realized_vol_history=realized_vol.loc[vol_fc.blended.index].values,
        sas_values=sas_values or None,
        heston_term_gap=heston.term_gap if (heston and heston.available) else None,
        heston_confident=bool(heston.data_quality_ok) if heston else True,
    )

    portfolio = build_portfolio_snapshot(client, settings.symbol, spot, risk_free_rate)
    deployed_long_vol_notional, deployed_short_vol_risk = existing_vol_exposure_notional(portfolio.option_positions)

    extra_size_multiplier = 1.0
    if heston and heston.available:
        extra_size_multiplier *= heston.size_multiplier
    if toxicity and toxicity.available:
        extra_size_multiplier *= toxicity.size_multiplier

    cost_floor_vol_pts: Optional[float] = None
    if settings.enable_cost_floor:
        cost_floor_vol_pts = leland_cost_bps_of_vega(
            sigma=max(live_iv, 1e-3), k=settings.hedge.transaction_cost_bps / 10_000.0, dt=1 / 252,
        ) / 10_000.0

    planned_orders: list[dict[str, Any]] = []
    exit_decisions_raw: list[ExitDecision] = []

    # --- Auto-exit: evaluate open positions for stop-loss / take-profit / near-expiry ---
    if settings.enable_auto_exit:
        try:
            raw_positions = client.get_positions()
            exit_decisions_raw = evaluate_exits(raw_positions, settings.risk, settings.symbol)
            for ed in exit_decisions_raw:
                # Use DELETE /v2/positions/{symbol} — Alpaca's preferred close method.
                # This avoids the HTTP 403 "no available quote" error that market orders
                # hit when an option has no live bid (illiquid / near-expiry).
                result_dict = submit_close_position(client, ed.symbol, settings, ed.reason)
                desc = (
                    f"[{ed.reason.replace('_',' ').upper()}] CLOSE {ed.qty:g}x {ed.symbol} "
                    f"(P&L {ed.pnl_pct:+.1%}, {ed.dte} DTE)"
                )
                planned_orders.append({
                    "kind": f"exit_{ed.reason}",
                    "description": desc,
                    **result_dict,
                })
                notes.append(
                    f"Auto-exit [{ed.reason.replace('_',' ').upper()}]: {ed.symbol} "
                    f"qty={ed.qty:g} P&L={ed.pnl_pct:+.1%} DTE={ed.dte} "
                    f"({'SUBMITTED' if not settings.dry_run else 'DRY RUN'})"
                )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Auto-exit evaluation failed: {type(exc).__name__}: {exc}")
            log.warning("Auto-exit evaluation failed: %s", exc, exc_info=True)

    sizing = None
    if not kill_switch:
        straddle = build_straddle(chain, expiry, spot, risk_free_rate)
        if straddle is not None:
            sizing = decide_sizing(
                signal.composite_edge, regime.current_label, equity,
                straddle.price, straddle.gamma, straddle.vega, settings.risk,
                deployed_long_vol_notional=deployed_long_vol_notional,
                deployed_short_vol_risk=deployed_short_vol_risk,
                extra_scale=extra_size_multiplier,
                vrp_raw=signal.vrp_raw,
                cost_floor_vol_pts=cost_floor_vol_pts,
            )
        else:
            notes.append(f"Could not build an ATM straddle for {settings.symbol} {expiry}; entries skipped.")

        if sizing and "cost_floor" in sizing.capped_by:
            notes.append(
                f"Edge {signal.vrp_raw:+.2%} does not clear the estimated round-trip hedging cost "
                f"({cost_floor_vol_pts:.2%} vol pts at {settings.hedge.transaction_cost_bps:.0f}bps) -- entry skipped."
            )
        elif sizing and sizing.already_at_target:
            notes.append(
                f"Already at target risk budget (${deployed_long_vol_notional + deployed_short_vol_risk:,.0f} "
                f"deployed vs. ${sizing.risk_budget_dollars:,.0f} target this cycle) -- no new entry."
            )

        extra_buffer_bps = toxicity.extra_limit_buffer_bps if (toxicity and toxicity.available) else 0.0

        if sizing and sizing.direction == "long_vol" and sizing.contracts > 0:
            plan = plan_straddle_buy(
                straddle, sizing.contracts,
                limit_price_buffer_bps=STRADDLE_BASE_LIMIT_BUFFER_BPS + extra_buffer_bps,
            )
            result = submit_orders(client, plan, settings)
            planned_orders.append({"kind": plan.kind, "description": plan.description, **result})
        elif sizing and sizing.direction == "short_vol" and settings.enable_credit_spreads:
            condor = build_iron_condor(
                chain, expiry, spot, risk_free_rate,
                settings.short_leg_target_delta, settings.wing_target_delta,
                sizing.max_structure_loss, settings.risk.max_contracts_per_leg,
            )
            if condor is not None:
                plan = plan_iron_condor(condor, limit_buffer_bps=extra_buffer_bps)
                result = submit_orders(client, plan, settings)
                planned_orders.append({"kind": plan.kind, "description": plan.description, **result})
            else:
                notes.append("Short-vol edge present but no iron condor could be built within the risk budget/chain liquidity.")

    hedge_decision = None
    if settings.enable_equity_hedge:
        # portfolio.net_delta_shares is already the BSM-delta-weighted
        # share-equivalent of every option position (see
        # strategy/portfolio.py), so a hedge_ratio of 1.0 hedges it
        # exactly. min_variance_hedge_ratio's covariance-based override
        # only improves on that once a persisted history of the book's
        # own option-value returns exists (see README "Productionizing"
        # -- v1 has no cross-cycle P&L series to regress on yet), so it
        # falls back to this 1.0 baseline every cycle for now.
        h_star = min_variance_hedge_ratio([], [], fallback_delta=1.0)
        hd = decide_hedge(
            portfolio_option_delta_shares=portfolio.net_delta_shares,
            portfolio_gamma_shares=portfolio.net_gamma_shares,
            current_hedge_shares=portfolio.equity_shares_held,
            regime_label=regime.current_label,
            spot=spot, time_to_expiry=max((expiry - end).days, 1) / 365.0,
            risk_free_rate=risk_free_rate, params=settings.hedge, hedge_ratio_override=h_star,
        )
        hedge_decision = {
            "should_trade": hd.should_trade, "target_shares": hd.target_shares,
            "current_shares": hd.current_shares, "trade_shares": hd.trade_shares,
            "band_half_width": hd.band_half_width, "reason": hd.reason,
        }
        if hd.should_trade:
            hedge_plan = plan_equity_hedge(settings.symbol, hd.trade_shares)
            if hedge_plan:
                result = submit_orders(client, hedge_plan, settings)
                planned_orders.append({"kind": hedge_plan.kind, "description": hedge_plan.description, **result})

    timestamp_utc = dt.datetime.utcnow().isoformat()
    record_equity(state, timestamp_utc, equity)
    risk_metrics = compute_risk_metrics(state)

    # For display: compute Kelly fraction + risk budget from the live signal
    # even when no straddle could be built (e.g. 0-DTE chain at close has
    # stale/crossed quotes). No execution happens -- purely for the UI.
    if sizing is not None:
        display_kelly = sizing.kelly_fraction
        display_budget = sizing.risk_budget_dollars
    else:
        display_kelly = _compute_kelly(
            signal.composite_edge, regime.current_label, settings.risk,
            extra_scale=extra_size_multiplier,
        )
        display_budget = abs(display_kelly) * equity

    result = AgentRunResult(
        timestamp_utc=timestamp_utc,
        symbol=settings.symbol, spot=spot,
        regime_label=regime.current_label, regime_confidence=regime.current_confidence,
        gamma_regime=signal.gamma_regime,
        forecast_vol=vol_fc.blended_next, live_atm_iv=live_iv,
        vrp_raw=signal.vrp_raw, vrp_z=signal.vrp_z, sas_mean=signal.sas_mean,
        composite_edge=signal.composite_edge, signal_rationale=signal.rationale,
        kelly_fraction=display_kelly,
        direction=sizing.direction if sizing else "flat",
        portfolio_delta_shares=portfolio.net_delta_shares, portfolio_gamma_shares=portfolio.net_gamma_shares,
        portfolio_vega_dollars=portfolio.net_vega_dollars, portfolio_theta_dollars=portfolio.net_theta_dollars,
        equity=equity, drawdown_pct=drawdown_pct, kill_switch=kill_switch,
        planned_orders=planned_orders, hedge_decision=hedge_decision, notes=notes,
        heston_cross_check=_heston_to_dict(heston), flow_toxicity=_toxicity_to_dict(toxicity),
        vol_path_hurst=vol_path_hurst,
        deployed_long_vol_notional=deployed_long_vol_notional, deployed_short_vol_risk=deployed_short_vol_risk,
        extra_size_multiplier=extra_size_multiplier,
        incremental_budget_dollars=sizing.incremental_budget_dollars if sizing else 0.0,
        already_at_target=sizing.already_at_target if sizing else False,
        risk_metrics=risk_metrics,
        risk_budget_dollars=display_budget,
        contracts_planned=sizing.contracts if sizing else 0,
        max_structure_loss=sizing.max_structure_loss if sizing else 0.0,
        capped_by=sizing.capped_by if sizing else [],
        regime_size_scale=REGIME_SIZE_SCALE.get(regime.current_label, 1.0),
        exit_decisions=[
            {"symbol": ed.symbol, "qty": ed.qty, "reason": ed.reason,
             "pnl_pct": ed.pnl_pct, "dte": ed.dte,
             "avg_entry_price": ed.avg_entry_price, "current_price": ed.current_price,
             "market_value": ed.market_value}
            for ed in exit_decisions_raw
        ],
    )

    state.last_run_utc = result.timestamp_utc
    record_decision(state, {
        "timestamp_utc": result.timestamp_utc, "symbol": result.symbol, "regime": result.regime_label,
        "composite_edge": result.composite_edge, "direction": result.direction,
        "orders": [o.get("description") for o in planned_orders], "dry_run": settings.dry_run,
    })
    save_state(state)
    return result


def _heston_to_dict(h: Optional[HestonCrossCheck]) -> Optional[dict[str, Any]]:
    if h is None:
        return None
    return {
        "available": h.available, "n_points": h.n_points, "n_expiries": h.n_expiries,
        "calibration_success": h.calibration_success, "rmse_vol_pts": h.rmse_vol_pts,
        "heston_atm_iv": h.heston_atm_iv, "atm_gap_vol_pts": h.atm_gap_vol_pts,
        "data_quality_ok": h.data_quality_ok, "fair_term_vol": h.fair_term_vol, "term_gap": h.term_gap,
        "feller_ratio": h.feller_ratio, "vol_of_vol_xi": h.vol_of_vol_xi,
        "vol_of_vol_regime": h.vol_of_vol_regime, "size_multiplier": h.size_multiplier, "notes": h.notes,
    }


def _toxicity_to_dict(t: Optional[FlowToxicity]) -> Optional[dict[str, Any]]:
    if t is None:
        return None
    return {
        "available": t.available, "n_bars": t.n_bars, "n_buckets": t.n_buckets,
        "current_vpin": t.current_vpin, "percentile_vs_history": t.percentile_vs_history,
        "regime": t.regime, "size_multiplier": t.size_multiplier,
        "extra_limit_buffer_bps": t.extra_limit_buffer_bps, "notes": t.notes,
    }
