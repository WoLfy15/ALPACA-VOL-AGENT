"""
api/server.py -- FastAPI backend. This is the seam the hackathon UI is
meant to be built against: every route returns plain JSON (dataclasses
via `asdict`), CORS is wide open for local UI development, and nothing
in here duplicates logic that already lives in agent/loop.py or the
strategy/ modules -- the routes are thin.

Run with:  uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.loop import run_cycle
from agent.state import compute_risk_metrics, load_state
from alpaca_client import AlpacaClient
from config import SETTINGS, Settings
from data.market_data import atm_implied_vol, fetch_chain
from strategy.portfolio import build_portfolio_snapshot

app = FastAPI(title="Alpaca Volatility Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def _client() -> AlpacaClient:
    SETTINGS.validate()
    return AlpacaClient(settings=SETTINGS)


@app.get("/health")
def health():
    return {"ok": True, "time_utc": dt.datetime.utcnow().isoformat(), "dry_run": SETTINGS.dry_run, "symbol": SETTINGS.symbol}


@app.get("/status")
def status():
    client = _client()
    account = client.get_account()
    clock = client.get_clock()
    return {
        "account": {
            "equity": account.get("equity"), "cash": account.get("cash"),
            "buying_power": account.get("buying_power"),
            "options_trading_level": account.get("options_trading_level"),
            "options_approved_level": account.get("options_approved_level"),
        },
        "market_open": clock.get("is_open"),
        "next_open": clock.get("next_open"), "next_close": clock.get("next_close"),
        "dry_run": SETTINGS.dry_run, "symbol": SETTINGS.symbol,
    }


@app.get("/chain")
def chain(symbol: Optional[str] = None, dte: int = Query(default=None)):
    sym = symbol or SETTINGS.symbol
    client = _client()
    spot = client.get_mid_price(sym)
    end = dt.date.today()
    target_dte = dte if dte is not None else SETTINGS.target_dte_days
    contracts = fetch_chain(
        client, sym,
        expiration_date_gte=end.isoformat(),
        expiration_date_lte=(end + dt.timedelta(days=target_dte + SETTINGS.dte_tolerance_days + 10)).isoformat(),
        min_open_interest=SETTINGS.min_open_interest,
    )
    atm_iv = atm_implied_vol(contracts, spot, 0.045)
    return {
        "symbol": sym, "spot": spot, "atm_implied_vol": atm_iv,
        "contracts": [
            {
                "symbol": c.symbol, "strike": c.strike, "expiry": c.expiry.isoformat(),
                "type": c.option_type.value, "bid": c.bid, "ask": c.ask, "mid": c.mid,
                "open_interest": c.open_interest, "delta": c.alpaca_delta, "iv": c.alpaca_iv,
            }
            for c in contracts
        ],
    }


@app.get("/positions")
def positions():
    client = _client()
    spot = client.get_mid_price(SETTINGS.symbol)
    snap = build_portfolio_snapshot(client, SETTINGS.symbol, spot, 0.045)
    return {
        "equity": snap.equity, "cash": snap.cash, "equity_shares_held": snap.equity_shares_held,
        "net_delta_shares": snap.net_delta_shares, "net_gamma_shares": snap.net_gamma_shares,
        "net_vega_dollars": snap.net_vega_dollars, "net_theta_dollars": snap.net_theta_dollars,
        "positions": [asdict(p) for p in snap.option_positions],
    }


@app.get("/decisions")
def decisions(limit: int = 50):
    state = load_state()
    return {"equity_high_water_mark": state.equity_high_water_mark, "last_run_utc": state.last_run_utc,
            "decisions": state.decision_log[-limit:]}


@app.get("/metrics")
def metrics():
    """Live risk metrics (Sharpe/Sortino/max-drawdown/win-rate) computed
    from the account equity curve agent/loop.py now persists every
    cycle (agent/state.record_equity + compute_risk_metrics). Returns
    Nones, not zeros, until enough cycles have run to say anything --
    see compute_risk_metrics's own docstring."""
    state = load_state()
    risk_metrics = compute_risk_metrics(state)
    return {
        **risk_metrics,
        "equity_curve": state.equity_curve[-500:],   # capped for a reasonably-sized response
        "equity_high_water_mark": state.equity_high_water_mark,
    }


class BacktestRequest(BaseModel):
    symbol: Optional[str] = None
    years: float = 3.0
    capital: float = 1_000_000.0
    cost_bps: float = 5.0


@app.post("/backtest")
def backtest(req: BacktestRequest = BacktestRequest()):
    """Runs backtest/run_backtest.py on demand (yfinance daily history,
    the fused regime+GARCH/HAR+Kelly strategy, WITH the v2 transaction-
    cost model -- see that module's docstring for why cost_bps=0
    recovers the old frictionless reading, and why that reading alone
    is not a trustworthy Sharpe for this strategy shape). Needs network
    + `pip install yfinance arch hmmlearn` in whatever environment this
    API server is actually running in -- returns a clear 502 rather than
    a bare stack trace if those aren't available."""
    try:
        from backtest.run_backtest import run_backtest as _run_backtest
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"Backtest dependencies not installed: {exc}") from exc
    try:
        summary = _run_backtest(
            req.symbol or SETTINGS.symbol, years=req.years, initial_capital=req.capital, cost_bps=req.cost_bps,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return asdict(summary)


class RunRequest(BaseModel):
    dry_run: Optional[bool] = None
    symbol: Optional[str] = None


@app.post("/run")
def run(req: RunRequest = RunRequest()):
    """Runs exactly one agent cycle: fetch -> analyze -> decide ->
    (execute unless dry_run). Returns the full AgentRunResult as JSON --
    this is the single endpoint a UI needs to drive a "Run Agent" button.
    Includes the v2 fields (heston_cross_check, flow_toxicity,
    risk_metrics, etc.) automatically since they're just more fields on
    the same dataclass this endpoint already asdict()s."""
    settings = SETTINGS
    if req.dry_run is not None or req.symbol is not None:
        settings = Settings(
            api_key=SETTINGS.api_key, api_secret=SETTINGS.api_secret,
            data_feed=SETTINGS.data_feed, stock_feed=SETTINGS.stock_feed,
            symbol=req.symbol or SETTINGS.symbol,
            target_dte_days=SETTINGS.target_dte_days, dte_tolerance_days=SETTINGS.dte_tolerance_days,
            short_leg_target_delta=SETTINGS.short_leg_target_delta, wing_target_delta=SETTINGS.wing_target_delta,
            dry_run=req.dry_run if req.dry_run is not None else SETTINGS.dry_run,
            i_understand_this_is_live=SETTINGS.i_understand_this_is_live,
            enable_credit_spreads=SETTINGS.enable_credit_spreads, enable_equity_hedge=SETTINGS.enable_equity_hedge,
            min_open_interest=SETTINGS.min_open_interest, risk=SETTINGS.risk, hedge=SETTINGS.hedge,
            enable_heston_cross_check=SETTINGS.enable_heston_cross_check,
            enable_toxicity_gate=SETTINGS.enable_toxicity_gate,
            enable_cost_floor=SETTINGS.enable_cost_floor,
            enable_auto_exit=SETTINGS.enable_auto_exit,
        )
    try:
        result = run_cycle(settings)
    except Exception as exc:  # noqa: BLE001 -- surface the real error to the UI rather than a bare 500
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return asdict(result)


@app.get("/")
def root():
    return {
        "service": "alpaca-vol-agent", "docs": "/docs",
        "endpoints": [
            "/health", "/status", "/chain", "/positions", "/decisions", "/metrics",
            "POST /run", "POST /backtest",
        ],
    }
