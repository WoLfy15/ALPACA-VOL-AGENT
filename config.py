"""
config.py -- single source of truth for every tunable in the agent.

Everything is read from the environment (via a .env file in development)
so the same code runs unmodified in a notebook, a container, or behind
the FastAPI service. Nothing here ever hard-codes a key.

Design note: DRY_RUN defaults to True. The agent will compute signals,
size positions, and build order payloads whether or not DRY_RUN is set --
the only thing DRY_RUN gates is the final `alpaca_client.submit_order`
call. This means the UI can always show "what the agent would do" even
before the user is comfortable flipping it live, and it means a code
review of execution/order_manager.py is enough to convince yourself
nothing fires without the flag being explicitly turned off.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("alpaca_vol_agent.config")


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so we don't force a python-dotenv dependency.
    Does nothing if the file isn't present; never overrides a variable
    already set in the real environment (real env wins, e.g. in Docker)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


DEFAULT_WATCHLIST = "NVDA,AAPL,TSLA,MSFT,AMZN,GOOGL,META,NFLX,AVGO,AMD,BRK.B,JPM,V,LLY,WMT,COST"


def _list(name: str, default: str) -> list[str]:
    val = os.getenv(name)
    raw = val if val not in (None, "") else default
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class RiskLimits:
    """Portfolio-overlay risk budget. All fractions are of current
    account equity, read live from Alpaca -- not a fixed notional --
    so the budget scales automatically as paper equity moves."""

    max_gamma_risk_pct: float = field(default_factory=lambda: _float("MAX_GAMMA_RISK_PCT", 0.0005))
    max_vega_risk_pct: float = field(default_factory=lambda: _float("MAX_VEGA_RISK_PCT", 0.02))
    target_portfolio_vol: float = field(default_factory=lambda: _float("TARGET_PORTFOLIO_VOL", 0.12))
    max_drawdown_pct: float = field(default_factory=lambda: _float("MAX_DRAWDOWN_PCT", 0.25))
    kelly_fraction: float = field(default_factory=lambda: _float("KELLY_FRACTION", 0.25))
    kelly_clip: float = field(default_factory=lambda: _float("KELLY_CLIP", 0.15))
    max_contracts_per_leg: int = field(default_factory=lambda: _int("MAX_CONTRACTS_PER_LEG", 20))
    # -- auto-exit thresholds (v2) --
    exit_take_profit_pct: float = field(default_factory=lambda: _float("EXIT_TAKE_PROFIT_PCT", 0.50))
    exit_stop_loss_pct: float = field(default_factory=lambda: _float("EXIT_STOP_LOSS_PCT", 0.35))
    exit_dte_close_days: int = field(default_factory=lambda: _int("EXIT_DTE_CLOSE_DAYS", 1))


@dataclass(frozen=True)
class HedgeParams:
    """Feeds pricing_engine.risk.friction_hedging -- see that module's
    docstring for the Leland / Whalley-Wilmott derivations."""

    transaction_cost_bps: float = field(default_factory=lambda: _float("TRANSACTION_COST_BPS", 5.0))
    risk_aversion: float = field(default_factory=lambda: _float("HEDGE_RISK_AVERSION", 1.0))
    rehedge_min_interval_min: int = field(default_factory=lambda: _int("REHEDGE_MIN_INTERVAL_MIN", 30))
    min_variance_lookback_days: int = field(default_factory=lambda: _int("MIN_VARIANCE_LOOKBACK_DAYS", 20))


@dataclass(frozen=True)
class Settings:
    # -- Alpaca credentials -- names match the CLI's own env vars exactly
    # (github.com/alpacahq/cli): ALPACA_API_KEY / ALPACA_SECRET_KEY. The CLI
    # itself resolves paper vs. live -- env-sourced credentials default to
    # paper and only route live if ALPACA_LIVE_TRADE is exactly "true".
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    data_feed: str = field(default_factory=lambda: os.getenv("ALPACA_OPTION_FEED", "indicative"))
    stock_feed: str = field(default_factory=lambda: os.getenv("ALPACA_STOCK_FEED", "iex"))

    # -- what to trade --
    symbol: str = field(default_factory=lambda: os.getenv("SYMBOL", "SPY"))
    watchlist: list = field(default_factory=lambda: _list("WATCHLIST", DEFAULT_WATCHLIST))
    target_dte_days: int = field(default_factory=lambda: _int("TARGET_DTE_DAYS", 7))
    dte_tolerance_days: int = field(default_factory=lambda: _int("DTE_TOLERANCE_DAYS", 3))
    short_leg_target_delta: float = field(default_factory=lambda: _float("SHORT_LEG_TARGET_DELTA", 0.16))
    wing_target_delta: float = field(default_factory=lambda: _float("WING_TARGET_DELTA", 0.05))

    # -- safety --
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))
    # Separate from dry_run: this gates whether the Alpaca CLI itself is even
    # allowed to see ALPACA_LIVE_TRADE=true (alpaca_client.AlpacaClient._env).
    # Leave false. The CLI defaults to paper regardless, but this is the
    # project's own explicit opt-in on top of that default.
    i_understand_this_is_live: bool = field(default_factory=lambda: _bool("I_UNDERSTAND_THIS_IS_LIVE", False))
    enable_credit_spreads: bool = field(default_factory=lambda: _bool("ENABLE_CREDIT_SPREADS", True))
    enable_equity_hedge: bool = field(default_factory=lambda: _bool("ENABLE_EQUITY_HEDGE", True))
    min_open_interest: int = field(default_factory=lambda: _int("MIN_OPEN_INTEREST", 10))
    # -- v2: cross-checks/gates that were previously vendored code with no
    # caller (research/heston_signal.py, research/toxicity.py) or dead
    # code (pricing_engine.risk.friction_hedging.leland_cost_bps_of_vega).
    # Each degrades gracefully to v1 behavior when disabled or when its
    # own data requirements aren't met this cycle -- see the respective
    # module docstrings.
    enable_heston_cross_check: bool = field(default_factory=lambda: _bool("ENABLE_HESTON_CROSS_CHECK", True))
    enable_toxicity_gate: bool = field(default_factory=lambda: _bool("ENABLE_TOXICITY_GATE", True))
    enable_cost_floor: bool = field(default_factory=lambda: _bool("ENABLE_COST_FLOOR", True))
    enable_auto_exit: bool = field(default_factory=lambda: _bool("ENABLE_AUTO_EXIT", True))

    # -- risk --
    risk: RiskLimits = field(default_factory=RiskLimits)
    hedge: HedgeParams = field(default_factory=HedgeParams)

    def validate(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Copy .env.example to .env "
                "and fill in a NEW, dedicated PAPER account's keys from "
                "https://app.alpaca.markets (Paper Trading) -- the hackathon playbook "
                "requires a fresh paper account created for this event, not a reused one."
            )
        if self.i_understand_this_is_live:
            log.warning(
                "I_UNDERSTAND_THIS_IS_LIVE=true -- the Alpaca CLI will be allowed to route "
                "orders to LIVE trading if ALPACA_LIVE_TRADE=true is also set in the "
                "environment. This project is built and validated for paper trading only."
            )


SETTINGS = Settings()
