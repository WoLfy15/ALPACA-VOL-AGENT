"""
alpaca_client.py -- every Alpaca call in this project goes through
direct REST API calls using the `requests` library.

This replaces the original Alpaca CLI (Go binary) implementation with
equivalent direct HTTPS calls to:
  - Trading API:  https://paper-api.alpaca.markets
  - Market Data:  https://data.alpaca.markets

Endpoints in use (confirmed against docs.alpaca.markets):
  Trading API paths
    GET  /v2/account
    GET  /v2/clock
    GET  /v2/positions
    GET  /v2/positions/{symbol_or_id}
    GET  /v2/orders
    POST /v2/orders                 (single-leg AND order_class="mleg")
    DEL  /v2/orders/{id}
    GET  /v2/options/contracts?underlying_symbols=...
  Market Data API paths
    GET  /v2/stocks/{symbol}/bars
    GET  /v2/stocks/{symbol}/quotes/latest
    GET  /v2/stocks/{symbol}/trades/latest
    GET  /v1beta1/options/snapshots/{underlying_symbol}   (option chain + Greeks)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode

import requests

from config import Settings

log = logging.getLogger("alpaca_vol_agent.client")

TRADING_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


class AlpacaAPIError(RuntimeError):
    """Raised on a non-2xx response from the Alpaca REST API."""

    def __init__(self, method: str, url: str, status: int, body: str):
        self.method = method
        self.url = url
        self.status = status
        self.body = body
        super().__init__(f"{method} {url} -> HTTP {status}: {body[:500]}")


@dataclass
class AlpacaClient:
    settings: Settings
    timeout: float = 30.0
    # cli_binary kept for API compatibility (unused now)
    cli_binary: str = "alpaca"

    def __post_init__(self):
        # Validate credentials are present
        if not self.settings.api_key or not self.settings.api_secret:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. "
                "Check your .env file."
            )
        log.info("AlpacaClient initialised (direct REST, paper trading endpoint)")

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.settings.api_key,
            "APCA-API-SECRET-KEY": self.settings.api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def doctor(self) -> str:
        """Connectivity check: fetch account info and return a summary string."""
        try:
            acct = self.get_account()
            return (
                f"OK -- account_id={acct.get('id')} "
                f"status={acct.get('status')} "
                f"equity={acct.get('equity')}"
            )
        except Exception as exc:
            return f"ERROR: {exc}"

    @staticmethod
    def _with_query(path: str, params: Optional[dict]) -> str:
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        return f"{path}?{urlencode(clean)}" if clean else path

    def _trading_request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> Any:
        url = TRADING_BASE + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            url = url + "?" + urlencode(clean) if clean else url
        resp = requests.request(
            method,
            url,
            headers=self._headers(),
            json=json,
            timeout=self.timeout,
        )
        log.debug("%s %s -> %s", method, url, resp.status_code)
        if not resp.ok:
            raise AlpacaAPIError(method, url, resp.status_code, resp.text)
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    def _data_request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
    ) -> Any:
        url = DATA_BASE + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            url = url + "?" + urlencode(clean) if clean else url
        resp = requests.request(
            method,
            url,
            headers=self._headers(),
            timeout=self.timeout,
        )
        log.debug("%s %s -> %s", method, url, resp.status_code)
        if not resp.ok:
            raise AlpacaAPIError(method, url, resp.status_code, resp.text)
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    def _trading_get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._trading_request("GET", path, params=params)

    def _trading_post(self, path: str, json: dict) -> Any:
        return self._trading_request("POST", path, json=json)

    def _trading_delete(self, path: str) -> Any:
        return self._trading_request("DELETE", path)

    def _data_get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._data_request("GET", path, params=params)

    # ---------------------------------------------------------------- account
    def get_account(self) -> dict:
        return self._trading_get("/v2/account")

    def get_clock(self) -> dict:
        return self._trading_get("/v2/clock")

    def is_market_open(self) -> bool:
        return bool(self.get_clock().get("is_open"))

    # --------------------------------------------------------------- positions
    def get_positions(self) -> list[dict]:
        return self._trading_get("/v2/positions") or []

    def get_position(self, symbol_or_id: str) -> Optional[dict]:
        try:
            return self._trading_get(f"/v2/positions/{symbol_or_id}")
        except AlpacaAPIError as exc:
            if exc.status == 404:
                return None
            raise

    def close_position(self, symbol_or_id: str) -> Optional[dict]:
        """Liquidate a single position at market via DELETE /v2/positions/{symbol}.
        Alpaca fills this as a market sell-to-close for long positions or
        buy-to-close for short positions. Returns the closing order dict,
        or None if the position was already gone (404)."""
        try:
            return self._trading_request("DELETE", f"/v2/positions/{symbol_or_id}")
        except AlpacaAPIError as exc:
            if exc.status == 404:
                log.warning("close_position: %s not found (already closed?)", symbol_or_id)
                return None
            raise

    # ------------------------------------------------------------------ orders
    def get_orders(self, status: str = "open", limit: int = 100) -> list[dict]:
        return self._trading_get("/v2/orders", params={"status": status, "limit": limit}) or []

    def submit_order(self, payload: dict) -> dict:
        """Submit any order payload as-is (single-leg or order_class='mleg').
        Callers build the payload via execution/order_manager.py, which
        already stamps every payload with a client_order_id for idempotency."""
        return self._trading_post("/v2/orders", json=payload)

    def cancel_order(self, order_id: str) -> None:
        self._trading_delete(f"/v2/orders/{order_id}")

    def cancel_all_orders(self) -> None:
        self._trading_delete("/v2/orders")

    # ------------------------------------------------------------ option chain
    def get_option_contracts(
        self,
        underlying_symbol: str,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Trading API's contract master list (status, tradable, OI, etc.) --
        NOT priced. Use get_option_chain_snapshot for live quotes/Greeks."""
        params: dict[str, Any] = {"underlying_symbols": underlying_symbol, "limit": limit}
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        contracts: list[dict] = []
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            resp = self._trading_get("/v2/options/contracts", params=params)
            contracts.extend(resp.get("option_contracts", []))
            page_token = resp.get("page_token")
            if not page_token:
                break
        return contracts

    def get_option_chain_snapshot(
        self,
        underlying_symbol: str,
        option_type: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        expiration_date: Optional[str] = None,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        limit: int = 1000,
    ) -> dict[str, dict]:
        """Data API's live snapshot per contract symbol: latestTrade,
        latestQuote, and greeks (delta/gamma/theta/vega/rho + impliedVolatility)
        when the feed supplies them. Returns {contract_symbol: snapshot}."""
        params: dict[str, Any] = {"feed": self.settings.data_feed, "limit": limit}
        if option_type:
            params["type"] = option_type
        if strike_price_gte is not None:
            params["strike_price_gte"] = strike_price_gte
        if strike_price_lte is not None:
            params["strike_price_lte"] = strike_price_lte
        if expiration_date:
            params["expiration_date"] = expiration_date
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte

        snapshots: dict[str, dict] = {}
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            resp = self._data_get(f"/v1beta1/options/snapshots/{underlying_symbol}", params=params)
            snapshots.update(resp.get("snapshots", {}))
            page_token = resp.get("next_page_token")
            if not page_token:
                break
        return snapshots

    # ------------------------------------------------------------- stock data
    def get_stock_bars(
        self, symbol: str, timeframe: str = "1Day", start: Optional[str] = None,
        end: Optional[str] = None, limit: int = 1000,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "timeframe": timeframe, "limit": limit, "feed": self.settings.stock_feed,
            "adjustment": "all",
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        bars: list[dict] = []
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            resp = self._data_get(f"/v2/stocks/{symbol}/bars", params=params)
            bars.extend(resp.get("bars", []))
            page_token = resp.get("next_page_token")
            if not page_token or len(bars) >= limit:
                break
        return bars

    def get_latest_stock_quote(self, symbol: str) -> dict:
        resp = self._data_get(
            f"/v2/stocks/{symbol}/quotes/latest",
            params={"feed": self.settings.stock_feed},
        )
        return resp.get("quote", {})

    def get_latest_stock_trade(self, symbol: str) -> dict:
        resp = self._data_get(
            f"/v2/stocks/{symbol}/trades/latest",
            params={"feed": self.settings.stock_feed},
        )
        return resp.get("trade", {})

    def get_mid_price(self, symbol: str) -> float:
        q = self.get_latest_stock_quote(symbol)
        bid, ask = q.get("bp"), q.get("ap")
        if bid and ask:
            return (bid + ask) / 2.0
        return float(self.get_latest_stock_trade(symbol).get("p", 0.0))
