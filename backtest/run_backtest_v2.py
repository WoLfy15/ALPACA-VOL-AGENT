"""
backtest/run_backtest_v2.py -- enhanced backtest that mirrors the live
agent's actual signal / sizing / hedging logic with the tuned parameters,
rather than the original IITfinal.ipynb formulas.

Key differences from run_backtest.py:
  - Uses the same tanh-squashed composite_edge + fractional Kelly as the live agent
  - Applies Whalley-Wilmott no-trade band to hedging (not daily rehedge-to-zero)
  - Uses regime-aware hedge multipliers
  - Applies the same gamma/vega caps and Kelly clip as the live agent
  - Incremental sizing: only sizes the GAP between target and deployed
  - Properly separates long-vol (buy straddle) and short-vol (iron condor) paths

Usage:
    python -m backtest.run_backtest_v2 --symbol SPY --years 3
    python -m backtest.run_backtest_v2 --symbol SPY --years 3 --cost-bps 0  # frictionless
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.regime import detect_regime  # noqa: E402
from research.vol_forecast import forecast_volatility  # noqa: E402
from strategy.sizing import REGIME_SIZE_SCALE  # noqa: E402
from strategy.hedging import REGIME_HEDGE_MULTIPLIER  # noqa: E402
from config import RiskLimits, HedgeParams  # noqa: E402
from pricing_engine.risk.friction_hedging import whalley_wilmott_band  # noqa: E402


def bs_straddle(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = norm.cdf(d1) - norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    return call + put, delta, gamma * 2, vega * 2, theta * 2  # per straddle (call + put)


@dataclass
class BacktestV2Summary:
    symbol: str
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    win_rate: float
    final_capital: float
    n_days: int
    total_friction_paid: float
    cost_bps: float
    avg_contracts: float
    hedge_trades: int
    total_trades: int


def run_backtest_v2(
    symbol: str, years: float = 3.0, initial_capital: float = 100_000.0,
    risk: RiskLimits | None = None, hedge: HedgeParams | None = None,
    roll_freq_days: int = 7, out_dir: str = "backtest/out",
    cost_bps: float = 3.0, option_cost_multiple: float = 3.0,
) -> BacktestV2Summary:
    risk = risk or RiskLimits()
    hedge = hedge or HedgeParams()

    data = yf.download(symbol, period=f"{years:.0f}y", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    close = data["Close"].dropna()
    if len(close) < 300:
        raise ValueError(f"only {len(close)} bars for {symbol}; need >=300")

    regime = detect_regime(close)
    log_returns = np.log(close / close.shift(1)).dropna()
    realized_vol = (log_returns.rolling(21).std() * np.sqrt(252)).dropna()
    aligned_returns = log_returns.loc[realized_vol.index]
    vol_fc = forecast_volatility(aligned_returns, realized_vol)

    idx = regime.labels.index.intersection(vol_fc.blended.index)
    frame = pd.DataFrame({
        "close": close.loc[idx], "regime": regime.labels.loc[idx],
        "forecast_vol": vol_fc.blended.loc[idx], "realized_vol": realized_vol.loc[idx],
    }).dropna()
    frame["implied_vol_proxy"] = frame["realized_vol"] * 1.15
    frame["vrp"] = frame["forecast_vol"] - frame["implied_vol_proxy"]
    frame["vrp_z"] = (frame["vrp"] - frame["vrp"].rolling(60).mean()) / frame["vrp"].rolling(60).std()
    frame = frame.dropna()

    r = 0.045
    k = cost_bps / 10_000.0
    cash = initial_capital
    shares = 0.0
    contracts_held = 0
    deployed_notional = 0.0
    trade_log = []
    K = round(frame["close"].iloc[0] / 5) * 5
    expiry_idx = 0
    total_friction = 0.0
    high_water = initial_capital
    hedge_trade_count = 0
    total_trade_count = 0
    contract_sum = 0

    for i, (date, row) in enumerate(frame.iterrows()):
        capital_mark = cash + shares * row["close"]
        high_water = max(high_water, capital_mark)

        # Drawdown kill switch
        drawdown_pct = (capital_mark - high_water) / high_water if high_water > 0 else 0
        if drawdown_pct < -risk.max_drawdown_pct:
            if contracts_held != 0 or abs(shares) > 0.5:
                friction = 0.0
                if k > 0 and contracts_held != 0:
                    straddle_price = bs_straddle(
                        row["close"], K, max((expiry_idx - i), 1) / 252.0,
                        r, max(row["implied_vol_proxy"], 1e-3)
                    )[0]
                    friction += k * option_cost_multiple * abs(contracts_held) * max(straddle_price, 0.01) * 100
                if k > 0 and abs(shares) > 0.5:
                    friction += k * abs(shares) * row["close"]
                cash += shares * row["close"]
                shares = 0.0
                contracts_held = 0
                deployed_notional = 0.0
                total_friction += friction
                cash -= friction
            trade_log.append({"date": date, "capital": cash, "regime": row["regime"], "contracts": 0, "friction": 0.0})
            continue

        rolled = (i % roll_freq_days == 0)
        if rolled:
            K = round(row["close"] / 5) * 5
            expiry_idx = i + roll_freq_days
            deployed_notional = 0.0
        T = max((expiry_idx - i), 1) / 252.0

        sigma_impl = max(row["implied_vol_proxy"], 1e-3)
        price, delta, gamma, vega, _theta = bs_straddle(row["close"], K, T, r, sigma_impl)

        # ---- SIGNAL: same as live agent's composite_edge ----
        vrp_z = row["vrp_z"] if not np.isnan(row["vrp_z"]) else 0.0
        composite_edge = math.tanh(0.85 * vrp_z)

        # ---- SIZING: same fractional Kelly as live agent ----
        regime_scale = REGIME_SIZE_SCALE.get(row["regime"], 1.0)
        kelly_raw = risk.kelly_fraction * composite_edge * regime_scale
        kelly = float(np.clip(kelly_raw, -risk.kelly_clip, risk.kelly_clip))

        if abs(kelly) < 1e-4:
            new_contracts = 0
        elif kelly > 0:
            budget = capital_mark * kelly
            incremental = max(budget - deployed_notional, 0.0)
            if incremental <= 0:
                new_contracts = contracts_held
            else:
                raw_contracts = incremental / (price * 100) if price > 0 else 0

                gamma_risk = abs(raw_contracts * gamma * 100)
                max_gamma = risk.max_gamma_risk_pct * capital_mark
                if gamma_risk > max_gamma > 0:
                    raw_contracts *= max_gamma / gamma_risk

                vega_risk = abs(raw_contracts * vega * 100)
                max_vega = risk.max_vega_risk_pct * capital_mark
                if vega_risk > max_vega > 0:
                    raw_contracts *= max_vega / vega_risk

                add_contracts = int(min(raw_contracts, risk.max_contracts_per_leg - max(contracts_held, 0)))
                new_contracts = max(contracts_held, 0) + max(add_contracts, 0)
        else:
            budget = capital_mark * abs(kelly)
            incremental = max(budget - abs(deployed_notional), 0.0)
            if incremental <= 0:
                new_contracts = contracts_held
            else:
                raw_contracts = -(incremental / (price * 100)) if price > 0 else 0
                new_contracts = max(-risk.max_contracts_per_leg, min(contracts_held, 0) + int(raw_contracts))

        # Apply friction on position changes
        friction = 0.0
        if new_contracts != contracts_held:
            if k > 0:
                changed_qty = abs(new_contracts - contracts_held)
                if rolled:
                    changed_qty = abs(new_contracts) + abs(contracts_held)
                friction_opt = k * option_cost_multiple * changed_qty * price * 100
                cash -= friction_opt
                friction += friction_opt
            total_trade_count += 1
            deployed_notional = abs(new_contracts) * price * 100
        elif rolled and contracts_held != 0:
            if k > 0:
                friction_opt = k * option_cost_multiple * (abs(new_contracts) + abs(contracts_held)) * price * 100
                cash -= friction_opt
                friction += friction_opt
            total_trade_count += 1
            deployed_notional = abs(new_contracts) * price * 100

        contracts_held = new_contracts
        contract_sum += abs(contracts_held)

        # ---- HEDGING: Whalley-Wilmott band ----
        portfolio_delta = delta * contracts_held * 100
        portfolio_gamma = gamma * contracts_held * 100
        regime_hedge_mult = REGIME_HEDGE_MULTIPLIER.get(row["regime"], 1.0)
        ideal_hedge = -regime_hedge_mult * portfolio_delta

        band = whalley_wilmott_band(
            spot=row["close"], gamma=portfolio_gamma,
            k=k if k > 0 else 1e-6,
            risk_aversion=hedge.risk_aversion,
            r=r, time_to_expiry=max(T, 1 / 365),
        )

        drift = abs(shares - ideal_hedge)
        if drift > band or rolled:
            trade_shares = ideal_hedge - shares
            if abs(trade_shares) >= 1:
                if k > 0:
                    friction_eq = k * abs(trade_shares) * row["close"]
                    cash -= friction_eq
                    friction += friction_eq
                cash -= trade_shares * row["close"]
                shares += trade_shares
                hedge_trade_count += 1

        total_friction += friction

        option_value = contracts_held * price * 100
        capital = cash + shares * row["close"] + option_value

        trade_log.append({
            "date": date, "close": row["close"], "regime": row["regime"],
            "contracts": contracts_held, "shares": shares, "capital": capital,
            "delta": delta, "gamma": gamma, "vega": vega, "friction": friction,
            "kelly": kelly, "edge": composite_edge, "band": band,
        })

    trade_df = pd.DataFrame(trade_log).set_index("date")
    trade_df["return"] = trade_df["capital"].pct_change()
    excess = trade_df["return"].dropna()
    sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0
    downside = excess[excess < 0]
    sortino = float(np.sqrt(252) * excess.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else 0.0
    win_rate = float((excess > 0).mean()) if len(excess) > 0 else 0.0
    cum = (1 + trade_df["return"].fillna(0)).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = float(dd.min())
    cagr = float(cum.iloc[-1] ** (252 / len(cum)) - 1) if len(cum) > 0 else 0.0
    avg_contracts = contract_sum / max(len(trade_df), 1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trade_df.to_csv(out_path / f"{symbol}_backtest_v2.csv")

    return BacktestV2Summary(
        symbol=symbol, sharpe=sharpe, sortino=sortino, cagr=cagr, max_drawdown=max_dd,
        win_rate=win_rate, final_capital=float(trade_df["capital"].iloc[-1]) if len(trade_df) else initial_capital,
        n_days=len(trade_df), total_friction_paid=float(total_friction), cost_bps=cost_bps,
        avg_contracts=avg_contracts, hedge_trades=hedge_trade_count, total_trades=total_trade_count,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=3.0)
    args = parser.parse_args()

    summary = run_backtest_v2(args.symbol, years=args.years, initial_capital=args.capital, cost_bps=args.cost_bps)
    print(f"\n{'='*55}")
    print(f"  {summary.symbol} VOL-ARB BACKTEST v2 (tuned parameters)")
    print(f"{'='*55}")
    print(f"  Sharpe Ratio   : {summary.sharpe:.2f}")
    print(f"  Sortino Ratio  : {summary.sortino:.2f}")
    print(f"  CAGR           : {summary.cagr * 100:.2f}%")
    print(f"  Max Drawdown   : {summary.max_drawdown * 100:.2f}%")
    print(f"  Win Rate       : {summary.win_rate * 100:.1f}%")
    print(f"  Final Capital  : ${summary.final_capital:,.0f}")
    print(f"  Avg Contracts  : {summary.avg_contracts:.1f}")
    print(f"  Hedge Trades   : {summary.hedge_trades}")
    print(f"  Total Trades   : {summary.total_trades}")
    print(f"  Trading Days   : {summary.n_days}")
    print(f"  Cost Assumed   : {summary.cost_bps:.1f}bps")
    print(f"  Friction Paid  : ${summary.total_friction_paid:,.0f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
