"""
Unit tests for the code this project actually adds on top of the vendored,
already-tested pricing_engine (see options_pricing_engine's own 59-test
suite for that layer). These focus on the new glue: signal fusion, Kelly
sizing + risk caps, the hedge no-trade band, and Alpaca-compliant order
payload construction -- run with `pytest tests/ -v` from the project root.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from config import HedgeParams, RiskLimits
from data.market_data import ChainContract
from pricing_engine.market_data.contract import OptionType
from research.signal import build_signal
from strategy.hedging import decide_hedge, min_variance_hedge_ratio
from strategy.legs import build_iron_condor, build_straddle, select_expiry
from strategy.portfolio import parse_occ_symbol
from strategy.sizing import decide_sizing, kelly_fraction


# --------------------------------------------------------------- signal ---
def _wobbly_history(base: float, n: int = 80, amplitude: float = 0.02, seed: int = 1) -> np.ndarray:
    """A historical series with genuine variance, so VRP z-scoring has a
    nonzero std to divide by -- a perfectly flat series (std=0) is an
    edge case build_signal deliberately treats as 'not enough signal',
    which is correct behavior but not what these tests want to exercise."""
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, amplitude, n)


def test_signal_positive_when_forecast_exceeds_iv():
    sig = build_signal(
        regime_label="Range", forecast_vol_now=0.25, live_atm_iv=0.18,
        forecast_vol_history=_wobbly_history(0.20), realized_vol_history=_wobbly_history(0.15, seed=2),
    )
    assert sig.vrp_raw == pytest.approx(0.07)
    assert sig.composite_edge > 0


def test_signal_sas_disagreement_is_flagged():
    sig = build_signal(
        regime_label="Range", forecast_vol_now=0.25, live_atm_iv=0.18,
        forecast_vol_history=_wobbly_history(0.20), realized_vol_history=_wobbly_history(0.15, seed=2),
        sas_values=[0.06, 0.05, 0.07],  # market priced RICH by history too -> should disagree w/ "cheap" VRP read
    )
    assert sig.sas_agrees is False


def test_signal_composite_edge_bounded():
    sig = build_signal(
        regime_label="Crash", forecast_vol_now=0.9, live_atm_iv=0.1,
        forecast_vol_history=_wobbly_history(0.2), realized_vol_history=_wobbly_history(0.15, seed=2),
    )
    assert -1.0 <= sig.composite_edge <= 1.0


# --------------------------------------------------------------- sizing ---
def test_kelly_fraction_clipped():
    risk = RiskLimits(kelly_fraction=0.25, kelly_clip=0.15)
    k = kelly_fraction(composite_edge=1.0, regime_label="Range", risk=risk)
    assert k == pytest.approx(risk.kelly_clip)
    k_neg = kelly_fraction(composite_edge=-1.0, regime_label="Range", risk=risk)
    assert k_neg == pytest.approx(-risk.kelly_clip)


def test_decide_sizing_long_vol_respects_gamma_cap():
    risk = RiskLimits(max_gamma_risk_pct=0.0001, max_contracts_per_leg=50)
    decision = decide_sizing(
        composite_edge=0.8, regime_label="Range", capital=1_000_000,
        straddle_price=5.0, straddle_gamma=1.0, straddle_vega=10.0, risk=risk,
    )
    assert decision.direction == "long_vol"
    assert decision.contracts >= 0
    # gamma_risk = contracts * gamma * 100 must stay <= max_gamma_risk_pct * capital
    assert decision.contracts * 1.0 * 100 <= risk.max_gamma_risk_pct * 1_000_000 * 1.01


def test_decide_sizing_short_vol_gives_dollar_budget_not_contracts():
    risk = RiskLimits()
    decision = decide_sizing(
        composite_edge=-0.5, regime_label="Trend", capital=200_000,
        straddle_price=5.0, straddle_gamma=0.05, straddle_vega=10.0, risk=risk,
    )
    assert decision.direction == "short_vol"
    assert decision.contracts == 0
    assert decision.max_structure_loss > 0


def test_decide_sizing_flat_when_edge_near_zero():
    risk = RiskLimits()
    decision = decide_sizing(
        composite_edge=0.0, regime_label="Range", capital=100_000,
        straddle_price=5.0, straddle_gamma=0.05, straddle_vega=10.0, risk=risk,
    )
    assert decision.direction == "flat"


# -------------------------------------------------------------- hedging ---
def test_hedge_within_band_does_not_trade():
    hd = decide_hedge(
        portfolio_option_delta_shares=10.0, portfolio_gamma_shares=0.5,
        current_hedge_shares=-10.0,  # already ~flat: net delta ~= 0
        regime_label="Range", spot=100.0, time_to_expiry=7 / 365,
        risk_free_rate=0.04, params=HedgeParams(transaction_cost_bps=5, risk_aversion=1.0),
    )
    assert hd.should_trade is False
    assert hd.trade_shares == 0.0


def test_hedge_outside_band_trades_towards_target():
    hd = decide_hedge(
        portfolio_option_delta_shares=500.0, portfolio_gamma_shares=5.0,
        current_hedge_shares=0.0,
        regime_label="Crash", spot=100.0, time_to_expiry=7 / 365,
        risk_free_rate=0.04, params=HedgeParams(transaction_cost_bps=5, risk_aversion=1.0),
    )
    assert hd.should_trade is True
    # Crash regime multiplier (1.3x) means we hedge MORE than 1:1 against the option delta.
    assert hd.target_shares == pytest.approx(-1.3 * 500.0)


def test_min_variance_hedge_ratio_falls_back_with_short_history():
    ratio = min_variance_hedge_ratio([0.01, -0.02], [0.02, -0.01], fallback_delta=0.42)
    assert ratio == 0.42


def test_min_variance_hedge_ratio_regresses_with_enough_history():
    rng = np.random.default_rng(0)
    stock = rng.normal(0, 0.01, 200)
    option = 2.5 * stock + rng.normal(0, 0.001, 200)  # true hedge ratio ~2.5
    ratio = min_variance_hedge_ratio(option, stock, fallback_delta=0.5)
    assert ratio == pytest.approx(2.5, abs=0.15)


# ----------------------------------------------------------------- legs ---
def _synthetic_chain(spot: float, expiry: dt.date) -> list[ChainContract]:
    """A small but realistic-enough chain: extrinsic value peaks ATM and
    decays with distance (like real time value), so further-OTM wings
    are always cheaper than nearer short strikes -- the property
    build_iron_condor's net-credit check depends on, same as any real
    listed chain."""
    chain = []
    for k in range(int(spot) - 80, int(spot) + 81, 5):
        for otype, sym_c in ((OptionType.CALL, "C"), (OptionType.PUT, "P")):
            intrinsic = max(spot - k, 0) if otype == OptionType.CALL else max(k - spot, 0)
            extrinsic = 8.0 * math.exp(-((k - spot) / (0.18 * spot)) ** 2)
            mid = round(intrinsic + extrinsic, 2)
            moneyness = (k - spot) / spot
            delta = max(0.02, min(0.98, 0.5 - moneyness * 3))
            if otype == OptionType.PUT:
                delta = -delta
            chain.append(ChainContract(
                symbol=f"TST{expiry.strftime('%y%m%d')}{sym_c}{int(k * 1000):08d}",
                underlying="TST", strike=float(k), expiry=expiry, option_type=otype,
                open_interest=50, tradable=True, bid=mid - 0.05, ask=mid + 0.05, last=mid,
                alpaca_iv=0.15 + 0.0003 * abs(k - spot), alpaca_delta=delta,
                alpaca_gamma=0.01, alpaca_theta=-0.02, alpaca_vega=0.3,
            ))
    return chain


def test_select_expiry_picks_closest_within_tolerance():
    today = dt.date.today()
    contracts = [
        ChainContract("A", "TST", 100, today + dt.timedelta(days=2), OptionType.CALL, 10, True, 1, 1.1, 1, None, None, None, None, None),
        ChainContract("B", "TST", 100, today + dt.timedelta(days=9), OptionType.CALL, 10, True, 1, 1.1, 1, None, None, None, None, None),
    ]
    picked = select_expiry(contracts, target_dte_days=7, tolerance_days=3)
    assert picked == today + dt.timedelta(days=9)


def test_select_expiry_returns_none_outside_tolerance():
    today = dt.date.today()
    contracts = [ChainContract("A", "TST", 100, today + dt.timedelta(days=40), OptionType.CALL, 10, True, 1, 1.1, 1, None, None, None, None, None)]
    assert select_expiry(contracts, target_dte_days=7, tolerance_days=3) is None


def test_build_straddle_picks_atm_strike():
    spot = 100.0
    expiry = dt.date.today() + dt.timedelta(days=7)
    chain = _synthetic_chain(spot, expiry)
    straddle = build_straddle(chain, expiry, spot, 0.04)
    assert straddle is not None
    assert straddle.call.strike == straddle.put.strike
    assert abs(straddle.call.strike - spot) <= 5


def test_build_iron_condor_is_covered_and_within_budget():
    spot = 100.0
    expiry = dt.date.today() + dt.timedelta(days=7)
    chain = _synthetic_chain(spot, expiry)
    condor = build_iron_condor(chain, expiry, spot, 0.04, short_delta=0.16, wing_delta=0.05, max_loss_budget=2000, max_contracts=20)
    assert condor is not None
    # every short leg has a further-OTM long leg protecting it -- "covered"
    assert condor.long_call.strike > condor.short_call.strike
    assert condor.long_put.strike < condor.short_put.strike
    assert condor.max_loss_per_contract * condor.contracts <= 2000 + 1e-6
    assert condor.contracts >= 1


# ------------------------------------------------------------- OCC parse ---
@pytest.mark.parametrize("symbol,expected", [
    ("AAPL240119C00100000", ("AAPL", dt.date(2024, 1, 19), OptionType.CALL, 100.0)),
    ("SPY260905P00575000", ("SPY", dt.date(2026, 9, 5), OptionType.PUT, 575.0)),
    ("not-a-symbol", None),
])
def test_parse_occ_symbol(symbol, expected):
    assert parse_occ_symbol(symbol) == expected


# ------------------------------------------------ v2: disagreement damping ---
def test_signal_disagreement_actually_shrinks_the_edge():
    """v1 computed sas_agrees but the composite_edge magnitude was
    IDENTICAL whether or not SAS agreed -- the exact bug
    test_signal_sas_disagreement_is_flagged (above) didn't catch, since
    it only ever checked the boolean flag. This is the regression test
    for the fix: same VRP inputs, only the SAS sign differs, and the
    disagreeing case MUST produce a smaller-magnitude edge."""
    common = dict(
        regime_label="Range", forecast_vol_now=0.25, live_atm_iv=0.18,
        forecast_vol_history=_wobbly_history(0.20), realized_vol_history=_wobbly_history(0.15, seed=2),
    )
    agree = build_signal(**common, sas_values=[-0.06, -0.05, -0.07])
    disagree = build_signal(**common, sas_values=[0.06, 0.05, 0.07])
    assert agree.sas_agrees is True
    assert disagree.sas_agrees is False
    assert disagree.damping_applied is True
    assert abs(disagree.composite_edge) < abs(agree.composite_edge)


def test_signal_heston_leg_low_confidence_is_excluded():
    common = dict(
        regime_label="Range", forecast_vol_now=0.25, live_atm_iv=0.18,
        forecast_vol_history=_wobbly_history(0.20), realized_vol_history=_wobbly_history(0.15, seed=2),
    )
    confident = build_signal(**common, heston_term_gap=0.05, heston_confident=True)
    low_conf = build_signal(**common, heston_term_gap=0.05, heston_confident=False)
    assert "heston" in confident.legs_used
    assert "heston" not in low_conf.legs_used
    assert low_conf.heston_term_gap is None


def test_signal_three_legs_blend_when_all_available():
    sig = build_signal(
        regime_label="Range", forecast_vol_now=0.25, live_atm_iv=0.18,
        forecast_vol_history=_wobbly_history(0.20), realized_vol_history=_wobbly_history(0.15, seed=2),
        sas_values=[-0.05], heston_term_gap=0.04, heston_confident=True,
    )
    assert sig.legs_used == ["vrp", "sas", "heston"]


# --------------------------------------------- v2: position-aware sizing ---
def test_decide_sizing_nets_against_existing_long_vol_exposure():
    risk = RiskLimits()
    fresh = decide_sizing(
        composite_edge=0.8, regime_label="Range", capital=1_000_000,
        straddle_price=5.0, straddle_gamma=0.01, straddle_vega=1.0, risk=risk,
    )
    assert fresh.contracts > 0 and not fresh.already_at_target

    at_target = decide_sizing(
        composite_edge=0.8, regime_label="Range", capital=1_000_000,
        straddle_price=5.0, straddle_gamma=0.01, straddle_vega=1.0, risk=risk,
        deployed_long_vol_notional=fresh.risk_budget_dollars,
    )
    assert at_target.contracts == 0
    assert at_target.already_at_target is True

    partial = decide_sizing(
        composite_edge=0.8, regime_label="Range", capital=1_000_000,
        straddle_price=5.0, straddle_gamma=0.01, straddle_vega=1.0, risk=risk,
        deployed_long_vol_notional=fresh.risk_budget_dollars * 0.5,
    )
    assert 0 < partial.contracts <= fresh.contracts


def test_decide_sizing_cost_floor_blocks_sub_cost_edges():
    risk = RiskLimits()
    blocked = decide_sizing(
        composite_edge=0.8, regime_label="Range", capital=1_000_000,
        straddle_price=5.0, straddle_gamma=0.01, straddle_vega=1.0, risk=risk,
        vrp_raw=0.001, cost_floor_vol_pts=0.01,
    )
    assert blocked.direction == "flat"
    assert "cost_floor" in blocked.capped_by

    allowed = decide_sizing(
        composite_edge=0.8, regime_label="Range", capital=1_000_000,
        straddle_price=5.0, straddle_gamma=0.01, straddle_vega=1.0, risk=risk,
        vrp_raw=0.05, cost_floor_vol_pts=0.01,
    )
    assert allowed.direction == "long_vol"


def test_kelly_fraction_extra_scale_throttles_size():
    risk = RiskLimits()
    full = kelly_fraction(composite_edge=0.5, regime_label="Range", risk=risk, extra_scale=1.0)
    throttled = kelly_fraction(composite_edge=0.5, regime_label="Range", risk=risk, extra_scale=0.5)
    assert abs(throttled) == pytest.approx(abs(full) * 0.5, rel=1e-6)


# -------------------------------------- v2: existing_vol_exposure_notional ---
def test_existing_vol_exposure_notional_splits_by_sign():
    from strategy.portfolio import SHORT_VOL_RISK_MULTIPLIER, PositionGreeks, existing_vol_exposure_notional

    positions = [
        PositionGreeks(symbol="LONG1", qty=2, market_value=1000.0, delta_shares=0, gamma_shares=0, vega_dollars=0, theta_dollars=0),
        PositionGreeks(symbol="SHORT1", qty=-3, market_value=-300.0, delta_shares=0, gamma_shares=0, vega_dollars=0, theta_dollars=0),
    ]
    long_notional, short_notional = existing_vol_exposure_notional(positions)
    assert long_notional == pytest.approx(1000.0)
    assert short_notional == pytest.approx(300.0 * SHORT_VOL_RISK_MULTIPLIER)
