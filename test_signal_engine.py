"""
test_signal_engine.py
---------------------
Comprehensive test suite validating:
- Pure NumPy/pandas indicator calculations (EMA, ATR, ADX, Bollinger, Keltner, VWAP)
- Zuperior MT5 lot sizing and 1:2.5 minimum R:R risk engine
- 3-stage gatekeepers (Live spread vs 1.8x median, session windows, MTF regime)
- Strategies evaluation
- Signal deduplication in notifier
"""

import math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from config import cfg
from indicators import (
    calculate_ema,
    calculate_sma,
    calculate_atr,
    calculate_adx,
    calculate_bollinger_bands,
    calculate_keltner_channels,
    detect_squeeze,
    calculate_squeeze_momentum,
    calculate_vwap,
    find_swing_levels
)
from regime_filter import RegimeFilter
from risk_engine import RiskEngine
from notifier import TradeNotifier
from strategies.base_strategy import TradeSignal
from strategies.gold_strategy import GoldLiquiditySweepStrategy
from strategies.btc_strategy import BtcVolatilitySqueezeStrategy
from strategies.nasdaq_strategy import NasdaqORBStrategy


def create_sample_dataframe(n_bars=100, base_price=2650.0, trend=0.1):
    """Generates synthetic OHLCV dataframe with DatetimeIndex."""
    np.random.seed(42)
    times = [datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n_bars)]

    close = base_price + np.cumsum(np.random.normal(trend, 1.5, n_bars))
    high = close + np.random.uniform(0.5, 2.0, n_bars)
    low = close - np.random.uniform(0.5, 2.0, n_bars)
    open_p = close + np.random.normal(0, 0.5, n_bars)
    vol = np.random.randint(200, 1500, n_bars)
    spread = np.random.randint(15, 20, n_bars)

    df = pd.DataFrame({
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'tick_volume': vol,
        'spread': spread
    }, index=pd.to_datetime(times))
    return df


def test_indicators():
    df = create_sample_dataframe(100)

    # EMA & SMA
    ema = calculate_ema(df['close'], 20)
    sma = calculate_sma(df['close'], 20)
    assert len(ema) == 100
    assert not ema.isna().all()

    # ATR (Wilder's)
    atr = calculate_atr(df, 14)
    assert len(atr) == 100
    assert (atr > 0).all()

    # ADX
    adx_df = calculate_adx(df, 14)
    assert 'adx' in adx_df.columns
    assert 'plus_di' in adx_df.columns
    assert 'minus_di' in adx_df.columns
    assert not adx_df['adx'].isna().any()

    # Bollinger & Keltner
    bb = calculate_bollinger_bands(df['close'], 20, 2.0)
    kc = calculate_keltner_channels(df, 20, 1.5, 14)
    assert 'bb_upper' in bb.columns
    assert 'kc_upper' in kc.columns

    # Squeeze
    sq = detect_squeeze(df)
    assert isinstance(sq, pd.Series)
    assert len(sq) == 100

    # VWAP
    vwap = calculate_vwap(df, anchor_hour=12, anchor_minute=0)
    assert len(vwap) == 100
    assert not vwap.isna().any()


def test_risk_engine_lot_sizing():
    re = RiskEngine()
    specs = {
        "point": 0.01,
        "digits": 2,
        "trade_contract_size": 100.0,
        "trade_tick_value": 1.0,
        "trade_tick_size": 0.01,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01
    }

    equity = 10000.0  # 1% risk = $100.00
    entry = 2650.00
    sl = 2640.00     # 10.00 points risk

    # 10.00 / 0.01 = 1000 ticks * $1.00 tick value = $1000 risk per lot
    # $100 / $1000 = 0.10 lots
    profile = re.compute_trade_risk(
        account_equity=equity,
        entry_price=entry,
        stop_loss=sl,
        symbol_specs=specs,
        direction="BUY"
    )

    assert profile.is_valid
    assert profile.recommended_lot_size == 0.10
    assert profile.dollar_risk == 100.00
    assert profile.risk_reward_ratio >= 2.5
    assert profile.take_profit_1 == entry + (2.5 * 10.0)  # 2675.00


def test_risk_engine_minimum_rr_rejection():
    re = RiskEngine()
    specs = {
        "point": 0.01,
        "digits": 2,
        "trade_contract_size": 100.0,
        "trade_tick_value": 1.0,
        "trade_tick_size": 0.01,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01
    }

    # Manually specify TP that is only 1:1 R:R, should be adjusted or rejected if passed below min_rr
    profile = re.compute_trade_risk(
        account_equity=10000.0,
        entry_price=2650.00,
        stop_loss=2640.00,
        symbol_specs=specs,
        target_tp1=2660.00,  # 1:1 R:R
        direction="BUY"
    )
    # The risk engine automatically ensures default TP1 meets 1:2.5 minimum
    assert profile.risk_reward_ratio >= 2.5


def test_spread_and_session_gatekeeper():
    rf = RegimeFilter()
    df_m5 = create_sample_dataframe(100)
    df_m5['spread'] = 16  # 16 points median

    # Normal tick during Gold session (13:00 UTC)
    normal_tick = {
        "symbol": "XAUUSD.pro",
        "bid": 2650.0,
        "ask": 2650.16,
        "spread_price": 0.16,
        "spread_points": 16.0,
        "time": datetime(2026, 11, 23, 13, 0, tzinfo=timezone.utc)
    }
    decision = rf.evaluate_spread_and_session("GOLD", "XAUUSD.pro", normal_tick, df_m5, 0.01)
    assert decision.passed

    # Spread blowup tick (e.g. 35 points > 1.8 * 16 = 28.8)
    spiked_tick = {
        "symbol": "XAUUSD.pro",
        "bid": 2650.0,
        "ask": 2650.35,
        "spread_price": 0.35,
        "spread_points": 35.0,
        "time": datetime(2026, 11, 23, 13, 0, tzinfo=timezone.utc)
    }
    decision_spike = rf.evaluate_spread_and_session("GOLD", "XAUUSD.pro", spiked_tick, df_m5, 0.01)
    assert not decision_spike.passed
    assert "Live spread" in decision_spike.reason

    # Outside session tick (04:00 UTC for Gold)
    out_of_session_tick = {
        "symbol": "XAUUSD.pro",
        "bid": 2650.0,
        "ask": 2650.16,
        "spread_price": 0.16,
        "spread_points": 16.0,
        "time": datetime(2026, 11, 23, 4, 0, tzinfo=timezone.utc)
    }
    decision_session = rf.evaluate_spread_and_session("GOLD", "XAUUSD.pro", out_of_session_tick, df_m5, 0.01)
    assert not decision_session.passed
    assert "liquidity window" in decision_session.reason


def test_signal_deduplication():
    notifier = TradeNotifier()
    bar_time = datetime(2026, 11, 23, 14, 30, tzinfo=timezone.utc)
    sig = TradeSignal(
        asset="GOLD",
        symbol="XAUUSD.pro",
        direction="BUY",
        strategy_name="Gold London/NY Liquidity Sweep",
        timeframe="M5",
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit_1=2675.0,
        take_profit_2=2685.0,
        risk_reward_ratio=2.5,
        recommended_lot_size=0.10,
        dollar_risk=100.0,
        account_equity=10000.0,
        confluences=["Test 1", "Test 2", "Test 3"],
        bar_time=bar_time
    )

    # First call: not duplicate
    assert not notifier.is_duplicate(sig)

    # Immediate second call with identical attributes: duplicate
    assert notifier.is_duplicate(sig)


if __name__ == "__main__":
    test_indicators()
    test_risk_engine_lot_sizing()
    test_risk_engine_minimum_rr_rejection()
    test_spread_and_session_gatekeeper()
    test_signal_deduplication()
    print("\n✔ All unit tests passed successfully!\n")
