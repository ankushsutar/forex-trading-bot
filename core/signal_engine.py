"""
core/signal_engine.py
---------------------
Zero-repaint quantitative signal generator operating strictly on closed candles.
Evaluates:
1. Volatility Squeeze Breakout (Bollinger / Keltner expansion + volume spike)
2. Liquidity Sweep Reversals (Session high/low sweep + wick rejection + MSS)
Ensures full alignment with RegimeFilter verdicts and multi-timeframe structure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from config.symbols import SymbolConfig, SYMBOL_REGISTRY
from indicators import (
    calculate_bollinger_bands,
    calculate_keltner_channels,
    detect_squeeze,
    calculate_squeeze_momentum,
    calculate_atr,
    calculate_sma,
    calculate_ema,
    get_session_high_low
)
from core.regime_filter import RegimeFilter, MarketRegime, RegimeVerdict


@dataclass
class SignalEvent:
    symbol: str
    direction: str          # "BUY" or "SELL"
    archetype: str          # "SQUEEZE_BREAKOUT" or "LIQUIDITY_SWEEP"
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    confluences: List[str]
    bar_time: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)


class SignalEngine:
    """
    Evaluates closed bars and regime verdicts to generate validated trading signals.
    """

    def __init__(self, regime_filter: Optional[RegimeFilter] = None):
        self.regime_filter = regime_filter or RegimeFilter()

    def evaluate(
        self,
        symbol: str,
        df_ltf: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        spread_bps: float = 0.0,
        funding_rate: float = 0.0,
        mark_price: Optional[float] = None
    ) -> Optional[SignalEvent]:
        """
        Evaluates strictly closed bars.
        df_ltf must contain completed closed candles (latest row is last closed bar).
        """
        if len(df_ltf) < 50:
            return None

        # 1. Evaluate Quantitative Market Regime
        verdict = self.regime_filter.classify_regime(
            symbol=symbol,
            df_ltf=df_ltf,
            df_htf=df_htf,
            spread_bps=spread_bps,
            funding_rate=funding_rate
        )

        if not verdict.is_tradeable:
            return None

        cfg = SYMBOL_REGISTRY.get(symbol)
        atr_mult = cfg.atr_stop_multiplier if cfg else 1.5

        # 2. Extract Indicator Matrices on Closed Bars
        atr_series = calculate_atr(df_ltf, period=14)
        latest_atr = float(atr_series.iloc[-1])

        # Current closed bar is index -1
        bar = df_ltf.iloc[-1]
        prev_bar = df_ltf.iloc[-2]
        close = float(bar['close'])
        open_p = float(bar['open'])
        high = float(bar['high'])
        low = float(bar['low'])
        bar_time = df_ltf.index[-1]
        candle_range = max(high - low, 0.0001)

        # ---------------------------------------------------------------------
        # ARCHETYPE 1: VOLATILITY SQUEEZE BREAKOUT (Trend Continuation)
        # ---------------------------------------------------------------------
        if verdict.regime in (MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR):
            bb_df = calculate_bollinger_bands(df_ltf['close'], 20, 2.0)
            kc_df = calculate_keltner_channels(df_ltf, 20, 1.5, 14)
            squeeze_series = detect_squeeze(df_ltf, 20, 2.0, 20, 1.5)
            momentum_series = calculate_squeeze_momentum(df_ltf, 20, 20)

            # Lookback 2 to 6 bars for squeeze compression
            prior_squeeze = bool(squeeze_series.iloc[-6:-1].any())
            currently_in_squeeze = bool(squeeze_series.iloc[-1])
            squeeze_fired = prior_squeeze and not currently_in_squeeze

            vol_col = 'volume' if 'volume' in df_ltf.columns else 'tick_volume'
            vol_sma = calculate_sma(df_ltf[vol_col].astype(float), 20)
            curr_vol = float(bar[vol_col])
            avg_vol = float(vol_sma.iloc[-1]) if not np.isnan(vol_sma.iloc[-1]) else curr_vol
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
            volume_spike = vol_ratio >= 1.4

            upper_bb = float(bb_df['bb_upper'].iloc[-1])
            lower_bb = float(bb_df['bb_lower'].iloc[-1])
            latest_mom = float(momentum_series.iloc[-1])

            # Bullish Breakout
            if "BUY" in verdict.allowed_directions:
                if (close > upper_bb) and (latest_mom > 0) and volume_spike and (squeeze_fired or not currently_in_squeeze):
                    sl = max(lower_bb, close - (atr_mult * latest_atr))
                    risk_dist = abs(close - sl)
                    if risk_dist > 0:
                        tp1 = close + (2.5 * risk_dist)
                        tp2 = close + (4.0 * risk_dist)
                        confluences = [
                            f"Volatility squeeze expansion release (Vol ratio: {vol_ratio:.2f}x)",
                            f"Squeeze momentum positive ({latest_mom:.2f})",
                            f"Regime confirmed: {verdict.regime.value} (ADX: {verdict.metrics.get('adx')})",
                            f"Spread healthy ({spread_bps:.1f} bps <= {cfg.max_spread_bps if cfg else 8.0} bps)"
                        ]
                        return SignalEvent(
                            symbol=symbol,
                            direction="BUY",
                            archetype="SQUEEZE_BREAKOUT",
                            entry_price=close,
                            stop_loss=sl,
                            take_profit_1=tp1,
                            take_profit_2=tp2,
                            risk_reward_ratio=2.5,
                            confluences=confluences,
                            bar_time=bar_time,
                            metrics={
                                "vol_ratio": vol_ratio,
                                "momentum": latest_mom,
                                "regime": verdict.regime.value
                            }
                        )

            # Bearish Breakdown
            if "SELL" in verdict.allowed_directions:
                if (close < lower_bb) and (latest_mom < 0) and volume_spike and (squeeze_fired or not currently_in_squeeze):
                    sl = min(upper_bb, close + (atr_mult * latest_atr))
                    risk_dist = abs(close - sl)
                    if risk_dist > 0:
                        tp1 = close - (2.5 * risk_dist)
                        tp2 = close - (4.0 * risk_dist)
                        confluences = [
                            f"Volatility squeeze breakdown release (Vol ratio: {vol_ratio:.2f}x)",
                            f"Squeeze momentum negative ({latest_mom:.2f})",
                            f"Regime confirmed: {verdict.regime.value} (ADX: {verdict.metrics.get('adx')})",
                            f"Spread healthy ({spread_bps:.1f} bps)"
                        ]
                        return SignalEvent(
                            symbol=symbol,
                            direction="SELL",
                            archetype="SQUEEZE_BREAKOUT",
                            entry_price=close,
                            stop_loss=sl,
                            take_profit_1=tp1,
                            take_profit_2=tp2,
                            risk_reward_ratio=2.5,
                            confluences=confluences,
                            bar_time=bar_time,
                            metrics={
                                "vol_ratio": vol_ratio,
                                "momentum": latest_mom,
                                "regime": verdict.regime.value
                            }
                        )

        # ---------------------------------------------------------------------
        # ARCHETYPE 2: LIQUIDITY SWEEP & REVERSAL (Mean Reversion)
        # ---------------------------------------------------------------------
        elif verdict.regime == MarketRegime.MEAN_REVERTING:
            # Lookback recent 20 bars high/low as liquidity reference
            recent_high = float(df_ltf['high'].iloc[-21:-1].max())
            recent_low = float(df_ltf['low'].iloc[-21:-1].min())
            ema9 = calculate_ema(df_ltf['close'], 9)

            # Bullish Sweep of Lows
            lower_wick = min(open_p, close) - low
            rejection_wick_bull = (lower_wick / candle_range) >= 0.40
            swept_low = (low < recent_low) and (close > recent_low)

            if "BUY" in verdict.allowed_directions and swept_low and rejection_wick_bull:
                sl = low - (0.5 * latest_atr)
                risk_dist = abs(close - sl)
                if risk_dist > 0:
                    tp1 = close + (2.5 * risk_dist)
                    tp2 = recent_high
                    confluences = [
                        f"Liquidity swept below recent low ({recent_low:.2f})",
                        f"Rejection wick printed ({(lower_wick / candle_range):.1%})",
                        f"Mean-reverting regime confirmed (ADX: {verdict.metrics.get('adx')})",
                        f"Spread healthy ({spread_bps:.1f} bps)"
                    ]
                    return SignalEvent(
                        symbol=symbol,
                        direction="BUY",
                        archetype="LIQUIDITY_SWEEP",
                        entry_price=close,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        risk_reward_ratio=2.5,
                        confluences=confluences,
                        bar_time=bar_time,
                        metrics={"swept_level": recent_low, "regime": verdict.regime.value}
                    )

            # Bearish Sweep of Highs
            upper_wick = high - max(open_p, close)
            rejection_wick_bear = (upper_wick / candle_range) >= 0.40
            swept_high = (high > recent_high) and (close < recent_high)

            if "SELL" in verdict.allowed_directions and swept_high and rejection_wick_bear:
                sl = high + (0.5 * latest_atr)
                risk_dist = abs(close - sl)
                if risk_dist > 0:
                    tp1 = close - (2.5 * risk_dist)
                    tp2 = recent_low
                    confluences = [
                        f"Liquidity swept above recent high ({recent_high:.2f})",
                        f"Rejection wick printed ({(upper_wick / candle_range):.1%})",
                        f"Mean-reverting regime confirmed (ADX: {verdict.metrics.get('adx')})",
                        f"Spread healthy ({spread_bps:.1f} bps)"
                    ]
                    return SignalEvent(
                        symbol=symbol,
                        direction="SELL",
                        archetype="LIQUIDITY_SWEEP",
                        entry_price=close,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        risk_reward_ratio=2.5,
                        confluences=confluences,
                        bar_time=bar_time,
                        metrics={"swept_level": recent_high, "regime": verdict.regime.value}
                    )

        return None
