"""
strategies/gold_strategy.py
---------------------------
Gold Liquidity Sweep & Pullback Reversal Strategy tailored for Zuperior MT5.
Operates during the London/NY session overlap (12:00 - 16:30 UTC).
Detects sweeps of Asian/early London highs/lows with rejection wicks,
market structure shift (MSS), and pullback confirmation.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

from config import cfg
from indicators import (
    calculate_ema,
    calculate_atr,
    calculate_adx,
    get_session_high_low
)
from regime_filter import RegimeFilter
from risk_engine import RiskEngine
from strategies.base_strategy import BaseStrategy, TradeSignal

logger = logging.getLogger("GoldStrategy")


class GoldLiquiditySweepStrategy(BaseStrategy):
    """
    Gold Liquidity Sweep Reversal Strategy for XAUUSD.
    """

    def __init__(self):
        super().__init__(name="Gold London/NY Liquidity Sweep", asset="GOLD")
        self.regime_filter = RegimeFilter()
        self.risk_engine = RiskEngine()

    def evaluate(
        self,
        symbol: str,
        data_frames: Dict[str, pd.DataFrame],
        live_tick: dict,
        symbol_specs: dict,
        account_equity: float
    ) -> Optional[TradeSignal]:
        """
        Evaluates Gold market conditions across M5, H1, and H4 timeframes.
        """
        df_m5 = data_frames.get("M5")
        df_h1 = data_frames.get("H1")
        df_h4 = data_frames.get("H4", df_h1)

        if df_m5 is None or df_h1 is None or len(df_m5) < 60 or len(df_h1) < 50:
            self.last_status = "Insufficient historical bars"
            return None

        # Stage 1: Live Spread & Session Liquidity Window Filter
        point = symbol_specs.get("point", 0.01)
        spread_decision = self.regime_filter.evaluate_spread_and_session(
            asset_key=self.asset,
            symbol=symbol,
            live_tick=live_tick,
            df_m5=df_m5,
            point=point
        )
        if not spread_decision.passed:
            self.last_status = spread_decision.reason
            logger.debug(f"[{symbol}] {spread_decision.stage} failed: {spread_decision.reason}")
            return None

        # Extract Asian Session High and Low (00:00 to 08:00 UTC)
        asian_high, asian_low = get_session_high_low(df_m5, start_hour=0, start_minute=0, end_hour=8, end_minute=0)
        # Extract Early London High and Low (08:00 to 12:00 UTC)
        london_high, london_low = get_session_high_low(df_m5, start_hour=8, start_minute=0, end_hour=12, end_minute=0)

        # Composite session reference levels
        valid_lows = [l for l in (asian_low, london_low) if l is not None]
        valid_highs = [h for h in (asian_high, london_high) if h is not None]

        if not valid_lows or not valid_highs:
            self.last_status = "Session levels (Asian/London) not established yet"
            return None

        key_low = min(valid_lows)
        key_high = max(valid_highs)

        # Technical Indicators on M5
        atr_m5 = calculate_atr(df_m5, period=cfg.regime.atr_period)
        latest_atr = float(atr_m5.iloc[-1])
        ema9 = calculate_ema(df_m5['close'], 9)

        adx_df = calculate_adx(df_m5, period=cfg.regime.adx_period)
        latest_adx = float(adx_df['adx'].iloc[-1])

        # Inspect latest completed bars
        curr_bar = df_m5.iloc[-1]
        prev_bar = df_m5.iloc[-2]
        bar_time = df_m5.index[-1]

        close = float(curr_bar['close'])
        open_price = float(curr_bar['open'])
        high = float(curr_bar['high'])
        low = float(curr_bar['low'])
        candle_range = max(high - low, 0.001)

        # ---------------------------------------------------------------------
        # BULLISH SWEEP: Liquidity grab below key session low
        # ---------------------------------------------------------------------
        swept_low = (low < key_low) or (float(prev_bar['low']) < key_low)
        closed_above_low = close > key_low
        lower_wick = min(open_price, close) - low
        rejection_wick_bull = (lower_wick / candle_range) >= 0.40
        pullback_confirmed_bull = close >= float(ema9.iloc[-1])

        if swept_low and closed_above_low and (rejection_wick_bull or pullback_confirmed_bull):
            regime_decision = self.regime_filter.evaluate_mtf_regime(
                df_h1=df_h1,
                df_h4=df_h4,
                df_m5=df_m5,
                strategy_type="LIQUIDITY_SWEEP",
                direction="BUY"
            )
            if not regime_decision.passed:
                self.last_status = regime_decision.reason
                return None

            confluences = [
                f"Session liquidity low swept ({key_low:.2f})",
                f"M5 rejection wick / close back inside range (Wick: {(lower_wick / candle_range):.1%})",
                f"Pullback confirmation above EMA9 ({close:.2f} >= {float(ema9.iloc[-1]):.2f})",
                f"Peak London/NY liquidity window active ({live_tick['time'].strftime('%H:%M')} UTC)",
                f"Zuperior live spread verified ({live_tick['spread_price']:.2f} <= {cfg.gatekeeper.max_spread_gold:.2f})"
            ]

            sweep_extreme_low = min(low, float(prev_bar['low']))
            stop_loss = sweep_extreme_low - (0.5 * latest_atr)

            risk_profile = self.risk_engine.compute_trade_risk(
                account_equity=account_equity,
                entry_price=close,
                stop_loss=stop_loss,
                symbol_specs=symbol_specs,
                target_tp1=close + (cfg.risk.tp1_rr_ratio * abs(close - stop_loss)),
                target_tp2=key_high,
                direction="BUY"
            )

            if not risk_profile.is_valid:
                self.last_status = f"Risk rejected: {risk_profile.rejection_reason}"
                return None

            passed_confluence, conf_msg = self.risk_engine.validate_confluences(confluences)
            if not passed_confluence:
                self.last_status = conf_msg
                return None

            self.last_status = f"BUY Signal Emitted @ {close:.2f} (SL: {stop_loss:.2f})"
            return TradeSignal(
                asset=self.asset,
                symbol=symbol,
                direction="BUY",
                strategy_name=self.name,
                timeframe="M5",
                entry_price=risk_profile.entry_price,
                stop_loss=risk_profile.stop_loss,
                take_profit_1=risk_profile.take_profit_1,
                take_profit_2=risk_profile.take_profit_2,
                risk_reward_ratio=risk_profile.risk_reward_ratio,
                recommended_lot_size=risk_profile.recommended_lot_size,
                dollar_risk=risk_profile.dollar_risk,
                account_equity=risk_profile.account_equity,
                confluences=confluences,
                bar_time=bar_time,
                metadata={
                    "session_key_low": key_low,
                    "session_key_high": key_high,
                    "atr": latest_atr,
                    "adx": latest_adx,
                    "spread_pts": live_tick["spread_points"],
                    "spread_price": live_tick["spread_price"]
                }
            )

        # ---------------------------------------------------------------------
        # BEARISH SWEEP: Liquidity grab above key session high
        # ---------------------------------------------------------------------
        swept_high = (high > key_high) or (float(prev_bar['high']) > key_high)
        closed_below_high = close < key_high
        upper_wick = high - max(open_price, close)
        rejection_wick_bear = (upper_wick / candle_range) >= 0.40
        pullback_confirmed_bear = close <= float(ema9.iloc[-1])

        if swept_high and closed_below_high and (rejection_wick_bear or pullback_confirmed_bear):
            regime_decision = self.regime_filter.evaluate_mtf_regime(
                df_h1=df_h1,
                df_h4=df_h4,
                df_m5=df_m5,
                strategy_type="LIQUIDITY_SWEEP",
                direction="SELL"
            )
            if not regime_decision.passed:
                self.last_status = regime_decision.reason
                return None

            confluences = [
                f"Session liquidity high swept ({key_high:.2f})",
                f"M5 rejection wick / close back inside range (Wick: {(upper_wick / candle_range):.1%})",
                f"Pullback confirmation below EMA9 ({close:.2f} <= {float(ema9.iloc[-1]):.2f})",
                f"Peak London/NY liquidity window active ({live_tick['time'].strftime('%H:%M')} UTC)",
                f"Zuperior live spread verified ({live_tick['spread_price']:.2f} <= {cfg.gatekeeper.max_spread_gold:.2f})"
            ]

            sweep_extreme_high = max(high, float(prev_bar['high']))
            stop_loss = sweep_extreme_high + (0.5 * latest_atr)

            risk_profile = self.risk_engine.compute_trade_risk(
                account_equity=account_equity,
                entry_price=close,
                stop_loss=stop_loss,
                symbol_specs=symbol_specs,
                target_tp1=close - (cfg.risk.tp1_rr_ratio * abs(close - stop_loss)),
                target_tp2=key_low,
                direction="SELL"
            )

            if not risk_profile.is_valid:
                self.last_status = f"Risk rejected: {risk_profile.rejection_reason}"
                return None

            passed_confluence, conf_msg = self.risk_engine.validate_confluences(confluences)
            if not passed_confluence:
                self.last_status = conf_msg
                return None

            self.last_status = f"SELL Signal Emitted @ {close:.2f} (SL: {stop_loss:.2f})"
            return TradeSignal(
                asset=self.asset,
                symbol=symbol,
                direction="SELL",
                strategy_name=self.name,
                timeframe="M5",
                entry_price=risk_profile.entry_price,
                stop_loss=risk_profile.stop_loss,
                take_profit_1=risk_profile.take_profit_1,
                take_profit_2=risk_profile.take_profit_2,
                risk_reward_ratio=risk_profile.risk_reward_ratio,
                recommended_lot_size=risk_profile.recommended_lot_size,
                dollar_risk=risk_profile.dollar_risk,
                account_equity=risk_profile.account_equity,
                confluences=confluences,
                bar_time=bar_time,
                metadata={
                    "session_key_high": key_high,
                    "session_key_low": key_low,
                    "atr": latest_atr,
                    "adx": latest_adx,
                    "spread_pts": live_tick["spread_points"],
                    "spread_price": live_tick["spread_price"]
                }
            )

        self.last_status = f"No liquidity sweep on M5 (Range: {key_low:.2f} - {key_high:.2f})"
        return None
