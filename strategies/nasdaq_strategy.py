"""
strategies/nasdaq_strategy.py
-----------------------------
Nasdaq US Cash Open (14:30 UTC) Opening Range Breakout (ORB) Strategy.
Computes the 15-minute Opening Range (14:30 - 14:45 UTC), anchors intraday
VWAP from the cash open, and trades trend-aligned breakouts with ADX > 25.
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, time
import pandas as pd
import numpy as np

from config import cfg
from indicators import (
    calculate_vwap,
    calculate_atr,
    calculate_adx,
    calculate_sma
)
from regime_filter import RegimeFilter
from risk_engine import RiskEngine
from strategies.base_strategy import BaseStrategy, TradeSignal

logger = logging.getLogger("NasdaqStrategy")


class NasdaqORBStrategy(BaseStrategy):
    """
    US Cash Open Opening Range Breakout (ORB) Strategy for USTEC / NAS100 / US100.
    """

    def __init__(self):
        super().__init__(name="Nasdaq US Cash Open ORB + VWAP", asset="NASDAQ")
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
        Evaluates Nasdaq ORB conditions on M5 and H1 timeframes.
        """
        df_m5 = data_frames.get("M5")
        df_h1 = data_frames.get("H1")
        df_h4 = data_frames.get("H4", df_h1)

        if df_m5 is None or df_h1 is None or len(df_m5) < 60 or len(df_h1) < 50:
            self.last_status = "Insufficient historical bars"
            return None

        # Stage 1: Live Spread and Session Window (14:30 - 17:00 UTC)
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
            logger.debug(f"[{symbol}] Stage 1 failed: {spread_decision.reason}")
            return None

        # 1. Extract the 15-minute Opening Range (14:30 - 14:45 UTC) for current day
        orb_high, orb_low = self._calculate_opening_range(df_m5)
        if orb_high is None or orb_low is None:
            self.last_status = "15-min Opening Range (14:30-14:45 UTC) not yet completed"
            return None

        orb_midpoint = (orb_high + orb_low) / 2.0

        # 2. Compute Intraday VWAP anchored to 14:30 UTC
        vwap_series = calculate_vwap(df_m5, anchor_hour=14, anchor_minute=30)
        curr_vwap = float(vwap_series.iloc[-1])

        # 3. Bar Dynamics and Technical Indicators
        curr_bar = df_m5.iloc[-1]
        prev_bar = df_m5.iloc[-2]
        close = float(curr_bar['close'])
        open_p = float(curr_bar['open'])
        bar_time = df_m5.index[-1]

        # Enforce execution window: must be AFTER opening range formed (>= 14:45 UTC)
        if bar_time.hour == 14 and bar_time.minute < 45:
            self.last_status = "Waiting for 15-min Opening Range to conclude (until 14:45 UTC)"
            return None

        atr_series = calculate_atr(df_m5, period=cfg.regime.atr_period)
        latest_atr = float(atr_series.iloc[-1])

        vol_col = 'tick_volume' if 'tick_volume' in df_m5.columns else 'volume'
        vol_sma = calculate_sma(df_m5[vol_col].astype(float), 20)
        curr_vol = float(curr_bar[vol_col])
        avg_vol = float(vol_sma.iloc[-1]) if not np.isnan(vol_sma.iloc[-1]) else curr_vol
        has_volume = curr_vol >= (avg_vol * 1.15)

        # ---------------------------------------------------------------------
        # BULLISH ORB BREAKOUT: Close above ORB High + Price above VWAP
        # ---------------------------------------------------------------------
        bullish_orb = (close > orb_high) and (float(prev_bar['close']) <= orb_high or open_p <= orb_high)
        above_vwap = close > curr_vwap

        if bullish_orb and above_vwap:
            regime_decision = self.regime_filter.evaluate_mtf_regime(
                df_h1=df_h1,
                df_h4=df_h4,
                df_m5=df_m5,
                strategy_type="BREAKOUT",
                direction="BUY"
            )
            if not regime_decision.passed:
                self.last_status = regime_decision.reason
                return None

            confluences = [
                f"15-min US Cash Opening Range High broken ({close:.2f} > {orb_high:.2f})",
                f"Intraday Cash Open VWAP alignment ({close:.2f} > {curr_vwap:.2f})",
                f"M5 ADX trend expansion confirmed ({regime_decision.metrics.get('adx', 0):.1f} > {cfg.regime.adx_breakout_threshold})",
                f"H1 macro trend aligned bullish (EMA50 > EMA200)",
                f"Volume surge on breakout ({curr_vol:.0f} > 1.15x SMA20)",
                f"Live spread verified ({live_tick['spread_price']:.2f} <= {cfg.gatekeeper.max_spread_nasdaq:.2f})"
            ]

            structural_sl = max(orb_midpoint, orb_high - (cfg.risk.atr_stop_multiplier * latest_atr))

            risk_profile = self.risk_engine.compute_trade_risk(
                account_equity=account_equity,
                entry_price=close,
                stop_loss=structural_sl,
                symbol_specs=symbol_specs,
                target_tp1=close + (cfg.risk.tp1_rr_ratio * abs(close - structural_sl)),
                target_tp2=close + (cfg.risk.tp2_rr_ratio * abs(close - structural_sl)),
                direction="BUY"
            )

            if not risk_profile.is_valid:
                self.last_status = f"Risk rejected: {risk_profile.rejection_reason}"
                return None

            passed_confluence, conf_msg = self.risk_engine.validate_confluences(confluences)
            if not passed_confluence:
                self.last_status = conf_msg
                return None

            self.last_status = f"BUY Breakout Emitted @ {close:.2f} (SL: {structural_sl:.2f})"
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
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "vwap": curr_vwap,
                    "adx": regime_decision.metrics.get("adx"),
                    "spread_price": live_tick["spread_price"]
                }
            )

        # ---------------------------------------------------------------------
        # BEARISH ORB BREAKDOWN: Close below ORB Low + Price below VWAP
        # ---------------------------------------------------------------------
        bearish_orb = (close < orb_low) and (float(prev_bar['close']) >= orb_low or open_p >= orb_low)
        below_vwap = close < curr_vwap

        if bearish_orb and below_vwap:
            regime_decision = self.regime_filter.evaluate_mtf_regime(
                df_h1=df_h1,
                df_h4=df_h4,
                df_m5=df_m5,
                strategy_type="BREAKOUT",
                direction="SELL"
            )
            if not regime_decision.passed:
                self.last_status = regime_decision.reason
                return None

            confluences = [
                f"15-min US Cash Opening Range Low broken ({close:.2f} < {orb_low:.2f})",
                f"Intraday Cash Open VWAP alignment ({close:.2f} < {curr_vwap:.2f})",
                f"M5 ADX trend expansion confirmed ({regime_decision.metrics.get('adx', 0):.1f} > {cfg.regime.adx_breakout_threshold})",
                f"H1 macro trend aligned bearish (EMA50 < EMA200)",
                f"Volume surge on breakdown ({curr_vol:.0f} > 1.15x SMA20)",
                f"Live spread verified ({live_tick['spread_price']:.2f} <= {cfg.gatekeeper.max_spread_nasdaq:.2f})"
            ]

            structural_sl = min(orb_midpoint, orb_low + (cfg.risk.atr_stop_multiplier * latest_atr))

            risk_profile = self.risk_engine.compute_trade_risk(
                account_equity=account_equity,
                entry_price=close,
                stop_loss=structural_sl,
                symbol_specs=symbol_specs,
                target_tp1=close - (cfg.risk.tp1_rr_ratio * abs(close - structural_sl)),
                target_tp2=close - (cfg.risk.tp2_rr_ratio * abs(close - structural_sl)),
                direction="SELL"
            )

            if not risk_profile.is_valid:
                self.last_status = f"Risk rejected: {risk_profile.rejection_reason}"
                return None

            passed_confluence, conf_msg = self.risk_engine.validate_confluences(confluences)
            if not passed_confluence:
                self.last_status = conf_msg
                return None

            self.last_status = f"SELL Breakdown Emitted @ {close:.2f} (SL: {structural_sl:.2f})"
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
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "vwap": curr_vwap,
                    "adx": regime_decision.metrics.get("adx"),
                    "spread_price": live_tick["spread_price"]
                }
            )

        self.last_status = f"Inside Opening Range ({orb_low:.2f} - {orb_high:.2f})"
        return None

    def _calculate_opening_range(self, df_m5: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Extracts high and low between 14:30 and 14:45 UTC for the most recent day.
        """
        times = df_m5.index
        # Bars at 14:30, 14:35, 14:40
        orb_mask = (times.hour == 14) & (times.minute >= 30) & (times.minute < 45)
        orb_bars = df_m5[orb_mask]

        if orb_bars.empty:
            return None, None

        latest_date = orb_bars.index[-1].date()
        today_orb = orb_bars[orb_bars.index.date == latest_date]

        if len(today_orb) < 2:
            return None, None

        return float(today_orb['high'].max()), float(today_orb['low'].min())
