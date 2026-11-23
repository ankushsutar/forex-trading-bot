"""
strategies/btc_strategy.py
--------------------------
Bitcoin Volatility Squeeze Breakout Strategy tailored for Zuperior MT5.
Detects Bollinger Bands compression inside Keltner Channels followed by
an explosive expansion confirmed by an institutional tick-volume spike.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

from config import cfg
from indicators import (
    calculate_bollinger_bands,
    calculate_keltner_channels,
    detect_squeeze,
    calculate_squeeze_momentum,
    calculate_atr,
    calculate_adx,
    calculate_sma,
    calculate_ema
)
from regime_filter import RegimeFilter
from risk_engine import RiskEngine
from strategies.base_strategy import BaseStrategy, TradeSignal

logger = logging.getLogger("BtcStrategy")


class BtcVolatilitySqueezeStrategy(BaseStrategy):
    """
    Bitcoin Volatility Squeeze Breakout Strategy.
    """

    def __init__(self):
        super().__init__(name="Bitcoin Volatility Squeeze Breakout", asset="BITCOIN")
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
        Evaluates Bitcoin market conditions across M5/M15 and H1 timeframes.
        """
        df_exec = data_frames.get("M15", data_frames.get("M5"))
        df_h1 = data_frames.get("H1")
        df_h4 = data_frames.get("H4", df_h1)

        if df_exec is None or df_h1 is None or len(df_exec) < 60 or len(df_h1) < 50:
            self.last_status = "Insufficient historical bars"
            return None

        # Stage 1: Live Spread and Session Gatekeeper (including weekend chop filter)
        point = symbol_specs.get("point", 0.01)
        spread_decision = self.regime_filter.evaluate_spread_and_session(
            asset_key=self.asset,
            symbol=symbol,
            live_tick=live_tick,
            df_m5=df_exec,
            point=point
        )
        if not spread_decision.passed:
            self.last_status = spread_decision.reason
            logger.debug(f"[{symbol}] Stage 1 failed: {spread_decision.reason}")
            return None

        # 1. Bollinger Bands & Keltner Channels
        bb_df = calculate_bollinger_bands(
            df_exec['close'],
            period=cfg.regime.bb_period,
            std_dev=cfg.regime.bb_std_dev
        )
        kc_df = calculate_keltner_channels(
            df_exec,
            period=cfg.regime.kc_period,
            atr_multiplier=cfg.regime.kc_atr_multiplier,
            atr_period=cfg.regime.atr_period
        )

        squeeze_series = detect_squeeze(
            df_exec,
            bb_period=cfg.regime.bb_period,
            bb_std=cfg.regime.bb_std_dev,
            kc_period=cfg.regime.kc_period,
            kc_mult=cfg.regime.kc_atr_multiplier
        )
        momentum_series = calculate_squeeze_momentum(df_exec)

        # Lookback to verify compression existed in the last 2 to 6 bars
        prior_squeeze = bool(squeeze_series.iloc[-6:-1].any())
        currently_in_squeeze = bool(squeeze_series.iloc[-1])
        squeeze_released = prior_squeeze and not currently_in_squeeze

        # 2. Volume Spike Confirmation
        vol_col = 'tick_volume' if 'tick_volume' in df_exec.columns else 'volume'
        volume_sma20 = calculate_sma(df_exec[vol_col].astype(float), 20)
        curr_vol = float(df_exec[vol_col].iloc[-1])
        avg_vol = float(volume_sma20.iloc[-1]) if not np.isnan(volume_sma20.iloc[-1]) else curr_vol
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
        has_volume_spike = vol_ratio >= cfg.regime.btc_volume_spike_multiplier

        # 3. Bar Dynamics
        curr_bar = df_exec.iloc[-1]
        close = float(curr_bar['close'])
        bar_time = df_exec.index[-1]
        latest_atr = float(calculate_atr(df_exec, period=cfg.regime.atr_period).iloc[-1])
        latest_mom = float(momentum_series.iloc[-1])

        upper_bb = float(bb_df['bb_upper'].iloc[-1])
        lower_bb = float(bb_df['bb_lower'].iloc[-1])
        upper_kc = float(kc_df['kc_upper'].iloc[-1])
        lower_kc = float(kc_df['kc_lower'].iloc[-1])

        # ---------------------------------------------------------------------
        # BULLISH BREAKOUT: Expansion above Upper Band + Positive Momentum Spike
        # ---------------------------------------------------------------------
        bullish_trigger = (close > upper_bb or close > upper_kc) and (latest_mom > 0) and (squeeze_released or not currently_in_squeeze)
        if bullish_trigger and has_volume_spike:
            regime_decision = self.regime_filter.evaluate_mtf_regime(
                df_h1=df_h1,
                df_h4=df_h4,
                df_m5=df_exec,
                strategy_type="BREAKOUT",
                direction="BUY"
            )
            if not regime_decision.passed:
                self.last_status = regime_decision.reason
                return None

            confluences = [
                "Bollinger/Keltner volatility squeeze expansion released",
                f"Tick-volume spike confirmed ({curr_vol:.0f} is {vol_ratio:.2f}x SMA20)",
                f"Squeeze momentum positive & rising ({latest_mom:.2f})",
                f"M15 ADX expansion confirmed ({regime_decision.metrics.get('adx', 0):.1f} > {cfg.regime.adx_breakout_threshold})",
                f"H1 macro trend alignment (EMA50: {regime_decision.metrics.get('h1_ema50', 0):.1f})",
                f"Live spread verified (${live_tick['spread_price']:.2f} <= ${cfg.gatekeeper.max_spread_btc:.2f})"
            ]

            structural_sl = max(lower_bb, close - (cfg.risk.atr_stop_multiplier * latest_atr))

            risk_profile = self.risk_engine.compute_trade_risk(
                account_equity=account_equity,
                entry_price=close,
                stop_loss=structural_sl,
                symbol_specs=symbol_specs,
                target_tp1=close + (cfg.risk.tp1_rr_ratio * abs(close - structural_sl)),
                target_tp2=close + (4.0 * abs(close - structural_sl)),
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
                timeframe="M15",
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
                    "volume_ratio": vol_ratio,
                    "momentum": latest_mom,
                    "adx": regime_decision.metrics.get("adx"),
                    "spread_price": live_tick["spread_price"]
                }
            )

        # ---------------------------------------------------------------------
        # BEARISH BREAKDOWN: Expansion below Lower Band + Negative Momentum Spike
        # ---------------------------------------------------------------------
        bearish_trigger = (close < lower_bb or close < lower_kc) and (latest_mom < 0) and (squeeze_released or not currently_in_squeeze)
        if bearish_trigger and has_volume_spike:
            regime_decision = self.regime_filter.evaluate_mtf_regime(
                df_h1=df_h1,
                df_h4=df_h4,
                df_m5=df_exec,
                strategy_type="BREAKOUT",
                direction="SELL"
            )
            if not regime_decision.passed:
                self.last_status = regime_decision.reason
                return None

            confluences = [
                "Bollinger/Keltner volatility squeeze breakdown released",
                f"Tick-volume spike confirmed ({curr_vol:.0f} is {vol_ratio:.2f}x SMA20)",
                f"Squeeze momentum negative ({latest_mom:.2f})",
                f"M15 ADX expansion confirmed ({regime_decision.metrics.get('adx', 0):.1f} > {cfg.regime.adx_breakout_threshold})",
                f"H1 macro trend alignment (EMA50: {regime_decision.metrics.get('h1_ema50', 0):.1f})",
                f"Live spread verified (${live_tick['spread_price']:.2f} <= ${cfg.gatekeeper.max_spread_btc:.2f})"
            ]

            structural_sl = min(upper_bb, close + (cfg.risk.atr_stop_multiplier * latest_atr))

            risk_profile = self.risk_engine.compute_trade_risk(
                account_equity=account_equity,
                entry_price=close,
                stop_loss=structural_sl,
                symbol_specs=symbol_specs,
                target_tp1=close - (cfg.risk.tp1_rr_ratio * abs(close - structural_sl)),
                target_tp2=close - (4.0 * abs(close - structural_sl)),
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
                timeframe="M15",
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
                    "volume_ratio": vol_ratio,
                    "momentum": latest_mom,
                    "adx": regime_decision.metrics.get("adx"),
                    "spread_price": live_tick["spread_price"]
                }
            )

        self.last_status = f"No squeeze breakout (In Squeeze: {currently_in_squeeze}, Vol Ratio: {vol_ratio:.2f}x)"
        return None
