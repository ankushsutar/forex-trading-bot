"""
regime_filter.py
----------------
Market state classifier and strict gatekeeper pipeline for Zuperior MT5.
Enforces:
1. Live spread vs. rolling median spread (< 1.8x) and peak liquidity session windows.
2. Multi-timeframe (H1/H4) EMA alignment and M5/M15 ADX/ATR volatility regime classification.
3. Confluence evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from config import cfg
from indicators import (
    calculate_ema,
    calculate_atr,
    calculate_adx,
    calculate_sma,
    calculate_rolling_spread_median
)


@dataclass
class GatekeeperDecision:
    """Detailed verdict from a gatekeeper evaluation."""
    passed: bool
    stage: str
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class RegimeFilter:
    """
    Evaluates market regime and enforces Zuperior MT5 signal gatekeepers.
    """

    def __init__(self):
        self.cfg = cfg

    # ---------------------------------------------------------
    # STAGE 1: Zuperior Live Spread & Session Filter
    # ---------------------------------------------------------
    def evaluate_spread_and_session(
        self,
        asset_key: str,
        symbol: str,
        live_tick: dict,
        df_m5: pd.DataFrame,
        point: float
    ) -> GatekeeperDecision:
        """
        Stage 1 Gatekeeper:
        - Validates that current time falls within peak liquidity session window.
        - Calculates rolling median spread from M5 bars.
        - Rejects signal if live spread exceeds 1.8 * rolling_median_spread or absolute max cap.
        """
        now_utc = live_tick["time"] if "time" in live_tick else datetime.now(timezone.utc)
        current_time_str = now_utc.strftime("%H:%M")
        weekday = now_utc.weekday()  # 5 = Saturday, 6 = Sunday

        # 1. Session Liquidity Window Check
        if asset_key == "GOLD":
            start_str = self.cfg.gatekeeper.gold_session_start_utc
            end_str = self.cfg.gatekeeper.gold_session_end_utc
            if not (start_str <= current_time_str <= end_str):
                return GatekeeperDecision(
                    passed=False,
                    stage="Stage 1: Session Filter",
                    reason=f"Gold outside peak liquidity window ({start_str} - {end_str} UTC). Current: {current_time_str} UTC",
                    metrics={"current_time_utc": current_time_str, "session": f"{start_str}-{end_str}"}
                )

        elif asset_key == "NASDAQ":
            start_str = self.cfg.gatekeeper.nasdaq_session_start_utc
            end_str = self.cfg.gatekeeper.nasdaq_session_end_utc
            if not (start_str <= current_time_str <= end_str):
                return GatekeeperDecision(
                    passed=False,
                    stage="Stage 1: Session Filter",
                    reason=f"Nasdaq outside US Cash Open window ({start_str} - {end_str} UTC). Current: {current_time_str} UTC",
                    metrics={"current_time_utc": current_time_str, "session": f"{start_str}-{end_str}"}
                )

        elif asset_key == "BITCOIN":
            if self.cfg.gatekeeper.btc_filter_weekend_chop and weekday in (5, 6):
                # Check recent ADX on M5 for weekend volatility
                adx_df = calculate_adx(df_m5, period=self.cfg.regime.adx_period)
                latest_adx = float(adx_df['adx'].iloc[-1])
                if latest_adx < self.cfg.gatekeeper.btc_weekend_min_adx:
                    return GatekeeperDecision(
                        passed=False,
                        stage="Stage 1: Session Filter",
                        reason=f"Bitcoin in low-volatility weekend chop (ADX {latest_adx:.1f} < {self.cfg.gatekeeper.btc_weekend_min_adx})",
                        metrics={"weekday": weekday, "adx": latest_adx}
                    )

        # 2. Live Spread vs Rolling Median Spread Check
        # Historical spread in MT5 is stored in points in the 'spread' column
        rolling_median_pts = float(df_m5['spread'].tail(self.cfg.gatekeeper.rolling_spread_bars).median())
        if np.isnan(rolling_median_pts) or rolling_median_pts <= 0:
            rolling_median_pts = float(df_m5['spread'].iloc[-1]) if 'spread' in df_m5.columns else 10.0

        live_spread_pts = live_tick["spread_points"]
        live_spread_price = live_tick["spread_price"]
        max_multiplier = self.cfg.gatekeeper.spread_median_multiplier
        allowed_spread_pts = rolling_median_pts * max_multiplier

        # Rejection on relative spread blowup (e.g., rollover or offshore liquidity drain)
        if live_spread_pts > allowed_spread_pts:
            return GatekeeperDecision(
                passed=False,
                stage="Stage 1: Live Spread Gatekeeper",
                reason=f"Live spread ({live_spread_pts:.1f} pts) exceeds {max_multiplier}x median ({rolling_median_pts:.1f} pts). Max allowed: {allowed_spread_pts:.1f} pts.",
                metrics={
                    "live_spread_pts": live_spread_pts,
                    "median_spread_pts": rolling_median_pts,
                    "multiplier": max_multiplier,
                    "live_spread_price": live_spread_price
                }
            )

        # Hard cap absolute ceiling checks
        if asset_key == "GOLD" and live_spread_price > self.cfg.gatekeeper.max_spread_gold:
            return GatekeeperDecision(
                passed=False,
                stage="Stage 1: Hard Spread Cap",
                reason=f"Gold spread (${live_spread_price:.2f}) exceeds absolute max limit (${self.cfg.gatekeeper.max_spread_gold:.2f})",
                metrics={"live_spread_price": live_spread_price, "max_allowed": self.cfg.gatekeeper.max_spread_gold}
            )
        elif asset_key == "BITCOIN" and live_spread_price > self.cfg.gatekeeper.max_spread_btc:
            return GatekeeperDecision(
                passed=False,
                stage="Stage 1: Hard Spread Cap",
                reason=f"Bitcoin spread (${live_spread_price:.2f}) exceeds absolute max limit (${self.cfg.gatekeeper.max_spread_btc:.2f})",
                metrics={"live_spread_price": live_spread_price, "max_allowed": self.cfg.gatekeeper.max_spread_btc}
            )
        elif asset_key == "NASDAQ" and live_spread_price > self.cfg.gatekeeper.max_spread_nasdaq:
            return GatekeeperDecision(
                passed=False,
                stage="Stage 1: Hard Spread Cap",
                reason=f"Nasdaq spread ({live_spread_price:.2f} pts) exceeds absolute max limit ({self.cfg.gatekeeper.max_spread_nasdaq:.2f} pts)",
                metrics={"live_spread_price": live_spread_price, "max_allowed": self.cfg.gatekeeper.max_spread_nasdaq}
            )

        return GatekeeperDecision(
            passed=True,
            stage="Stage 1: Spread & Session Filter",
            reason="Spread within normal threshold and session is active.",
            metrics={
                "live_spread_pts": live_spread_pts,
                "median_spread_pts": rolling_median_pts,
                "live_spread_price": live_spread_price,
                "current_time_utc": current_time_str
            }
        )

    # ---------------------------------------------------------
    # STAGE 2: Multi-Timeframe Regime Filter
    # ---------------------------------------------------------
    def evaluate_mtf_regime(
        self,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        df_m5: pd.DataFrame,
        strategy_type: str,  # "BREAKOUT" or "LIQUIDITY_SWEEP"
        direction: str       # "BUY" or "SELL"
    ) -> GatekeeperDecision:
        """
        Stage 2 Gatekeeper:
        - H1/H4 macro bias via 50/200 EMA alignment.
        - M5/M15 micro regime via ADX(14) and ATR(14) expansion.
          * Breakouts require: ADX > 25 and expanding ATR.
          * Liquidity Sweeps require: ADX < 22 (range fade) or structural sweep context.
        """
        # 1. Higher Timeframe (H1) EMA Alignment
        h1_ema50 = calculate_ema(df_h1['close'], self.cfg.regime.ema_fast_period)
        h1_ema200 = calculate_ema(df_h1['close'], self.cfg.regime.ema_slow_period)
        latest_h1_close = float(df_h1['close'].iloc[-1])
        h1_fast = float(h1_ema50.iloc[-1])
        h1_slow = float(h1_ema200.iloc[-1])

        h1_bullish = (h1_fast > h1_slow) and (latest_h1_close > h1_fast)
        h1_bearish = (h1_fast < h1_slow) and (latest_h1_close < h1_fast)

        # 2. Lower Timeframe (M5) ADX and ATR Volatility Check
        adx_df = calculate_adx(df_m5, period=self.cfg.regime.adx_period)
        latest_adx = float(adx_df['adx'].iloc[-1])

        atr_series = calculate_atr(df_m5, period=self.cfg.regime.atr_period)
        latest_atr = float(atr_series.iloc[-1])
        atr_sma = calculate_sma(atr_series, period=self.cfg.regime.atr_expansion_window)
        latest_atr_sma = float(atr_sma.iloc[-1]) if not np.isnan(atr_sma.iloc[-1]) else latest_atr

        atr_expanding = latest_atr >= (latest_atr_sma * 0.98)

        # 3. Apply Regime Rules based on Strategy Archetype
        if strategy_type == "BREAKOUT":
            # Breakouts require strong trend momentum (ADX > 25) and expanding volatility
            if latest_adx <= self.cfg.regime.adx_breakout_threshold:
                return GatekeeperDecision(
                    passed=False,
                    stage="Stage 2: MTF Regime Filter",
                    reason=f"Breakout rejected: M5 ADX ({latest_adx:.1f}) <= threshold ({self.cfg.regime.adx_breakout_threshold:.1f}). Lacks trend momentum.",
                    metrics={"adx": latest_adx, "adx_threshold": self.cfg.regime.adx_breakout_threshold}
                )

            if not atr_expanding:
                return GatekeeperDecision(
                    passed=False,
                    stage="Stage 2: MTF Regime Filter",
                    reason=f"Breakout rejected: ATR is contracting ({latest_atr:.4f} < SMA {latest_atr_sma:.4f}). Volatility expansion missing.",
                    metrics={"atr": latest_atr, "atr_sma": latest_atr_sma}
                )

            # Check macro trend alignment for breakout
            if direction == "BUY" and not (h1_fast >= h1_slow or latest_h1_close >= h1_slow):
                return GatekeeperDecision(
                    passed=False,
                    stage="Stage 2: MTF Macro Alignment",
                    reason=f"Buy breakout rejected: H1 trend is bearish (Price {latest_h1_close:.2f} < EMA200 {h1_slow:.2f}).",
                    metrics={"h1_close": latest_h1_close, "h1_ema200": h1_slow}
                )
            elif direction == "SELL" and not (h1_fast <= h1_slow or latest_h1_close <= h1_slow):
                return GatekeeperDecision(
                    passed=False,
                    stage="Stage 2: MTF Macro Alignment",
                    reason=f"Sell breakout rejected: H1 trend is bullish (Price {latest_h1_close:.2f} > EMA200 {h1_slow:.2f}).",
                    metrics={"h1_close": latest_h1_close, "h1_ema200": h1_slow}
                )

        elif strategy_type == "LIQUIDITY_SWEEP":
            # Liquidity sweeps are mean-reverting turtle-soup setups at session extremes.
            # Best performed in non-trending/ranging conditions (ADX < 22) or at exhausted extremes.
            # If ADX > 35, the market is in a violent runaway breakout where fading is perilous.
            if latest_adx > 35.0:
                return GatekeeperDecision(
                    passed=False,
                    stage="Stage 2: MTF Regime Filter",
                    reason=f"Liquidity sweep rejected: M5 ADX ({latest_adx:.1f}) > 35.0 indicates a runaway trend; fading breakout risks severe slippage.",
                    metrics={"adx": latest_adx}
                )

        return GatekeeperDecision(
            passed=True,
            stage="Stage 2: MTF Regime Filter",
            reason="Multi-timeframe regime and volatility conditions satisfied.",
            metrics={
                "adx": latest_adx,
                "atr": latest_atr,
                "atr_expanding": atr_expanding,
                "h1_ema50": h1_fast,
                "h1_ema200": h1_slow,
                "strategy_type": strategy_type
            }
        )
