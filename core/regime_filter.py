"""
core/regime_filter.py
---------------------
Quantitative market regime classifier and pre-trade gatekeeper.
Combines:
- ADX(14) trend strength
- ATR(14) volatility percentile (relative to rolling 100-bar distribution)
- Dual EMA (50 / 200) trend and slope alignment
- Top-of-book bid-ask spread in basis points (guarding against illiquidity)
- 8-hour Perpetual Funding Rate skew (guarding against crowded carry drag and liquidation cascades)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from config.symbols import SymbolConfig, SYMBOL_REGISTRY
from indicators import calculate_ema, calculate_atr, calculate_adx


class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    MEAN_REVERTING = "MEAN_REVERTING"
    CHOP_NO_TRADE = "CHOP_NO_TRADE"


@dataclass
class RegimeVerdict:
    regime: MarketRegime
    is_tradeable: bool
    allowed_directions: list  # ["BUY"], ["SELL"], ["BUY", "SELL"], or []
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class RegimeFilter:
    """
    Classifies market regime and validates whether current market conditions
    permit high-expectancy trading.
    """

    def __init__(self, adx_trending_threshold: float = 25.0, adx_ranging_threshold: float = 20.0):
        self.adx_trending = adx_trending_threshold
        self.adx_ranging = adx_ranging_threshold

    def classify_regime(
        self,
        symbol: str,
        df_ltf: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        spread_bps: float = 0.0,
        funding_rate: float = 0.0
    ) -> RegimeVerdict:
        """
        Evaluates multi-timeframe bar data, spread, and funding rate to produce a comprehensive regime verdict.
        """
        cfg = SYMBOL_REGISTRY.get(symbol)
        max_spread_bps = cfg.max_spread_bps if cfg else 8.0
        max_funding_bps = cfg.max_funding_rate_bps if cfg else 5.0
        funding_bps = funding_rate * 10000.0

        metrics = {
            "symbol": symbol,
            "spread_bps": round(spread_bps, 2),
            "funding_bps": round(funding_bps, 2)
        }

        # 1. Gatekeeper: Spread Guard (illiquid/news spread blowup)
        if spread_bps > max_spread_bps:
            return RegimeVerdict(
                regime=MarketRegime.CHOP_NO_TRADE,
                is_tradeable=False,
                allowed_directions=[],
                reason=f"Spread ({spread_bps:.1f} bps) exceeds max limit ({max_spread_bps:.1f} bps). Order-book illiquid.",
                metrics=metrics
            )

        # 2. Gatekeeper: Extreme Funding Rate Skew Guard
        # If funding rate is heavily skewed positive, longs pay excessive funding and face liquidation risk
        longs_permitted = True
        shorts_permitted = True

        if funding_bps > max_funding_bps:
            longs_permitted = False
            metrics["funding_warning"] = "High positive funding: Longs throttled due to crowded carry drag."
        elif funding_bps < -max_funding_bps:
            shorts_permitted = False
            metrics["funding_warning"] = "Extreme negative funding: Shorts throttled due to short-squeeze risk."

        if not longs_permitted and not shorts_permitted:
            return RegimeVerdict(
                regime=MarketRegime.CHOP_NO_TRADE,
                is_tradeable=False,
                allowed_directions=[],
                reason=f"Extreme funding skew ({funding_bps:.1f} bps). Market heavily crowded.",
                metrics=metrics
            )

        if len(df_ltf) < 50:
            return RegimeVerdict(
                regime=MarketRegime.CHOP_NO_TRADE,
                is_tradeable=False,
                allowed_directions=[],
                reason="Insufficient historical bars for statistical regime classification.",
                metrics=metrics
            )

        # 3. Micro Indicators (LTF)
        adx_df = calculate_adx(df_ltf, period=14)
        latest_adx = float(adx_df['adx'].iloc[-1])
        plus_di = float(adx_df['plus_di'].iloc[-1])
        minus_di = float(adx_df['minus_di'].iloc[-1])
        metrics["adx"] = round(latest_adx, 2)
        metrics["plus_di"] = round(plus_di, 2)
        metrics["minus_di"] = round(minus_di, 2)

        atr_series = calculate_atr(df_ltf, period=14)
        latest_atr = float(atr_series.iloc[-1])
        # Calculate ATR percentile over last 100 bars
        atr_window = atr_series.tail(100)
        atr_percentile = float((atr_window < latest_atr).mean() * 100.0)
        metrics["atr"] = round(latest_atr, 4)
        metrics["atr_percentile"] = round(atr_percentile, 1)

        # 4. Macro Structure (HTF if provided, else LTF 50/200)
        target_df = df_htf if (df_htf is not None and len(df_htf) >= 50) else df_ltf
        ema50 = calculate_ema(target_df['close'], 50)
        ema200 = calculate_ema(target_df['close'], 200)
        latest_close = float(target_df['close'].iloc[-1])
        curr_ema50 = float(ema50.iloc[-1])
        curr_ema200 = float(ema200.iloc[-1])

        metrics["ema50"] = round(curr_ema50, 2)
        metrics["ema200"] = round(curr_ema200, 2)

        is_macro_bull = (curr_ema50 > curr_ema200) and (latest_close >= curr_ema50)
        is_macro_bear = (curr_ema50 < curr_ema200) and (latest_close <= curr_ema50)

        # 5. Regime Synthesis
        # State A: Strong Trend
        if latest_adx >= self.adx_trending and atr_percentile >= 40.0:
            if is_macro_bull and plus_di > minus_di:
                dirs = ["BUY"] if longs_permitted else []
                reason = "Bullish trend confirmed: ADX > 25, expanding ATR, EMA50 > EMA200."
                return RegimeVerdict(
                    regime=MarketRegime.TRENDING_BULL,
                    is_tradeable=len(dirs) > 0,
                    allowed_directions=dirs,
                    reason=reason,
                    metrics=metrics
                )
            elif is_macro_bear and minus_di > plus_di:
                dirs = ["SELL"] if shorts_permitted else []
                reason = "Bearish trend confirmed: ADX > 25, expanding ATR, EMA50 < EMA200."
                return RegimeVerdict(
                    regime=MarketRegime.TRENDING_BEAR,
                    is_tradeable=len(dirs) > 0,
                    allowed_directions=dirs,
                    reason=reason,
                    metrics=metrics
                )

        # State B: Mean-Reverting / Ranging
        if latest_adx < self.adx_ranging and atr_percentile <= 60.0:
            dirs = []
            if longs_permitted:
                dirs.append("BUY")
            if shorts_permitted:
                dirs.append("SELL")
            return RegimeVerdict(
                regime=MarketRegime.MEAN_REVERTING,
                is_tradeable=len(dirs) > 0,
                allowed_directions=dirs,
                reason="Mean-reverting regime: Low ADX (< 20) with compressed volatility. Suitable for liquidity sweep reversals.",
                metrics=metrics
            )

        # State C: Indeterminate Chop / Low-Expectancy Zone
        return RegimeVerdict(
            regime=MarketRegime.CHOP_NO_TRADE,
            is_tradeable=False,
            allowed_directions=[],
            reason=f"Chop / Indeterminate state: ADX {latest_adx:.1f} lacks strong trend or clear range bounds.",
            metrics=metrics
        )
