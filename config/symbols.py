"""
config/symbols.py
-----------------
Symbol-specific quantitative limits, leverage constraints, and regime thresholds.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    base_asset: str
    quote_asset: str
    max_spread_bps: float        # Max allowed top-of-book bid-ask spread in basis points (1 bps = 0.01%)
    max_funding_rate_bps: float  # Absolute funding threshold in bps (e.g. 5.0 bps = 0.05%)
    leverage_cap: int            # Max leverage allowed under isolated margin
    is_isolated: bool            # Enforce isolated margin
    atr_stop_multiplier: float   # Dynamic ATR multiplier for Stop Loss
    primary_tf: str              # Execution timeframe ('5m' or '15m')
    macro_tf: str                # Trend filter timeframe ('1h' or '4h')
    default_tick_size: float     # Fallback tick size before exchangeInfo is loaded
    default_step_size: float     # Fallback step size before exchangeInfo is loaded
    default_min_notional: float  # Fallback minimum order notional value


SYMBOL_REGISTRY: Dict[str, SymbolConfig] = {
    "BTCUSDT": SymbolConfig(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        max_spread_bps=5.0,          # 0.05% max spread (e.g. $30 on $60,000 BTC)
        max_funding_rate_bps=5.0,    # 0.05% per 8h
        leverage_cap=5,
        is_isolated=True,
        atr_stop_multiplier=1.5,
        primary_tf="5m",
        macro_tf="1h",
        default_tick_size=0.10,
        default_step_size=0.001,
        default_min_notional=5.0
    ),
    "ETHUSDT": SymbolConfig(
        symbol="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT",
        max_spread_bps=8.0,
        max_funding_rate_bps=6.0,
        leverage_cap=5,
        is_isolated=True,
        atr_stop_multiplier=1.6,
        primary_tf="5m",
        macro_tf="1h",
        default_tick_size=0.01,
        default_step_size=0.01,
        default_min_notional=5.0
    ),
    "SOLUSDT": SymbolConfig(
        symbol="SOLUSDT",
        base_asset="SOL",
        quote_asset="USDT",
        max_spread_bps=12.0,
        max_funding_rate_bps=8.0,
        leverage_cap=3,
        is_isolated=True,
        atr_stop_multiplier=1.8,
        primary_tf="5m",
        macro_tf="1h",
        default_tick_size=0.01,
        default_step_size=0.01,
        default_min_notional=5.0
    ),
    "PAXGUSDT": SymbolConfig(
        symbol="PAXGUSDT",
        base_asset="PAXG",
        quote_asset="USDT",
        max_spread_bps=10.0,
        max_funding_rate_bps=5.0,
        leverage_cap=3,
        is_isolated=True,
        atr_stop_multiplier=1.5,
        primary_tf="5m",
        macro_tf="1h",
        default_tick_size=0.01,
        default_step_size=0.0001,
        default_min_notional=5.0
    )
}
