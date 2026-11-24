"""
core/risk_manager.py
--------------------
Institutional quantitative risk engine for Binance Spot & USD-M Futures.
Enforces:
- Fixed-fractional equity risk sizing (0.5% - 1.0%) based on exact price distance to Stop Loss.
- Precision normalization to LOT_SIZE.stepSize and MIN_NOTIONAL verification.
- Minimum 1:2.5 Risk-to-Reward ratio validation.
- Intraday High-Water Mark Circuit Breaker (halts execution if daily drawdown >= 3.0%).
- Maximum concurrent open positions guardrail.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Dict, Optional, Tuple, Any

from config.settings import settings
from config.symbols import SYMBOL_REGISTRY
from core.exchange_metadata import ExchangeMetadata
from core.signal_engine import SignalEvent

logger = logging.getLogger("RiskManager")


@dataclass
class SizingResult:
    is_valid: bool
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    quantity: float
    dollar_risk: float
    notional_value: float
    risk_reward_ratio: float
    reduce_only: bool
    rejection_reason: str = ""


class RiskManager:
    """
    Manages portfolio risk, position sizing, order brackets, and circuit breakers.
    """

    def __init__(self, metadata: Optional[ExchangeMetadata] = None):
        self.metadata = metadata or ExchangeMetadata()
        self.risk_percent = settings.risk_percent
        self.max_risk_usd = settings.max_risk_usd
        self.min_rr_ratio = settings.min_rr_ratio
        self.max_daily_dd = settings.max_daily_drawdown_percent
        self.max_positions = settings.max_open_positions

        # Circuit Breaker state
        self._current_date: date = datetime.now(timezone.utc).date()
        self._daily_start_equity: float = 10000.0
        self._daily_peak_equity: float = 10000.0
        self._circuit_breaker_tripped: bool = False
        self._open_positions: Dict[str, dict] = {}

    def set_account_equity(self, equity: float):
        """Updates internal equity tracking and monitors intraday drawdown."""
        today = datetime.now(timezone.utc).date()

        # Reset daily baseline at 00:00 UTC
        if today != self._current_date:
            self._current_date = today
            self._daily_start_equity = equity
            self._daily_peak_equity = equity
            self._circuit_breaker_tripped = False
            logger.info(f"Daily risk baseline reset: ${equity:,.2f} on {today}")

        if equity > self._daily_peak_equity:
            self._daily_peak_equity = equity

        # Check Drawdown from intraday peak
        drawdown_pct = ((self._daily_peak_equity - equity) / self._daily_peak_equity) * 100.0
        if drawdown_pct >= self.max_daily_dd:
            if not self._circuit_breaker_tripped:
                logger.error(
                    f"PORTFOLIO KILL-SWITCH TRIPPED! Daily drawdown {drawdown_pct:.2f}% >= "
                    f"max limit {self.max_daily_dd:.2f}%. Halting new entries until 00:00 UTC."
                )
                self._circuit_breaker_tripped = True

    @property
    def is_trading_halted(self) -> bool:
        """Returns True if the daily drawdown circuit breaker is active."""
        return self._circuit_breaker_tripped

    def calculate_order_sizing(
        self,
        signal: SignalEvent,
        account_equity: float
    ) -> SizingResult:
        """
        Calculates exact crypto base quantity normalized to Binance stepSize and verified against minNotional.
        """
        self.set_account_equity(account_equity)

        # 1. Gatekeeper: Portfolio Circuit Breaker
        if self._circuit_breaker_tripped:
            return SizingResult(
                is_valid=False, symbol=signal.symbol, direction=signal.direction,
                entry_price=signal.entry_price, stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1, take_profit_2=signal.take_profit_2,
                quantity=0.0, dollar_risk=0.0, notional_value=0.0,
                risk_reward_ratio=signal.risk_reward_ratio, reduce_only=False,
                rejection_reason="Trading halted: Intraday high-water mark drawdown circuit breaker is active."
            )

        # 2. Gatekeeper: Max Concurrent Positions
        if len(self._open_positions) >= self.max_positions and signal.symbol not in self._open_positions:
            return SizingResult(
                is_valid=False, symbol=signal.symbol, direction=signal.direction,
                entry_price=signal.entry_price, stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1, take_profit_2=signal.take_profit_2,
                quantity=0.0, dollar_risk=0.0, notional_value=0.0,
                risk_reward_ratio=signal.risk_reward_ratio, reduce_only=False,
                rejection_reason=f"Max concurrent positions reached ({len(self._open_positions)}/{self.max_positions})."
            )

        # 3. Gatekeeper: Risk-to-Reward Ratio
        risk_dist = abs(signal.entry_price - signal.stop_loss)
        if risk_dist <= 0:
            return SizingResult(
                is_valid=False, symbol=signal.symbol, direction=signal.direction,
                entry_price=signal.entry_price, stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1, take_profit_2=signal.take_profit_2,
                quantity=0.0, dollar_risk=0.0, notional_value=0.0,
                risk_reward_ratio=0.0, reduce_only=False,
                rejection_reason="Invalid stop loss: distance to entry is zero."
            )

        reward_dist = abs(signal.take_profit_1 - signal.entry_price)
        actual_rr = reward_dist / risk_dist

        if round(actual_rr, 2) < self.min_rr_ratio:
            return SizingResult(
                is_valid=False, symbol=signal.symbol, direction=signal.direction,
                entry_price=signal.entry_price, stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1, take_profit_2=signal.take_profit_2,
                quantity=0.0, dollar_risk=0.0, notional_value=0.0,
                risk_reward_ratio=actual_rr, reduce_only=False,
                rejection_reason=f"Risk-to-Reward ratio (1:{actual_rr:.2f}) < minimum required (1:{self.min_rr_ratio:.1f})."
            )

        # 4. Position Sizing Math
        # Risk USD = Equity * Risk%
        intended_risk_usd = min(account_equity * (self.risk_percent / 100.0), self.max_risk_usd)

        # Crypto base quantity formula: Qty = Risk_USD / |Entry - StopLoss|
        raw_quantity = intended_risk_usd / risk_dist

        # 5. Rounding via ExchangeMetadata
        rounded_entry = self.metadata.round_price(signal.symbol, signal.entry_price)
        rounded_sl = self.metadata.round_price(signal.symbol, signal.stop_loss)
        rounded_tp1 = self.metadata.round_price(signal.symbol, signal.take_profit_1)
        rounded_tp2 = self.metadata.round_price(signal.symbol, signal.take_profit_2)

        rounded_qty = self.metadata.round_quantity(signal.symbol, raw_quantity)

        # 6. Validate against Binance LOT_SIZE and MIN_NOTIONAL filters
        is_valid_order, validation_msg = self.metadata.validate_order(signal.symbol, rounded_entry, rounded_qty)
        if not is_valid_order:
            return SizingResult(
                is_valid=False, symbol=signal.symbol, direction=signal.direction,
                entry_price=rounded_entry, stop_loss=rounded_sl,
                take_profit_1=rounded_tp1, take_profit_2=rounded_tp2,
                quantity=rounded_qty, dollar_risk=0.0, notional_value=0.0,
                risk_reward_ratio=actual_rr, reduce_only=False,
                rejection_reason=f"Binance filter violation: {validation_msg}"
            )

        actual_dollar_risk = rounded_qty * abs(rounded_entry - rounded_sl)
        notional_value = rounded_qty * rounded_entry

        return SizingResult(
            is_valid=True,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=rounded_entry,
            stop_loss=rounded_sl,
            take_profit_1=rounded_tp1,
            take_profit_2=rounded_tp2,
            quantity=rounded_qty,
            dollar_risk=round(actual_dollar_risk, 2),
            notional_value=round(notional_value, 2),
            risk_reward_ratio=round(actual_rr, 2),
            reduce_only=False
        )

    def register_position_opened(self, symbol: str, position_data: dict):
        self._open_positions[symbol] = position_data

    def register_position_closed(self, symbol: str):
        if symbol in self._open_positions:
            del self._open_positions[symbol]
