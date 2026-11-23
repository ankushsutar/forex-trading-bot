"""
risk_engine.py
--------------
Production-grade risk engine for Zuperior MetaTrader 5 accounts.
Calculates:
- Exact position sizing based on live account equity, risk %, tick value, and volume step.
- Dynamic ATR-based Stop Losses and structural swing invalidation levels.
- Minimum 1:2.5 Risk-to-Reward Take Profit 1 and Take Profit 2 levels.
- Confluence checklist validation gatekeeper.
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

from config import cfg

logger = logging.getLogger("RiskEngine")


@dataclass
class RiskProfile:
    """Calculated trade risk, lot sizing, and targets."""
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    recommended_lot_size: float
    dollar_risk: float
    account_equity: float
    is_valid: bool
    rejection_reason: str = ""


class RiskEngine:
    """
    Precision risk engine enforcing Zuperior MT5 contract sizing and risk management rules.
    """

    def __init__(self):
        self.cfg = cfg

    def compute_trade_risk(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
        symbol_specs: dict,
        target_tp1: Optional[float] = None,
        target_tp2: Optional[float] = None,
        direction: str = "BUY"
    ) -> RiskProfile:
        """
        Computes the complete risk profile, Take Profit targets, and normalized MT5 lot size.
        Enforces:
        - Minimum 1:2.5 Risk-to-Reward ratio.
        - Exact MT5 lot size clamped to volume_min, volume_max, and volume_step.
        """
        price_distance = abs(entry_price - stop_loss)
        if price_distance <= 0:
            return RiskProfile(
                entry_price=entry_price, stop_loss=stop_loss,
                take_profit_1=entry_price, take_profit_2=entry_price,
                risk_reward_ratio=0.0, recommended_lot_size=0.0,
                dollar_risk=0.0, account_equity=account_equity,
                is_valid=False, rejection_reason="Entry price equals Stop Loss price."
            )

        # 1. Compute Take Profit 1 and Take Profit 2 adhering to minimum 1:2.5 R:R
        min_rr = self.cfg.risk.min_risk_reward
        tp1_mult = max(self.cfg.risk.tp1_rr_ratio, min_rr)
        tp2_mult = max(self.cfg.risk.tp2_rr_ratio, tp1_mult + 1.0)

        if direction.upper() == "BUY":
            default_tp1 = entry_price + (tp1_mult * price_distance)
            default_tp2 = entry_price + (tp2_mult * price_distance)
            actual_tp1 = max(target_tp1, default_tp1) if target_tp1 else default_tp1
            actual_tp2 = max(target_tp2, default_tp2) if target_tp2 else default_tp2
        else:
            default_tp1 = entry_price - (tp1_mult * price_distance)
            default_tp2 = entry_price - (tp2_mult * price_distance)
            actual_tp1 = min(target_tp1, default_tp1) if target_tp1 else default_tp1
            actual_tp2 = min(target_tp2, default_tp2) if target_tp2 else default_tp2

        # 2. Risk-to-Reward Ratio Validation
        rr_tp1 = abs(actual_tp1 - entry_price) / price_distance
        if round(rr_tp1, 2) < min_rr:
            return RiskProfile(
                entry_price=entry_price, stop_loss=stop_loss,
                take_profit_1=actual_tp1, take_profit_2=actual_tp2,
                risk_reward_ratio=rr_tp1, recommended_lot_size=0.0,
                dollar_risk=0.0, account_equity=account_equity,
                is_valid=False,
                rejection_reason=f"Risk-to-Reward ratio (1:{rr_tp1:.2f}) is below minimum required 1:{min_rr:.1f}."
            )

        # 3. Exact MT5 Lot Sizing using Zuperior contract specifications
        # Risk amount in deposit currency (USD)
        risk_usd = account_equity * (self.cfg.risk.risk_percent / 100.0)
        risk_usd = min(risk_usd, self.cfg.risk.max_risk_usd)

        tick_size = symbol_specs.get("trade_tick_size") or symbol_specs.get("point") or 0.01
        tick_value = symbol_specs.get("trade_tick_value", 0.0)
        contract_size = symbol_specs.get("trade_contract_size", 100.0)

        # MT5 tick value formula:
        # Profit/Loss = (Ticks) * (Tick Value per Lot) * (Lots)
        # Therefore: Risk_USD = (Price_Distance / Tick_Size) * Tick_Value * Lots
        if tick_value > 0 and tick_size > 0:
            ticks = price_distance / tick_size
            risk_per_lot = ticks * tick_value
        else:
            # Fallback for standard USD quote
            risk_per_lot = price_distance * contract_size

        if risk_per_lot <= 0:
            return RiskProfile(
                entry_price=entry_price, stop_loss=stop_loss,
                take_profit_1=actual_tp1, take_profit_2=actual_tp2,
                risk_reward_ratio=rr_tp1, recommended_lot_size=0.0,
                dollar_risk=0.0, account_equity=account_equity,
                is_valid=False, rejection_reason="Calculated risk per lot is zero or negative."
            )

        raw_lots = risk_usd / risk_per_lot

        # 4. Normalize to Zuperior broker lot constraints
        min_lot = symbol_specs.get("volume_min", 0.01)
        max_lot = symbol_specs.get("volume_max", 100.0)
        step = symbol_specs.get("volume_step", 0.01)

        # Floor to nearest volume_step to prevent exceeding risk
        normalized_lots = math.floor(raw_lots / step) * step
        decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
        normalized_lots = round(normalized_lots, decimals)

        # Handle min lot clamping
        if normalized_lots < min_lot:
            # Check if minimum lot risks more than 1.5x intended risk
            min_lot_risk = min_lot * risk_per_lot
            if min_lot_risk > (risk_usd * 1.5):
                return RiskProfile(
                    entry_price=entry_price, stop_loss=stop_loss,
                    take_profit_1=actual_tp1, take_profit_2=actual_tp2,
                    risk_reward_ratio=rr_tp1, recommended_lot_size=min_lot,
                    dollar_risk=min_lot_risk, account_equity=account_equity,
                    is_valid=False,
                    rejection_reason=f"Account equity (${account_equity:.2f}) too small for minimum lot {min_lot} (Risk: ${min_lot_risk:.2f} vs Allowed: ${risk_usd:.2f})."
                )
            normalized_lots = min_lot

        normalized_lots = min(normalized_lots, max_lot)
        actual_dollar_risk = normalized_lots * risk_per_lot

        return RiskProfile(
            entry_price=round(entry_price, symbol_specs.get("digits", 2)),
            stop_loss=round(stop_loss, symbol_specs.get("digits", 2)),
            take_profit_1=round(actual_tp1, symbol_specs.get("digits", 2)),
            take_profit_2=round(actual_tp2, symbol_specs.get("digits", 2)),
            risk_reward_ratio=round(rr_tp1, 2),
            recommended_lot_size=normalized_lots,
            dollar_risk=round(actual_dollar_risk, 2),
            account_equity=account_equity,
            is_valid=True,
            rejection_reason=""
        )

    def validate_confluences(self, confluences: List[str]) -> Tuple[bool, str]:
        """
        Gatekeeper Stage 3: Requires at least 3 aligned technical confluences.
        """
        count = len(confluences)
        required = self.cfg.regime.min_confluences
        if count < required:
            return False, f"Insufficient confluences ({count}/{required} required). Confluences: {', '.join(confluences)}"
        return True, f"Confluence gatekeeper passed ({count} confluences satisfied)."
