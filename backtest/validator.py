"""
backtest/validator.py
---------------------
Event-driven walk-forward backtest validator for Binance Spot & USD-M Futures.
Simulates:
- Binance Taker fees (0.05%) on market entry and exits.
- Dynamic half-spread crossing slippage.
- 8-hour Perpetual funding rate debits/credits.
- Bracket management: 50% partial exit at TP1 (1:2.5 R:R), moving stop loss to breakeven, and runner exit at TP2.
Computes institutional tear sheet: Profit Factor, Win Rate %, Max Drawdown %, Net Return after Fees/Funding, and Expectancy in R-multiples.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from config.symbols import SYMBOL_REGISTRY
from core.exchange_metadata import ExchangeMetadata
from core.regime_filter import RegimeFilter
from core.signal_engine import SignalEngine, SignalEvent
from core.risk_manager import RiskManager


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: float
    tp2: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    fees_paid: float
    funding_paid: float
    r_multiple: float
    exit_reason: str


@dataclass
class BacktestPerformanceReport:
    symbol: str
    timeframe: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    gross_profit: float
    gross_loss: float
    total_fees: float
    total_funding: float
    net_pnl: float
    ending_equity: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    expectancy_usd: float
    expectancy_r: float
    sharpe_ratio: float


class EventDrivenBacktester:
    """
    Event-driven walk-forward backtest simulator with Binance fee and funding mechanics.
    """

    def __init__(
        self,
        taker_fee: float = 0.0005,   # 0.05% taker fee
        maker_fee: float = 0.0002,   # 0.02% maker fee
        avg_funding_rate_8h: float = 0.0001 # 0.01% per 8h baseline
    ):
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.funding_rate = avg_funding_rate_8h

        self.metadata = ExchangeMetadata()
        self.regime_filter = RegimeFilter()
        self.signal_engine = SignalEngine(regime_filter=self.regime_filter)
        self.risk_manager = RiskManager(metadata=self.metadata)

    def run(
        self,
        symbol: str,
        df_ltf: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        initial_equity: float = 10000.0,
        spread_bps: float = 1.0
    ) -> BacktestPerformanceReport:
        """
        Walks forward candle-by-candle through df_ltf, strictly passing historical closed bars.
        """
        equity = initial_equity
        equity_curve = [initial_equity]
        trades: List[BacktestTrade] = []

        active_trade: Optional[Dict[str, Any]] = None
        min_bars = 60

        for i in range(min_bars, len(df_ltf)):
            # Closed slice up to candle i (zero future leak)
            current_closed_slice = df_ltf.iloc[:i + 1]
            current_bar = df_ltf.iloc[i]
            bar_time = df_ltf.index[i]
            high = float(current_bar['high'])
            low = float(current_bar['low'])
            close = float(current_bar['close'])
            open_p = float(current_bar['open'])

            # 1. Manage active position against current candle
            if active_trade is not None:
                self._update_position(active_trade, high, low, close, bar_time, spread_bps)
                if active_trade["is_closed"]:
                    t = self._finalize_trade(active_trade)
                    trades.append(t)
                    equity += t.net_pnl
                    equity_curve.append(equity)
                    active_trade = None
                    continue

            # 2. Check for new signals if no position is active
            if active_trade is None:
                htf_slice = df_htf[df_htf.index <= bar_time] if df_htf is not None else None

                signal = self.signal_engine.evaluate(
                    symbol=symbol,
                    df_ltf=current_closed_slice,
                    df_htf=htf_slice,
                    spread_bps=spread_bps,
                    funding_rate=self.funding_rate
                )

                if signal is not None:
                    sizing = self.risk_manager.calculate_order_sizing(signal, equity)
                    if sizing.is_valid and sizing.quantity > 0:
                        # Half-spread crossing slippage on market entry
                        slip = (spread_bps / 10000.0) * 0.5 * sizing.entry_price
                        exec_entry = sizing.entry_price + (slip if sizing.direction == "BUY" else -slip)

                        # Taker fee on entry notional
                        entry_fee = exec_entry * sizing.quantity * self.taker_fee

                        active_trade = {
                            "symbol": symbol,
                            "direction": sizing.direction,
                            "entry_time": bar_time,
                            "entry_price": exec_entry,
                            "stop_loss": sizing.stop_loss,
                            "tp1": sizing.take_profit_1,
                            "tp2": sizing.take_profit_2,
                            "total_quantity": sizing.quantity,
                            "remaining_quantity": sizing.quantity,
                            "dollar_risk": sizing.dollar_risk,
                            "fees_paid": entry_fee,
                            "funding_paid": 0.0,
                            "realized_pnl": 0.0,
                            "tp1_hit": False,
                            "is_closed": False,
                            "exit_reason": "",
                            "exit_price": 0.0,
                            "exit_time": None
                        }

        report = self._generate_report(symbol, df_ltf, trades, initial_equity, equity_curve)
        return report

    def _update_position(
        self,
        pos: Dict[str, Any],
        high: float,
        low: float,
        close: float,
        current_time: datetime,
        spread_bps: float
    ):
        """Simulates intra-candle bracket hit logic with funding payments."""
        # Accumulate 8h funding rate (every 96 M5 bars ~= 8 hours)
        funding_tick = pos["remaining_quantity"] * close * (self.funding_rate / 96.0)
        pos["funding_paid"] += funding_tick if pos["direction"] == "BUY" else -funding_tick

        is_buy = pos["direction"] == "BUY"

        if is_buy:
            # Check Stop Loss first (worst-case execution)
            if low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                loss_pnl = (exit_price - pos["entry_price"]) * pos["remaining_quantity"]
                exit_fee = exit_price * pos["remaining_quantity"] * self.taker_fee
                pos["fees_paid"] += exit_fee
                pos["realized_pnl"] += loss_pnl
                pos["is_closed"] = True
                pos["exit_price"] = exit_price
                pos["exit_time"] = current_time
                pos["exit_reason"] = "Stop Loss"
                return

            # Check Take Profit 1 (50% scale-out, move SL to breakeven)
            if not pos["tp1_hit"] and high >= pos["tp1"]:
                exit_price = pos["tp1"]
                closed_qty = pos["total_quantity"] * 0.5
                gain_pnl = (exit_price - pos["entry_price"]) * closed_qty
                exit_fee = exit_price * closed_qty * self.maker_fee
                pos["fees_paid"] += exit_fee
                pos["realized_pnl"] += gain_pnl
                pos["remaining_quantity"] -= closed_qty
                pos["tp1_hit"] = True
                pos["stop_loss"] = pos["entry_price"]  # Move SL to Breakeven

            # Check Take Profit 2 (close remaining runner)
            if pos["tp1_hit"] and high >= pos["tp2"]:
                exit_price = pos["tp2"]
                runner_qty = pos["remaining_quantity"]
                gain_pnl = (exit_price - pos["entry_price"]) * runner_qty
                exit_fee = exit_price * runner_qty * self.maker_fee
                pos["fees_paid"] += exit_fee
                pos["realized_pnl"] += gain_pnl
                pos["is_closed"] = True
                pos["exit_price"] = exit_price
                pos["exit_time"] = current_time
                pos["exit_reason"] = "Take Profit 2"
                return

        else:  # SELL / SHORT
            if high >= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                loss_pnl = (pos["entry_price"] - exit_price) * pos["remaining_quantity"]
                exit_fee = exit_price * pos["remaining_quantity"] * self.taker_fee
                pos["fees_paid"] += exit_fee
                pos["realized_pnl"] += loss_pnl
                pos["is_closed"] = True
                pos["exit_price"] = exit_price
                pos["exit_time"] = current_time
                pos["exit_reason"] = "Stop Loss"
                return

            if not pos["tp1_hit"] and low <= pos["tp1"]:
                exit_price = pos["tp1"]
                closed_qty = pos["total_quantity"] * 0.5
                gain_pnl = (pos["entry_price"] - exit_price) * closed_qty
                exit_fee = exit_price * closed_qty * self.maker_fee
                pos["fees_paid"] += exit_fee
                pos["realized_pnl"] += gain_pnl
                pos["remaining_quantity"] -= closed_qty
                pos["tp1_hit"] = True
                pos["stop_loss"] = pos["entry_price"]  # Move to Breakeven

            if pos["tp1_hit"] and low <= pos["tp2"]:
                exit_price = pos["tp2"]
                runner_qty = pos["remaining_quantity"]
                gain_pnl = (pos["entry_price"] - exit_price) * runner_qty
                exit_fee = exit_price * runner_qty * self.maker_fee
                pos["fees_paid"] += exit_fee
                pos["realized_pnl"] += gain_pnl
                pos["is_closed"] = True
                pos["exit_price"] = exit_price
                pos["exit_time"] = current_time
                pos["exit_reason"] = "Take Profit 2"
                return

    def _finalize_trade(self, pos: Dict[str, Any]) -> BacktestTrade:
        """Converts position state into finalized BacktestTrade record."""
        gross_pnl = pos["realized_pnl"]
        fees = pos["fees_paid"]
        funding = pos["funding_paid"]
        net_pnl = gross_pnl - fees - funding

        dollar_risk = pos["dollar_risk"]
        r_multiple = net_pnl / dollar_risk if dollar_risk > 0 else 0.0

        return BacktestTrade(
            symbol=pos["symbol"],
            direction=pos["direction"],
            entry_time=pos["entry_time"],
            exit_time=pos["exit_time"],
            entry_price=pos["entry_price"],
            exit_price=pos["exit_price"],
            stop_loss=pos["stop_loss"],
            tp1=pos["tp1"],
            tp2=pos["tp2"],
            quantity=pos["total_quantity"],
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            fees_paid=round(fees, 2),
            funding_paid=round(funding, 2),
            r_multiple=round(r_multiple, 2),
            exit_reason=pos["exit_reason"]
        )

    def _generate_report(
        self,
        symbol: str,
        df_ltf: pd.DataFrame,
        trades: List[BacktestTrade],
        initial_equity: float,
        equity_curve: List[float]
    ) -> BacktestPerformanceReport:
        """Computes comprehensive quantitative performance statistics."""
        total_trades = len(trades)
        if total_trades == 0:
            return BacktestPerformanceReport(
                symbol=symbol, timeframe="5m", total_trades=0,
                winning_trades=0, losing_trades=0, win_rate=0.0,
                profit_factor=0.0, gross_profit=0.0, gross_loss=0.0,
                total_fees=0.0, total_funding=0.0, net_pnl=0.0,
                ending_equity=initial_equity, max_drawdown_usd=0.0,
                max_drawdown_pct=0.0, expectancy_usd=0.0, expectancy_r=0.0,
                sharpe_ratio=0.0
            )

        net_pnls = [t.net_pnl for t in trades]
        r_mults = [t.r_multiple for t in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p < 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades) * 100.0

        gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
        gross_loss = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
        total_fees = sum(t.fees_paid for t in trades)
        total_funding = sum(t.funding_paid for t in trades)
        net_pnl = sum(net_pnls)

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        # Drawdown
        peak = equity_curve[0]
        max_dd_usd = 0.0
        max_dd_pct = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd_usd = peak - eq
            dd_pct = (dd_usd / peak) * 100.0 if peak > 0 else 0.0
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        # Expectancy
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        win_prob = win_count / total_trades
        loss_prob = loss_count / total_trades
        expectancy_usd = (win_prob * avg_win) - (loss_prob * avg_loss)
        expectancy_r = float(np.mean(r_mults)) if r_mults else 0.0

        # Sharpe ratio
        daily_returns = pd.Series(net_pnls) / initial_equity
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0

        return BacktestPerformanceReport(
            symbol=symbol,
            timeframe="5m",
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            total_fees=round(total_fees, 2),
            total_funding=round(total_funding, 2),
            net_pnl=round(net_pnl, 2),
            ending_equity=round(equity_curve[-1], 2),
            max_drawdown_usd=round(max_dd_usd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            expectancy_usd=round(expectancy_usd, 2),
            expectancy_r=round(expectancy_r, 2),
            sharpe_ratio=round(sharpe, 2)
        )
