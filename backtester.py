"""
backtester.py
-------------
Historical backtesting engine for Zuperior MT5 algorithmic strategies.
Pulls historical rates directly from Zuperior's server via copy_rates_range / copy_rates_from_pos,
utilizes Zuperior's actual historical broker spread column, and computes institutional metrics:
Win Rate, Profit Factor, Max Drawdown, Expectancy ($ and R-multiple), and Sharpe Ratio.
"""

import sys
import argparse
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from config import cfg
from mt5_connector import MT5Connector
from strategies.base_strategy import TradeSignal
from strategies.gold_strategy import GoldLiquiditySweepStrategy
from strategies.btc_strategy import BtcVolatilitySqueezeStrategy
from strategies.nasdaq_strategy import NasdaqORBStrategy

logger = logging.getLogger("Backtester")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class SimulatedPosition:
    """Tracks the state of an open trade during backtesting."""

    def __init__(self, signal: TradeSignal, point: float, spread_price: float):
        self.signal = signal
        self.point = point
        self.direction = signal.direction
        self.entry_price = signal.entry_price + (spread_price if signal.direction == "BUY" else 0.0)
        self.stop_loss = signal.stop_loss
        self.tp1 = signal.take_profit_1
        self.tp2 = signal.take_profit_2
        self.lot_size = signal.recommended_lot_size
        self.remaining_lot = signal.recommended_lot_size
        self.dollar_risk = signal.dollar_risk
        self.entry_time = signal.bar_time
        self.exit_time: Optional[datetime] = None
        self.pnl_usd: float = 0.0
        self.r_multiple: float = 0.0
        self.tp1_hit: bool = False
        self.is_closed: bool = False
        self.exit_reason: str = ""


class Backtester:
    """
    Simulates multi-timeframe execution using Zuperior MT5 historical data and actual spreads.
    """

    def __init__(self, connector: Optional[MT5Connector] = None):
        self.connector = connector or MT5Connector()
        self.strategies = {
            "GOLD": GoldLiquiditySweepStrategy(),
            "BITCOIN": BtcVolatilitySqueezeStrategy(),
            "NASDAQ": NasdaqORBStrategy()
        }

    def run_backtest(
        self,
        asset_key: str,
        days: int = 30,
        initial_equity: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Executes historical backtest for a specific asset over the specified number of days.
        """
        if not self.connector.is_connected:
            if not self.connector.connect():
                raise RuntimeError("Failed to connect to MT5 terminal for backtesting.")

        symbol = self.connector.resolve_symbol(asset_key)
        if not symbol:
            raise ValueError(f"Could not resolve broker symbol for {asset_key}")

        specs = self.connector.get_symbol_specs(symbol)
        point = specs.get("point", 0.01)

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        logger.info(f"Fetching {days} days of historical rates for {symbol} ({start_dt.date()} to {end_dt.date()})...")

        # Fetch required timeframe data
        df_m5 = self.connector.fetch_rates_range(symbol, "M5", start_dt, end_dt)
        df_m15 = self.connector.fetch_rates_range(symbol, "M15", start_dt, end_dt)
        df_h1 = self.connector.fetch_rates_range(symbol, "H1", start_dt - timedelta(days=20), end_dt)
        df_h4 = self.connector.fetch_rates_range(symbol, "H4", start_dt - timedelta(days=40), end_dt)

        strategy = self.strategies.get(asset_key)
        if not strategy:
            raise ValueError(f"No strategy available for asset {asset_key}")

        logger.info(f"Loaded {len(df_m5)} M5 bars and {len(df_h1)} H1 bars. Commencing simulation...")

        equity = initial_equity
        equity_curve = [initial_equity]
        trades: List[SimulatedPosition] = []
        open_position: Optional[SimulatedPosition] = None

        min_lookback = 65
        step_increment = 1  # Bar by bar simulation

        for i in range(min_lookback, len(df_m5)):
            current_bar = df_m5.iloc[i]
            current_time = df_m5.index[i]
            bar_high = float(current_bar['high'])
            bar_low = float(current_bar['low'])
            bar_close = float(current_bar['close'])
            bar_open = float(current_bar['open'])
            spread_pts = int(current_bar['spread'])
            spread_price = spread_pts * point

            # 1. Manage existing open position
            if open_position is not None:
                self._update_position(open_position, bar_high, bar_low, bar_close, bar_open, current_time, specs)
                if open_position.is_closed:
                    equity += open_position.pnl_usd
                    equity_curve.append(equity)
                    trades.append(open_position)
                    open_position = None
                    continue

            # 2. Check for new signals if no position is active
            if open_position is None:
                m5_slice = df_m5.iloc[:i + 1]
                # Filter HTF bars matching current_time
                h1_slice = df_h1[df_h1.index <= current_time]
                h4_slice = df_h4[df_h4.index <= current_time]
                m15_slice = df_m15[df_m15.index <= current_time]

                if len(h1_slice) < 50:
                    continue

                data_frames = {
                    "M5": m5_slice,
                    "M15": m15_slice if len(m15_slice) > 20 else m5_slice,
                    "H1": h1_slice,
                    "H4": h4_slice if len(h4_slice) > 10 else h1_slice
                }

                live_tick = {
                    "symbol": symbol,
                    "bid": bar_close,
                    "ask": bar_close + spread_price,
                    "last": bar_close,
                    "spread_price": spread_price,
                    "spread_points": float(spread_pts),
                    "time": current_time
                }

                signal = strategy.evaluate(
                    symbol=symbol,
                    data_frames=data_frames,
                    live_tick=live_tick,
                    symbol_specs=specs,
                    account_equity=equity
                )

                if signal is not None:
                    open_position = SimulatedPosition(signal, point, spread_price)

        # Performance analytics
        report = self._calculate_metrics(trades, initial_equity, equity_curve)
        self._print_report(report, asset_key, symbol, days)
        return report

    def _update_position(
        self,
        pos: SimulatedPosition,
        high: float,
        low: float,
        close: float,
        open_p: float,
        current_time: datetime,
        specs: dict
    ):
        """Simulates price movement against Stop Loss and Take Profit levels."""
        tick_size = specs.get("trade_tick_size") or specs.get("point") or 0.01
        tick_value = specs.get("trade_tick_value", 1.0)
        risk_per_price_unit = (1.0 / tick_size) * tick_value if tick_size > 0 else specs.get("trade_contract_size", 1.0)

        if pos.direction == "BUY":
            # Check Stop Loss first (worst-case execution)
            if low <= pos.stop_loss:
                loss_dist = abs(pos.entry_price - pos.stop_loss)
                pnl = - (loss_dist * risk_per_price_unit * pos.remaining_lot)
                pos.pnl_usd += pnl
                pos.r_multiple = -1.0 if not pos.tp1_hit else 0.5
                pos.is_closed = True
                pos.exit_time = current_time
                pos.exit_reason = "Stop Loss Hit"
                return

            # Check Take Profit 1 (partial 50% scale-out, move SL to breakeven)
            if not pos.tp1_hit and high >= pos.tp1:
                closed_lot = pos.lot_size * 0.5
                gain_dist = abs(pos.tp1 - pos.entry_price)
                partial_pnl = gain_dist * risk_per_price_unit * closed_lot
                pos.pnl_usd += partial_pnl
                pos.remaining_lot -= closed_lot
                pos.tp1_hit = True
                pos.stop_loss = pos.entry_price  # Move to breakeven

            # Check Take Profit 2 (close runner)
            if pos.tp1_hit and high >= pos.tp2:
                gain_dist = abs(pos.tp2 - pos.entry_price)
                runner_pnl = gain_dist * risk_per_price_unit * pos.remaining_lot
                pos.pnl_usd += runner_pnl
                pos.r_multiple = pos.signal.risk_reward_ratio
                pos.is_closed = True
                pos.exit_time = current_time
                pos.exit_reason = "Take Profit 2 Hit"
                return

        elif pos.direction == "SELL":
            # Check Stop Loss
            if high >= pos.stop_loss:
                loss_dist = abs(pos.stop_loss - pos.entry_price)
                pnl = - (loss_dist * risk_per_price_unit * pos.remaining_lot)
                pos.pnl_usd += pnl
                pos.r_multiple = -1.0 if not pos.tp1_hit else 0.5
                pos.is_closed = True
                pos.exit_time = current_time
                pos.exit_reason = "Stop Loss Hit"
                return

            # Check Take Profit 1
            if not pos.tp1_hit and low <= pos.tp1:
                closed_lot = pos.lot_size * 0.5
                gain_dist = abs(pos.entry_price - pos.tp1)
                partial_pnl = gain_dist * risk_per_price_unit * closed_lot
                pos.pnl_usd += partial_pnl
                pos.remaining_lot -= closed_lot
                pos.tp1_hit = True
                pos.stop_loss = pos.entry_price  # Breakeven

            # Check Take Profit 2
            if pos.tp1_hit and low <= pos.tp2:
                gain_dist = abs(pos.entry_price - pos.tp2)
                runner_pnl = gain_dist * risk_per_price_unit * pos.remaining_lot
                pos.pnl_usd += runner_pnl
                pos.r_multiple = pos.signal.risk_reward_ratio
                pos.is_closed = True
                pos.exit_time = current_time
                pos.exit_reason = "Take Profit 2 Hit"
                return

    def _calculate_metrics(
        self,
        trades: List[SimulatedPosition],
        initial_equity: float,
        equity_curve: List[float]
    ) -> Dict[str, Any]:
        """Calculates institutional backtesting performance metrics."""
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_usd": 0.0,
                "max_drawdown_pct": 0.0,
                "expectancy_usd": 0.0,
                "expectancy_r": 0.0,
                "sharpe_ratio": 0.0,
                "net_profit": 0.0,
                "ending_equity": initial_equity
            }

        pnls = [t.pnl_usd for t in trades]
        r_mults = [t.r_multiple for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades) * 100.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        net_profit = sum(pnls)

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        # Max Drawdown Calculation
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
        expectancy_r = np.mean(r_mults) if r_mults else 0.0

        # Sharpe ratio
        daily_returns = pd.Series(pnls) / initial_equity
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0

        return {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_profit": round(net_profit, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_usd": round(max_dd_usd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "expectancy_usd": round(expectancy_usd, 2),
            "expectancy_r": round(expectancy_r, 2),
            "sharpe_ratio": round(sharpe, 2),
            "ending_equity": round(equity_curve[-1], 2)
        }

    def _print_report(self, m: Dict[str, Any], asset: str, symbol: str, days: int):
        """Prints formatted institutional backtest summary table."""
        bold = "\033[1m"
        reset = "\033[0m"
        green = "\033[92m"
        red = "\033[91m"
        cyan = "\033[96m"

        print(f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════╗{reset}")
        print(f"{bold}{cyan}║           ZUPERIOR MT5 STRATEGY BACKTEST RESULTS ({days} DAYS)           ║{reset}")
        print(f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════╝{reset}")
        print(f"Asset: {bold}{asset}{reset} | Symbol: {bold}{symbol}{reset}")
        print("──────────────────────────────────────────────────────────────────────")
        print(f"Total Trades:           {bold}{m['total_trades']}{reset}  (Wins: {green}{m.get('win_count', 0)}{reset} | Losses: {red}{m.get('loss_count', 0)}{reset})")
        print(f"Win Rate:               {bold}{green if m['win_rate'] >= 50 else red}{m['win_rate']}%{reset}")
        print(f"Profit Factor:          {bold}{green if m['profit_factor'] >= 1.5 else red}{m['profit_factor']}{reset}")
        print(f"Net Profit ($):         {bold}{green if m['net_profit'] >= 0 else red}${m['net_profit']:,.2f}{reset}")
        print(f"Ending Equity ($):      {bold}${m['ending_equity']:,.2f}{reset}")
        print(f"Max Drawdown:           {bold}{red}{m['max_drawdown_pct']}% (${m['max_drawdown_usd']:,.2f}){reset}")
        print(f"Expectancy:             {bold}${m['expectancy_usd']:.2f} / trade ({m['expectancy_r']:.2f} R){reset}")
        print(f"Sharpe Ratio:           {bold}{m['sharpe_ratio']}{reset}")
        print("──────────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zuperior MT5 Strategy Backtester")
    parser.add_argument("--asset", choices=["GOLD", "BITCOIN", "NASDAQ"], default="GOLD", help="Asset to backtest")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--equity", type=float, default=10000.0, help="Starting account equity")
    args = parser.parse_args()

    backtester = Backtester()
    backtester.run_backtest(asset_key=args.asset, days=args.days, initial_equity=args.equity)
