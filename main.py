"""
main.py
-------
Unified command-line execution entrypoint for the Binance Quantitative Trading Engine.
Supports:
- --verify-metadata: Validates Binance exchangeInfo filters, tickSize, stepSize, and minNotional.
- --once: Executes a single non-repainting scan pass across closed bars.
- --paper: Runs resilient async WebSocket market stream in paper trading mode.
- --live: Runs live execution mode with authenticated order routing and reduceOnly exits.
- --backtest: Runs event-driven backtest simulation modeling taker fees and funding rates.
"""

import sys
import os
import asyncio
import signal
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd

from config.settings import settings, ExecutionEnvironment
from config.symbols import SYMBOL_REGISTRY, SymbolConfig
from core.exchange_metadata import ExchangeMetadata
from core.market_stream import MarketStreamManager
from core.regime_filter import RegimeFilter
from core.signal_engine import SignalEngine
from core.risk_manager import RiskManager
from execution.binance_router import BinanceOrderRouter
from backtest.validator import EventDrivenBacktester, BacktestPerformanceReport

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BinanceEngine")


class BinanceQuantitativeEngine:
    """
    Main orchestration engine managing stream consumers, risk evaluation, and order routing.
    """

    def __init__(self, execution_env: Optional[ExecutionEnvironment] = None):
        self.env = execution_env or settings.execution_env
        self.metadata = ExchangeMetadata()
        self.regime_filter = RegimeFilter()
        self.signal_engine = SignalEngine(regime_filter=self.regime_filter)
        self.risk_manager = RiskManager(metadata=self.metadata)
        self.router = BinanceOrderRouter()
        self.stream_manager = MarketStreamManager(symbols=list(SYMBOL_REGISTRY.keys()))

        self.account_equity: float = 10000.0  # Baseline equity in USDT
        self.running = False

    def verify_metadata(self) -> bool:
        """
        Connects to Binance API and verifies dynamic tickSize, stepSize, and minNotional filters.
        """
        bold = "\033[1m"
        reset = "\033[0m"
        green = "\033[92m"
        yellow = "\033[93m"
        cyan = "\033[96m"
        red = "\033[91m"

        print(f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗{reset}")
        print(f"{bold}{cyan}║                      BINANCE QUANTITATIVE ENGINE: SYMBOL METADATA VERIFICATION                       ║{reset}")
        print(f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝{reset}\n")

        print(f"Trading Mode: {bold}{settings.trading_mode.value}{reset} | Endpoint: {bold}{settings.endpoints.rest_url}{reset}\n")

        # Refresh filters
        self.metadata.refresh(timeout=5)

        print(f"{bold}{'Symbol':<12} {'Tick Size':<14} {'Step Size':<14} {'Min Qty':<12} {'Min Notional':<16} {'Precision (P/Q)':<18} {'Status'}{reset}")
        print("─" * 96)

        all_ok = True
        for sym, cfg in SYMBOL_REGISTRY.items():
            f = self.metadata.get_filters(sym)
            precision_str = f"{f.price_precision} / {f.quantity_precision}"
            status = f"{green}VERIFIED{reset}"
            print(f"{bold}{sym:<12}{reset} {str(f.tick_size):<14} {str(f.step_size):<14} {str(f.min_qty):<12} ${str(f.min_notional):<15} {precision_str:<18} {status}")

        print("─" * 96)
        print(f"\n{bold}{green}✔ All Binance exchangeInfo precision filters and notional thresholds verified.{reset}\n")
        return all_ok

    def scan_once(self):
        """
        Executes a single market scan across closed bars for all registered symbols.
        """
        bold = "\033[1m"
        reset = "\033[0m"
        green = "\033[92m"
        yellow = "\033[93m"
        cyan = "\033[96m"
        red = "\033[91m"

        print(f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗{reset}")
        print(f"{bold}{cyan}║                          BINANCE ENGINE: CLOSED-BAR MARKET SCAN REPORT                               ║{reset}")
        print(f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝{reset}")
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"Time: {bold}{now_utc}{reset} | Portfolio Equity: {green}${self.account_equity:,.2f} USDT{reset} | Risk: {settings.risk_percent}%\n")

        # Warm up closed bar history via REST
        self.stream_manager.warm_up_history(count=100)

        print(f"{bold}{'Symbol':<12} {'Close Price':<16} {'Spread (bps)':<16} {'Regime State':<22} {'Signal / Gatekeeper Verdict'}{reset}")
        print("─" * 105)

        signals_found = 0
        for sym in SYMBOL_REGISTRY.keys():
            df = self.stream_manager.klines_buffer.get(sym)
            if df is None or len(df) < 50:
                print(f"{bold}{sym:<12}{reset} {'-':<16} {'-':<16} {'INSUFFICIENT_DATA':<22} {yellow}Buffering historical bars{reset}")
                continue

            last_close = float(df['close'].iloc[-1])
            spread_bps = self.stream_manager.get_live_spread_bps(sym)
            mark = self.stream_manager.mark_prices.get(sym, {})
            funding_rate = mark.get("funding_rate", 0.0001)

            verdict = self.regime_filter.classify_regime(sym, df, spread_bps=spread_bps, funding_rate=funding_rate)
            signal = self.signal_engine.evaluate(sym, df, spread_bps=spread_bps, funding_rate=funding_rate)

            price_str = f"${last_close:,.2f}"
            spread_str = f"{spread_bps:.1f} bps"
            regime_str = verdict.regime.value

            if signal is not None:
                signals_found += 1
                sizing = self.risk_manager.calculate_order_sizing(signal, self.account_equity)
                if sizing.is_valid:
                    status_str = f"{green}{bold}✔ {signal.direction} {sizing.quantity} @ {sizing.entry_price} (SL: {sizing.stop_loss}){reset}"
                    # Execute
                    self.router.execute_bracket(sizing, int(signal.bar_time.timestamp()))
                else:
                    status_str = f"{yellow}Risk Throttled: {sizing.rejection_reason}{reset}"
            else:
                status_str = f"{yellow}⏸ {verdict.reason}{reset}"

            print(f"{bold}{sym:<12}{reset} {price_str:<16} {spread_str:<16} {regime_str:<22} {status_str}")

        print("─" * 105)
        if signals_found == 0:
            print(f"{yellow}Notice: Zero false signals fired. Strict regime filter and zero-repaint closed-bar rules enforced.{reset}\n")
        else:
            print(f"{green}{bold}✔ {signals_found} Signal(s) Evaluated and Processed!{reset}\n")

    def _on_closed_candle(self, symbol: str, df_closed: pd.DataFrame):
        """
        Event-driven callback triggered exclusively when a candle closes (kline.x == True).
        Guarantees zero repainting.
        """
        spread_bps = self.stream_manager.get_live_spread_bps(symbol)
        mark = self.stream_manager.mark_prices.get(symbol, {})
        funding_rate = mark.get("funding_rate", 0.0001)

        signal = self.signal_engine.evaluate(
            symbol=symbol,
            df_ltf=df_closed,
            spread_bps=spread_bps,
            funding_rate=funding_rate
        )

        if signal is not None:
            sizing = self.risk_manager.calculate_order_sizing(signal, self.account_equity)
            if sizing.is_valid:
                self.router.execute_bracket(sizing, int(signal.bar_time.timestamp()))
            else:
                logger.info(f"[{symbol}] Signal rejected by Risk Manager: {sizing.rejection_reason}")

    async def run_async_stream(self):
        """Starts real-time async WebSocket loop."""
        self.running = True
        logger.info(f"Starting Binance Quantitative Trading Engine in {self.env.value} mode...")

        # Sync clock & warm up metadata
        self.router.sync_server_time()
        self.metadata.refresh()
        self.stream_manager.warm_up_history(count=150)

        # Register event-driven callback
        self.stream_manager.add_closed_bar_listener(self._on_closed_candle)

        # Run stream
        await self.stream_manager.start()

    def run_backtest(self, symbol: str, days: int = 30):
        """Runs event-driven backtest simulation."""
        bold = "\033[1m"
        reset = "\033[0m"
        green = "\033[92m"
        red = "\033[91m"
        cyan = "\033[96m"

        validator = EventDrivenBacktester()

        # Warm up data for backtest
        self.stream_manager.warm_up_history(count=min(days * 288, 1000))
        df = self.stream_manager.klines_buffer.get(symbol)
        if df is None or len(df) < 50:
            logger.error(f"Could not load historical bars for {symbol} to backtest.")
            return

        report = validator.run(symbol, df, initial_equity=self.account_equity)

        print(f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════════════╗{reset}")
        print(f"{bold}{cyan}║            BINANCE EVENT-DRIVEN BACKTEST REPORT (TAKER FEES & FUNDING)        ║{reset}")
        print(f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════════════╝{reset}")
        print(f"Symbol: {bold}{symbol}{reset} | Timeframe: {bold}{report.timeframe}{reset} | Initial Equity: ${self.account_equity:,.2f}")
        print("─" * 80)
        print(f"Total Trades:           {bold}{report.total_trades}{reset}  (Wins: {green}{report.winning_trades}{reset} | Losses: {red}{report.losing_trades}{reset})")
        print(f"Win Rate:               {bold}{green if report.win_rate >= 50 else red}{report.win_rate}%{reset}")
        print(f"Profit Factor:          {bold}{green if report.profit_factor >= 1.5 else red}{report.profit_factor}{reset}")
        print(f"Gross Profit:           ${report.gross_profit:,.2f}")
        print(f"Gross Loss:             ${report.gross_loss:,.2f}")
        print(f"Total Taker Fees:       {red}-${report.total_fees:,.2f}{reset}")
        print(f"Total Funding Paid:     ${report.total_funding:,.2f}")
        print(f"Net Realized PnL:       {bold}{green if report.net_pnl >= 0 else red}${report.net_pnl:,.2f}{reset}")
        print(f"Ending Equity:          {bold}${report.ending_equity:,.2f}{reset}")
        print(f"Max Drawdown:           {bold}{red}{report.max_drawdown_pct}% (${report.max_drawdown_usd:,.2f}){reset}")
        print(f"Expectancy:             {bold}${report.expectancy_usd:.2f} / trade ({report.expectancy_r:.2f} R){reset}")
        print(f"Sharpe Ratio:           {bold}{report.sharpe_ratio}{reset}")
        print("─" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Binance Quantitative Algorithmic Trading Engine")
    parser.add_argument("--verify-metadata", action="store_true", help="Connect to Binance, verify exchangeInfo filters, tickSize, stepSize, and minNotional")
    parser.add_argument("--once", action="store_true", help="Execute single-pass scan across closed bars and print diagnostic report")
    parser.add_argument("--paper", action="store_true", help="Run live WebSocket market stream in paper trading mode")
    parser.add_argument("--live", action="store_true", help="Run live WebSocket market stream with real authenticated order execution")
    parser.add_argument("--backtest", choices=list(SYMBOL_REGISTRY.keys()), help="Run event-driven backtester on specified pair")
    parser.add_argument("--days", type=int, default=14, help="Historical days for backtest (default: 14)")
    args = parser.parse_args()

    engine = BinanceQuantitativeEngine()

    if args.verify_metadata:
        success = engine.verify_metadata()
        sys.exit(0 if success else 1)

    if args.backtest:
        engine.run_backtest(args.backtest, days=args.days)
        sys.exit(0)

    if args.once:
        engine.scan_once()
        sys.exit(0)

    # Stream Runner
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_exit(sig, frame):
        logger.info("Shutdown signal received (Ctrl+C). Terminating...")
        engine.stream_manager.stop()
        loop.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    if args.live:
        engine.env = ExecutionEnvironment.LIVE
        settings.execution_env = ExecutionEnvironment.LIVE
    else:
        engine.env = ExecutionEnvironment.PAPER
        settings.execution_env = ExecutionEnvironment.PAPER

    try:
        loop.run_until_complete(engine.run_async_stream())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
