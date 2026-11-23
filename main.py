"""
main.py
-------
Main execution entrypoint for the Zuperior MetaTrader 5 Algorithmic Signal Engine.
Coordinates:
- Dynamic symbol resolution and spec verification (--verify-symbols)
- Live multi-timeframe bar fetching and live tick streams
- 3-stage gatekeeper validation (Spread/Session, MTF Regime, Confluence/Lot Sizing)
- Real-time alert dispatch to Console, Telegram, and Webhooks
"""

import sys
import time
import signal
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd

from config import cfg
from mt5_connector import MT5Connector
from notifier import TradeNotifier
from strategies.base_strategy import BaseStrategy, TradeSignal
from strategies.gold_strategy import GoldLiquiditySweepStrategy
from strategies.btc_strategy import BtcVolatilitySqueezeStrategy
from strategies.nasdaq_strategy import NasdaqORBStrategy
from backtester import Backtester

logger = logging.getLogger("SignalEngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class ZuperiorSignalEngine:
    """
    Master orchestrator for real-time market data scanning and signal generation.
    """

    def __init__(self, connector: Optional[MT5Connector] = None):
        self.connector = connector or MT5Connector()
        self.notifier = TradeNotifier()
        self.strategies: Dict[str, BaseStrategy] = {
            "GOLD": GoldLiquiditySweepStrategy(),
            "BITCOIN": BtcVolatilitySqueezeStrategy(),
            "NASDAQ": NasdaqORBStrategy()
        }
        self.running = False

    def verify_symbols(self) -> bool:
        """
        Connects to Zuperior MT5, queries account details, detects exact broker symbols,
        validates contract specifications (contract size, tick value, spread, lot steps),
        and tests live OHLCV data flow.
        """
        bold = "\033[1m"
        reset = "\033[0m"
        green = "\033[92m"
        yellow = "\033[93m"
        cyan = "\033[96m"
        red = "\033[91m"

        print(f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════════════╗{reset}")
        print(f"{bold}{cyan}║           ZUPERIOR METATRADER 5: SYMBOL & DATA FLOW VERIFICATION             ║{reset}")
        print(f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════════════╝{reset}\n")

        # 1. Connect to Terminal
        if not self.connector.connect():
            print(f"{red}[ERROR] Failed to connect to MT5 terminal.{reset}")
            return False

        # 2. Account Information
        try:
            acc = self.connector.get_account_info()
            print(f"{bold}Zuperior Account Profile:{reset}")
            print(f"  • Login:      {bold}{acc['login']}{reset}")
            print(f"  • Server:     {bold}{acc['server']}{reset} ({acc['company']})")
            print(f"  • Currency:   {bold}{acc['currency']}{reset} | Leverage: 1:{acc['leverage']}")
            print(f"  • Balance:    {green}${acc['balance']:,.2f}{reset}")
            print(f"  • Live Equity:{green}${acc['equity']:,.2f}{reset}\n")
        except Exception as e:
            print(f"{red}[ERROR] Could not fetch account info: {e}{reset}")
            return False

        # 3. Dynamic Symbol Verification
        print(f"{bold}{'Asset':<10} {'Broker Symbol':<14} {'Bid/Ask':<20} {'Spread (Pts/$)':<18} {'Contract':<10} {'Tick Val':<10} {'Min Step':<10} {'Data Flow':<10}{reset}")
        print("─" * 105)

        all_ok = True
        for asset in ["GOLD", "BITCOIN", "NASDAQ"]:
            symbol = self.connector.resolve_symbol(asset)
            if not symbol:
                print(f"{red}{asset:<10} {'NOT FOUND':<14} {'-':<20} {'-':<18} {'-':<10} {'-':<10} {'-':<10} {'FAIL':<10}{reset}")
                all_ok = False
                continue

            try:
                specs = self.connector.get_symbol_specs(symbol)
                tick = self.connector.get_live_tick(symbol)

                # Test data flow
                df_test = self.connector.fetch_rates(symbol, "M5", n_bars=100)
                data_status = f"{green}OK ({len(df_test)}b){reset}" if len(df_test) >= 100 else f"{yellow}LOW{reset}"

                bid_ask = f"{tick['bid']:.2f} / {tick['ask']:.2f}"
                spread_str = f"{tick['spread_points']:.0f} pts (${tick['spread_price']:.2f})"
                contract_str = f"{specs['trade_contract_size']:g}"
                tick_val_str = f"${specs['trade_tick_value']:.2f}"
                lot_step_str = f"{specs['volume_step']:g} (min {specs['volume_min']:g})"

                print(f"{bold}{asset:<10}{reset} {green}{symbol:<14}{reset} {bid_ask:<20} {spread_str:<18} {contract_str:<10} {tick_val_str:<10} {lot_step_str:<10} {data_status}")

            except Exception as e:
                print(f"{red}{asset:<10} {symbol:<14} ERROR: {e}{reset}")
                all_ok = False

        print("─" * 105)
        if all_ok:
            print(f"\n{bold}{green}✔ ALL ASSET SYMBOLS, CONTRACT SPECIFICATIONS & DATA FLOW CONFIRMED.{reset}\n")
        else:
            print(f"\n{bold}{yellow}⚠ WARNING: Some symbols or data feeds could not be verified.{reset}\n")

        return all_ok

    def scan_once(self, verbose: bool = True):
        """
        Executes one full scanning pass across all configured instruments.
        Displays diagnostic summary of market conditions and gatekeeper states.
        """
        bold = "\033[1m"
        reset = "\033[0m"
        green = "\033[92m"
        yellow = "\033[93m"
        cyan = "\033[96m"
        red = "\033[91m"

        try:
            account_info = self.connector.get_account_info()
            equity = account_info["equity"]
        except Exception as e:
            logger.error(f"Failed to fetch live account equity: {e}")
            return

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        scan_records = []
        signals_emitted = 0

        for asset, strategy in self.strategies.items():
            symbol = self.connector.resolve_symbol(asset)
            if not symbol:
                scan_records.append((asset, "NOT FOUND", "-", "-", f"{red}Symbol not detected{reset}"))
                continue

            try:
                specs = self.connector.get_symbol_specs(symbol)
                live_tick = self.connector.get_live_tick(symbol)

                # Pull timeframe rates
                df_m5 = self.connector.fetch_rates(symbol, "M5", n_bars=250)
                df_m15 = self.connector.fetch_rates(symbol, "M15", n_bars=250)
                df_h1 = self.connector.fetch_rates(symbol, "H1", n_bars=250)
                df_h4 = self.connector.fetch_rates(symbol, "H4", n_bars=250)

                data_frames = {
                    "M5": df_m5,
                    "M15": df_m15,
                    "H1": df_h1,
                    "H4": df_h4
                }

                # Strategy evaluation (includes 3-stage gatekeeper pipeline)
                signal = strategy.evaluate(
                    symbol=symbol,
                    data_frames=data_frames,
                    live_tick=live_tick,
                    symbol_specs=specs,
                    account_equity=equity
                )

                bid_ask = f"{live_tick['bid']:.2f} / {live_tick['ask']:.2f}"
                spread_str = f"{live_tick['spread_points']:.0f} pts (${live_tick['spread_price']:.2f})"

                if signal is not None:
                    signals_emitted += 1
                    status_str = f"{green}{bold}✔ SIGNAL: {signal.direction} @ {signal.entry_price} (SL: {signal.stop_loss}){reset}"
                    self.notifier.notify(signal)
                else:
                    status_str = f"{yellow}⏸ {strategy.last_status}{reset}"

                scan_records.append((asset, symbol, bid_ask, spread_str, status_str))

            except Exception as e:
                logger.error(f"Error evaluating {asset} ({symbol}): {e}", exc_info=False)
                scan_records.append((asset, symbol, "-", "-", f"{red}Error: {e}{reset}"))

        if verbose:
            print(f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗{reset}")
            print(f"{bold}{cyan}║                               ZUPERIOR MT5: MARKET SCAN REPORT                                       ║{reset}")
            print(f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝{reset}")
            print(f"Time: {bold}{now_utc}{reset} | Live Equity: {green}${equity:,.2f}{reset} (Risk: {cfg.risk.risk_percent}% = ${equity * cfg.risk.risk_percent / 100:.2f})\n")
            print(f"{bold}{'Asset':<10} {'Broker Symbol':<14} {'Bid / Ask':<22} {'Spread':<18} {'Gatekeeper & Strategy Status'}{reset}")
            print("─" * 105)
            for asset, sym, ba, sp, st in scan_records:
                print(f"{bold}{asset:<10}{reset} {sym:<14} {ba:<22} {sp:<18} {st}")
            print("─" * 105)
            if signals_emitted == 0:
                print(f"{yellow}Notice: No setups met all 3 gatekeepers (Session Window, ADX/ATR Volatility, 3+ Confluences & 1:2.5 R:R).{reset}\n")
            else:
                print(f"{green}{bold}✔ {signals_emitted} High-Probability Signal(s) Emitted!{reset}\n")

    def start_loop(self, interval_seconds: int = 30):
        """
        Runs the continuous algorithmic scanning loop until interrupted.
        """
        if not self.connector.is_connected:
            if not self.connector.connect():
                logger.error("Failed to connect to MT5 terminal. Exiting.")
                return

        self.running = True
        logger.info(f"Zuperior Signal Engine started. Scanning every {interval_seconds}s...")

        while self.running:
            try:
                self.scan_once()
            except Exception as e:
                logger.error(f"Unexpected error in scan iteration: {e}")

            # Sleep in 1-second chunks for responsive SIGINT handling
            for _ in range(interval_seconds):
                if not self.running:
                    break
                time.sleep(1)

        self.connector.disconnect()
        logger.info("Signal Engine stopped cleanly.")

    def stop(self):
        """Stops the scanning loop."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Zuperior MT5 Production Algorithmic Signal Engine")
    parser.add_argument("--verify-symbols", action="store_true", help="Connect to MT5, verify broker symbols, spreads, contract sizes, and data flow")
    parser.add_argument("--once", action="store_true", help="Perform a single market scan across all assets and exit")
    parser.add_argument("--backtest", choices=["GOLD", "BITCOIN", "NASDAQ", "ALL"], help="Run backtester on specified asset")
    parser.add_argument("--days", type=int, default=30, help="Number of historical days for backtest (default 30)")
    parser.add_argument("--interval", type=int, default=cfg.scan_interval_seconds, help="Scan interval in seconds (default 30)")
    args = parser.parse_args()

    engine = ZuperiorSignalEngine()

    # Graceful shutdown handler
    def handle_sigint(signum, frame):
        logger.info("Shutdown signal received (Ctrl+C). Terminating...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    if args.verify_symbols:
        success = engine.verify_symbols()
        sys.exit(0 if success else 1)

    if args.backtest:
        backtester = Backtester(connector=engine.connector)
        assets = ["GOLD", "BITCOIN", "NASDAQ"] if args.backtest == "ALL" else [args.backtest]
        for a in assets:
            backtester.run_backtest(asset_key=a, days=args.days)
        sys.exit(0)

    if args.once:
        if engine.connector.connect():
            engine.scan_once()
            engine.connector.disconnect()
        sys.exit(0)

    # Default: start continuous scanning loop
    engine.start_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
