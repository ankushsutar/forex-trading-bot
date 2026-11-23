"""
mt5_connector.py
----------------
Robust wrapper around the MetaTrader 5 (MT5) Python API tailored for Zuperior accounts.
Handles connection lifecycle, auto-resolving broker-specific symbol suffixes (.pro, .raw, .ecn, m),
contract specifications, live ticks, and OHLCV bar fetching across M5, M15, H1, and H4 timeframes.
Includes a built-in Zuperior mock bridge for Linux testing and dry-run verification.
"""

import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config import cfg

logger = logging.getLogger("MT5Connector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Attempt importing the official MetaTrader5 package
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5_lib
    MT5_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    mt5_lib = None
    logger.warning("Official MetaTrader5 package is not installed or unavailable in this environment (common on non-Windows/Wine).")


class ZuperiorMockMT5:
    """
    High-fidelity simulation bridge mimicking a live Zuperior MetaTrader 5 broker terminal.
    Enables symbol verification, dry runs, unit tests, and strategy backtesting outside Windows.
    """
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440

    def __init__(self):
        self._connected = False
        self._rates_cache = {}
        # Zuperior broker mock specifications
        self._symbols_catalog = {
            "XAUUSD.pro": {
                "name": "XAUUSD.pro", "point": 0.01, "digits": 2, "spread": 16,
                "trade_contract_size": 100.0, "trade_tick_value": 1.0, "trade_tick_size": 0.01,
                "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
                "currency_base": "XAU", "currency_profit": "USD", "currency_margin": "USD",
                "visible": True, "base_price": 2650.00, "spread_price": 0.16
            },
            "BTCUSD.pro": {
                "name": "BTCUSD.pro", "point": 0.01, "digits": 2, "spread": 1200,
                "trade_contract_size": 1.0, "trade_tick_value": 0.01, "trade_tick_size": 0.01,
                "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
                "currency_base": "BTC", "currency_profit": "USD", "currency_margin": "USD",
                "visible": True, "base_price": 96500.00, "spread_price": 12.00
            },
            "USTEC.pro": {
                "name": "USTEC.pro", "point": 0.01, "digits": 2, "spread": 90,
                "trade_contract_size": 20.0, "trade_tick_value": 0.20, "trade_tick_size": 0.01,
                "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
                "currency_base": "USD", "currency_profit": "USD", "currency_margin": "USD",
                "visible": True, "base_price": 20850.00, "spread_price": 0.90
            }
        }

    def initialize(self, path=None, login=None, password=None, server=None, timeout=None):
        self._connected = True
        return True

    def login(self, login, password=None, server=None, timeout=None):
        self._connected = True
        return True

    def shutdown(self):
        self._connected = False

    def terminal_info(self):
        class TerminalInfo:
            connected = True
            name = "MetaTrader 5 Zuperior Edition"
            build = 4450
            company = "Zuperior Prime Liquidity Ltd"
        return TerminalInfo()

    def version(self):
        return (500, 4450, "15 Nov 2026")

    def last_error(self):
        return (1, "Success")

    def account_info(self):
        class AccountInfo:
            login = cfg.mt5.login if cfg.mt5.login else 1089542
            trade_mode = 0  # 0 demo, 2 real
            leverage = 500
            balance = 10000.00
            equity = 10000.00
            margin = 0.00
            margin_free = 10000.00
            currency = "USD"
            server = cfg.mt5.server if cfg.mt5.server else "Zuperior-Live"
            company = "Zuperior Financial Services"
        return AccountInfo()

    def symbols_get(self, group=None):
        symbols = []
        for name, data in self._symbols_catalog.items():
            class SymbolObj:
                pass
            obj = SymbolObj()
            for k, v in data.items():
                setattr(obj, k, v)
            symbols.append(obj)
        return symbols

    def symbol_select(self, symbol, enable=True):
        return symbol in self._symbols_catalog

    def symbol_info(self, symbol):
        if symbol not in self._symbols_catalog:
            return None
        data = self._symbols_catalog[symbol]
        class SymbolInfo:
            pass
        obj = SymbolInfo()
        for k, v in data.items():
            setattr(obj, k, v)
        return obj

    def symbol_info_tick(self, symbol):
        if symbol not in self._symbols_catalog:
            return None
        data = self._symbols_catalog[symbol]
        base = data["base_price"]
        spread_val = data["spread_price"]

        class Tick:
            time = int(datetime.now(timezone.utc).timestamp())
            bid = base
            ask = base + spread_val
            last = base
            volume = 120
            flags = 6
        return Tick()

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        if symbol not in self._symbols_catalog:
            return None
        return self._generate_synthetic_rates(symbol, timeframe, count)

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        if symbol not in self._symbols_catalog:
            return None
        # Estimate count based on timeframe
        tf_minutes = timeframe if isinstance(timeframe, int) else 5
        duration_mins = int((date_to - date_from).total_seconds() / 60)
        count = max(10, duration_mins // tf_minutes)
        return self._generate_synthetic_rates(symbol, timeframe, min(count, 5000), end_time=date_to)

    def _generate_synthetic_rates(self, symbol, timeframe, count, end_time=None):
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        elif end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        tf_minutes = timeframe if isinstance(timeframe, int) else 5
        data = self._symbols_catalog[symbol]
        base_price = data["base_price"]
        spread_pts = data["spread"]

        np.random.seed(42 + tf_minutes)
        # Generate random walk with trend and volatility
        volatility = base_price * 0.0008
        steps = np.random.normal(0, volatility, count)
        close_prices = base_price + np.cumsum(steps)

        records = []
        for i in range(count):
            bar_time = end_time - timedelta(minutes=(count - 1 - i) * tf_minutes)
            bar_ts = int(bar_time.timestamp())
            c = float(close_prices[i])
            spread_noise = spread_pts + int(np.random.choice([0, 1, 2, -1]))
            spread_noise = max(data["spread"] // 2, spread_noise)

            h = c + abs(np.random.normal(0, volatility * 0.5))
            l = c - abs(np.random.normal(0, volatility * 0.5))
            o = float(close_prices[i - 1]) if i > 0 else (c - np.random.normal(0, volatility * 0.2))

            h = max(h, o, c)
            l = min(l, o, c)
            vol = int(np.random.randint(100, 1500))

            records.append((bar_ts, o, h, l, c, vol, spread_noise, 0))

        dtype = [
            ('time', '<i8'), ('open', '<f8'), ('high', '<f8'),
            ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<u8'),
            ('spread', '<i4'), ('real_volume', '<u8')
        ]
        return np.array(records, dtype=dtype)


class MT5Connector:
    """
    Production-grade MT5 connector supporting official MetaTrader5 API
    with automatic failover to Zuperior mock bridge when running on Linux.
    """

    TIMEFRAME_MAP = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
    }

    def __init__(self):
        self.is_connected = False
        self.is_mock = False
        self.is_binance = False
        self.resolved_symbols: Dict[str, str] = {}
        self.symbol_specs_cache: Dict[str, dict] = {}

        # Decide whether to use real MT5, live Binance feed, or mock bridge
        if not MT5_AVAILABLE or cfg.mt5.use_mock:
            if cfg.mt5.use_binance:
                from binance_feed import BinanceLiveFeed
                logger.info("Initializing in Real-Time Live Market Feed mode (Binance for Crypto/Gold, Live Futures for Nasdaq).")
                self.api = BinanceLiveFeed()
                self.is_binance = True
                self.is_mock = False
            else:
                logger.info("Initializing in Zuperior Simulation / Mock Bridge mode.")
                self.api = ZuperiorMockMT5()
                self.is_mock = True
        else:
            self.api = mt5_lib
            self.is_mock = False

    def connect(self) -> bool:
        """
        Initializes connection to the running Zuperior MT5 terminal, Binance Live Feed, or Mock Bridge.
        """
        if self.is_binance:
            self.is_connected = self.api.initialize()
            logger.info("Connected to Real-Time Live Binance & Market Feed (fetching real live ticks and candles).")
            return self.is_connected

        if self.is_mock:
            self.is_connected = self.api.initialize()
            logger.info("Connected to Zuperior Mock Bridge (Linux / Testing mode).")
            return self.is_connected

        # Real MT5 connection
        init_kwargs = {}
        if cfg.mt5.path:
            init_kwargs["path"] = cfg.mt5.path
        if cfg.mt5.login:
            init_kwargs["login"] = cfg.mt5.login
        if cfg.mt5.password:
            init_kwargs["password"] = cfg.mt5.password
        if cfg.mt5.server:
            init_kwargs["server"] = cfg.mt5.server
        if cfg.mt5.timeout_ms:
            init_kwargs["timeout"] = cfg.mt5.timeout_ms

        logger.info(f"Attempting connection to MT5 terminal at server '{cfg.mt5.server}'...")
        if not self.api.initialize(**init_kwargs):
            err = self.api.last_error()
            logger.error(f"MT5 initialize failed: {err}")
            return False

        # If login was not handled in initialize, explicitly log in
        if cfg.mt5.login and cfg.mt5.password:
            authorized = self.api.login(
                login=cfg.mt5.login,
                password=cfg.mt5.password,
                server=cfg.mt5.server
            )
            if not authorized:
                err = self.api.last_error()
                logger.error(f"Failed to log into MT5 account {cfg.mt5.login} on {cfg.mt5.server}: {err}")
                return False

        terminal = self.api.terminal_info()
        account = self.api.account_info()
        self.is_connected = True
        logger.info(f"Successfully connected to {terminal.name} (Build {terminal.build}) on {account.server}")
        logger.info(f"Account: {account.login} | Currency: {account.currency} | Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f}")
        return True

    def disconnect(self):
        """Disconnects cleanly from the MT5 terminal."""
        if self.is_connected and self.api:
            self.api.shutdown()
            self.is_connected = False
            logger.info("Disconnected from MT5 terminal.")

    def get_account_info(self) -> dict:
        """Retrieves live account balance, equity, margin, leverage, and currency."""
        if not self.is_connected:
            raise RuntimeError("MT5 connector is not connected.")

        acc = self.api.account_info()
        if acc is None:
            raise RuntimeError(f"Failed to retrieve account info: {self.api.last_error()}")

        return {
            "login": acc.login,
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "margin_free": acc.margin_free,
            "leverage": acc.leverage,
            "currency": acc.currency,
            "server": acc.server,
            "company": acc.company
        }

    def resolve_symbol(self, asset_key: str) -> Optional[str]:
        """
        Dynamically detects the exact broker symbol in MT5 Market Watch.
        Checks candidate list and matches Zuperior broker suffixes like .pro, .raw, .ecn, m.
        """
        if asset_key in self.resolved_symbols:
            return self.resolved_symbols[asset_key]

        candidates = cfg.symbols.get(asset_key, [asset_key])
        all_symbols = self.api.symbols_get()
        if not all_symbols:
            logger.error("Unable to query broker symbols from MT5 terminal.")
            return None

        available_symbol_names = {s.name: s for s in all_symbols}

        # 1. Exact match against candidates list
        for candidate in candidates:
            if candidate in available_symbol_names:
                exact_name = candidate
                self.api.symbol_select(exact_name, True)
                self.resolved_symbols[asset_key] = exact_name
                logger.info(f"Resolved {asset_key} -> exact broker symbol: '{exact_name}'")
                return exact_name

        # 2. Case-insensitive / prefix matching
        for candidate in candidates:
            cand_lower = candidate.lower().replace(".", "").replace("_", "")
            for broker_symbol in available_symbol_names.keys():
                b_lower = broker_symbol.lower().replace(".", "").replace("_", "")
                if cand_lower == b_lower:
                    self.api.symbol_select(broker_symbol, True)
                    self.resolved_symbols[asset_key] = broker_symbol
                    logger.info(f"Resolved {asset_key} -> broker symbol (fuzzy): '{broker_symbol}'")
                    return broker_symbol

        logger.error(f"Could not resolve any broker symbol for asset '{asset_key}'. Tested: {candidates}")
        return None

    def get_symbol_specs(self, symbol: str) -> dict:
        """
        Fetches full contract specs required for exact risk and lot-size calculation:
        trade_contract_size, trade_tick_value, trade_tick_size, point, digits, volume steps.
        """
        if symbol in self.symbol_specs_cache:
            return self.symbol_specs_cache[symbol]

        info = self.api.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol '{symbol}' not found on MT5 broker.")

        # Ensure selected in Market Watch
        if not info.visible:
            self.api.symbol_select(symbol, True)

        specs = {
            "name": info.name,
            "point": getattr(info, "point", 0.01),
            "digits": getattr(info, "digits", 2),
            "spread": getattr(info, "spread", 0),
            "trade_contract_size": getattr(info, "trade_contract_size", 1.0),
            "trade_tick_value": getattr(info, "trade_tick_value", 1.0),
            "trade_tick_size": getattr(info, "trade_tick_size", getattr(info, "point", 0.01)),
            "volume_min": getattr(info, "volume_min", 0.01),
            "volume_max": getattr(info, "volume_max", 100.0),
            "volume_step": getattr(info, "volume_step", 0.01),
            "currency_base": getattr(info, "currency_base", "USD"),
            "currency_profit": getattr(info, "currency_profit", "USD"),
            "currency_margin": getattr(info, "currency_margin", "USD")
        }

        # Safety fallbacks for tick value if broker returns 0 outside active market hours
        if specs["trade_tick_value"] <= 0:
            specs["trade_tick_value"] = specs["trade_contract_size"] * specs["trade_tick_size"]

        self.symbol_specs_cache[symbol] = specs
        return specs

    def get_live_tick(self, symbol: str) -> dict:
        """
        Fetches the latest live tick for symbol and computes live spread in price units and points.
        """
        tick = self.api.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Could not fetch live tick for '{symbol}': {self.api.last_error()}")

        point = self.get_symbol_specs(symbol)["point"]
        spread_price = tick.ask - tick.bid
        spread_points = spread_price / point if point > 0 else 0.0

        return {
            "symbol": symbol,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "last": float(tick.last) if tick.last > 0 else float(tick.bid),
            "spread_price": float(spread_price),
            "spread_points": float(spread_points),
            "volume": getattr(tick, "volume", 0),
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc)
        }

    def fetch_rates(self, symbol: str, timeframe_str: str, n_bars: int = 500) -> pd.DataFrame:
        """
        Fetches the last n_bars of OHLCV + spread data and returns a cleaned pandas DataFrame with UTC DatetimeIndex.
        Timeframe options: 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'.
        """
        tf_code = self._get_timeframe_code(timeframe_str)
        rates = self.api.copy_rates_from_pos(symbol, tf_code, 0, n_bars)

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Failed to copy rates for {symbol} on {timeframe_str}: {self.api.last_error()}")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)

        # Standardize column types
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        df['tick_volume'] = df['tick_volume'].astype(int)
        df['spread'] = df['spread'].astype(int)

        return df

    def fetch_rates_range(
        self,
        symbol: str,
        timeframe_str: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Fetches historical rates within a specific datetime range for backtesting.
        Pulls actual historical spread column from Zuperior broker server.
        """
        tf_code = self._get_timeframe_code(timeframe_str)

        # MT5 requires naive UTC or localized datetime objects
        if start_dt.tzinfo is not None:
            start_dt = start_dt.astimezone(timezone.utc)
        if end_dt.tzinfo is not None:
            end_dt = end_dt.astimezone(timezone.utc)

        rates = self.api.copy_rates_range(symbol, tf_code, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Failed to copy rates range for {symbol}: {self.api.last_error()}")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)

        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        df['tick_volume'] = df['tick_volume'].astype(int)
        df['spread'] = df['spread'].astype(int)

        return df

    def _get_timeframe_code(self, tf_str: str) -> Any:
        """Translates timeframe string ('M5', 'H1', etc.) to MT5 timeframe constant."""
        tf_str = tf_str.upper()
        if self.is_mock or self.is_binance:
            return self.TIMEFRAME_MAP.get(tf_str, 5)

        mt5_tf_attr = f"TIMEFRAME_{tf_str}"
        if hasattr(self.api, mt5_tf_attr):
            return getattr(self.api, mt5_tf_attr)
        raise ValueError(f"Unknown MT5 timeframe '{tf_str}'")
