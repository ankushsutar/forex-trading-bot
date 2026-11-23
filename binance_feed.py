"""
binance_feed.py
---------------
Real-time live market data provider fetching live ticks and OHLCV klines from Binance
(for Bitcoin and Gold) and real-time futures feeds (for Nasdaq).
Provides live market data directly on Linux without requiring a running MT5 terminal.
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger("BinanceFeed")

# Mapping of engine assets to live symbols
FEED_SYMBOL_MAP = {
    "BITCOIN": "BTCUSDT",
    "GOLD": "PAXGUSDT",       # PAX Gold on Binance tracks 1 troy oz of physical gold 1:1
    "NASDAQ": "NQ=F"          # Nasdaq 100 E-mini Futures
}

TIMEFRAME_CONVERSION = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d"
}


class BinanceLiveFeed:
    """
    Direct REST feed pulling actual live market prices from Binance public API
    and live index futures quotes. No API keys required.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self._connected = True
        self._cache_ticks: Dict[str, dict] = {}
        self._last_tick_time: Dict[str, float] = {}

    def initialize(self, **kwargs) -> bool:
        self._connected = True
        return True

    def shutdown(self):
        self._connected = False

    def last_error(self):
        return (1, "Success")

    def terminal_info(self):
        class TerminalInfo:
            connected = True
            name = "Binance Real-Time Live Feed Bridge"
            build = 2026
            company = "Binance & Institutional Live Liquidity"
        return TerminalInfo()

    def account_info(self):
        from config import cfg
        class AccountInfo:
            login = cfg.mt5.login if cfg.mt5.login else 1089542
            leverage = 500
            balance = 10000.00
            equity = 10000.00
            margin = 0.00
            margin_free = 10000.00
            currency = "USD"
            server = "Binance-Live-Bridge (Zuperior Specs)"
            company = "Zuperior Financial Services"
        return AccountInfo()

    def symbols_get(self, group=None):
        symbols = []
        for name in ["XAUUSD.pro", "BTCUSD.pro", "USTEC.pro"]:
            class S:
                pass
            s = S()
            s.name = name
            s.visible = True
            symbols.append(s)
        return symbols

    def symbol_select(self, symbol, enable=True):
        return True

    def symbol_info(self, symbol: str):
        class Info:
            pass
        obj = Info()
        obj.name = symbol
        obj.visible = True

        if "XAU" in symbol or "GOLD" in symbol:
            obj.point = 0.01
            obj.digits = 2
            obj.spread = 16
            obj.trade_contract_size = 100.0
            obj.trade_tick_value = 1.00
            obj.trade_tick_size = 0.01
            obj.volume_min = 0.01
            obj.volume_max = 100.0
            obj.volume_step = 0.01
            obj.currency_base = "XAU"
            obj.currency_profit = "USD"
        elif "BTC" in symbol:
            obj.point = 0.01
            obj.digits = 2
            obj.spread = 1200
            obj.trade_contract_size = 1.0
            obj.trade_tick_value = 0.01
            obj.trade_tick_size = 0.01
            obj.volume_min = 0.01
            obj.volume_max = 50.0
            obj.volume_step = 0.01
            obj.currency_base = "BTC"
            obj.currency_profit = "USD"
        else:  # Nasdaq / USTEC
            obj.point = 0.01
            obj.digits = 2
            obj.spread = 90
            obj.trade_contract_size = 20.0
            obj.trade_tick_value = 0.20
            obj.trade_tick_size = 0.01
            obj.volume_min = 0.01
            obj.volume_max = 100.0
            obj.volume_step = 0.01
            obj.currency_base = "USD"
            obj.currency_profit = "USD"

        return obj

    def symbol_info_tick(self, symbol: str):
        """Fetches live bid and ask from Binance API or live quote feed."""
        now_ts = time.time()
        # 1-second caching to prevent hitting rate limits
        if symbol in self._cache_ticks and (now_ts - self._last_tick_time.get(symbol, 0)) < 1.0:
            return self._cache_ticks[symbol]

        tick_data = self._fetch_live_tick(symbol)
        if tick_data:
            self._cache_ticks[symbol] = tick_data
            self._last_tick_time[symbol] = now_ts
            return tick_data

        # Fallback to cached tick if request fails
        if symbol in self._cache_ticks:
            return self._cache_ticks[symbol]

        return None

    def _fetch_live_tick(self, symbol: str):
        """Internal fetcher for live ticker data."""
        class Tick:
            pass
        t = Tick()
        t.time = int(datetime.now(timezone.utc).timestamp())
        t.volume = 150
        t.flags = 6

        try:
            if "BTC" in symbol:
                url = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT"
                resp = self.session.get(url, timeout=4)
                if resp.status_code == 200:
                    d = resp.json()
                    t.bid = float(d["bidPrice"])
                    t.ask = float(d["askPrice"])
                    t.last = t.bid
                    return t

            elif "XAU" in symbol or "GOLD" in symbol:
                url = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=PAXGUSDT"
                resp = self.session.get(url, timeout=4)
                if resp.status_code == 200:
                    d = resp.json()
                    t.bid = float(d["bidPrice"])
                    t.ask = float(d["askPrice"])
                    t.last = t.bid
                    return t

            else:  # Nasdaq
                url = "https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?interval=5m&range=1d"
                resp = self.session.get(url, timeout=4)
                if resp.status_code == 200:
                    d = resp.json()
                    meta = d["chart"]["result"][0]["meta"]
                    price = float(meta["regularMarketPrice"])
                    # Use realistic institutional CFD spread of 0.90 points
                    t.bid = price
                    t.ask = price + 0.90
                    t.last = price
                    return t

        except Exception as e:
            logger.warning(f"Error fetching live tick for {symbol}: {e}")

        return None

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int):
        """Fetches live OHLCV bars from Binance / public feed."""
        tf_str = self._int_to_tf_str(timeframe)
        return self._fetch_live_klines(symbol, tf_str, count)

    def copy_rates_range(self, symbol: str, timeframe: int, date_from: datetime, date_to: datetime):
        tf_str = self._int_to_tf_str(timeframe)
        tf_mins = timeframe if isinstance(timeframe, int) else 5
        duration_mins = int((date_to - date_from).total_seconds() / 60)
        count = min(max(50, duration_mins // tf_mins), 1000)
        return self._fetch_live_klines(symbol, tf_str, count)

    def _fetch_live_klines(self, symbol: str, tf_str: str, count: int):
        """Queries Binance klines API and formats numpy structured array."""
        binance_interval = TIMEFRAME_CONVERSION.get(tf_str, "5m")
        limit = min(max(count, 50), 1000)

        records = []
        try:
            if "BTC" in symbol:
                url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={binance_interval}&limit={limit}"
                resp = self.session.get(url, timeout=6)
                if resp.status_code == 200:
                    for k in resp.json():
                        bar_ts = int(k[0] // 1000)
                        o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                        vol = int(k[8])  # Number of trades
                        spread = 1200    # points
                        records.append((bar_ts, o, h, l, c, vol, spread, 0))

            elif "XAU" in symbol or "GOLD" in symbol:
                url = f"https://api.binance.com/api/v3/klines?symbol=PAXGUSDT&interval={binance_interval}&limit={limit}"
                resp = self.session.get(url, timeout=6)
                if resp.status_code == 200:
                    for k in resp.json():
                        bar_ts = int(k[0] // 1000)
                        o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                        vol = int(k[8])
                        spread = 16
                        records.append((bar_ts, o, h, l, c, vol, spread, 0))

            else:  # Nasdaq
                # Yahoo Finance range mapping
                range_map = {"5m": "5d", "15m": "10d", "1h": "1mo", "4h": "1mo", "1d": "3mo"}
                r_val = range_map.get(binance_interval, "5d")
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?interval={binance_interval}&range={r_val}"
                resp = self.session.get(url, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()["chart"]["result"][0]
                    ts_list = data["timestamp"]
                    quote = data["indicators"]["quote"][0]
                    opens = quote.get("open", [])
                    highs = quote.get("high", [])
                    lows = quote.get("low", [])
                    closes = quote.get("close", [])
                    volumes = quote.get("volume", [])

                    for i in range(len(ts_list)):
                        if opens[i] is None or closes[i] is None:
                            continue
                        bar_ts = int(ts_list[i])
                        o = float(opens[i])
                        h = float(highs[i])
                        l = float(lows[i])
                        c = float(closes[i])
                        vol = int(volumes[i]) if volumes and volumes[i] else 100
                        spread = 90
                        records.append((bar_ts, o, h, l, c, vol, spread, 0))

        except Exception as e:
            logger.warning(f"Error fetching klines for {symbol}: {e}")

        if not records:
            return None

        # Sort chronologically
        records.sort(key=lambda x: x[0])
        records = records[-count:]

        dtype = [
            ('time', '<i8'), ('open', '<f8'), ('high', '<f8'),
            ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<u8'),
            ('spread', '<i4'), ('real_volume', '<u8')
        ]
        return np.array(records, dtype=dtype)

    def _int_to_tf_str(self, tf: int) -> str:
        reverse_map = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4", 1440: "D1"}
        return reverse_map.get(tf, "M5")
