"""
core/market_stream.py
---------------------
Resilient asynchronous WebSocket client managing Binance market data streams:
- <symbol>@kline_<interval>: filters strictly for closed bars (kline.x == True) to prevent repainting.
- <symbol>@markPrice: real-time mark price and 8-hour funding rate skew surveillance.
- get_live_spread_bps: low-latency cached REST / ticker spread monitoring to eliminate socket flooding.
Built on Python's native `websockets` library with exponential backoff reconnects and RFC 6455 ping compliance.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import requests
import websockets

from config.settings import settings, TradingMode
from config.symbols import SYMBOL_REGISTRY

logger = logging.getLogger("MarketStream")


class MarketStreamManager:
    """
    Manages multiplexed WebSocket subscriptions and maintains real-time closed-bar buffers.
    """

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "5m"):
        self.symbols = [s.upper() for s in (symbols or list(SYMBOL_REGISTRY.keys()))]
        self.timeframe = timeframe
        self.ws_base_url = settings.endpoints.ws_url
        self.rest_base_url = settings.endpoints.rest_url

        # In-memory storage of latest market state
        self.book_tickers: Dict[str, dict] = {}    # symbol -> {bid, ask, spread_bps, time}
        self.mark_prices: Dict[str, dict] = {}     # symbol -> {mark_price, funding_rate, time}
        self.klines_buffer: Dict[str, pd.DataFrame] = {} # symbol -> DataFrame of closed bars

        # Event listeners / callbacks
        self._on_closed_bar_callbacks: List[Callable[[str, pd.DataFrame], Any]] = []

        self._running = False

    def add_closed_bar_listener(self, callback: Callable[[str, pd.DataFrame], Any]):
        """Registers a callback to be triggered whenever a kline closes (kline.x == True)."""
        self._on_closed_bar_callbacks.append(callback)

    def warm_up_history(self, count: int = 150):
        """
        Synchronously fetches historical closed candles via REST to pre-warm indicator buffers.
        """
        endpoint = "/fapi/v1/klines" if settings.trading_mode == TradingMode.USD_M_FUTURES else "/api/v3/klines"
        url = f"{self.rest_base_url}{endpoint}"

        for sym in self.symbols:
            params = {
                "symbol": sym,
                "interval": self.timeframe,
                "limit": count
            }
            try:
                resp = requests.get(url, params=params, timeout=5)
                if resp.status_code == 200:
                    raw_bars = resp.json()
                    records = []
                    # All except possibly the very latest (unclosed) bar
                    for b in raw_bars[:-1]:
                        bar_time = pd.to_datetime(b[0], unit='ms', utc=True)
                        records.append({
                            'time': bar_time,
                            'open': float(b[1]),
                            'high': float(b[2]),
                            'low': float(b[3]),
                            'close': float(b[4]),
                            'volume': float(b[5]),
                            'tick_volume': int(b[8]),
                            'spread': 10
                        })
                    df = pd.DataFrame(records).set_index('time')
                    self.klines_buffer[sym] = df
                    logger.info(f"Warmed up {len(df)} historical {self.timeframe} closed bars for {sym}")
            except Exception as e:
                logger.warning(f"Could not warm up history for {sym}: {e}")

    def get_live_spread_bps(self, symbol: str) -> float:
        """
        Retrieves real-time top-of-book bid-ask spread in basis points.
        Cached for up to 5 seconds to eliminate socket flooding, otherwise refreshes from REST (Weight: 1).
        """
        now = time.time()
        cached = self.book_tickers.get(symbol)
        if cached and (now - cached.get("time", 0.0) < 5.0):
            return cached.get("spread_bps", 1.0)

        endpoint = "/fapi/v1/ticker/bookTicker" if settings.trading_mode == TradingMode.USD_M_FUTURES else "/api/v3/ticker/bookTicker"
        url = f"{self.rest_base_url}{endpoint}"

        try:
            resp = requests.get(url, params={"symbol": symbol}, timeout=3)
            if resp.status_code == 200:
                d = resp.json()
                bid = float(d.get("bidPrice", 0.0))
                ask = float(d.get("askPrice", 0.0))
                spread_bps = ((ask - bid) / ask * 10000.0) if ask > 0 else 1.0
                self.book_tickers[symbol] = {
                    "bid": bid,
                    "ask": ask,
                    "spread_bps": spread_bps,
                    "time": now
                }
                return spread_bps
        except Exception as e:
            logger.debug(f"Could not fetch REST bookTicker for {symbol}: {e}")

        return cached.get("spread_bps", 1.0) if cached else 1.0

    def build_stream_url(self) -> str:
        """Constructs multiplex stream URL for all registered symbols."""
        streams = []
        for s in self.symbols:
            s_lower = s.lower()
            streams.append(f"{s_lower}@kline_{self.timeframe}")
            if settings.trading_mode == TradingMode.USD_M_FUTURES:
                streams.append(f"{s_lower}@markPrice")

        stream_path = "/".join(streams)
        if "/stream" in self.ws_base_url:
            return f"{self.ws_base_url}?streams={stream_path}"
        return f"{self.ws_base_url}/stream?streams={stream_path}"

    async def start(self):
        """Starts the persistent WebSocket connection with exponential backoff reconnects."""
        self._running = True
        backoff = 1

        while self._running:
            url = self.build_stream_url()
            logger.info(f"Connecting to Binance WebSocket multiplex stream ({len(self.symbols)} pairs)...")
            try:
                # ping_interval=None relies on automatic RFC 6455 pong replies to server pings,
                # preventing spurious client-side timeouts and transport write contention.
                async with websockets.connect(url, ping_interval=None, close_timeout=5.0) as ws:
                    logger.info("WebSocket connected. Subscribed to live market stream.")
                    backoff = 1  # Reset backoff upon successful connection

                    while self._running:
                        try:
                            msg = await ws.recv()
                            self._handle_stream_message(msg)
                        except websockets.ConnectionClosed:
                            logger.warning("Binance WebSocket stream closed by remote server.")
                            break

            except Exception as e:
                logger.warning(f"WebSocket disconnected: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def stop(self):
        """Stops the market stream."""
        self._running = False

    def _handle_stream_message(self, raw_data: str):
        """Dispatches multiplexed stream payloads to appropriate state handlers."""
        try:
            payload = json.loads(raw_data)
            stream_name = payload.get("stream", "")
            data = payload.get("data", {})

            if "@kline" in stream_name:
                self._process_kline(data)
            elif "@markPrice" in stream_name:
                self._process_mark_price(data)

        except Exception as e:
            logger.debug(f"Error handling stream message: {e}")

    def _process_mark_price(self, d: dict):
        """Tracks Futures Mark Price and current 8-hour funding rate."""
        sym = d.get("s")
        if not sym:
            return
        mark = float(d.get("p", 0.0))
        funding_rate = float(d.get("r", 0.0))
        funding_bps = funding_rate * 10000.0

        self.mark_prices[sym] = {
            "mark_price": mark,
            "funding_rate": funding_rate,
            "funding_bps": funding_bps,
            "time": time.time()
        }

    def _process_kline(self, d: dict):
        """
        Processes kline updates.
        CRITICAL: Only triggers strategy callbacks when kline.x == True (closed candle).
        """
        k = d.get("k", {})
        sym = d.get("s")
        is_closed = k.get("x", False)

        if not is_closed or not sym:
            return  # Ignore intra-candle forming updates to prevent repainting

        bar_time = pd.to_datetime(k.get("t"), unit='ms', utc=True)
        new_row = pd.DataFrame([{
            'open': float(k.get("o")),
            'high': float(k.get("h")),
            'low': float(k.get("l")),
            'close': float(k.get("c")),
            'volume': float(k.get("v")),
            'tick_volume': int(k.get("n")),
            'spread': 10
        }], index=[bar_time])

        if sym in self.klines_buffer:
            self.klines_buffer[sym] = pd.concat([self.klines_buffer[sym], new_row])
            # Keep last 500 closed bars
            if len(self.klines_buffer[sym]) > 500:
                self.klines_buffer[sym] = self.klines_buffer[sym].iloc[-500:]
        else:
            self.klines_buffer[sym] = new_row

        logger.info(f"[{sym}] Confirmed CLOSED {self.timeframe} bar @ {new_row['close'].iloc[0]} (Vol: {new_row['volume'].iloc[0]})")

        # Notify listeners with guaranteed non-repainting closed dataframe
        for cb in self._on_closed_bar_callbacks:
            try:
                cb(sym, self.klines_buffer[sym])
            except Exception as e:
                logger.error(f"Error in closed bar listener callback: {e}", exc_info=True)
