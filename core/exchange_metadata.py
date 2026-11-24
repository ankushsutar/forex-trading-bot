"""
core/exchange_metadata.py
-------------------------
Loads and caches Binance exchangeInfo filters (PRICE_FILTER, LOT_SIZE, MIN_NOTIONAL).
Enforces exact precision rounding on order prices and quantities to eliminate
Binance API errors (-1111 Precision over maximum, -1013 Filter failure: MIN_NOTIONAL).
"""

import math
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import requests

from config.settings import settings, TradingMode
from config.symbols import SYMBOL_REGISTRY

logger = logging.getLogger("ExchangeMetadata")


@dataclass
class SymbolFilters:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_price: Decimal
    max_price: Decimal
    min_notional: Decimal
    price_precision: int
    quantity_precision: int


class ExchangeMetadata:
    """
    Caches and enforces Binance market filters and precision rules.
    """

    def __init__(self, rest_url: Optional[str] = None):
        self.rest_url = rest_url or settings.endpoints.rest_url
        self.trading_mode = settings.trading_mode
        self._filters: Dict[str, SymbolFilters] = {}
        self._load_fallback_defaults()

    def _load_fallback_defaults(self):
        """Pre-populates filters from SYMBOL_REGISTRY as resilient defaults."""
        for sym, cfg in SYMBOL_REGISTRY.items():
            tick_d = Decimal(str(cfg.default_tick_size))
            step_d = Decimal(str(cfg.default_step_size))
            min_notional_d = Decimal(str(cfg.default_min_notional))

            self._filters[sym] = SymbolFilters(
                symbol=sym,
                tick_size=tick_d,
                step_size=step_d,
                min_qty=step_d,
                max_qty=Decimal("1000000"),
                min_price=tick_d,
                max_price=Decimal("1000000"),
                min_notional=min_notional_d,
                price_precision=self._decimal_precision(tick_d),
                quantity_precision=self._decimal_precision(step_d)
            )

    @staticmethod
    def _decimal_precision(d: Decimal) -> int:
        """Determines decimal places of a Decimal step/tick."""
        tuple_form = d.as_tuple()
        exponent = tuple_form.exponent
        return max(0, -exponent) if exponent < 0 else 0

    def refresh(self, timeout: int = 5) -> bool:
        """
        Pulls live exchangeInfo from Binance REST endpoint and updates cached filters.
        """
        endpoint = "/fapi/v1/exchangeInfo" if self.trading_mode == TradingMode.USD_M_FUTURES else "/api/v3/exchangeInfo"
        url = f"{self.rest_url}{endpoint}"

        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch exchangeInfo from {url} (HTTP {resp.status_code}). Retaining defaults.")
                return False

            data = resp.json()
            symbols_list = data.get("symbols", [])
            for s in symbols_list:
                sym_name = s.get("symbol")
                if sym_name not in SYMBOL_REGISTRY and sym_name not in self._filters:
                    continue

                filters_dict = {f["filterType"]: f for f in s.get("filters", [])}

                # PRICE_FILTER
                pf = filters_dict.get("PRICE_FILTER", {})
                tick_size = Decimal(str(pf.get("tickSize", "0.01")))
                min_price = Decimal(str(pf.get("minPrice", "0.01")))
                max_price = Decimal(str(pf.get("maxPrice", "1000000")))

                # LOT_SIZE
                ls = filters_dict.get("LOT_SIZE", {})
                step_size = Decimal(str(ls.get("stepSize", "0.001")))
                min_qty = Decimal(str(ls.get("minQty", "0.001")))
                max_qty = Decimal(str(ls.get("maxQty", "1000000")))

                # MIN_NOTIONAL / NOTIONAL
                mnf = filters_dict.get("MIN_NOTIONAL", filters_dict.get("NOTIONAL", {}))
                min_notional = Decimal(str(mnf.get("notional", mnf.get("minNotional", "5.0"))))

                self._filters[sym_name] = SymbolFilters(
                    symbol=sym_name,
                    tick_size=tick_size,
                    step_size=step_size,
                    min_qty=min_qty,
                    max_qty=max_qty,
                    min_price=min_price,
                    max_price=max_price,
                    min_notional=min_notional,
                    price_precision=self._decimal_precision(tick_size),
                    quantity_precision=self._decimal_precision(step_size)
                )

            logger.info(f"Loaded live exchangeInfo filters for {len(self._filters)} registered symbols.")
            return True

        except Exception as e:
            logger.warning(f"Error fetching live exchangeInfo: {e}. Using cached fallback filters.")
            return False

    def get_filters(self, symbol: str) -> SymbolFilters:
        """Retrieves cached filters for symbol, falling back to registered defaults."""
        if symbol not in self._filters:
            self._load_fallback_defaults()
        return self._filters[symbol]

    def round_price(self, symbol: str, price: float) -> float:
        """
        Rounds price to symbol's exact tickSize using standard half-up rounding.
        """
        filt = self.get_filters(symbol)
        p_dec = Decimal(str(price))
        # Quantize to tickSize
        rounded = (p_dec / filt.tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * filt.tick_size
        return float(rounded)

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """
        Rounds quantity DOWN to symbol's stepSize (never exceeds intended risk).
        """
        filt = self.get_filters(symbol)
        q_dec = Decimal(str(quantity))
        # Floor quantize to stepSize
        rounded = (q_dec / filt.step_size).quantize(Decimal("1"), rounding=ROUND_FLOOR) * filt.step_size
        return float(rounded)

    def validate_order(self, symbol: str, price: float, quantity: float) -> Tuple[bool, str]:
        """
        Validates order parameters against LOT_SIZE, PRICE_FILTER, and MIN_NOTIONAL.
        """
        filt = self.get_filters(symbol)
        p_dec = Decimal(str(price))
        q_dec = Decimal(str(quantity))

        if q_dec < filt.min_qty:
            return False, f"Quantity {quantity} < minimum allowed {filt.min_qty} for {symbol}"

        if q_dec > filt.max_qty:
            return False, f"Quantity {quantity} > maximum allowed {filt.max_qty} for {symbol}"

        if p_dec < filt.min_price or p_dec > filt.max_price:
            return False, f"Price {price} outside allowed range [{filt.min_price}, {filt.max_price}] for {symbol}"

        notional = p_dec * q_dec
        if notional < filt.min_notional:
            return False, f"Order notional value ${float(notional):.2f} < minimum required ${filt.min_notional} for {symbol}"

        return True, "Valid"
