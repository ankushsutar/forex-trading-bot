"""
config/settings.py
------------------
Centralized configuration for the Binance Quantitative Algorithmic Trading Engine.
Manages API endpoints, operational modes (Spot / USD-M Futures), risk parameters,
and telemetry destinations.
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


class TradingMode(str, Enum):
    SPOT = "SPOT"
    USD_M_FUTURES = "USD_M_FUTURES"


class ExecutionEnvironment(str, Enum):
    LIVE = "LIVE"
    PAPER = "PAPER"
    BACKTEST = "BACKTEST"


@dataclass(frozen=True)
class BinanceEndpoints:
    mode: TradingMode
    testnet: bool

    @property
    def rest_url(self) -> str:
        if self.mode == TradingMode.USD_M_FUTURES:
            return "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
        else:
            return "https://testnet.binance.vision" if self.testnet else "https://api.binance.com"

    @property
    def ws_url(self) -> str:
        if self.mode == TradingMode.USD_M_FUTURES:
            return "wss://stream.binancefuture.com" if self.testnet else "wss://fstream.binance.com"
        else:
            return "wss://testnet.binance.vision" if self.testnet else "wss://stream.binance.com:9443"


@dataclass
class AppSettings:
    """Master configuration container."""
    # API Credentials
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    trading_mode: TradingMode = TradingMode(os.getenv("TRADING_MODE", "USD_M_FUTURES").upper())
    testnet: bool = os.getenv("BINANCE_TESTNET", "false").lower() in ("true", "1", "yes")
    execution_env: ExecutionEnvironment = ExecutionEnvironment(os.getenv("EXECUTION_ENV", "PAPER").upper())

    # Clock Synchronization & Network Tolerance
    recv_window: int = int(os.getenv("RECV_WINDOW", "5000"))
    max_used_weight_threshold: int = int(os.getenv("MAX_USED_WEIGHT_THRESHOLD", "1000"))  # Throttle if 1m weight > 1000

    # Risk Management & Portfolio Circuit Breakers
    risk_percent: float = float(os.getenv("RISK_PERCENT", "1.0"))  # 1.0% equity risk per trade
    max_risk_usd: float = float(os.getenv("MAX_RISK_USD", "500.0"))
    min_rr_ratio: float = float(os.getenv("MIN_RR_RATIO", "2.5"))
    tp1_rr_ratio: float = float(os.getenv("TP1_RR_RATIO", "2.5"))
    tp2_rr_ratio: float = float(os.getenv("TP2_RR_RATIO", "4.0"))
    max_daily_drawdown_percent: float = float(os.getenv("MAX_DAILY_DRAWDOWN_PERCENT", "3.0"))  # 3% daily halt
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

    # Telemetry & Storage
    db_path: str = os.getenv("DATABASE_PATH", "data/trades.db")
    csv_telemetry_path: str = os.getenv("CSV_TELEMETRY_PATH", "data/telemetry.csv")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Notifications
    enable_telegram: bool = os.getenv("ENABLE_TELEGRAM", "false").lower() in ("true", "1", "yes")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    webhook_url: str = os.getenv("WEBHOOK_URL", "")
    enable_webhook: bool = os.getenv("ENABLE_WEBHOOK", "false").lower() in ("true", "1", "yes")

    @property
    def endpoints(self) -> BinanceEndpoints:
        return BinanceEndpoints(mode=self.trading_mode, testnet=self.testnet)


settings = AppSettings()
