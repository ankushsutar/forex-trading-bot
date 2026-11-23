"""
config.py
---------
Configuration management for the Zuperior MetaTrader 5 Algorithmic Signal Engine.
Loads credentials, risk settings, session windows, regime thresholds, and symbol candidates from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass(frozen=True)
class MT5Credentials:
    """Zuperior MetaTrader 5 account credentials and connection settings."""
    login: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "Zuperior-Demo")
    path: str = os.getenv("MT5_PATH", "")  # Optional path to terminal64.exe
    timeout_ms: int = int(os.getenv("MT5_TIMEOUT_MS", "60000"))
    use_mock: bool = os.getenv("USE_MOCK_MT5", "false").lower() in ("true", "1", "yes")
    use_binance: bool = os.getenv("USE_BINANCE", "true").lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class RiskParameters:
    """Dynamic risk management and lot-sizing parameters."""
    risk_percent: float = float(os.getenv("RISK_PERCENT", "1.0"))  # Default 1.0% of live equity
    max_risk_usd: float = float(os.getenv("MAX_RISK_USD", "1000.0"))  # Hard cap per single trade
    min_risk_reward: float = float(os.getenv("MIN_RR_RATIO", "2.5"))  # Minimum 1:2.5 R:R gatekeeper
    tp1_rr_ratio: float = float(os.getenv("TP1_RR_RATIO", "2.5"))  # TP1 target at 1:2.5
    tp2_rr_ratio: float = float(os.getenv("TP2_RR_RATIO", "3.5"))  # TP2 target at 1:3.5
    atr_stop_multiplier: float = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))
    max_daily_loss_percent: float = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "3.0"))
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))


@dataclass(frozen=True)
class GatekeeperLimits:
    """Strict Zuperior live spread and session window filtering rules."""
    # Signals rejected if live spread > rolling_median_spread * multiplier
    spread_median_multiplier: float = float(os.getenv("SPREAD_MEDIAN_MULTIPLIER", "1.8"))
    rolling_spread_bars: int = int(os.getenv("ROLLING_SPREAD_BARS", "50"))

    # Hard absolute maximum spread allowed in points/price units
    max_spread_gold: float = float(os.getenv("MAX_SPREAD_GOLD", "0.60"))  # $0.60 (60 points on 2 decimals)
    max_spread_btc: float = float(os.getenv("MAX_SPREAD_BTC", "50.0"))    # $50.00
    max_spread_nasdaq: float = float(os.getenv("MAX_SPREAD_NASDAQ", "3.5"))  # 3.5 index points

    # Peak Liquidity Windows in UTC (HH:MM format)
    # Gold: London/NY morning overlap 12:00 - 16:30 UTC
    gold_session_start_utc: str = os.getenv("GOLD_SESSION_START_UTC", "12:00")
    gold_session_end_utc: str = os.getenv("GOLD_SESSION_END_UTC", "16:30")

    # Nasdaq: US Cash Open (14:30 UTC) through peak liquidity (17:00 UTC)
    nasdaq_session_start_utc: str = os.getenv("NASDAQ_SESSION_START_UTC", "14:30")
    nasdaq_session_end_utc: str = os.getenv("NASDAQ_SESSION_END_UTC", "17:00")

    # Bitcoin: 24/7 trading, but weekend chop filter enabled
    btc_filter_weekend_chop: bool = os.getenv("BTC_FILTER_WEEKEND_CHOP", "true").lower() in ("true", "1", "yes")
    btc_weekend_min_adx: float = float(os.getenv("BTC_WEEKEND_MIN_ADX", "22.0"))


@dataclass(frozen=True)
class RegimeThresholds:
    """Multi-timeframe technical indicator thresholds."""
    adx_period: int = 14
    adx_breakout_threshold: float = float(os.getenv("ADX_BREAKOUT_THRESHOLD", "25.0"))
    adx_reversal_threshold: float = float(os.getenv("ADX_REVERSAL_THRESHOLD", "22.0"))
    atr_period: int = 14
    atr_expansion_window: int = 20
    ema_fast_period: int = 50
    ema_slow_period: int = 200

    # Bollinger & Keltner Squeeze parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0
    kc_period: int = 20
    kc_atr_multiplier: float = 1.5

    # Bitcoin volume spike threshold
    btc_volume_spike_multiplier: float = float(os.getenv("BTC_VOLUME_SPIKE_MULTIPLIER", "1.8"))

    # Minimum confluence factors required to emit alert
    min_confluences: int = int(os.getenv("MIN_CONFLUENCES", "3"))


@dataclass(frozen=True)
class NotificationConfig:
    """Telegram and Webhook notification credentials."""
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    webhook_url: str = os.getenv("WEBHOOK_URL", "")
    enable_telegram: bool = os.getenv("ENABLE_TELEGRAM", "true").lower() in ("true", "1", "yes")
    enable_webhook: bool = os.getenv("ENABLE_WEBHOOK", "true").lower() in ("true", "1", "yes")
    enable_console: bool = os.getenv("ENABLE_CONSOLE", "true").lower() in ("true", "1", "yes")
    dedup_cooldown_minutes: int = int(os.getenv("DEDUP_COOLDOWN_MINUTES", "15"))


# Dynamic symbol mapping candidate lists tailored to Zuperior broker specifications
# Zuperior accounts feature suffixes like .pro, .raw, .ecn, m, s, _i, +
SYMBOL_CANDIDATES: Dict[str, List[str]] = {
    "GOLD": [
        "XAUUSD.pro", "XAUUSD.raw", "XAUUSD.ecn", "XAUUSDm", "XAUUSD.s",
        "XAUUSD_i", "XAUUSD+", "XAUUSD", "GOLD.pro", "GOLD.raw", "GOLD"
    ],
    "BITCOIN": [
        "BTCUSD.pro", "BTCUSD.raw", "BTCUSD.ecn", "BTCUSDm", "BTCUSD.s",
        "BTCUSD+", "BTCUSD", "BTCUSDT.pro", "BTCUSDT.raw", "BTCUSDT", "BITCOIN"
    ],
    "NASDAQ": [
        "USTEC.pro", "USTEC.raw", "USTEC.ecn", "USTECm", "USTEC",
        "NAS100.pro", "NAS100.raw", "NAS100.ecn", "NAS100m", "NAS100",
        "US100.pro", "US100.raw", "US100.ecn", "US100m", "US100", "USTECH"
    ]
}


@dataclass
class EngineConfig:
    """Master configuration container for the signal engine."""
    mt5: MT5Credentials = field(default_factory=MT5Credentials)
    risk: RiskParameters = field(default_factory=RiskParameters)
    gatekeeper: GatekeeperLimits = field(default_factory=GatekeeperLimits)
    regime: RegimeThresholds = field(default_factory=RegimeThresholds)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    symbols: Dict[str, List[str]] = field(default_factory=lambda: SYMBOL_CANDIDATES)
    scan_interval_seconds: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "30"))


# Global singleton instance
cfg = EngineConfig()
