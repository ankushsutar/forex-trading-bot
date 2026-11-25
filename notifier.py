"""
notifier.py
-----------
Multi-channel notification engine for trade alerts.
Dispatches formatted HTML/Markdown and JSON payloads to Telegram, Webhooks (Discord/Slack), and Terminal Console.
Features:
- Full support for both Binance Quantitative Signals and MT5 Trade Signals.
- In-memory signal deduplication with configurable cooldown to prevent alert spam.
- Built-in Telegram connection validator and test alert generator.
"""

import os
import sys
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

from config.settings import settings
from config import cfg

logger = logging.getLogger("Notifier")


class TradeNotifier:
    """
    Handles formatting, deduplication, and dispatching of trade alerts.
    """

    def __init__(self):
        # Prefer settings from config/settings.py, fallback to cfg if available
        self.enable_console: bool = True
        self.enable_telegram: bool = settings.enable_telegram
        self.telegram_bot_token: str = settings.telegram_bot_token
        self.telegram_chat_id: str = settings.telegram_chat_id
        self.enable_webhook: bool = bool(settings.webhook_url)
        self.webhook_url: str = settings.webhook_url
        self.cooldown_minutes: int = 15

        # Fallback to cfg if available and settings fields are empty
        if cfg is not None and hasattr(cfg, "notifications"):
            if not self.telegram_bot_token and cfg.notifications.telegram_bot_token:
                self.telegram_bot_token = cfg.notifications.telegram_bot_token
                self.enable_telegram = cfg.notifications.enable_telegram
            if not self.telegram_chat_id and cfg.notifications.telegram_chat_id:
                self.telegram_chat_id = cfg.notifications.telegram_chat_id
            if not self.webhook_url and cfg.notifications.webhook_url:
                self.webhook_url = cfg.notifications.webhook_url
                self.enable_webhook = cfg.notifications.enable_webhook
            self.enable_console = getattr(cfg.notifications, "enable_console", True)
            self.cooldown_minutes = getattr(cfg.notifications, "dedup_cooldown_minutes", 15)

        # In-memory deduplication cache: key -> timestamp of last dispatch
        self.signal_cache: Dict[str, float] = {}

    def is_duplicate(self, signal: Any) -> bool:
        """
        Determines if an identical signal has already been sent within the cooldown window.
        Key is composed of symbol, direction, strategy/archetype, and bar_time.
        """
        symbol = getattr(signal, "symbol", "UNKNOWN")
        direction = getattr(signal, "direction", "UNKNOWN")
        strategy = getattr(signal, "strategy_name", getattr(signal, "archetype", "STRATEGY"))
        bar_time = getattr(signal, "bar_time", "")

        key = f"{symbol}_{direction}_{strategy}_{bar_time}"
        now_ts = time.time()
        cooldown_sec = self.cooldown_minutes * 60

        # Purge stale keys older than 2 hours
        stale_threshold = now_ts - 7200
        self.signal_cache = {k: v for k, v in self.signal_cache.items() if v > stale_threshold}

        if key in self.signal_cache:
            last_sent = self.signal_cache[key]
            if (now_ts - last_sent) < cooldown_sec:
                return True

        self.signal_cache[key] = now_ts
        return False

    def notify_binance(self, signal: Any, sizing: Any, equity: float = 10000.0) -> bool:
        """
        Dispatches institutional Binance quantitative trade alerts.
        """
        if self.is_duplicate(signal):
            logger.info(f"Binance signal deduplicated: {sizing.symbol} {sizing.direction}")
            return False

        # 1. Console Output
        if self.enable_console:
            self._print_binance_console(signal, sizing, equity)

        # 2. Telegram Alert
        if self.enable_telegram and self.telegram_bot_token and self.telegram_chat_id:
            html = self._format_binance_telegram_html(signal, sizing, equity)
            self._send_telegram(html)

        # 3. Webhook Alert (Discord/Slack/JSON)
        if self.enable_webhook and self.webhook_url:
            self._send_binance_webhook(signal, sizing, equity)

        return True

    def notify(self, signal: Any) -> bool:
        """
        Dispatches MT5 trade alert (backward compatible).
        """
        if self.is_duplicate(signal):
            logger.info(f"Signal deduplicated: {signal.symbol} {signal.direction}")
            return False

        if self.enable_console:
            self._print_legacy_console(signal)

        if self.enable_telegram and self.telegram_bot_token and self.telegram_chat_id:
            html = self._format_legacy_telegram_html(signal)
            self._send_telegram(html)

        if self.enable_webhook and self.webhook_url:
            self._send_legacy_webhook(signal)

        return True

    def _format_binance_telegram_html(self, signal: Any, sizing: Any, equity: float) -> str:
        """Formats a Binance quantitative trade signal into Telegram HTML."""
        icon = "🟢" if sizing.direction == "BUY" else "🔴"
        action_text = "BUY / LONG" if sizing.direction == "BUY" else "SELL / SHORT"
        env_badge = settings.execution_env.value
        mode_badge = settings.trading_mode.value.replace("_", " ")

        # Metrics & spread
        spread_bps = signal.metrics.get("spread_bps", 0.0) if hasattr(signal, "metrics") else 0.0
        funding_rate = signal.metrics.get("funding_rate", 0.0) if hasattr(signal, "metrics") else 0.0
        funding_str = f"{funding_rate * 100:+.4f}%" if funding_rate else "0.0000%"

        confluences = getattr(signal, "confluences", [])
        confluences_html = "\n".join([f"  • {c}" for c in confluences]) if confluences else "  • Quantitative Regime & Volatility Alignment"

        notional_val = sizing.quantity * sizing.entry_price
        risk_pct = (sizing.dollar_risk / equity * 100.0) if equity > 0 else settings.risk_percent

        bar_time_str = signal.bar_time.strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(signal.bar_time, datetime) else str(signal.bar_time)

        html = (
            f"🚀 <b>BINANCE QUANTITATIVE SIGNAL ALERT</b> 🚀\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Pair:</b> <code>{sizing.symbol}</code> ({mode_badge})\n"
            f"⚡ <b>Action:</b> {icon} <b>{action_text}</b>\n"
            f"📊 <b>Archetype:</b> {signal.archetype.replace('_', ' ').title()}\n"
            f"🌐 <b>Mode:</b> <code>{env_badge}</code>\n"
            f"\n"
            f"💰 <b>ENTRY:</b> <code>${sizing.entry_price:,.2f}</code>\n"
            f"🛑 <b>STOP LOSS:</b> <code>${sizing.stop_loss:,.2f}</code> (Dist: ${abs(sizing.entry_price - sizing.stop_loss):,.2f})\n"
            f"🎯 <b>TAKE PROFIT 1:</b> <code>${sizing.take_profit_1:,.2f}</code> (<b>R:R 1:{sizing.risk_reward_ratio:.2f}</b>)\n"
            f"🚀 <b>TAKE PROFIT 2:</b> <code>${sizing.take_profit_2:,.2f}</code> (Runner)\n"
            f"\n"
            f"⚖️ <b>Position & Risk Allocation:</b>\n"
            f"• <b>Order Quantity:</b> <code>{sizing.quantity}</code> (~${notional_val:,.2f} Notional)\n"
            f"• <b>Risk Capital:</b> ${sizing.dollar_risk:,.2f} ({risk_pct:.1f}% of ${equity:,.2f})\n"
            f"• <b>Live Spread:</b> {spread_bps:.1f} bps | <b>8h Funding:</b> {funding_str}\n"
            f"\n"
            f"🔍 <b>Confluences Verified:</b>\n"
            f"{confluences_html}\n"
            f"\n"
            f"🕒 <b>Bar Timestamp:</b> <code>{bar_time_str}</code>"
        )
        return html

    def _format_legacy_telegram_html(self, signal: Any) -> str:
        """Formats an MT5 trade signal into clean HTML for Telegram."""
        icon = "🟢" if signal.direction == "BUY" else "🔴"
        spread_price = signal.metadata.get("spread_price", "N/A") if hasattr(signal, "metadata") else "N/A"
        confluences = getattr(signal, "confluences", [])
        confluences_html = "\n".join([f"• {c}" for c in confluences])

        html = (
            f"🚨 <b>ZUPERIOR MT5 SIGNAL ALERT</b> 🚨\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Asset:</b> {getattr(signal, 'asset', signal.symbol)} (<code>{signal.symbol}</code>)\n"
            f"⚡ <b>Action:</b> {icon} <b>{signal.direction}</b> ({getattr(signal, 'timeframe', 'M5')})\n"
            f"📊 <b>Strategy:</b> {getattr(signal, 'strategy_name', 'Quant Strategy')}\n"
            f"\n"
            f"💰 <b>ENTRY:</b> <code>{signal.entry_price}</code>\n"
            f"🛑 <b>STOP LOSS:</b> <code>{signal.stop_loss}</code> (Dist: {abs(signal.entry_price - signal.stop_loss):.2f})\n"
            f"🎯 <b>TAKE PROFIT 1:</b> <code>{signal.take_profit_1}</code> (<b>R:R 1:{signal.risk_reward_ratio:.2f}</b>)\n"
            f"🚀 <b>TAKE PROFIT 2:</b> <code>{signal.take_profit_2}</code>\n"
            f"\n"
            f"⚖️ <b>Sizing & Risk:</b>\n"
            f"• <b>Recommended Lot Size:</b> <code>{getattr(signal, 'recommended_lot_size', '0.01')}</code> Lots\n"
            f"• <b>Risk:</b> ${signal.dollar_risk:,.2f}\n"
            f"• <b>Live Spread:</b> {spread_price}\n"
            f"\n"
            f"🔍 <b>Confluences Verified:</b>\n"
            f"{confluences_html}\n"
            f"\n"
            f"🕒 <b>Bar Timestamp:</b> <code>{signal.bar_time} UTC</code>"
        )
        return html

    def _print_binance_console(self, signal: Any, sizing: Any, equity: float):
        """Prints formatted Binance signal alert to stdout."""
        dir_color = "\033[92m" if sizing.direction == "BUY" else "\033[91m"
        bold = "\033[1m"
        reset = "\033[0m"
        cyan = "\033[96m"
        yellow = "\033[93m"

        banner = (
            f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════╗{reset}\n"
            f"{bold}{cyan}║                 BINANCE QUANTITATIVE SIGNAL ALERT                    ║{reset}\n"
            f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════╝{reset}\n"
            f"{bold}Pair:{reset}   {sizing.symbol} ({settings.trading_mode.value}) | Mode: {settings.execution_env.value}\n"
            f"{bold}Action:{reset} {dir_color}{sizing.direction}{reset}\n"
            f"{bold}Archetype:{reset} {signal.archetype}\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"{bold}Entry Price:{reset}    ${sizing.entry_price:,.2f}\n"
            f"{bold}Stop Loss:{reset}      ${sizing.stop_loss:,.2f}  (Risk Dist: ${abs(sizing.entry_price - sizing.stop_loss):,.2f})\n"
            f"{bold}Take Profit 1:{reset}  ${sizing.take_profit_1:,.2f}  ({yellow}R:R 1:{sizing.risk_reward_ratio:.2f}{reset})\n"
            f"{bold}Take Profit 2:{reset}  ${sizing.take_profit_2:,.2f}\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"{bold}Position Sizing:{reset}\n"
            f"  • Quantity:    {bold}{yellow}{sizing.quantity}{reset} (~${sizing.quantity * sizing.entry_price:,.2f} Notional)\n"
            f"  • Risk USD:    ${sizing.dollar_risk:.2f} ({settings.risk_percent}% of ${equity:,.2f})\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"{bold}Confluences Verified:{reset}\n"
        )
        for conf in getattr(signal, "confluences", []):
            banner += f"  [✓] {conf}\n"
        banner += f"──────────────────────────────────────────────────────────────────────\n"
        banner += f"Bar Time: {signal.bar_time} UTC\n"
        print(banner)

    def _print_legacy_console(self, signal: Any):
        """Prints a colorized, human-readable alert to stdout for MT5."""
        dir_color = "\033[92m" if signal.direction == "BUY" else "\033[91m"
        bold = "\033[1m"
        reset = "\033[0m"
        cyan = "\033[96m"
        yellow = "\033[93m"

        banner = (
            f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════╗{reset}\n"
            f"{bold}{cyan}║                 ZUPERIOR MT5 SIGNAL ENGINE ALERT                     ║{reset}\n"
            f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════╝{reset}\n"
            f"{bold}Asset:{reset} {getattr(signal, 'asset', signal.symbol)} ({signal.symbol})\n"
            f"{bold}Action:{reset} {dir_color}{signal.direction}{reset}\n"
            f"{bold}Strategy:{reset} {getattr(signal, 'strategy_name', 'Strategy')}\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"{bold}Entry Price:{reset}    {signal.entry_price}\n"
            f"{bold}Stop Loss:{reset}      {signal.stop_loss}  (Risk: {abs(signal.entry_price - signal.stop_loss):.2f})\n"
            f"{bold}Take Profit 1:{reset}  {signal.take_profit_1}  ({yellow}R:R 1:{signal.risk_reward_ratio:.2f}{reset})\n"
            f"{bold}Take Profit 2:{reset}  {signal.take_profit_2}\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"Bar Time: {signal.bar_time} UTC\n"
        )
        print(banner)

    def _send_telegram(self, html_text: str):
        """Dispatches HTML message to Telegram via Bot API."""
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                logger.info(f"Telegram alert successfully dispatched to chat {self.telegram_chat_id}")
            else:
                logger.warning(f"Telegram API responded with {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to deliver Telegram alert: {e}")

    def _send_binance_webhook(self, signal: Any, sizing: Any, equity: float):
        """Dispatches structured JSON payload to Webhook (Discord / Slack / custom endpoint)."""
        payload = {
            "event": "BINANCE_QUANT_SIGNAL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": sizing.symbol,
            "direction": sizing.direction,
            "entry_price": sizing.entry_price,
            "stop_loss": sizing.stop_loss,
            "take_profit_1": sizing.take_profit_1,
            "take_profit_2": sizing.take_profit_2,
            "quantity": sizing.quantity,
            "dollar_risk": sizing.dollar_risk,
            "mode": settings.execution_env.value
        }
        if "discord.com" in self.webhook_url.lower():
            color = 0x00FF00 if sizing.direction == "BUY" else 0xFF0000
            payload["embeds"] = [{
                "title": f"🚨 BINANCE SIGNAL: {sizing.direction} {sizing.symbol}",
                "color": color,
                "fields": [
                    {"name": "Pair", "value": sizing.symbol, "inline": True},
                    {"name": "Direction", "value": sizing.direction, "inline": True},
                    {"name": "Entry", "value": f"${sizing.entry_price:,.2f}", "inline": True},
                    {"name": "Stop Loss", "value": f"${sizing.stop_loss:,.2f}", "inline": True},
                    {"name": "TP 1", "value": f"${sizing.take_profit_1:,.2f} (1:{sizing.risk_reward_ratio:.2f})", "inline": True},
                    {"name": "TP 2", "value": f"${sizing.take_profit_2:,.2f}", "inline": True},
                    {"name": "Quantity", "value": str(sizing.quantity), "inline": True},
                    {"name": "Risk USD", "value": f"${sizing.dollar_risk:.2f}", "inline": True}
                ],
                "footer": {"text": f"Bar Time: {signal.bar_time} UTC"}
            }]

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=8)
            if resp.status_code in (200, 204):
                logger.info(f"Webhook alert successfully dispatched for {sizing.symbol}")
            else:
                logger.warning(f"Webhook endpoint returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to deliver Webhook alert: {e}")

    def _send_legacy_webhook(self, signal: Any):
        """Dispatches legacy structured JSON payload to Webhook."""
        payload = {
            "event": "ZUPERIOR_MT5_SIGNAL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal.to_dict() if hasattr(signal, "to_dict") else vars(signal)
        }
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=8)
            if resp.status_code in (200, 204):
                logger.info(f"Webhook alert successfully dispatched for {signal.symbol}")
        except Exception as e:
            logger.error(f"Failed to deliver Webhook alert: {e}")

    def send_test_message(self) -> Tuple[bool, str]:
        """
        Validates the Telegram Bot Token and Chat ID by sending a test alert.
        Returns (success: bool, message: str).
        """
        if not self.telegram_bot_token:
            return False, "TELEGRAM_BOT_TOKEN is empty. Please set it in your .env file."
        if not self.telegram_chat_id:
            return False, "TELEGRAM_CHAT_ID is empty. Please set it in your .env file."

        # 1. Test getMe to verify bot token validity
        get_me_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getMe"
        try:
            r = requests.get(get_me_url, timeout=5)
            if r.status_code != 200:
                return False, f"Telegram Bot Token rejected by Telegram (HTTP {r.status_code}): {r.text}"
            bot_info = r.json().get("result", {})
            bot_username = bot_info.get("username", "UnknownBot")
        except Exception as e:
            return False, f"Network error contacting Telegram API: {e}"

        # 2. Dispatch test message to chat_id
        send_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        test_html = (
            f"✅ <b>TELEGRAM NOTIFIER TEST SUCCESSFUL</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 <b>Bot:</b> @{bot_username}\n"
            f"📱 <b>Chat ID:</b> <code>{self.telegram_chat_id}</code>\n"
            f"⚡ <b>Engine:</b> Institutional Binance Quantitative Engine\n"
            f"🕒 <b>Time:</b> <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Connection verified! Live trading signals, ATR stops, and bracket allocations will be delivered here."
        )
        try:
            resp = requests.post(send_url, json={
                "chat_id": self.telegram_chat_id,
                "text": test_html,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }, timeout=8)
            if resp.status_code == 200:
                return True, f"Telegram test message successfully delivered to chat {self.telegram_chat_id} via bot @{bot_username}!"
            else:
                return False, f"Failed to deliver to Chat ID '{self.telegram_chat_id}' (HTTP {resp.status_code}): {resp.text}"
        except Exception as e:
            return False, f"Failed to deliver Telegram test message: {e}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    notifier = TradeNotifier()
    if "--test" in sys.argv:
        print("Testing Telegram configuration...")
        success, msg = notifier.send_test_message()
        if success:
            print(f"\033[92m✔ {msg}\033[0m")
            sys.exit(0)
        else:
            print(f"\033[91m✖ {msg}\033[0m")
            sys.exit(1)
    else:
        print("Usage: python3 notifier.py --test")
