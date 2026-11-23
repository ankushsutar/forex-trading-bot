"""
notifier.py
-----------
Multi-channel notification engine for trade alerts.
Dispatches formatted Markdown/HTML and JSON payloads to Telegram, Webhooks, and Terminal Console.
Includes signal deduplication and cooldown caching to eliminate alert spam.
"""

import time
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from config import cfg
from strategies.base_strategy import TradeSignal

logger = logging.getLogger("Notifier")


class TradeNotifier:
    """
    Handles formatting, deduplication, and dispatching of trade alerts.
    """

    def __init__(self):
        self.cfg = cfg
        # In-memory deduplication cache: key -> timestamp of last dispatch
        self.signal_cache: Dict[str, float] = {}

    def is_duplicate(self, signal: TradeSignal) -> bool:
        """
        Determines if an identical signal has already been sent within the cooldown window.
        Key is composed of symbol, direction, strategy, and bar_time.
        """
        key = f"{signal.symbol}_{signal.direction}_{signal.strategy_name}_{signal.bar_time}"
        now_ts = time.time()
        cooldown_sec = self.cfg.notifications.dedup_cooldown_minutes * 60

        # Purge stale keys older than 2 hours
        stale_threshold = now_ts - 7200
        self.signal_cache = {k: v for k, v in self.signal_cache.items() if v > stale_threshold}

        if key in self.signal_cache:
            last_sent = self.signal_cache[key]
            if (now_ts - last_sent) < cooldown_sec:
                return True

        self.signal_cache[key] = now_ts
        return False

    def notify(self, signal: TradeSignal) -> bool:
        """
        Dispatches signal across all enabled channels (Console, Telegram, Webhook).
        Returns True if signal was newly dispatched, False if deduplicated or silenced.
        """
        if self.is_duplicate(signal):
            logger.info(f"Signal deduplicated (already alerted): {signal.symbol} {signal.direction}")
            return False

        # 1. Console Output
        if self.cfg.notifications.enable_console:
            self._print_console(signal)

        # 2. Telegram Alert
        if self.cfg.notifications.enable_telegram and self.cfg.notifications.telegram_bot_token and self.cfg.notifications.telegram_chat_id:
            self._send_telegram(signal)

        # 3. Webhook Alert (JSON payload)
        if self.cfg.notifications.enable_webhook and self.cfg.notifications.webhook_url:
            self._send_webhook(signal)

        return True

    def _print_console(self, signal: TradeSignal):
        """Prints a colorized, human-readable alert to stdout."""
        dir_color = "\033[92m" if signal.direction == "BUY" else "\033[91m"  # Green / Red
        bold = "\033[1m"
        reset = "\033[0m"
        cyan = "\033[96m"
        yellow = "\033[93m"

        spread_info = ""
        if "spread_price" in signal.metadata:
            spread_info = f" | Spread: {signal.metadata['spread_price']}"

        banner = (
            f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════╗{reset}\n"
            f"{bold}{cyan}║                 ZUPERIOR MT5 SIGNAL ENGINE ALERT                     ║{reset}\n"
            f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════╝{reset}\n"
            f"{bold}Asset:{reset} {signal.asset} ({signal.symbol}){spread_info}\n"
            f"{bold}Action:{reset} {dir_color}{signal.direction} ({signal.timeframe}){reset}\n"
            f"{bold}Strategy:{reset} {signal.strategy_name}\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"{bold}Entry Price:{reset}    {signal.entry_price}\n"
            f"{bold}Stop Loss:{reset}      {signal.stop_loss}  (Risk: {abs(signal.entry_price - signal.stop_loss):.2f})\n"
            f"{bold}Take Profit 1:{reset}  {signal.take_profit_1}  ({yellow}R:R 1:{signal.risk_reward_ratio:.2f}{reset})\n"
            f"{bold}Take Profit 2:{reset}  {signal.take_profit_2}\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"{bold}Risk Sizing:{reset}\n"
            f"  • Recommended Lot Size: {bold}{yellow}{signal.recommended_lot_size} Lots{reset}\n"
            f"  • Cash Risk (Equity %): ${signal.dollar_risk:.2f} ({self.cfg.risk.risk_percent}% of ${signal.account_equity:,.2f})\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"{bold}Confluences Verified:{reset}\n"
        )
        for conf in signal.confluences:
            banner += f"  [✓] {conf}\n"
        banner += f"──────────────────────────────────────────────────────────────────────\n"
        banner += f"Bar Time: {signal.bar_time} UTC\n"

        print(banner)

    def _format_telegram_html(self, signal: TradeSignal) -> str:
        """Formats the trade signal into clean HTML for Telegram."""
        icon = "🟢" if signal.direction == "BUY" else "🔴"
        spread_price = signal.metadata.get("spread_price", "N/A")

        confluences_html = "\n".join([f"• {c}" for c in signal.confluences])

        html = (
            f"🚨 <b>ZUPERIOR MT5 SIGNAL ENGINE</b> 🚨\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Asset:</b> {signal.asset} (<code>{signal.symbol}</code>)\n"
            f"⚡ <b>Action:</b> {icon} <b>{signal.direction}</b> ({signal.timeframe})\n"
            f"📊 <b>Strategy:</b> {signal.strategy_name}\n"
            f"\n"
            f"💰 <b>ENTRY:</b> <code>{signal.entry_price}</code>\n"
            f"🛑 <b>STOP LOSS:</b> <code>{signal.stop_loss}</code> (Dist: {abs(signal.entry_price - signal.stop_loss):.2f})\n"
            f"🎯 <b>TAKE PROFIT 1:</b> <code>{signal.take_profit_1}</code> (<b>R:R 1:{signal.risk_reward_ratio:.2f}</b>)\n"
            f"🚀 <b>TAKE PROFIT 2:</b> <code>{signal.take_profit_2}</code>\n"
            f"\n"
            f"⚖️ <b>Zuperior Sizing & Risk:</b>\n"
            f"• <b>Recommended Lot Size:</b> <code>{signal.recommended_lot_size}</code> Lots\n"
            f"• <b>Risk:</b> ${signal.dollar_risk:,.2f} ({self.cfg.risk.risk_percent}% of ${signal.account_equity:,.2f})\n"
            f"• <b>Live Spread:</b> {spread_price}\n"
            f"\n"
            f"🔍 <b>Confluence Checklist (3+ Aligned):</b>\n"
            f"{confluences_html}\n"
            f"\n"
            f"🕒 <b>Bar Timestamp:</b> <code>{signal.bar_time} UTC</code>"
        )
        return html

    def _send_telegram(self, signal: TradeSignal):
        """Dispatches HTML message to Telegram via Bot API."""
        url = f"https://api.telegram.org/bot{self.cfg.notifications.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.cfg.notifications.telegram_chat_id,
            "text": self._format_telegram_html(signal),
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                logger.info(f"Telegram alert successfully dispatched for {signal.symbol} {signal.direction}")
            else:
                logger.warning(f"Telegram API responded with {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to deliver Telegram alert: {e}")

    def _send_webhook(self, signal: TradeSignal):
        """Dispatches structured JSON payload to Webhook (Discord / Slack / custom endpoint)."""
        payload = {
            "event": "ZUPERIOR_MT5_SIGNAL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal.to_dict()
        }

        # Discord webhook compatibility: add 'content' or 'embeds'
        if "discord.com" in self.cfg.notifications.webhook_url.lower():
            color = 0x00FF00 if signal.direction == "BUY" else 0xFF0000
            payload["embeds"] = [{
                "title": f"🚨 ZUPERIOR MT5 SIGNAL: {signal.direction} {signal.symbol}",
                "color": color,
                "fields": [
                    {"name": "Asset", "value": signal.asset, "inline": True},
                    {"name": "Strategy", "value": signal.strategy_name, "inline": True},
                    {"name": "Timeframe", "value": signal.timeframe, "inline": True},
                    {"name": "Entry", "value": str(signal.entry_price), "inline": True},
                    {"name": "Stop Loss", "value": str(signal.stop_loss), "inline": True},
                    {"name": "TP 1", "value": f"{signal.take_profit_1} (1:{signal.risk_reward_ratio:.2f})", "inline": True},
                    {"name": "TP 2", "value": str(signal.take_profit_2), "inline": True},
                    {"name": "Lot Size", "value": f"{signal.recommended_lot_size} Lots", "inline": True},
                    {"name": "Risk USD", "value": f"${signal.dollar_risk:.2f}", "inline": True},
                    {"name": "Confluences", "value": "\n".join([f"• {c}" for c in signal.confluences]), "inline": False}
                ],
                "footer": {"text": f"Bar Time: {signal.bar_time} UTC"}
            }]

        try:
            resp = requests.post(self.cfg.notifications.webhook_url, json=payload, timeout=8)
            if resp.status_code in (200, 204):
                logger.info(f"Webhook alert successfully dispatched for {signal.symbol}")
            else:
                logger.warning(f"Webhook endpoint returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to deliver Webhook alert: {e}")
