"""
Telegram Bot Integration for V7 Ultimate Signals

Sends real-time trading signals to Telegram with:
- Signal type (BUY/SELL/HOLD)
- Confidence level
- Mathematical theory analysis
- LLM reasoning
- Risk metrics

Usage:
    notifier = TelegramNotifier(token, chat_id)
    notifier.send_v7_signal(signal_result)
"""

import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from loguru import logger

# Try requests first (simpler, no async issues)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Fallback to python-telegram-bot
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

if not REQUESTS_AVAILABLE and not TELEGRAM_AVAILABLE:
    logger.warning("Neither requests nor python-telegram-bot installed. Telegram notifications disabled.")


class TelegramNotifier:
    """
    Telegram notification service for V7 Ultimate signals
    """

    def __init__(self, token: str, chat_id: str, enabled: bool = True):
        """
        Initialize Telegram notifier

        Args:
            token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
            enabled: Whether notifications are enabled
        """
        self.token = token
        self.chat_id = chat_id
        self.use_requests = REQUESTS_AVAILABLE  # Prefer requests (simpler, no async issues)
        self.enabled = enabled and (REQUESTS_AVAILABLE or TELEGRAM_AVAILABLE)

        if not REQUESTS_AVAILABLE and not TELEGRAM_AVAILABLE:
            logger.warning("Telegram library not available. Install with: pip install requests OR pip install python-telegram-bot")
            self.enabled = False
            return

        if not token or not chat_id:
            logger.warning("Telegram token or chat_id not provided. Notifications disabled.")
            self.enabled = False
            return

        if self.use_requests:
            # Using requests - no initialization needed
            logger.info(f"✅ Telegram notifier initialized (chat_id: {chat_id}, method: requests)")
        else:
            # Using python-telegram-bot
            try:
                self.bot = Bot(token=token)
                logger.info(f"✅ Telegram notifier initialized (chat_id: {chat_id}, method: python-telegram-bot)")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.enabled = False

    async def send_message_async(self, message: str) -> bool:
        """
        Send message to Telegram (async)

        Args:
            message: Message text (supports HTML formatting)

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            return True
        except TelegramError as e:
            logger.error(f"Telegram send error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_message_requests(self, message: str) -> bool:
        """
        Send message to Telegram using requests library (sync, simple)

        Args:
            message: Message text (supports HTML formatting)

        Returns:
            True if sent successfully
        """
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            return response.json().get('ok', False)
        except Exception as e:
            logger.error(f"Telegram send error (requests): {e}")
            return False

    def send_message(self, message: str) -> bool:
        """
        Send message to Telegram (sync wrapper)

        Args:
            message: Message text (supports HTML formatting)

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        # Prefer requests if available (simpler, no event loop issues)
        if self.use_requests:
            return self.send_message_requests(message)

        # Fallback to async method
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, we can't use run_until_complete
                # This shouldn't happen in our use case, but handle gracefully
                logger.warning("Already in event loop, cannot send synchronously")
                return False
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.send_message_async(message))
                    return result
                finally:
                    loop.close()
                    # Clean up to avoid issues with next call
                    asyncio.set_event_loop(None)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def format_v7_signal(self, symbol: str, signal_result) -> str:
        """
        Format V7 signal result for Telegram

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            signal_result: SignalGenerationResult from V7

        Returns:
            Formatted HTML message for Telegram
        """
        sig = signal_result.parsed_signal
        theories = signal_result.theory_analysis

        # Determine emoji based on signal
        if sig.signal.value == "BUY":
            emoji = "🟢"
            action = "BUY"
        elif sig.signal.value == "SELL":
            emoji = "🔴"
            action = "SELL"
        else:
            emoji = "🟡"
            action = "HOLD"

        # Confidence bar
        conf_pct = int(sig.confidence * 100)
        conf_bars = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)

        # Format timestamp
        ts = sig.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build message
        lines = [
            f"{emoji} <b>V7 ULTIMATE SIGNAL</b> {emoji}",
            "",
            f"<b>Symbol:</b> {symbol}",
            f"<b>Signal:</b> {action}",
            f"<b>Confidence:</b> {conf_pct}% {conf_bars}",
            f"<b>Time:</b> {ts}",
            "",
            "📊 <b>MATHEMATICAL ANALYSIS</b>",
        ]

        # Add theory analysis
        if theories.entropy is not None:
            entropy_interp = "Low" if theories.entropy < 0.4 else ("High" if theories.entropy > 0.7 else "Medium")
            lines.append(f"• <b>Shannon Entropy:</b> {theories.entropy:.3f} ({entropy_interp} randomness)")

        if theories.hurst is not None:
            hurst_interp = "Trending" if theories.hurst > 0.5 else "Mean-reverting"
            lines.append(f"• <b>Hurst Exponent:</b> {theories.hurst:.3f} ({hurst_interp})")

        if theories.current_regime:
            # Get highest probability regime for confidence
            regime_conf = max(theories.regime_probabilities.values()) if theories.regime_probabilities else 0
            lines.append(f"• <b>Market Regime:</b> {theories.current_regime} ({regime_conf*100:.0f}% conf)")

        sharpe = theories.risk_metrics.get('sharpe_ratio')
        if sharpe is not None:
            lines.append(f"• <b>Sharpe Ratio:</b> {sharpe:.2f}")

        var_95 = theories.risk_metrics.get('var_95')
        if var_95 is not None:
            lines.append(f"• <b>VaR (95%):</b> {var_95*100:.1f}%")

        profit_prob = theories.risk_metrics.get('profit_probability')
        if profit_prob is not None:
            lines.append(f"• <b>Profit Probability:</b> {profit_prob*100:.0f}%")

        # Add LLM reasoning
        lines.extend([
            "",
            "🤖 <b>LLM REASONING</b>",
            f"<i>{sig.reasoning}</i>",
        ])

        # Add cost info
        lines.extend([
            "",
            f"💰 <b>Cost:</b> ${signal_result.total_cost_usd:.6f}",
        ])

        # Add validity status
        if not sig.is_valid:
            lines.extend([
                "",
                "⚠️ <b>BLOCKED</b> - Signal did not pass FTMO rules",
            ])

        return "\n".join(lines)

    def send_v7_signal(self, symbol: str, signal_result) -> bool:
        """
        Send V7 signal to Telegram

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            signal_result: SignalGenerationResult from V7

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug("Telegram notifications disabled, skipping")
            return False

        try:
            message = self.format_v7_signal(symbol, signal_result)
            success = self.send_message(message)

            if success:
                logger.info(f"✅ V7 signal sent to Telegram: {symbol} {signal_result.parsed_signal.signal.value}")
            else:
                logger.warning(f"⚠️  Failed to send V7 signal to Telegram")

            return success

        except Exception as e:
            logger.error(f"Error sending V7 signal to Telegram: {e}")
            return False

    def send_test_message(self) -> bool:
        """
        Send test message to verify Telegram configuration

        Returns:
            True if sent successfully
        """
        message = """
🔬 <b>V7 ULTIMATE - TEST MESSAGE</b>

✅ Telegram notifications are working!

You will receive real-time V7 signals here with:
• Signal type (BUY/SELL/HOLD)
• Confidence levels
• Mathematical theory analysis
• LLM reasoning
• Risk metrics

Bot is ready! 🚀
        """.strip()

        return self.send_message(message)

    def send_runtime_status(self, status: str, details: str = "") -> bool:
        """
        Send V7 runtime status update

        Args:
            status: Status message (e.g., "Started", "Stopped", "Error")
            details: Additional details

        Returns:
            True if sent successfully
        """
        emoji_map = {
            "started": "🟢",
            "stopped": "🔴",
            "error": "⚠️",
            "warning": "🟡"
        }

        emoji = emoji_map.get(status.lower(), "ℹ️")

        message = f"{emoji} <b>V7 Runtime:</b> {status}"
        if details:
            message += f"\n\n{details}"

        return self.send_message(message)
