"""
HYDRA Unified Alert Manager

Coordinates alerts across multiple channels:
- Telegram (primary, all signals)
- SMS (critical alerts only)
- Email (backup, summaries)

Alert Routing:
- INFO: Telegram only
- WARNING: Telegram + Email
- CRITICAL: Telegram + SMS + Email

Usage:
    alert_manager = get_alert_manager()
    alert_manager.send_critical("Kill switch activated!")
    alert_manager.send_trade_signal(signal_data)
"""

import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from loguru import logger

from .telegram_bot import TelegramNotifier
from .twilio_sms import TwilioSMSNotifier, get_sms_notifier
from .email_notifier import EmailNotifier, get_email_notifier


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertResult:
    """Result of sending an alert."""
    telegram_sent: bool = False
    sms_sent: bool = False
    email_sent: bool = False

    @property
    def any_sent(self) -> bool:
        return self.telegram_sent or self.sms_sent or self.email_sent

    @property
    def all_sent(self) -> bool:
        return self.telegram_sent and self.sms_sent and self.email_sent


class AlertManager:
    """
    Unified alert manager for HYDRA.

    Routes alerts to appropriate channels based on severity.
    Handles failures gracefully (if one channel fails, others still try).
    """

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        sms_notifier: Optional[TwilioSMSNotifier] = None,
        email_notifier: Optional[EmailNotifier] = None
    ):
        """
        Initialize AlertManager with all notification channels.

        Args:
            telegram_token: Telegram bot token (or TELEGRAM_TOKEN env)
            telegram_chat_id: Telegram chat ID (or TELEGRAM_CHAT_ID env)
            sms_notifier: Optional TwilioSMSNotifier instance
            email_notifier: Optional EmailNotifier instance
        """
        # Initialize Telegram
        token = telegram_token or os.getenv("TELEGRAM_TOKEN", "")
        chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.telegram = TelegramNotifier(token, chat_id) if token and chat_id else None

        # Initialize SMS (uses singleton if not provided)
        self.sms = sms_notifier or get_sms_notifier()

        # Initialize Email (uses singleton if not provided)
        self.email = email_notifier or get_email_notifier()

        # Track channels status
        self._log_channel_status()

    def _log_channel_status(self):
        """Log status of all notification channels."""
        channels = []
        if self.telegram and self.telegram.enabled:
            channels.append("Telegram")
        if self.sms and self.sms.enabled:
            channels.append("SMS")
        if self.email and self.email.enabled:
            channels.append("Email")

        if channels:
            logger.info(f"AlertManager initialized: {', '.join(channels)}")
        else:
            logger.warning("AlertManager: No notification channels configured!")

    def _get_channels_for_severity(self, severity: AlertSeverity) -> Dict[str, bool]:
        """
        Determine which channels to use based on severity.

        Returns:
            Dict of channel -> should_use
        """
        if severity == AlertSeverity.INFO:
            return {"telegram": True, "sms": False, "email": False}
        elif severity == AlertSeverity.WARNING:
            return {"telegram": True, "sms": False, "email": True}
        else:  # CRITICAL
            return {"telegram": True, "sms": True, "email": True}

    def send_alert(
        self,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        alert_type: str = "general"
    ) -> AlertResult:
        """
        Send alert to appropriate channels based on severity.

        Args:
            message: Alert message
            severity: Alert severity (determines routing)
            alert_type: Type for rate limiting

        Returns:
            AlertResult with status for each channel
        """
        result = AlertResult()
        channels = self._get_channels_for_severity(severity)

        # Telegram
        if channels["telegram"] and self.telegram and self.telegram.enabled:
            try:
                emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[severity.value]
                formatted = f"{emoji} <b>HYDRA {severity.value.upper()}</b>\n\n{message}"
                result.telegram_sent = self.telegram.send_message(formatted)
            except Exception as e:
                logger.error(f"Telegram alert failed: {e}")

        # SMS (critical only)
        if channels["sms"] and self.sms and self.sms.enabled:
            try:
                result.sms_sent = self.sms.send_critical_alert(message, alert_type)
            except Exception as e:
                logger.error(f"SMS alert failed: {e}")

        # Email
        if channels["email"] and self.email and self.email.enabled:
            try:
                result.email_sent = self.email.send_alert(message, severity.value.upper(), alert_type)
            except Exception as e:
                logger.error(f"Email alert failed: {e}")

        return result

    def send_info(self, message: str, alert_type: str = "info") -> AlertResult:
        """Send info level alert (Telegram only)."""
        return self.send_alert(message, AlertSeverity.INFO, alert_type)

    def send_warning(self, message: str, alert_type: str = "warning") -> AlertResult:
        """Send warning level alert (Telegram + Email)."""
        return self.send_alert(message, AlertSeverity.WARNING, alert_type)

    def send_critical(self, message: str, alert_type: str = "critical") -> AlertResult:
        """Send critical level alert (Telegram + SMS + Email)."""
        return self.send_alert(message, AlertSeverity.CRITICAL, alert_type)

    # ==================== Specific Alert Types ====================

    def send_kill_switch_alert(self, reason: str) -> AlertResult:
        """Send kill switch activation alert."""
        message = f"KILL SWITCH ACTIVATED!\n\nReason: {reason}\n\nAll trading has been halted. Manual intervention required."
        return self.send_critical(message, "kill_switch")

    def send_loss_limit_warning(
        self,
        loss_type: str,
        current_pct: float,
        limit_pct: float
    ) -> AlertResult:
        """Send loss limit warning."""
        severity = AlertSeverity.CRITICAL if current_pct >= limit_pct * 0.9 else AlertSeverity.WARNING

        message = (
            f"{loss_type.upper()} LOSS WARNING\n\n"
            f"Current: {current_pct*100:.2f}%\n"
            f"Limit: {limit_pct*100:.2f}%\n\n"
            f"{'REDUCE EXPOSURE IMMEDIATELY!' if severity == AlertSeverity.CRITICAL else 'Monitor closely.'}"
        )
        return self.send_alert(message, severity, f"{loss_type}_loss")

    def send_engine_blown_alert(self, engine_name: str, balance: float) -> AlertResult:
        """Send engine blow-up alert."""
        message = (
            f"ENGINE {engine_name} BLOWN!\n\n"
            f"Final Balance: ${balance:.2f}\n\n"
            f"Engine has been disabled. Review and restart manually."
        )
        return self.send_critical(message, f"engine_blown_{engine_name}")

    def send_connection_failure_alert(self, service: str, error: str) -> AlertResult:
        """Send connection failure alert."""
        message = f"CONNECTION FAILURE: {service}\n\nError: {error}\n\nAttempting reconnection..."
        return self.send_warning(message, f"connection_{service}")

    def send_trade_executed(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        engine: str
    ) -> AlertResult:
        """Send trade execution notification."""
        emoji = "🟢" if direction == "BUY" else "🔴"
        message = (
            f"{emoji} Trade Executed\n\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Entry: ${entry_price:,.2f}\n"
            f"SL: ${stop_loss:,.2f}\n"
            f"TP: ${take_profit:,.2f}\n"
            f"Engine: {engine}"
        )
        return self.send_info(message, "trade_executed")

    def send_trade_closed(
        self,
        symbol: str,
        direction: str,
        outcome: str,
        pnl_usd: float,
        pnl_pct: float,
        engine: str
    ) -> AlertResult:
        """Send trade closure notification."""
        emoji = "✅" if outcome == "win" else "❌"
        message = (
            f"{emoji} Trade Closed\n\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Outcome: {outcome.upper()}\n"
            f"P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)\n"
            f"Engine: {engine}"
        )
        return self.send_info(message, "trade_closed")

    def send_daily_summary(self, summary: Dict[str, Any]) -> AlertResult:
        """Send daily performance summary."""
        result = AlertResult()

        # Telegram summary
        if self.telegram and self.telegram.enabled:
            try:
                pnl = summary.get('pnl', 0)
                emoji = "📈" if pnl >= 0 else "📉"
                message = (
                    f"{emoji} <b>HYDRA Daily Summary</b>\n\n"
                    f"<b>Trades:</b> {summary.get('total_trades', 0)}\n"
                    f"<b>Win Rate:</b> {summary.get('win_rate', 0)*100:.1f}%\n"
                    f"<b>P&L:</b> ${pnl:+.2f}\n"
                    f"<b>Balance:</b> ${summary.get('balance', 0):,.2f}"
                )
                result.telegram_sent = self.telegram.send_message(message)
            except Exception as e:
                logger.error(f"Telegram summary failed: {e}")

        # Email summary (detailed)
        if self.email and self.email.enabled:
            try:
                result.email_sent = self.email.send_daily_summary(summary)
            except Exception as e:
                logger.error(f"Email summary failed: {e}")

        return result

    def test_all_channels(self) -> AlertResult:
        """Test all notification channels."""
        result = AlertResult()

        # Test Telegram
        if self.telegram and self.telegram.enabled:
            try:
                result.telegram_sent = self.telegram.send_test_message()
                logger.info(f"Telegram test: {'OK' if result.telegram_sent else 'FAILED'}")
            except Exception as e:
                logger.error(f"Telegram test error: {e}")

        # Test SMS
        if self.sms and self.sms.enabled:
            try:
                result.sms_sent = self.sms.send_test_sms()
                logger.info(f"SMS test: {'OK' if result.sms_sent else 'FAILED'}")
            except Exception as e:
                logger.error(f"SMS test error: {e}")

        # Test Email
        if self.email and self.email.enabled:
            try:
                result.email_sent = self.email.send_test_email()
                logger.info(f"Email test: {'OK' if result.email_sent else 'FAILED'}")
            except Exception as e:
                logger.error(f"Email test error: {e}")

        return result


# Global singleton
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create global AlertManager singleton."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
