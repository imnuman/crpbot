"""
Twilio SMS Integration for HYDRA Critical Alerts

Sends critical SMS alerts for:
- Kill switch activation
- Daily/total loss limit approaching
- Connection failures
- Trade execution errors
- Engine blow-ups (balance < floor)

Usage:
    notifier = TwilioSMSNotifier(account_sid, auth_token, from_number, to_number)
    notifier.send_critical_alert("Kill switch activated!")
"""

import os
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass
from loguru import logger

# Try to import Twilio
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not installed. SMS notifications disabled. Install with: pip install twilio")


@dataclass
class SMSConfig:
    """Configuration for Twilio SMS."""
    account_sid: str
    auth_token: str
    from_number: str  # Your Twilio phone number
    to_number: str    # Phone to receive alerts
    enabled: bool = True


class TwilioSMSNotifier:
    """
    Twilio SMS notification service for critical HYDRA alerts.

    Only for CRITICAL alerts - not regular signals (use Telegram for those).
    SMS is reserved for:
    - Kill switch activation
    - Loss limit breaches
    - System failures requiring immediate attention
    """

    # Rate limiting - max 1 SMS per minute for same alert type
    MAX_SMS_PER_MINUTE = 1
    ALERT_COOLDOWN_SECONDS = 60

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Twilio SMS notifier.

        Args:
            account_sid: Twilio account SID (or TWILIO_ACCOUNT_SID env var)
            auth_token: Twilio auth token (or TWILIO_AUTH_TOKEN env var)
            from_number: Twilio phone number (or TWILIO_FROM_NUMBER env var)
            to_number: Phone to receive alerts (or TWILIO_TO_NUMBER env var)
            enabled: Whether SMS notifications are enabled
        """
        # Get credentials from params or environment
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER", "")
        self.to_number = to_number or os.getenv("TWILIO_TO_NUMBER", "")

        self.enabled = enabled and TWILIO_AVAILABLE
        self.client: Optional[Client] = None

        # Rate limiting tracking
        self._last_alert_times: dict[str, datetime] = {}

        # Validate and initialize
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio library not available. SMS notifications disabled.")
            self.enabled = False
            return

        if not all([self.account_sid, self.auth_token, self.from_number, self.to_number]):
            logger.warning(
                "Twilio credentials incomplete. Set TWILIO_ACCOUNT_SID, "
                "TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER env vars."
            )
            self.enabled = False
            return

        try:
            self.client = Client(self.account_sid, self.auth_token)
            logger.info(f"Twilio SMS notifier initialized (to: {self._mask_phone(self.to_number)})")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
            self.enabled = False

    def _mask_phone(self, phone: str) -> str:
        """Mask phone number for logging."""
        if len(phone) < 6:
            return "***"
        return f"{phone[:3]}***{phone[-2:]}"

    def _can_send_alert(self, alert_type: str) -> bool:
        """Check if we can send this alert (rate limiting)."""
        now = datetime.now(timezone.utc)
        last_sent = self._last_alert_times.get(alert_type)

        if last_sent:
            elapsed = (now - last_sent).total_seconds()
            if elapsed < self.ALERT_COOLDOWN_SECONDS:
                logger.debug(
                    f"SMS rate limited for {alert_type}: "
                    f"{self.ALERT_COOLDOWN_SECONDS - elapsed:.0f}s remaining"
                )
                return False

        return True

    def _mark_alert_sent(self, alert_type: str):
        """Mark alert as sent for rate limiting."""
        self._last_alert_times[alert_type] = datetime.now(timezone.utc)

    def send_sms(self, message: str, alert_type: str = "general") -> bool:
        """
        Send SMS message.

        Args:
            message: SMS text (max 160 chars recommended)
            alert_type: Type of alert for rate limiting

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug("SMS notifications disabled")
            return False

        if not self._can_send_alert(alert_type):
            return False

        try:
            # Twilio SMS has 1600 char limit, but 160 is standard
            if len(message) > 1600:
                message = message[:1597] + "..."

            result = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )

            self._mark_alert_sent(alert_type)
            logger.info(f"SMS sent: {result.sid} ({alert_type})")
            return True

        except TwilioRestException as e:
            logger.error(f"Twilio API error: {e.code} - {e.msg}")
            return False
        except Exception as e:
            logger.error(f"SMS send error: {e}")
            return False

    def send_critical_alert(self, message: str, alert_type: str = "critical") -> bool:
        """
        Send critical alert SMS.

        Args:
            message: Alert message
            alert_type: Type for rate limiting

        Returns:
            True if sent successfully
        """
        timestamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
        full_message = f"HYDRA ALERT [{timestamp}]: {message}"
        return self.send_sms(full_message, alert_type)

    def send_kill_switch_alert(self, reason: str) -> bool:
        """Send kill switch activation alert."""
        return self.send_critical_alert(
            f"KILL SWITCH ACTIVATED! {reason}. Trading halted.",
            alert_type="kill_switch"
        )

    def send_loss_limit_warning(self, loss_type: str, current_pct: float, limit_pct: float) -> bool:
        """
        Send loss limit warning.

        Args:
            loss_type: "daily" or "total"
            current_pct: Current loss percentage (e.g., 0.035 for 3.5%)
            limit_pct: Limit percentage (e.g., 0.045 for 4.5%)
        """
        return self.send_critical_alert(
            f"{loss_type.upper()} LOSS WARNING: {current_pct*100:.1f}% "
            f"(limit: {limit_pct*100:.1f}%). Reduce exposure!",
            alert_type=f"{loss_type}_loss_warning"
        )

    def send_connection_failure(self, service: str, error: str) -> bool:
        """Send connection failure alert."""
        return self.send_critical_alert(
            f"{service} connection failed: {error[:50]}. Check system.",
            alert_type=f"connection_{service}"
        )

    def send_engine_blown(self, engine_name: str, balance: float) -> bool:
        """Send engine blow-up alert."""
        return self.send_critical_alert(
            f"Engine {engine_name} BLOWN! Balance: ${balance:.2f}. Engine disabled.",
            alert_type=f"engine_blown_{engine_name}"
        )

    def send_trade_error(self, error: str) -> bool:
        """Send trade execution error alert."""
        return self.send_critical_alert(
            f"Trade error: {error[:80]}. Manual review needed.",
            alert_type="trade_error"
        )

    def send_test_sms(self) -> bool:
        """Send test SMS to verify configuration."""
        return self.send_sms(
            "HYDRA SMS Test: Notifications working!",
            alert_type="test"
        )


# Global singleton
_sms_notifier: Optional[TwilioSMSNotifier] = None


def get_sms_notifier() -> TwilioSMSNotifier:
    """Get or create global SMS notifier singleton."""
    global _sms_notifier
    if _sms_notifier is None:
        _sms_notifier = TwilioSMSNotifier()
    return _sms_notifier
