"""
Notification modules for HYDRA trading system.

Provides multi-channel alerts:
- Telegram: All signals and notifications
- SMS (Twilio): Critical alerts only
- Email: Backup alerts and daily summaries

Usage:
    from libs.notifications import get_alert_manager
    alert_manager = get_alert_manager()
    alert_manager.send_critical("Kill switch activated!")
"""

from .telegram_bot import TelegramNotifier
from .twilio_sms import TwilioSMSNotifier, get_sms_notifier
from .email_notifier import EmailNotifier, get_email_notifier
from .alert_manager import AlertManager, AlertSeverity, get_alert_manager

__all__ = [
    'TelegramNotifier',
    'TwilioSMSNotifier',
    'get_sms_notifier',
    'EmailNotifier',
    'get_email_notifier',
    'AlertManager',
    'AlertSeverity',
    'get_alert_manager',
]
