"""
Email Integration for HYDRA Critical Alerts

Sends email alerts as backup notification channel.
Uses SMTP (supports Gmail, SendGrid, AWS SES, etc.)

Usage:
    notifier = EmailNotifier(smtp_host, smtp_port, username, password, from_email, to_email)
    notifier.send_alert("Kill switch activated!", "CRITICAL")
"""

import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class EmailConfig:
    """Configuration for Email notifications."""
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_email: str
    to_emails: List[str]
    use_tls: bool = True
    enabled: bool = True


class EmailNotifier:
    """
    Email notification service for HYDRA alerts.

    Supports SMTP servers including:
    - Gmail (smtp.gmail.com:587)
    - SendGrid (smtp.sendgrid.net:587)
    - AWS SES (email-smtp.us-east-1.amazonaws.com:587)
    """

    # Rate limiting - max 1 email per 5 minutes for same alert type
    ALERT_COOLDOWN_SECONDS = 300

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[List[str]] = None,
        use_tls: bool = True,
        enabled: bool = True
    ):
        """
        Initialize Email notifier.

        Args:
            smtp_host: SMTP server host (or SMTP_HOST env var)
            smtp_port: SMTP server port (or SMTP_PORT env var, default 587)
            username: SMTP username (or SMTP_USERNAME env var)
            password: SMTP password (or SMTP_PASSWORD env var)
            from_email: From email address (or SMTP_FROM_EMAIL env var)
            to_emails: List of recipient emails (or SMTP_TO_EMAILS env var, comma-separated)
            use_tls: Use TLS encryption
            enabled: Whether email notifications are enabled
        """
        # Get config from params or environment
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.username = username or os.getenv("SMTP_USERNAME", "")
        self.password = password or os.getenv("SMTP_PASSWORD", "")
        self.from_email = from_email or os.getenv("SMTP_FROM_EMAIL", "")

        # Parse to_emails (can be comma-separated string from env)
        if to_emails:
            self.to_emails = to_emails
        else:
            to_env = os.getenv("SMTP_TO_EMAILS", "")
            self.to_emails = [e.strip() for e in to_env.split(",") if e.strip()]

        self.use_tls = use_tls
        self.enabled = enabled

        # Rate limiting
        self._last_alert_times: dict[str, datetime] = {}

        # Validate configuration
        if not all([self.smtp_host, self.username, self.password, self.from_email, self.to_emails]):
            logger.warning(
                "Email config incomplete. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, "
                "SMTP_FROM_EMAIL, SMTP_TO_EMAILS env vars."
            )
            self.enabled = False
            return

        logger.info(f"Email notifier initialized (to: {len(self.to_emails)} recipients)")

    def _can_send_alert(self, alert_type: str) -> bool:
        """Check if we can send this alert (rate limiting)."""
        now = datetime.now(timezone.utc)
        last_sent = self._last_alert_times.get(alert_type)

        if last_sent:
            elapsed = (now - last_sent).total_seconds()
            if elapsed < self.ALERT_COOLDOWN_SECONDS:
                logger.debug(
                    f"Email rate limited for {alert_type}: "
                    f"{self.ALERT_COOLDOWN_SECONDS - elapsed:.0f}s remaining"
                )
                return False

        return True

    def _mark_alert_sent(self, alert_type: str):
        """Mark alert as sent for rate limiting."""
        self._last_alert_times[alert_type] = datetime.now(timezone.utc)

    def send_email(
        self,
        subject: str,
        body: str,
        alert_type: str = "general",
        is_html: bool = False
    ) -> bool:
        """
        Send email.

        Args:
            subject: Email subject
            body: Email body text
            alert_type: Type of alert for rate limiting
            is_html: Whether body is HTML formatted

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug("Email notifications disabled")
            return False

        if not self._can_send_alert(alert_type):
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)

            # Attach body
            content_type = "html" if is_html else "plain"
            msg.attach(MIMEText(body, content_type))

            # Send via SMTP
            context = ssl.create_default_context()

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())

            self._mark_alert_sent(alert_type)
            logger.info(f"Email sent: {subject} ({alert_type})")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Email send error: {e}")
            return False

    def send_alert(self, message: str, severity: str = "INFO", alert_type: str = "alert") -> bool:
        """
        Send alert email.

        Args:
            message: Alert message
            severity: Alert severity (INFO, WARNING, CRITICAL)
            alert_type: Type for rate limiting

        Returns:
            True if sent successfully
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        severity_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "CRITICAL": "🚨"
        }.get(severity.upper(), "ℹ️")

        subject = f"[HYDRA {severity}] {message[:50]}..."
        body = f"""
HYDRA Trading System Alert
===========================

Severity: {severity_emoji} {severity}
Time: {timestamp}

Message:
{message}

---
This is an automated alert from HYDRA Trading System.
Do not reply to this email.
        """.strip()

        return self.send_email(subject, body, alert_type)

    def send_critical_alert(self, message: str, alert_type: str = "critical") -> bool:
        """Send critical alert email."""
        return self.send_alert(message, "CRITICAL", alert_type)

    def send_daily_summary(self, summary: dict) -> bool:
        """
        Send daily performance summary email.

        Args:
            summary: Performance summary dictionary
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        subject = f"[HYDRA] Daily Summary - {timestamp}"

        # Build HTML body
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: #333;">HYDRA Daily Summary</h2>
    <p><strong>Date:</strong> {timestamp}</p>

    <h3 style="color: #666;">Performance</h3>
    <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Total Trades:</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{summary.get('total_trades', 0)}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Win Rate:</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{summary.get('win_rate', 0)*100:.1f}%</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>P&L:</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd; color: {'green' if summary.get('pnl', 0) >= 0 else 'red'};">
                ${summary.get('pnl', 0):+.2f}
            </td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Total Balance:</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">${summary.get('balance', 0):,.2f}</td>
        </tr>
    </table>

    <h3 style="color: #666;">Engine Performance</h3>
    <table style="border-collapse: collapse; width: 100%;">
        <tr style="background: #f5f5f5;">
            <th style="padding: 8px; border: 1px solid #ddd;">Engine</th>
            <th style="padding: 8px; border: 1px solid #ddd;">Balance</th>
            <th style="padding: 8px; border: 1px solid #ddd;">Trades</th>
            <th style="padding: 8px; border: 1px solid #ddd;">P&L</th>
        </tr>
"""
        for engine, data in summary.get('engines', {}).items():
            pnl_color = 'green' if data.get('pnl', 0) >= 0 else 'red'
            html_body += f"""
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;">{engine}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">${data.get('balance', 0):,.2f}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{data.get('trades', 0)}</td>
            <td style="padding: 8px; border: 1px solid #ddd; color: {pnl_color};">
                ${data.get('pnl', 0):+.2f}
            </td>
        </tr>
"""

        html_body += """
    </table>

    <p style="color: #999; font-size: 12px; margin-top: 20px;">
        This is an automated summary from HYDRA Trading System.
    </p>
</body>
</html>
        """

        return self.send_email(subject, html_body, "daily_summary", is_html=True)

    def send_test_email(self) -> bool:
        """Send test email to verify configuration."""
        return self.send_alert(
            "HYDRA Email Test: Notifications working!",
            "INFO",
            "test"
        )


# Global singleton
_email_notifier: Optional[EmailNotifier] = None


def get_email_notifier() -> EmailNotifier:
    """Get or create global email notifier singleton."""
    global _email_notifier
    if _email_notifier is None:
        _email_notifier = EmailNotifier()
    return _email_notifier
