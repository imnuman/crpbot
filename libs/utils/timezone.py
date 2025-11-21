"""Timezone utilities for consistent EST timestamps across the application."""
from datetime import datetime
from zoneinfo import ZoneInfo

# EST Timezone (US Eastern Time)
EST = ZoneInfo("America/New_York")


def now_est() -> datetime:
    """Get current time in EST timezone.

    Returns:
        datetime: Current datetime in EST timezone
    """
    return datetime.now(EST)


def utc_to_est(dt: datetime) -> datetime:
    """Convert UTC datetime to EST.

    Args:
        dt: Datetime in UTC

    Returns:
        datetime: Datetime converted to EST
    """
    if dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(EST)


def format_timestamp_est(dt: datetime) -> str:
    """Format datetime as EST timestamp string.

    Args:
        dt: Datetime object (any timezone)

    Returns:
        str: Formatted timestamp in EST (e.g., "2025-11-20 18:30:45")
    """
    dt_est = utc_to_est(dt)
    return dt_est.strftime("%Y-%m-%d %H:%M:%S")
