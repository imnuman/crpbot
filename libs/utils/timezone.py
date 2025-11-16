"""Timezone utilities for consistent EST timestamps across the application."""
from datetime import datetime
from zoneinfo import ZoneInfo

# EST Timezone
EST = ZoneInfo("America/Toronto")


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
