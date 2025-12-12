"""
Time-based Trading Filter - Blocks trades during historically bad hours/days.

Based on 46-trade analysis (2025-12-11):
- 22:00 UTC: 0% WR, -$499
- 21:00 UTC: 33% WR, -$453
- 06:00 UTC: 8% WR, -$40
- Wednesday: 6% WR, -$950

Created: 2025-12-11
"""

import logging
from datetime import datetime, timezone
from typing import Tuple

logger = logging.getLogger(__name__)

# Hours to AVOID (UTC) - based on backtest data
# 21:00-23:00 = US market close volatility, 0-33% WR
# 06:00-09:00 = Low liquidity period, 8-17% WR
BAD_HOURS_UTC = {21, 22, 6, 9}

# Days to reduce size (not block entirely)
# Wednesday: 6% WR, -$950 loss
REDUCED_SIZE_DAYS = {"Wednesday"}
REDUCED_SIZE_MULTIPLIER = 0.25  # 75% size reduction on bad days

# Best hours (67%+ WR)
BEST_HOURS_UTC = {15}
BEST_HOURS_MULTIPLIER = 1.25  # 25% size increase on best hours


def check_trading_time() -> Tuple[bool, float, str]:
    """
    Check if current time is good for trading.

    Returns:
        (allowed, size_multiplier, reason)
        - allowed: True if trading is permitted
        - size_multiplier: Position size multiplier (0.25-1.25)
        - reason: Explanation string
    """
    now = datetime.now(timezone.utc)
    hour = now.hour
    day = now.strftime("%A")

    # Check if in bad hours
    if hour in BAD_HOURS_UTC:
        return False, 0.0, f"Bad hour {hour:02d}:00 UTC (historically 0-17% WR)"

    # Check if best hour
    if hour in BEST_HOURS_UTC:
        return True, BEST_HOURS_MULTIPLIER, f"Best hour {hour:02d}:00 UTC (67% WR)"

    # Check if bad day
    if day in REDUCED_SIZE_DAYS:
        return True, REDUCED_SIZE_MULTIPLIER, f"{day} (6% WR historically) - reduced size"

    # Normal trading
    return True, 1.0, "OK"


def should_trade_now() -> bool:
    """Simple check if trading is allowed now."""
    allowed, _, _ = check_trading_time()
    return allowed


def get_time_size_multiplier() -> float:
    """Get position size multiplier based on current time."""
    _, multiplier, _ = check_trading_time()
    return multiplier


def log_time_check():
    """Log current time filter status."""
    allowed, multiplier, reason = check_trading_time()
    now = datetime.now(timezone.utc)
    logger.info(
        f"[TimeFilter] {now.strftime('%A %H:%M')} UTC - "
        f"Trading={'ALLOWED' if allowed else 'BLOCKED'}, "
        f"Size={multiplier:.0%}, Reason={reason}"
    )
