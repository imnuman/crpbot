"""Rate limiting for signal generation."""
from collections import deque
from datetime import datetime, timedelta

from loguru import logger


class RateLimiter:
    """Rate limiter for controlling signal frequency."""

    def __init__(
        self,
        max_signals_per_hour: int = 10,
        max_high_tier_per_hour: int = 5,
    ):
        """
        Initialize rate limiter.

        Args:
            max_signals_per_hour: Maximum total signals per hour
            max_high_tier_per_hour: Maximum high-tier signals per hour
        """
        self.max_signals_per_hour = max_signals_per_hour
        self.max_high_tier_per_hour = max_high_tier_per_hour

        # Track signal timestamps (last hour)
        self.all_signals: deque = deque()
        self.high_tier_signals: deque = deque()

    def _cleanup_old_signals(self) -> None:
        """Remove signals older than 1 hour."""
        cutoff = datetime.now() - timedelta(hours=1)

        while self.all_signals and self.all_signals[0] < cutoff:
            self.all_signals.popleft()

        while self.high_tier_signals and self.high_tier_signals[0] < cutoff:
            self.high_tier_signals.popleft()

    def can_emit_signal(self, tier: str) -> bool:
        """
        Check if a signal can be emitted.

        Args:
            tier: Signal tier ('high', 'medium', 'low')

        Returns:
            True if signal can be emitted, False if rate limit exceeded
        """
        self._cleanup_old_signals()

        # Check total signal limit
        if len(self.all_signals) >= self.max_signals_per_hour:
            logger.warning(
                f"❌ Total signal rate limit reached: "
                f"{len(self.all_signals)}/{self.max_signals_per_hour} signals in last hour"
            )
            return False

        # Check high-tier signal limit
        if tier == "high" and len(self.high_tier_signals) >= self.max_high_tier_per_hour:
            logger.warning(
                f"❌ High-tier signal rate limit reached: "
                f"{len(self.high_tier_signals)}/{self.max_high_tier_per_hour} signals in last hour"
            )
            return False

        return True

    def record_signal(self, tier: str) -> None:
        """
        Record that a signal was emitted.

        Args:
            tier: Signal tier ('high', 'medium', 'low')
        """
        now = datetime.now()
        self.all_signals.append(now)

        if tier == "high":
            self.high_tier_signals.append(now)

        logger.info(
            f"📊 Signals in last hour: {len(self.all_signals)} total, "
            f"{len(self.high_tier_signals)} high-tier"
        )

    def get_stats(self) -> dict[str, int]:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with signal counts
        """
        self._cleanup_old_signals()
        return {
            "total_signals_last_hour": len(self.all_signals),
            "high_tier_signals_last_hour": len(self.high_tier_signals),
            "remaining_total": self.max_signals_per_hour - len(self.all_signals),
            "remaining_high_tier": self.max_high_tier_per_hour - len(self.high_tier_signals),
        }
