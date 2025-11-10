"""Rate limiting for signal emission."""
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger


class RateLimiter:
    """Rate limiter for signals per hour per tier."""

    def __init__(
        self,
        max_signals_per_hour: int = 10,
        max_signals_per_hour_high: int | None = 5,
        max_high_tier_per_hour: int | None = None,
        backoff_losses: int = 2,
        backoff_window_minutes: int = 60,
        backoff_risk_reduction: float = 0.5,
    ):
        """
        Initialize rate limiter.

        Args:
            max_signals_per_hour: Max total signals per hour
            max_signals_per_hour_high: Max HIGH tier signals per hour
            backoff_losses: Number of losses to trigger backoff
            backoff_window_minutes: Time window for backoff trigger
            backoff_risk_reduction: Risk reduction multiplier during backoff
        """
        self.max_signals_per_hour = max_signals_per_hour

        if max_signals_per_hour_high is None and max_high_tier_per_hour is not None:
            max_signals_per_hour_high = max_high_tier_per_hour
        if max_signals_per_hour_high is None:
            max_signals_per_hour_high = 5

        self.max_signals_per_hour_high = max_signals_per_hour_high
        self.backoff_losses = backoff_losses
        self.backoff_window_minutes = backoff_window_minutes
        self.backoff_risk_reduction = backoff_risk_reduction

        # Track signals by hour (UTC-aware timestamps)
        self.signal_history: list[datetime] = []
        self.signal_history_high: list[datetime] = []

        # Track losses for backoff (UTC-aware timestamps)
        self.loss_history: list[datetime] = []
        self.backoff_active = False
        self.backoff_until: datetime | None = None

        logger.info(
            f"Rate limiter initialized: max={max_signals_per_hour}/hour, "
            f"max_high={max_signals_per_hour_high}/hour"
        )

    @staticmethod
    def _to_utc(value: datetime) -> datetime:
        """Return a UTC-aware datetime."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _clean_old_signals(self, history: list[datetime], window_hours: int = 1) -> None:
        """Remove signals older than window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        cleaned: list[datetime] = []
        for stamp in history:
            stamp_utc = self._to_utc(stamp)
            if stamp_utc > cutoff:
                cleaned.append(stamp_utc)
        history[:] = cleaned

    def _clean_old_losses(self) -> None:
        """Remove losses older than backoff window."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.backoff_window_minutes)
        cleaned: list[datetime] = []
        for loss in self.loss_history:
            loss_utc = self._to_utc(loss)
            if loss_utc > cutoff:
                cleaned.append(loss_utc)
        self.loss_history[:] = cleaned

    def can_emit(self, tier: str) -> tuple[bool, str]:
        """
        Check if signal can be emitted.

        Args:
            tier: Signal tier ('high', 'medium', 'low')

        Returns:
            Tuple of (can_emit, reason)
        """
        now = datetime.now(timezone.utc)

        # Check backoff status
        if self.backoff_active and self.backoff_until:
            backoff_until = self._to_utc(self.backoff_until)
            if now >= backoff_until:
                logger.info("Backoff window elapsed; resuming normal operation")
                self.backoff_active = False
                self.backoff_until = None
                self._clean_old_losses()
            else:
                # Check if backoff period should remain active
                self._clean_old_losses()
                if len(self.loss_history) < self.backoff_losses:
                    self.backoff_active = False
                    self.backoff_until = None
                    logger.info("Backoff period ended")
                else:
                    remaining = (backoff_until - now).total_seconds() / 60
                    return False, f"Backoff active (remaining: {remaining:.1f} minutes)"

        # Clean old signals
        self._clean_old_signals(self.signal_history)
        self._clean_old_signals(self.signal_history_high)

        # Check total rate limit
        if len(self.signal_history) >= self.max_signals_per_hour:
            return False, f"Rate limit exceeded: {len(self.signal_history)}/{self.max_signals_per_hour} signals/hour"

        # Check high tier rate limit
        if tier == "high" and len(self.signal_history_high) >= self.max_signals_per_hour_high:
            return (
                False,
                f"High tier rate limit exceeded: {len(self.signal_history_high)}/{self.max_signals_per_hour_high} signals/hour",
            )

        return True, ""



    def can_emit_signal(self, tier: str) -> bool:
        """Compatibility wrapper used by the runtime."""
        can_emit, reason = self.can_emit(tier)
        if not can_emit and reason:
            logger.warning(reason)
        return can_emit
    def record_signal(self, tier: str) -> None:
        """
        Record a signal emission.

        Args:
            tier: Signal tier
        """
        now = datetime.now(timezone.utc)
        self.signal_history.append(now)

        if tier == "high":
            self.signal_history_high.append(now)

        logger.debug(
            f"Recorded {tier} signal (total: {len(self.signal_history)}, high: {len(self.signal_history_high)})"
        )

    def record_loss(self) -> None:
        """Record a loss (for backoff logic)."""
        now = datetime.now(timezone.utc)
        self.loss_history.append(now)

        # Clean old losses
        self._clean_old_losses()

        # Check if backoff should be triggered
        if len(self.loss_history) >= self.backoff_losses and not self.backoff_active:
            self.backoff_active = True
            # Backoff for the remainder of the session (until end of backoff window)
            self.backoff_until = now + timedelta(minutes=self.backoff_window_minutes)
            logger.warning(
                f"Backoff triggered: {len(self.loss_history)} losses in {self.backoff_window_minutes} minutes. "
                f"Backoff until {self.backoff_until}"
            )

    def get_backoff_multiplier(self) -> float:
        """
        Get risk reduction multiplier during backoff.

        Returns:
            Multiplier (1.0 if no backoff, backoff_risk_reduction if backoff active)
        """
        if self.backoff_active:
            return self.backoff_risk_reduction
        return 1.0

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        self._clean_old_signals(self.signal_history)
        self._clean_old_signals(self.signal_history_high)
        self._clean_old_losses()

        backoff_until = (
            self._to_utc(self.backoff_until).isoformat() if self.backoff_until else None
        )

        return {
            "signals_last_hour": len(self.signal_history),
            "signals_high_last_hour": len(self.signal_history_high),
            "max_signals_per_hour": self.max_signals_per_hour,
            "max_signals_high_per_hour": self.max_signals_per_hour_high,
            "backoff_active": self.backoff_active,
            "backoff_until": backoff_until,
            "losses_in_window": len(self.loss_history),
        }
