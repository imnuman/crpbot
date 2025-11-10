"""Unit tests for rate limiter."""

from datetime import datetime, timedelta, timezone

import pytest

from apps.runtime.rate_limiter import RateLimiter


def test_rate_limiter_blocks_after_limit():
    """Rate limiter should block when hourly limit is reached."""
    limiter = RateLimiter(max_signals_per_hour=2, max_signals_per_hour_high=1)

    can_emit, _ = limiter.can_emit("medium")
    assert can_emit is True

    limiter.record_signal("medium")
    limiter.record_signal("medium")

    can_emit, reason = limiter.can_emit("medium")
    assert can_emit is False
    assert "Rate limit" in reason


def test_high_tier_limit():
    """High tier signals have a tighter limit."""
    limiter = RateLimiter(max_signals_per_hour=5, max_signals_per_hour_high=1)

    limiter.record_signal("high")
    can_emit, reason = limiter.can_emit("high")

    assert can_emit is False
    assert "High tier rate limit" in reason


def test_backoff_triggers_after_losses():
    """Two losses within the window should trigger backoff."""
    limiter = RateLimiter(backoff_losses=2, backoff_window_minutes=60, backoff_risk_reduction=0.4)

    limiter.record_loss()
    limiter.record_loss()

    assert limiter.backoff_active is True
    assert limiter.get_backoff_multiplier() == pytest.approx(0.4)

    can_emit, reason = limiter.can_emit("high")
    assert can_emit is False
    assert "Backoff active" in reason


def test_backoff_expires_after_window(monkeypatch):
    """Backoff should lift once losses fall out of the monitoring window."""
    limiter = RateLimiter(backoff_losses=1, backoff_window_minutes=1)

    limiter.record_loss()
    assert limiter.backoff_active is True

    # Simulate time passing by placing loss timestamps well in the past
    past_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    limiter.loss_history = [past_time for _ in limiter.loss_history]
    limiter.backoff_until = past_time

    can_emit, _ = limiter.can_emit("medium")
    assert can_emit is True
    assert limiter.backoff_active is False
    assert limiter.get_backoff_multiplier() == pytest.approx(1.0)
