"""Unit tests for FTMO rules enforcement."""

from apps.runtime.ftmo_rules import (
    FTMOState,
    calculate_position_size,
    check_daily_loss,
    check_ftmo_limits,
    check_total_loss,
)


def test_daily_loss_limit_exceeded():
    """Daily loss check should fail when loss exceeds 4.5%."""
    state = FTMOState(account_balance=10000.0)
    state.update_balance(9400.0)  # 6% drawdown

    assert check_daily_loss(state) is False
    ok, reason = check_ftmo_limits(state)
    assert ok is False
    assert "Daily loss" in reason


def test_total_loss_limit_exceeded():
    """Total loss check should fail when account draws down more than 9%."""
    state = FTMOState(account_balance=10000.0)
    state.daily_loss_limit = 0.20  # relax daily limit so total loss triggers first
    state.update_balance(8900.0)  # 11% drawdown

    assert check_total_loss(state) is False
    ok, reason = check_ftmo_limits(state)
    assert ok is False
    assert "Total loss" in reason


def test_position_size_calculation():
    """Position sizing should scale with risk settings."""
    size_default = calculate_position_size(
        account_balance=10000.0,
        risk_per_trade=0.01,
        entry_price=50000.0,
        sl_price=49500.0,
    )

    size_lower_risk = calculate_position_size(
        account_balance=10000.0,
        risk_per_trade=0.005,
        entry_price=50000.0,
        sl_price=49500.0,
    )

    assert size_default > 0
    assert size_lower_risk < size_default


def test_daily_reset_on_new_session():
    """Daily counters should reset when a new day begins."""
    state = FTMOState(account_balance=10000.0)
    state.update_balance(9500.0)

    # Simulate next day
    from datetime import timedelta

    state.daily_start_time = state.daily_start_time - timedelta(days=1)
    state.reset_daily()

    assert state.daily_loss == 0.0
    assert state.daily_start_balance == state.account_balance
