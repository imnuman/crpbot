"""FTMO rules enforcement library."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from loguru import logger


@dataclass
class FTMOState:
    """FTMO account state tracking."""

    account_balance: float = 10000.0
    daily_loss: float = 0.0
    total_loss: float = 0.0
    daily_start_balance: float = 10000.0
    daily_start_time: datetime = None
    daily_loss_limit: float = 0.045  # 4.5% of account
    total_loss_limit: float = 0.09  # 9% of account

    def __post_init__(self):
        """Initialize daily start time if not set."""
        if self.daily_start_time is None:
            self.daily_start_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    def reset_daily(self) -> None:
        """Reset daily loss tracking (call at start of new trading day)."""
        now = datetime.utcnow()
        if now.date() > self.daily_start_time.date():
            logger.info("Resetting daily loss tracking (new trading day)")
            self.daily_start_balance = self.account_balance
            self.daily_loss = 0.0
            self.daily_start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def update_balance(self, new_balance: float) -> None:
        """Update account balance and recalculate losses."""
        self.reset_daily()  # Check if new day

        old_balance = self.account_balance
        self.account_balance = new_balance

        # Calculate losses
        daily_change = self.account_balance - self.daily_start_balance
        total_change = self.account_balance - 10000.0  # Assuming 10K starting balance

        if daily_change < 0:
            self.daily_loss = abs(daily_change)
        else:
            self.daily_loss = 0.0

        if total_change < 0:
            self.total_loss = abs(total_change)
        else:
            self.total_loss = 0.0

        logger.debug(
            f"Balance update: {old_balance:.2f} → {new_balance:.2f} "
            f"(daily_loss: {self.daily_loss:.2f}, total_loss: {self.total_loss:.2f})"
        )

    def apply_loss(self, loss_amount: float) -> None:
        """Apply a loss to the account."""
        self.account_balance -= loss_amount
        self.update_balance(self.account_balance)


def check_daily_loss(state: FTMOState, account_balance: float | None = None) -> bool:
    """
    Check if daily loss limit would be exceeded.

    Args:
        state: FTMO state object
        account_balance: Current account balance (if None, uses state.account_balance)

    Returns:
        True if daily loss limit is OK, False if exceeded
    """
    if account_balance is not None:
        state.update_balance(account_balance)
    else:
        state.reset_daily()

    daily_loss_pct = state.daily_loss / state.daily_start_balance if state.daily_start_balance > 0 else 0.0
    limit_pct = state.daily_loss_limit

    if daily_loss_pct >= limit_pct:
        logger.warning(
            f"Daily loss limit exceeded: {daily_loss_pct:.2%} >= {limit_pct:.2%} "
            f"(loss: ${state.daily_loss:.2f}, limit: ${state.daily_start_balance * limit_pct:.2f})"
        )
        return False

    logger.debug(
        f"Daily loss check OK: {daily_loss_pct:.2%} < {limit_pct:.2%} "
        f"(loss: ${state.daily_loss:.2f}, remaining: ${state.daily_start_balance * limit_pct - state.daily_loss:.2f})"
    )
    return True


def check_total_loss(state: FTMOState, account_balance: float | None = None) -> bool:
    """
    Check if total loss limit would be exceeded.

    Args:
        state: FTMO state object
        account_balance: Current account balance (if None, uses state.account_balance)

    Returns:
        True if total loss limit is OK, False if exceeded
    """
    if account_balance is not None:
        state.update_balance(account_balance)
    else:
        state.reset_daily()

    initial_balance = 10000.0  # Starting balance
    total_loss_pct = state.total_loss / initial_balance if initial_balance > 0 else 0.0
    limit_pct = state.total_loss_limit

    if total_loss_pct >= limit_pct:
        logger.warning(
            f"Total loss limit exceeded: {total_loss_pct:.2%} >= {limit_pct:.2%} "
            f"(loss: ${state.total_loss:.2f}, limit: ${initial_balance * limit_pct:.2f})"
        )
        return False

    logger.debug(
        f"Total loss check OK: {total_loss_pct:.2%} < {limit_pct:.2%} "
        f"(loss: ${state.total_loss:.2f}, remaining: ${initial_balance * limit_pct - state.total_loss:.2f})"
    )
    return True


def check_ftmo_limits(state: FTMOState, account_balance: float | None = None) -> tuple[bool, str]:
    """
    Check both daily and total loss limits.

    Args:
        state: FTMO state object
        account_balance: Current account balance (if None, uses state.account_balance)

    Returns:
        Tuple of (is_ok, reason)
        - is_ok: True if both limits OK, False if either exceeded
        - reason: Empty string if OK, reason if not OK
    """
    daily_ok = check_daily_loss(state, account_balance)
    total_ok = check_total_loss(state, account_balance)

    if not daily_ok:
        return False, "Daily loss limit exceeded"
    if not total_ok:
        return False, "Total loss limit exceeded"

    return True, ""


def calculate_position_size(
    account_balance: float, risk_per_trade: float = 0.01, entry_price: float = 50000.0, sl_price: float = 49000.0
) -> float:
    """
    Calculate position size based on risk management.

    Args:
        account_balance: Current account balance
        risk_per_trade: Risk per trade as fraction (default: 1%)
        entry_price: Entry price
        sl_price: Stop loss price

    Returns:
        Position size in base currency (e.g., BTC units)
    """
    risk_amount = account_balance * risk_per_trade
    price_risk = abs(entry_price - sl_price)

    if price_risk <= 0:
        logger.warning(f"Invalid price risk: {price_risk}, using default risk")
        price_risk = entry_price * 0.02  # 2% default risk

    position_size = risk_amount / price_risk

    logger.debug(
        f"Position size calculation: balance=${account_balance:.2f}, "
        f"risk={risk_per_trade:.1%}, risk_amount=${risk_amount:.2f}, "
        f"price_risk=${price_risk:.2f}, position_size={position_size:.6f}"
    )

    return position_size

