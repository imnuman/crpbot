"""FTMO rules enforcement library."""
from dataclasses import dataclass
from datetime import datetime

from loguru import logger


@dataclass
class FTMOState:
    """FTMO account state tracking."""

    account_balance: float = 10000.0
    daily_loss: float = 0.0
    total_loss: float = 0.0
    daily_start_balance: float = 10000.0
    daily_start_time: datetime | None = None
    daily_loss_limit: float = 0.045  # 4.5% of account
    total_loss_limit: float = 0.09  # 9% of account

    def __post_init__(self) -> None:
        """Initialize daily start time if not set."""
        if self.daily_start_time is None:
            today = datetime.utcnow()
            self.daily_start_time = today.replace(hour=0, minute=0, second=0, microsecond=0)

    def reset_daily(self) -> None:
        """Reset daily loss tracking (call at start of new trading day)."""
        now = datetime.utcnow()
        if self.daily_start_time and now.date() > self.daily_start_time.date():
            logger.info("Resetting daily loss tracking (new trading day)")
            self.daily_start_balance = self.account_balance
            self.daily_loss = 0.0
            self.daily_start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def update_balance(self, new_balance: float) -> None:
        """Update account balance and recalculate losses."""
        self.reset_daily()

        old_balance = self.account_balance
        self.account_balance = new_balance

        daily_change = self.account_balance - self.daily_start_balance
        total_change = self.account_balance - 10000.0  # Assuming 10K starting balance

        self.daily_loss = abs(daily_change) if daily_change < 0 else 0.0
        self.total_loss = abs(total_change) if total_change < 0 else 0.0

        logger.debug(
            "Balance update: %.2f → %.2f (daily_loss: %.2f, total_loss: %.2f)",
            old_balance,
            new_balance,
            self.daily_loss,
            self.total_loss,
        )

    def apply_loss(self, loss_amount: float) -> None:
        """Apply a loss to the account (for simulation/testing)."""
        self.account_balance -= loss_amount
        self.update_balance(self.account_balance)


def check_daily_loss(state: FTMOState, account_balance: float | None = None) -> bool:
    """Check if the daily loss limit is respected."""
    if account_balance is not None:
        state.update_balance(account_balance)
    else:
        state.reset_daily()

    if state.daily_start_balance <= 0:
        return False

    daily_loss_pct = state.daily_loss / state.daily_start_balance

    if daily_loss_pct >= state.daily_loss_limit:
        logger.warning(
            "Daily loss limit exceeded: %.2f%% >= %.2f%% (loss: $%.2f, limit: $%.2f)",
            daily_loss_pct * 100,
            state.daily_loss_limit * 100,
            state.daily_loss,
            state.daily_start_balance * state.daily_loss_limit,
        )
        return False

    return True


def check_total_loss(state: FTMOState, account_balance: float | None = None) -> bool:
    """Check if the total loss limit is respected."""
    if account_balance is not None:
        state.update_balance(account_balance)

    initial_balance = 10000.0
    total_loss_pct = state.total_loss / initial_balance if initial_balance > 0 else 0.0

    if total_loss_pct >= state.total_loss_limit:
        logger.warning(
            "Total loss limit exceeded: %.2f%% >= %.2f%% (loss: $%.2f, limit: $%.2f)",
            total_loss_pct * 100,
            state.total_loss_limit * 100,
            state.total_loss,
            initial_balance * state.total_loss_limit,
        )
        return False

    return True


def check_ftmo_limits(state: FTMOState, account_balance: float | None = None) -> tuple[bool, str]:
    """Check both FTMO daily and total loss limits."""
    if not check_daily_loss(state, account_balance):
        return False, "Daily loss limit exceeded"
    if not check_total_loss(state, account_balance):
        return False, "Total loss limit exceeded"
    return True, ""


def calculate_position_size(
    account_balance: float,
    risk_per_trade: float = 0.01,
    entry_price: float = 50000.0,
    sl_price: float = 49000.0,
) -> float:
    """Calculate position size based on risk per trade."""
    risk_amount = account_balance * risk_per_trade
    price_risk = abs(entry_price - sl_price)

    if price_risk <= 0:
        logger.warning("Invalid price risk %.2f; falling back to 2%% of entry", price_risk)
        price_risk = entry_price * 0.02

    position_size = risk_amount / price_risk

    logger.debug(
        "Position size calculation: balance=$%.2f risk=%.2f%% position_size=%.6f",
        account_balance,
        risk_per_trade * 100,
        position_size,
    )

    return position_size


def check_daily_loss_limit(balance: float, daily_pnl: float, daily_loss_limit_pct: float = 0.05) -> bool:
    """Function wrapper to check daily loss limit using scalar inputs."""
    limit = balance * daily_loss_limit_pct
    if daily_pnl < -limit:
        logger.warning(
            "Daily loss limit exceeded: %.2f < -%.2f", daily_pnl, limit
        )
        return False
    return True


def check_total_loss_limit(
    initial_balance: float, current_balance: float, total_loss_limit_pct: float = 0.10
) -> bool:
    """Function wrapper to check total loss limit."""
    total_loss = current_balance - initial_balance
    limit = initial_balance * total_loss_limit_pct
    if total_loss < -limit:
        logger.warning(
            "Total loss limit exceeded: %.2f < -%.2f", total_loss, limit
        )
        return False
    return True


def check_position_size(
    balance: float, position_size: float, max_risk_pct: float = 0.01
) -> bool:
    """Function wrapper for position sizing checks."""
    max_position = balance * max_risk_pct
    if position_size > max_position:
        logger.warning(
            "Position size too large: %.2f > %.2f", position_size, max_position
        )
        return False
    return True
