"""FTMO trading rules enforcement."""
from loguru import logger


def check_daily_loss_limit(balance: float, daily_pnl: float, daily_loss_limit_pct: float = 0.05) -> bool:
    """
    Check if daily loss limit is violated.

    Args:
        balance: Current account balance
        daily_pnl: Today's profit/loss
        daily_loss_limit_pct: Maximum daily loss as percentage (default: 5%)

    Returns:
        True if within limits, False if limit exceeded
    """
    daily_loss_limit = balance * daily_loss_limit_pct
    if daily_pnl < -daily_loss_limit:
        logger.warning(
            f"❌ Daily loss limit exceeded: ${daily_pnl:.2f} < -${daily_loss_limit:.2f}"
        )
        return False
    return True


def check_total_loss_limit(
    initial_balance: float, current_balance: float, total_loss_limit_pct: float = 0.10
) -> bool:
    """
    Check if total loss limit is violated.

    Args:
        initial_balance: Starting account balance
        current_balance: Current account balance
        total_loss_limit_pct: Maximum total loss as percentage (default: 10%)

    Returns:
        True if within limits, False if limit exceeded
    """
    total_loss = current_balance - initial_balance
    total_loss_limit = initial_balance * total_loss_limit_pct

    if total_loss < -total_loss_limit:
        logger.warning(
            f"❌ Total loss limit exceeded: ${total_loss:.2f} < -${total_loss_limit:.2f}"
        )
        return False
    return True


def check_position_size(
    balance: float, position_size: float, max_risk_pct: float = 0.01
) -> bool:
    """
    Check if position size is within risk limits.

    Args:
        balance: Current account balance
        position_size: Proposed position size
        max_risk_pct: Maximum risk per trade as percentage (default: 1%)

    Returns:
        True if within limits, False if exceeds limit
    """
    max_position = balance * max_risk_pct
    if position_size > max_position:
        logger.warning(
            f"❌ Position size too large: ${position_size:.2f} > ${max_position:.2f}"
        )
        return False
    return True
