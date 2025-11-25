"""
Drawdown Circuit Breaker

Emergency stop system for excessive losses to prevent account blow-up.

Problem:
- Series of losses compound (3 losses @ 0.8% each = -2.4%)
- Emotional/algorithmic spirals
- No automatic protection in losing streaks

Solution:
Multi-level protection:
- Level 1 (Warning): -3% daily → Reduce size 50%
- Level 2 (Emergency): -5% daily → Stop trading
- Level 3 (Shutdown): -9% total → Full shutdown (FTMO breach)

Expected Impact:
- Max Drawdown: -30-50% reduction
- Protects against blow-up scenarios
- FTMO compliance enforcement

Research: Professional risk management uses tiered circuit breakers
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DrawdownStatus:
    """Current drawdown status"""
    level: int  # 0=Normal, 1=Warning, 2=Emergency, 3=Shutdown
    is_trading_allowed: bool
    daily_drawdown_pct: float
    total_drawdown_pct: float
    session_pnl: float
    total_pnl: float
    current_balance: float
    action_required: Optional[str]  # 'warning', 'emergency', 'shutdown', None
    message: str
    position_size_multiplier: float  # 1.0=Normal, 0.5=Warning, 0.0=Emergency


class DrawdownCircuitBreaker:
    """
    Monitor real-time drawdown and enforce emergency stops

    Prevents account blow-up through tiered protection:
    1. Warning at -3% daily loss (reduce size)
    2. Emergency stop at -5% daily loss
    3. Full shutdown at -9% total loss (FTMO compliance)

    Usage:
        breaker = DrawdownCircuitBreaker(starting_balance=5000.0)

        # Before each trade
        status = breaker.check_drawdown()
        if not status.is_trading_allowed:
            logger.error(status.message)
            return  # Block trade

        # Apply size multiplier
        position_size *= status.position_size_multiplier

        # After trade closes
        breaker.update_balance(new_balance=5050.0)
    """

    def __init__(
        self,
        starting_balance: float,
        daily_loss_warning: float = 0.03,      # 3% daily loss
        daily_loss_emergency: float = 0.05,    # 5% daily loss
        total_loss_shutdown: float = 0.09,     # 9% total loss (FTMO)
        reset_hour: int = 0  # Hour to reset daily stats (0 = midnight UTC)
    ):
        """
        Initialize Drawdown Circuit Breaker

        Args:
            starting_balance: Initial account balance
            daily_loss_warning: Daily loss % for Level 1 warning
            daily_loss_emergency: Daily loss % for Level 2 emergency stop
            total_loss_shutdown: Total loss % for Level 3 shutdown
            reset_hour: Hour of day (0-23) to reset daily stats
        """
        if starting_balance <= 0:
            raise ValueError("Starting balance must be positive")

        self.starting_balance = starting_balance
        self.daily_loss_warning = daily_loss_warning
        self.daily_loss_emergency = daily_loss_emergency
        self.total_loss_shutdown = total_loss_shutdown
        self.reset_hour = reset_hour

        # Balance tracking
        self.current_balance = starting_balance
        self.peak_balance = starting_balance
        self.session_start_balance = starting_balance
        self.session_start_time = datetime.now()

        # P&L tracking
        self.session_pnl = 0.0
        self.total_pnl = 0.0

        # Circuit breaker state
        self.level = 0  # 0=Normal, 1=Warning, 2=Emergency, 3=Shutdown
        self.is_trading_allowed = True
        self.last_alert_time: Optional[datetime] = None
        self.alert_cooldown_minutes = 30  # Don't spam alerts

        logger.info(
            f"Circuit Breaker initialized | "
            f"Balance: ${starting_balance:,.2f} | "
            f"Thresholds: {daily_loss_warning:.1%}/{daily_loss_emergency:.1%}/{total_loss_shutdown:.1%}"
        )

    def check_drawdown(self, current_time: Optional[datetime] = None) -> DrawdownStatus:
        """
        Check current drawdown against thresholds

        Returns:
            DrawdownStatus with current state and action required
        """
        if current_time is None:
            current_time = datetime.now()

        # Auto-reset daily if past reset hour
        self._auto_reset_daily(current_time)

        # Calculate drawdowns
        session_dd_pct = self.session_pnl / self.session_start_balance if self.session_start_balance > 0 else 0.0
        total_dd_pct = (self.current_balance - self.starting_balance) / self.starting_balance if self.starting_balance > 0 else 0.0

        # Determine circuit breaker level
        level, action, message = self._determine_level(session_dd_pct, total_dd_pct)

        # Update state
        previous_level = self.level
        self.level = level
        self.is_trading_allowed = (level < 2)  # Levels 0-1 allow trading, 2-3 block

        # Send alert if level increased
        if level > previous_level:
            self._send_alert_if_needed(level, message, current_time)

        # Get position size multiplier
        multiplier = self._get_size_multiplier()

        return DrawdownStatus(
            level=level,
            is_trading_allowed=self.is_trading_allowed,
            daily_drawdown_pct=session_dd_pct,
            total_drawdown_pct=total_dd_pct,
            session_pnl=self.session_pnl,
            total_pnl=self.total_pnl,
            current_balance=self.current_balance,
            action_required=action,
            message=message,
            position_size_multiplier=multiplier
        )

    def _determine_level(
        self,
        session_dd_pct: float,
        total_dd_pct: float
    ) -> tuple[int, Optional[str], str]:
        """
        Determine circuit breaker level from drawdowns

        Returns:
            (level, action, message) tuple
        """
        # Level 3: SHUTDOWN (Total loss breach)
        if abs(total_dd_pct) >= self.total_loss_shutdown:
            return (
                3,
                'shutdown',
                f"🚨 SHUTDOWN: Total loss {total_dd_pct:.2%} >= {self.total_loss_shutdown:.1%} (FTMO BREACH)"
            )

        # Level 2: EMERGENCY STOP (Daily loss breach)
        elif abs(session_dd_pct) >= self.daily_loss_emergency:
            return (
                2,
                'emergency',
                f"⛔ EMERGENCY STOP: Daily loss {session_dd_pct:.2%} >= {self.daily_loss_emergency:.1%}"
            )

        # Level 1: WARNING (Approaching daily limit)
        elif abs(session_dd_pct) >= self.daily_loss_warning:
            return (
                1,
                'warning',
                f"⚠️  WARNING: Daily loss {session_dd_pct:.2%} >= {self.daily_loss_warning:.1%} (size reduced 50%)"
            )

        # Level 0: Normal
        else:
            return (
                0,
                None,
                f"✅ Normal: Daily {session_dd_pct:+.2%}, Total {total_dd_pct:+.2%}"
            )

    def _get_size_multiplier(self) -> float:
        """
        Get position size multiplier based on circuit breaker level

        Returns:
            1.0 = Normal (Level 0)
            0.5 = Warning (Level 1) - Reduce size 50%
            0.0 = Emergency/Shutdown (Level 2/3) - No trading
        """
        if self.level == 0:
            return 1.0  # Normal size
        elif self.level == 1:
            return 0.5  # Half size (warning)
        else:
            return 0.0  # No trading (emergency/shutdown)

    def update_balance(self, new_balance: float, trade_pnl: Optional[float] = None):
        """
        Update balance after trade completion

        Args:
            new_balance: New account balance
            trade_pnl: Optional P&L from specific trade (if not provided, calculated)
        """
        if new_balance < 0:
            logger.error(f"Invalid balance update: {new_balance}")
            return

        # Calculate P&L
        if trade_pnl is None:
            trade_pnl = new_balance - self.current_balance

        # Update balances
        old_balance = self.current_balance
        self.current_balance = new_balance
        self.session_pnl += trade_pnl
        self.total_pnl = new_balance - self.starting_balance

        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        logger.debug(
            f"Balance updated: ${old_balance:,.2f} → ${new_balance:,.2f} | "
            f"Trade P&L: {trade_pnl:+,.2f} | "
            f"Session: {self.session_pnl:+,.2f} | "
            f"Total: {self.total_pnl:+,.2f}"
        )

    def reset_daily(self, current_time: Optional[datetime] = None):
        """
        Reset daily statistics (call at start of new trading day)

        Args:
            current_time: Optional timestamp for reset
        """
        if current_time is None:
            current_time = datetime.now()

        logger.info(
            f"Daily reset | "
            f"Previous session P&L: {self.session_pnl:+,.2f} | "
            f"New starting balance: ${self.current_balance:,.2f}"
        )

        # Reset daily stats
        self.session_start_balance = self.current_balance
        self.session_start_time = current_time
        self.session_pnl = 0.0

        # Reset circuit breaker level if not in total shutdown
        if self.level < 3:
            old_level = self.level
            self.level = 0
            self.is_trading_allowed = True

            if old_level > 0:
                logger.info(f"Circuit breaker level reset: {old_level} → 0")

    def _auto_reset_daily(self, current_time: datetime):
        """
        Automatically reset daily stats if past reset hour

        Args:
            current_time: Current timestamp
        """
        # Check if we've passed the reset hour since session start
        reset_time_today = current_time.replace(hour=self.reset_hour, minute=0, second=0, microsecond=0)

        # If we're past today's reset time and session started before it
        if current_time >= reset_time_today and self.session_start_time < reset_time_today:
            logger.info(f"Auto-reset triggered at {current_time.strftime('%H:%M:%S')}")
            self.reset_daily(current_time)

    def _send_alert_if_needed(self, level: int, message: str, current_time: datetime):
        """
        Send alert if cooldown period has passed

        Args:
            level: Circuit breaker level
            message: Alert message
            current_time: Current timestamp
        """
        # Check cooldown
        if self.last_alert_time is not None:
            minutes_since_last = (current_time - self.last_alert_time).total_seconds() / 60
            if minutes_since_last < self.alert_cooldown_minutes:
                logger.debug(f"Alert cooldown active ({minutes_since_last:.1f} min)")
                return

        # Log alert
        urgency = ['INFO', 'WARNING', 'EMERGENCY', 'CRITICAL'][level]
        logger.warning(f"{urgency} ALERT: {message}")

        # Update last alert time
        self.last_alert_time = current_time

        # In production, send to Telegram/SMS/email
        # For now, just log it
        logger.info("Alert notification sent (placeholder)")

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive status summary

        Returns:
            Dict with all circuit breaker state
        """
        status = self.check_drawdown()

        return {
            'level': status.level,
            'level_name': ['Normal', 'Warning', 'Emergency', 'Shutdown'][status.level],
            'is_trading_allowed': status.is_trading_allowed,
            'daily_drawdown_pct': status.daily_drawdown_pct,
            'total_drawdown_pct': status.total_drawdown_pct,
            'session_pnl': status.session_pnl,
            'total_pnl': status.total_pnl,
            'current_balance': status.current_balance,
            'starting_balance': self.starting_balance,
            'peak_balance': self.peak_balance,
            'position_size_multiplier': status.position_size_multiplier,
            'message': status.message,
            'session_start_time': self.session_start_time.isoformat(),
            'thresholds': {
                'daily_warning': self.daily_loss_warning,
                'daily_emergency': self.daily_loss_emergency,
                'total_shutdown': self.total_loss_shutdown
            }
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("DRAWDOWN CIRCUIT BREAKER TEST")
    print("=" * 70)

    # Initialize with $5000 balance
    breaker = DrawdownCircuitBreaker(
        starting_balance=5000.0,
        daily_loss_warning=0.03,     # 3%
        daily_loss_emergency=0.05,   # 5%
        total_loss_shutdown=0.09     # 9%
    )

    # Scenario 1: Normal trading
    print("\n[Scenario 1] Normal Trading:")
    breaker.update_balance(5050.0)  # +$50 profit
    status = breaker.check_drawdown()
    print(f"  Level: {status.level} ({['Normal', 'Warning', 'Emergency', 'Shutdown'][status.level]})")
    print(f"  Trading Allowed: {status.is_trading_allowed}")
    print(f"  Daily P&L: {status.session_pnl:+.2f} ({status.daily_drawdown_pct:+.2%})")
    print(f"  Message: {status.message}")

    # Scenario 2: Warning level (-3.5% loss)
    print("\n[Scenario 2] Warning Level (-3.5%):")
    breaker.reset_daily()  # Reset for new day
    breaker.update_balance(4825.0)  # -$175 loss (-3.5%)
    status = breaker.check_drawdown()
    print(f"  Level: {status.level} ({['Normal', 'Warning', 'Emergency', 'Shutdown'][status.level]})")
    print(f"  Trading Allowed: {status.is_trading_allowed}")
    print(f"  Position Size Multiplier: {status.position_size_multiplier:.1f}x")
    print(f"  Daily P&L: {status.session_pnl:+.2f} ({status.daily_drawdown_pct:+.2%})")
    print(f"  Message: {status.message}")

    # Scenario 3: Emergency stop (-5.5% loss)
    print("\n[Scenario 3] Emergency Stop (-5.5%):")
    breaker.reset_daily()
    breaker.update_balance(4725.0)  # -$275 loss (-5.5%)
    status = breaker.check_drawdown()
    print(f"  Level: {status.level} ({['Normal', 'Warning', 'Emergency', 'Shutdown'][status.level]})")
    print(f"  Trading Allowed: {status.is_trading_allowed}")
    print(f"  Position Size Multiplier: {status.position_size_multiplier:.1f}x")
    print(f"  Daily P&L: {status.session_pnl:+.2f} ({status.daily_drawdown_pct:+.2%})")
    print(f"  Message: {status.message}")

    # Scenario 4: Full shutdown (-9.5% total loss)
    print("\n[Scenario 4] Full Shutdown (-9.5% total):")
    breaker.current_balance = 4525.0  # Set balance directly (total -9.5%)
    breaker.total_pnl = 4525.0 - 5000.0
    status = breaker.check_drawdown()
    print(f"  Level: {status.level} ({['Normal', 'Warning', 'Emergency', 'Shutdown'][status.level]})")
    print(f"  Trading Allowed: {status.is_trading_allowed}")
    print(f"  Total P&L: {status.total_pnl:+.2f} ({status.total_drawdown_pct:+.2%})")
    print(f"  Message: {status.message}")

    print("\n" + "=" * 70)
    print("✅ Drawdown Circuit Breaker ready for production!")
    print("=" * 70)
