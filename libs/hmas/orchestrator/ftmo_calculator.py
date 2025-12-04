"""
FTMO Risk Calculator
Ensures all trades comply with FTMO risk limits
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta


class FTMOCalculator:
    """
    FTMO Risk Management Calculator

    Hard Limits:
    - Daily Loss: 4.5% max
    - Total Loss: 9% max
    - Position Sizing: Risk-based (1-2% per trade)
    """

    def __init__(
        self,
        account_balance: float,
        daily_loss_limit: float = 0.045,  # 4.5%
        max_loss_limit: float = 0.09,      # 9.0%
        max_drawdown: float = 0.10         # 10%
    ):
        self.account_balance = account_balance
        self.daily_loss_limit = daily_loss_limit
        self.max_loss_limit = max_loss_limit
        self.max_drawdown = max_drawdown

        # Tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.last_reset = datetime.now(timezone.utc)

    def calculate_lot_size(
        self,
        risk_percent: float,
        entry_price: float,
        stop_loss: float,
        pip_value: float = 10.0
    ) -> float:
        """
        Calculate lot size for exact risk percentage

        Args:
            risk_percent: Risk as decimal (0.01 = 1.0%)
            entry_price: Entry price
            stop_loss: Stop loss price
            pip_value: Value per pip for 1 lot (default $10 for forex)

        Returns:
            Lot size (e.g., 0.67 lots)
        """
        # Risk amount in USD
        risk_amount = self.account_balance * risk_percent

        # Stop loss in pips
        sl_pips = abs(entry_price - stop_loss) * 10000

        # Lot size = Risk Amount / (SL pips × pip value)
        lot_size = risk_amount / (sl_pips * pip_value)

        # Round to 2 decimals
        return round(lot_size, 2)

    def validate_trade(
        self,
        risk_amount: float,
        current_daily_loss: Optional[float] = None,
        current_total_loss: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate if trade complies with FTMO limits

        Args:
            risk_amount: Risk amount in USD for this trade
            current_daily_loss: Current daily loss (if tracking externally)
            current_total_loss: Current total loss (if tracking externally)

        Returns:
            Validation result dict
        """
        # Use provided values or internal tracking
        daily_loss = current_daily_loss if current_daily_loss is not None else abs(min(self.daily_pnl, 0))
        total_loss = current_total_loss if current_total_loss is not None else abs(min(self.total_pnl, 0))

        # Calculate potential losses after this trade
        potential_daily_loss = daily_loss + risk_amount
        potential_total_loss = total_loss + risk_amount

        # Check limits
        daily_loss_pct = potential_daily_loss / self.account_balance
        total_loss_pct = potential_total_loss / self.account_balance

        daily_ok = daily_loss_pct <= self.daily_loss_limit
        total_ok = total_loss_pct <= self.max_loss_limit

        return {
            'valid': daily_ok and total_ok,
            'daily_loss_pct': daily_loss_pct,
            'daily_loss_limit': self.daily_loss_limit,
            'daily_ok': daily_ok,
            'total_loss_pct': total_loss_pct,
            'total_loss_limit': self.max_loss_limit,
            'total_ok': total_ok,
            'risk_amount': risk_amount,
            'remaining_daily_risk': (self.daily_loss_limit * self.account_balance) - daily_loss,
            'remaining_total_risk': (self.max_loss_limit * self.account_balance) - total_loss
        }

    def reset_daily_stats(self):
        """Reset daily P&L (call at start of each trading day)"""
        self.daily_pnl = 0.0
        self.last_reset = datetime.now(timezone.utc)

    def record_trade_result(self, pnl: float):
        """Record trade result for tracking"""
        self.daily_pnl += pnl
        self.total_pnl += pnl

    def get_stats(self) -> Dict[str, Any]:
        """Get current FTMO statistics"""
        return {
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'daily_loss_pct': abs(min(self.daily_pnl, 0)) / self.account_balance,
            'total_loss_pct': abs(min(self.total_pnl, 0)) / self.account_balance,
            'daily_limit': self.daily_loss_limit,
            'total_limit': self.max_loss_limit,
            'daily_ok': abs(min(self.daily_pnl, 0)) / self.account_balance <= self.daily_loss_limit,
            'total_ok': abs(min(self.total_pnl, 0)) / self.account_balance <= self.max_loss_limit,
            'last_reset': self.last_reset.isoformat()
        }

    def __repr__(self) -> str:
        return f"FTMOCalculator(balance=${self.account_balance:,.2f}, daily={self.daily_loss_limit:.1%}, max={self.max_loss_limit:.1%})"
