"""
Advanced Exit Strategy for V7 Ultimate

Implements multiple exit conditions:
1. Trailing Stop - Lock in profits after 50% gain
2. Time-based Exit - Max hold time (24 hours)
3. Break-even Stop - Move SL to entry after 25% profit
4. Take Profit Levels - Partial exits at profit milestones
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ExitStrategy:
    """Advanced exit management for trades"""

    def __init__(
        self,
        trailing_stop_activation=0.005,  # 0.5% profit to activate
        trailing_stop_distance=0.002,     # 0.2% trailing distance
        max_hold_hours=24,                # 24 hour max hold
        breakeven_profit_threshold=0.0025, # 0.25% profit to move to breakeven
        partial_exit_enabled=False         # Partial exits (future feature)
    ):
        """
        Initialize exit strategy

        Args:
            trailing_stop_activation: Profit % to activate trailing stop
            trailing_stop_distance: Distance from peak for trailing stop
            max_hold_hours: Maximum hold time in hours
            breakeven_profit_threshold: Profit % to move SL to entry
            partial_exit_enabled: Enable partial position exits
        """
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
        self.max_hold_hours = max_hold_hours
        self.breakeven_profit_threshold = breakeven_profit_threshold
        self.partial_exit_enabled = partial_exit_enabled

    def calculate_exit_levels(
        self,
        entry_price: float,
        direction: str,
        initial_stop_loss: float,
        initial_take_profit: float,
        entry_timestamp: datetime
    ) -> Dict:
        """
        Calculate all exit levels for a trade

        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            initial_stop_loss: Initial SL price
            initial_take_profit: Initial TP price
            entry_timestamp: When trade was entered

        Returns:
            Dict with exit parameters
        """
        return {
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss': initial_stop_loss,
            'take_profit': initial_take_profit,
            'trailing_stop': None,  # Not activated yet
            'highest_price': entry_price if direction == 'long' else entry_price,
            'lowest_price': entry_price if direction == 'short' else entry_price,
            'entry_timestamp': entry_timestamp,
            'max_exit_time': entry_timestamp + timedelta(hours=self.max_hold_hours),
            'breakeven_activated': False
        }

    def update_exit_levels(
        self,
        current_price: float,
        current_time: datetime,
        exit_params: Dict
    ) -> Tuple[Dict, Optional[str]]:
        """
        Update exit levels based on current price

        Args:
            current_price: Current market price
            current_time: Current timestamp
            exit_params: Current exit parameters

        Returns:
            Tuple of (updated_params, exit_reason)
            exit_reason is None if no exit, or string describing exit reason
        """
        direction = exit_params['direction']
        entry_price = exit_params['entry_price']

        # Update highest/lowest prices
        if direction == 'long':
            exit_params['highest_price'] = max(exit_params['highest_price'], current_price)
            current_profit_pct = (current_price - entry_price) / entry_price
        else:  # short
            exit_params['lowest_price'] = min(exit_params['lowest_price'], current_price)
            current_profit_pct = (entry_price - current_price) / entry_price

        # Check time-based exit
        if current_time >= exit_params['max_exit_time']:
            return exit_params, f"time_exit (held {self.max_hold_hours}h)"

        # Check breakeven stop activation
        if not exit_params['breakeven_activated'] and current_profit_pct >= self.breakeven_profit_threshold:
            exit_params['stop_loss'] = entry_price
            exit_params['breakeven_activated'] = True
            logger.info(f"Breakeven stop activated at {current_profit_pct*100:.2f}% profit")

        # Check trailing stop activation
        if current_profit_pct >= self.trailing_stop_activation:
            if direction == 'long':
                trailing_stop = exit_params['highest_price'] * (1 - self.trailing_stop_distance)
                exit_params['trailing_stop'] = max(
                    exit_params['trailing_stop'] or 0,
                    trailing_stop
                )
            else:  # short
                trailing_stop = exit_params['lowest_price'] * (1 + self.trailing_stop_distance)
                exit_params['trailing_stop'] = min(
                    exit_params['trailing_stop'] or float('inf'),
                    trailing_stop
                )
            logger.debug(f"Trailing stop updated to {exit_params['trailing_stop']:.2f}")

        # Check exit conditions
        exit_reason = self._check_exit_hit(current_price, exit_params)

        return exit_params, exit_reason

    def _check_exit_hit(self, current_price: float, exit_params: Dict) -> Optional[str]:
        """
        Check if any exit condition is hit

        Returns:
            Exit reason string or None
        """
        direction = exit_params['direction']

        if direction == 'long':
            # Check stop loss
            if current_price <= exit_params['stop_loss']:
                if exit_params['breakeven_activated']:
                    return "breakeven_stop"
                return "stop_loss"

            # Check trailing stop
            if exit_params['trailing_stop'] and current_price <= exit_params['trailing_stop']:
                return "trailing_stop"

            # Check take profit
            if current_price >= exit_params['take_profit']:
                return "take_profit"

        else:  # short
            # Check stop loss
            if current_price >= exit_params['stop_loss']:
                if exit_params['breakeven_activated']:
                    return "breakeven_stop"
                return "stop_loss"

            # Check trailing stop
            if exit_params['trailing_stop'] and current_price >= exit_params['trailing_stop']:
                return "trailing_stop"

            # Check take profit
            if current_price <= exit_params['take_profit']:
                return "take_profit"

        return None

    def get_exit_summary(self, exit_params: Dict) -> str:
        """Get human-readable summary of exit parameters"""
        summary = [
            f"Entry: ${exit_params['entry_price']:.2f}",
            f"SL: ${exit_params['stop_loss']:.2f}",
            f"TP: ${exit_params['take_profit']:.2f}",
        ]

        if exit_params['trailing_stop']:
            summary.append(f"Trailing SL: ${exit_params['trailing_stop']:.2f}")

        if exit_params['breakeven_activated']:
            summary.append("Breakeven: ACTIVE")

        return " | ".join(summary)


# Example usage
if __name__ == "__main__":
    # Test exit strategy
    strategy = ExitStrategy(
        trailing_stop_activation=0.005,  # 0.5%
        trailing_stop_distance=0.002,    # 0.2%
        max_hold_hours=24,
        breakeven_profit_threshold=0.0025  # 0.25%
    )

    # Simulate LONG trade
    entry_price = 100.0
    entry_time = datetime.now()

    exit_params = strategy.calculate_exit_levels(
        entry_price=entry_price,
        direction='long',
        initial_stop_loss=98.0,  # -2%
        initial_take_profit=102.0,  # +2%
        entry_timestamp=entry_time
    )

    print("="*70)
    print("EXIT STRATEGY SIMULATION")
    print("="*70)
    print(f"\nInitial Parameters:")
    print(f"  Entry: ${entry_price:.2f}")
    print(f"  Initial SL: ${exit_params['stop_loss']:.2f}")
    print(f"  Initial TP: ${exit_params['take_profit']:.2f}")
    print(f"  Max Hold: {strategy.max_hold_hours} hours")

    # Simulate price movement
    print(f"\nSimulating price movements:")

    test_prices = [
        (100.1, "Small profit"),
        (100.26, "Breakeven activated (0.26% profit)"),
        (100.6, "Trailing stop activated (0.6% profit)"),
        (101.0, "Trailing stop updates"),
        (100.8, "Price pulls back (trailing stop should exit)"),
    ]

    for price, description in test_prices:
        current_time = entry_time + timedelta(minutes=len(test_prices))
        exit_params, exit_reason = strategy.update_exit_levels(price, current_time, exit_params)

        print(f"\n  Price: ${price:.2f} - {description}")
        print(f"    Current SL: ${exit_params['stop_loss']:.2f}")
        if exit_params['trailing_stop']:
            print(f"    Trailing SL: ${exit_params['trailing_stop']:.2f}")
        if exit_params['breakeven_activated']:
            print(f"    Breakeven: ACTIVE")
        if exit_reason:
            print(f"    ❌ EXIT TRIGGERED: {exit_reason}")
            break

    print("\n" + "="*70)
