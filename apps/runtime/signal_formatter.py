"""Signal formatter for creating detailed, actionable trading signals.

Formats signals with:
- Entry zones with current price indication
- Order types based on price position (BUY STOP, LIMIT BUY, SELL STOP, LIMIT SELL)
- Stop loss and take profit levels
- Position sizing
- Clear visual hierarchy
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple

from libs.constants import RISK_PER_TRADE
from libs.utils.timezone import now_est


class SignalFormatter:
    """Formats trading signals for Telegram and dashboard display."""

    def __init__(self, initial_balance: float = 10000, leverage: int = 10):
        """Initialize signal formatter.

        Args:
            initial_balance: Account balance in USD
            leverage: Trading leverage (default 10x)
        """
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.signal_counter = 0  # Track signal numbers

    def calculate_entry_zone(self, current_price: float, direction: str) -> Tuple[float, float]:
        """Calculate entry zone range (2% spread from current price).

        Args:
            current_price: Current market price
            direction: Trade direction ('long' or 'short')

        Returns:
            Tuple of (entry_min, entry_max)
        """
        # 2% spread for entry zone
        spread_pct = 0.02
        spread_amount = current_price * spread_pct

        if direction.lower() == 'long':
            # For LONG: entry zone below current (buy lower)
            entry_min = current_price - spread_amount
            entry_max = current_price
        else:  # SHORT
            # For SHORT: entry zone above current (sell higher)
            entry_min = current_price
            entry_max = current_price + spread_amount

        return (entry_min, entry_max)

    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """Calculate stop loss level (1.5% from entry).

        Args:
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')

        Returns:
            Stop loss price
        """
        sl_pct = 0.015  # 1.5%

        if direction.lower() == 'long':
            # LONG: stop below entry
            return entry_price * (1 - sl_pct)
        else:  # SHORT
            # SHORT: stop above entry
            return entry_price * (1 + sl_pct)

    def calculate_take_profit(self, entry_price: float, direction: str, risk_reward: float = 1.8) -> float:
        """Calculate take profit level based on risk:reward ratio.

        Args:
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            risk_reward: Risk:reward ratio (default 1.8)

        Returns:
            Take profit price
        """
        sl = self.calculate_stop_loss(entry_price, direction)
        risk_amount = abs(entry_price - sl)
        reward_amount = risk_amount * risk_reward

        if direction.lower() == 'long':
            # LONG: TP above entry
            return entry_price + reward_amount
        else:  # SHORT
            # SHORT: TP below entry
            return entry_price - reward_amount

    def calculate_position_size(self, entry_price: float, stop_loss: float, risk_usd: float) -> Tuple[float, float]:
        """Calculate position size based on risk.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_usd: Risk amount in USD

        Returns:
            Tuple of (position_size_units, notional_value_usd)
        """
        risk_per_unit = abs(entry_price - stop_loss)
        position_size = risk_usd / risk_per_unit if risk_per_unit > 0 else 0
        notional_value = position_size * entry_price

        return (position_size, notional_value)

    def determine_order_type(self, current_price: float, entry_min: float, entry_max: float,
                            direction: str) -> Tuple[str, float, str]:
        """Determine order type based on current price position.

        Args:
            current_price: Current market price
            entry_min: Entry zone minimum
            entry_max: Entry zone maximum
            direction: Trade direction ('long' or 'short')

        Returns:
            Tuple of (order_type, order_price, price_indicator)
        """
        if direction.lower() == 'long':
            if current_price < entry_min:
                # Price BELOW zone → BUY STOP (wait for breakout up)
                return ("BUY STOP ⬆️", entry_min, "⬇️")
            elif entry_min <= current_price <= entry_max:
                # Price IN zone → LIMIT BUY (buy the dip)
                return ("LIMIT BUY 💰", entry_min, "✅")
            else:  # current_price > entry_max
                # Price ABOVE zone → LIMIT BUY or Skip
                return ("LIMIT BUY", entry_max, "⬆️")
        else:  # SHORT
            if current_price > entry_max:
                # Price ABOVE zone → SELL STOP (wait for breakdown)
                return ("SELL STOP ⬇️", entry_max, "⬆️")
            elif entry_min <= current_price <= entry_max:
                # Price IN zone → LIMIT SELL (sell the rip)
                return ("LIMIT SELL 💰", entry_max, "✅")
            else:  # current_price < entry_min
                # Price BELOW zone → LIMIT SELL or Skip
                return ("LIMIT SELL", entry_min, "⬇️")

    def generate_reasoning(self, signal_data: dict) -> str:
        """Generate reasoning/why section from model predictions.

        Args:
            signal_data: Signal information dictionary

        Returns:
            Formatted reasoning text
        """
        confidence = signal_data.get('confidence', 0)
        direction = signal_data.get('direction', 'unknown')

        # Extract model confidence
        lstm_pred = signal_data.get('lstm_prediction', 0.5)
        trans_pred = signal_data.get('transformer_prediction', 0.5)

        # Determine model agreement
        signals_aligned = 0
        if direction == 'long' and lstm_pred > 0.5:
            signals_aligned += 1
        elif direction == 'short' and lstm_pred < 0.5:
            signals_aligned += 1

        if direction == 'long' and trans_pred > 0.5:
            signals_aligned += 1
        elif direction == 'short' and trans_pred < 0.5:
            signals_aligned += 1

        reasoning_parts = []

        # Confidence-based reasoning
        if confidence >= 0.80:
            reasoning_parts.append("Very strong signal")
        elif confidence >= 0.75:
            reasoning_parts.append("Strong trend indication")
        elif confidence >= 0.65:
            reasoning_parts.append("Moderate confidence")

        # Model alignment
        if signals_aligned >= 2:
            reasoning_parts.append(f"{signals_aligned} models aligned")

        return "• " + "\n• ".join(reasoning_parts) if reasoning_parts else "• Model prediction"

    def format_telegram_signal(self, signal_data: dict) -> str:
        """Format signal for Telegram notification with detailed entry instructions.

        Args:
            signal_data: Signal information dictionary containing:
                - symbol: Trading pair
                - direction: 'long' or 'short'
                - confidence: Model confidence (0-1)
                - tier: 'high', 'medium', or 'low'
                - entry_price: Current market price

        Returns:
            Formatted Telegram message
        """
        self.signal_counter += 1
        signal_num = self.signal_counter

        # Extract data
        symbol = signal_data.get('symbol', 'UNKNOWN')
        direction = signal_data.get('direction', 'unknown').upper()
        confidence = signal_data.get('confidence', 0)
        tier = signal_data.get('tier', 'unknown').upper()
        current_price = signal_data.get('entry_price', 0)

        # Calculate entry zone
        entry_min, entry_max = self.calculate_entry_zone(current_price, direction)

        # Use mid-point of entry zone as reference entry
        entry_ref = (entry_min + entry_max) / 2

        # Calculate SL/TP
        stop_loss = self.calculate_stop_loss(entry_ref, direction)
        take_profit = self.calculate_take_profit(entry_ref, direction)

        # Calculate position sizing
        risk_usd = self.initial_balance * RISK_PER_TRADE
        position_size, notional_value = self.calculate_position_size(entry_ref, stop_loss, risk_usd)

        # Determine order type
        order_type, order_price, price_indicator = self.determine_order_type(
            current_price, entry_min, entry_max, direction
        )

        # Format symbol for display (BTC-USD → BTC/USD)
        display_symbol = symbol.replace('-', '/')

        # Calculate risk:reward
        risk_amount = abs(entry_ref - stop_loss)
        reward_amount = abs(take_profit - entry_ref)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        # Generate reasoning
        reasoning = self.generate_reasoning(signal_data)

        # Get current timestamp
        timestamp = now_est().strftime('%Y-%m-%d %H:%M:%S EST')

        # Expiry time (30 minutes from now)
        expiry_time = now_est() + timedelta(minutes=30)
        expiry_str = expiry_time.strftime('%H:%M')

        # Build message
        message = f"""🎯 SIGNAL #{signal_num}

{display_symbol} {direction} | {confidence*100:.0f}% confidence

━━━━━━━━━━━━━━━━━━━━━━━
📍 ENTRY ZONE
━━━━━━━━━━━━━━━━━━━━━━━
Range: ${entry_min:,.2f} - ${entry_max:,.2f}
Current: ${current_price:,.2f} {price_indicator}

⚡ ENTRY ORDER
━━━━━━━━━━━━━━━━━━━━━━━
Type: {order_type}
Order Price: ${order_price:,.2f}

━━━━━━━━━━━━━━━━━━━━━━━
🛡️ STOP LOSS
━━━━━━━━━━━━━━━━━━━━━━━
Price: ${stop_loss:,.2f}
Type: STOP MARKET ⬇️

━━━━━━━━━━━━━━━━━━━━━━━
🎯 TAKE PROFIT
━━━━━━━━━━━━━━━━━━━━━━━
Price: ${take_profit:,.2f}
Type: LIMIT ⬆️

━━━━━━━━━━━━━━━━━━━━━━━
💰 POSITION SIZING
━━━━━━━━━━━━━━━━━━━━━━━
Risk: ${risk_usd:.0f} ({RISK_PER_TRADE*100:.1f}%)
Size: {position_size:.0f} {symbol.split('-')[0]}
Leverage: {self.leverage}x
Risk:Reward: 1:{rr_ratio:.1f}

━━━━━━━━━━━━━━━━━━━━━━━
📋 REASONING
━━━━━━━━━━━━━━━━━━━━━━━
{reasoning}

━━━━━━━━━━━━━━━━━━━━━━━
🟢 STATUS: READY
⏰ Valid until {expiry_str}
━━━━━━━━━━━━━━━━━━━━━━━

{timestamp}"""

        return message

    def format_dashboard_card(self, signal_data: dict) -> dict:
        """Format signal for dashboard display.

        Args:
            signal_data: Signal information dictionary

        Returns:
            Dictionary with formatted display fields
        """
        # Extract data
        symbol = signal_data.get('symbol', 'UNKNOWN')
        direction = signal_data.get('direction', 'unknown').upper()
        confidence = signal_data.get('confidence', 0)
        tier = signal_data.get('tier', 'unknown').upper()
        current_price = signal_data.get('entry_price', 0)

        # Calculate entry zone
        entry_min, entry_max = self.calculate_entry_zone(current_price, direction)
        entry_ref = (entry_min + entry_max) / 2

        # Calculate SL/TP
        stop_loss = self.calculate_stop_loss(entry_ref, direction)
        take_profit = self.calculate_take_profit(entry_ref, direction)

        # Calculate position sizing
        risk_usd = self.initial_balance * RISK_PER_TRADE
        position_size, _ = self.calculate_position_size(entry_ref, stop_loss, risk_usd)

        # Determine order type
        order_type, order_price, price_indicator = self.determine_order_type(
            current_price, entry_min, entry_max, direction
        )

        # Calculate RR
        risk_amount = abs(entry_ref - stop_loss)
        reward_amount = abs(take_profit - entry_ref)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        return {
            'signal_number': self.signal_counter,
            'symbol': symbol.replace('-', '/'),
            'direction': direction,
            'confidence': confidence,
            'tier': tier,
            'current_price': current_price,
            'price_indicator': price_indicator,
            'entry_min': entry_min,
            'entry_max': entry_max,
            'order_type': order_type,
            'order_price': order_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_usd': risk_usd,
            'position_size': position_size,
            'rr_ratio': rr_ratio,
            'timestamp': now_est(),
            'expiry_time': now_est() + timedelta(minutes=30)
        }
