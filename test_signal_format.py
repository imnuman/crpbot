#!/usr/bin/env python3
"""Test script to demonstrate the new signal format."""

from apps.runtime.signal_formatter import SignalFormatter


def main():
    """Test signal formatter with sample data."""
    formatter = SignalFormatter(initial_balance=10000, leverage=10)

    # Test signals
    test_signals = [
        # Example 1: LONG - Price below entry (BUY STOP scenario)
        {
            'symbol': 'BTC-USD',
            'direction': 'long',
            'confidence': 0.82,
            'tier': 'high',
            'entry_price': 94100.00,  # Below entry zone
            'lstm_prediction': 0.75,
            'transformer_prediction': 0.68,
            'rl_prediction': 0.50
        },
        # Example 2: LONG - Price in entry (LIMIT BUY scenario)
        {
            'symbol': 'ETH-USD',
            'direction': 'long',
            'confidence': 0.80,
            'tier': 'high',
            'entry_price': 2430.00,  # In entry zone
            'lstm_prediction': 0.72,
            'transformer_prediction': 0.65,
            'rl_prediction': 0.50
        },
        # Example 3: SHORT - Price above entry (SELL STOP scenario)
        {
            'symbol': 'SOL-USD',
            'direction': 'short',
            'confidence': 0.79,
            'tier': 'high',
            'entry_price': 245.00,  # Above entry zone
            'lstm_prediction': 0.30,
            'transformer_prediction': 0.35,
            'rl_prediction': 0.50
        },
        # Example 4: SHORT - Price in entry (LIMIT SELL scenario)
        {
            'symbol': 'BNB-USD',
            'direction': 'short',
            'confidence': 0.77,
            'tier': 'high',
            'entry_price': 615.00,  # In entry zone
            'lstm_prediction': 0.28,
            'transformer_prediction': 0.32,
            'rl_prediction': 0.50
        },
    ]

    print("=" * 80)
    print("NEW SIGNAL FORMAT EXAMPLES")
    print("=" * 80)
    print()

    for i, signal in enumerate(test_signals, 1):
        print(f"\n{'=' * 80}")
        print(f"EXAMPLE {i}: {signal['symbol']} {signal['direction'].upper()}")
        print(f"Current Price: ${signal['entry_price']:,.2f}")
        print('=' * 80)
        print()

        # Format Telegram message
        message = formatter.format_telegram_signal(signal)
        print(message)

        # Also get dashboard card format
        card_data = formatter.format_dashboard_card(signal)
        print(f"\n--- Dashboard Card Data ---")
        print(f"Signal #{card_data['signal_number']}")
        print(f"Order Type: {card_data['order_type']}")
        print(f"Order Price: ${card_data['order_price']:,.2f}")
        print(f"Price Indicator: {card_data['price_indicator']}")
        print(f"SL: ${card_data['stop_loss']:,.2f}")
        print(f"TP: ${card_data['take_profit']:,.2f}")
        print(f"Risk: ${card_data['risk_usd']:.0f}")
        print(f"Size: {card_data['position_size']:.0f} {signal['symbol'].split('-')[0]}")
        print(f"RR: 1:{card_data['rr_ratio']:.1f}")

        print("\n" + "=" * 80)

        if i < len(test_signals):
            input("\nPress Enter for next example...")


if __name__ == '__main__':
    main()
