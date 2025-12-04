#!/usr/bin/env python3
"""
Test V7 price display with manual BUY/SELL signal insertion

Since the market is currently choppy and V7 keeps recommending HOLD,
this test manually creates a BUY signal with prices to verify the display works.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.db.models import Signal, create_tables, get_session
from libs.config.config import Settings

def create_test_signals():
    """Create test BUY and SELL signals with prices"""
    config = Settings()
    create_tables(config.db_url)
    session = get_session(config.db_url)

    print("=" * 80)
    print("Creating test V7 signals with price targets...")
    print("=" * 80)

    # Test BUY signal
    buy_signal = Signal(
        timestamp=datetime.utcnow(),
        symbol='BTC-USD',
        direction='long',
        confidence=0.78,
        tier='high',
        ensemble_prediction=0.78,
        entry_price=91234.56,
        sl_price=90500.00,
        tp_price=92800.00,
        model_version='v7_ultimate',
        notes='{"theories": {"shannon_entropy": 0.523, "hurst": 0.72}, "reasoning": "Test BUY signal with price targets"}'
    )

    session.add(buy_signal)

    # Test SELL signal
    sell_signal = Signal(
        timestamp=datetime.utcnow(),
        symbol='ETH-USD',
        direction='short',
        confidence=0.81,
        tier='high',
        ensemble_prediction=0.81,
        entry_price=3245.67,
        sl_price=3310.00,
        tp_price=3120.50,
        model_version='v7_ultimate',
        notes='{"theories": {"shannon_entropy": 0.42, "hurst": 0.35}, "reasoning": "Test SELL signal with price targets"}'
    )

    session.add(sell_signal)

    session.commit()

    print("\n✅ Created 2 test signals:")
    print(f"  1. BTC-USD BUY  @ ${buy_signal.entry_price:,.2f} (SL: ${buy_signal.sl_price:,.2f}, TP: ${buy_signal.tp_price:,.2f})")
    print(f"  2. ETH-USD SELL @ ${sell_signal.entry_price:,.2f} (SL: ${sell_signal.sl_price:,.2f}, TP: ${sell_signal.tp_price:,.2f})")

    # Calculate R:R ratios
    buy_risk = abs(buy_signal.entry_price - buy_signal.sl_price)
    buy_reward = abs(buy_signal.tp_price - buy_signal.entry_price)
    buy_rr = buy_reward / buy_risk if buy_risk > 0 else 0

    sell_risk = abs(sell_signal.entry_price - sell_signal.sl_price)
    sell_reward = abs(sell_signal.tp_price - sell_signal.entry_price)
    sell_rr = sell_reward / sell_risk if sell_risk > 0 else 0

    print(f"\n  BUY  R:R = 1:{buy_rr:.2f}")
    print(f"  SELL R:R = 1:{sell_rr:.2f}")

    session.close()

    print("\n" + "=" * 80)
    print("Test signals created successfully!")
    print("=" * 80)
    print("\nTo view in dashboard:")
    print("1. cd apps/dashboard")
    print("2. uv run python app.py")
    print("3. Open http://localhost:5000")
    print("4. Check the 'V7 Ultimate Signals' section")
    print("=" * 80)

def verify_signals():
    """Verify signals were created correctly"""
    config = Settings()
    session = get_session(config.db_url)

    print("\n" + "=" * 80)
    print("Verifying signals in database...")
    print("=" * 80)

    signals = session.query(Signal).filter(
        Signal.model_version == 'v7_ultimate'
    ).order_by(Signal.timestamp.desc()).limit(5).all()

    print(f"\nFound {len(signals)} V7 signals:\n")
    print(f"{'Timestamp':<20} {'Symbol':<10} {'Dir':<6} {'Conf':<8} {'Entry':<12} {'SL':<12} {'TP':<12} {'R:R':<8}")
    print("-" * 100)

    for s in signals:
        # Calculate R:R
        rr = "N/A"
        if s.entry_price and s.sl_price and s.tp_price:
            risk = abs(s.entry_price - s.sl_price)
            reward = abs(s.tp_price - s.entry_price)
            if risk > 0:
                rr = f"1:{reward/risk:.2f}"

        entry = f"${s.entry_price:,.2f}" if s.entry_price else "N/A"
        sl = f"${s.sl_price:,.2f}" if s.sl_price else "N/A"
        tp = f"${s.tp_price:,.2f}" if s.tp_price else "N/A"

        print(f"{str(s.timestamp)[:19]:<20} {s.symbol:<10} {s.direction:<6} {s.confidence*100:>6.1f}% {entry:<12} {sl:<12} {tp:<12} {rr:<8}")

    session.close()

    print("\n" + "=" * 80)
    print("✅ Verification complete!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        create_test_signals()
        verify_signals()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
