#!/usr/bin/env python3
"""V6 Enhanced Model Monitoring Dashboard.

Real-time monitoring of V6 model predictions, signals, and accuracy.
"""
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, desc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.config.config import Settings
from libs.db.models import Signal


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name == 'posix' else 'cls')


def get_recent_signals(session, hours=24):
    """Get signals from last N hours."""
    since = datetime.utcnow() - timedelta(hours=hours)
    return session.query(Signal).filter(
        Signal.timestamp >= since
    ).order_by(desc(Signal.timestamp)).all()


def calculate_stats(signals):
    """Calculate statistics from signals."""
    if not signals:
        return None

    df = pd.DataFrame([{
        'symbol': s.symbol,
        'direction': s.direction,
        'confidence': s.confidence,
        'tier': s.tier,
        'timestamp': s.timestamp
    } for s in signals])

    stats = {
        'total_signals': len(df),
        'by_symbol': df['symbol'].value_counts().to_dict(),
        'by_direction': df['direction'].value_counts().to_dict(),
        'by_tier': df['tier'].value_counts().to_dict(),
        'avg_confidence': df['confidence'].mean(),
        'max_confidence': df['confidence'].max(),
        'min_confidence': df['confidence'].min(),
    }

    return stats


def print_dashboard(signals, stats, config):
    """Print monitoring dashboard."""
    clear_screen()

    # Header
    print("=" * 80)
    print("🚀 V6 ENHANCED MODEL MONITORING DASHBOARD")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {config.runtime_mode.upper()}")
    print(f"Confidence Threshold: {config.confidence_threshold * 100:.0f}%")
    print(f"Database: {config.db_url}")
    print("=" * 80)

    if not stats:
        print("\n⏳ No signals generated yet. Waiting for data...")
        print("\nThe runtime scans every 60 seconds and generates signals when:")
        print("  • Confidence ≥ 65%")
        print("  • Model detects trading opportunity")
        print("  • Rate limits not exceeded")
        return

    # Statistics
    print(f"\n📊 STATISTICS (Last 24 Hours)")
    print("-" * 80)
    print(f"Total Signals:     {stats['total_signals']}")
    print(f"Avg Confidence:    {stats['avg_confidence']:.1%}")
    print(f"Max Confidence:    {stats['max_confidence']:.1%}")
    print(f"Min Confidence:    {stats['min_confidence']:.1%}")

    # By Symbol
    print(f"\n📈 BY SYMBOL")
    print("-" * 40)
    for symbol, count in stats['by_symbol'].items():
        pct = (count / stats['total_signals']) * 100
        print(f"  {symbol:12} {count:3} signals ({pct:5.1f}%)")

    # By Direction
    print(f"\n🎯 BY DIRECTION")
    print("-" * 40)
    for direction, count in stats['by_direction'].items():
        pct = (count / stats['total_signals']) * 100
        emoji = "📈" if direction == "long" else "📉"
        print(f"  {emoji} {direction.upper():6} {count:3} signals ({pct:5.1f}%)")

    # By Tier
    print(f"\n⭐ BY CONFIDENCE TIER")
    print("-" * 40)
    tier_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_tiers = sorted(stats['by_tier'].items(), key=lambda x: tier_order.get(x[0], 99))
    for tier, count in sorted_tiers:
        pct = (count / stats['total_signals']) * 100
        emoji = "🔥" if tier == "high" else "⚡" if tier == "medium" else "💡"
        print(f"  {emoji} {tier.upper():8} {count:3} signals ({pct:5.1f}%)")

    # Recent Signals
    print(f"\n🔔 RECENT SIGNALS (Last 10)")
    print("-" * 80)
    print(f"{'Time':<20} {'Symbol':<12} {'Dir':<6} {'Conf':<8} {'Tier':<8}")
    print("-" * 80)

    for signal in signals[:10]:
        time_str = signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        dir_emoji = "📈" if signal.direction == "long" else "📉"
        tier_emoji = "🔥" if signal.tier == "high" else "⚡" if signal.tier == "medium" else "💡"

        print(
            f"{time_str:<20} "
            f"{signal.symbol:<12} "
            f"{dir_emoji} {signal.direction:<4} "
            f"{signal.confidence:>6.1%}  "
            f"{tier_emoji} {signal.tier:<6}"
        )

    print("-" * 80)
    print("\n💡 Press Ctrl+C to exit")


def main():
    """Main monitoring loop."""
    config = Settings()

    # Connect to database
    engine = create_engine(config.db_url)
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)

    logger.info("🚀 Starting V6 Enhanced Model Monitoring Dashboard")
    logger.info(f"   Database: {config.db_url}")
    logger.info(f"   Refresh: Every 5 seconds")

    try:
        while True:
            session = Session()
            try:
                signals = get_recent_signals(session, hours=24)
                stats = calculate_stats(signals)
                print_dashboard(signals, stats, config)
            finally:
                session.close()

            time.sleep(5)  # Refresh every 5 seconds

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped. Goodbye!")
        sys.exit(0)


if __name__ == '__main__':
    main()
