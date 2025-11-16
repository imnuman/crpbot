#!/usr/bin/env python3
"""Track signal outcomes by comparing predictions vs actual price movements.

This script evaluates pending signals after a configured time period and updates
their outcomes (win/loss) based on actual market movement.
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import and_

from apps.runtime.data_fetcher import get_data_fetcher
from libs.config.config import Settings
from libs.db.models import Signal, get_session


class SignalOutcomeTracker:
    """Track and evaluate signal outcomes."""

    def __init__(self, config: Settings | None = None, evaluation_period_minutes: int = 15):
        """
        Initialize outcome tracker.

        Args:
            config: Settings object
            evaluation_period_minutes: Minutes to wait before evaluating signal
        """
        self.config = config or Settings()
        self.evaluation_period = timedelta(minutes=evaluation_period_minutes)
        self.data_fetcher = get_data_fetcher(self.config)
        self.session = get_session(self.config.db_url)

    def get_pending_signals(self):
        """Get signals that are ready to be evaluated."""
        cutoff_time = datetime.utcnow() - self.evaluation_period

        # Find signals that:
        # 1. Have result='pending'
        # 2. Were created more than evaluation_period ago
        signals = self.session.query(Signal).filter(
            and_(
                Signal.result == 'pending',
                Signal.timestamp <= cutoff_time
            )
        ).all()

        logger.info(f"Found {len(signals)} pending signals ready for evaluation")
        return signals

    def evaluate_signal(self, signal: Signal) -> dict:
        """
        Evaluate a single signal outcome.

        Args:
            signal: Signal record to evaluate

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Calculate exit time (entry time + evaluation period)
            exit_time = signal.timestamp + self.evaluation_period

            # Fetch latest market data and filter to evaluation period
            # Get more candles than needed to ensure we have the data
            df = self.data_fetcher.fetch_latest_candles(
                symbol=signal.symbol,
                num_candles=60  # 1 hour of 1-minute candles
            )

            if df.empty:
                logger.warning(f"No market data available for signal {signal.id}")
                return {'result': 'skipped', 'reason': 'no_data'}

            # Filter to signals within our evaluation window
            # Find candles between signal timestamp and exit time
            df_filtered = df[
                (df['timestamp'] >= signal.timestamp) &
                (df['timestamp'] <= exit_time)
            ]

            if df_filtered.empty:
                # If we don't have exact timestamp match, use closest candles
                df_filtered = df.iloc[:int(self.evaluation_period.total_seconds() / 60)]

            if df_filtered.empty:
                logger.warning(f"No data in evaluation window for signal {signal.id}")
                return {'result': 'skipped', 'reason': 'no_data_in_window'}

            # Get entry and exit prices
            entry_price = signal.entry_price or df_filtered.iloc[0]['close']
            exit_price = df_filtered.iloc[-1]['close']

            # Calculate price change percentage
            price_change_pct = ((exit_price - entry_price) / entry_price) * 100

            # Determine win/loss based on signal direction
            if signal.direction == 'long':
                # Long signal wins if price went up
                is_win = price_change_pct > 0
            else:  # short
                # Short signal wins if price went down
                is_win = price_change_pct < 0

            # Calculate PnL (simplified - assumes 1% risk per trade)
            risk_amount = 1.0  # 1% of capital
            pnl = risk_amount if is_win else -risk_amount

            result = {
                'result': 'win' if is_win else 'loss',
                'exit_time': exit_time,
                'exit_price': exit_price,
                'pnl': pnl,
                'price_change_pct': price_change_pct,
                'entry_price': entry_price
            }

            logger.info(
                f"Signal {signal.id} ({signal.symbol} {signal.direction.upper()}): "
                f"{'WIN' if is_win else 'LOSS'} - "
                f"Price change: {price_change_pct:+.2f}%, PnL: {pnl:+.2f}%"
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating signal {signal.id}: {e}")
            return {'result': 'skipped', 'reason': str(e)}

    def update_signal(self, signal: Signal, result: dict) -> None:
        """
        Update signal with evaluation results.

        Args:
            signal: Signal to update
            result: Evaluation results
        """
        try:
            signal.result = result['result']
            signal.exit_time = result.get('exit_time')
            signal.exit_price = result.get('exit_price')
            signal.pnl = result.get('pnl')

            # If entry_price wasn't set, update it
            if not signal.entry_price and 'entry_price' in result:
                signal.entry_price = result['entry_price']

            # Add notes about the evaluation
            if 'price_change_pct' in result:
                signal.notes = f"Price change: {result['price_change_pct']:+.2f}%"
            elif 'reason' in result:
                signal.notes = f"Skipped: {result['reason']}"

            self.session.commit()
            logger.debug(f"Updated signal {signal.id} with result: {result['result']}")

        except Exception as e:
            logger.error(f"Error updating signal {signal.id}: {e}")
            self.session.rollback()

    def run(self) -> dict:
        """
        Run outcome tracking for all pending signals.

        Returns:
            Dictionary with run statistics
        """
        logger.info("Starting signal outcome tracking...")

        signals = self.get_pending_signals()

        if not signals:
            logger.info("No signals to evaluate")
            return {'evaluated': 0, 'wins': 0, 'losses': 0, 'skipped': 0}

        stats = {'evaluated': 0, 'wins': 0, 'losses': 0, 'skipped': 0}

        for signal in signals:
            logger.debug(f"Evaluating signal {signal.id} ({signal.symbol} {signal.direction})")
            result = self.evaluate_signal(signal)
            self.update_signal(signal, result)

            stats['evaluated'] += 1
            if result['result'] == 'win':
                stats['wins'] += 1
            elif result['result'] == 'loss':
                stats['losses'] += 1
            else:
                stats['skipped'] += 1

        # Calculate win rate
        total_decided = stats['wins'] + stats['losses']
        win_rate = (stats['wins'] / total_decided * 100) if total_decided > 0 else 0

        logger.info(
            f"Evaluation complete: {stats['evaluated']} signals processed - "
            f"Wins: {stats['wins']}, Losses: {stats['losses']}, "
            f"Skipped: {stats['skipped']}, Win Rate: {win_rate:.1f}%"
        )

        return stats

    def get_performance_stats(self, hours: int = 24) -> dict:
        """
        Get performance statistics for a time period.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dictionary with performance metrics
        """
        since = datetime.utcnow() - timedelta(hours=hours)

        # Get all evaluated signals in time period
        signals = self.session.query(Signal).filter(
            and_(
                Signal.timestamp >= since,
                Signal.result.in_(['win', 'loss'])
            )
        ).all()

        if not signals:
            return {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_per_trade': 0.0,
                'by_symbol': {},
                'by_tier': {},
                'by_direction': {}
            }

        # Calculate overall stats
        wins = sum(1 for s in signals if s.result == 'win')
        losses = sum(1 for s in signals if s.result == 'loss')
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        total_pnl = sum(s.pnl or 0 for s in signals)
        avg_pnl = total_pnl / total if total > 0 else 0

        # Break down by symbol
        by_symbol = {}
        for s in signals:
            if s.symbol not in by_symbol:
                by_symbol[s.symbol] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if s.result == 'win':
                by_symbol[s.symbol]['wins'] += 1
            else:
                by_symbol[s.symbol]['losses'] += 1
            by_symbol[s.symbol]['pnl'] += (s.pnl or 0)

        # Break down by tier
        by_tier = {}
        for s in signals:
            if s.tier not in by_tier:
                by_tier[s.tier] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if s.result == 'win':
                by_tier[s.tier]['wins'] += 1
            else:
                by_tier[s.tier]['losses'] += 1
            by_tier[s.tier]['pnl'] += (s.pnl or 0)

        # Break down by direction
        by_direction = {}
        for s in signals:
            if s.direction not in by_direction:
                by_direction[s.direction] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if s.result == 'win':
                by_direction[s.direction]['wins'] += 1
            else:
                by_direction[s.direction]['losses'] += 1
            by_direction[s.direction]['pnl'] += (s.pnl or 0)

        return {
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'by_symbol': by_symbol,
            'by_tier': by_tier,
            'by_direction': by_direction
        }

    def close(self):
        """Close database session."""
        self.session.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Track trading signal outcomes")
    parser.add_argument(
        "--evaluation-period",
        type=int,
        default=15,
        help="Minutes to wait before evaluating signal (default: 15)"
    )
    parser.add_argument(
        "--stats-hours",
        type=int,
        default=24,
        help="Hours to include in performance stats (default: 24)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show performance stats, don't evaluate signals"
    )

    args = parser.parse_args()

    tracker = SignalOutcomeTracker(evaluation_period_minutes=args.evaluation_period)

    try:
        if not args.stats_only:
            # Run signal evaluation
            run_stats = tracker.run()
            print("\n" + "=" * 60)
            print("Signal Evaluation Results")
            print("=" * 60)
            print(f"Evaluated: {run_stats['evaluated']}")
            print(f"Wins: {run_stats['wins']}")
            print(f"Losses: {run_stats['losses']}")
            print(f"Skipped: {run_stats['skipped']}")
            if run_stats['wins'] + run_stats['losses'] > 0:
                wr = run_stats['wins'] / (run_stats['wins'] + run_stats['losses']) * 100
                print(f"Win Rate: {wr:.1f}%")
            print("=" * 60)

        # Show performance stats
        print(f"\n{'=' * 60}")
        print(f"Performance Stats (Last {args.stats_hours}h)")
        print("=" * 60)

        stats = tracker.get_performance_stats(hours=args.stats_hours)

        print(f"\nOverall:")
        print(f"  Total Signals: {stats['total_signals']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Total PnL: {stats['total_pnl']:+.2f}%")
        print(f"  Avg PnL/Trade: {stats['avg_pnl_per_trade']:+.2f}%")

        if stats['by_symbol']:
            print(f"\nBy Symbol:")
            for symbol, data in stats['by_symbol'].items():
                total = data['wins'] + data['losses']
                wr = (data['wins'] / total * 100) if total > 0 else 0
                print(f"  {symbol}: {data['wins']}W/{data['losses']}L ({wr:.1f}%) PnL: {data['pnl']:+.2f}%")

        if stats['by_tier']:
            print(f"\nBy Tier:")
            for tier, data in stats['by_tier'].items():
                total = data['wins'] + data['losses']
                wr = (data['wins'] / total * 100) if total > 0 else 0
                print(f"  {tier.upper()}: {data['wins']}W/{data['losses']}L ({wr:.1f}%) PnL: {data['pnl']:+.2f}%")

        if stats['by_direction']:
            print(f"\nBy Direction:")
            for direction, data in stats['by_direction'].items():
                total = data['wins'] + data['losses']
                wr = (data['wins'] / total * 100) if total > 0 else 0
                print(f"  {direction.upper()}: {data['wins']}W/{data['losses']}L ({wr:.1f}%) PnL: {data['pnl']:+.2f}%")

        print("=" * 60 + "\n")

    finally:
        tracker.close()


if __name__ == "__main__":
    main()
