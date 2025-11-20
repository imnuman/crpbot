"""
Signal Results Tracker

Monitors active BUY/SELL signals and updates database when they hit TP/SL or expire.
Tracks actual outcomes for performance metrics calculation.
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from libs.db.models import Signal
from libs.config.config import Settings

# Load settings
settings = Settings()

# Coinbase client for price fetching
COINBASE_AVAILABLE = False
try:
    import os
    from coinbase.rest import RESTClient

    api_key = os.getenv('COINBASE_API_KEY_NAME') or settings.dict().get('coinbase_api_key_name')
    api_secret = os.getenv('COINBASE_API_PRIVATE_KEY') or settings.dict().get('coinbase_api_private_key')

    if api_key and api_secret:
        coinbase_client = RESTClient(api_key=api_key, api_secret=api_secret)
        COINBASE_AVAILABLE = True
    else:
        coinbase_client = None
        print("⚠️  Coinbase API keys not configured, price fetching will be skipped")
except Exception as e:
    coinbase_client = None
    print(f"⚠️  Coinbase client not available: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalTracker:
    """Tracks active signals and updates their results when TP/SL hit"""

    def __init__(self, db_url: str = None):
        """Initialize tracker with database connection"""
        self.db_url = db_url or "sqlite:///tradingai.db"
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Tracking config
        self.check_interval_seconds = 60  # Check every 60 seconds
        self.signal_expiry_hours = 24  # Signals expire after 24 hours

        logger.info(f"Signal Tracker initialized with DB: {self.db_url}")
        logger.info(f"Check interval: {self.check_interval_seconds}s, Expiry: {self.signal_expiry_hours}h")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price from Coinbase"""
        if not COINBASE_AVAILABLE:
            logger.warning(f"Coinbase not available, cannot fetch price for {symbol}")
            return None

        try:
            # Get latest candle (1m)
            product = coinbase_client.get_product(product_id=symbol)
            if product and 'price' in product:
                return float(product['price'])

            # Fallback: use candles endpoint
            candles = coinbase_client.get_candles(
                product_id=symbol,
                start=int((datetime.now(timezone.utc) - timedelta(minutes=2)).timestamp()),
                end=int(datetime.now(timezone.utc).timestamp()),
                granularity="ONE_MINUTE"
            )

            if candles and 'candles' in candles and len(candles['candles']) > 0:
                latest = candles['candles'][0]
                return float(latest['close'])

            logger.warning(f"No price data returned for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def check_signal_exit(
        self,
        signal: Signal,
        current_price: float
    ) -> Optional[Dict]:
        """
        Check if signal has hit TP or SL

        Returns:
            Dict with exit info if signal exited, None otherwise
        """
        direction = signal.direction
        entry = signal.entry_price
        tp = signal.tp_price
        sl = signal.sl_price

        if not entry or not tp or not sl:
            logger.warning(f"Signal {signal.id} missing price data: entry={entry}, tp={tp}, sl={sl}")
            return None

        # Check for exit conditions
        exit_info = None

        if direction == 'long':
            # LONG: Win if price >= TP, Loss if price <= SL
            if current_price >= tp:
                pnl = current_price - entry
                pnl_pct = (pnl / entry) * 100
                exit_info = {
                    'result': 'win',
                    'exit_price': current_price,
                    'exit_time': datetime.now(timezone.utc),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': 'TP_HIT'
                }
                logger.info(f"✅ LONG WIN: {signal.symbol} Entry ${entry:.2f} → TP ${current_price:.2f} (+${pnl:.2f}, +{pnl_pct:.2f}%)")

            elif current_price <= sl:
                pnl = current_price - entry
                pnl_pct = (pnl / entry) * 100
                exit_info = {
                    'result': 'loss',
                    'exit_price': current_price,
                    'exit_time': datetime.now(timezone.utc),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': 'SL_HIT'
                }
                logger.info(f"❌ LONG LOSS: {signal.symbol} Entry ${entry:.2f} → SL ${current_price:.2f} (-${abs(pnl):.2f}, {pnl_pct:.2f}%)")

        elif direction == 'short':
            # SHORT: Win if price <= TP, Loss if price >= SL
            if current_price <= tp:
                pnl = entry - current_price
                pnl_pct = (pnl / entry) * 100
                exit_info = {
                    'result': 'win',
                    'exit_price': current_price,
                    'exit_time': datetime.now(timezone.utc),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': 'TP_HIT'
                }
                logger.info(f"✅ SHORT WIN: {signal.symbol} Entry ${entry:.2f} → TP ${current_price:.2f} (+${pnl:.2f}, +{pnl_pct:.2f}%)")

            elif current_price >= sl:
                pnl = entry - current_price
                pnl_pct = (pnl / entry) * 100
                exit_info = {
                    'result': 'loss',
                    'exit_price': current_price,
                    'exit_time': datetime.now(timezone.utc),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': 'SL_HIT'
                }
                logger.info(f"❌ SHORT LOSS: {signal.symbol} Entry ${entry:.2f} → SL ${current_price:.2f} (-${abs(pnl):.2f}, {pnl_pct:.2f}%)")

        return exit_info

    def check_signal_expiry(self, signal: Signal) -> Optional[Dict]:
        """Check if signal has expired (no TP/SL hit within expiry window)"""
        # Make signal.timestamp timezone-aware if it's naive
        signal_time = signal.timestamp
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)

        age_hours = (datetime.now(timezone.utc) - signal_time).total_seconds() / 3600

        if age_hours >= self.signal_expiry_hours:
            logger.info(f"⏰ Signal {signal.id} expired after {age_hours:.1f} hours")
            return {
                'result': 'expired',
                'exit_price': None,
                'exit_time': datetime.now(timezone.utc),
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'reason': 'EXPIRED'
            }

        return None

    def update_signal_result(self, session: Session, signal: Signal, exit_info: Dict):
        """Update signal in database with exit information"""
        try:
            signal.result = exit_info['result']
            signal.exit_time = exit_info['exit_time']
            signal.exit_price = exit_info['exit_price']
            signal.pnl = exit_info['pnl']

            session.commit()
            logger.info(f"✅ Updated signal {signal.id} result: {exit_info['result']}")

        except Exception as e:
            logger.error(f"Error updating signal {signal.id}: {e}")
            session.rollback()

    def process_active_signals(self) -> Dict:
        """
        Main processing loop: check all active signals for exits

        Returns:
            Stats dict with counts of signals processed
        """
        session = self.SessionLocal()
        stats = {
            'active_checked': 0,
            'wins': 0,
            'losses': 0,
            'expired': 0,
            'errors': 0
        }

        try:
            # Query active signals (BUY/SELL with no result yet)
            active_signals = session.query(Signal).filter(
                Signal.model_version == 'v7_ultimate',
                Signal.direction.in_(['long', 'short']),
                Signal.result.is_(None)
            ).all()

            logger.info(f"📊 Checking {len(active_signals)} active signals...")
            stats['active_checked'] = len(active_signals)

            for signal in active_signals:
                try:
                    # Check expiry first
                    exit_info = self.check_signal_expiry(signal)
                    if exit_info:
                        self.update_signal_result(session, signal, exit_info)
                        stats['expired'] += 1
                        continue

                    # Get current price
                    current_price = self.get_current_price(signal.symbol)
                    if current_price is None:
                        logger.warning(f"⚠️  Could not fetch price for {signal.symbol}, skipping")
                        continue

                    # Check if TP/SL hit
                    exit_info = self.check_signal_exit(signal, current_price)
                    if exit_info:
                        self.update_signal_result(session, signal, exit_info)
                        if exit_info['result'] == 'win':
                            stats['wins'] += 1
                        elif exit_info['result'] == 'loss':
                            stats['losses'] += 1

                except Exception as e:
                    logger.error(f"Error processing signal {signal.id}: {e}")
                    stats['errors'] += 1

            return stats

        except Exception as e:
            logger.error(f"Error in process_active_signals: {e}")
            return stats

        finally:
            session.close()

    def run_continuous(self):
        """Run tracker in continuous loop"""
        logger.info("🚀 Starting Signal Tracker (continuous mode)")
        logger.info(f"Check interval: {self.check_interval_seconds}s")

        iteration = 0
        try:
            while True:
                iteration += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"Iteration #{iteration} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
                logger.info(f"{'='*80}")

                stats = self.process_active_signals()

                logger.info(f"📈 Results: {stats['wins']} wins, {stats['losses']} losses, "
                           f"{stats['expired']} expired, {stats['errors']} errors")

                if stats['active_checked'] == 0:
                    logger.info("💤 No active signals to track")

                logger.info(f"⏱️  Sleeping {self.check_interval_seconds}s...\n")
                time.sleep(self.check_interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n🛑 Signal Tracker stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in tracker loop: {e}")
            raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='V7 Signal Results Tracker')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds (default: 60)')
    parser.add_argument('--expiry-hours', type=int, default=24,
                       help='Signal expiry time in hours (default: 24)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (for testing)')

    args = parser.parse_args()

    tracker = SignalTracker()
    tracker.check_interval_seconds = args.interval
    tracker.signal_expiry_hours = args.expiry_hours

    if args.once:
        logger.info("Running once and exiting...")
        stats = tracker.process_active_signals()
        logger.info(f"Final stats: {stats}")
    else:
        tracker.run_continuous()


if __name__ == '__main__':
    main()
