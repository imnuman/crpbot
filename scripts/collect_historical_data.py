"""
Collect 2 years of historical OHLCV data for all V7 symbols

This script collects hourly candles for all 10 trading symbols
and saves them in parquet format for fast backtesting.
"""
import sys
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_historical_data(
    symbols,
    days_back=730,
    granularity=3600,
    output_dir='data/historical'
):
    """
    Collect historical data for all symbols

    Args:
        symbols: List of trading symbols
        days_back: Days of historical data (730 = 2 years)
        granularity: Candle size in seconds (3600 = 1 hour)
        output_dir: Output directory
    """
    from apps.runtime.data_fetcher import get_data_fetcher
    from libs.config.config import Settings

    config = Settings()
    fetcher = get_data_fetcher(config)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    logger.info(f"Collecting {days_back} days of {granularity}s candles for {len(symbols)} symbols")
    logger.info(f"Date range: {start_time.date()} to {end_time.date()}")
    logger.info(f"Output directory: {output_path}")

    results = {}

    for i, symbol in enumerate(symbols):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i+1}/{len(symbols)}] Fetching {symbol}...")
        logger.info(f"{'='*70}")

        try:
            # Fetch historical candles
            df = fetcher.fetch_historical_candles(
                symbol=symbol,
                start=start_time,
                end=end_time,
                granularity=granularity
            )

            if df.empty:
                logger.warning(f"No data collected for {symbol}")
                continue

            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

            # Validate data
            logger.info(f"  Candles collected: {len(df)}")
            logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

            # Check for gaps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_diffs = df['timestamp'].diff()
            expected_diff = pd.Timedelta(seconds=granularity)
            gaps = time_diffs[time_diffs > expected_diff * 1.5]

            if len(gaps) > 0:
                logger.warning(f"  Found {len(gaps)} gaps in data")
            else:
                logger.info(f"  ✅ No gaps detected")

            # Save to parquet
            symbol_safe = symbol.replace('-', '_')
            filename = f"{symbol_safe}_{granularity}s_{days_back}d.parquet"
            filepath = output_path / filename

            df.to_parquet(filepath, compression='snappy', index=False)
            file_size_mb = filepath.stat().st_size / 1024 / 1024

            logger.info(f"  ✅ Saved to {filename} ({file_size_mb:.2f} MB)")

            results[symbol] = df

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"  ❌ Failed to collect {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary report
    logger.info(f"\n{'='*70}")
    logger.info("COLLECTION SUMMARY")
    logger.info(f"{'='*70}\n")

    successful = len(results)
    failed = len(symbols) - successful

    logger.info(f"Total symbols: {len(symbols)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}\n")

    for symbol, df in results.items():
        logger.info(f"{symbol}:")
        logger.info(f"  Candles: {len(df)}")
        logger.info(f"  Size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        logger.info("")

    logger.info(f"{'='*70}")
    logger.info(f"✅ Collection complete! Files saved to {output_path}")
    logger.info(f"{'='*70}")

    return results


def main():
    # V7 symbols
    symbols = [
        'BTC-USD', 'ETH-USD', 'SOL-USD',
        'XRP-USD', 'DOGE-USD', 'ADA-USD', 'AVAX-USD',
        'LINK-USD', 'POL-USD', 'LTC-USD'
    ]

    # Production: collect 2 years (730 days)
    DAYS_BACK = 730

    logger.info("="*70)
    logger.info("HISTORICAL DATA COLLECTION")
    logger.info("="*70)
    logger.info(f"\nCollecting {DAYS_BACK} days of hourly data (2 years)...")
    logger.info("This will take approximately 5-10 minutes...\n")

    results = collect_historical_data(
        symbols=symbols,
        days_back=DAYS_BACK,
        granularity=3600,  # 1 hour
        output_dir='data/historical'
    )

    logger.info("\n✅ Done!")


if __name__ == "__main__":
    main()
