#!/usr/bin/env python3
"""Comprehensive test script for data pipeline with larger datasets."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from apps.trainer.data_pipeline import (
    clean_and_validate_data,
    create_walk_forward_splits,
    fetch_historical_data,
    load_data,
    save_data,
)
from libs.config.config import Settings


def test_large_fetch(symbol: str, days: int, interval: str = "1m"):
    """Test fetching large amounts of data."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing large data fetch: {symbol} for {days} days ({interval})")
    logger.info(f"{'='*60}")

    config = Settings()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Expected approximate candles: {days * 24 * 60 if interval == '1m' else days * 24}")

    # Fetch data
    df = fetch_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        config=config,
    )

    if df.empty:
        logger.error(f"❌ No data fetched for {symbol}")
        return False

    logger.info(f"✅ Fetched {len(df)} candles")

    # Clean and validate
    df_clean, report = clean_and_validate_data(df, interval=interval)

    if report:
        logger.info(f"Data Quality Report:\n{report}")

    # Verify data integrity
    logger.info("\n📊 Data Integrity Checks:")
    logger.info(f"  Date range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
    logger.info(f"  Price range: ${df_clean['low'].min():.2f} - ${df_clean['high'].max():.2f}")
    logger.info(f"  Total volume: {df_clean['volume'].sum():.2f}")
    logger.info(f"  Avg volume: {df_clean['volume'].mean():.2f}")

    # Check for gaps
    if report and report.missing_periods:
        logger.warning(f"⚠️  Found {len(report.missing_periods)} missing periods")
        for i, (start, end) in enumerate(report.missing_periods[:5]):  # Show first 5
            logger.warning(f"  Gap {i+1}: {start} to {end}")

    # Save to file
    output_file = Path(f"data/raw/test_{symbol}_{interval}_{days}d.parquet")
    save_data(df_clean, output_file)

    # Verify we can load it back
    df_loaded = load_data(output_file)
    assert len(df_loaded) == len(df_clean), "Data mismatch after save/load"
    logger.info(f"✅ Data saved and verified: {output_file}")

    return True


def test_multiple_symbols(symbols: list[str], days: int = 7):
    """Test fetching data for multiple symbols."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing multiple symbols: {symbols}")
    logger.info(f"{'='*60}")

    results = {}
    for symbol in symbols:
        try:
            logger.info(f"\n📈 Testing {symbol}...")
            success = test_large_fetch(symbol, days=days, interval="1h")
            results[symbol] = success
        except Exception as e:
            logger.error(f"❌ Failed to fetch {symbol}: {e}")
            results[symbol] = False

    logger.info(f"\n{'='*60}")
    logger.info("Multi-symbol test results:")
    for symbol, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {symbol}: {status}")
    logger.info(f"{'='*60}")

    return all(results.values())


def test_walk_forward_splits(symbol: str, days: int = 30):
    """Test walk-forward split functionality."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing walk-forward splits for {symbol}")
    logger.info(f"{'='*60}")

    config = Settings()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    # Fetch data
    df = fetch_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="1h",
        config=config,
    )

    if df.empty:
        logger.error(f"❌ No data fetched for {symbol}")
        return False

    # Clean data
    df_clean, _ = clean_and_validate_data(df, interval="1h")

    # Define split dates
    train_end = start_date + timedelta(days=int(days * 0.7))  # 70% for training
    val_end = start_date + timedelta(days=int(days * 0.85))  # 15% for validation

    # Create splits
    train_df, val_df, test_df = create_walk_forward_splits(df_clean, train_end, val_end)

    logger.info(f"\n📊 Walk-Forward Split Results:")
    logger.info(f"  Train: {len(train_df)} candles ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    logger.info(f"  Val:   {len(val_df)} candles ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    logger.info(f"  Test:  {len(test_df)} candles ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    # Verify no overlap
    assert train_df["timestamp"].max() <= val_df["timestamp"].min(), "Train/Val overlap!"
    assert val_df["timestamp"].max() <= test_df["timestamp"].min(), "Val/Test overlap!"

    logger.info("✅ No temporal leakage detected")
    logger.info("✅ Walk-forward splits working correctly")

    return True


def test_different_intervals(symbol: str = "BTC-USD"):
    """Test fetching data with different intervals."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing different intervals for {symbol}")
    logger.info(f"{'='*60}")

    intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
    days = 7
    config = Settings()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    results = {}
    for interval in intervals:
        try:
            logger.info(f"\n⏱️  Testing {interval} interval...")
            df = fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                config=config,
            )

            if df.empty:
                logger.warning(f"⚠️  No data for {interval}")
                results[interval] = False
                continue

            df_clean, report = clean_and_validate_data(df, interval=interval)
            logger.info(f"  ✅ Fetched {len(df_clean)} candles")
            if report:
                logger.info(f"  Completeness: {report.completeness_pct:.2f}%")

            results[interval] = True
        except Exception as e:
            logger.error(f"  ❌ Failed for {interval}: {e}")
            results[interval] = False

    logger.info(f"\n{'='*60}")
    logger.info("Interval test results:")
    for interval, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {interval}: {status}")
    logger.info(f"{'='*60}")

    return all(results.values())


def main():
    """Run comprehensive data pipeline tests."""
    parser = argparse.ArgumentParser(description="Test data pipeline with larger datasets")
    parser.add_argument("--symbol", default="BTC-USD", help="Symbol to test")
    parser.add_argument("--days", type=int, default=7, help="Number of days to fetch")
    parser.add_argument("--interval", default="1h", help="Time interval")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--test-multi", action="store_true", help="Test multiple symbols")
    parser.add_argument("--test-splits", action="store_true", help="Test walk-forward splits")
    parser.add_argument("--test-intervals", action="store_true", help="Test different intervals")

    args = parser.parse_args()

    # Create output directory
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    if args.test_all:
        logger.info("🧪 Running ALL comprehensive tests...")
        test_large_fetch(args.symbol, days=args.days, interval=args.interval)
        test_multiple_symbols(["BTC-USD", "ETH-USD"], days=7)
        test_walk_forward_splits(args.symbol, days=30)
        test_different_intervals(args.symbol)
        logger.info("\n✅ All tests completed!")
    elif args.test_multi:
        test_multiple_symbols(["BTC-USD", "ETH-USD"], days=args.days)
    elif args.test_splits:
        test_walk_forward_splits(args.symbol, days=args.days)
    elif args.test_intervals:
        test_different_intervals(args.symbol)
    else:
        # Default: single large fetch test
        test_large_fetch(args.symbol, days=args.days, interval=args.interval)


if __name__ == "__main__":
    main()

