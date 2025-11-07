#!/usr/bin/env python3
"""Script to fetch historical cryptocurrency data."""
import argparse
from datetime import datetime, timedelta, timezone

from loguru import logger

from apps.trainer.data_pipeline import (
    clean_and_validate_data,
    create_walk_forward_splits,
    fetch_historical_data,
    save_data,
)
from libs.config.config import Settings


def main():
    """Fetch and save historical data."""
    parser = argparse.ArgumentParser(description="Fetch historical cryptocurrency data")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading pair (e.g., BTC-USD)")
    parser.add_argument(
        "--start", default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--interval", default="1m", help="Time interval (1m, 5m, 1h, 1d)")
    parser.add_argument(
        "--output", default="data/raw", help="Output directory for parquet files"
    )
    parser.add_argument("--test", action="store_true", help="Test mode: fetch only 100 candles")

    args = parser.parse_args()

    # Load config
    config = Settings()
    logger.info(f"Using data provider: {config.data_provider}")

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)

    # Test mode: limit to 100 candles
    if args.test:
        logger.info("TEST MODE: Limiting to ~100 candles")
        end_date = start_date + timedelta(minutes=100)  # For 1m interval

    # Fetch data
    logger.info(f"Fetching {args.symbol} from {start_date} to {end_date}")
    df = fetch_historical_data(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
        config=config,
    )

    if df.empty:
        logger.error("No data fetched!")
        return

    # Clean and validate
    df, quality_report = clean_and_validate_data(df, interval=args.interval)
    if quality_report:
        logger.info(f"Data quality report:\n{quality_report}")

    # Save raw data
    from pathlib import Path

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.symbol}_{args.interval}_{start_date.date()}_{end_date.date()}.parquet"
    save_data(df, output_file)

    logger.info(f"✅ Data saved to {output_file}")
    logger.info(f"Total candles: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()

