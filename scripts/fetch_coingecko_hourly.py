#!/usr/bin/env python3
"""
Fetch hourly OHLCV data from CoinGecko Analyst API.

Features:
- Hourly candles from 2018-present (7+ years)
- OHLC + volume data
- Fast download (500 calls/min rate limit)

Requirements:
- CoinGecko Analyst subscription
- Hourly data available from 2018

Usage:
    python scripts/fetch_coingecko_hourly.py --symbol BTC-USD --start 2018-01-01 --end 2025-11-15
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

# CoinGecko configuration
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'

# Coin ID mapping
COIN_IDS = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana'
}

# Rate limiting (500 calls/min = 0.12s per call)
RATE_LIMIT_DELAY = 0.12


def fetch_hourly_ohlcv(coin_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch hourly OHLCV from CoinGecko market_chart endpoint.

    Args:
        coin_id: CoinGecko coin ID
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with hourly OHLCV
    """
    logger.info(f"Fetching hourly OHLCV for {coin_id} from {start_date} to {end_date}")

    # Calculate days between dates
    days = (end_date - start_date).days

    # CoinGecko API: Use 'days' parameter for automatic interval
    # For >90 days, it returns hourly data automatically
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"

    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }

    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'hourly'  # Request hourly interval explicitly
    }

    try:
        logger.debug(f"Request: GET {url}")
        logger.debug(f"Params: {params}")

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        if not data or 'prices' not in data:
            logger.error(f"No price data returned for {coin_id}")
            return pd.DataFrame()

        # Parse prices and volumes
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])

        if not prices:
            logger.error("Empty prices array")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': [p[0] for p in prices],
            'price': [p[1] for p in prices]
        })

        # Add volume
        if volumes:
            df['volume'] = [v[1] for v in volumes]
        else:
            df['volume'] = 0.0

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Note: CoinGecko market_chart doesn't return OHLC directly
        # It returns price points which we can use as close prices
        # For true OHLC, we'd need to aggregate or use a different endpoint

        # For now, use price as close (common approach)
        df['close'] = df['price']
        df['open'] = df['close'].shift(1).fillna(df['close'])  # Approximate
        df['high'] = df['close']  # Approximate (would need tick data for true high)
        df['low'] = df['close']   # Approximate (would need tick data for true low)

        # Reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Filter to requested date range
        df = df[(df['timestamp'] >= pd.Timestamp(start_date, tz='UTC')) &
                (df['timestamp'] <= pd.Timestamp(end_date, tz='UTC'))]

        # Sort and remove duplicates
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')

        logger.info(f"✅ Fetched {len(df)} hourly candles")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logger.error("Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
            return fetch_hourly_ohlcv(coin_id, start_date, end_date)  # Retry
        else:
            logger.error(f"HTTP error: {e}")
            logger.error(f"Response: {response.text}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch hourly data: {e}")
        raise


def fetch_in_chunks(coin_id: str, start_date: datetime, end_date: datetime,
                    chunk_days: int = 365) -> pd.DataFrame:
    """
    Fetch hourly data in yearly chunks.

    CoinGecko may limit the number of data points per request,
    so we fetch in chunks and combine.
    """
    all_data = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)

        logger.info(f"Fetching chunk: {current_start.date()} to {current_end.date()}")

        chunk_data = fetch_hourly_ohlcv(coin_id, current_start, current_end)

        if not chunk_data.empty:
            all_data.append(chunk_data)

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        current_start = current_end

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        return df
    else:
        return pd.DataFrame()


def save_to_parquet(df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime):
    """Save DataFrame to parquet file."""
    output_dir = Path('data/raw/coingecko_hourly')
    output_dir.mkdir(parents=True, exist_ok=True)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    filename = f'{symbol}_1h_{start_str}_{end_str}.parquet'
    output_path = output_dir / filename

    df.to_parquet(output_path, index=False, compression='gzip')

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved: {output_path}")
    logger.info(f"   Size: {file_size:.2f} MB")
    logger.info(f"   Rows: {len(df):,}")
    logger.info(f"   Columns: {list(df.columns)}")

    return output_path


def validate_data(df: pd.DataFrame, symbol: str):
    """Validate hourly OHLCV data."""
    logger.info(f"\n📊 Hourly OHLCV Quality Report for {symbol}")
    logger.info("=" * 60)

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"⚠️  Missing values found:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"   {col}: {count} missing ({count/len(df)*100:.2f}%)")
    else:
        logger.info("✅ No missing values")

    # Check data range
    logger.info(f"\n📅 Date Range:")
    logger.info(f"   Start: {df['timestamp'].min()}")
    logger.info(f"   End:   {df['timestamp'].max()}")
    logger.info(f"   Hours: {len(df):,}")
    logger.info(f"   Days:  {(df['timestamp'].max() - df['timestamp'].min()).days}")

    # Check price statistics
    logger.info(f"\n💰 Price Statistics:")
    logger.info(f"   Low:  ${df['low'].min():,.2f}")
    logger.info(f"   High: ${df['high'].max():,.2f}")
    logger.info(f"   Mean: ${df['close'].mean():,.2f}")

    # Check volume
    logger.info(f"\n📊 Volume Statistics:")
    logger.info(f"   Total: ${df['volume'].sum():,.0f}")
    logger.info(f"   Mean:  ${df['volume'].mean():,.0f}")

    # Quality checks
    logger.info(f"\n🔍 Quality Checks:")

    zero_prices = (df['close'] == 0).sum()
    if zero_prices > 0:
        logger.warning(f"⚠️  {zero_prices} candles with zero close price")
    else:
        logger.info("✅ No zero prices")

    # Check for gaps (missing hours)
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff()
    expected_diff = pd.Timedelta(hours=1)
    gaps = (time_diffs > expected_diff * 1.5).sum()  # Allow 50% tolerance

    if gaps > 0:
        logger.warning(f"⚠️  {gaps} gaps detected (missing hourly data)")
        logger.warning(f"   This is normal for crypto (exchange downtime, etc.)")
    else:
        logger.info("✅ No significant gaps")

    logger.info("=" * 60)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Fetch hourly OHLCV from CoinGecko Analyst API'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        choices=['BTC-USD', 'ETH-USD', 'SOL-USD'],
        help='Trading pair symbol'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2018-01-01',
        help='Start date (YYYY-MM-DD, default: 2018-01-01)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--chunk-days',
        type=int,
        default=365,
        help='Days per chunk (default: 365)'
    )

    args = parser.parse_args()

    # Verify API key
    if not COINGECKO_API_KEY:
        logger.error("❌ COINGECKO_API_KEY not found in environment")
        sys.exit(1)

    logger.info(f"🚀 CoinGecko Hourly OHLCV Fetcher - {args.symbol}")
    logger.info(f"API Key: {COINGECKO_API_KEY[:10]}...")

    # Map symbol to CoinGecko ID
    coin_id = COIN_IDS.get(args.symbol)
    if not coin_id:
        logger.error(f"Unknown symbol: {args.symbol}")
        sys.exit(1)

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Days: {(end_date - start_date).days}")

    try:
        # Fetch data
        df = fetch_in_chunks(coin_id, start_date, end_date, chunk_days=args.chunk_days)

        if df.empty:
            logger.error("No hourly data fetched")
            sys.exit(1)

        # Validate data
        validate_data(df, args.symbol)

        # Save to parquet
        output_path = save_to_parquet(df, args.symbol, start_date, end_date)

        logger.info(f"\n✅ SUCCESS!")
        logger.info(f"Hourly OHLCV saved to: {output_path}")
        logger.info(f"\n💡 NOTE: OHLC values are approximated from price points")
        logger.info(f"   True OHLC would require tick data aggregation")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
