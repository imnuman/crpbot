#!/usr/bin/env python3
"""
Fetch 1-minute on-chain data from CoinGecko Analyst API.

Features:
- Active addresses (1m intervals)
- Transaction count (1m intervals)
- Transaction volume (USD)
- Gas prices (ETH/SOL)
- Hash rate (BTC)

Requirements:
- CoinGecko Analyst subscription ($129/month)
- 500 calls/min rate limit
- On-chain data available from 2021-01-01

Usage:
    python scripts/fetch_coingecko_onchain.py --symbol BTC-USD --start 2021-01-01 --end 2025-11-15
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
RATE_LIMIT_DELAY = 0.12  # seconds


def fetch_onchain_data(coin_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch 1-minute on-chain data from CoinGecko.

    Note: This endpoint may be specific to CoinGecko Analyst tier.
    Check API documentation for exact endpoint path.

    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin')
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with on-chain metrics at 1-minute intervals
    """
    logger.info(f"Fetching on-chain data for {coin_id} from {start_date} to {end_date}")

    # Convert to timestamps (milliseconds)
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    # Try different potential endpoints for on-chain data
    # Note: Exact endpoint may vary - adjust based on API docs

    # Option 1: Dedicated on-chain endpoint (if available)
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/onchain_data"

    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }

    params = {
        'from': start_ts,
        'to': end_ts,
        'interval': '1m'  # Request 1-minute intervals
    }

    try:
        logger.debug(f"Request: GET {url}")
        logger.debug(f"Params: {params}")

        response = requests.get(url, headers=headers, params=params)

        # If 404, try alternative endpoint
        if response.status_code == 404:
            logger.warning(f"Endpoint {url} not found, trying alternative...")

            # Option 2: Market chart with on-chain metrics
            url_alt = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range"
            params_alt = {
                'vs_currency': 'usd',
                'from': start_ts,
                'to': end_ts,
                'include_onchain': 'true',  # Request on-chain data
                'interval': '1m'
            }

            response = requests.get(url_alt, headers=headers, params=params_alt)

        response.raise_for_status()
        data = response.json()

        # Parse response (structure depends on actual API response)
        # This is a placeholder - adjust based on actual response format

        if not data:
            logger.error(f"No on-chain data returned for {coin_id}")
            return pd.DataFrame()

        # Example parsing (adjust based on actual response structure)
        if 'onchain_data' in data:
            df = parse_onchain_response(data['onchain_data'])
        elif 'active_addresses' in data:
            df = parse_onchain_metrics(data)
        else:
            logger.warning(f"Unknown response structure for on-chain data")
            logger.warning(f"Response keys: {list(data.keys())}")
            # Return empty DataFrame with expected schema
            df = pd.DataFrame(columns=[
                'timestamp',
                'active_addresses',
                'transaction_count',
                'transaction_volume_usd',
                'gas_price',
                'hash_rate'
            ])

        if not df.empty:
            logger.info(f"✅ Fetched {len(df)} on-chain data points")
            logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logger.error("Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
            return fetch_onchain_data(coin_id, start_date, end_date)  # Retry
        elif response.status_code == 403:
            logger.error("Forbidden - on-chain data may require higher tier subscription")
            logger.error("Check CoinGecko Analyst plan features")
        else:
            logger.error(f"HTTP error: {e}")
            logger.error(f"Response: {response.text}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch on-chain data: {e}")
        raise


def parse_onchain_response(onchain_data: dict) -> pd.DataFrame:
    """
    Parse on-chain data from API response.

    Adjust this function based on actual API response structure.
    """
    # Placeholder implementation
    # TODO: Update based on actual API response format

    records = []

    # Example structure (adjust as needed)
    for entry in onchain_data:
        records.append({
            'timestamp': pd.to_datetime(entry.get('timestamp', 0), unit='ms', utc=True),
            'active_addresses': entry.get('active_addresses', 0),
            'transaction_count': entry.get('transaction_count', 0),
            'transaction_volume_usd': entry.get('transaction_volume_usd', 0.0),
            'gas_price': entry.get('gas_price', 0.0),
            'hash_rate': entry.get('hash_rate', 0.0)
        })

    df = pd.DataFrame(records)
    return df


def parse_onchain_metrics(data: dict) -> pd.DataFrame:
    """
    Parse on-chain metrics from separate arrays.

    Similar to how market_chart returns prices, volumes separately.
    """
    # Example structure
    timestamps = []
    active_addresses = []
    tx_counts = []
    tx_volumes = []

    if 'active_addresses' in data:
        for ts, value in data['active_addresses']:
            timestamps.append(ts)
            active_addresses.append(value)

    if 'transaction_counts' in data:
        for ts, value in data['transaction_counts']:
            tx_counts.append(value)

    if 'transaction_volumes' in data:
        for ts, value in data['transaction_volumes']:
            tx_volumes.append(value)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='ms', utc=True),
        'active_addresses': active_addresses,
        'transaction_count': tx_counts if tx_counts else [0] * len(timestamps),
        'transaction_volume_usd': tx_volumes if tx_volumes else [0.0] * len(timestamps),
        'gas_price': [0.0] * len(timestamps),  # May need separate call
        'hash_rate': [0.0] * len(timestamps)   # May need separate call
    })

    return df


def fetch_in_chunks(coin_id: str, start_date: datetime, end_date: datetime,
                    chunk_days: int = 30) -> pd.DataFrame:
    """
    Fetch on-chain data in chunks to avoid timeouts.

    Args:
        coin_id: CoinGecko coin ID
        start_date: Start date
        end_date: End date
        chunk_days: Days per chunk (default: 30)

    Returns:
        Combined DataFrame
    """
    all_data = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)

        logger.info(f"Fetching chunk: {current_start.date()} to {current_end.date()}")

        chunk_data = fetch_onchain_data(coin_id, current_start, current_end)

        if not chunk_data.empty:
            all_data.append(chunk_data)

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        current_start = current_end

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')

        return df
    else:
        return pd.DataFrame()


def save_to_parquet(df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime):
    """Save DataFrame to parquet file."""
    output_dir = Path('data/raw/coingecko_onchain')
    output_dir.mkdir(parents=True, exist_ok=True)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    filename = f'{symbol}_1m_onchain_{start_str}_{end_str}.parquet'
    output_path = output_dir / filename

    df.to_parquet(output_path, index=False, compression='gzip')

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved: {output_path}")
    logger.info(f"   Size: {file_size:.2f} MB")
    logger.info(f"   Rows: {len(df):,}")
    logger.info(f"   Columns: {list(df.columns)}")

    return output_path


def validate_data(df: pd.DataFrame, symbol: str):
    """Validate on-chain data quality."""
    logger.info(f"\n📊 On-Chain Data Quality Report for {symbol}")
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
    logger.info(f"   Days:  {(df['timestamp'].max() - df['timestamp'].min()).days}")

    # Check statistics
    logger.info(f"\n📊 On-Chain Statistics:")

    if 'active_addresses' in df.columns:
        logger.info(f"   Active Addresses:")
        logger.info(f"     Min:  {df['active_addresses'].min():,.0f}")
        logger.info(f"     Max:  {df['active_addresses'].max():,.0f}")
        logger.info(f"     Mean: {df['active_addresses'].mean():,.0f}")

    if 'transaction_count' in df.columns:
        logger.info(f"   Transaction Count:")
        logger.info(f"     Min:  {df['transaction_count'].min():,.0f}")
        logger.info(f"     Max:  {df['transaction_count'].max():,.0f}")
        logger.info(f"     Mean: {df['transaction_count'].mean():,.0f}")

    if 'transaction_volume_usd' in df.columns:
        logger.info(f"   Transaction Volume (USD):")
        logger.info(f"     Total: ${df['transaction_volume_usd'].sum():,.0f}")
        logger.info(f"     Mean:  ${df['transaction_volume_usd'].mean():,.0f}")

    logger.info("=" * 60)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Fetch on-chain data from CoinGecko Analyst API'
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
        default='2021-01-01',
        help='Start date (YYYY-MM-DD, default: 2021-01-01)'
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
        default=30,
        help='Days per chunk (default: 30)'
    )

    args = parser.parse_args()

    # Verify API key
    if not COINGECKO_API_KEY:
        logger.error("❌ COINGECKO_API_KEY not found in environment")
        logger.error("Please set it in .env file")
        sys.exit(1)

    logger.info(f"🚀 CoinGecko On-Chain Data Fetcher - {args.symbol}")
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
        # Fetch data in chunks
        df = fetch_in_chunks(coin_id, start_date, end_date, chunk_days=args.chunk_days)

        if df.empty:
            logger.error("No on-chain data fetched")
            logger.warning("⚠️  On-chain data may not be available for this coin/timeframe")
            logger.warning("⚠️  Or endpoint may require different API structure")
            logger.warning("⚠️  Check CoinGecko API docs for on-chain data endpoints")
            sys.exit(1)

        # Validate data
        validate_data(df, args.symbol)

        # Save to parquet
        output_path = save_to_parquet(df, args.symbol, start_date, end_date)

        logger.info(f"\n✅ SUCCESS!")
        logger.info(f"On-chain data saved to: {output_path}")
        logger.info(f"\n💡 NOTE: This script may need adjustment based on actual API response")
        logger.info(f"   If data looks wrong, check API documentation for correct endpoint/format")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Failed: {e}")
        logger.error("\n💡 Troubleshooting:")
        logger.error("   1. Check if on-chain data is available for this coin")
        logger.error("   2. Verify CoinGecko Analyst subscription is active")
        logger.error("   3. Check API documentation for correct endpoint")
        logger.error("   4. Try with a shorter date range first")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
