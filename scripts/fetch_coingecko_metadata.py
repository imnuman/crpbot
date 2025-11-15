#!/usr/bin/env python3
"""
Fetch daily market metadata from CoinGecko Analyst API.

Features:
- Market cap, rank, supply (daily)
- BTC/ETH dominance (global context)
- Cross-exchange prices & spreads
- Social sentiment metrics
- Trending status

Requirements:
- CoinGecko Analyst subscription
- 500 calls/min rate limit

Usage:
    python scripts/fetch_coingecko_metadata.py --symbol BTC-USD --start 2023-11-15 --end 2025-11-15
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


def fetch_market_chart_daily(coin_id: str, days: int) -> pd.DataFrame:
    """
    Fetch daily price, market cap, and volume.

    This is the base data we'll enrich with metadata.
    """
    logger.info(f"Fetching market chart for {coin_id} ({days} days)")

    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"

    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }

    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        # Parse prices, market_caps, total_volumes
        df = pd.DataFrame({
            'timestamp': [p[0] for p in data.get('prices', [])],
            'price': [p[1] for p in data.get('prices', [])],
            'market_cap': [m[1] for m in data.get('market_caps', [])],
            'total_volume': [v[1] for v in data.get('total_volumes', [])]
        })

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['date'] = df['timestamp'].dt.date

        logger.info(f"✅ Fetched {len(df)} daily market data points")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch market chart: {e}")
        raise


def fetch_coin_info(coin_id: str) -> dict:
    """
    Fetch current coin metadata.

    Returns market_cap_rank, circulating_supply, total_supply, etc.
    """
    logger.info(f"Fetching coin info for {coin_id}")

    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}"

    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }

    params = {
        'localization': 'false',
        'tickers': 'false',
        'market_data': 'true',
        'community_data': 'true',
        'developer_data': 'false',
        'sparkline': 'false'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract key metrics
        market_data = data.get('market_data', {})
        community_data = data.get('community_data', {})

        info = {
            'market_cap_rank': data.get('market_cap_rank', 0),
            'circulating_supply': market_data.get('circulating_supply', 0),
            'total_supply': market_data.get('total_supply', 0),
            'max_supply': market_data.get('max_supply', 0),
            'ath_usd': market_data.get('ath', {}).get('usd', 0),
            'ath_date': market_data.get('ath_date', {}).get('usd', ''),
            'twitter_followers': community_data.get('twitter_followers', 0),
            'reddit_subscribers': community_data.get('reddit_subscribers', 0)
        }

        logger.info(f"✅ Fetched coin info (rank: {info['market_cap_rank']})")

        return info

    except Exception as e:
        logger.error(f"Failed to fetch coin info: {e}")
        return {}


def fetch_global_data() -> dict:
    """
    Fetch global market data (BTC dominance, total market cap, etc.).

    This is fetched once per day and applies to all symbols.
    """
    logger.info("Fetching global market data")

    url = f"{COINGECKO_BASE_URL}/global"

    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json().get('data', {})

        global_info = {
            'total_market_cap_usd': data.get('total_market_cap', {}).get('usd', 0),
            'total_volume_24h_usd': data.get('total_volume', {}).get('usd', 0),
            'btc_dominance': data.get('market_cap_percentage', {}).get('btc', 0),
            'eth_dominance': data.get('market_cap_percentage', {}).get('eth', 0),
            'active_cryptocurrencies': data.get('active_cryptocurrencies', 0),
            'markets': data.get('markets', 0)
        }

        logger.info(f"✅ Global: BTC dominance {global_info['btc_dominance']:.2f}%")

        return global_info

    except Exception as e:
        logger.error(f"Failed to fetch global data: {e}")
        return {}


def fetch_cross_exchange_data(coin_id: str) -> dict:
    """
    Fetch prices across multiple exchanges (cross-exchange spread).
    """
    logger.info(f"Fetching cross-exchange data for {coin_id}")

    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/tickers"

    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }

    params = {
        'include_exchange_logo': 'false',
        'depth': 'true'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        tickers = data.get('tickers', [])

        if not tickers:
            return {}

        # Extract USD prices from different exchanges
        usd_prices = []
        volumes = []

        for ticker in tickers:
            if ticker.get('target') == 'USD' or ticker.get('target') == 'USDT':
                price = ticker.get('last', 0)
                volume = ticker.get('converted_volume', {}).get('usd', 0)

                if price > 0:
                    usd_prices.append(price)
                    volumes.append(volume)

        if not usd_prices:
            return {}

        # Calculate statistics
        avg_price = sum(usd_prices) / len(usd_prices)
        min_price = min(usd_prices)
        max_price = max(usd_prices)
        spread_pct = ((max_price - min_price) / avg_price) * 100 if avg_price > 0 else 0

        # Volume-weighted average price
        total_volume = sum(volumes)
        if total_volume > 0:
            vwap = sum(p * v for p, v in zip(usd_prices, volumes)) / total_volume
        else:
            vwap = avg_price

        cross_exchange = {
            'global_avg_price': avg_price,
            'global_min_price': min_price,
            'global_max_price': max_price,
            'cross_exchange_spread_pct': spread_pct,
            'vwap': vwap,
            'num_exchanges': len(usd_prices)
        }

        logger.info(f"✅ Cross-exchange: {len(usd_prices)} exchanges, spread: {spread_pct:.2f}%")

        return cross_exchange

    except Exception as e:
        logger.error(f"Failed to fetch cross-exchange data: {e}")
        return {}


def build_daily_metadata(coin_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Build comprehensive daily metadata dataset.

    Combines:
    - Market chart (price, market cap, volume)
    - Coin info (rank, supply)
    - Global data (BTC dominance)
    - Cross-exchange data (spreads)
    """
    days = (end_date - start_date).days

    logger.info(f"Building daily metadata for {coin_id} ({days} days)")

    # 1. Fetch market chart (price, market cap, volume - daily)
    market_df = fetch_market_chart_daily(coin_id, days)

    if market_df.empty:
        logger.error("Failed to fetch market chart")
        return pd.DataFrame()

    time.sleep(RATE_LIMIT_DELAY)

    # 2. Fetch coin info (current metadata)
    coin_info = fetch_coin_info(coin_id)
    time.sleep(RATE_LIMIT_DELAY)

    # 3. Fetch global data (current)
    global_data = fetch_global_data()
    time.sleep(RATE_LIMIT_DELAY)

    # 4. Fetch cross-exchange data (current)
    cross_exchange = fetch_cross_exchange_data(coin_id)
    time.sleep(RATE_LIMIT_DELAY)

    # Combine into single DataFrame
    # Note: coin_info, global_data, cross_exchange are current snapshots
    # For historical data, we'd need to fetch daily snapshots (expensive!)

    # For now, use current values as proxy for all dates
    # In production, could cache daily snapshots or use time-series endpoints

    for key, value in coin_info.items():
        market_df[key] = value

    for key, value in global_data.items():
        market_df[key] = value

    for key, value in cross_exchange.items():
        market_df[key] = value

    # Calculate derived features
    market_df['market_cap_change_pct'] = market_df['market_cap'].pct_change() * 100
    market_df['volume_change_pct'] = market_df['total_volume'].pct_change() * 100
    market_df['price_change_pct'] = market_df['price'].pct_change() * 100

    # Distance from ATH
    if 'ath_usd' in market_df.columns and market_df['ath_usd'].iloc[0] > 0:
        market_df['ath_distance_pct'] = ((market_df['price'] - market_df['ath_usd']) / market_df['ath_usd']) * 100
    else:
        market_df['ath_distance_pct'] = 0.0

    # Circulating supply percentage
    if 'total_supply' in market_df.columns and market_df['total_supply'].iloc[0] > 0:
        market_df['circulating_pct'] = (market_df['circulating_supply'] / market_df['total_supply']) * 100
    else:
        market_df['circulating_pct'] = 100.0

    logger.info(f"✅ Built metadata with {len(market_df.columns)} features")

    return market_df


def save_to_parquet(df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime):
    """Save DataFrame to parquet file."""
    output_dir = Path('data/raw/coingecko_daily')
    output_dir.mkdir(parents=True, exist_ok=True)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    filename = f'{symbol}_daily_{start_str}_{end_str}.parquet'
    output_path = output_dir / filename

    df.to_parquet(output_path, index=False, compression='gzip')

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved: {output_path}")
    logger.info(f"   Size: {file_size:.2f} MB")
    logger.info(f"   Rows: {len(df):,}")
    logger.info(f"   Columns: {len(df.columns)}")

    return output_path


def validate_data(df: pd.DataFrame, symbol: str):
    """Validate daily metadata."""
    logger.info(f"\n📊 Daily Metadata Quality Report for {symbol}")
    logger.info("=" * 60)

    # Show columns
    logger.info(f"\nFeatures ({len(df.columns)}):")
    for col in df.columns:
        logger.info(f"   - {col}")

    # Check for missing values
    logger.info(f"\n📋 Missing Values:")
    missing = df.isnull().sum()
    if missing.any():
        for col, count in missing[missing > 0].items():
            logger.warning(f"   {col}: {count} missing ({count/len(df)*100:.2f}%)")
    else:
        logger.info("   ✅ No missing values")

    # Date range
    logger.info(f"\n📅 Date Range:")
    logger.info(f"   Start: {df['timestamp'].min()}")
    logger.info(f"   End:   {df['timestamp'].max()}")
    logger.info(f"   Days:  {len(df)}")

    logger.info("=" * 60)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Fetch daily market metadata from CoinGecko'
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
        default='2023-11-15',
        help='Start date (YYYY-MM-DD, default: 2023-11-15)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD, default: today)'
    )

    args = parser.parse_args()

    # Verify API key
    if not COINGECKO_API_KEY:
        logger.error("❌ COINGECKO_API_KEY not found in environment")
        sys.exit(1)

    logger.info(f"🚀 CoinGecko Daily Metadata Fetcher - {args.symbol}")
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
        # Build metadata
        df = build_daily_metadata(coin_id, start_date, end_date)

        if df.empty:
            logger.error("No metadata fetched")
            sys.exit(1)

        # Validate
        validate_data(df, args.symbol)

        # Save
        output_path = save_to_parquet(df, args.symbol, start_date, end_date)

        logger.info(f"\n✅ SUCCESS!")
        logger.info(f"Daily metadata saved to: {output_path}")
        logger.info(f"\n💡 NOTE: Some metadata (rank, supply) uses current values")
        logger.info(f"   For true historical metadata, would need daily snapshots")

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
