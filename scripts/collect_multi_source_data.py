#!/usr/bin/env python3
"""Collect historical data from multiple sources for V6 rebuild.

This script fetches OHLCV data from multiple exchanges (Kraken, Coinbase)
and compares data quality to select the best source for training.
"""

import argparse
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import time


def fetch_kraken_data(symbol: str, timeframe: str, since: datetime, limit: int = 720):
    """Fetch OHLCV data from Kraken.

    Args:
        symbol: Trading pair (e.g., 'BTC/USD')
        timeframe: Candle interval (1m, 5m, 15m, 1h)
        since: Start timestamp
        limit: Number of candles per request (max 720 for Kraken)

    Returns:
        DataFrame with OHLCV data
    """
    kraken = ccxt.kraken({
        'enableRateLimit': True,
        'timeout': 30000
    })

    all_ohlcv = []
    current_since = int(since.timestamp() * 1000)

    logger.info(f"Fetching {symbol} {timeframe} data from Kraken...")

    while True:
        try:
            ohlcv = kraken.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)

            # Update timestamp for next batch
            last_timestamp = ohlcv[-1][0]
            if last_timestamp <= current_since:
                break  # No more data

            current_since = last_timestamp + 1

            logger.info(f"  Fetched {len(all_ohlcv)} candles so far...")

            # Check if we've reached present time
            if last_timestamp >= datetime.now().timestamp() * 1000:
                break

            time.sleep(kraken.rateLimit / 1000)  # Respect rate limit

        except Exception as e:
            logger.error(f"Error fetching from Kraken: {e}")
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['source'] = 'kraken'

    logger.info(f"✅ Kraken: {len(df)} candles fetched")
    return df


def fetch_coinbase_data(symbol: str, timeframe: str, since: datetime, limit: int = 300):
    """Fetch OHLCV data from Coinbase.

    Args:
        symbol: Trading pair (e.g., 'BTC/USD')
        timeframe: Candle interval (1m, 5m, 15m, 1h)
        since: Start timestamp
        limit: Number of candles per request (max 300 for Coinbase)

    Returns:
        DataFrame with OHLCV data
    """
    coinbase = ccxt.coinbase({
        'enableRateLimit': True,
        'timeout': 30000
    })

    all_ohlcv = []
    current_since = int(since.timestamp() * 1000)

    logger.info(f"Fetching {symbol} {timeframe} data from Coinbase...")

    while True:
        try:
            ohlcv = coinbase.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)

            # Update timestamp for next batch
            last_timestamp = ohlcv[-1][0]
            if last_timestamp <= current_since:
                break  # No more data

            current_since = last_timestamp + 1

            logger.info(f"  Fetched {len(all_ohlcv)} candles so far...")

            # Check if we've reached present time
            if last_timestamp >= datetime.now().timestamp() * 1000:
                break

            time.sleep(coinbase.rateLimit / 1000)  # Respect rate limit

        except Exception as e:
            logger.error(f"Error fetching from Coinbase: {e}")
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['source'] = 'coinbase'

    logger.info(f"✅ Coinbase: {len(df)} candles fetched")
    return df


def compare_data_quality(df_kraken: pd.DataFrame, df_coinbase: pd.DataFrame) -> dict:
    """Compare data quality between sources.

    Returns:
        dict with quality metrics for each source
    """
    logger.info("Comparing data quality...")

    metrics = {}

    for name, df in [('kraken', df_kraken), ('coinbase', df_coinbase)]:
        metrics[name] = {
            'total_candles': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_timestamps': df['timestamp'].duplicated().sum(),
            'zero_volume_candles': (df['volume'] == 0).sum(),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'avg_volume': df['volume'].mean(),
            'price_range': f"${df['low'].min():.2f} - ${df['high'].max():.2f}"
        }

    # Print comparison
    logger.info("=" * 80)
    logger.info("Data Quality Comparison:")
    logger.info("=" * 80)

    for source, m in metrics.items():
        logger.info(f"\n{source.upper()}:")
        logger.info(f"  Total candles: {m['total_candles']:,}")
        logger.info(f"  Missing values: {m['missing_values']}")
        logger.info(f"  Duplicate timestamps: {m['duplicate_timestamps']}")
        logger.info(f"  Zero volume candles: {m['zero_volume_candles']}")
        logger.info(f"  Date range: {m['date_range']}")
        logger.info(f"  Avg volume: {m['avg_volume']:,.2f}")
        logger.info(f"  Price range: {m['price_range']}")

    # Determine winner
    kraken_score = metrics['kraken']['total_candles'] - (
        metrics['kraken']['missing_values'] * 10 +
        metrics['kraken']['duplicate_timestamps'] * 5 +
        metrics['kraken']['zero_volume_candles']
    )

    coinbase_score = metrics['coinbase']['total_candles'] - (
        metrics['coinbase']['missing_values'] * 10 +
        metrics['coinbase']['duplicate_timestamps'] * 5 +
        metrics['coinbase']['zero_volume_candles']
    )

    winner = 'kraken' if kraken_score > coinbase_score else 'coinbase'

    logger.info(f"\n🏆 Winner: {winner.upper()} (score: {max(kraken_score, coinbase_score):,})")
    logger.info("=" * 80)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Collect multi-source data for V6 rebuild")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading pair (default: BTC/USD)")
    parser.add_argument("--timeframe", default="1m", help="Candle interval (default: 1m)")
    parser.add_argument("--days", type=int, default=730, help="Days of historical data (default: 730 = 2 years)")
    parser.add_argument("--output-dir", default="data/multi_source", help="Output directory")
    parser.add_argument("--sources", nargs='+', default=['kraken', 'coinbase'],
                       help="Data sources to fetch from")

    args = parser.parse_args()

    # Calculate start date
    since = datetime.now() - timedelta(days=args.days)

    logger.info("=" * 80)
    logger.info("V6 Multi-Source Data Collection")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Date range: {since} to {datetime.now()}")
    logger.info(f"Sources: {', '.join(args.sources)}")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch from each source
    dataframes = {}

    if 'kraken' in args.sources:
        df_kraken = fetch_kraken_data(args.symbol, args.timeframe, since)
        dataframes['kraken'] = df_kraken

        # Save Kraken data
        symbol_clean = args.symbol.replace('/', '-')
        kraken_path = output_dir / f"{symbol_clean}_{args.timeframe}_kraken.parquet"
        df_kraken.to_parquet(kraken_path, index=False)
        logger.info(f"Saved: {kraken_path}")

    if 'coinbase' in args.sources:
        df_coinbase = fetch_coinbase_data(args.symbol, args.timeframe, since)
        dataframes['coinbase'] = df_coinbase

        # Save Coinbase data
        symbol_clean = args.symbol.replace('/', '-')
        coinbase_path = output_dir / f"{symbol_clean}_{args.timeframe}_coinbase.parquet"
        df_coinbase.to_parquet(coinbase_path, index=False)
        logger.info(f"Saved: {coinbase_path}")

    # Compare quality if we have multiple sources
    if len(dataframes) >= 2:
        compare_data_quality(dataframes['kraken'], dataframes['coinbase'])

    logger.info("\n✅ Multi-source data collection complete!")
    logger.info(f"Files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
