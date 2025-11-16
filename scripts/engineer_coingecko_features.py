#!/usr/bin/env python3
"""
Engineer strategic features from CoinGecko data.

Inputs:
- On-chain 1m data (active addresses, tx volume, etc.)
- Hourly OHLCV (1h trends)
- Daily metadata (market cap, dominance, etc.)

Outputs:
- Engineered features ready to merge with Coinbase tactical features

Usage:
    python scripts/engineer_coingecko_features.py --symbol BTC-USD
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger


def load_onchain_data(symbol: str) -> pd.DataFrame:
    """Load on-chain 1m data if available."""
    pattern = f"data/raw/coingecko_onchain/{symbol}_1m_onchain_*.parquet"
    files = list(Path('.').glob(pattern))

    if not files:
        logger.warning(f"No on-chain data found for {symbol}")
        return pd.DataFrame()

    logger.info(f"Loading on-chain data: {files[0]}")
    df = pd.read_parquet(files[0])

    logger.info(f"✅ Loaded {len(df):,} on-chain rows")
    return df


def load_hourly_data(symbol: str) -> pd.DataFrame:
    """Load hourly OHLCV data."""
    pattern = f"data/raw/coingecko_hourly/{symbol}_1h_*.parquet"
    files = list(Path('.').glob(pattern))

    if not files:
        logger.warning(f"No hourly data found for {symbol}")
        return pd.DataFrame()

    logger.info(f"Loading hourly data: {files[0]}")
    df = pd.read_parquet(files[0])

    logger.info(f"✅ Loaded {len(df):,} hourly rows")
    return df


def load_daily_metadata(symbol: str) -> pd.DataFrame:
    """Load daily metadata."""
    pattern = f"data/raw/coingecko_daily/{symbol}_daily_*.parquet"
    files = list(Path('.').glob(pattern))

    if not files:
        logger.warning(f"No daily metadata found for {symbol}")
        return pd.DataFrame()

    logger.info(f"Loading daily metadata: {files[0]}")
    df = pd.read_parquet(files[0])

    logger.info(f"✅ Loaded {len(df):,} daily rows")
    return df


def engineer_onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer on-chain features.

    Features:
    - Active address trends
    - Transaction volume patterns
    - Network activity spikes
    """
    if df.empty:
        return df

    logger.info("Engineering on-chain features...")

    # Active addresses features
    if 'active_addresses' in df.columns:
        df['active_addresses_1h_ma'] = df['active_addresses'].rolling(60).mean()
        df['active_addresses_change_pct'] = df['active_addresses'].pct_change() * 100
        df['active_addresses_spike'] = (df['active_addresses'] > df['active_addresses_1h_ma'] * 1.5).astype(int)

    # Transaction count features
    if 'transaction_count' in df.columns:
        df['tx_count_1h_ma'] = df['transaction_count'].rolling(60).mean()
        df['tx_count_change_pct'] = df['transaction_count'].pct_change() * 100
        df['tx_count_spike'] = (df['transaction_count'] > df['tx_count_1h_ma'] * 1.5).astype(int)

    # Transaction volume features
    if 'transaction_volume_usd' in df.columns:
        df['tx_volume_1h_ma'] = df['transaction_volume_usd'].rolling(60).mean()
        df['tx_volume_change_pct'] = df['transaction_volume_usd'].pct_change() * 100

    # Gas price features (ETH/SOL)
    if 'gas_price' in df.columns:
        df['gas_price_1h_ma'] = df['gas_price'].rolling(60).mean()
        df['gas_price_change_pct'] = df['gas_price'].pct_change() * 100

    # Hash rate features (BTC)
    if 'hash_rate' in df.columns:
        df['hash_rate_7d_ma'] = df['hash_rate'].rolling(10080).mean()  # 7 days * 24h * 60m
        df['hash_rate_change_pct'] = df['hash_rate'].pct_change() * 100

    # Drop NaN from rolling windows
    df = df.bfill().fillna(0)

    logger.info(f"✅ Engineered {len(df.columns)} on-chain features")
    return df


def engineer_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer hourly OHLCV features.

    Features:
    - 1h RSI, MACD, Bollinger Bands
    - 1h volume patterns
    - Price momentum
    """
    if df.empty:
        return df

    logger.info("Engineering hourly features...")

    # RSI (14-period on hourly)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_1h'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9 on hourly)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_1h'] = ema12 - ema26
    df['macd_signal_1h'] = df['macd_1h'].ewm(span=9, adjust=False).mean()
    df['macd_hist_1h'] = df['macd_1h'] - df['macd_signal_1h']

    # Bollinger Bands (20-period on hourly)
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper_1h'] = sma20 + (std20 * 2)
    df['bb_lower_1h'] = sma20 - (std20 * 2)
    df['bb_position_1h'] = (df['close'] - df['bb_lower_1h']) / (df['bb_upper_1h'] - df['bb_lower_1h'])

    # Volume features
    df['volume_1h_ma'] = df['volume'].rolling(24).mean()  # 24-hour MA
    df['volume_1h_ratio'] = df['volume'] / df['volume_1h_ma']

    # Price momentum
    df['price_1h_change_pct'] = df['close'].pct_change() * 100
    df['price_4h_change_pct'] = df['close'].pct_change(4) * 100
    df['price_24h_change_pct'] = df['close'].pct_change(24) * 100

    # Drop NaN
    df = df.bfill().fillna(0)

    logger.info(f"✅ Engineered {len(df.columns)} hourly features")
    return df


def engineer_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer daily metadata features.

    Features:
    - Market cap trends
    - Dominance indicators
    - Supply metrics
    - Social sentiment
    """
    if df.empty:
        return df

    logger.info("Engineering daily features...")

    # Market cap features
    if 'market_cap' in df.columns:
        df['market_cap_7d_ma'] = df['market_cap'].rolling(7).mean()
        df['market_cap_30d_ma'] = df['market_cap'].rolling(30).mean()
        df['market_cap_change_7d_pct'] = df['market_cap'].pct_change(7) * 100
        df['market_cap_trend'] = np.where(df['market_cap'] > df['market_cap_7d_ma'], 1, -1)

    # Volume features
    if 'total_volume' in df.columns:
        df['volume_7d_ma'] = df['total_volume'].rolling(7).mean()
        df['volume_change_7d_pct'] = df['total_volume'].pct_change(7) * 100

    # BTC dominance features
    if 'btc_dominance' in df.columns:
        df['btc_dominance_change_pct'] = df['btc_dominance'].pct_change() * 100
        df['btc_dominance_7d_ma'] = df['btc_dominance'].rolling(7).mean()
        df['btc_dominance_trend'] = np.where(df['btc_dominance'] > df['btc_dominance_7d_ma'], 1, -1)

    # ETH dominance features
    if 'eth_dominance' in df.columns:
        df['eth_dominance_change_pct'] = df['eth_dominance'].pct_change() * 100
        df['eth_dominance_7d_ma'] = df['eth_dominance'].rolling(7).mean()

    # Cross-exchange spread features
    if 'cross_exchange_spread_pct' in df.columns:
        df['spread_7d_ma'] = df['cross_exchange_spread_pct'].rolling(7).mean()
        df['spread_volatility'] = df['cross_exchange_spread_pct'].rolling(7).std()

    # Rank changes
    if 'market_cap_rank' in df.columns:
        df['rank_change'] = df['market_cap_rank'].diff()

    # Social metrics (if available)
    if 'twitter_followers' in df.columns:
        df['twitter_change_pct'] = df['twitter_followers'].pct_change() * 100

    if 'reddit_subscribers' in df.columns:
        df['reddit_change_pct'] = df['reddit_subscribers'].pct_change() * 100

    # Drop NaN
    df = df.bfill().fillna(0)

    logger.info(f"✅ Engineered {len(df.columns)} daily features")
    return df


def save_features(df: pd.DataFrame, symbol: str, data_type: str):
    """Save engineered features."""
    output_dir = Path('data/features/coingecko')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{symbol}_{data_type}_features.parquet'
    output_path = output_dir / filename

    df.to_parquet(output_path, index=False, compression='gzip')

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved: {output_path}")
    logger.info(f"   Size: {file_size:.2f} MB")
    logger.info(f"   Rows: {len(df):,}")
    logger.info(f"   Features: {len(df.columns)}")

    return output_path


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Engineer features from CoinGecko data'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        choices=['BTC-USD', 'ETH-USD', 'SOL-USD'],
        help='Trading pair symbol'
    )

    args = parser.parse_args()

    logger.info(f"🚀 CoinGecko Feature Engineering - {args.symbol}")
    logger.info("=" * 60)

    try:
        # Load and engineer on-chain features
        logger.info("\n📊 Processing On-Chain Data...")
        onchain_df = load_onchain_data(args.symbol)

        if not onchain_df.empty:
            onchain_df = engineer_onchain_features(onchain_df)
            save_features(onchain_df, args.symbol, 'onchain_1m')
        else:
            logger.warning("⚠️  Skipping on-chain features (no data)")

        # Load and engineer hourly features
        logger.info("\n📊 Processing Hourly Data...")
        hourly_df = load_hourly_data(args.symbol)

        if not hourly_df.empty:
            hourly_df = engineer_hourly_features(hourly_df)
            save_features(hourly_df, args.symbol, 'hourly')
        else:
            logger.warning("⚠️  Skipping hourly features (no data)")

        # Load and engineer daily features
        logger.info("\n📊 Processing Daily Metadata...")
        daily_df = load_daily_metadata(args.symbol)

        if not daily_df.empty:
            daily_df = engineer_daily_features(daily_df)
            save_features(daily_df, args.symbol, 'daily')
        else:
            logger.warning("⚠️  Skipping daily features (no data)")

        logger.info("\n✅ SUCCESS!")
        logger.info("=" * 60)
        logger.info(f"CoinGecko features engineered for {args.symbol}")
        logger.info("\nNext step: Merge with Coinbase tactical features")

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
