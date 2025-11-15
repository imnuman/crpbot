#!/usr/bin/env python3
"""
Merge all feature sources into ultimate hybrid dataset.

Combines:
1. Coinbase 1m tactical features (30-35 features)
2. CoinGecko on-chain 1m features (10-15 features)
3. CoinGecko hourly features (10-12 features, resampled to 1m)
4. CoinGecko daily features (15-20 features, resampled to 1m)

Output:
- Hybrid dataset at 1-minute granularity with 65-82 features
- Ready for model training

Usage:
    python scripts/merge_hybrid_features.py --symbol BTC-USD
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger


def load_coinbase_features(symbol: str) -> pd.DataFrame:
    """Load Coinbase tactical features (1m granularity)."""
    pattern = f"data/features/features_{symbol}_1m_*.parquet"
    files = list(Path('.').glob(pattern))

    if not files:
        logger.error(f"No Coinbase features found for {symbol}")
        logger.error(f"Run: python scripts/engineer_features.py --symbol {symbol}")
        return pd.DataFrame()

    logger.info(f"Loading Coinbase features: {files[0]}")
    df = pd.read_parquet(files[0])

    logger.info(f"✅ Coinbase: {len(df):,} rows, {len(df.columns)} features")
    return df


def load_coingecko_onchain(symbol: str) -> pd.DataFrame:
    """Load CoinGecko on-chain features (1m granularity)."""
    pattern = f"data/features/coingecko/{symbol}_onchain_1m_features.parquet"
    files = list(Path('.').glob(pattern))

    if not files:
        logger.warning(f"No on-chain features found for {symbol}")
        return pd.DataFrame()

    logger.info(f"Loading on-chain features: {files[0]}")
    df = pd.read_parquet(files[0])

    logger.info(f"✅ On-chain: {len(df):,} rows, {len(df.columns)} features")
    return df


def load_coingecko_hourly(symbol: str) -> pd.DataFrame:
    """Load CoinGecko hourly features."""
    pattern = f"data/features/coingecko/{symbol}_hourly_features.parquet"
    files = list(Path('.').glob(pattern))

    if not files:
        logger.warning(f"No hourly features found for {symbol}")
        return pd.DataFrame()

    logger.info(f"Loading hourly features: {files[0]}")
    df = pd.read_parquet(files[0])

    logger.info(f"✅ Hourly: {len(df):,} rows, {len(df.columns)} features")
    return df


def load_coingecko_daily(symbol: str) -> pd.DataFrame:
    """Load CoinGecko daily features."""
    pattern = f"data/features/coingecko/{symbol}_daily_features.parquet"
    files = list(Path('.').glob(pattern))

    if not files:
        logger.warning(f"No daily features found for {symbol}")
        return pd.DataFrame()

    logger.info(f"Loading daily features: {files[0]}")
    df = pd.read_parquet(files[0])

    logger.info(f"✅ Daily: {len(df):,} rows, {len(df.columns)} features")
    return df


def merge_onchain_1m(base_df: pd.DataFrame, onchain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge on-chain 1m features with base 1m data.

    Both are at 1m granularity, so this is a straightforward merge.
    """
    if onchain_df.empty:
        logger.warning("Skipping on-chain merge (no data)")
        return base_df

    logger.info("Merging on-chain 1m features...")

    # Ensure timestamp column exists
    if 'timestamp' not in onchain_df.columns:
        logger.error("On-chain data missing 'timestamp' column")
        return base_df

    # Select only feature columns (exclude raw OHLCV)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'active_addresses', 'transaction_count', 'transaction_volume_usd',
                    'gas_price', 'hash_rate']

    feature_cols = [col for col in onchain_df.columns if col not in exclude_cols]

    if not feature_cols:
        logger.warning("No engineered on-chain features found")
        return base_df

    # Merge on timestamp
    merged = base_df.merge(
        onchain_df[['timestamp'] + feature_cols],
        on='timestamp',
        how='left',
        suffixes=('', '_onchain')
    )

    # Forward-fill missing values
    merged[feature_cols] = merged[feature_cols].fillna(method='ffill')

    logger.info(f"✅ Added {len(feature_cols)} on-chain features")
    return merged


def merge_hourly(base_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge hourly features with base 1m data.

    Hourly data needs to be downsampled to 1m (forward-fill).
    """
    if hourly_df.empty:
        logger.warning("Skipping hourly merge (no data)")
        return base_df

    logger.info("Merging hourly features (downsampling to 1m)...")

    # Ensure timestamp column
    if 'timestamp' not in hourly_df.columns:
        logger.error("Hourly data missing 'timestamp' column")
        return base_df

    # Select feature columns (exclude raw OHLCV)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'price']
    feature_cols = [col for col in hourly_df.columns if col not in exclude_cols]

    if not feature_cols:
        logger.warning("No engineered hourly features found")
        return base_df

    # Round base_df timestamps to hour for merging
    base_df['hour'] = base_df['timestamp'].dt.floor('H')

    # Round hourly_df timestamps to hour
    hourly_df['hour'] = hourly_df['timestamp'].dt.floor('H')

    # Merge on hour
    merged = base_df.merge(
        hourly_df[['hour'] + feature_cols],
        on='hour',
        how='left',
        suffixes=('', '_hourly')
    )

    # Drop temporary hour column
    merged = merged.drop('hour', axis=1)

    # Forward-fill missing values (for any gaps)
    merged[feature_cols] = merged[feature_cols].fillna(method='ffill')

    logger.info(f"✅ Added {len(feature_cols)} hourly features (forward-filled to 1m)")
    return merged


def merge_daily(base_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily features with base 1m data.

    Daily data needs to be downsampled to 1m (forward-fill).
    """
    if daily_df.empty:
        logger.warning("Skipping daily merge (no data)")
        return base_df

    logger.info("Merging daily features (downsampling to 1m)...")

    # Ensure timestamp column
    if 'timestamp' not in daily_df.columns:
        # Try 'date' column
        if 'date' in daily_df.columns:
            daily_df['timestamp'] = pd.to_datetime(daily_df['date'], utc=True)
        else:
            logger.error("Daily data missing 'timestamp' or 'date' column")
            return base_df

    # Select feature columns (exclude raw market data)
    exclude_cols = ['timestamp', 'date', 'price', 'market_cap', 'total_volume']
    feature_cols = [col for col in daily_df.columns if col not in exclude_cols]

    if not feature_cols:
        logger.warning("No engineered daily features found")
        return base_df

    # Round base_df timestamps to day
    base_df['date'] = base_df['timestamp'].dt.floor('D')

    # Round daily_df timestamps to day
    daily_df['date_key'] = daily_df['timestamp'].dt.floor('D')

    # Merge on date
    merged = base_df.merge(
        daily_df[['date_key'] + feature_cols],
        left_on='date',
        right_on='date_key',
        how='left',
        suffixes=('', '_daily')
    )

    # Drop temporary columns
    merged = merged.drop(['date', 'date_key'], axis=1)

    # Forward-fill missing values
    merged[feature_cols] = merged[feature_cols].fillna(method='ffill')

    logger.info(f"✅ Added {len(feature_cols)} daily features (forward-filled to 1m)")
    return merged


def save_hybrid_dataset(df: pd.DataFrame, symbol: str):
    """Save final hybrid dataset."""
    output_dir = Path('data/features/hybrid')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{symbol}_hybrid_features.parquet'
    output_path = output_dir / filename

    df.to_parquet(output_path, index=False, compression='gzip')

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"\n✅ HYBRID DATASET SAVED!")
    logger.info(f"   Path: {output_path}")
    logger.info(f"   Size: {file_size:.2f} MB")
    logger.info(f"   Rows: {len(df):,}")
    logger.info(f"   Features: {len(df.columns)}")
    logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return output_path


def validate_hybrid(df: pd.DataFrame, symbol: str):
    """Validate hybrid dataset."""
    logger.info(f"\n📊 Hybrid Dataset Validation - {symbol}")
    logger.info("=" * 60)

    # Check missing values
    logger.info("\n🔍 Missing Values Check:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    critical_missing = missing_pct[missing_pct > 10]  # >10% missing

    if not critical_missing.empty:
        logger.warning(f"⚠️  Critical missing values (>10%):")
        for col, pct in critical_missing.items():
            logger.warning(f"   {col}: {pct:.2f}%")
    else:
        logger.info("✅ No critical missing values (<10% for all features)")

    # Check feature groups
    logger.info(f"\n📋 Feature Groups:")

    # Count features by type (based on column names)
    tactical = [c for c in df.columns if not any(x in c for x in ['_1h', '_daily', '_onchain', 'timestamp'])]
    onchain = [c for c in df.columns if '_onchain' in c or any(x in c for x in ['active_addresses', 'tx_count', 'hash_rate'])]
    hourly = [c for c in df.columns if '_1h' in c]
    daily = [c for c in df.columns if any(x in c for x in ['_daily', 'btc_dominance', 'market_cap', 'rank'])]

    logger.info(f"   Tactical (Coinbase 1m): {len(tactical)} features")
    logger.info(f"   On-Chain (1m): {len(onchain)} features")
    logger.info(f"   Hourly (1h → 1m): {len(hourly)} features")
    logger.info(f"   Daily (daily → 1m): {len(daily)} features")
    logger.info(f"   TOTAL: {len(df.columns)} features")

    # Check data quality
    logger.info(f"\n✅ Quality Checks:")
    logger.info(f"   Rows: {len(df):,}")
    logger.info(f"   Date range: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    logger.info("=" * 60)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Merge all features into hybrid dataset'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        choices=['BTC-USD', 'ETH-USD', 'SOL-USD'],
        help='Trading pair symbol'
    )

    args = parser.parse_args()

    logger.info(f"🚀 Hybrid Feature Merger - {args.symbol}")
    logger.info("=" * 60)
    logger.info("\nThis will create the ULTIMATE V5 hybrid dataset:")
    logger.info("  - Coinbase 1m tactical features (30-35)")
    logger.info("  - CoinGecko on-chain 1m features (10-15)")
    logger.info("  - CoinGecko hourly features (10-12)")
    logger.info("  - CoinGecko daily features (15-20)")
    logger.info("  = 65-82 total features at 1-minute granularity")
    logger.info("=" * 60)

    try:
        # Load base data (Coinbase 1m tactical)
        logger.info("\n📊 Step 1: Loading Coinbase tactical features...")
        base_df = load_coinbase_features(args.symbol)

        if base_df.empty:
            logger.error("Cannot proceed without Coinbase features")
            sys.exit(1)

        # Load CoinGecko features
        logger.info("\n📊 Step 2: Loading CoinGecko features...")
        onchain_df = load_coingecko_onchain(args.symbol)
        hourly_df = load_coingecko_hourly(args.symbol)
        daily_df = load_coingecko_daily(args.symbol)

        # Merge on-chain 1m
        logger.info("\n📊 Step 3: Merging on-chain features...")
        hybrid_df = merge_onchain_1m(base_df, onchain_df)

        # Merge hourly
        logger.info("\n📊 Step 4: Merging hourly features...")
        hybrid_df = merge_hourly(hybrid_df, hourly_df)

        # Merge daily
        logger.info("\n📊 Step 5: Merging daily features...")
        hybrid_df = merge_daily(hybrid_df, daily_df)

        # Validate
        logger.info("\n📊 Step 6: Validating hybrid dataset...")
        validate_hybrid(hybrid_df, args.symbol)

        # Save
        logger.info("\n📊 Step 7: Saving hybrid dataset...")
        output_path = save_hybrid_dataset(hybrid_df, args.symbol)

        logger.info("\n" + "=" * 60)
        logger.info("✅ SUCCESS! Hybrid dataset created!")
        logger.info("=" * 60)
        logger.info(f"\nOutput: {output_path}")
        logger.info(f"\nNext step: Train models with this hybrid dataset")
        logger.info(f"  uv run python apps/trainer/main.py --task lstm --coin {args.symbol.split('-')[0]}")

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
