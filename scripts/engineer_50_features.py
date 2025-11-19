#!/usr/bin/env python3
"""Engineer 50-feature datasets with multi-timeframe features.

This script combines:
- 31 base features (from engineer_features.py)
- 19 multi-TF features:
  - 15 higher TF OHLCV (5m, 15m, 1h × 5 columns)
  - 3 cross-TF alignment (score, direction, strength)
  - 1 volatility (atr_percentile)
= 50 total features
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from apps.trainer.features import engineer_features
from apps.trainer.multi_tf_features import (
    calculate_cross_tf_alignment,
    calculate_volatility_regime_features,
    load_multi_tf_data,
    resample_to_base_tf,
)


def engineer_50_features(
    symbol: str,
    intervals: list[str] = ["1m", "5m", "15m", "1h"],
    data_dir: Path | str = "data/raw",
    output_dir: Path | str = "data/features",
    start_date: str = "2023-11-10",
) -> pd.DataFrame:
    """
    Engineer 50 features combining base + multi-TF features.

    Args:
        symbol: Trading pair (e.g., 'BTC-USD')
        intervals: List of timeframes (default: ['1m', '5m', '15m', '1h'])
        data_dir: Directory with raw data files
        output_dir: Directory to save feature files
        start_date: Start date for data

    Returns:
        DataFrame with 50 features engineered
    """
    logger.info(f"Engineering 50 features for {symbol}")

    # Step 1: Load multi-TF data
    logger.info("Step 1/5: Loading multi-timeframe data...")
    multi_tf_data = load_multi_tf_data(symbol, intervals, data_dir, start_date)

    if "1m" not in multi_tf_data:
        raise ValueError("Base timeframe '1m' not found in loaded data")

    base_df = multi_tf_data["1m"].copy()
    logger.info(f"  Base (1m): {len(base_df):,} rows")

    # Step 2: Engineer base 31 features
    logger.info("Step 2/5: Engineering base 31 features...")
    base_df = engineer_features(base_df)
    logger.info(f"  Base features: {len([c for c in base_df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'session', 'volatility_regime']])} features")

    # Step 3: Resample higher TF data to base
    logger.info("Step 3/5: Resampling higher timeframe data to 1m...")
    for interval in intervals:
        if interval == "1m":
            continue  # Skip base timeframe

        if interval in multi_tf_data:
            logger.info(f"  Resampling {interval} → 1m...")
            base_df = resample_to_base_tf(
                base_df,
                multi_tf_data[interval],
                interval,
                feature_prefix=f"{interval}_",
            )

    # Step 4: Calculate cross-TF alignment
    logger.info("Step 4/5: Calculating cross-timeframe alignment...")
    base_df = calculate_cross_tf_alignment(base_df, intervals)

    # Step 5: Calculate volatility regime features
    logger.info("Step 5/5: Calculating volatility regime features...")
    base_df = calculate_volatility_regime_features(base_df)

    # Count final features
    exclude_cols = ["timestamp", "open", "high", "low", "close", "volume", "session", "volatility_regime"]
    feature_cols = [col for col in base_df.columns if col not in exclude_cols]

    logger.info(f"✅ Feature engineering complete!")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Total rows: {len(base_df):,}")
    logger.info(f"  Columns: {len(base_df.columns)}")

    # Verify 50 features
    if len(feature_cols) != 50:
        logger.warning(f"⚠️ Expected 50 features, got {len(feature_cols)}")
        logger.warning(f"Features: {sorted(feature_cols)}")
    else:
        logger.info(f"✅ Verified: Exactly 50 features!")

    # Save to parquet
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_suffix = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"features_{symbol}_1m_{date_suffix}_50feat.parquet"

    logger.info(f"Saving to {output_file}...")
    base_df.to_parquet(output_file, index=False)
    logger.info(f"✅ Saved: {output_file} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # Create "latest" symlink
    latest_file = output_dir / f"features_{symbol}_1m_latest.parquet"
    if latest_file.exists() or latest_file.is_symlink():
        latest_file.unlink()
    latest_file.symlink_to(output_file.name)
    logger.info(f"✅ Symlink: {latest_file} -> {output_file.name}")

    return base_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Engineer 50-feature datasets with multi-TF features")
    parser.add_argument("--symbol", required=True, help="Trading pair (e.g., BTC-USD)")
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["1m", "5m", "15m", "1h"],
        help="Timeframes to use (default: 1m 5m 15m 1h)",
    )
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/features", help="Output directory")
    parser.add_argument("--start-date", default="2023-11-10", help="Start date (YYYY-MM-DD)")

    args = parser.parse_args()

    try:
        df = engineer_50_features(
            symbol=args.symbol,
            intervals=args.intervals,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            start_date=args.start_date,
        )

        logger.info(f"✅ SUCCESS: {args.symbol} 50-feature dataset ready!")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
