#!/usr/bin/env python3
"""Test feature engineering with various datasets."""
import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from apps.trainer.data_pipeline import (
    detect_leakage,
    load_data,
    save_features_to_parquet,
)
from apps.trainer.features import engineer_features, normalize_features


def test_feature_engineering(input_file: str):
    """Test feature engineering on a dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing feature engineering: {input_file}")
    logger.info(f"{'='*60}")

    # Load raw data
    df_raw = load_data(input_file)
    logger.info(f"Loaded {len(df_raw)} rows of raw data")

    # Engineer features
    df_features = engineer_features(df_raw)
    logger.info(f"✅ Engineered {len(df_features.columns)} total columns")
    logger.info(f"   Features: {len(df_features.columns) - 6} (excluding OHLCV + timestamp)")

    # List feature categories
    feature_cols = [
        c
        for c in df_features.columns
        if c not in ["timestamp", "open", "high", "low", "close", "volume"]
    ]
    logger.info("\n📊 Feature Breakdown:")
    logger.info(f"   Session features: {len([c for c in feature_cols if 'session' in c])}")
    logger.info(
        f"   Time features: {len([c for c in feature_cols if 'day' in c or 'weekend' in c])}"
    )
    logger.info(f"   Spread features: {len([c for c in feature_cols if 'spread' in c])}")
    logger.info(f"   ATR features: {len([c for c in feature_cols if 'atr' in c.lower()])}")
    logger.info(f"   Volume features: {len([c for c in feature_cols if 'volume' in c])}")
    logger.info(f"   SMA features: {len([c for c in feature_cols if 'sma' in c.lower()])}")
    logger.info(f"   RSI features: {len([c for c in feature_cols if 'rsi' in c.lower()])}")
    logger.info(f"   MACD features: {len([c for c in feature_cols if 'macd' in c.lower()])}")
    logger.info(f"   Bollinger features: {len([c for c in feature_cols if 'bb_' in c])}")
    logger.info(f"   Volatility regime: {len([c for c in feature_cols if 'volatility' in c])}")

    # Check for NaN values
    nan_counts = df_features.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        logger.warning(f"⚠️  Found NaN values in {len(nan_cols)} columns:")
        for col, count in nan_cols.items():
            logger.warning(f"   {col}: {count} NaN values")
    else:
        logger.info("✅ No NaN values in features")

    # Test leakage detection
    logger.info("\n🔍 Testing for data leakage...")
    has_leakage = detect_leakage(df_features, feature_cols)
    if has_leakage:
        logger.error("❌ Data leakage detected!")
    else:
        logger.info("✅ No data leakage detected")

    # Test normalization
    logger.info("\n📏 Testing normalization...")
    df_norm, norm_params = normalize_features(df_features, method="standard")
    logger.info(f"✅ Normalized {len(norm_params)} features")
    logger.info("   Normalized data stats:")
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_norm[c])]
    if numeric_features:
        logger.info(f"   Mean: {df_norm[numeric_features[:5]].mean().values}")
        logger.info(f"   Std: {df_norm[numeric_features[:5]].std().values}")

    # Verify session features
    logger.info("\n🌍 Verifying session features...")
    sessions = df_features["session"].value_counts()
    logger.info("   Session distribution:")
    for session, count in sessions.items():
        logger.info(f"     {session}: {count} ({count/len(df_features)*100:.1f}%)")

    # Verify volatility regime
    logger.info("\n📈 Verifying volatility regime...")
    if "volatility_regime" in df_features.columns:
        regimes = df_features["volatility_regime"].value_counts()
        logger.info("   Volatility regime distribution:")
        for regime, count in regimes.items():
            logger.info(f"     {regime}: {count} ({count/len(df_features)*100:.1f}%)")

    logger.info("\n✅ Feature engineering test complete!")

    return df_features


def main():
    """Run feature engineering tests."""
    parser = argparse.ArgumentParser(description="Test feature engineering")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--save", action="store_true", help="Save engineered features")

    args = parser.parse_args()

    # Import pandas here to avoid circular import
    import pandas as pd

    globals()["pd"] = pd

    df_features = test_feature_engineering(args.input)

    if args.save:
        # Extract symbol and interval from filename
        input_path = Path(args.input)
        parts = input_path.stem.split("_")
        symbol = parts[1] if len(parts) > 1 else "BTC-USD"
        interval = parts[2] if len(parts) > 2 else "1h"

        output_file = save_features_to_parquet(df_features, symbol=symbol, interval=interval)
        logger.info(f"✅ Features saved to {output_file}")


if __name__ == "__main__":
    main()
