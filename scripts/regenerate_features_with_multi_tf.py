"""Regenerate source features with multi-timeframe data for model training.

This script recreates the feature files in data/features/ with the full multi-TF
pipeline that matches what the runtime uses, ensuring training and inference alignment.
"""
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from apps.runtime.data_fetcher import MarketDataFetcher
from apps.runtime.runtime_features import engineer_runtime_features
from libs.config.config import load_settings


def regenerate_features_for_symbol(
    symbol: str,
    output_dir: Path = Path("data/features"),
    num_candles: int = 1_100_000
):
    """
    Regenerate features for a symbol with multi-TF pipeline.

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        output_dir: Directory to save feature files
        num_candles: Number of 1m candles to fetch
    """
    logger.info(f"{'='*60}")
    logger.info(f"Regenerating features for {symbol}")
    logger.info(f"{'='*60}")

    # Initialize data fetcher
    settings = load_settings()
    data_fetcher = MarketDataFetcher(settings)

    # Fetch historical data (1m candles)
    logger.info(f"Fetching {num_candles:,} 1m candles for {symbol}...")
    try:
        df_raw = data_fetcher.fetch_latest_candles(symbol, num_candles)
        logger.info(f"✅ Fetched {len(df_raw):,} candles")
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return False

    if len(df_raw) < 200:
        logger.error(f"Insufficient data: {len(df_raw)} rows")
        return False

    # Engineer features using runtime pipeline (includes multi-TF)
    logger.info("Engineering features with multi-TF pipeline...")
    try:
        df_features = engineer_runtime_features(
            df=df_raw,
            symbol=symbol,
            data_fetcher=data_fetcher,
            include_multi_tf=True,
            include_coingecko=False  # Use placeholders for now
        )
        logger.info(f"✅ Engineered {len(df_features):,} rows with {len(df_features.columns)} columns")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify feature count
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'session', 'volatility_regime']
    feature_cols = [c for c in df_features.columns if c not in exclude]
    numeric_features = [c for c in feature_cols if df_features[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    logger.info(f"Feature summary:")
    logger.info(f"  Total columns: {len(df_features.columns)}")
    logger.info(f"  After exclusions: {len(feature_cols)}")
    logger.info(f"  Numeric features: {len(numeric_features)}")

    # Expected feature counts (from training splits)
    expected = {
        'BTC-USD': 73,
        'ETH-USD': 54,  # ETH doesn't have multi-TF
        'SOL-USD': 73
    }

    if symbol in expected and len(numeric_features) != expected[symbol]:
        logger.warning(f"⚠️  Feature count mismatch: expected {expected[symbol]}, got {len(numeric_features)}")
        logger.warning(f"Columns: {sorted(df_features.columns.tolist())}")

    # Save to parquet
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dated filename
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"features_{symbol}_1m_{date_str}.parquet"

    logger.info(f"Saving to {output_file}...")
    df_features.to_parquet(output_file, index=False)

    # Update symlink to latest
    latest_symlink = output_dir / f"features_{symbol}_1m_latest.parquet"
    if latest_symlink.exists() or latest_symlink.is_symlink():
        latest_symlink.unlink()
    latest_symlink.symlink_to(output_file.name)

    logger.success(f"✅ Saved {len(df_features):,} rows to {output_file.name}")
    logger.success(f"✅ Updated symlink: {latest_symlink.name} -> {output_file.name}")

    return True


def main():
    """Main entry point."""
    logger.info("Starting feature regeneration with multi-TF pipeline...")
    logger.info("This will recreate source features to match runtime pipeline")
    logger.info("")

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    for symbol in symbols:
        success = regenerate_features_for_symbol(
            symbol=symbol,
            num_candles=1_100_000  # ~2 years of 1m data
        )

        if not success:
            logger.error(f"Failed to regenerate features for {symbol}")
            sys.exit(1)

        logger.info("")

    logger.success("✅ All features regenerated successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Verify feature counts match expectations (BTC/SOL: 73, ETH: 54)")
    logger.info("2. Create training splits: uv run python scripts/create_training_splits.py")
    logger.info("3. Retrain models: uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15")


if __name__ == "__main__":
    main()
