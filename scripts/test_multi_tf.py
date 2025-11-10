"""Quick test of multi-TF feature engineering module."""
from pathlib import Path

from loguru import logger

from apps.trainer.multi_tf_features import (
    calculate_cross_tf_alignment,
    calculate_volatility_regime_features,
    engineer_multi_tf_features,
    load_multi_tf_data,
)


def test_load_multi_tf_data():
    """Test loading multi-TF data (with graceful handling of missing timeframes)."""
    logger.info("Testing load_multi_tf_data()...")

    try:
        # Try to load with all timeframes (some may be missing)
        data = load_multi_tf_data(
            symbol="BTC-USD",
            intervals=["1m", "5m", "15m", "1h"],
            data_dir="data/raw",
            start_date="2023-11-10",
        )

        logger.info(f"✅ Loaded {len(data)} timeframes: {list(data.keys())}")

        for interval, df in data.items():
            logger.info(f"  {interval}: {len(df)} rows, {list(df.columns)[:10]}...")

        return data

    except ValueError as e:
        logger.warning(f"Expected error (missing TFs): {e}")
        return None


def test_volatility_regime():
    """Test volatility regime calculation on 1m data."""
    logger.info("Testing calculate_volatility_regime_features()...")

    try:
        data = load_multi_tf_data(
            symbol="BTC-USD",
            intervals=["1m"],
            data_dir="data/raw",
            start_date="2023-11-10",
        )

        df = data["1m"]
        df_with_vol = calculate_volatility_regime_features(df)

        vol_cols = [c for c in df_with_vol.columns if "volatility" in c]
        logger.info(f"✅ Added volatility features: {vol_cols}")

        # Show distribution
        logger.info(f"  Low: {df_with_vol['volatility_low'].sum()}")
        logger.info(f"  Medium: {df_with_vol['volatility_medium'].sum()}")
        logger.info(f"  High: {df_with_vol['volatility_high'].sum()}")

    except Exception as e:
        logger.error(f"❌ Volatility regime test failed: {e}")


def test_alignment_single_tf():
    """Test alignment calculation with single TF (should handle gracefully)."""
    logger.info("Testing calculate_cross_tf_alignment() with single TF...")

    try:
        data = load_multi_tf_data(
            symbol="BTC-USD",
            intervals=["1m"],
            data_dir="data/raw",
            start_date="2023-11-10",
        )

        df = data["1m"]
        df_with_align = calculate_cross_tf_alignment(df, intervals=["1m"])

        align_cols = [c for c in df_with_align.columns if "tf_alignment" in c]
        logger.info(f"✅ Added alignment features: {align_cols}")

        # Should have neutral values (0.5) since only 1 TF
        logger.info(f"  Mean alignment: {df_with_align['tf_alignment_score'].mean():.3f}")

    except Exception as e:
        logger.error(f"❌ Alignment test failed: {e}")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Multi-TF Feature Engineering Module Tests")
    logger.info("=" * 60)

    # Test 1: Load data
    data = test_load_multi_tf_data()
    logger.info("")

    # Test 2: Volatility regime (should work with 1m only)
    test_volatility_regime()
    logger.info("")

    # Test 3: Alignment (should handle single TF gracefully)
    test_alignment_single_tf()
    logger.info("")

    logger.info("=" * 60)
    logger.info("Tests complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
