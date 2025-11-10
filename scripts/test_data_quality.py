#!/usr/bin/env python3
"""Comprehensive test suite for data quality checks."""
import sys
from datetime import timedelta
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.trainer.data_pipeline import (
    create_walk_forward_splits,
    load_data,
)
from apps.trainer.features import engineer_features
from libs.data.quality import (
    check_data_completeness,
    check_data_leakage,
    check_data_ranges,
    check_data_types,
    check_missing_values,
    validate_data_quality,
    validate_feature_quality,
)


def test_raw_data_quality():
    """Test raw data quality checks."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: Raw Data Quality Checks")
    logger.info("=" * 60)

    # Load test data
    df = load_data("data/raw/test_BTC-USD_1h_7d.parquet")
    logger.info(f"Loaded {len(df)} rows")

    # Run comprehensive validation
    report = validate_data_quality(
        df=df,
        interval="1h",
        check_leakage=False,  # Raw data doesn't have features yet
        check_completeness=True,
        check_missing=True,
        check_types=True,
        check_ranges=True,
    )

    logger.info(f"\n{report}")
    assert report.is_valid, "Raw data quality check failed!"
    logger.info("✅ Raw data quality checks PASSED")


def test_feature_quality():
    """Test feature quality checks."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Feature Quality Checks")
    logger.info("=" * 60)

    # Load and engineer features
    df_raw = load_data("data/raw/test_BTC-USD_1h_7d.parquet")
    df_features = engineer_features(df_raw)
    logger.info(f"Engineered {len(df_features.columns)} columns")

    # Run feature validation
    report = validate_feature_quality(df_features)
    logger.info(f"\n{report}")
    assert report.is_valid, "Feature quality check failed!"
    logger.info("✅ Feature quality checks PASSED")


def test_leakage_detection():
    """Test leakage detection with train/test split."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Leakage Detection with Train/Test Split")
    logger.info("=" * 60)

    # Load and engineer features
    df_raw = load_data("data/raw/test_BTC-USD_1h_7d.parquet")
    df_features = engineer_features(df_raw)

    # Create train/test split
    end_date = df_features["timestamp"].max()
    train_end = end_date - timedelta(days=2)
    val_end = train_end + timedelta(days=1)

    train_df, val_df, test_df = create_walk_forward_splits(df_features, train_end, val_end)

    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

    # Check for leakage in train set
    leakage_check_train = check_data_leakage(
        train_df, split_timestamp=train_end, timestamp_col="timestamp"
    )
    logger.info(f"Train leakage check: {leakage_check_train}")

    # Check for leakage in test set
    leakage_check_test = check_data_leakage(
        test_df, split_timestamp=val_end, timestamp_col="timestamp"
    )
    logger.info(f"Test leakage check: {leakage_check_test}")

    # Verify no temporal overlap
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Train and Val overlap!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Val and Test overlap!"

    logger.info("✅ Leakage detection checks PASSED")


def test_data_completeness():
    """Test data completeness checks."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Data Completeness Checks")
    logger.info("=" * 60)

    # Test with different intervals
    intervals = ["1m", "1h", "1d"]
    for interval in intervals:
        df = load_data("data/raw/test_BTC-USD_1h_7d.parquet")  # Use same data for all
        completeness_check = check_data_completeness(df, interval=interval)
        logger.info(f"{interval} interval: {completeness_check}")
        # For 1h interval, should pass
        if interval == "1h":
            assert completeness_check.passed, f"Completeness check failed for {interval}"


def test_missing_values():
    """Test missing value detection."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Missing Value Detection")
    logger.info("=" * 60)

    # Load features
    df_raw = load_data("data/raw/test_BTC-USD_1h_7d.parquet")
    df_features = engineer_features(df_raw)

    # Check for missing values
    missing_checks = check_missing_values(df_features)
    for check in missing_checks:
        logger.info(f"{check}")

    # All checks should pass (features are cleaned)
    assert all(c.passed for c in missing_checks), "Missing value checks failed!"
    logger.info("✅ Missing value checks PASSED")


def test_data_types():
    """Test data type validation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Data Type Validation")
    logger.info("=" * 60)

    df = load_data("data/raw/test_BTC-USD_1h_7d.parquet")
    type_checks = check_data_types(df)

    for check in type_checks:
        logger.info(f"{check}")

    assert all(c.passed for c in type_checks), "Data type checks failed!"
    logger.info("✅ Data type checks PASSED")


def test_data_ranges():
    """Test data range validation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 7: Data Range Validation")
    logger.info("=" * 60)

    df = load_data("data/raw/test_BTC-USD_1h_7d.parquet")
    range_checks = check_data_ranges(df)

    for check in range_checks:
        logger.info(f"{check}")

    assert all(c.passed for c in range_checks), "Data range checks failed!"
    logger.info("✅ Data range checks PASSED")


def main():
    """Run all data quality tests."""
    logger.info("Starting Data Quality Test Suite")
    logger.info("=" * 60)

    try:
        test_raw_data_quality()
        test_feature_quality()
        test_leakage_detection()
        test_data_completeness()
        test_missing_values()
        test_data_types()
        test_data_ranges()

        logger.info("\n" + "=" * 60)
        logger.info("✅ All Data Quality Tests PASSED!")
        logger.info("=" * 60)
        return 0

    except AssertionError as e:
        logger.error(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        logger.exception(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
