"""Tests for data pipeline."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from apps.trainer.data_pipeline import (
    clean_and_validate_data,
    create_walk_forward_splits,
    detect_outliers,
    interval_map,
)


def test_clean_and_validate_data_basic():
    """Test that clean_and_validate_data processes data correctly."""
    # Create clean test data
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="1min", tz=timezone.utc),
        "open": [100.0 + i for i in range(10)],
        "high": [101.0 + i for i in range(10)],
        "low": [99.0 + i for i in range(10)],
        "close": [100.5 + i for i in range(10)],
        "volume": [1000 + i * 100 for i in range(10)],
    })

    # clean_and_validate_data returns (DataFrame, DataQualityReport)
    cleaned, report = clean_and_validate_data(df)

    # Should return a DataFrame
    assert isinstance(cleaned, pd.DataFrame)
    assert len(cleaned) > 0
    assert "timestamp" in cleaned.columns
    assert "close" in cleaned.columns

    # Should not have NaN values
    assert not cleaned.isnull().any().any()


def test_create_walk_forward_splits_basic():
    """Test walk-forward splits with basic configuration."""
    # Create test data with required columns
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            start="2023-01-01", periods=100, freq="1h", tz=timezone.utc
        ),
        "close": np.random.randn(100).cumsum() + 100,
    })

    # Define split points (2 days and 3 days into the data)
    train_end = datetime(2023, 1, 3, 0, 0, tzinfo=timezone.utc)
    val_end = datetime(2023, 1, 4, 0, 0, tzinfo=timezone.utc)

    # create_walk_forward_splits returns (train, val, test) DataFrames
    train, val, test = create_walk_forward_splits(df, train_end, val_end)

    # Should have all three splits
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    # All should be non-empty
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_interval_mapping():
    """Test that interval mapping constants are defined."""
    # Check that common intervals are mapped
    assert "1m" in interval_map
    assert "5m" in interval_map
    assert "15m" in interval_map
    assert "1h" in interval_map
    assert "4h" in interval_map
    assert "1d" in interval_map

    # Check values are in pandas-compatible format
    assert interval_map["1m"] == "1min"
    assert interval_map["1h"] == "1h"


