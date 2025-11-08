"""Leakage tests for trading dataset."""

import numpy as np
import pandas as pd
import pytest

from apps.trainer.train.dataset import TradingDataset


def test_trading_dataset_direction_labels_without_leakage():
    """Labels should reflect future movement while features remain historical."""
    timestamps = pd.date_range("2025-01-01", periods=120, freq="min")
    close = np.linspace(100.0, 120.0, len(timestamps))
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": close,
            "feature1": np.arange(len(timestamps), dtype=float),
        }
    )

    dataset = TradingDataset(
        df=df,
        feature_columns=["feature1", "close"],
        sequence_length=10,
        horizon=5,
        prediction_type="direction",
    )

    sample = dataset[0]
    features = sample["features"].numpy()
    label = sample["label"].item()

    # Features should correspond to historical window ending at index 9
    assert features.shape == (10, 2)
    assert np.isclose(features[-1, 1], df.loc[9, "close"])

    # Monotonic price increase -> label should predict upward movement (1.0)
    assert label == pytest.approx(1.0)

    # Ensure dataset length respects horizon (no sequences with NaN labels)
    assert len(dataset) == len(df) - dataset.sequence_length - dataset.horizon + 1
