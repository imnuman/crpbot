"""Dataset classes for model training."""
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from loguru import logger


class TradingDataset(Dataset):
    """
    Dataset for trading model training.

    Creates sequences of features and labels for time series prediction.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str | None = None,
        sequence_length: int = 60,
        horizon: int = 15,
        prediction_type: str = "direction",
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with features and labels
            feature_columns: List of feature column names to use
            label_column: Column name for labels (if None, creates from price data)
            sequence_length: Number of time steps to use as input
            horizon: Number of time steps ahead to predict (default: 15 for 15-min horizon)
            prediction_type: Type of prediction ('direction' or 'trend')
        """
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.prediction_type = prediction_type

        # Extract features
        if not all(col in df.columns for col in feature_columns):
            missing = [col for col in feature_columns if col not in df.columns]
            raise ValueError(f"Missing feature columns: {missing}")

        self.features = df[feature_columns].values.astype(np.float32)

        # Create labels
        if label_column and label_column in df.columns:
            self.labels = df[label_column].values
        else:
            # Create labels from price data
            self.labels = self._create_labels()

        # Remove sequences that would go beyond data boundaries
        self.valid_indices = self._get_valid_indices()

        logger.info(
            f"Created dataset: {len(self.valid_indices)} sequences, "
            f"{len(feature_columns)} features, prediction_type={prediction_type}"
        )

    def _create_labels(self) -> np.ndarray:
        """
        Create labels from price data based on prediction type.

        Returns:
            Array of labels
        """
        if "close" not in self.df.columns:
            raise ValueError("Cannot create labels: 'close' column not found")

        prices = self.df["close"].values

        if self.prediction_type == "direction":
            # Binary classification: 1 if price goes up, 0 if down
            # Compare price at t+horizon with price at t
            future_prices = np.roll(prices, -self.horizon)
            labels = (future_prices > prices).astype(np.float32)
            # Set labels beyond data boundaries to NaN (will be filtered)
            labels[-self.horizon :] = np.nan
        elif self.prediction_type == "trend":
            # Regression: trend strength (0-1)
            # Normalize price change over horizon
            future_prices = np.roll(prices, -self.horizon)
            price_changes = (future_prices - prices) / prices
            # Normalize to 0-1 range (using percentile-based normalization)
            p99 = np.nanpercentile(np.abs(price_changes), 99)
            labels = np.clip((price_changes / (2 * p99)) + 0.5, 0, 1).astype(np.float32)
            labels[-self.horizon :] = np.nan
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return labels

    def _get_valid_indices(self) -> list[int]:
        """
        Get indices of valid sequences (those that don't go beyond data boundaries).

        Returns:
            List of valid indices
        """
        valid_indices = []
        for i in range(len(self.df) - self.sequence_length - self.horizon + 1):
            # Check if label is valid (not NaN)
            if not np.isnan(self.labels[i + self.sequence_length - 1]):
                valid_indices.append(i)
        return valid_indices

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sequence and its label.

        Args:
            idx: Index in valid_indices

        Returns:
            Dictionary with 'features' and 'label' tensors
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        # Extract sequence
        sequence = self.features[start_idx:end_idx]
        label_idx = start_idx + self.sequence_length - 1
        label = self.labels[label_idx]

        # Convert to tensors
        features_tensor = torch.FloatTensor(sequence)
        label_tensor = torch.FloatTensor([label])

        return {"features": features_tensor, "label": label_tensor}

