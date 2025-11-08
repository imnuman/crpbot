"""Platt and Isotonic scaling for confidence calibration."""
from typing import Any

import numpy as np
from loguru import logger

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Platt/Isotonic scaling will be disabled.")


class PlattScaling:
    """Platt scaling (logistic regression) for probability calibration."""

    def __init__(self):
        """Initialize Platt scaler."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Platt scaling")
        self.model = LogisticRegression()

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit Platt scaler to scores and labels.

        Args:
            scores: Confidence scores (0.0-1.0)
            labels: True labels (0 or 1)
        """
        scores_2d = scores.reshape(-1, 1)
        self.model.fit(scores_2d, labels)
        logger.info("Platt scaling model fitted")

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores using fitted Platt scaler.

        Args:
            scores: Confidence scores (0.0-1.0)

        Returns:
            Calibrated scores (0.0-1.0)
        """
        scores_2d = scores.reshape(-1, 1)
        calibrated = self.model.predict_proba(scores_2d)[:, 1]
        return np.clip(calibrated, 0.0, 1.0)


class IsotonicScaling:
    """Isotonic regression for probability calibration."""

    def __init__(self):
        """Initialize Isotonic scaler."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Isotonic scaling")
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit Isotonic scaler to scores and labels.

        Args:
            scores: Confidence scores (0.0-1.0)
            labels: True labels (0 or 1)
        """
        self.model.fit(scores, labels)
        logger.info("Isotonic scaling model fitted")

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores using fitted Isotonic scaler.

        Args:
            scores: Confidence scores (0.0-1.0)

        Returns:
            Calibrated scores (0.0-1.0)
        """
        calibrated = self.model.predict(scores)
        return np.clip(calibrated, 0.0, 1.0)


def calculate_calibration_error(
    predicted_scores: np.ndarray, actual_labels: np.ndarray, n_bins: int = 10
) -> float:
    """
    Calculate calibration error (ECE - Expected Calibration Error).

    Args:
        predicted_scores: Predicted confidence scores
        actual_labels: Actual binary labels (0 or 1)
        n_bins: Number of bins for calibration error calculation

    Returns:
        Calibration error (0.0-1.0, lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predicted_scores > bin_lower) & (predicted_scores <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = actual_labels[in_bin].mean()
            # Average confidence in this bin
            avg_confidence_in_bin = predicted_scores[in_bin].mean()
            # Calibration error for this bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def should_apply_calibration(
    calibration_error: float, threshold: float = 0.05
) -> tuple[bool, str]:
    """
    Determine if calibration should be applied.

    Args:
        calibration_error: Current calibration error
        threshold: Calibration error threshold (default: 5%)

    Returns:
        Tuple of (should_apply, method)
        - should_apply: True if calibration needed
        - method: 'platt' or 'isotonic' (recommended method)
    """
    if calibration_error <= threshold:
        return False, "none"

    # Use Isotonic for larger errors, Platt for smaller
    if calibration_error > 0.10:
        return True, "isotonic"
    else:
        return True, "platt"

