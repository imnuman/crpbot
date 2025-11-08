"""Unit tests for enhanced confidence scoring."""

from datetime import datetime

import numpy as np
import pytest

from libs.confidence.enhanced import EnhancedConfidenceScorer, TierHysteresis
from libs.confidence.scaling import (
    SKLEARN_AVAILABLE,
    calculate_calibration_error,
)


def test_hysteresis_requires_consecutive_scans():
    """Tier hysteresis should require consecutive confirmations to upgrade."""
    hysteresis = TierHysteresis(hysteresis_threshold=0.01, consecutive_scans=2)

    assert hysteresis.update(0.60) == "low"
    assert hysteresis.update(0.76) == "low"  # first attempt should not upgrade yet
    assert hysteresis.update(0.77) == "high"  # second consecutive high should upgrade


def test_ensemble_scoring_weights():
    """Ensemble scoring should respect custom weights and bias."""
    scorer = EnhancedConfidenceScorer(
        ensemble_weights={"lstm": 0.5, "transformer": 0.5, "rl": 0.0, "sentiment": 0.0},
        conservative_bias=0.0,
        enable_boosters=False,
        enable_hysteresis=False,
        enable_calibration=False,
    )

    confidence = scorer.score(lstm_pred=0.8, transformer_pred=0.6)
    assert pytest.approx(confidence, rel=1e-3) == 0.7


def test_free_boosters_and_pattern_adjustments():
    """Boosters should lift confidence when multi-timeframe and pattern win rates align."""
    scorer = EnhancedConfidenceScorer(conservative_bias=0.0, enable_calibration=False)

    base_confidence = scorer.score(lstm_pred=0.7, transformer_pred=0.7)

    boosted_confidence = scorer.score(
        lstm_pred=0.7,
        transformer_pred=0.7,
        timestamp=datetime(2025, 1, 1, 10, 0),
        multi_tf_aligned=True,
        volatility_regime="high",
        pattern_win_rate=0.65,
        pattern_sample_count=25,
    )

    assert boosted_confidence > base_confidence


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_calibration_fits_when_error_high():
    """Calibration should fit when error exceeds 5%."""
    scorer = EnhancedConfidenceScorer(conservative_bias=0.0, enable_boosters=False)

    scores = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
    labels = np.concatenate([np.ones(25), np.zeros(25), np.zeros(45), np.ones(5)])

    calibration_error = calculate_calibration_error(scores, labels)
    assert calibration_error > 0.05

    scorer.fit_calibration(scores, labels)

    assert scorer.calibration_fitted is True
    assert scorer.calibration_method in {"platt", "isotonic"}
