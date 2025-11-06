"""Tests for configuration system."""

import pytest

from libs.config.config import EnsembleWeights, Settings


def test_ensemble_weights_normalization():
    """Test ensemble weights normalization."""
    weights = EnsembleWeights(lstm=0.35, transformer=0.40, rl=0.25)
    normalized = weights.normalize()
    total = normalized.lstm + normalized.transformer + normalized.rl
    assert abs(total - 1.0) < 0.001


def test_ensemble_weights_fallback():
    """Test fallback weights when RL is 0."""
    weights = EnsembleWeights(lstm=0.0, transformer=0.0, rl=0.0)
    normalized = weights.normalize()
    assert normalized.lstm == 0.5
    assert normalized.transformer == 0.5
    assert normalized.rl == 0.0


def test_settings_validation():
    """Test settings validation."""
    settings = Settings(
        confidence_threshold=0.75, max_signals_per_hour=10, max_signals_per_hour_high=5
    )
    settings.validate()  # Should not raise


def test_settings_validation_fails():
    """Test that invalid settings raise errors."""
    settings = Settings(
        confidence_threshold=0.3,  # Too low
        max_signals_per_hour=10,
        max_signals_per_hour_high=5,
    )
    with pytest.raises(ValueError):
        settings.validate()
