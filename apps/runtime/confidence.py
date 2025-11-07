"""Confidence scoring system (basic implementation, enhanced in Phase 5)."""
from typing import Any

from loguru import logger


def score_confidence(
    lstm_pred: float = 0.0,
    transformer_pred: float = 0.0,
    rl_pred: float = 0.0,
    ensemble_weights: dict[str, float] | None = None,
    conservative_bias: float = -0.05,
) -> float:
    """
    Score confidence from ensemble predictions.

    Args:
        lstm_pred: LSTM prediction (0.0-1.0)
        transformer_pred: Transformer prediction (0.0-1.0)
        rl_pred: RL prediction (0.0-1.0, optional)
        ensemble_weights: Ensemble weights dict (default: 50/50 LSTM/Transformer if no RL)
        conservative_bias: Conservative bias to apply (default: -5%)

    Returns:
        Confidence score (0.0-1.0)
    """
    if ensemble_weights is None:
        # Default: 50/50 LSTM/Transformer if no RL, or 35/40/25 if RL provided
        if rl_pred > 0:
            ensemble_weights = {"lstm": 0.35, "transformer": 0.40, "rl": 0.25}
        else:
            ensemble_weights = {"lstm": 0.5, "transformer": 0.5, "rl": 0.0}

    # Normalize weights
    total_weight = sum(ensemble_weights.values())
    if total_weight == 0:
        logger.warning("All ensemble weights are zero, using default")
        ensemble_weights = {"lstm": 0.5, "transformer": 0.5, "rl": 0.0}
        total_weight = 1.0

    normalized_weights = {k: v / total_weight for k, v in ensemble_weights.items()}

    # Calculate weighted average
    confidence = (
        lstm_pred * normalized_weights.get("lstm", 0.0)
        + transformer_pred * normalized_weights.get("transformer", 0.0)
        + rl_pred * normalized_weights.get("rl", 0.0)
    )

    # Apply conservative bias
    confidence = max(0.0, min(1.0, confidence + conservative_bias))

    logger.debug(
        f"Confidence scored: LSTM={lstm_pred:.2f}, Transformer={transformer_pred:.2f}, "
        f"RL={rl_pred:.2f}, Weighted={confidence:.2f}, Final={confidence:.2f}"
    )

    return confidence


def apply_tier_hysteresis(
    current_confidence: float, previous_tier: str | None = None, hysteresis_threshold: float = 0.02
) -> str:
    """
    Apply tier hysteresis to avoid flapping.

    Args:
        current_confidence: Current confidence score
        previous_tier: Previous tier (if available)
        hysteresis_threshold: Threshold for tier changes (default: 2%)

    Returns:
        Tier: 'high', 'medium', or 'low'
    """
    # Determine tier from confidence
    if current_confidence >= 0.75:
        new_tier = "high"
    elif current_confidence >= 0.65:
        new_tier = "medium"
    else:
        new_tier = "low"

    # Apply hysteresis if previous tier exists
    if previous_tier:
        if previous_tier == "high" and new_tier == "medium":
            # Require 2% drop to downgrade from high
            if current_confidence >= 0.75 - hysteresis_threshold:
                return "high"
        elif previous_tier == "medium" and new_tier == "low":
            # Require 2% drop to downgrade from medium
            if current_confidence >= 0.65 - hysteresis_threshold:
                return "medium"
        elif previous_tier == "low" and new_tier == "medium":
            # Require 2% increase to upgrade from low
            if current_confidence < 0.65 + hysteresis_threshold:
                return "low"
        elif previous_tier == "medium" and new_tier == "high":
            # Require 2% increase to upgrade from medium
            if current_confidence < 0.75 + hysteresis_threshold:
                return "medium"

    return new_tier

