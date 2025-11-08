"""Enhanced confidence scoring with calibration, hysteresis, and boosters."""
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

from apps.trainer.features import get_trading_session
from libs.confidence.scaling import (
    IsotonicScaling,
    PlattScaling,
    calculate_calibration_error,
    should_apply_calibration,
    SKLEARN_AVAILABLE,
)


class TierHysteresis:
    """Tier hysteresis to avoid flapping between tiers."""

    def __init__(self, hysteresis_threshold: float = 0.02, consecutive_scans: int = 2):
        """
        Initialize tier hysteresis.

        Args:
            hysteresis_threshold: Threshold for tier changes (default: 2%)
            consecutive_scans: Number of consecutive scans required to upgrade tier
        """
        self.hysteresis_threshold = hysteresis_threshold
        self.consecutive_scans = consecutive_scans
        self.scan_history: deque = deque(maxlen=consecutive_scans)
        self.current_tier: str | None = None

    def update(self, confidence: float) -> str:
        """
        Update tier with hysteresis logic.

        Args:
            confidence: Current confidence score

        Returns:
            Tier after applying hysteresis
        """
        # Determine target tier
        if confidence >= 0.75:
            target_tier = "high"
        elif confidence >= 0.65:
            target_tier = "medium"
        else:
            target_tier = "low"

        # Add to history
        self.scan_history.append({"confidence": confidence, "target_tier": target_tier})

        # Apply hysteresis
        if self.current_tier is None:
            # First scan, use target tier
            self.current_tier = target_tier
            return target_tier

        # Check if we need to upgrade/downgrade
        if target_tier != self.current_tier:
            # Check if we have enough consecutive scans
            recent_scans = list(self.scan_history)[-self.consecutive_scans :]
            if len(recent_scans) >= self.consecutive_scans:
                # Check if all recent scans point to target tier
                if all(scan["target_tier"] == target_tier for scan in recent_scans):
                    # Check if confidence change is significant enough
                    if self._is_significant_change(confidence, target_tier):
                        self.current_tier = target_tier
                        logger.debug(f"Tier changed: {self.current_tier} (confidence: {confidence:.2%})")
                    else:
                        logger.debug(
                            f"Tier change prevented by hysteresis: {target_tier} (confidence: {confidence:.2%})"
                        )

        return self.current_tier

    def _is_significant_change(self, confidence: float, target_tier: str) -> bool:
        """Check if confidence change is significant enough."""
        if target_tier == "high" and self.current_tier == "medium":
            # Upgrading to high: need 0.75 + threshold
            return confidence >= 0.75 + self.hysteresis_threshold
        elif target_tier == "medium" and self.current_tier == "low":
            # Upgrading to medium: need 0.65 + threshold
            return confidence >= 0.65 + self.hysteresis_threshold
        elif target_tier == "low" and self.current_tier == "medium":
            # Downgrading to low: need < 0.65 - threshold
            return confidence < 0.65 - self.hysteresis_threshold
        elif target_tier == "medium" and self.current_tier == "high":
            # Downgrading to medium: need < 0.75 - threshold
            return confidence < 0.75 - self.hysteresis_threshold
        return True

    def reset(self) -> None:
        """Reset hysteresis state."""
        self.scan_history.clear()
        self.current_tier = None


class EnhancedConfidenceScorer:
    """
    Enhanced confidence scorer with calibration, hysteresis, and boosters.

    Features:
    - Ensemble weighting (LSTM + Transformer + RL + Sentiment)
    - Conservative bias
    - Platt/Isotonic scaling (if calibration error > 5%)
    - Tier hysteresis
    - FREE boosters (multi-TF, session timing, volatility)
    - Per-pattern sample floor (via database)
    """

    def __init__(
        self,
        ensemble_weights: dict[str, float] | None = None,
        conservative_bias: float = -0.05,
        enable_calibration: bool = True,
        calibration_threshold: float = 0.05,
        enable_hysteresis: bool = True,
        enable_boosters: bool = True,
    ):
        """
        Initialize enhanced confidence scorer.

        Args:
            ensemble_weights: Ensemble weights dict
            conservative_bias: Conservative bias to apply (default: -5%)
            enable_calibration: Enable Platt/Isotonic scaling
            calibration_threshold: Calibration error threshold (default: 5%)
            enable_hysteresis: Enable tier hysteresis
            enable_boosters: Enable FREE boosters
        """
        self.ensemble_weights = ensemble_weights or {"lstm": 0.35, "transformer": 0.40, "rl": 0.25, "sentiment": 0.0}
        self.conservative_bias = conservative_bias
        self.enable_calibration = enable_calibration
        self.calibration_threshold = calibration_threshold
        self.enable_hysteresis = enable_hysteresis
        self.enable_boosters = enable_boosters

        # Normalize weights
        total = sum(self.ensemble_weights.values())
        if total > 0:
            self.ensemble_weights = {k: v / total for k, v in self.ensemble_weights.items()}
        else:
            # Fallback: 50/50 LSTM/Transformer if all weights are 0
            self.ensemble_weights = {"lstm": 0.5, "transformer": 0.5, "rl": 0.0, "sentiment": 0.0}

        # Calibration models
        self.platt_scaler: PlattScaling | None = None
        self.isotonic_scaler: IsotonicScaling | None = None
        self.calibration_method: str = "none"
        self.calibration_fitted = False

        # Hysteresis
        self.hysteresis = TierHysteresis() if enable_hysteresis else None

        # Calibration data (for fitting)
        self.calibration_scores: list[float] = []
        self.calibration_labels: list[int] = []

        logger.info(f"Enhanced confidence scorer initialized: weights={self.ensemble_weights}")

    def score(
        self,
        lstm_pred: float = 0.0,
        transformer_pred: float = 0.0,
        rl_pred: float = 0.0,
        sentiment_pred: float = 0.0,
        timestamp: datetime | None = None,
        multi_tf_aligned: bool = False,
        volatility_regime: str = "medium",
        pattern_win_rate: float | None = None,
        pattern_sample_count: int = 0,
        pattern_sample_floor: int = 10,
    ) -> float:
        """
        Score confidence with all enhancements.

        Args:
            lstm_pred: LSTM prediction (0.0-1.0)
            transformer_pred: Transformer prediction (0.0-1.0)
            rl_pred: RL prediction (0.0-1.0)
            sentiment_pred: Sentiment prediction (0.0-1.0)
            timestamp: Timestamp for session timing
            multi_tf_aligned: Multi-timeframe alignment (V2)
            volatility_regime: Volatility regime ('high', 'medium', 'low')
            pattern_win_rate: Historical pattern win rate (from database)
            pattern_sample_count: Pattern sample count
            pattern_sample_floor: Minimum samples for pattern influence

        Returns:
            Confidence score (0.0-1.0)
        """
        # Calculate weighted ensemble
        confidence = (
            lstm_pred * self.ensemble_weights.get("lstm", 0.0)
            + transformer_pred * self.ensemble_weights.get("transformer", 0.0)
            + rl_pred * self.ensemble_weights.get("rl", 0.0)
            + sentiment_pred * self.ensemble_weights.get("sentiment", 0.0)
        )

        # Apply FREE boosters (V2)
        if self.enable_boosters:
            # Multi-timeframe alignment bonus
            if multi_tf_aligned:
                confidence += 0.02  # +2% boost
                logger.debug("Multi-timeframe alignment bonus applied")

            # Session timing boost
            if timestamp:
                session = get_trading_session(timestamp)
                # Optimal session times (simplified)
                hour = timestamp.hour
                if session == "london" and 9 <= hour <= 12:
                    confidence += 0.01  # +1% boost during optimal London hours
                elif session == "new_york" and 14 <= hour <= 17:
                    confidence += 0.01  # +1% boost during optimal NY hours
                logger.debug(f"Session timing boost applied: {session}")

            # Volatility regime adjustment
            if volatility_regime == "high":
                confidence += 0.01  # +1% boost in high volatility
            elif volatility_regime == "low":
                confidence -= 0.01  # -1% penalty in low volatility
            logger.debug(f"Volatility regime adjustment: {volatility_regime}")

        # Apply pattern win rate adjustment (if pattern has enough samples)
        if pattern_win_rate is not None and pattern_sample_count >= pattern_sample_floor:
            # Adjust confidence based on historical pattern performance
            pattern_adjustment = (pattern_win_rate - 0.5) * 0.1  # Scale adjustment
            confidence += pattern_adjustment
            logger.debug(f"Pattern adjustment applied: {pattern_adjustment:.2%} (win_rate: {pattern_win_rate:.2%})")

        # Apply conservative bias
        confidence = max(0.0, min(1.0, confidence + self.conservative_bias))

        # Apply calibration (if enabled and fitted)
        if self.enable_calibration and self.calibration_fitted:
            confidence = self._apply_calibration(confidence)

        return max(0.0, min(1.0, confidence))

    def _apply_calibration(self, score: float) -> float:
        """Apply calibration scaling to score."""
        if self.calibration_method == "platt" and self.platt_scaler:
            return float(self.platt_scaler.transform(np.array([score]))[0])
        elif self.calibration_method == "isotonic" and self.isotonic_scaler:
            return float(self.isotonic_scaler.transform(np.array([score]))[0])
        return score

    def fit_calibration(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit calibration models on validation data.

        Args:
            scores: Confidence scores (0.0-1.0)
            labels: True labels (0 or 1)
        """
        if not self.enable_calibration:
            return

        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Skipping calibration fitting.")
            return

        # Calculate calibration error
        calibration_error = calculate_calibration_error(scores, labels)
        logger.info(f"Calibration error: {calibration_error:.2%}")

        # Determine if calibration needed
        should_apply, method = should_apply_calibration(calibration_error, self.calibration_threshold)

        if not should_apply:
            logger.info(f"Calibration not needed (error: {calibration_error:.2%} <= {self.calibration_threshold:.2%})")
            return

        # Fit calibration model
        try:
            if method == "platt":
                self.platt_scaler = PlattScaling()
                self.platt_scaler.fit(scores, labels)
                self.calibration_method = "platt"
                logger.info(f"Platt scaling fitted (error: {calibration_error:.2%})")
            elif method == "isotonic":
                self.isotonic_scaler = IsotonicScaling()
                self.isotonic_scaler.fit(scores, labels)
                self.calibration_method = "isotonic"
                logger.info(f"Isotonic scaling fitted (error: {calibration_error:.2%})")

            self.calibration_fitted = True
        except Exception as e:
            logger.error(f"Failed to fit calibration model: {e}")
            self.calibration_fitted = False

    def get_tier(self, confidence: float) -> str:
        """
        Get tier with hysteresis applied.

        Args:
            confidence: Confidence score

        Returns:
            Tier: 'high', 'medium', or 'low'
        """
        if self.enable_hysteresis and self.hysteresis:
            return self.hysteresis.update(confidence)
        else:
            # Simple tier determination
            if confidence >= 0.75:
                return "high"
            elif confidence >= 0.65:
                return "medium"
            else:
                return "low"

