"""
Random Forest Ensemble Validator

Uses Random Forest classifier to validate trading signals based on:
- Technical indicator patterns
- Price momentum features
- Volatility characteristics
- Volume patterns

Provides ensemble voting confidence for signal validation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RandomForestValidator:
    """
    Random Forest ensemble model for signal validation

    Uses multiple decision trees to vote on signal quality based on
    technical features extracted from price data.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        """
        Initialize Random Forest validator

        Args:
            n_estimators: Number of trees in forest (default: 100)
            max_depth: Maximum depth of each tree (default: 10)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced'
        )
        self.is_trained = False

    def extract_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extract features from price data for Random Forest

        Features include:
        - Returns (1, 5, 10, 20 periods)
        - Volatility (rolling std)
        - Momentum indicators
        - Price range characteristics

        Args:
            prices: Array of prices

        Returns:
            Feature vector for classification
        """
        if len(prices) < 50:
            # Return default features if insufficient data
            return np.zeros(15)

        features = []

        # Returns over different periods
        for period in [1, 5, 10, 20]:
            if len(prices) > period:
                ret = (prices[-1] - prices[-period]) / prices[-period]
                features.append(ret)
            else:
                features.append(0.0)

        # Volatility (rolling std)
        for window in [5, 10, 20]:
            if len(prices) > window:
                vol = np.std(prices[-window:]) / np.mean(prices[-window:])
                features.append(vol)
            else:
                features.append(0.0)

        # Price momentum (rate of change)
        if len(prices) > 10:
            momentum = (prices[-1] - prices[-10]) / prices[-10]
            features.append(momentum)
        else:
            features.append(0.0)

        # Price acceleration (second derivative)
        if len(prices) > 20:
            acc = ((prices[-1] - prices[-10]) - (prices[-10] - prices[-20])) / prices[-20]
            features.append(acc)
        else:
            features.append(0.0)

        # High-low range
        if len(prices) > 20:
            hl_range = (np.max(prices[-20:]) - np.min(prices[-20:])) / np.mean(prices[-20:])
            features.append(hl_range)
        else:
            features.append(0.0)

        # Trend strength (linear regression slope)
        if len(prices) > 20:
            x = np.arange(20)
            y = prices[-20:]
            slope = np.polyfit(x, y, 1)[0] / np.mean(y)
            features.append(slope)
        else:
            features.append(0.0)

        # Mean reversion indicator
        if len(prices) > 20:
            mean_20 = np.mean(prices[-20:])
            distance_from_mean = (prices[-1] - mean_20) / mean_20
            features.append(distance_from_mean)
        else:
            features.append(0.0)

        # Relative position in recent range
        if len(prices) > 20:
            min_20 = np.min(prices[-20:])
            max_20 = np.max(prices[-20:])
            if max_20 > min_20:
                position = (prices[-1] - min_20) / (max_20 - min_20)
            else:
                position = 0.5
            features.append(position)
        else:
            features.append(0.5)

        return np.array(features)

    def analyze(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Analyze price data using Random Forest ensemble

        Args:
            prices: Array of historical prices

        Returns:
            Dictionary with RF analysis results:
            - rf_bullish_prob: Probability of bullish outcome (0.0-1.0)
            - rf_bearish_prob: Probability of bearish outcome (0.0-1.0)
            - rf_confidence: Ensemble confidence (0.0-1.0)
            - rf_signal: Recommended action (-1=sell, 0=hold, 1=buy)
        """
        try:
            # Extract features
            features = self.extract_features(prices)

            # If model is not trained, use heuristic approach
            if not self.is_trained:
                return self._heuristic_analysis(features)

            # Reshape for sklearn
            X = features.reshape(1, -1)

            # Get prediction probabilities
            proba = self.model.predict_proba(X)[0]

            # Extract probabilities (assuming classes: [0=bearish, 1=neutral, 2=bullish])
            bearish_prob = proba[0] if len(proba) > 0 else 0.33
            neutral_prob = proba[1] if len(proba) > 1 else 0.34
            bullish_prob = proba[2] if len(proba) > 2 else 0.33

            # Confidence is the max probability
            confidence = max(bullish_prob, bearish_prob, neutral_prob)

            # Determine signal
            if bullish_prob > bearish_prob and bullish_prob > 0.4:
                signal = 1  # Buy
            elif bearish_prob > bullish_prob and bearish_prob > 0.4:
                signal = -1  # Sell
            else:
                signal = 0  # Hold

            return {
                'rf_bullish_prob': float(bullish_prob),
                'rf_bearish_prob': float(bearish_prob),
                'rf_neutral_prob': float(neutral_prob),
                'rf_confidence': float(confidence),
                'rf_signal': int(signal)
            }

        except Exception as e:
            logger.warning(f"Random Forest analysis failed: {e}")
            return {
                'rf_bullish_prob': 0.33,
                'rf_bearish_prob': 0.33,
                'rf_neutral_prob': 0.34,
                'rf_confidence': 0.34,
                'rf_signal': 0
            }

    def _heuristic_analysis(self, features: np.ndarray) -> Dict[str, float]:
        """
        Heuristic analysis when model is not trained

        Uses feature values directly to estimate probabilities

        Args:
            features: Extracted feature vector

        Returns:
            Dictionary with estimated probabilities
        """
        # Extract key features
        ret_1 = features[0]  # 1-period return
        ret_20 = features[3]  # 20-period return
        momentum = features[7]  # Momentum
        trend_strength = features[10]  # Trend strength

        # Calculate bullish/bearish scores
        bullish_score = 0.0
        bearish_score = 0.0

        # Recent return
        if ret_1 > 0.01:
            bullish_score += 0.2
        elif ret_1 < -0.01:
            bearish_score += 0.2

        # Medium-term return
        if ret_20 > 0.05:
            bullish_score += 0.3
        elif ret_20 < -0.05:
            bearish_score += 0.3

        # Momentum
        if momentum > 0.02:
            bullish_score += 0.3
        elif momentum < -0.02:
            bearish_score += 0.3

        # Trend strength
        if trend_strength > 0:
            bullish_score += 0.2
        elif trend_strength < 0:
            bearish_score += 0.2

        # Normalize to probabilities
        total = bullish_score + bearish_score + 0.5  # Add baseline for neutral
        bullish_prob = bullish_score / total
        bearish_prob = bearish_score / total
        neutral_prob = 0.5 / total

        # Confidence is the max
        confidence = max(bullish_prob, bearish_prob, neutral_prob)

        # Determine signal
        if bullish_prob > bearish_prob and bullish_prob > 0.4:
            signal = 1
        elif bearish_prob > bullish_prob and bearish_prob > 0.4:
            signal = -1
        else:
            signal = 0

        return {
            'rf_bullish_prob': float(bullish_prob),
            'rf_bearish_prob': float(bearish_prob),
            'rf_neutral_prob': float(neutral_prob),
            'rf_confidence': float(confidence),
            'rf_signal': int(signal)
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Train the Random Forest model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=bearish, 1=neutral, 2=bullish)

        Returns:
            True if training successful
        """
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logger.info(f"Random Forest trained on {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return False

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores from trained model

        Returns:
            Array of feature importance scores, or None if not trained
        """
        if not self.is_trained:
            return None
        return self.model.feature_importances_
