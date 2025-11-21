"""
Variance Tests - Homoscedasticity and Heteroscedasticity Detection

Tests for variance stability in price returns:
- Homoscedasticity: Constant variance over time (stable market)
- Heteroscedasticity: Changing variance over time (regime changes, breakouts)

Used to detect market regime changes and adjust position sizing.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class VarianceAnalyzer:
    """
    Analyze variance stability in price returns

    Detects heteroscedasticity (changing variance) which indicates:
    - Market regime changes
    - Increased uncertainty
    - Potential breakouts
    """

    def __init__(self, window_size: int = 20):
        """
        Initialize variance analyzer

        Args:
            window_size: Window for variance calculations (default: 20)
        """
        self.window_size = window_size

    def analyze(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Test for variance stability

        Args:
            prices: Array of historical prices

        Returns:
            Dictionary with variance analysis results:
            - variance_ratio: Recent variance / historical variance
            - is_heteroscedastic: True if variance is changing
            - variance_stability: 0.0-1.0 (1.0 = stable variance)
            - regime_change_prob: 0.0-1.0 probability of regime change
            - volatility_trend: 'increasing', 'decreasing', or 'stable'
        """
        try:
            if len(prices) < self.window_size * 2:
                logger.warning(f"Insufficient data for variance test: {len(prices)} < {self.window_size * 2}")
                return self._default_results()

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Split into historical and recent periods
            split_point = len(returns) // 2
            historical_returns = returns[:split_point]
            recent_returns = returns[split_point:]

            # Calculate variances
            historical_var = np.var(historical_returns)
            recent_var = np.var(recent_returns)

            # Variance ratio (>1 = increasing volatility, <1 = decreasing)
            if historical_var > 0:
                variance_ratio = recent_var / historical_var
            else:
                variance_ratio = 1.0

            # Test for heteroscedasticity using simple ratio test
            # Threshold: variance ratio significantly different from 1.0
            HETEROSCEDASTIC_THRESHOLD = 1.5
            is_heteroscedastic = (variance_ratio > HETEROSCEDASTIC_THRESHOLD or
                                variance_ratio < 1.0 / HETEROSCEDASTIC_THRESHOLD)

            # Variance stability score (inverse of variance ratio deviation)
            variance_stability = 1.0 / (1.0 + abs(variance_ratio - 1.0))

            # Regime change probability (based on variance change magnitude)
            regime_change_prob = min(abs(variance_ratio - 1.0), 1.0)

            # Volatility trend
            if variance_ratio > 1.2:
                volatility_trend = 'increasing'
            elif variance_ratio < 0.8:
                volatility_trend = 'decreasing'
            else:
                volatility_trend = 'stable'

            return {
                'variance_ratio': float(variance_ratio),
                'is_heteroscedastic': bool(is_heteroscedastic),
                'variance_stability': float(variance_stability),
                'regime_change_prob': float(regime_change_prob),
                'volatility_trend': volatility_trend
            }

        except Exception as e:
            logger.error(f"Variance analysis failed: {e}")
            return self._default_results()

    def _default_results(self) -> Dict[str, float]:
        """Return default results when analysis fails"""
        return {
            'variance_ratio': 1.0,
            'is_heteroscedastic': False,
            'variance_stability': 1.0,
            'regime_change_prob': 0.0,
            'volatility_trend': 'stable'
        }
