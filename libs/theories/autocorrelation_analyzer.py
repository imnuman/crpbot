"""
Autocorrelation Analysis

Measures correlation of returns with their own lagged values:
- High positive autocorrelation: Trends persist (momentum strategies work)
- Low/zero autocorrelation: Random walk (no predictable patterns)
- Negative autocorrelation: Mean reversion (reversal strategies work)

Used to determine optimal strategy type for current market conditions.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AutocorrelationAnalyzer:
    """
    Analyze autocorrelation structure of returns

    Determines whether returns are:
    - Trending (momentum strategies work)
    - Mean-reverting (reversal strategies work)
    - Random (no edge)
    """

    def __init__(self, max_lags: int = 10):
        """
        Initialize autocorrelation analyzer

        Args:
            max_lags: Maximum number of lags to test (default: 10)
        """
        self.max_lags = max_lags

    def analyze(self, prices: np.ndarray, max_lags: int = None) -> Dict[str, float]:
        """
        Analyze autocorrelation structure of returns

        Args:
            prices: Array of historical prices
            max_lags: Override default max_lags if provided

        Returns:
            Dictionary with autocorrelation analysis results:
            - acf_lag1: Autocorrelation at lag 1
            - acf_lag5: Autocorrelation at lag 5
            - acf_mean: Mean autocorrelation across all lags
            - trend_strength: 0.0-1.0 (based on positive autocorrelation)
            - mean_reversion_score: 0.0-1.0 (based on negative autocorrelation)
            - optimal_strategy: 'momentum' or 'mean_reversion'
        """
        try:
            if len(prices) < 30:
                logger.warning(f"Insufficient data for autocorrelation: {len(prices)} < 30")
                return self._default_results()

            # Use override if provided, otherwise use instance max_lags
            lags = max_lags if max_lags is not None else self.max_lags

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Calculate autocorrelation at different lags
            acf_values = []
            for lag in range(1, min(lags + 1, len(returns) // 2)):
                acf = self._calculate_acf(returns, lag)
                acf_values.append(acf)

            # Extract key lag values
            acf_lag1 = acf_values[0] if len(acf_values) > 0 else 0.0
            acf_lag5 = acf_values[4] if len(acf_values) >= 5 else 0.0
            acf_mean = np.mean(acf_values) if acf_values else 0.0

            # Trend strength (based on positive autocorrelation)
            # Positive ACF = momentum, negative ACF = mean reversion
            positive_acf = [acf for acf in acf_values if acf > 0]
            trend_strength = np.mean(positive_acf) if positive_acf else 0.0
            trend_strength = max(0.0, min(1.0, trend_strength))  # Clamp to [0, 1]

            # Mean reversion score (based on negative autocorrelation)
            negative_acf = [abs(acf) for acf in acf_values if acf < 0]
            mean_reversion_score = np.mean(negative_acf) if negative_acf else 0.0
            mean_reversion_score = max(0.0, min(1.0, mean_reversion_score))  # Clamp to [0, 1]

            # Determine optimal strategy
            if trend_strength > mean_reversion_score:
                optimal_strategy = 'momentum'
            else:
                optimal_strategy = 'mean_reversion'

            return {
                'acf_lag1': float(acf_lag1),
                'acf_lag5': float(acf_lag5),
                'acf_mean': float(acf_mean),
                'trend_strength': float(trend_strength),
                'mean_reversion_score': float(mean_reversion_score),
                'optimal_strategy': optimal_strategy
            }

        except Exception as e:
            logger.error(f"Autocorrelation analysis failed: {e}")
            return self._default_results()

    def _calculate_acf(self, returns: np.ndarray, lag: int) -> float:
        """
        Calculate autocorrelation function at given lag

        Args:
            returns: Array of returns
            lag: Lag period

        Returns:
            Autocorrelation coefficient
        """
        try:
            if len(returns) < lag + 1:
                return 0.0

            # Split into current and lagged series
            y = returns[lag:]
            y_lagged = returns[:-lag]

            # Calculate correlation
            if len(y) == 0 or len(y_lagged) == 0:
                return 0.0

            # Pearson correlation coefficient
            mean_y = np.mean(y)
            mean_y_lagged = np.mean(y_lagged)

            numerator = np.sum((y - mean_y) * (y_lagged - mean_y_lagged))
            denominator = np.sqrt(np.sum((y - mean_y)**2) * np.sum((y_lagged - mean_y_lagged)**2))

            if denominator == 0:
                return 0.0

            acf = numerator / denominator
            return acf

        except Exception as e:
            logger.error(f"ACF calculation failed at lag {lag}: {e}")
            return 0.0

    def _default_results(self) -> Dict[str, float]:
        """Return default results when analysis fails"""
        return {
            'acf_lag1': 0.0,
            'acf_lag5': 0.0,
            'acf_mean': 0.0,
            'trend_strength': 0.5,
            'mean_reversion_score': 0.5,
            'optimal_strategy': 'momentum'
        }
