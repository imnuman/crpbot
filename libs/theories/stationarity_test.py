"""
Stationarity Test - Augmented Dickey-Fuller (ADF) Test

Tests whether time series is:
- Stationary: Mean-reverting (mean and variance are constant)
- Non-stationary: Trending (mean and/or variance change over time)

Used to determine optimal strategy:
- Stationary series → Mean reversion strategies
- Non-stationary series → Momentum/trend-following strategies
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class StationarityAnalyzer:
    """
    Test for stationarity in price series

    Simplified ADF-like test to determine if series is:
    - Stationary (suitable for mean reversion)
    - Non-stationary (suitable for momentum)
    """

    def __init__(self, window_size: int = 20):
        """
        Initialize stationarity analyzer

        Args:
            window_size: Window for rolling statistics (default: 20)
        """
        self.window_size = window_size

    def analyze(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Test for stationarity using simplified ADF-like approach

        Args:
            prices: Array of historical prices

        Returns:
            Dictionary with stationarity test results:
            - is_stationary: Boolean (True if stationary)
            - adf_score: Stationarity score (-2.0 to +2.0, negative = stationary)
            - trend_strength: 0.0-1.0 (non-stationary = high trend)
            - mean_reversion_strength: 0.0-1.0 (stationary = high mean reversion)
            - recommended_strategy: 'momentum' or 'mean_reversion'
        """
        try:
            if len(prices) < self.window_size * 2:
                logger.warning(f"Insufficient data for stationarity test: {len(prices)} < {self.window_size * 2}")
                return self._default_results()

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Test 1: Mean stability
            first_half_mean = np.mean(returns[:len(returns)//2])
            second_half_mean = np.mean(returns[len(returns)//2:])
            mean_stability = 1.0 - abs(first_half_mean - second_half_mean)

            # Test 2: Variance stability
            first_half_var = np.var(returns[:len(returns)//2])
            second_half_var = np.var(returns[len(returns)//2:])

            if first_half_var > 0:
                var_ratio = second_half_var / first_half_var
                var_stability = 1.0 / (1.0 + abs(var_ratio - 1.0))
            else:
                var_stability = 0.5

            # Test 3: Autocorrelation of first differences
            # Stationary series should have low autocorrelation in first differences
            diff_returns = np.diff(returns)
            if len(diff_returns) > 1:
                autocorr = self._calculate_autocorr(diff_returns, lag=1)
                # Low autocorr in differences = stationary
                diff_stationarity = 1.0 - abs(autocorr)
            else:
                diff_stationarity = 0.5

            # Test 4: Mean reversion tendency
            # Calculate how often price crosses the mean
            mean_price = np.mean(prices)
            crossings = 0
            for i in range(1, len(prices)):
                if (prices[i-1] < mean_price and prices[i] > mean_price) or \
                   (prices[i-1] > mean_price and prices[i] < mean_price):
                    crossings += 1

            # More crossings = more mean reversion = more stationary
            crossing_rate = crossings / len(prices)
            mean_reversion_tendency = min(crossing_rate * 10, 1.0)  # Scale to 0-1

            # Combine tests into ADF-like score
            # More negative = more stationary
            stationarity_score = (mean_stability + var_stability + diff_stationarity + mean_reversion_tendency) / 4

            # Convert to ADF-like score range (-2 to +2)
            # Stationary: score > 0.6 → ADF < -1.0
            # Non-stationary: score < 0.4 → ADF > +1.0
            adf_score = 2.0 - (stationarity_score * 4.0)  # Maps [0, 1] to [+2, -2]

            # Determine stationarity (ADF critical value typically -1.95 for 5% significance)
            is_stationary = adf_score < -1.0

            # Trend strength (inverse of stationarity)
            trend_strength = 1.0 - stationarity_score
            trend_strength = max(0.0, min(1.0, trend_strength))

            # Mean reversion strength (same as stationarity score)
            mean_reversion_strength = stationarity_score

            # Recommended strategy
            if is_stationary:
                recommended_strategy = 'mean_reversion'
            else:
                recommended_strategy = 'momentum'

            return {
                'is_stationary': bool(is_stationary),
                'adf_score': float(adf_score),
                'trend_strength': float(trend_strength),
                'mean_reversion_strength': float(mean_reversion_strength),
                'recommended_strategy': recommended_strategy
            }

        except Exception as e:
            logger.error(f"Stationarity analysis failed: {e}")
            return self._default_results()

    def _calculate_autocorr(self, series: np.ndarray, lag: int) -> float:
        """
        Calculate autocorrelation at given lag

        Args:
            series: Time series data
            lag: Lag period

        Returns:
            Autocorrelation coefficient
        """
        try:
            if len(series) < lag + 1:
                return 0.0

            y = series[lag:]
            y_lagged = series[:-lag]

            if len(y) == 0 or len(y_lagged) == 0:
                return 0.0

            mean_y = np.mean(y)
            mean_y_lagged = np.mean(y_lagged)

            numerator = np.sum((y - mean_y) * (y_lagged - mean_y_lagged))
            denominator = np.sqrt(np.sum((y - mean_y)**2) * np.sum((y_lagged - mean_y_lagged)**2))

            if denominator == 0:
                return 0.0

            return numerator / denominator

        except Exception as e:
            logger.error(f"Autocorrelation calculation failed: {e}")
            return 0.0

    def _default_results(self) -> Dict[str, float]:
        """Return default results when analysis fails"""
        return {
            'is_stationary': False,
            'adf_score': 0.0,
            'trend_strength': 0.5,
            'mean_reversion_strength': 0.5,
            'recommended_strategy': 'momentum'
        }
