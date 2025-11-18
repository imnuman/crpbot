"""
Hurst Exponent Analysis for Trend vs Mean-Reversion Detection

The Hurst Exponent (H) measures the long-term memory of a time series.
- H > 0.5: Persistent/Trending behavior (momentum continues)
- H = 0.5: Random walk (no predictable pattern)
- H < 0.5: Anti-persistent/Mean-reverting behavior (reversals likely)

Method: Rescaled Range (R/S) Analysis
Formula: E[R(n)/S(n)] = C * n^H
where:
  - R(n) = Range of cumulative deviations
  - S(n) = Standard deviation
  - H = Hurst exponent (estimated from log-log slope)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from scipy import stats

logger = logging.getLogger(__name__)


class HurstExponentAnalyzer:
    """Calculate Hurst Exponent for trend vs mean-reversion detection"""

    def __init__(self, min_window: int = 10, max_window: Optional[int] = None):
        """
        Initialize Hurst Exponent Analyzer

        Args:
            min_window: Minimum window size for R/S analysis (default: 10)
            max_window: Maximum window size (default: data_length // 2)
        """
        self.min_window = min_window
        self.max_window = max_window
        logger.debug(f"Hurst Exponent Analyzer initialized (min_window={min_window})")

    def _rs_statistic(self, series: np.ndarray) -> float:
        """
        Calculate Rescaled Range (R/S) statistic for a time series

        Args:
            series: Time series data (returns or prices)

        Returns:
            R/S statistic value
        """
        n = len(series)
        if n < 2:
            return 0.0

        # Calculate mean
        mean = np.mean(series)

        # Calculate cumulative deviations from mean
        deviations = series - mean
        cumulative_deviations = np.cumsum(deviations)

        # Calculate range (R)
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

        # Calculate standard deviation (S)
        S = np.std(series, ddof=1)

        # Avoid division by zero
        if S == 0 or np.isnan(S):
            return 0.0

        # Return R/S statistic
        return R / S

    def calculate_hurst(
        self,
        prices: np.ndarray,
        method: str = 'rs',
        simplified: bool = False
    ) -> float:
        """
        Calculate Hurst Exponent using R/S analysis

        Args:
            prices: Array of price values
            method: Calculation method ('rs' for Rescaled Range)
            simplified: If True, use simplified single-window calculation

        Returns:
            Hurst exponent value (0-1, typically 0.3-0.7)
        """
        if len(prices) < self.min_window * 2:
            logger.warning(f"Insufficient data: {len(prices)} < {self.min_window * 2}")
            return 0.5  # Return neutral value

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < self.min_window:
            logger.warning("Insufficient valid returns for Hurst calculation")
            return 0.5

        if simplified:
            # Simplified approach: single window R/S
            rs_stat = self._rs_statistic(returns)
            if rs_stat <= 0:
                return 0.5

            # Estimate H from single R/S value
            # E[R/S] ≈ (n/2)^H for random walk
            n = len(returns)
            H = np.log(rs_stat) / np.log(n / 2)
            H = np.clip(H, 0.0, 1.0)

            logger.debug(f"Hurst (simplified): {H:.4f} from R/S={rs_stat:.4f}, n={n}")
            return float(H)

        # Full R/S analysis: multiple window sizes
        max_window = self.max_window or len(returns) // 2

        # Create logarithmically spaced window sizes
        # This gives better coverage across scales
        num_windows = min(10, (max_window - self.min_window) // 2)
        if num_windows < 3:
            # Fall back to simplified if not enough windows
            return self.calculate_hurst(prices, method='rs', simplified=True)

        window_sizes = np.logspace(
            np.log10(self.min_window),
            np.log10(max_window),
            num=num_windows,
            dtype=int
        )
        window_sizes = np.unique(window_sizes)  # Remove duplicates

        rs_values = []
        valid_windows = []

        for window in window_sizes:
            if window > len(returns):
                continue

            # Calculate R/S for this window by averaging over subseries
            rs_stats = []
            num_subseries = len(returns) // window

            for i in range(num_subseries):
                subseries = returns[i * window:(i + 1) * window]
                rs = self._rs_statistic(subseries)
                if rs > 0 and np.isfinite(rs):
                    rs_stats.append(rs)

            if len(rs_stats) > 0:
                avg_rs = np.mean(rs_stats)
                rs_values.append(avg_rs)
                valid_windows.append(window)

        if len(rs_values) < 3:
            logger.warning("Insufficient R/S values for regression, using simplified method")
            return self.calculate_hurst(prices, method='rs', simplified=True)

        # Estimate H from log-log regression
        # log(R/S) = log(c) + H * log(n)
        log_windows = np.log10(valid_windows)
        log_rs = np.log10(rs_values)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_windows, log_rs)

        # Hurst exponent is the slope
        H = slope
        H = np.clip(H, 0.0, 1.0)  # Bound to valid range

        logger.debug(
            f"Hurst exponent: {H:.4f} (R²={r_value**2:.3f}, "
            f"windows={len(valid_windows)}, p={p_value:.4f})"
        )

        return float(H)

    def interpret_hurst(self, hurst: float) -> Dict:
        """
        Interpret Hurst Exponent value for trading context

        Args:
            hurst: Hurst exponent value (0-1)

        Returns:
            Dictionary with interpretation:
                - behavior: str ('trending', 'random', 'mean_reverting')
                - persistence: str ('high', 'medium', 'low')
                - strategy: str (recommended trading strategy)
                - strength: float (0-1, strength of the signal)
        """
        # Hurst interpretation thresholds
        # Strong Mean-Reversion: 0.0 - 0.4
        # Weak Mean-Reversion: 0.4 - 0.5
        # Random Walk: 0.5
        # Weak Trending: 0.5 - 0.6
        # Strong Trending: 0.6 - 1.0

        if hurst < 0.4:
            behavior = 'mean_reverting'
            persistence = 'high'
            strategy = 'fade_extremes'  # Trade against the trend
            strength = (0.5 - hurst) * 2  # 0.4 → 0.2, 0.0 → 1.0
        elif hurst < 0.5:
            behavior = 'mean_reverting'
            persistence = 'medium'
            strategy = 'range_trading'
            strength = (0.5 - hurst) * 2  # 0.5 → 0.0, 0.4 → 0.2
        elif hurst < 0.6:
            behavior = 'trending'
            persistence = 'medium'
            strategy = 'momentum_following'
            strength = (hurst - 0.5) * 2  # 0.5 → 0.0, 0.6 → 0.2
        else:
            behavior = 'trending'
            persistence = 'high'
            strategy = 'trend_following'
            strength = (hurst - 0.5) * 2  # 0.6 → 0.2, 1.0 → 1.0

        # Cap strength at 1.0
        strength = min(strength, 1.0)

        return {
            'hurst': hurst,
            'behavior': behavior,
            'persistence': persistence,
            'strategy': strategy,
            'strength': strength,
            'is_trending': hurst > 0.5,
            'is_mean_reverting': hurst < 0.5,
            'timestamp': pd.Timestamp.now()
        }

    def calculate_rolling_hurst(
        self,
        prices: np.ndarray,
        window: int = 200,
        step: int = 10,
        simplified: bool = True
    ) -> np.ndarray:
        """
        Calculate rolling Hurst Exponent

        Args:
            prices: Array of price values
            window: Lookback window for each Hurst calculation
            step: Step size between calculations
            simplified: Use simplified calculation for speed

        Returns:
            Array of Hurst exponent values
        """
        hurst_values = []

        for i in range(window, len(prices), step):
            window_prices = prices[i - window:i]
            hurst = self.calculate_hurst(window_prices, simplified=simplified)
            hurst_values.append(hurst)

        return np.array(hurst_values)

    def analyze_market(
        self,
        prices: np.ndarray,
        simplified: bool = False
    ) -> Dict:
        """
        Complete market Hurst analysis

        Args:
            prices: Array of price values
            simplified: Use simplified calculation

        Returns:
            Dictionary with Hurst analysis results
        """
        hurst = self.calculate_hurst(prices, simplified=simplified)
        interpretation = self.interpret_hurst(hurst)

        # Additional metrics
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]

        result = {
            **interpretation,
            'num_prices': len(prices),
            'volatility': float(np.std(returns)) if len(returns) > 0 else 0.0,
            'mean_return': float(np.mean(returns)) if len(returns) > 0 else 0.0,
        }

        logger.info(
            f"Hurst Analysis: {hurst:.3f} | "
            f"Behavior: {interpretation['behavior']} | "
            f"Strategy: {interpretation['strategy']} | "
            f"Strength: {interpretation['strength']:.2f}"
        )

        return result


# Convenience function for V7 runtime
def calculate_hurst_exponent(
    prices: np.ndarray,
    simplified: bool = True,
    min_window: int = 10
) -> Dict:
    """
    Calculate Hurst exponent and return analysis

    Args:
        prices: Array of price values
        simplified: Use simplified (faster) calculation
        min_window: Minimum window size for R/S analysis

    Returns:
        Dictionary with Hurst analysis
    """
    analyzer = HurstExponentAnalyzer(min_window=min_window)
    return analyzer.analyze_market(prices, simplified=simplified)


if __name__ == "__main__":
    # Test Hurst Exponent implementation
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Hurst Exponent Analyzer - Test Run")
    print("=" * 80)

    # Generate test data
    np.random.seed(42)

    # Test 1: Random walk (H ≈ 0.5)
    print("\n1. Testing Random Walk (Expected: H ≈ 0.5)")
    random_walk = np.cumsum(np.random.randn(1000)) + 100
    analyzer = HurstExponentAnalyzer(min_window=10)
    result1 = analyzer.analyze_market(random_walk, simplified=False)
    print(f"   Hurst:      {result1['hurst']:.4f}")
    print(f"   Behavior:   {result1['behavior']}")
    print(f"   Strategy:   {result1['strategy']}")
    print(f"   Strength:   {result1['strength']:.2f}")

    # Test 2: Trending market (H > 0.5)
    print("\n2. Testing Trending Market (Expected: H > 0.5)")
    trend = np.cumsum(np.random.randn(1000) + 0.1) + 100  # Drift added
    result2 = analyzer.analyze_market(trend, simplified=False)
    print(f"   Hurst:      {result2['hurst']:.4f}")
    print(f"   Behavior:   {result2['behavior']}")
    print(f"   Strategy:   {result2['strategy']}")
    print(f"   Strength:   {result2['strength']:.2f}")

    # Test 3: Mean-reverting (H < 0.5)
    print("\n3. Testing Mean-Reverting Series (Expected: H < 0.5)")
    # Create mean-reverting series using AR(1) with negative coefficient
    mean_reverting = [100]
    for _ in range(999):
        # Mean reversion: X(t) = μ + φ(X(t-1) - μ) + ε, φ < 0
        mean_reverting.append(
            100 + (-0.3) * (mean_reverting[-1] - 100) + np.random.randn() * 2
        )
    mean_reverting = np.array(mean_reverting)
    result3 = analyzer.analyze_market(mean_reverting, simplified=False)
    print(f"   Hurst:      {result3['hurst']:.4f}")
    print(f"   Behavior:   {result3['behavior']}")
    print(f"   Strategy:   {result3['strategy']}")
    print(f"   Strength:   {result3['strength']:.2f}")

    # Test 4: Strong trending (H > 0.6)
    print("\n4. Testing Strong Trending Series (Expected: H > 0.6)")
    strong_trend = np.cumsum(np.random.randn(1000) + 0.5) + 100  # Strong drift
    result4 = analyzer.analyze_market(strong_trend, simplified=False)
    print(f"   Hurst:      {result4['hurst']:.4f}")
    print(f"   Behavior:   {result4['behavior']}")
    print(f"   Strategy:   {result4['strategy']}")
    print(f"   Strength:   {result4['strength']:.2f}")

    # Test 5: Rolling Hurst analysis
    print("\n5. Testing Rolling Hurst (Simplified Method)")
    # Mixed signal: trending → random → mean-reverting
    mixed_signal = np.concatenate([
        np.cumsum(np.random.randn(300) + 0.3) + 100,  # Trending
        np.cumsum(np.random.randn(400)) + 130,        # Random walk
        mean_reverting[600:900] + 30                   # Mean-reverting
    ])

    rolling_hurst = analyzer.calculate_rolling_hurst(
        mixed_signal,
        window=200,
        step=20,
        simplified=True
    )
    print(f"   Calculated {len(rolling_hurst)} Hurst values")
    print(f"   Min Hurst: {np.min(rolling_hurst):.4f}")
    print(f"   Max Hurst: {np.max(rolling_hurst):.4f}")
    print(f"   Avg Hurst: {np.mean(rolling_hurst):.4f}")

    # Visualization
    print("\n6. Creating Visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Random Walk
    axes[0, 0].plot(random_walk)
    axes[0, 0].set_title(f"Random Walk (H={result1['hurst']:.3f})")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Trending
    axes[0, 1].plot(trend)
    axes[0, 1].set_title(f"Trending Market (H={result2['hurst']:.3f})")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Price")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mean-Reverting
    axes[1, 0].plot(mean_reverting)
    axes[1, 0].set_title(f"Mean-Reverting (H={result3['hurst']:.3f})")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Price")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Rolling Hurst
    axes[1, 1].plot(rolling_hurst)
    axes[1, 1].axhline(y=0.5, color='black', linestyle='--', label='Random Walk', alpha=0.5)
    axes[1, 1].axhline(y=0.6, color='g', linestyle='--', label='Trending Threshold', alpha=0.5)
    axes[1, 1].axhline(y=0.4, color='r', linestyle='--', label='Mean-Reversion Threshold', alpha=0.5)
    axes[1, 1].set_title("Rolling Hurst Exponent (Mixed Signal)")
    axes[1, 1].set_xlabel("Window")
    axes[1, 1].set_ylabel("Hurst Exponent")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('/tmp/hurst_exponent_test.png', dpi=100, bbox_inches='tight')
    print(f"   Saved visualization to /tmp/hurst_exponent_test.png")

    print("\n" + "=" * 80)
    print("Hurst Exponent Test Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  - H ≈ 0.5: Random walk (no predictable pattern)")
    print("  - H > 0.5: Trending/persistent behavior (momentum)")
    print("  - H < 0.5: Mean-reverting/anti-persistent (reversals)")
    print("  - Use Hurst to select trading strategy (trend vs mean-reversion)")
    print("=" * 80)
