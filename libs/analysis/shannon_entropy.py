"""
Shannon Entropy Analysis for Market Predictability

Shannon Entropy measures the uncertainty/randomness in a price series.
- High entropy (→1.0) = Unpredictable, random movements (hard to trade)
- Low entropy (→0.0) = Predictable patterns (easier to trade)

Formula: H(X) = -Σ p(x_i) * log2(p(x_i))

where p(x_i) is the probability of return being in bin i
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class ShannonEntropyAnalyzer:
    """Calculate Shannon Entropy for market predictability analysis"""

    def __init__(self, num_bins: int = 10):
        """
        Initialize Shannon Entropy Analyzer

        Args:
            num_bins: Number of bins to discretize returns (default: 10)
                     More bins = finer granularity but needs more data
        """
        self.num_bins = num_bins
        logger.debug(f"Shannon Entropy Analyzer initialized with {num_bins} bins")

    def calculate_entropy(
        self,
        prices: np.ndarray,
        window: int = 100,
        normalize: bool = True
    ) -> float:
        """
        Calculate Shannon Entropy for a price series

        Args:
            prices: Array of price values
            window: Lookback window for entropy calculation
            normalize: If True, normalize entropy to [0, 1] range

        Returns:
            Entropy value (0-1 if normalized, 0-log2(num_bins) otherwise)
        """
        if len(prices) < window:
            logger.warning(f"Insufficient data: {len(prices)} < {window}")
            return 0.5 if normalize else np.log2(self.num_bins) / 2

        # Use last 'window' prices
        recent_prices = prices[-window:]

        # Calculate returns
        returns = np.diff(recent_prices) / recent_prices[:-1]

        # Remove any NaN or inf values
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            logger.warning("No valid returns for entropy calculation")
            return 0.5 if normalize else np.log2(self.num_bins) / 2

        # Discretize returns into bins
        hist, _ = np.histogram(returns, bins=self.num_bins)

        # Calculate probabilities
        probabilities = hist / np.sum(hist)

        # Remove zero probabilities (log(0) undefined)
        probabilities = probabilities[probabilities > 0]

        # Calculate Shannon Entropy: H(X) = -Σ p(x) * log2(p(x))
        entropy = -np.sum(probabilities * np.log2(probabilities))

        if normalize:
            # Normalize to [0, 1] range
            # Max entropy = log2(num_bins) (uniform distribution)
            max_entropy = np.log2(self.num_bins)
            entropy = entropy / max_entropy

        logger.debug(f"Calculated entropy: {entropy:.4f} (normalized={normalize})")
        return float(entropy)

    def calculate_rolling_entropy(
        self,
        prices: np.ndarray,
        window: int = 100,
        step: int = 1
    ) -> np.ndarray:
        """
        Calculate rolling Shannon Entropy

        Args:
            prices: Array of price values
            window: Lookback window for each entropy calculation
            step: Step size between calculations

        Returns:
            Array of entropy values (normalized to [0, 1])
        """
        entropies = []

        for i in range(window, len(prices), step):
            window_prices = prices[i-window:i]
            entropy = self.calculate_entropy(window_prices, window, normalize=True)
            entropies.append(entropy)

        return np.array(entropies)

    def interpret_entropy(self, entropy: float) -> Dict:
        """
        Interpret Shannon Entropy value for trading context

        Args:
            entropy: Normalized entropy value (0-1)

        Returns:
            Dictionary with interpretation:
                - predictability: str ('high', 'medium', 'low')
                - confidence_impact: str (trading confidence adjustment)
                - regime: str (market regime)
                - trading_difficulty: str ('easy', 'moderate', 'hard')
        """
        # Entropy thresholds
        # Low: 0.0 - 0.4 (predictable)
        # Med: 0.4 - 0.7 (mixed)
        # High: 0.7 - 1.0 (random)

        if entropy < 0.4:
            predictability = 'high'
            confidence_impact = 'boost'  # Increase signal confidence
            regime = 'trending'
            difficulty = 'easy'
        elif entropy < 0.7:
            predictability = 'medium'
            confidence_impact = 'neutral'
            regime = 'mixed'
            difficulty = 'moderate'
        else:
            predictability = 'low'
            confidence_impact = 'reduce'  # Decrease signal confidence
            regime = 'random_walk'
            difficulty = 'hard'

        return {
            'entropy': entropy,
            'predictability': predictability,
            'confidence_impact': confidence_impact,
            'regime': regime,
            'trading_difficulty': difficulty,
            'timestamp': pd.Timestamp.now()
        }

    def analyze_market(
        self,
        prices: np.ndarray,
        window: int = 100
    ) -> Dict:
        """
        Complete market entropy analysis

        Args:
            prices: Array of price values
            window: Lookback window

        Returns:
            Dictionary with entropy analysis results
        """
        entropy = self.calculate_entropy(prices, window, normalize=True)
        interpretation = self.interpret_entropy(entropy)

        # Additional metrics
        recent_prices = prices[-window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        returns = returns[np.isfinite(returns)]

        result = {
            **interpretation,
            'window_size': window,
            'num_prices': len(prices),
            'volatility': float(np.std(returns)) if len(returns) > 0 else 0.0,
            'mean_return': float(np.mean(returns)) if len(returns) > 0 else 0.0,
        }

        logger.info(
            f"Market Entropy Analysis: {entropy:.3f} | "
            f"Predictability: {interpretation['predictability']} | "
            f"Regime: {interpretation['regime']}"
        )

        return result


# Convenience function for V7 runtime
def calculate_market_entropy(
    prices: np.ndarray,
    window: int = 100,
    num_bins: int = 10
) -> Dict:
    """
    Calculate market entropy and return analysis

    Args:
        prices: Array of price values
        window: Lookback window
        num_bins: Number of bins for discretization

    Returns:
        Dictionary with entropy analysis
    """
    analyzer = ShannonEntropyAnalyzer(num_bins=num_bins)
    return analyzer.analyze_market(prices, window)


if __name__ == "__main__":
    # Test Shannon Entropy implementation
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
    print("Shannon Entropy Analyzer - Test Run")
    print("=" * 80)

    # Generate test data
    np.random.seed(42)

    # Test 1: Random walk (high entropy)
    print("\n1. Testing Random Walk (Expected: High Entropy)")
    random_walk = np.cumsum(np.random.randn(1000)) + 100
    analyzer = ShannonEntropyAnalyzer(num_bins=10)
    result1 = analyzer.analyze_market(random_walk, window=100)
    print(f"   Entropy:         {result1['entropy']:.4f}")
    print(f"   Predictability:  {result1['predictability']}")
    print(f"   Regime:          {result1['regime']}")
    print(f"   Difficulty:      {result1['trading_difficulty']}")

    # Test 2: Trending market (low entropy)
    print("\n2. Testing Trending Market (Expected: Low Entropy)")
    trend = np.linspace(100, 150, 1000) + np.random.randn(1000) * 0.5
    result2 = analyzer.analyze_market(trend, window=100)
    print(f"   Entropy:         {result2['entropy']:.4f}")
    print(f"   Predictability:  {result2['predictability']}")
    print(f"   Regime:          {result2['regime']}")
    print(f"   Difficulty:      {result2['trading_difficulty']}")

    # Test 3: Sinusoidal pattern (medium entropy)
    print("\n3. Testing Sinusoidal Pattern (Expected: Medium Entropy)")
    t = np.linspace(0, 4*np.pi, 1000)
    sine_wave = 100 + 10 * np.sin(t) + np.random.randn(1000) * 0.3
    result3 = analyzer.analyze_market(sine_wave, window=100)
    print(f"   Entropy:         {result3['entropy']:.4f}")
    print(f"   Predictability:  {result3['predictability']}")
    print(f"   Regime:          {result3['regime']}")
    print(f"   Difficulty:      {result3['trading_difficulty']}")

    # Test 4: Rolling entropy analysis
    print("\n4. Testing Rolling Entropy")
    mixed_signal = np.concatenate([
        trend[:300],         # Low entropy (trending)
        random_walk[300:600],  # High entropy (random)
        sine_wave[600:900]     # Medium entropy (cyclical)
    ])

    rolling_entropy = analyzer.calculate_rolling_entropy(mixed_signal, window=100, step=10)
    print(f"   Calculated {len(rolling_entropy)} entropy values")
    print(f"   Min Entropy: {np.min(rolling_entropy):.4f}")
    print(f"   Max Entropy: {np.max(rolling_entropy):.4f}")
    print(f"   Avg Entropy: {np.mean(rolling_entropy):.4f}")

    # Visualization
    print("\n5. Creating Visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Random Walk
    axes[0, 0].plot(random_walk)
    axes[0, 0].set_title(f"Random Walk (Entropy: {result1['entropy']:.3f})")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Trend
    axes[0, 1].plot(trend)
    axes[0, 1].set_title(f"Trending Market (Entropy: {result2['entropy']:.3f})")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Price")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Sine Wave
    axes[1, 0].plot(sine_wave)
    axes[1, 0].set_title(f"Cyclical Pattern (Entropy: {result3['entropy']:.3f})")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Price")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Rolling Entropy
    axes[1, 1].plot(rolling_entropy)
    axes[1, 1].axhline(y=0.4, color='g', linestyle='--', label='Low Threshold', alpha=0.5)
    axes[1, 1].axhline(y=0.7, color='r', linestyle='--', label='High Threshold', alpha=0.5)
    axes[1, 1].set_title("Rolling Entropy (Mixed Signal)")
    axes[1, 1].set_xlabel("Window")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/shannon_entropy_test.png', dpi=100, bbox_inches='tight')
    print(f"   Saved visualization to /tmp/shannon_entropy_test.png")

    print("\n" + "=" * 80)
    print("Shannon Entropy Test Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  - Random walks show HIGH entropy (unpredictable)")
    print("  - Trending markets show LOW entropy (predictable)")
    print("  - Cyclical patterns show MEDIUM entropy")
    print("  - Entropy can be used to adjust trading confidence")
    print("=" * 80)
