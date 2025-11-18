"""
Markov Chain Analysis for Market Regime Detection

A Markov Chain models the probability of transitioning between market regimes.
It assumes the current regime depends only on the previous regime (memoryless property).

6 Market Regimes:
1. BULL_TREND: Strong upward momentum, high returns
2. BEAR_TREND: Strong downward momentum, negative returns
3. HIGH_VOL_RANGE: Choppy, high volatility, no clear direction
4. LOW_VOL_RANGE: Quiet consolidation, low volatility
5. BREAKOUT: Volatility expansion with directional move
6. CONSOLIDATION: Tight range after major move

Transition Matrix P[i,j] = probability of moving from regime i to regime j
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
from collections import Counter

logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    """Enumeration of market regimes"""
    BULL_TREND = 0
    BEAR_TREND = 1
    HIGH_VOL_RANGE = 2
    LOW_VOL_RANGE = 3
    BREAKOUT = 4
    CONSOLIDATION = 5


# Regime metadata for interpretation
REGIME_METADATA = {
    MarketRegime.BULL_TREND: {
        'name': 'Bull Trend',
        'description': 'Strong upward momentum',
        'strategy': 'trend_following_long',
        'risk_level': 'medium'
    },
    MarketRegime.BEAR_TREND: {
        'name': 'Bear Trend',
        'description': 'Strong downward momentum',
        'strategy': 'trend_following_short',
        'risk_level': 'medium'
    },
    MarketRegime.HIGH_VOL_RANGE: {
        'name': 'High Volatility Range',
        'description': 'Choppy market, no clear direction',
        'strategy': 'mean_reversion',
        'risk_level': 'high'
    },
    MarketRegime.LOW_VOL_RANGE: {
        'name': 'Low Volatility Range',
        'description': 'Quiet consolidation',
        'strategy': 'wait_for_breakout',
        'risk_level': 'low'
    },
    MarketRegime.BREAKOUT: {
        'name': 'Breakout',
        'description': 'Volatility expansion with direction',
        'strategy': 'momentum_trading',
        'risk_level': 'high'
    },
    MarketRegime.CONSOLIDATION: {
        'name': 'Consolidation',
        'description': 'Tight range after major move',
        'strategy': 'range_trading',
        'risk_level': 'low'
    }
}


class MarkovRegimeDetector:
    """Markov Chain model for market regime detection"""

    def __init__(
        self,
        lookback_window: int = 100,
        trend_threshold: float = 0.02,  # 2% return threshold for trends
        vol_threshold: float = 0.03     # 3% volatility threshold
    ):
        """
        Initialize Markov Chain regime detector

        Args:
            lookback_window: Window for calculating regime metrics
            trend_threshold: Return threshold for trend classification
            vol_threshold: Volatility threshold for high/low vol classification
        """
        self.lookback_window = lookback_window
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold

        # Transition matrix: P[i,j] = prob(state j | state i)
        # Initially uniform (will be learned from data)
        self.transition_matrix = np.ones((6, 6)) / 6.0

        # Stationary distribution (equilibrium probabilities)
        self.stationary_dist = np.ones(6) / 6.0

        logger.debug(
            f"Markov regime detector initialized: "
            f"lookback={lookback_window}, trend_thresh={trend_threshold}, "
            f"vol_thresh={vol_threshold}"
        )

    def _classify_regime(
        self,
        returns: np.ndarray,
        volatility: float,
        price_range: float,
        prev_regime: Optional[int] = None
    ) -> int:
        """
        Classify current market regime based on metrics

        Args:
            returns: Recent returns
            volatility: Current volatility (std of returns)
            price_range: Recent price range (high - low) / mean
            prev_regime: Previous regime (for transition detection)

        Returns:
            Regime ID (0-5)
        """
        # Calculate mean return over window
        mean_return = np.mean(returns)

        # Trend detection (based on mean return)
        is_bullish = mean_return > self.trend_threshold
        is_bearish = mean_return < -self.trend_threshold

        # Volatility classification
        is_high_vol = volatility > self.vol_threshold
        is_low_vol = volatility < self.vol_threshold / 2

        # Breakout detection: sudden volatility increase
        recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else volatility
        is_breakout = recent_vol > volatility * 1.5 and abs(mean_return) > 0.01

        # Consolidation detection: tight range after move
        is_consolidating = price_range < 0.02 and volatility < self.vol_threshold / 2

        # Regime classification logic
        if is_breakout:
            return MarketRegime.BREAKOUT
        elif is_consolidating:
            return MarketRegime.CONSOLIDATION
        elif is_bullish and not is_high_vol:
            return MarketRegime.BULL_TREND
        elif is_bearish and not is_high_vol:
            return MarketRegime.BEAR_TREND
        elif is_high_vol:
            return MarketRegime.HIGH_VOL_RANGE
        else:
            return MarketRegime.LOW_VOL_RANGE

    def detect_regime(
        self,
        prices: np.ndarray,
        prev_regime: Optional[int] = None
    ) -> int:
        """
        Detect current market regime

        Args:
            prices: Array of recent prices
            prev_regime: Previous regime (optional)

        Returns:
            Current regime ID (0-5)
        """
        if len(prices) < 20:
            logger.warning(f"Insufficient data: {len(prices)} < 20")
            return MarketRegime.LOW_VOL_RANGE  # Default to low-vol range

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            return MarketRegime.LOW_VOL_RANGE

        # Calculate metrics
        volatility = float(np.std(returns))
        price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)

        # Classify regime
        regime = self._classify_regime(
            returns=returns,
            volatility=volatility,
            price_range=price_range,
            prev_regime=prev_regime
        )

        logger.debug(
            f"Regime detected: {REGIME_METADATA[regime]['name']} "
            f"(vol={volatility:.4f}, range={price_range:.4f})"
        )

        return int(regime)

    def detect_regime_sequence(
        self,
        prices: np.ndarray,
        window: int = 50,
        step: int = 1
    ) -> np.ndarray:
        """
        Detect regime sequence over time (for learning transitions)

        Args:
            prices: Full price series
            window: Window size for each detection
            step: Step size between detections

        Returns:
            Array of regime IDs
        """
        regimes = []
        prev_regime = None

        for i in range(window, len(prices), step):
            window_prices = prices[i - window:i]
            regime = self.detect_regime(window_prices, prev_regime)
            regimes.append(regime)
            prev_regime = regime

        return np.array(regimes)

    def learn_transition_matrix(
        self,
        prices: np.ndarray,
        window: int = 50,
        step: int = 1
    ) -> np.ndarray:
        """
        Learn transition matrix from historical price data

        Args:
            prices: Historical price series
            window: Window size for regime detection
            step: Step size

        Returns:
            6x6 transition matrix
        """
        # Detect regime sequence
        regimes = self.detect_regime_sequence(prices, window, step)

        if len(regimes) < 2:
            logger.warning("Insufficient regimes detected for learning")
            return self.transition_matrix

        # Count transitions
        transition_counts = np.zeros((6, 6))

        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            transition_counts[current_regime, next_regime] += 1

        # Calculate transition probabilities
        # P[i,j] = count(i→j) / sum(count(i→*))
        for i in range(6):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                self.transition_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # If no transitions from this state, use uniform
                self.transition_matrix[i, :] = 1.0 / 6.0

        logger.info(
            f"Learned transition matrix from {len(regimes)} regimes, "
            f"{len(regimes) - 1} transitions"
        )

        # Calculate stationary distribution
        self._calculate_stationary_distribution()

        return self.transition_matrix

    def _calculate_stationary_distribution(self):
        """
        Calculate stationary distribution (equilibrium probabilities)

        π = πP, where π is the stationary distribution
        Solved as the eigenvector of P^T with eigenvalue 1
        """
        try:
            # Find eigenvectors of transpose
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

            # Find eigenvector corresponding to eigenvalue ≈ 1
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])

            # Normalize to sum to 1
            stationary = stationary / np.sum(stationary)

            # Ensure non-negative
            stationary = np.abs(stationary)
            stationary = stationary / np.sum(stationary)

            self.stationary_dist = stationary

            logger.debug(f"Stationary distribution: {self.stationary_dist}")

        except Exception as e:
            logger.warning(f"Failed to calculate stationary distribution: {e}")
            # Fall back to uniform
            self.stationary_dist = np.ones(6) / 6.0

    def predict_next_regime(
        self,
        current_regime: int,
        n_steps: int = 1
    ) -> Dict:
        """
        Predict next regime(s) using transition matrix

        Args:
            current_regime: Current regime ID
            n_steps: Number of steps ahead to predict

        Returns:
            Dictionary with predictions and probabilities
        """
        if n_steps == 1:
            # Direct transition
            probabilities = self.transition_matrix[current_regime, :]
            most_likely = int(np.argmax(probabilities))

            return {
                'most_likely_regime': most_likely,
                'regime_name': REGIME_METADATA[most_likely]['name'],
                'probability': float(probabilities[most_likely]),
                'all_probabilities': probabilities.tolist()
            }
        else:
            # Multi-step transition: P^n
            P_n = np.linalg.matrix_power(self.transition_matrix, n_steps)
            probabilities = P_n[current_regime, :]
            most_likely = int(np.argmax(probabilities))

            return {
                'most_likely_regime': most_likely,
                'regime_name': REGIME_METADATA[most_likely]['name'],
                'probability': float(probabilities[most_likely]),
                'steps_ahead': n_steps,
                'all_probabilities': probabilities.tolist()
            }

    def interpret_regime(self, regime: int) -> Dict:
        """
        Get interpretation of a regime for trading

        Args:
            regime: Regime ID

        Returns:
            Dictionary with regime metadata
        """
        metadata = REGIME_METADATA[regime].copy()
        metadata['regime_id'] = int(regime)
        metadata['stationary_probability'] = float(self.stationary_dist[regime])

        return metadata

    def analyze_market(
        self,
        prices: np.ndarray,
        learn_transitions: bool = True
    ) -> Dict:
        """
        Complete market regime analysis

        Args:
            prices: Price series
            learn_transitions: Whether to learn transition matrix

        Returns:
            Dictionary with regime analysis
        """
        # Learn transition matrix if requested
        if learn_transitions and len(prices) >= 100:
            self.learn_transition_matrix(prices)

        # Detect current regime
        current_regime = self.detect_regime(prices[-self.lookback_window:])

        # Get interpretation
        interpretation = self.interpret_regime(current_regime)

        # Predict next regime
        prediction = self.predict_next_regime(current_regime)

        # Additional metrics
        regimes = self.detect_regime_sequence(prices, window=50, step=10)
        regime_counts = Counter(regimes)

        result = {
            'current_regime': current_regime,
            'regime_name': interpretation['name'],
            'description': interpretation['description'],
            'strategy': interpretation['strategy'],
            'risk_level': interpretation['risk_level'],
            'stationary_prob': interpretation['stationary_probability'],
            'next_regime_prediction': prediction,
            'regime_distribution': {
                REGIME_METADATA[i]['name']: int(regime_counts.get(i, 0))
                for i in range(6)
            },
            'transition_matrix': self.transition_matrix.tolist(),
            'timestamp': pd.Timestamp.now()
        }

        logger.info(
            f"Market Regime: {interpretation['name']} | "
            f"Strategy: {interpretation['strategy']} | "
            f"Risk: {interpretation['risk_level']} | "
            f"Next: {prediction['regime_name']} ({prediction['probability']:.2%})"
        )

        return result


# Convenience function for V7 runtime
def detect_market_regime(
    prices: np.ndarray,
    learn_transitions: bool = True,
    lookback_window: int = 100
) -> Dict:
    """
    Detect market regime and return analysis

    Args:
        prices: Price series
        learn_transitions: Learn transitions from data
        lookback_window: Window for regime detection

    Returns:
        Dictionary with regime analysis
    """
    detector = MarkovRegimeDetector(lookback_window=lookback_window)
    return detector.analyze_market(prices, learn_transitions=learn_transitions)


if __name__ == "__main__":
    # Test Markov Chain implementation
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Markov Chain Regime Detector - Test Run")
    print("=" * 80)

    # Generate synthetic market data with regime changes
    np.random.seed(42)

    # Create price series with different regimes
    price_series = [100.0]

    # Bull trend (200 candles)
    for _ in range(200):
        price_series.append(price_series[-1] * (1 + np.random.normal(0.002, 0.01)))

    # High vol range (150 candles)
    for _ in range(150):
        price_series.append(price_series[-1] * (1 + np.random.normal(0, 0.03)))

    # Bear trend (200 candles)
    for _ in range(200):
        price_series.append(price_series[-1] * (1 + np.random.normal(-0.002, 0.01)))

    # Low vol consolidation (150 candles)
    consolidation_center = price_series[-1]
    for _ in range(150):
        price_series.append(consolidation_center + np.random.normal(0, 2))

    # Breakout (100 candles)
    for _ in range(100):
        price_series.append(price_series[-1] * (1 + np.random.normal(0.005, 0.02)))

    prices = np.array(price_series)

    # Test regime detection
    print("\n1. Testing Regime Detection")
    detector = MarkovRegimeDetector()

    # Detect regime sequence
    regimes = detector.detect_regime_sequence(prices, window=50, step=10)
    print(f"   Detected {len(regimes)} regimes over {len(prices)} prices")

    # Count each regime
    regime_counts = Counter(regimes)
    print("\n   Regime Distribution:")
    for regime_id, count in sorted(regime_counts.items()):
        name = REGIME_METADATA[regime_id]['name']
        percentage = (count / len(regimes)) * 100
        print(f"     {name:25s}: {count:3d} ({percentage:5.1f}%)")

    # Test transition matrix learning
    print("\n2. Testing Transition Matrix Learning")
    transition_matrix = detector.learn_transition_matrix(prices, window=50, step=10)
    print(f"   Learned transition matrix (shape: {transition_matrix.shape})")
    print("\n   Transition probabilities (top 3 per regime):")
    for i in range(6):
        from_regime = REGIME_METADATA[i]['name']
        # Get top 3 transitions
        top_indices = np.argsort(transition_matrix[i, :])[-3:][::-1]
        print(f"\n     From {from_regime}:")
        for idx in top_indices:
            to_regime = REGIME_METADATA[idx]['name']
            prob = transition_matrix[i, idx]
            print(f"       → {to_regime:25s}: {prob:.2%}")

    # Test current regime analysis
    print("\n3. Testing Current Regime Analysis")
    analysis = detector.analyze_market(prices, learn_transitions=False)
    print(f"   Current Regime:  {analysis['regime_name']}")
    print(f"   Description:     {analysis['description']}")
    print(f"   Strategy:        {analysis['strategy']}")
    print(f"   Risk Level:      {analysis['risk_level']}")
    print(f"   Next Predicted:  {analysis['next_regime_prediction']['regime_name']}")
    print(f"   Probability:     {analysis['next_regime_prediction']['probability']:.2%}")

    # Test multi-step prediction
    print("\n4. Testing Multi-Step Prediction (5 steps ahead)")
    current_regime = analysis['current_regime']
    prediction_5 = detector.predict_next_regime(current_regime, n_steps=5)
    print(f"   Most Likely:     {prediction_5['regime_name']}")
    print(f"   Probability:     {prediction_5['probability']:.2%}")

    # Visualization
    print("\n5. Creating Visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Price series with regime colors
    ax1 = axes[0]
    regime_colors = {
        0: 'green',      # Bull trend
        1: 'red',        # Bear trend
        2: 'orange',     # High vol range
        3: 'blue',       # Low vol range
        4: 'purple',     # Breakout
        5: 'gray'        # Consolidation
    }

    ax1.plot(prices, color='black', alpha=0.5, linewidth=0.5, label='Price')

    # Color background by regime
    regime_indices = np.arange(50, len(prices), 10)[:len(regimes)]
    for i in range(len(regimes)):
        if i < len(regime_indices):
            start_idx = regime_indices[i]
            end_idx = regime_indices[i+1] if i+1 < len(regime_indices) else len(prices)
            regime = regimes[i]
            ax1.axvspan(start_idx, end_idx, alpha=0.2, color=regime_colors[regime])

    ax1.set_title("Price Series with Market Regimes")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=regime_colors[i], alpha=0.5, label=REGIME_METADATA[i]['name'])
        for i in range(6)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # Plot 2: Regime sequence
    ax2 = axes[1]
    ax2.plot(regime_indices, regimes, drawstyle='steps-post', linewidth=2)
    ax2.set_title("Regime Sequence Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Regime ID")
    ax2.set_yticks(range(6))
    ax2.set_yticklabels([REGIME_METADATA[i]['name'] for i in range(6)], fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Transition matrix heatmap
    ax3 = axes[2]
    im = ax3.imshow(transition_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
    ax3.set_xticks(range(6))
    ax3.set_yticks(range(6))
    ax3.set_xticklabels([REGIME_METADATA[i]['name'] for i in range(6)], rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels([REGIME_METADATA[i]['name'] for i in range(6)], fontsize=8)
    ax3.set_title("Transition Probability Matrix")
    ax3.set_xlabel("To Regime")
    ax3.set_ylabel("From Regime")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Transition Probability')

    # Add transition probabilities as text
    for i in range(6):
        for j in range(6):
            text = ax3.text(j, i, f'{transition_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=6)

    plt.tight_layout()
    plt.savefig('/tmp/markov_regime_test.png', dpi=100, bbox_inches='tight')
    print(f"   Saved visualization to /tmp/markov_regime_test.png")

    print("\n" + "=" * 80)
    print("Markov Chain Test Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  - 6 distinct market regimes detected")
    print("  - Transition probabilities learned from data")
    print("  - Can predict likely next regime")
    print("  - Trading strategies adapt to current regime")
    print("=" * 80)
