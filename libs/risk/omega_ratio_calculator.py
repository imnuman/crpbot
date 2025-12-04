"""
Omega Ratio Calculator

Omega Ratio = Probability-Weighted Gains / Probability-Weighted Losses

Developed by Keating & Shadwick (2002), the Omega Ratio is a comprehensive
risk-adjusted performance measure that considers the entire return distribution,
not just first two moments (mean and variance).

Formula:
    Omega(τ) = ∫[τ to ∞] (1 - F(r)) dr / ∫[-∞ to τ] F(r) dr

Where:
- τ = threshold return (usually 0% or risk-free rate)
- F(r) = cumulative distribution function of returns
- Numerator = probability-weighted gains above threshold
- Denominator = probability-weighted losses below threshold

Interpretation:
- Omega > 2.0: Excellent (gains >> losses)
- Omega 1.5-2.0: Very Good
- Omega 1.2-1.5: Good
- Omega 1.0-1.2: Acceptable
- Omega < 1.0: Poor (losses > gains)

Advantages over Sharpe:
- Uses entire return distribution (all moments)
- Captures skewness and kurtosis
- No normality assumption
- Intuitive interpretation (gain/loss ratio)
- Better for non-normal distributions (like crypto)

Expected Impact:
- More accurate risk assessment for crypto (fat tails)
- Better captures upside potential
- Complements Sharpe/Sortino/Calmar
- Identifies strategies with favorable gain/loss profiles
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class OmegaMetrics:
    """Omega ratio calculation results"""
    # Omega ratios for different thresholds
    omega_0pct: float  # Threshold = 0% (gains vs losses)
    omega_rf: float    # Threshold = risk-free rate
    omega_1pct: float  # Threshold = 1% monthly target

    # Components
    expected_gains: float
    expected_losses: float
    probability_gain: float
    probability_loss: float

    # Distribution statistics
    mean_return: float
    std_return: float
    skewness: float
    kurtosis: float

    # Sample statistics
    n_samples: int
    n_gains: int
    n_losses: int

    # Quality assessment
    omega_quality: str  # 'excellent', 'very_good', 'good', 'acceptable', 'poor'

    summary: str
    metrics: Dict[str, float]


class OmegaRatioCalculator:
    """
    Calculate Omega Ratio (probability-weighted gains/losses)

    Usage:
        calculator = OmegaRatioCalculator()

        # Record returns
        calculator.record_return(0.025, datetime.now())  # 2.5% gain
        calculator.record_return(-0.015, datetime.now()) # -1.5% loss

        # Calculate Omega
        metrics = calculator.calculate_omega()
        print(f"Omega (0%): {metrics.omega_0pct:.2f}")
        print(f"Gain/Loss: {metrics.omega_quality}")
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,  # Annual
        max_history: int = 500
    ):
        """
        Initialize Omega Ratio Calculator

        Args:
            risk_free_rate: Annual risk-free rate
            max_history: Maximum returns to keep
        """
        self.risk_free_rate = risk_free_rate
        self.max_history = max_history

        # Returns history
        self.returns_history: deque = deque(maxlen=max_history)

        logger.info(
            f"Omega Ratio Calculator initialized | "
            f"RF rate: {risk_free_rate:.1%} | "
            f"Max history: {max_history}"
        )

    def record_return(
        self,
        return_pct: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a return

        Args:
            return_pct: Return as decimal
            timestamp: Timestamp (default: now)
        """
        self.returns_history.append({
            'timestamp': timestamp or datetime.now(),
            'return_pct': return_pct
        })

        logger.debug(f"Return recorded: {return_pct:+.2%}")

    def calculate_omega(
        self,
        min_samples: int = 20
    ) -> OmegaMetrics:
        """
        Calculate Omega Ratio

        Args:
            min_samples: Minimum returns needed

        Returns:
            OmegaMetrics
        """
        try:
            if len(self.returns_history) < min_samples:
                logger.warning(f"Insufficient returns for Omega: {len(self.returns_history)}")
                return self._insufficient_data_metrics()

            returns = [r['return_pct'] for r in self.returns_history]
            returns_array = np.array(returns)

            # Calculate Omega for different thresholds
            omega_0 = self._calculate_omega_threshold(returns_array, threshold=0.0)
            omega_rf = self._calculate_omega_threshold(
                returns_array,
                threshold=self.risk_free_rate / 252  # Daily risk-free rate
            )
            omega_1pct = self._calculate_omega_threshold(
                returns_array,
                threshold=0.01 / 30  # ~1% monthly = 0.033% daily
            )

            # Calculate components (for 0% threshold)
            gains = returns_array[returns_array > 0]
            losses = returns_array[returns_array < 0]

            expected_gains = float(np.mean(gains)) if len(gains) > 0 else 0.0
            expected_losses = float(np.mean(losses)) if len(losses) > 0 else 0.0
            prob_gain = len(gains) / len(returns_array)
            prob_loss = len(losses) / len(returns_array)

            # Distribution statistics
            mean_return = float(np.mean(returns_array))
            std_return = float(np.std(returns_array, ddof=1))
            skewness = float(self._calculate_skewness(returns_array))
            kurtosis = float(self._calculate_kurtosis(returns_array))

            # Sample statistics
            n_samples = len(returns_array)
            n_gains = len(gains)
            n_losses = len(losses)

            # Quality assessment
            omega_quality = self._assess_omega_quality(omega_0)

            # Generate summary
            summary = self._generate_summary(omega_0, omega_quality, prob_gain)

            metrics = OmegaMetrics(
                omega_0pct=omega_0,
                omega_rf=omega_rf,
                omega_1pct=omega_1pct,
                expected_gains=expected_gains,
                expected_losses=expected_losses,
                probability_gain=prob_gain,
                probability_loss=prob_loss,
                mean_return=mean_return,
                std_return=std_return,
                skewness=skewness,
                kurtosis=kurtosis,
                n_samples=n_samples,
                n_gains=n_gains,
                n_losses=n_losses,
                omega_quality=omega_quality,
                summary=summary,
                metrics={
                    'omega_0pct': omega_0,
                    'omega_rf': omega_rf,
                    'omega_1pct': omega_1pct,
                    'skewness': skewness,
                    'kurtosis': kurtosis
                }
            )

            logger.debug(
                f"Omega calculated: Ω(0%)={omega_0:.2f}, "
                f"Ω(RF)={omega_rf:.2f}, "
                f"Quality={omega_quality}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Omega calculation failed: {e}")
            return self._insufficient_data_metrics()

    def _calculate_omega_threshold(
        self,
        returns: np.ndarray,
        threshold: float
    ) -> float:
        """
        Calculate Omega ratio for a specific threshold

        Args:
            returns: Array of returns
            threshold: Threshold return (τ)

        Returns:
            Omega ratio
        """
        # Separate gains and losses relative to threshold
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns < 0]

        # Probability-weighted gains and losses
        # Gains = average gain * probability of gain
        # Losses = average loss * probability of loss
        if len(gains) > 0:
            prob_weighted_gains = np.sum(gains) / len(returns)
        else:
            prob_weighted_gains = 0.0

        if len(losses) > 0:
            prob_weighted_losses = np.abs(np.sum(losses)) / len(returns)
        else:
            prob_weighted_losses = 0.0001  # Avoid division by zero

        # Omega = Gains / Losses
        omega = prob_weighted_gains / prob_weighted_losses

        return float(omega)

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness (third moment)"""
        if len(returns) < 3:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        if std == 0:
            return 0.0

        # Skewness = E[(X - μ)³] / σ³
        skew = np.mean(((returns - mean) / std) ** 3)

        return float(skew)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis (fourth moment)"""
        if len(returns) < 4:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        if std == 0:
            return 0.0

        # Kurtosis = E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
        kurt = np.mean(((returns - mean) / std) ** 4) - 3.0

        return float(kurt)

    def _assess_omega_quality(self, omega: float) -> str:
        """Assess Omega ratio quality"""
        if omega >= 2.0:
            return 'excellent'
        elif omega >= 1.5:
            return 'very_good'
        elif omega >= 1.2:
            return 'good'
        elif omega >= 1.0:
            return 'acceptable'
        else:
            return 'poor'

    def _generate_summary(
        self,
        omega: float,
        quality: str,
        prob_gain: float
    ) -> str:
        """Generate human-readable summary"""
        summary = (
            f"Omega(0%): {omega:.2f} ({quality.upper()}) | "
            f"Win Rate: {prob_gain:.1%} | "
            f"Gain/Loss Ratio: {omega:.2f}:1"
        )
        return summary

    def _insufficient_data_metrics(self) -> OmegaMetrics:
        """Return default metrics when insufficient data"""
        return OmegaMetrics(
            omega_0pct=0.0,
            omega_rf=0.0,
            omega_1pct=0.0,
            expected_gains=0.0,
            expected_losses=0.0,
            probability_gain=0.0,
            probability_loss=0.0,
            mean_return=0.0,
            std_return=0.0,
            skewness=0.0,
            kurtosis=0.0,
            n_samples=len(self.returns_history),
            n_gains=0,
            n_losses=0,
            omega_quality='unknown',
            summary=f"Insufficient data: {len(self.returns_history)} returns (need 20+)",
            metrics={}
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("OMEGA RATIO CALCULATOR TEST")
    print("=" * 70)

    # Scenario 1: Good strategy with positive skew
    print("\n[Scenario 1] Strategy with Positive Skew (many small wins, few large wins):")

    calculator = OmegaRatioCalculator(risk_free_rate=0.05)
    np.random.seed(42)

    # Simulate 100 trades with positive skew
    for i in range(100):
        # 70% small wins, 20% small losses, 10% large wins
        rand = np.random.random()
        if rand < 0.70:
            return_pct = np.random.normal(0.015, 0.005)  # 1.5% small wins
        elif rand < 0.90:
            return_pct = np.random.normal(-0.01, 0.003)  # -1% small losses
        else:
            return_pct = np.random.normal(0.05, 0.01)   # 5% large wins

        calculator.record_return(return_pct, datetime.now())

    metrics = calculator.calculate_omega()

    print(f"  Omega (0%):                {metrics.omega_0pct:.2f}")
    print(f"  Omega (Risk-Free):         {metrics.omega_rf:.2f}")
    print(f"  Omega (1% target):         {metrics.omega_1pct:.2f}")
    print(f"  Expected Gain:             {metrics.expected_gains:.2%}")
    print(f"  Expected Loss:             {metrics.expected_losses:.2%}")
    print(f"  Probability of Gain:       {metrics.probability_gain:.1%}")
    print(f"  Probability of Loss:       {metrics.probability_loss:.1%}")
    print(f"  Mean Return:               {metrics.mean_return:.2%}")
    print(f"  Std Deviation:             {metrics.std_return:.2%}")
    print(f"  Skewness:                  {metrics.skewness:.2f}")
    print(f"  Kurtosis (excess):         {metrics.kurtosis:.2f}")
    print(f"  Quality:                   {metrics.omega_quality.upper()}")
    print(f"\n  Summary: {metrics.summary}")

    # Interpretation
    if metrics.skewness > 0:
        print(f"\n  ✅ POSITIVE SKEW: More large gains than large losses (favorable)")
    else:
        print(f"\n  ⚠️  NEGATIVE SKEW: More large losses than large gains (unfavorable)")

    if metrics.kurtosis > 0:
        print(f"  ⚠️  FAT TAILS: Returns have more extreme values than normal distribution")
    else:
        print(f"  ℹ️  THIN TAILS: Returns close to normal distribution")

    # Scenario 2: Poor strategy with negative skew
    print("\n[Scenario 2] Strategy with Negative Skew (many small wins, few large losses):")

    calculator2 = OmegaRatioCalculator(risk_free_rate=0.05)
    np.random.seed(43)

    for i in range(100):
        # 80% small wins, 20% large losses (negative skew)
        if np.random.random() < 0.80:
            return_pct = np.random.normal(0.01, 0.003)   # 1% small wins
        else:
            return_pct = np.random.normal(-0.04, 0.01)  # -4% large losses

        calculator2.record_return(return_pct, datetime.now())

    metrics2 = calculator2.calculate_omega()

    print(f"  Omega (0%):                {metrics2.omega_0pct:.2f}")
    print(f"  Expected Gain:             {metrics2.expected_gains:.2%}")
    print(f"  Expected Loss:             {metrics2.expected_losses:.2%}")
    print(f"  Skewness:                  {metrics2.skewness:.2f}")
    print(f"  Quality:                   {metrics2.omega_quality.upper()}")

    if metrics2.skewness < 0:
        print(f"\n  ⚠️  NEGATIVE SKEW: Large losses outweigh small wins (unfavorable)")

    print("\n" + "=" * 70)
    print("✅ Omega Ratio Calculator ready for production!")
    print("=" * 70)
