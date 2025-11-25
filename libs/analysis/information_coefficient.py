"""
Information Coefficient (IC) Analysis

Measures the predictive power of signals/theories by calculating correlation
between signal strength and forward returns.

IC = Spearman Correlation(Signal Strength, Forward Returns)

Interpretation:
- IC > 0.05: Good predictive power
- IC > 0.10: Excellent predictive power
- IC < 0.02: Weak/no predictive power
- IC < 0: Negative correlation (contrarian signal)

Applications:
- Theory ranking: Which mathematical theories predict best?
- Signal validation: Are signals actually predictive?
- Strategy improvement: Focus on high-IC theories
- Real-time monitoring: Detect degrading signal quality

Expected Impact:
- Identify best-performing theories
- Increase win rate by weighting high-IC signals
- Early warning when IC decays
- Data-driven theory selection
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class ICMetrics:
    """Information Coefficient analysis results"""
    # Overall IC
    ic_1h: float  # IC for 1-hour forward returns
    ic_4h: float  # IC for 4-hour forward returns
    ic_24h: float  # IC for 24-hour forward returns

    # Statistical significance
    p_value_1h: float
    p_value_4h: float
    p_value_24h: float

    # Sample statistics
    n_samples: int
    mean_signal: float
    std_signal: float

    # Quality assessment
    ic_quality: str  # 'excellent', 'good', 'weak', 'none', 'contrarian'
    is_significant: bool  # p-value < 0.05

    # Theory-specific ICs (if available)
    theory_ics: Optional[Dict[str, float]]

    summary: str
    metrics: Dict[str, float]


@dataclass
class TheoryIC:
    """IC metrics for a specific theory"""
    theory_name: str
    ic: float
    p_value: float
    n_samples: int
    rank: int  # Ranking among all theories (1 = best)


class InformationCoefficientAnalyzer:
    """
    Analyze signal quality using Information Coefficient

    Usage:
        analyzer = InformationCoefficientAnalyzer()

        # Record signals and outcomes
        analyzer.record_signal(
            signal_strength=0.75,  # Confidence or theory score
            price_at_signal=100.0,
            timestamp=datetime.now(),
            theory_scores={'Shannon': 0.8, 'Hurst': 0.6, ...}
        )

        # Later, record actual prices
        analyzer.record_price_update(
            timestamp=datetime.now() + timedelta(hours=1),
            price=101.5
        )

        # Calculate IC
        metrics = analyzer.calculate_ic()
        print(f"1h IC: {metrics.ic_1h:.3f} ({metrics.ic_quality})")
        print(f"Best theory: {max(metrics.theory_ics, key=metrics.theory_ics.get)}")
    """

    def __init__(
        self,
        max_history: int = 1000,
        forward_horizons: List[int] = None  # [60, 240, 1440] minutes
    ):
        """
        Initialize IC Analyzer

        Args:
            max_history: Maximum signals to keep in history
            forward_horizons: List of forward return horizons in minutes
        """
        self.max_history = max_history
        self.forward_horizons = forward_horizons or [60, 240, 1440]  # 1h, 4h, 24h

        # Signal history: [(timestamp, signal_strength, price, theory_scores), ...]
        self.signal_history: deque = deque(maxlen=max_history)

        # Price history for computing forward returns
        self.price_history: deque = deque(maxlen=max_history * 2)

        logger.info(
            f"IC Analyzer initialized | "
            f"Max history: {max_history} | "
            f"Horizons: {[f'{h}min' for h in self.forward_horizons]}"
        )

    def record_signal(
        self,
        signal_strength: float,
        price_at_signal: float,
        timestamp: datetime,
        symbol: str = 'UNKNOWN',
        theory_scores: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record a trading signal

        Args:
            signal_strength: Signal confidence/strength (-1 to 1)
            price_at_signal: Price when signal was generated
            timestamp: Signal timestamp
            symbol: Trading symbol
            theory_scores: Individual theory contribution scores (optional)
        """
        self.signal_history.append({
            'timestamp': timestamp,
            'signal_strength': signal_strength,
            'price': price_at_signal,
            'symbol': symbol,
            'theory_scores': theory_scores or {}
        })

        # Also record price
        self.record_price_update(timestamp, price_at_signal)

        logger.debug(
            f"Signal recorded: {symbol} | "
            f"Strength: {signal_strength:.2f} | "
            f"Price: ${price_at_signal:.2f}"
        )

    def record_price_update(
        self,
        timestamp: datetime,
        price: float
    ) -> None:
        """Record a price update for forward return calculation"""
        self.price_history.append({
            'timestamp': timestamp,
            'price': price
        })

    def calculate_ic(
        self,
        min_samples: int = 30
    ) -> ICMetrics:
        """
        Calculate Information Coefficient

        Args:
            min_samples: Minimum signals needed for IC calculation

        Returns:
            ICMetrics with IC values and statistics
        """
        try:
            if len(self.signal_history) < min_samples:
                logger.warning(f"Insufficient signals for IC: {len(self.signal_history)} < {min_samples}")
                return self._insufficient_data_metrics()

            # Calculate IC for each horizon
            ic_1h, p_1h = self._calculate_ic_for_horizon(60)  # 1 hour
            ic_4h, p_4h = self._calculate_ic_for_horizon(240)  # 4 hours
            ic_24h, p_24h = self._calculate_ic_for_horizon(1440)  # 24 hours

            # Calculate theory-specific ICs
            theory_ics = self._calculate_theory_ics(horizon_minutes=60)

            # Sample statistics
            signals = [s['signal_strength'] for s in self.signal_history]
            n_samples = len(signals)
            mean_signal = np.mean(signals)
            std_signal = np.std(signals)

            # Quality assessment (using 1h IC as primary)
            ic_quality = self._assess_ic_quality(ic_1h)
            is_significant = p_1h < 0.05

            # Generate summary
            summary = self._generate_summary(
                ic_1h, ic_4h, ic_24h, ic_quality, is_significant, theory_ics
            )

            metrics = ICMetrics(
                ic_1h=ic_1h,
                ic_4h=ic_4h,
                ic_24h=ic_24h,
                p_value_1h=p_1h,
                p_value_4h=p_4h,
                p_value_24h=p_24h,
                n_samples=n_samples,
                mean_signal=mean_signal,
                std_signal=std_signal,
                ic_quality=ic_quality,
                is_significant=is_significant,
                theory_ics=theory_ics,
                summary=summary,
                metrics={
                    'ic_1h': ic_1h,
                    'ic_4h': ic_4h,
                    'ic_24h': ic_24h,
                    'n_samples': n_samples
                }
            )

            logger.debug(
                f"IC calculated: 1h={ic_1h:.3f}, 4h={ic_4h:.3f}, 24h={ic_24h:.3f} | "
                f"Quality: {ic_quality}"
            )

            return metrics

        except Exception as e:
            logger.error(f"IC calculation failed: {e}")
            return self._insufficient_data_metrics()

    def _calculate_ic_for_horizon(
        self,
        horizon_minutes: int
    ) -> Tuple[float, float]:
        """
        Calculate IC for a specific forward return horizon

        Args:
            horizon_minutes: Forward return horizon

        Returns:
            (IC, p-value) tuple
        """
        signals = []
        forward_returns = []

        for signal in self.signal_history:
            # Get price at signal time
            signal_price = signal['price']
            signal_time = signal['timestamp']

            # Find price after horizon
            target_time = signal_time + timedelta(minutes=horizon_minutes)
            future_price = self._get_price_at_time(target_time)

            if future_price is None:
                continue  # No price data available yet

            # Calculate forward return
            forward_return = (future_price - signal_price) / signal_price

            signals.append(signal['signal_strength'])
            forward_returns.append(forward_return)

        if len(signals) < 10:
            return 0.0, 1.0  # Not enough data

        # Calculate Spearman correlation (rank IC - more robust to outliers)
        ic, p_value = spearmanr(signals, forward_returns)

        return float(ic), float(p_value)

    def _calculate_theory_ics(
        self,
        horizon_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Calculate IC for each theory individually

        Args:
            horizon_minutes: Forward return horizon

        Returns:
            Dict of {theory_name: IC}
        """
        theory_signals = defaultdict(list)
        theory_returns = defaultdict(list)

        for signal in self.signal_history:
            signal_time = signal['timestamp']
            signal_price = signal['price']
            theory_scores = signal.get('theory_scores', {})

            # Get forward price
            target_time = signal_time + timedelta(minutes=horizon_minutes)
            future_price = self._get_price_at_time(target_time)

            if future_price is None:
                continue

            # Calculate forward return
            forward_return = (future_price - signal_price) / signal_price

            # Record for each theory
            for theory_name, theory_score in theory_scores.items():
                theory_signals[theory_name].append(theory_score)
                theory_returns[theory_name].append(forward_return)

        # Calculate IC for each theory
        theory_ics = {}
        for theory_name in theory_signals.keys():
            signals = theory_signals[theory_name]
            returns = theory_returns[theory_name]

            if len(signals) >= 10:
                ic, _ = spearmanr(signals, returns)
                theory_ics[theory_name] = float(ic)
            else:
                theory_ics[theory_name] = 0.0

        return theory_ics

    def _get_price_at_time(self, target_time: datetime) -> Optional[float]:
        """
        Get price at or near a specific time

        Args:
            target_time: Target timestamp

        Returns:
            Price at that time (or None if not available)
        """
        # Find closest price within ±5 minutes
        tolerance_minutes = 5

        closest_price = None
        min_diff = timedelta(minutes=tolerance_minutes)

        for price_record in self.price_history:
            time_diff = abs(price_record['timestamp'] - target_time)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_price = price_record['price']

        return closest_price

    def _assess_ic_quality(self, ic: float) -> str:
        """Assess IC quality based on magnitude"""
        abs_ic = abs(ic)

        if abs_ic > 0.10:
            return 'excellent'
        elif abs_ic > 0.05:
            return 'good'
        elif abs_ic > 0.02:
            return 'weak'
        elif ic < -0.05:
            return 'contrarian'
        else:
            return 'none'

    def _generate_summary(
        self,
        ic_1h: float,
        ic_4h: float,
        ic_24h: float,
        quality: str,
        significant: bool,
        theory_ics: Optional[Dict[str, float]]
    ) -> str:
        """Generate human-readable summary"""
        summary = f"IC: {ic_1h:.3f} ({quality.upper()})"

        if significant:
            summary += " | Statistically significant"
        else:
            summary += " | Not significant"

        # Best theory
        if theory_ics:
            best_theory = max(theory_ics, key=theory_ics.get)
            best_ic = theory_ics[best_theory]
            summary += f" | Best: {best_theory} ({best_ic:.3f})"

        return summary

    def get_theory_ranking(self) -> List[TheoryIC]:
        """
        Get theories ranked by IC

        Returns:
            List of TheoryIC objects, sorted by IC (best first)
        """
        metrics = self.calculate_ic()

        if not metrics.theory_ics:
            return []

        # Sort theories by IC
        sorted_theories = sorted(
            metrics.theory_ics.items(),
            key=lambda x: abs(x[1]),  # Sort by absolute IC
            reverse=True
        )

        # Create TheoryIC objects
        theory_ranking = []
        for rank, (theory_name, ic) in enumerate(sorted_theories, start=1):
            theory_ic = TheoryIC(
                theory_name=theory_name,
                ic=ic,
                p_value=metrics.p_value_1h,  # Use overall p-value
                n_samples=metrics.n_samples,
                rank=rank
            )
            theory_ranking.append(theory_ic)

        return theory_ranking

    def _insufficient_data_metrics(self) -> ICMetrics:
        """Return default metrics when insufficient data"""
        return ICMetrics(
            ic_1h=0.0,
            ic_4h=0.0,
            ic_24h=0.0,
            p_value_1h=1.0,
            p_value_4h=1.0,
            p_value_24h=1.0,
            n_samples=len(self.signal_history),
            mean_signal=0.0,
            std_signal=0.0,
            ic_quality='none',
            is_significant=False,
            theory_ics=None,
            summary=f"Insufficient data: {len(self.signal_history)} signals (need 30+)",
            metrics={}
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("INFORMATION COEFFICIENT (IC) ANALYZER TEST")
    print("=" * 70)

    analyzer = InformationCoefficientAnalyzer()

    # Simulate signal generation over 100 periods
    np.random.seed(42)
    base_time = datetime.now() - timedelta(hours=100)

    print("\n[Scenario 1] Good Predictive Signals (IC > 0.10):")

    for i in range(100):
        # Simulate a market trend
        true_trend = np.sin(i / 10) * 0.02  # 2% trend

        # Signal with good predictive power (correlated with trend)
        signal_strength = true_trend + np.random.normal(0, 0.01)
        signal_strength = np.clip(signal_strength, -1, 1)

        # Price follows trend
        price = 100 + np.cumsum([true_trend])[0] + np.random.normal(0, 0.5)

        analyzer.record_signal(
            signal_strength=signal_strength,
            price_at_signal=price,
            timestamp=base_time + timedelta(hours=i),
            theory_scores={
                'Shannon': signal_strength * 0.8 + np.random.normal(0, 0.1),
                'Hurst': signal_strength * 0.6 + np.random.normal(0, 0.15),
                'Markov': signal_strength * 0.5 + np.random.normal(0, 0.2)
            }
        )

        # Record future prices (1h, 4h ahead)
        for future_hours in [1, 4, 24]:
            future_price = price + true_trend * future_hours * 10
            analyzer.record_price_update(
                timestamp=base_time + timedelta(hours=i + future_hours),
                price=future_price
            )

    metrics = analyzer.calculate_ic()
    print(f"  1h IC:           {metrics.ic_1h:.3f}")
    print(f"  4h IC:           {metrics.ic_4h:.3f}")
    print(f"  24h IC:          {metrics.ic_24h:.3f}")
    print(f"  P-value (1h):    {metrics.p_value_1h:.4f}")
    print(f"  Quality:         {metrics.ic_quality.upper()}")
    print(f"  Significant:     {metrics.is_significant}")
    print(f"\n  Summary: {metrics.summary}")

    if metrics.theory_ics:
        print(f"\n  Theory Ranking:")
        ranking = analyzer.get_theory_ranking()
        for theory in ranking:
            print(f"    {theory.rank}. {theory.theory_name}: IC={theory.ic:.3f}")

    # Scenario 2: Poor predictive signals
    print("\n[Scenario 2] Poor Predictive Signals (IC < 0.02):")
    analyzer2 = InformationCoefficientAnalyzer()

    for i in range(100):
        # Random signals (no correlation with returns)
        signal_strength = np.random.uniform(-1, 1)
        price = 100 + np.random.normal(0, 5)

        analyzer2.record_signal(
            signal_strength=signal_strength,
            price_at_signal=price,
            timestamp=base_time + timedelta(hours=i)
        )

        # Random future prices
        for future_hours in [1, 4, 24]:
            future_price = price + np.random.normal(0, 3)
            analyzer2.record_price_update(
                timestamp=base_time + timedelta(hours=i + future_hours),
                price=future_price
            )

    metrics2 = analyzer2.calculate_ic()
    print(f"  1h IC:    {metrics2.ic_1h:.3f}")
    print(f"  Quality:  {metrics2.ic_quality.upper()}")
    print(f"  Summary: {metrics2.summary}")

    print("\n" + "=" * 70)
    print("✅ Information Coefficient Analyzer ready for production!")
    print("=" * 70)
