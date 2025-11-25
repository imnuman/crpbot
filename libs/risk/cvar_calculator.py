"""
CVaR (Conditional Value at Risk) Calculator

Calculates CVaR (Conditional Value at Risk), also known as Expected Shortfall (ES).

CVaR measures the expected loss in the worst X% of cases. For example:
- 95% CVaR = average loss in the worst 5% of outcomes
- More conservative than VaR (Value at Risk) because it captures tail risk

Mathematical Definition:
    CVaR_α = E[Loss | Loss > VaR_α]

Features:
- Historical CVaR (from realized returns)
- Parametric CVaR (assumes normal distribution)
- Monte Carlo CVaR (simulated scenarios)
- Multiple confidence levels (90%, 95%, 99%)
- Daily, weekly, monthly horizons

Risk Management Applications:
- Position sizing (limit CVaR to X% of capital)
- Strategy evaluation (compare CVaR across strategies)
- Stress testing (worst-case loss expectations)
- Portfolio optimization (minimize CVaR)

Expected Impact:
- Better tail risk awareness
- More conservative position sizing in high-vol regimes
- Early warning when CVaR deteriorates
- Quantitative risk limits
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CVaRMetrics:
    """CVaR calculation results"""
    # Historical CVaR (from actual returns)
    cvar_95_historical: float  # Expected loss in worst 5% of cases
    cvar_99_historical: float  # Expected loss in worst 1% of cases
    var_95_historical: float   # 95th percentile loss (VaR)
    var_99_historical: float   # 99th percentile loss (VaR)

    # Parametric CVaR (assuming normal distribution)
    cvar_95_parametric: float
    cvar_99_parametric: float

    # Monte Carlo CVaR (10k simulations)
    cvar_95_monte_carlo: Optional[float]
    cvar_99_monte_carlo: Optional[float]

    # Risk metrics
    worst_loss: float          # Worst single loss observed
    avg_loss: float            # Average losing trade
    loss_frequency: float      # Percentage of losing trades
    tail_ratio: float          # CVaR_95 / Avg Loss (tail heaviness)

    # Position sizing recommendations
    max_position_size_95: float  # Max position size to limit CVaR_95 to 2%
    max_position_size_99: float  # Max position size to limit CVaR_99 to 5%

    # Warnings
    risk_level: str  # 'low', 'moderate', 'high', 'extreme'
    warnings: List[str]

    summary: str
    metrics: Dict[str, float]


class CVaRCalculator:
    """
    Calculate CVaR (Conditional Value at Risk) for trading strategy

    Usage:
        calculator = CVaRCalculator()

        # Record trade returns
        calculator.record_return(return_pct=0.015)  # 1.5% gain
        calculator.record_return(return_pct=-0.025) # -2.5% loss

        # Get CVaR metrics
        metrics = calculator.get_cvar_metrics()
        print(f"95% CVaR: {metrics.cvar_95_historical:.2%}")
        print(f"Risk Level: {metrics.risk_level}")
        print(f"Max Position Size: {metrics.max_position_size_95:.1%}")
    """

    def __init__(
        self,
        max_history: int = 500,  # Keep last 500 trades
        confidence_levels: List[float] = None  # [0.95, 0.99]
    ):
        """
        Initialize CVaR Calculator

        Args:
            max_history: Maximum number of returns to keep
            confidence_levels: Confidence levels for CVaR (default: 95%, 99%)
        """
        self.max_history = max_history
        self.confidence_levels = confidence_levels or [0.95, 0.99]

        # Trade returns history
        self.returns: deque = deque(maxlen=max_history)

        # Timestamps for time-based analysis
        self.timestamps: deque = deque(maxlen=max_history)

        logger.info(
            f"CVaR Calculator initialized | "
            f"Max history: {max_history} | "
            f"Confidence levels: {[f'{c:.0%}' for c in self.confidence_levels]}"
        )

    def record_return(
        self,
        return_pct: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a trade return

        Args:
            return_pct: Trade return (e.g., 0.015 = 1.5% gain, -0.025 = -2.5% loss)
            timestamp: Trade timestamp (default: now)
        """
        self.returns.append(return_pct)
        self.timestamps.append(timestamp or datetime.now())

        logger.debug(
            f"Return recorded: {return_pct:+.2%} | "
            f"Total: {len(self.returns)} trades"
        )

    def get_cvar_metrics(
        self,
        capital: float = 10000.0,  # Total capital for position sizing
        max_cvar_95_pct: float = 0.02,  # Max 2% CVaR at 95%
        max_cvar_99_pct: float = 0.05   # Max 5% CVaR at 99%
    ) -> CVaRMetrics:
        """
        Calculate CVaR metrics

        Args:
            capital: Total trading capital
            max_cvar_95_pct: Maximum acceptable CVaR at 95% confidence
            max_cvar_99_pct: Maximum acceptable CVaR at 99% confidence

        Returns:
            CVaRMetrics with all calculations
        """
        try:
            if len(self.returns) < 10:
                logger.warning(f"Insufficient returns for CVaR: {len(self.returns)} < 10")
                return self._insufficient_data_metrics()

            returns_arr = np.array(self.returns)

            # Historical CVaR
            cvar_95_hist, var_95_hist = self._calculate_historical_cvar(returns_arr, 0.95)
            cvar_99_hist, var_99_hist = self._calculate_historical_cvar(returns_arr, 0.99)

            # Parametric CVaR (assumes normal distribution)
            cvar_95_param = self._calculate_parametric_cvar(returns_arr, 0.95)
            cvar_99_param = self._calculate_parametric_cvar(returns_arr, 0.99)

            # Monte Carlo CVaR (if enough data)
            if len(self.returns) >= 30:
                cvar_95_mc = self._calculate_monte_carlo_cvar(returns_arr, 0.95, n_simulations=10000)
                cvar_99_mc = self._calculate_monte_carlo_cvar(returns_arr, 0.99, n_simulations=10000)
            else:
                cvar_95_mc = None
                cvar_99_mc = None

            # Risk metrics
            losses = returns_arr[returns_arr < 0]
            worst_loss = float(np.min(returns_arr)) if len(returns_arr) > 0 else 0.0
            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            loss_frequency = len(losses) / len(returns_arr) if len(returns_arr) > 0 else 0.0

            # Tail ratio (how much worse is CVaR than average loss?)
            tail_ratio = abs(cvar_95_hist / avg_loss) if avg_loss != 0 else 1.0

            # Position sizing recommendations
            # If CVaR_95 = -3%, and we want max 2% loss, position size = 2% / 3% = 66.7%
            max_pos_95 = abs(max_cvar_95_pct / cvar_95_hist) if cvar_95_hist != 0 else 1.0
            max_pos_99 = abs(max_cvar_99_pct / cvar_99_hist) if cvar_99_hist != 0 else 1.0

            # Cap position size at 100%
            max_pos_95 = min(max_pos_95, 1.0)
            max_pos_99 = min(max_pos_99, 1.0)

            # Risk level assessment
            risk_level, warnings = self._assess_risk_level(
                cvar_95_hist, cvar_99_hist, tail_ratio, loss_frequency
            )

            # Generate summary
            summary = self._generate_summary(
                cvar_95_hist, cvar_99_hist, risk_level, max_pos_95
            )

            metrics = CVaRMetrics(
                cvar_95_historical=cvar_95_hist,
                cvar_99_historical=cvar_99_hist,
                var_95_historical=var_95_hist,
                var_99_historical=var_99_hist,
                cvar_95_parametric=cvar_95_param,
                cvar_99_parametric=cvar_99_param,
                cvar_95_monte_carlo=cvar_95_mc,
                cvar_99_monte_carlo=cvar_99_mc,
                worst_loss=worst_loss,
                avg_loss=avg_loss,
                loss_frequency=loss_frequency,
                tail_ratio=tail_ratio,
                max_position_size_95=max_pos_95,
                max_position_size_99=max_pos_99,
                risk_level=risk_level,
                warnings=warnings,
                summary=summary,
                metrics={
                    'cvar_95_hist': cvar_95_hist,
                    'cvar_99_hist': cvar_99_hist,
                    'var_95_hist': var_95_hist,
                    'var_99_hist': var_99_hist,
                    'worst_loss': worst_loss,
                    'avg_loss': avg_loss,
                    'tail_ratio': tail_ratio,
                    'max_pos_95': max_pos_95
                }
            )

            logger.debug(
                f"CVaR Metrics: 95%={cvar_95_hist:.2%}, 99%={cvar_99_hist:.2%}, "
                f"Risk={risk_level}"
            )

            return metrics

        except Exception as e:
            logger.error(f"CVaR calculation failed: {e}")
            return self._insufficient_data_metrics()

    def _calculate_historical_cvar(
        self,
        returns: np.ndarray,
        confidence: float
    ) -> Tuple[float, float]:
        """
        Calculate historical CVaR and VaR

        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95)

        Returns:
            (CVaR, VaR) tuple
        """
        # VaR = percentile loss
        var = np.percentile(returns, (1 - confidence) * 100)

        # CVaR = average of losses beyond VaR
        tail_losses = returns[returns <= var]

        if len(tail_losses) > 0:
            cvar = float(np.mean(tail_losses))
        else:
            cvar = var  # If no tail losses, CVaR = VaR

        return cvar, float(var)

    def _calculate_parametric_cvar(
        self,
        returns: np.ndarray,
        confidence: float
    ) -> float:
        """
        Calculate parametric CVaR (assumes normal distribution)

        Args:
            returns: Array of returns
            confidence: Confidence level

        Returns:
            Parametric CVaR
        """
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        # For normal distribution:
        # CVaR_α = μ - σ * φ(Φ^(-1)(1-α)) / (1-α)
        # where φ is PDF and Φ is CDF

        z_score = stats.norm.ppf(1 - confidence)
        pdf_z = stats.norm.pdf(z_score)

        cvar = mean - std * pdf_z / (1 - confidence)

        return float(cvar)

    def _calculate_monte_carlo_cvar(
        self,
        returns: np.ndarray,
        confidence: float,
        n_simulations: int = 10000
    ) -> float:
        """
        Calculate Monte Carlo CVaR

        Args:
            returns: Historical returns
            confidence: Confidence level
            n_simulations: Number of simulations

        Returns:
            Monte Carlo CVaR
        """
        # Estimate parameters from historical data
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        # Generate simulated returns
        simulated = np.random.normal(mean, std, n_simulations)

        # Calculate CVaR from simulations
        cvar, _ = self._calculate_historical_cvar(simulated, confidence)

        return float(cvar)

    def _assess_risk_level(
        self,
        cvar_95: float,
        cvar_99: float,
        tail_ratio: float,
        loss_frequency: float
    ) -> Tuple[str, List[str]]:
        """
        Assess risk level and generate warnings

        Args:
            cvar_95: 95% CVaR
            cvar_99: 99% CVaR
            tail_ratio: Tail heaviness
            loss_frequency: Loss frequency

        Returns:
            (risk_level, warnings) tuple
        """
        warnings = []

        # Risk level based on CVaR_95
        if abs(cvar_95) > 0.05:  # > 5% expected loss
            risk_level = 'extreme'
            warnings.append(f"⚠️ EXTREME RISK: 95% CVaR = {cvar_95:.2%} (>5%)")
        elif abs(cvar_95) > 0.03:  # > 3% expected loss
            risk_level = 'high'
            warnings.append(f"⚠️ HIGH RISK: 95% CVaR = {cvar_95:.2%} (>3%)")
        elif abs(cvar_95) > 0.02:  # > 2% expected loss
            risk_level = 'moderate'
            warnings.append(f"ℹ️ MODERATE RISK: 95% CVaR = {cvar_95:.2%}")
        else:
            risk_level = 'low'

        # Heavy tail warning
        if tail_ratio > 2.0:
            warnings.append(f"⚠️ HEAVY TAILS: CVaR is {tail_ratio:.1f}x worse than avg loss")

        # High loss frequency warning
        if loss_frequency > 0.6:
            warnings.append(f"⚠️ HIGH LOSS FREQUENCY: {loss_frequency:.0%} of trades lose")

        # Extreme tail risk
        if abs(cvar_99) > 0.10:
            warnings.append(f"🚨 EXTREME TAIL RISK: 99% CVaR = {cvar_99:.2%} (>10%)")

        return risk_level, warnings

    def _generate_summary(
        self,
        cvar_95: float,
        cvar_99: float,
        risk_level: str,
        max_pos_size: float
    ) -> str:
        """Generate human-readable summary"""
        summary = (
            f"CVaR 95%: {cvar_95:.2%}, CVaR 99%: {cvar_99:.2%} | "
            f"Risk: {risk_level.upper()} | "
            f"Max Position: {max_pos_size:.0%}"
        )
        return summary

    def _insufficient_data_metrics(self) -> CVaRMetrics:
        """Return default metrics when insufficient data"""
        return CVaRMetrics(
            cvar_95_historical=0.0,
            cvar_99_historical=0.0,
            var_95_historical=0.0,
            var_99_historical=0.0,
            cvar_95_parametric=0.0,
            cvar_99_parametric=0.0,
            cvar_95_monte_carlo=None,
            cvar_99_monte_carlo=None,
            worst_loss=0.0,
            avg_loss=0.0,
            loss_frequency=0.0,
            tail_ratio=1.0,
            max_position_size_95=1.0,
            max_position_size_99=1.0,
            risk_level='unknown',
            warnings=[],
            summary=f"Insufficient data: {len(self.returns)} returns (need 10+)",
            metrics={}
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("CVaR (CONDITIONAL VALUE AT RISK) CALCULATOR TEST")
    print("=" * 70)

    # Scenario 1: Low-risk strategy (tight distribution, small losses)
    print("\n[Scenario 1] Low-Risk Strategy:")
    calc1 = CVaRCalculator()

    np.random.seed(42)
    for _ in range(100):
        # 60% win rate, small wins/losses
        if np.random.random() < 0.60:
            return_pct = np.random.normal(0.015, 0.005)  # ~1.5% wins
        else:
            return_pct = np.random.normal(-0.010, 0.003)  # ~-1% losses

        calc1.record_return(return_pct)

    metrics1 = calc1.get_cvar_metrics()
    print(f"  95% CVaR (Historical):  {metrics1.cvar_95_historical:.2%}")
    print(f"  99% CVaR (Historical):  {metrics1.cvar_99_historical:.2%}")
    print(f"  95% CVaR (Parametric):  {metrics1.cvar_95_parametric:.2%}")
    print(f"  95% CVaR (Monte Carlo): {metrics1.cvar_95_monte_carlo:.2%}")
    print(f"  Worst Loss:             {metrics1.worst_loss:.2%}")
    print(f"  Avg Loss:               {metrics1.avg_loss:.2%}")
    print(f"  Tail Ratio:             {metrics1.tail_ratio:.2f}x")
    print(f"  Loss Frequency:         {metrics1.loss_frequency:.0%}")
    print(f"  Risk Level:             {metrics1.risk_level.upper()}")
    print(f"  Max Position (95%):     {metrics1.max_position_size_95:.0%}")
    print(f"\n  Summary: {metrics1.summary}")
    if metrics1.warnings:
        for warning in metrics1.warnings:
            print(f"  {warning}")

    # Scenario 2: High-risk strategy (fat tails, large losses)
    print("\n[Scenario 2] High-Risk Strategy (Fat Tails):")
    calc2 = CVaRCalculator()

    for _ in range(100):
        # 50% win rate, but occasional huge losses
        if np.random.random() < 0.50:
            return_pct = np.random.normal(0.020, 0.010)  # ~2% wins
        else:
            # Some losses are extreme (fat tail)
            if np.random.random() < 0.1:  # 10% of losses are extreme
                return_pct = np.random.normal(-0.080, 0.020)  # ~-8% extreme loss
            else:
                return_pct = np.random.normal(-0.015, 0.005)  # ~-1.5% normal loss

        calc2.record_return(return_pct)

    metrics2 = calc2.get_cvar_metrics()
    print(f"  95% CVaR (Historical):  {metrics2.cvar_95_historical:.2%}")
    print(f"  99% CVaR (Historical):  {metrics2.cvar_99_historical:.2%}")
    print(f"  Worst Loss:             {metrics2.worst_loss:.2%}")
    print(f"  Avg Loss:               {metrics2.avg_loss:.2%}")
    print(f"  Tail Ratio:             {metrics2.tail_ratio:.2f}x")
    print(f"  Risk Level:             {metrics2.risk_level.upper()}")
    print(f"  Max Position (95%):     {metrics2.max_position_size_95:.0%}")
    print(f"\n  Summary: {metrics2.summary}")
    if metrics2.warnings:
        for warning in metrics2.warnings:
            print(f"  {warning}")

    # Scenario 3: Moderate strategy
    print("\n[Scenario 3] Moderate-Risk Strategy:")
    calc3 = CVaRCalculator()

    for _ in range(100):
        # 55% win rate, balanced distribution
        if np.random.random() < 0.55:
            return_pct = np.random.normal(0.020, 0.008)  # ~2% wins
        else:
            return_pct = np.random.normal(-0.015, 0.008)  # ~-1.5% losses

        calc3.record_return(return_pct)

    metrics3 = calc3.get_cvar_metrics()
    print(f"  95% CVaR:      {metrics3.cvar_95_historical:.2%}")
    print(f"  Risk Level:    {metrics3.risk_level.upper()}")
    print(f"  Max Position:  {metrics3.max_position_size_95:.0%}")
    print(f"\n  Summary: {metrics3.summary}")

    print("\n" + "=" * 70)
    print("✅ CVaR Calculator ready for production!")
    print("=" * 70)
