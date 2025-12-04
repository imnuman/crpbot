"""
Calmar Ratio Tracker

Calmar Ratio = Annualized Return / Maximum Drawdown

Named after California Managed Accounts Reports (1991), the Calmar Ratio
measures risk-adjusted returns by comparing annualized performance to worst
drawdown. It's particularly useful for evaluating downside risk.

Interpretation:
- Calmar > 3.0: Excellent (high returns relative to drawdown)
- Calmar 2.0-3.0: Very Good
- Calmar 1.0-2.0: Good
- Calmar 0.5-1.0: Acceptable
- Calmar < 0.5: Poor (high drawdown relative to returns)

Advantages:
- Simple and intuitive (return per unit of max loss)
- Focuses on worst-case scenario (max drawdown)
- Preferred by CTAs and hedge funds
- Good for comparing strategies with different volatility profiles

Comparison with Sharpe:
- Sharpe uses volatility (standard deviation)
- Calmar uses max drawdown (actual worst loss)
- Calmar is more conservative (focuses on tail risk)

Expected Impact:
- Better assessment of downside risk
- Complements Sharpe/Sortino ratios
- Identifies strategies with favorable return/risk profiles
- Early warning when drawdowns become excessive
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CalmarMetrics:
    """Calmar ratio calculation results"""
    calmar_ratio_30d: float
    calmar_ratio_90d: Optional[float]
    calmar_ratio_180d: Optional[float]
    calmar_ratio_365d: Optional[float]

    # Components
    annualized_return: float
    max_drawdown: float
    current_drawdown: float

    # Drawdown statistics
    max_drawdown_duration_days: int
    current_drawdown_duration_days: int
    time_to_recovery_days: Optional[int]

    # Return statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Quality assessment
    calmar_quality: str  # 'excellent', 'very_good', 'good', 'acceptable', 'poor'

    summary: str
    metrics: Dict[str, float]


class CalmarRatioTracker:
    """
    Track Calmar Ratio (return/max drawdown)

    Usage:
        tracker = CalmarRatioTracker()

        # Record returns
        tracker.record_return(0.025, datetime.now())  # 2.5% gain
        tracker.record_return(-0.015, datetime.now()) # -1.5% loss

        # Get metrics
        metrics = tracker.get_calmar_metrics()
        print(f"30-day Calmar: {metrics.calmar_ratio_30d:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.1%}")
    """

    def __init__(
        self,
        max_history_days: int = 365
    ):
        """
        Initialize Calmar Ratio Tracker

        Args:
            max_history_days: Maximum days of history to keep
        """
        self.max_history_days = max_history_days

        # Returns history: [(timestamp, return_pct), ...]
        self.returns_history: deque = deque(maxlen=2000)

        # Track equity curve for drawdown calculation
        self.equity_curve: deque = deque(maxlen=2000)

        logger.info(f"Calmar Ratio Tracker initialized | Max history: {max_history_days} days")

    def record_return(
        self,
        return_pct: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a trade return

        Args:
            return_pct: Return as decimal (0.025 = 2.5%)
            timestamp: Return timestamp (default: now)
        """
        ts = timestamp or datetime.now(timezone.utc)

        self.returns_history.append({
            'timestamp': ts,
            'return_pct': return_pct
        })

        # Update equity curve
        if len(self.equity_curve) == 0:
            equity = 1.0 + return_pct
        else:
            equity = self.equity_curve[-1]['equity'] * (1.0 + return_pct)

        self.equity_curve.append({
            'timestamp': ts,
            'equity': equity,
            'return_pct': return_pct
        })

        logger.debug(f"Return recorded: {return_pct:+.2%} | Equity: {equity:.4f}")

    def get_calmar_metrics(self) -> CalmarMetrics:
        """
        Calculate Calmar Ratio metrics

        Returns:
            CalmarMetrics with Calmar ratios and drawdown statistics
        """
        try:
            if len(self.returns_history) < 5:
                logger.warning(f"Insufficient returns for Calmar: {len(self.returns_history)}")
                return self._insufficient_data_metrics()

            # Calculate Calmar for different windows
            calmar_30d = self._calculate_calmar_window(days=30)
            calmar_90d = self._calculate_calmar_window(days=90) if len(self.returns_history) >= 20 else None
            calmar_180d = self._calculate_calmar_window(days=180) if len(self.returns_history) >= 40 else None
            calmar_365d = self._calculate_calmar_window(days=365) if len(self.returns_history) >= 80 else None

            # Calculate overall metrics
            returns = [r['return_pct'] for r in self.returns_history]
            ann_return = self._annualize_return(returns)

            # Drawdown statistics
            dd_stats = self._calculate_drawdown_stats()
            max_dd = dd_stats['max_drawdown']
            current_dd = dd_stats['current_drawdown']
            max_dd_duration = dd_stats['max_drawdown_duration_days']
            current_dd_duration = dd_stats['current_drawdown_duration_days']
            recovery_time = dd_stats['time_to_recovery_days']

            # Trade statistics
            total_trades = len(returns)
            winning_trades = len([r for r in returns if r > 0])
            losing_trades = len([r for r in returns if r < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # Quality assessment
            calmar_quality = self._assess_calmar_quality(calmar_30d)

            # Generate summary
            summary = self._generate_summary(calmar_30d, ann_return, max_dd, calmar_quality)

            metrics = CalmarMetrics(
                calmar_ratio_30d=calmar_30d,
                calmar_ratio_90d=calmar_90d,
                calmar_ratio_180d=calmar_180d,
                calmar_ratio_365d=calmar_365d,
                annualized_return=ann_return,
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                max_drawdown_duration_days=max_dd_duration,
                current_drawdown_duration_days=current_dd_duration,
                time_to_recovery_days=recovery_time,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                calmar_quality=calmar_quality,
                summary=summary,
                metrics={
                    'calmar_30d': calmar_30d,
                    'calmar_90d': calmar_90d or 0.0,
                    'annualized_return': ann_return,
                    'max_drawdown': max_dd,
                    'current_drawdown': current_dd
                }
            )

            logger.debug(
                f"Calmar Metrics: 30d={calmar_30d:.2f}, Ann Return={ann_return:.1%}, "
                f"Max DD={max_dd:.1%}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Calmar calculation failed: {e}")
            return self._insufficient_data_metrics()

    def _calculate_calmar_window(self, days: int) -> float:
        """
        Calculate Calmar ratio for a specific window

        Args:
            days: Window size in days

        Returns:
            Calmar ratio
        """
        if len(self.returns_history) < 3:
            return 0.0

        # Get returns within window
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        window_returns = [
            r['return_pct'] for r in self.returns_history
            if r['timestamp'] >= cutoff
        ]

        if len(window_returns) < 3:
            return 0.0

        # Annualized return
        ann_return = self._annualize_return(window_returns)

        # Max drawdown for this window
        window_equity = []
        equity = 1.0
        for ret in window_returns:
            equity *= (1.0 + ret)
            window_equity.append(equity)

        max_dd = self._calculate_max_drawdown_from_equity(window_equity)

        # Calmar = Annual Return / |Max Drawdown|
        if max_dd == 0:
            return 0.0

        calmar = ann_return / abs(max_dd)

        return float(calmar)

    def _calculate_max_drawdown_from_equity(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(equity_curve) == 0:
            return 0.0

        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_dd = np.min(drawdown)

        return float(max_dd)

    def _calculate_drawdown_stats(self) -> Dict:
        """Calculate comprehensive drawdown statistics"""
        if len(self.equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'max_drawdown_duration_days': 0,
                'current_drawdown_duration_days': 0,
                'time_to_recovery_days': None
            }

        # Extract equity curve
        equity_values = [e['equity'] for e in self.equity_curve]
        timestamps = [e['timestamp'] for e in self.equity_curve]

        equity_array = np.array(equity_values)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max

        # Max drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = float(drawdown[max_dd_idx])

        # Current drawdown
        current_dd = float(drawdown[-1])

        # Max drawdown duration
        max_dd_start_idx = np.argmax(running_max[:max_dd_idx+1])
        max_dd_duration_days = (timestamps[max_dd_idx] - timestamps[max_dd_start_idx]).days

        # Current drawdown duration
        if current_dd < -0.001:  # In drawdown (> 0.1%)
            # Find when current drawdown started
            current_peak_idx = len(equity_array) - 1
            for i in range(len(equity_array) - 1, -1, -1):
                if equity_array[i] >= running_max[i] * 0.999:  # Within 0.1% of peak
                    current_peak_idx = i
                    break
            current_dd_duration_days = (timestamps[-1] - timestamps[current_peak_idx]).days
        else:
            current_dd_duration_days = 0

        # Time to recovery (for max drawdown)
        recovery_time = None
        if max_dd_idx < len(equity_array) - 1:
            peak_before_dd = running_max[max_dd_idx]
            for i in range(max_dd_idx + 1, len(equity_array)):
                if equity_array[i] >= peak_before_dd * 0.999:
                    recovery_time = (timestamps[i] - timestamps[max_dd_idx]).days
                    break

        return {
            'max_drawdown': max_dd,
            'current_drawdown': current_dd,
            'max_drawdown_duration_days': max_dd_duration_days,
            'current_drawdown_duration_days': current_dd_duration_days,
            'time_to_recovery_days': recovery_time
        }

    def _annualize_return(self, returns: List[float]) -> float:
        """Annualize return from list of trade returns"""
        if len(returns) < 2:
            return 0.0

        # Calculate total period
        first_timestamp = self.returns_history[0]['timestamp']
        last_timestamp = self.returns_history[-1]['timestamp']
        period_days = (last_timestamp - first_timestamp).days

        if period_days == 0:
            period_days = 1

        # Trades per day
        trades_per_day = len(returns) / period_days
        trades_per_year = trades_per_day * 365

        # Annualized return
        mean_return = np.mean(returns)
        ann_return = mean_return * trades_per_year

        return float(ann_return)

    def _assess_calmar_quality(self, calmar: float) -> str:
        """Assess Calmar ratio quality"""
        if calmar >= 3.0:
            return 'excellent'
        elif calmar >= 2.0:
            return 'very_good'
        elif calmar >= 1.0:
            return 'good'
        elif calmar >= 0.5:
            return 'acceptable'
        else:
            return 'poor'

    def _generate_summary(
        self,
        calmar_30d: float,
        ann_return: float,
        max_dd: float,
        quality: str
    ) -> str:
        """Generate human-readable summary"""
        summary = (
            f"30-day Calmar: {calmar_30d:.2f} ({quality.upper()}) | "
            f"Ann. Return: {ann_return:+.1%} | "
            f"Max DD: {max_dd:.1%}"
        )
        return summary

    def _insufficient_data_metrics(self) -> CalmarMetrics:
        """Return default metrics when insufficient data"""
        return CalmarMetrics(
            calmar_ratio_30d=0.0,
            calmar_ratio_90d=None,
            calmar_ratio_180d=None,
            calmar_ratio_365d=None,
            annualized_return=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            max_drawdown_duration_days=0,
            current_drawdown_duration_days=0,
            time_to_recovery_days=None,
            total_trades=len(self.returns_history),
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            calmar_quality='unknown',
            summary=f"Insufficient data: {len(self.returns_history)} trades (need 5+)",
            metrics={}
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("CALMAR RATIO TRACKER TEST")
    print("=" * 70)

    # Scenario: Strategy with good returns but significant drawdown
    print("\n[Scenario 1] Good Returns with Moderate Drawdown:")

    tracker = CalmarRatioTracker()
    np.random.seed(42)
    base_date = datetime.now() - timedelta(days=90)

    # Simulate 100 trades over 90 days
    equity = 1.0
    for i in range(100):
        # 65% win rate, but occasional large losses
        if np.random.random() < 0.65:
            return_pct = np.random.normal(0.02, 0.01)  # 2% avg win
        else:
            return_pct = np.random.normal(-0.03, 0.015)  # -3% avg loss

        timestamp = base_date + timedelta(days=i*0.9)
        tracker.record_return(return_pct, timestamp)
        equity *= (1.0 + return_pct)

    metrics = tracker.get_calmar_metrics()

    print(f"  30-day Calmar:             {metrics.calmar_ratio_30d:.2f}")
    print(f"  90-day Calmar:             {metrics.calmar_ratio_90d:.2f}")
    print(f"  Ann. Return:               {metrics.annualized_return:+.1%}")
    print(f"  Max Drawdown:              {metrics.max_drawdown:.1%}")
    print(f"  Current Drawdown:          {metrics.current_drawdown:.1%}")
    print(f"  Max DD Duration:           {metrics.max_drawdown_duration_days} days")
    print(f"  Time to Recovery:          {metrics.time_to_recovery_days} days" if metrics.time_to_recovery_days else "  Still in drawdown")
    print(f"  Win Rate:                  {metrics.win_rate:.1%}")
    print(f"  Quality:                   {metrics.calmar_quality.upper()}")
    print(f"\n  Summary: {metrics.summary}")

    # Interpretation
    if metrics.calmar_ratio_30d >= 2.0:
        print(f"\n  ✅ STRONG: Calmar >= 2.0 indicates excellent return/drawdown ratio")
    elif metrics.calmar_ratio_30d >= 1.0:
        print(f"\n  ✅ GOOD: Calmar >= 1.0 indicates acceptable return/drawdown ratio")
    else:
        print(f"\n  ⚠️  WEAK: Calmar < 1.0 indicates high drawdown relative to returns")

    print("\n" + "=" * 70)
    print("✅ Calmar Ratio Tracker ready for production!")
    print("=" * 70)
