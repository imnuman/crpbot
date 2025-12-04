"""
Sortino Ratio Tracker

Like Sharpe Ratio, but only penalizes downside volatility (more realistic for trading).

Sortino Ratio = (Mean Return - Risk-Free Rate) / Downside Deviation

Advantages over Sharpe:
- Only penalizes losses (not upside volatility)
- Better for asymmetric return distributions
- More realistic risk assessment
- Preferred by professional traders

Interpretation:
- Sortino > 3.0: Excellent
- Sortino 2.0-3.0: Very Good
- Sortino 1.0-2.0: Good
- Sortino < 1.0: Poor

Expected Impact:
- More accurate risk-adjusted performance
- Better suited for crypto (asymmetric returns)
- Complements Sharpe ratio
- Improved strategy comparison
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
class SortinoMetrics:
    """Sortino ratio calculation results"""
    sortino_ratio_7d: float
    sortino_ratio_14d: float
    sortino_ratio_30d: float
    sortino_ratio_90d: Optional[float]

    annualized_return: float
    downside_deviation: float  # Only negative returns
    upside_deviation: float    # Only positive returns

    total_trades: int
    losing_trades: int
    avg_loss: float
    max_loss: float

    # Comparison with Sharpe
    sharpe_ratio: Optional[float]
    sortino_sharpe_ratio: float  # Sortino / Sharpe (should be > 1.0)

    summary: str
    metrics: Dict[str, float]


class SortinoRatioTracker:
    """
    Track Sortino Ratio (downside risk-adjusted returns)

    Usage:
        tracker = SortinoRatioTracker()

        # Record returns
        tracker.record_return(0.015, datetime.now())  # 1.5% gain
        tracker.record_return(-0.025, datetime.now()) # -2.5% loss

        # Get metrics
        metrics = tracker.get_sortino_metrics()
        print(f"30-day Sortino: {metrics.sortino_ratio_30d:.2f}")
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        max_history_days: int = 90,
        mar: float = 0.0  # Minimum Acceptable Return
    ):
        """
        Initialize Sortino Ratio Tracker

        Args:
            risk_free_rate: Annual risk-free rate
            max_history_days: Max days of history
            mar: Minimum Acceptable Return (MAR) - returns below this are "downside"
        """
        self.risk_free_rate = risk_free_rate
        self.max_history_days = max_history_days
        self.mar = mar  # Usually 0% or risk-free rate

        # Returns history: [(timestamp, return_pct), ...]
        self.returns_history: deque = deque(maxlen=1000)

        logger.info(
            f"Sortino Tracker initialized | "
            f"RF rate: {risk_free_rate:.1%} | "
            f"MAR: {mar:.1%}"
        )

    def record_return(
        self,
        return_pct: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a trade return"""
        self.returns_history.append({
            'timestamp': timestamp or datetime.now(timezone.utc),
            'return_pct': return_pct
        })

        logger.debug(f"Return recorded: {return_pct:+.2%}")

    def get_sortino_metrics(
        self,
        sharpe_ratio: Optional[float] = None
    ) -> SortinoMetrics:
        """
        Calculate Sortino metrics

        Args:
            sharpe_ratio: Optional Sharpe ratio for comparison

        Returns:
            SortinoMetrics
        """
        try:
            if len(self.returns_history) < 5:
                logger.warning(f"Insufficient returns for Sortino: {len(self.returns_history)}")
                return self._insufficient_data_metrics()

            # Calculate Sortino for different windows
            sortino_7d = self._calculate_sortino_window(days=7)
            sortino_14d = self._calculate_sortino_window(days=14)
            sortino_30d = self._calculate_sortino_window(days=30)
            sortino_90d = self._calculate_sortino_window(days=90) if len(self.returns_history) >= 20 else None

            # Calculate annualized metrics
            returns = [r['return_pct'] for r in self.returns_history]
            ann_return, downside_dev, upside_dev = self._annualize_metrics(returns)

            # Loss statistics
            losses = [r for r in returns if r < 0]
            total_trades = len(returns)
            losing_trades = len(losses)
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            max_loss = np.min(returns) if len(returns) > 0 else 0.0

            # Sortino/Sharpe ratio (if Sharpe provided)
            sortino_sharpe_ratio = sortino_30d / sharpe_ratio if (sharpe_ratio and sharpe_ratio != 0) else 0.0

            # Generate summary
            summary = self._generate_summary(sortino_30d, ann_return, downside_dev)

            metrics = SortinoMetrics(
                sortino_ratio_7d=sortino_7d,
                sortino_ratio_14d=sortino_14d,
                sortino_ratio_30d=sortino_30d,
                sortino_ratio_90d=sortino_90d,
                annualized_return=ann_return,
                downside_deviation=downside_dev,
                upside_deviation=upside_dev,
                total_trades=total_trades,
                losing_trades=losing_trades,
                avg_loss=avg_loss,
                max_loss=max_loss,
                sharpe_ratio=sharpe_ratio,
                sortino_sharpe_ratio=sortino_sharpe_ratio,
                summary=summary,
                metrics={
                    'sortino_7d': sortino_7d,
                    'sortino_14d': sortino_14d,
                    'sortino_30d': sortino_30d,
                    'downside_dev': downside_dev,
                    'upside_dev': upside_dev
                }
            )

            logger.debug(
                f"Sortino Metrics: 7d={sortino_7d:.2f}, 14d={sortino_14d:.2f}, 30d={sortino_30d:.2f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Sortino calculation failed: {e}")
            return self._insufficient_data_metrics()

    def _calculate_sortino_window(self, days: int) -> float:
        """Calculate Sortino ratio for a specific window"""
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

        returns_arr = np.array(window_returns)
        mean_return = np.mean(returns_arr)

        # Downside deviation: only consider returns below MAR
        downside_returns = returns_arr[returns_arr < self.mar]
        if len(downside_returns) < 2:
            downside_dev = np.std(returns_arr, ddof=1)  # Fallback to normal std
        else:
            # Calculate downside deviation
            downside_dev = np.sqrt(np.mean((downside_returns - self.mar) ** 2))

        if downside_dev == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / 365

        # Average trades per day in this window
        trades_per_day = len(window_returns) / days

        # Sortino = (Mean Return - RF) / Downside Deviation
        # Annualize: multiply by sqrt(trades_per_year)
        trades_per_year = trades_per_day * 365
        sortino = (mean_return - daily_rf) / downside_dev * np.sqrt(trades_per_year)

        return float(sortino)

    def _annualize_metrics(self, returns: List[float]) -> tuple:
        """Annualize return and volatility metrics"""
        if len(returns) < 2:
            return 0.0, 0.0, 0.0

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

        # Downside deviation (annualized)
        downside_returns = [r for r in returns if r < self.mar]
        if len(downside_returns) > 0:
            downside_dev = np.sqrt(np.mean([(r - self.mar)**2 for r in downside_returns]))
            ann_downside_dev = downside_dev * np.sqrt(trades_per_year)
        else:
            ann_downside_dev = 0.0

        # Upside deviation (annualized)
        upside_returns = [r for r in returns if r > self.mar]
        if len(upside_returns) > 0:
            upside_dev = np.sqrt(np.mean([(r - self.mar)**2 for r in upside_returns]))
            ann_upside_dev = upside_dev * np.sqrt(trades_per_year)
        else:
            ann_upside_dev = 0.0

        return float(ann_return), float(ann_downside_dev), float(ann_upside_dev)

    def _generate_summary(
        self,
        sortino_30d: float,
        ann_return: float,
        downside_dev: float
    ) -> str:
        """Generate human-readable summary"""
        # Sortino interpretation
        if sortino_30d >= 3.0:
            quality = "EXCELLENT"
        elif sortino_30d >= 2.0:
            quality = "VERY GOOD"
        elif sortino_30d >= 1.0:
            quality = "GOOD"
        elif sortino_30d >= 0:
            quality = "ACCEPTABLE"
        else:
            quality = "NEGATIVE"

        summary = (
            f"30-day Sortino: {sortino_30d:.2f} ({quality}) | "
            f"Ann. Return: {ann_return:+.1%} | "
            f"Downside Dev: {downside_dev:.1%}"
        )

        return summary

    def _insufficient_data_metrics(self) -> SortinoMetrics:
        """Return default metrics when insufficient data"""
        return SortinoMetrics(
            sortino_ratio_7d=0.0,
            sortino_ratio_14d=0.0,
            sortino_ratio_30d=0.0,
            sortino_ratio_90d=None,
            annualized_return=0.0,
            downside_deviation=0.0,
            upside_deviation=0.0,
            total_trades=len(self.returns_history),
            losing_trades=0,
            avg_loss=0.0,
            max_loss=0.0,
            sharpe_ratio=None,
            sortino_sharpe_ratio=0.0,
            summary=f"Insufficient data: {len(self.returns_history)} trades (need 5+)",
            metrics={}
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("SORTINO RATIO TRACKER TEST")
    print("=" * 70)

    # Scenario: Asymmetric returns (big wins, small losses)
    print("\n[Scenario] Asymmetric Strategy (Sortino > Sharpe):")

    tracker = SortinoRatioTracker()
    np.random.seed(42)
    base_date = datetime.now() - timedelta(days=30)

    for i in range(50):
        # 60% win rate, wins larger than losses
        if np.random.random() < 0.60:
            return_pct = np.random.normal(0.030, 0.010)  # ~3% wins
        else:
            return_pct = np.random.normal(-0.015, 0.005)  # ~-1.5% losses (smaller)

        timestamp = base_date + timedelta(days=i*0.6)
        tracker.record_return(return_pct, timestamp)

    # Calculate Sharpe (for comparison)
    returns = [r['return_pct'] for r in tracker.returns_history]
    sharpe = (np.mean(returns) - 0.05/365) / np.std(returns, ddof=1) * np.sqrt(len(returns)/30 * 365)

    metrics = tracker.get_sortino_metrics(sharpe_ratio=sharpe)

    print(f"  7-day Sortino:        {metrics.sortino_ratio_7d:.2f}")
    print(f"  14-day Sortino:       {metrics.sortino_ratio_14d:.2f}")
    print(f"  30-day Sortino:       {metrics.sortino_ratio_30d:.2f}")
    print(f"  Ann. Return:          {metrics.annualized_return:+.1%}")
    print(f"  Downside Deviation:   {metrics.downside_deviation:.1%}")
    print(f"  Upside Deviation:     {metrics.upside_deviation:.1%}")
    print(f"  Losing Trades:        {metrics.losing_trades} / {metrics.total_trades}")
    print(f"  Avg Loss:             {metrics.avg_loss:.2%}")
    print(f"  Max Loss:             {metrics.max_loss:.2%}")
    print(f"\n  Sharpe Ratio:         {sharpe:.2f}")
    print(f"  Sortino / Sharpe:     {metrics.sortino_sharpe_ratio:.2f}x")
    print(f"\n  Summary: {metrics.summary}")

    if metrics.sortino_sharpe_ratio > 1.0:
        print(f"\n  ✅ Sortino > Sharpe: Strategy has favorable asymmetry (larger wins than losses)")
    else:
        print(f"\n  ⚠️ Sortino < Sharpe: Strategy may have unfavorable asymmetry")

    print("\n" + "=" * 70)
    print("✅ Sortino Ratio Tracker ready for production!")
    print("=" * 70)
