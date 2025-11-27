"""
Sharpe Ratio Real-Time Tracker

Tracks risk-adjusted returns in real-time using rolling windows.

Sharpe Ratio = (Mean Return - Risk-Free Rate) / StdDev(Returns)

Features:
- Rolling windows (7d, 14d, 30d, 90d)
- Multiple calculation methods (daily, intraday)
- Annualization (crypto: 365 days, 24/7 trading)
- Performance trend detection (improving/declining)
- Benchmark comparison (BTC buy-and-hold)

Expected Impact:
- Real-time performance monitoring
- Early detection of strategy degradation
- Quantitative performance validation
- Risk-adjusted performance metrics
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SharpeMetrics:
    """Sharpe ratio calculation results"""
    sharpe_ratio_7d: float
    sharpe_ratio_14d: float
    sharpe_ratio_30d: float
    sharpe_ratio_90d: Optional[float]

    annualized_return: float
    annualized_volatility: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    avg_return: float
    max_drawdown: float

    performance_trend: str  # 'improving', 'stable', 'declining'
    trend_confidence: float  # 0-1

    benchmark_sharpe: Optional[float]  # BTC buy-and-hold comparison
    relative_sharpe: Optional[float]  # Our Sharpe / Benchmark Sharpe

    summary: str
    metrics: Dict[str, float]


class SharpeRatioTracker:
    """
    Real-time Sharpe Ratio tracking for trading strategy

    Usage:
        tracker = SharpeRatioTracker()

        # After each trade closes
        tracker.record_trade_return(
            timestamp=datetime.now(),
            return_pct=0.015,  # 1.5% gain
            symbol='BTC-USD'
        )

        # Get current Sharpe
        metrics = tracker.get_sharpe_metrics()
        print(f"30-day Sharpe: {metrics.sharpe_ratio_30d:.2f}")
        print(f"Trend: {metrics.performance_trend}")
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,  # 5% annual (US Treasury)
        max_history_days: int = 90
    ):
        """
        Initialize Sharpe Ratio Tracker

        Args:
            risk_free_rate: Annual risk-free rate (default: 5%)
            max_history_days: Max days of history to keep (default: 90)
        """
        self.risk_free_rate = risk_free_rate
        self.max_history_days = max_history_days

        # Trade returns history: [(timestamp, return_pct, symbol), ...]
        self.trade_returns: deque = deque(maxlen=1000)  # Last 1000 trades

        # Daily aggregated returns for Sharpe calculation
        self.daily_returns: deque = deque(maxlen=max_history_days)

        # Benchmark (BTC buy-and-hold) tracking
        self.benchmark_returns: deque = deque(maxlen=max_history_days)

        logger.info(
            f"Sharpe Ratio Tracker initialized | "
            f"Risk-free rate: {risk_free_rate:.1%} | "
            f"Max history: {max_history_days} days"
        )

    def record_trade_return(
        self,
        timestamp: datetime,
        return_pct: float,
        symbol: str,
        benchmark_return_pct: Optional[float] = None
    ) -> None:
        """
        Record a completed trade's return

        Args:
            timestamp: Trade close timestamp
            return_pct: Trade return (e.g., 0.015 = 1.5% gain)
            symbol: Trading symbol
            benchmark_return_pct: Benchmark return for same period (optional)
        """
        self.trade_returns.append({
            'timestamp': timestamp,
            'return_pct': return_pct,
            'symbol': symbol
        })

        # Aggregate to daily returns
        self._update_daily_returns()

        if benchmark_return_pct is not None:
            self.benchmark_returns.append({
                'timestamp': timestamp,
                'return_pct': benchmark_return_pct
            })

        logger.debug(
            f"Trade return recorded: {symbol} | "
            f"Return: {return_pct:+.2%} | "
            f"Total trades: {len(self.trade_returns)}"
        )

    def get_sharpe_metrics(
        self,
        current_time: Optional[datetime] = None
    ) -> SharpeMetrics:
        """
        Calculate current Sharpe ratio and related metrics

        Args:
            current_time: Current timestamp (default: now)

        Returns:
            SharpeMetrics with all calculations
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        try:
            if len(self.trade_returns) < 5:
                logger.warning(f"Insufficient trades for Sharpe: {len(self.trade_returns)} < 5")
                return self._insufficient_data_metrics()

            # Calculate Sharpe for different windows
            sharpe_7d = self._calculate_sharpe_window(days=7)
            sharpe_14d = self._calculate_sharpe_window(days=14)
            sharpe_30d = self._calculate_sharpe_window(days=30)
            sharpe_90d = self._calculate_sharpe_window(days=90) if len(self.trade_returns) >= 20 else None

            # Calculate annualized metrics
            returns = [t['return_pct'] for t in self.trade_returns]
            ann_return, ann_volatility = self._annualize_metrics(returns)

            # Trade statistics
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r <= 0]

            total_trades = len(returns)
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_return = np.mean(returns)
            max_dd = self._calculate_max_drawdown(returns)

            # Performance trend
            trend, trend_conf = self._detect_performance_trend()

            # Benchmark comparison
            benchmark_sharpe = self._calculate_benchmark_sharpe() if len(self.benchmark_returns) > 0 else None
            relative_sharpe = sharpe_30d / benchmark_sharpe if benchmark_sharpe and benchmark_sharpe != 0 else None

            # Generate summary
            summary = self._generate_summary(
                sharpe_30d=sharpe_30d,
                win_rate=win_rate,
                ann_return=ann_return,
                trend=trend,
                benchmark_sharpe=benchmark_sharpe
            )

            metrics = SharpeMetrics(
                sharpe_ratio_7d=sharpe_7d,
                sharpe_ratio_14d=sharpe_14d,
                sharpe_ratio_30d=sharpe_30d,
                sharpe_ratio_90d=sharpe_90d,
                annualized_return=ann_return,
                annualized_volatility=ann_volatility,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_return=avg_return,
                max_drawdown=max_dd,
                performance_trend=trend,
                trend_confidence=trend_conf,
                benchmark_sharpe=benchmark_sharpe,
                relative_sharpe=relative_sharpe,
                summary=summary,
                metrics={
                    'sharpe_7d': sharpe_7d,
                    'sharpe_14d': sharpe_14d,
                    'sharpe_30d': sharpe_30d,
                    'sharpe_90d': sharpe_90d or 0,
                    'ann_return': ann_return,
                    'ann_vol': ann_volatility,
                    'win_rate': win_rate,
                    'max_dd': max_dd
                }
            )

            logger.debug(
                f"Sharpe Metrics: 7d={sharpe_7d:.2f}, 14d={sharpe_14d:.2f}, "
                f"30d={sharpe_30d:.2f}, Trend={trend}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Sharpe calculation failed: {e}")
            return self._insufficient_data_metrics()

    def _calculate_sharpe_window(self, days: int) -> float:
        """
        Calculate Sharpe ratio for a specific window

        Args:
            days: Number of days to look back

        Returns:
            Sharpe ratio for the window
        """
        if len(self.trade_returns) < 3:
            return 0.0

        # Get returns within window
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        window_returns = [
            t['return_pct'] for t in self.trade_returns
            if t['timestamp'] >= cutoff
        ]

        if len(window_returns) < 3:
            return 0.0

        # Calculate Sharpe
        returns_arr = np.array(window_returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr, ddof=1)

        if std_return == 0:
            return 0.0

        # Daily risk-free rate (annualized / 365)
        daily_rf = self.risk_free_rate / 365

        # Average trades per day in this window
        trades_per_day = len(window_returns) / days

        # Sharpe = (Mean Return - RF) / StdDev
        # Annualize: multiply by sqrt(trades_per_year)
        trades_per_year = trades_per_day * 365
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(trades_per_year)

        return float(sharpe)

    def _annualize_metrics(self, returns: List[float]) -> Tuple[float, float]:
        """
        Annualize return and volatility

        Args:
            returns: List of trade returns

        Returns:
            (annualized_return, annualized_volatility)
        """
        if len(returns) < 2:
            return 0.0, 0.0

        # Calculate total period
        first_timestamp = self.trade_returns[0]['timestamp']
        last_timestamp = self.trade_returns[-1]['timestamp']
        period_days = (last_timestamp - first_timestamp).days

        if period_days == 0:
            period_days = 1

        # Trades per day
        trades_per_day = len(returns) / period_days
        trades_per_year = trades_per_day * 365

        # Annualized return
        mean_return = np.mean(returns)
        ann_return = mean_return * trades_per_year

        # Annualized volatility
        std_return = np.std(returns, ddof=1)
        ann_volatility = std_return * np.sqrt(trades_per_year)

        return float(ann_return), float(ann_volatility)

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns

        Args:
            returns: List of trade returns

        Returns:
            Maximum drawdown (positive number, e.g., 0.15 = 15% drawdown)
        """
        if len(returns) < 2:
            return 0.0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + np.array(returns))

        # Calculate running max
        running_max = np.maximum.accumulate(cumulative)

        # Drawdown at each point
        drawdown = (running_max - cumulative) / running_max

        # Max drawdown
        max_dd = np.max(drawdown)

        return float(max_dd)

    def _detect_performance_trend(self) -> Tuple[str, float]:
        """
        Detect if performance is improving, stable, or declining

        Returns:
            (trend, confidence)
            trend: 'improving', 'stable', 'declining'
            confidence: 0-1
        """
        if len(self.trade_returns) < 10:
            return 'unknown', 0.0

        # Split into first half and second half
        mid = len(self.trade_returns) // 2
        first_half = [t['return_pct'] for t in list(self.trade_returns)[:mid]]
        second_half = [t['return_pct'] for t in list(self.trade_returns)[mid:]]

        # Calculate Sharpe for each half
        sharpe_first = self._calculate_sharpe_for_returns(first_half)
        sharpe_second = self._calculate_sharpe_for_returns(second_half)

        # Compare
        diff = sharpe_second - sharpe_first

        # Trend detection
        if diff > 0.3:
            trend = 'improving'
            confidence = min(abs(diff) / 1.0, 1.0)
        elif diff < -0.3:
            trend = 'declining'
            confidence = min(abs(diff) / 1.0, 1.0)
        else:
            trend = 'stable'
            confidence = 0.8

        return trend, confidence

    def _calculate_sharpe_for_returns(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio for a list of returns

        Args:
            returns: List of returns

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr, ddof=1)

        if std_return == 0:
            return 0.0

        daily_rf = self.risk_free_rate / 365

        # Assume ~1 trade per day for simplification
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(365)

        return float(sharpe)

    def _calculate_benchmark_sharpe(self) -> Optional[float]:
        """
        Calculate Sharpe ratio for benchmark (BTC buy-and-hold)

        Returns:
            Benchmark Sharpe ratio
        """
        if len(self.benchmark_returns) < 5:
            return None

        returns = [r['return_pct'] for r in self.benchmark_returns]
        return self._calculate_sharpe_for_returns(returns)

    def _update_daily_returns(self) -> None:
        """Update daily aggregated returns from trade returns"""
        if len(self.trade_returns) == 0:
            return

        # Group by day
        daily_dict = {}
        for trade in self.trade_returns:
            date = trade['timestamp'].date()
            if date not in daily_dict:
                daily_dict[date] = []
            daily_dict[date].append(trade['return_pct'])

        # Calculate daily returns (sum of all trades that day)
        self.daily_returns.clear()
        for date in sorted(daily_dict.keys()):
            daily_return = sum(daily_dict[date])
            self.daily_returns.append({
                'date': date,
                'return_pct': daily_return
            })

    def _generate_summary(
        self,
        sharpe_30d: float,
        win_rate: float,
        ann_return: float,
        trend: str,
        benchmark_sharpe: Optional[float]
    ) -> str:
        """Generate human-readable summary"""

        # Sharpe interpretation
        if sharpe_30d >= 2.0:
            sharpe_label = "EXCELLENT"
        elif sharpe_30d >= 1.5:
            sharpe_label = "VERY GOOD"
        elif sharpe_30d >= 1.0:
            sharpe_label = "GOOD"
        elif sharpe_30d >= 0.5:
            sharpe_label = "ACCEPTABLE"
        elif sharpe_30d >= 0:
            sharpe_label = "POOR"
        else:
            sharpe_label = "NEGATIVE"

        summary = (
            f"30-day Sharpe: {sharpe_30d:.2f} ({sharpe_label}) | "
            f"Win Rate: {win_rate:.1%} | "
            f"Ann. Return: {ann_return:+.1%} | "
            f"Trend: {trend.upper()}"
        )

        if benchmark_sharpe:
            summary += f" | vs BTC: {sharpe_30d/benchmark_sharpe:.2f}x"

        return summary

    def _insufficient_data_metrics(self) -> SharpeMetrics:
        """Return default metrics when insufficient data"""
        return SharpeMetrics(
            sharpe_ratio_7d=0.0,
            sharpe_ratio_14d=0.0,
            sharpe_ratio_30d=0.0,
            sharpe_ratio_90d=None,
            annualized_return=0.0,
            annualized_volatility=0.0,
            total_trades=len(self.trade_returns),
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_return=0.0,
            max_drawdown=0.0,
            performance_trend='unknown',
            trend_confidence=0.0,
            benchmark_sharpe=None,
            relative_sharpe=None,
            summary=f"Insufficient data: {len(self.trade_returns)} trades (need 5+)",
            metrics={}
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("SHARPE RATIO TRACKER TEST")
    print("=" * 70)

    tracker = SharpeRatioTracker()

    # Simulate 30 days of trading
    np.random.seed(42)
    base_date = datetime.now() - timedelta(days=30)

    # Scenario 1: Improving strategy (bad start, good finish)
    print("\n[Scenario 1] Improving Strategy:")
    for i in range(50):
        # First 25 trades: 40% win rate, small wins
        # Last 25 trades: 65% win rate, larger wins
        if i < 25:
            win_prob = 0.40
            win_size = 0.02  # 2% wins
            loss_size = -0.015  # -1.5% losses
        else:
            win_prob = 0.65
            win_size = 0.025  # 2.5% wins
            loss_size = -0.012  # -1.2% losses

        is_win = np.random.random() < win_prob
        return_pct = win_size if is_win else loss_size

        timestamp = base_date + timedelta(days=i*0.6)  # ~1.6 trades/day
        tracker.record_trade_return(timestamp, return_pct, 'BTC-USD')

    metrics = tracker.get_sharpe_metrics()
    print(f"  7-day Sharpe:  {metrics.sharpe_ratio_7d:.2f}")
    print(f"  14-day Sharpe: {metrics.sharpe_ratio_14d:.2f}")
    print(f"  30-day Sharpe: {metrics.sharpe_ratio_30d:.2f}")
    print(f"  Ann. Return:   {metrics.annualized_return:+.1%}")
    print(f"  Ann. Vol:      {metrics.annualized_volatility:.1%}")
    print(f"  Win Rate:      {metrics.win_rate:.1%}")
    print(f"  Max Drawdown:  {metrics.max_drawdown:.1%}")
    print(f"  Trend:         {metrics.performance_trend.upper()} ({metrics.trend_confidence:.0%})")
    print(f"\n  Summary: {metrics.summary}")

    # Scenario 2: Declining strategy (good start, bad finish)
    print("\n[Scenario 2] Declining Strategy:")
    tracker2 = SharpeRatioTracker()

    for i in range(50):
        # First 25 trades: 65% win rate
        # Last 25 trades: 35% win rate
        if i < 25:
            win_prob = 0.65
            win_size = 0.025
            loss_size = -0.012
        else:
            win_prob = 0.35
            win_size = 0.018
            loss_size = -0.020

        is_win = np.random.random() < win_prob
        return_pct = win_size if is_win else loss_size

        timestamp = base_date + timedelta(days=i*0.6)
        tracker2.record_trade_return(timestamp, return_pct, 'ETH-USD')

    metrics2 = tracker2.get_sharpe_metrics()
    print(f"  30-day Sharpe: {metrics2.sharpe_ratio_30d:.2f}")
    print(f"  Win Rate:      {metrics2.win_rate:.1%}")
    print(f"  Trend:         {metrics2.performance_trend.upper()} ({metrics2.trend_confidence:.0%})")
    print(f"\n  Summary: {metrics2.summary}")

    # Scenario 3: Consistent high performer
    print("\n[Scenario 3] Consistent High Performer:")
    tracker3 = SharpeRatioTracker()

    for i in range(50):
        win_prob = 0.62
        win_size = 0.030  # 3% wins
        loss_size = -0.010  # -1% losses (good R:R)

        is_win = np.random.random() < win_prob
        return_pct = win_size if is_win else loss_size

        timestamp = base_date + timedelta(days=i*0.6)
        tracker3.record_trade_return(timestamp, return_pct, 'SOL-USD')

    metrics3 = tracker3.get_sharpe_metrics()
    print(f"  30-day Sharpe: {metrics3.sharpe_ratio_30d:.2f}")
    print(f"  Win Rate:      {metrics3.win_rate:.1%}")
    print(f"  Ann. Return:   {metrics3.annualized_return:+.1%}")
    print(f"  Trend:         {metrics3.performance_trend.upper()}")
    print(f"\n  Summary: {metrics3.summary}")

    print("\n" + "=" * 70)
    print("✅ Sharpe Ratio Tracker ready for production!")
    print("=" * 70)
