"""
HYDRA 3.0 - Walk-Forward Validator

Validates trading strategy performance using walk-forward analysis.

Walk-forward validation splits historical data into rolling windows:
1. Training window (in-sample): Where strategy "learns"
2. Testing window (out-of-sample): Where we measure true performance

This detects overfitting by comparing in-sample vs out-of-sample results.

Key Metrics:
- Sharpe ratio consistency across windows
- Win rate stability
- Drawdown behavior
- Performance decay detection
- Overfitting score

Phase 2, Week 2 - Step 17
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
from enum import Enum
import json
import math
import statistics


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"           # Strategy is robust
    WARNING = "warning"         # Minor concerns
    FAILED = "failed"           # Strategy shows overfitting
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TradeRecord:
    """Single trade for validation."""
    timestamp: datetime
    asset: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    pnl_percent: float
    outcome: str  # "win" or "loss"
    holding_period_hours: float


@dataclass
class WindowMetrics:
    """Performance metrics for a single validation window."""
    window_id: int
    start_date: datetime
    end_date: datetime
    is_training: bool  # True = in-sample, False = out-of-sample

    # Core metrics
    num_trades: int
    win_rate: float
    total_pnl_percent: float
    avg_pnl_percent: float
    sharpe_ratio: float
    max_drawdown: float

    # Trade characteristics
    avg_win_pnl: float
    avg_loss_pnl: float
    profit_factor: float  # Gross profit / Gross loss

    # Consistency
    win_streak_max: int
    loss_streak_max: int


@dataclass
class OverfitMetrics:
    """Overfitting detection metrics."""
    train_sharpe: float
    test_sharpe: float
    sharpe_decay: float  # (train - test) / train

    train_win_rate: float
    test_win_rate: float
    win_rate_decay: float

    train_pnl: float
    test_pnl: float
    pnl_decay: float

    overfit_score: float  # 0-1 score (higher = more overfit)
    overfit_detected: bool


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""
    timestamp: datetime
    engine: str
    status: ValidationStatus

    # Summary metrics
    total_trades: int
    overall_sharpe: float
    overall_win_rate: float
    overall_pnl: float

    # Window breakdown
    num_windows: int
    training_windows: List[WindowMetrics]
    testing_windows: List[WindowMetrics]

    # Overfitting analysis
    overfit_metrics: OverfitMetrics

    # Stability metrics
    sharpe_stability: float  # Std dev of Sharpe across test windows
    win_rate_stability: float

    # Recommendations
    recommendations: List[str]


class WalkForwardValidator:
    """
    Walk-Forward Validator for HYDRA 3.0.

    Validates trading strategies using rolling window analysis
    to detect overfitting and assess robustness.
    """

    # Configuration
    MIN_TRADES_FOR_VALIDATION = 20  # Minimum trades needed
    MIN_TRADES_PER_WINDOW = 5       # Minimum trades per window

    # Default window sizes (in days)
    DEFAULT_TRAIN_WINDOW_DAYS = 7   # 1 week training
    DEFAULT_TEST_WINDOW_DAYS = 2    # 2 days testing
    DEFAULT_STEP_DAYS = 2           # Roll forward 2 days

    # Thresholds
    SHARPE_THRESHOLD = 0.5          # Minimum acceptable Sharpe
    OVERFIT_THRESHOLD = 0.3         # Max acceptable decay (30%)
    WIN_RATE_THRESHOLD = 0.45       # Minimum win rate

    def __init__(self, data_dir: Optional[Path] = None):
        # Auto-detect data directory based on environment
        if data_dir is None:
            from ..config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Validation history
        self.validation_history: List[WalkForwardResult] = []

        # Persistence
        self.results_file = self.data_dir / "walk_forward_results.jsonl"

        logger.info("[WalkForwardValidator] Initialized")

    def validate(
        self,
        trades: List[TradeRecord],
        engine: str,
        train_window_days: int = None,
        test_window_days: int = None,
        step_days: int = None
    ) -> WalkForwardResult:
        """
        Run walk-forward validation on trade history.

        Args:
            trades: List of historical trades
            engine: Engine name being validated
            train_window_days: Training window size
            test_window_days: Testing window size
            step_days: Step size for rolling

        Returns:
            WalkForwardResult with full analysis
        """
        train_days = train_window_days or self.DEFAULT_TRAIN_WINDOW_DAYS
        test_days = test_window_days or self.DEFAULT_TEST_WINDOW_DAYS
        step = step_days or self.DEFAULT_STEP_DAYS

        now = datetime.now(timezone.utc)

        # Check minimum data requirement
        if len(trades) < self.MIN_TRADES_FOR_VALIDATION:
            return self._insufficient_data_result(engine, len(trades))

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Generate validation windows
        windows = self._generate_windows(
            sorted_trades, train_days, test_days, step
        )

        if not windows:
            return self._insufficient_data_result(engine, len(trades))

        # Calculate metrics for each window
        training_windows = []
        testing_windows = []

        for window_id, (train_trades, test_trades, start, mid, end) in enumerate(windows):
            # Training window metrics
            if len(train_trades) >= self.MIN_TRADES_PER_WINDOW:
                train_metrics = self._calculate_window_metrics(
                    window_id, train_trades, start, mid, is_training=True
                )
                training_windows.append(train_metrics)

            # Testing window metrics
            if len(test_trades) >= self.MIN_TRADES_PER_WINDOW:
                test_metrics = self._calculate_window_metrics(
                    window_id, test_trades, mid, end, is_training=False
                )
                testing_windows.append(test_metrics)

        # Calculate overfitting metrics
        overfit_metrics = self._calculate_overfit_metrics(
            training_windows, testing_windows
        )

        # Calculate stability metrics
        sharpe_stability = self._calculate_stability(
            [w.sharpe_ratio for w in testing_windows]
        )
        win_rate_stability = self._calculate_stability(
            [w.win_rate for w in testing_windows]
        )

        # Overall metrics (from test windows only - true out-of-sample)
        if testing_windows:
            overall_sharpe = statistics.mean([w.sharpe_ratio for w in testing_windows])
            overall_win_rate = statistics.mean([w.win_rate for w in testing_windows])
            overall_pnl = sum([w.total_pnl_percent for w in testing_windows])
        else:
            overall_sharpe = 0.0
            overall_win_rate = 0.0
            overall_pnl = 0.0

        # Determine validation status
        status = self._determine_status(
            overfit_metrics, overall_sharpe, overall_win_rate, testing_windows
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            status, overfit_metrics, overall_sharpe, overall_win_rate,
            sharpe_stability, testing_windows
        )

        result = WalkForwardResult(
            timestamp=now,
            engine=engine,
            status=status,
            total_trades=len(trades),
            overall_sharpe=overall_sharpe,
            overall_win_rate=overall_win_rate,
            overall_pnl=overall_pnl,
            num_windows=len(windows),
            training_windows=training_windows,
            testing_windows=testing_windows,
            overfit_metrics=overfit_metrics,
            sharpe_stability=sharpe_stability,
            win_rate_stability=win_rate_stability,
            recommendations=recommendations
        )

        # Save and log result
        self._save_result(result)
        self._log_result(result)

        self.validation_history.append(result)

        return result

    def _generate_windows(
        self,
        trades: List[TradeRecord],
        train_days: int,
        test_days: int,
        step_days: int
    ) -> List[Tuple[List[TradeRecord], List[TradeRecord], datetime, datetime, datetime]]:
        """
        Generate rolling train/test windows.

        Returns:
            List of (train_trades, test_trades, start, mid, end)
        """
        if not trades:
            return []

        windows = []

        first_trade = trades[0].timestamp
        last_trade = trades[-1].timestamp

        total_days = (last_trade - first_trade).days
        window_size = train_days + test_days

        if total_days < window_size:
            return []

        # Generate rolling windows
        current_start = first_trade

        while True:
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)

            if test_end > last_trade:
                break

            # Get trades in each window
            train_trades = [
                t for t in trades
                if current_start <= t.timestamp < train_end
            ]

            test_trades = [
                t for t in trades
                if train_end <= t.timestamp < test_end
            ]

            if train_trades and test_trades:
                windows.append((
                    train_trades,
                    test_trades,
                    current_start,
                    train_end,
                    test_end
                ))

            # Step forward
            current_start += timedelta(days=step_days)

        return windows

    def _calculate_window_metrics(
        self,
        window_id: int,
        trades: List[TradeRecord],
        start: datetime,
        end: datetime,
        is_training: bool
    ) -> WindowMetrics:
        """Calculate performance metrics for a window."""
        if not trades:
            return self._empty_window_metrics(window_id, start, end, is_training)

        num_trades = len(trades)
        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]

        win_rate = len(wins) / num_trades if num_trades > 0 else 0

        # P&L metrics
        pnls = [t.pnl_percent for t in trades]
        total_pnl = sum(pnls)
        avg_pnl = statistics.mean(pnls) if pnls else 0

        # Win/Loss averages
        avg_win = statistics.mean([t.pnl_percent for t in wins]) if wins else 0
        avg_loss = statistics.mean([t.pnl_percent for t in losses]) if losses else 0

        # Profit factor
        gross_profit = sum([t.pnl_percent for t in wins]) if wins else 0
        gross_loss = abs(sum([t.pnl_percent for t in losses])) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (simplified)
        sharpe = self._calculate_sharpe(pnls)

        # Max drawdown
        max_dd = self._calculate_max_drawdown(pnls)

        # Streaks
        win_streak, loss_streak = self._calculate_streaks(trades)

        return WindowMetrics(
            window_id=window_id,
            start_date=start,
            end_date=end,
            is_training=is_training,
            num_trades=num_trades,
            win_rate=win_rate,
            total_pnl_percent=total_pnl,
            avg_pnl_percent=avg_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_win_pnl=avg_win,
            avg_loss_pnl=avg_loss,
            profit_factor=profit_factor,
            win_streak_max=win_streak,
            loss_streak_max=loss_streak
        )

    def _calculate_sharpe(self, returns: List[float], risk_free: float = 0) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0.0

        # Annualized (assuming daily returns, ~252 trading days)
        sharpe = (mean_return - risk_free) / std_return * math.sqrt(252)

        return sharpe

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0

        cumulative = []
        running_total = 0

        for r in returns:
            running_total += r
            cumulative.append(running_total)

        peak = cumulative[0]
        max_dd = 0

        for value in cumulative:
            if value > peak:
                peak = value
            dd = (peak - value) / max(abs(peak), 0.0001)
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_streaks(self, trades: List[TradeRecord]) -> Tuple[int, int]:
        """Calculate max win and loss streaks."""
        if not trades:
            return 0, 0

        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0

        for trade in trades:
            if trade.outcome == "win":
                current_win += 1
                current_loss = 0
                max_win_streak = max(max_win_streak, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss_streak = max(max_loss_streak, current_loss)

        return max_win_streak, max_loss_streak

    def _calculate_overfit_metrics(
        self,
        train_windows: List[WindowMetrics],
        test_windows: List[WindowMetrics]
    ) -> OverfitMetrics:
        """Calculate overfitting detection metrics."""
        if not train_windows or not test_windows:
            return OverfitMetrics(
                train_sharpe=0, test_sharpe=0, sharpe_decay=0,
                train_win_rate=0, test_win_rate=0, win_rate_decay=0,
                train_pnl=0, test_pnl=0, pnl_decay=0,
                overfit_score=0, overfit_detected=False
            )

        # Average metrics across windows
        train_sharpe = statistics.mean([w.sharpe_ratio for w in train_windows])
        test_sharpe = statistics.mean([w.sharpe_ratio for w in test_windows])

        train_wr = statistics.mean([w.win_rate for w in train_windows])
        test_wr = statistics.mean([w.win_rate for w in test_windows])

        train_pnl = statistics.mean([w.total_pnl_percent for w in train_windows])
        test_pnl = statistics.mean([w.total_pnl_percent for w in test_windows])

        # Calculate decay (performance drop from train to test)
        sharpe_decay = (train_sharpe - test_sharpe) / max(abs(train_sharpe), 0.001)
        wr_decay = (train_wr - test_wr) / max(train_wr, 0.001)
        pnl_decay = (train_pnl - test_pnl) / max(abs(train_pnl), 0.001)

        # Overall overfit score (0-1)
        # High decay = high overfit
        overfit_score = min(1.0, max(0.0, (
            0.4 * max(0, sharpe_decay) +
            0.3 * max(0, wr_decay) +
            0.3 * max(0, pnl_decay)
        )))

        overfit_detected = overfit_score > self.OVERFIT_THRESHOLD

        return OverfitMetrics(
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            sharpe_decay=sharpe_decay,
            train_win_rate=train_wr,
            test_win_rate=test_wr,
            win_rate_decay=wr_decay,
            train_pnl=train_pnl,
            test_pnl=test_pnl,
            pnl_decay=pnl_decay,
            overfit_score=overfit_score,
            overfit_detected=overfit_detected
        )

    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability (inverse of coefficient of variation)."""
        if len(values) < 2:
            return 1.0

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        if mean_val == 0:
            return 0.0

        cv = std_val / abs(mean_val)
        stability = 1 / (1 + cv)  # 0-1 scale, higher = more stable

        return stability

    def _determine_status(
        self,
        overfit: OverfitMetrics,
        sharpe: float,
        win_rate: float,
        test_windows: List[WindowMetrics]
    ) -> ValidationStatus:
        """Determine overall validation status."""
        if not test_windows:
            return ValidationStatus.INSUFFICIENT_DATA

        # Failed conditions
        if overfit.overfit_detected:
            return ValidationStatus.FAILED

        if sharpe < 0:
            return ValidationStatus.FAILED

        if win_rate < 0.35:
            return ValidationStatus.FAILED

        # Warning conditions
        if sharpe < self.SHARPE_THRESHOLD:
            return ValidationStatus.WARNING

        if win_rate < self.WIN_RATE_THRESHOLD:
            return ValidationStatus.WARNING

        if overfit.overfit_score > 0.15:
            return ValidationStatus.WARNING

        return ValidationStatus.PASSED

    def _generate_recommendations(
        self,
        status: ValidationStatus,
        overfit: OverfitMetrics,
        sharpe: float,
        win_rate: float,
        stability: float,
        test_windows: List[WindowMetrics]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        if status == ValidationStatus.FAILED:
            if overfit.overfit_detected:
                recs.append(
                    f"CRITICAL: Overfitting detected (score: {overfit.overfit_score:.2f}). "
                    "Strategy performs much worse on unseen data."
                )
                recs.append(
                    "Consider: Simpler decision rules, fewer parameters, "
                    "or more diverse training data."
                )

            if sharpe < 0:
                recs.append(
                    "CRITICAL: Negative Sharpe ratio indicates strategy "
                    "underperforms risk-free return."
                )

            if win_rate < 0.35:
                recs.append(
                    f"CRITICAL: Win rate ({win_rate:.1%}) is dangerously low. "
                    "Review entry criteria."
                )

        elif status == ValidationStatus.WARNING:
            if sharpe < self.SHARPE_THRESHOLD:
                recs.append(
                    f"Sharpe ratio ({sharpe:.2f}) below threshold ({self.SHARPE_THRESHOLD}). "
                    "Consider improving risk-adjusted returns."
                )

            if win_rate < self.WIN_RATE_THRESHOLD:
                recs.append(
                    f"Win rate ({win_rate:.1%}) marginally low. "
                    "May need tighter entry criteria."
                )

            if stability < 0.5:
                recs.append(
                    f"Performance stability is low ({stability:.2f}). "
                    "Results vary significantly across time periods."
                )

        else:  # PASSED
            recs.append(
                f"Strategy passed validation with Sharpe {sharpe:.2f}, "
                f"WR {win_rate:.1%}, Overfit score {overfit.overfit_score:.2f}"
            )

            if stability > 0.7:
                recs.append("Performance is stable across time periods.")

        return recs

    def _empty_window_metrics(
        self,
        window_id: int,
        start: datetime,
        end: datetime,
        is_training: bool
    ) -> WindowMetrics:
        """Return empty metrics for window with no trades."""
        return WindowMetrics(
            window_id=window_id,
            start_date=start,
            end_date=end,
            is_training=is_training,
            num_trades=0,
            win_rate=0,
            total_pnl_percent=0,
            avg_pnl_percent=0,
            sharpe_ratio=0,
            max_drawdown=0,
            avg_win_pnl=0,
            avg_loss_pnl=0,
            profit_factor=0,
            win_streak_max=0,
            loss_streak_max=0
        )

    def _insufficient_data_result(self, engine: str, num_trades: int) -> WalkForwardResult:
        """Return result for insufficient data."""
        return WalkForwardResult(
            timestamp=datetime.now(timezone.utc),
            engine=engine,
            status=ValidationStatus.INSUFFICIENT_DATA,
            total_trades=num_trades,
            overall_sharpe=0,
            overall_win_rate=0,
            overall_pnl=0,
            num_windows=0,
            training_windows=[],
            testing_windows=[],
            overfit_metrics=OverfitMetrics(
                train_sharpe=0, test_sharpe=0, sharpe_decay=0,
                train_win_rate=0, test_win_rate=0, win_rate_decay=0,
                train_pnl=0, test_pnl=0, pnl_decay=0,
                overfit_score=0, overfit_detected=False
            ),
            sharpe_stability=0,
            win_rate_stability=0,
            recommendations=[
                f"Insufficient data: {num_trades} trades "
                f"(need {self.MIN_TRADES_FOR_VALIDATION})"
            ]
        )

    def _save_result(self, result: WalkForwardResult):
        """Save validation result to file."""
        try:
            entry = {
                "timestamp": result.timestamp.isoformat(),
                "engine": result.engine,
                "status": result.status.value,
                "total_trades": result.total_trades,
                "overall_sharpe": result.overall_sharpe,
                "overall_win_rate": result.overall_win_rate,
                "overall_pnl": result.overall_pnl,
                "num_windows": result.num_windows,
                "overfit_score": result.overfit_metrics.overfit_score,
                "overfit_detected": result.overfit_metrics.overfit_detected,
                "sharpe_stability": result.sharpe_stability,
                "recommendations": result.recommendations
            }

            with open(self.results_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.warning(f"[WalkForwardValidator] Failed to save result: {e}")

    def _log_result(self, result: WalkForwardResult):
        """Log validation result."""
        status_icon = {
            ValidationStatus.PASSED: "✅",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.INSUFFICIENT_DATA: "📊"
        }.get(result.status, "?")

        logger.info("=" * 60)
        logger.info(f"WALK-FORWARD VALIDATION - Engine {result.engine}")
        logger.info("=" * 60)
        logger.info(f"Status: {status_icon} {result.status.value.upper()}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Windows Analyzed: {result.num_windows}")
        logger.info(f"")
        logger.info(f"OUT-OF-SAMPLE PERFORMANCE:")
        logger.info(f"  Sharpe Ratio: {result.overall_sharpe:.2f}")
        logger.info(f"  Win Rate: {result.overall_win_rate:.1%}")
        logger.info(f"  Total P&L: {result.overall_pnl:.2f}%")
        logger.info(f"")
        logger.info(f"OVERFITTING ANALYSIS:")
        logger.info(f"  Overfit Score: {result.overfit_metrics.overfit_score:.2f}")
        logger.info(f"  Sharpe Decay: {result.overfit_metrics.sharpe_decay:.1%}")
        logger.info(f"  WR Decay: {result.overfit_metrics.win_rate_decay:.1%}")
        logger.info(f"")
        logger.info(f"RECOMMENDATIONS:")
        for rec in result.recommendations:
            logger.info(f"  • {rec}")
        logger.info("=" * 60)

    def validate_engine(
        self,
        engine: str,
        get_portfolio_fn
    ) -> WalkForwardResult:
        """
        Validate an engine using its portfolio data.

        Args:
            engine: Engine name (A, B, C, D)
            get_portfolio_fn: Function to get portfolio for engine

        Returns:
            WalkForwardResult
        """
        try:
            portfolio = get_portfolio_fn(engine)
            closed_trades = portfolio.get_closed_trades()

            # Convert to TradeRecords
            trades = []
            for trade in closed_trades:
                trades.append(TradeRecord(
                    timestamp=trade.get("open_time", datetime.now(timezone.utc)),
                    asset=trade.get("asset", "UNKNOWN"),
                    direction=trade.get("direction", "LONG"),
                    entry_price=trade.get("entry_price", 0),
                    exit_price=trade.get("exit_price", 0),
                    pnl_percent=trade.get("pnl_percent", 0),
                    outcome=trade.get("outcome", "loss"),
                    holding_period_hours=trade.get("holding_hours", 0)
                ))

            return self.validate(trades, engine)

        except Exception as e:
            logger.error(f"[WalkForwardValidator] Failed to validate {engine}: {e}")
            return self._insufficient_data_result(engine, 0)

    def get_summary(self) -> Dict[str, Any]:
        """Get validator summary."""
        return {
            "validations_run": len(self.validation_history),
            "last_validation": (
                self.validation_history[-1].timestamp.isoformat()
                if self.validation_history else None
            ),
            "config": {
                "min_trades": self.MIN_TRADES_FOR_VALIDATION,
                "train_window_days": self.DEFAULT_TRAIN_WINDOW_DAYS,
                "test_window_days": self.DEFAULT_TEST_WINDOW_DAYS,
                "sharpe_threshold": self.SHARPE_THRESHOLD,
                "overfit_threshold": self.OVERFIT_THRESHOLD
            }
        }


# ==================== SINGLETON PATTERN ====================

_walk_forward_validator: Optional[WalkForwardValidator] = None

def get_walk_forward_validator() -> WalkForwardValidator:
    """Get singleton instance of WalkForwardValidator."""
    global _walk_forward_validator
    if _walk_forward_validator is None:
        _walk_forward_validator = WalkForwardValidator()
    return _walk_forward_validator
