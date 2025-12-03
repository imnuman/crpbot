"""
HYDRA 3.0 - Weight Adjustment System

Manages dynamic weight allocation for tournament engines.

Weight Strategies:
1. RANK_BASED: Fixed weights by rank (40/30/20/10)
2. SHARPE_BASED: Proportional to risk-adjusted returns
3. WIN_RATE_BASED: Proportional to win rates
4. HYBRID: Combination of all metrics

Features:
- Smooth transitions (exponential moving average)
- Minimum weight floor (5% - no engine goes to zero)
- Maximum weight cap (50% - prevent over-concentration)
- Weight history tracking
- Momentum bonus for consistent performers

Phase 2, Week 2 - Step 16
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
from enum import Enum
import json
import math


class WeightStrategy(Enum):
    """Available weight calculation strategies."""
    RANK_BASED = "rank_based"       # Fixed weights by rank
    SHARPE_BASED = "sharpe_based"   # Proportional to Sharpe ratio
    WIN_RATE_BASED = "win_rate"     # Proportional to win rate
    HYBRID = "hybrid"                # Combination of metrics


@dataclass
class WeightSnapshot:
    """Snapshot of weights at a point in time."""
    timestamp: datetime
    strategy_used: str
    weights: Dict[str, float]
    rankings: Dict[str, int]
    metrics: Dict[str, Dict[str, float]]  # Engine -> {sharpe, win_rate, pnl}
    adjustment_reason: str


@dataclass
class EngineWeight:
    """Complete weight info for an engine."""
    engine: str
    current_weight: float
    previous_weight: float
    target_weight: float
    rank: int
    sharpe_ratio: float
    win_rate: float
    momentum_score: float  # Recent performance trend
    weight_change: float   # current - previous


@dataclass
class WeightAdjustmentResult:
    """Result of a weight adjustment cycle."""
    timestamp: datetime
    strategy: WeightStrategy
    engine_weights: Dict[str, EngineWeight]
    total_weight: float
    adjustment_made: bool
    reason: str


class WeightAdjuster:
    """
    Weight Adjustment System for HYDRA 3.0.

    Manages tournament weight allocation with multiple strategies
    and smooth transitions.
    """

    # Configuration
    ADJUSTMENT_INTERVAL_HOURS = 24  # Adjust weights every 24 hours
    MIN_TRADES_FOR_ADJUSTMENT = 10  # Need minimum trades before adjusting

    # Weight bounds
    MIN_WEIGHT = 0.05   # 5% minimum (no engine goes to zero)
    MAX_WEIGHT = 0.50   # 50% maximum (prevent over-concentration)
    DEFAULT_WEIGHT = 0.25  # Equal distribution

    # Smoothing factor (0.3 = 30% new, 70% old)
    SMOOTHING_FACTOR = 0.3

    # Rank-based weights
    RANK_WEIGHTS = {
        1: 0.40,  # 40% for rank #1
        2: 0.30,  # 30% for rank #2
        3: 0.20,  # 20% for rank #3
        4: 0.10   # 10% for rank #4
    }

    # Momentum window (recent cycles to consider)
    MOMENTUM_WINDOW = 5

    def __init__(self, data_dir: Optional[Path] = None):
        # Auto-detect data directory based on environment
        if data_dir is None:
            from ..config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Current weights
        self.current_weights: Dict[str, float] = {
            "A": self.DEFAULT_WEIGHT,
            "B": self.DEFAULT_WEIGHT,
            "C": self.DEFAULT_WEIGHT,
            "D": self.DEFAULT_WEIGHT
        }

        # Weight history
        self.weight_history: List[WeightSnapshot] = []
        self.adjustment_count = 0
        self.last_adjustment_time: Optional[datetime] = None

        # Recent performance for momentum calculation
        self.performance_history: Dict[str, List[float]] = {
            "A": [], "B": [], "C": [], "D": []
        }

        # Strategy to use
        self.strategy = WeightStrategy.HYBRID

        # Persistence
        self.state_file = self.data_dir / "weight_adjuster_state.json"
        self.history_file = self.data_dir / "weight_history.jsonl"

        # Load existing state
        self._load_state()

        logger.info(f"[WeightAdjuster] Initialized. Strategy: {self.strategy.value}")

    def adjust_weights(
        self,
        rankings: List[tuple],
        force: bool = False
    ) -> Optional[WeightAdjustmentResult]:
        """
        Adjust engine weights based on current performance.

        Args:
            rankings: List of (engine_name, stats) sorted by rank
            force: Force adjustment even if interval hasn't passed

        Returns:
            WeightAdjustmentResult if adjustment made, None otherwise
        """
        now = datetime.now(timezone.utc)

        # Check if adjustment is due
        if not force and not self._should_adjust():
            return None

        # Check minimum trades requirement
        total_trades = sum(stats.total_trades for _, stats in rankings)
        if total_trades < self.MIN_TRADES_FOR_ADJUSTMENT:
            logger.info(
                f"[WeightAdjuster] Skipping - insufficient trades "
                f"({total_trades} < {self.MIN_TRADES_FOR_ADJUSTMENT})"
            )
            return None

        # Calculate target weights based on strategy
        target_weights = self._calculate_target_weights(rankings)

        # Apply smoothing for gradual transitions
        new_weights = self._apply_smoothing(target_weights)

        # Enforce bounds
        new_weights = self._enforce_bounds(new_weights)

        # Calculate engine weight details
        engine_weights = {}
        for rank, (name, stats) in enumerate(rankings, 1):
            engine_weights[name] = EngineWeight(
                engine=name,
                current_weight=new_weights[name],
                previous_weight=self.current_weights[name],
                target_weight=target_weights[name],
                rank=rank,
                sharpe_ratio=getattr(stats, 'sharpe_ratio', 0.0) or 0.0,
                win_rate=stats.win_rate,
                momentum_score=self._calculate_momentum(name),
                weight_change=new_weights[name] - self.current_weights[name]
            )

        # Update current weights
        old_weights = self.current_weights.copy()
        self.current_weights = new_weights

        # Update performance history for momentum
        self._update_performance_history(rankings)

        # Create result
        result = WeightAdjustmentResult(
            timestamp=now,
            strategy=self.strategy,
            engine_weights=engine_weights,
            total_weight=sum(new_weights.values()),
            adjustment_made=True,
            reason=f"24-hour adjustment using {self.strategy.value} strategy"
        )

        # Log changes
        self._log_weight_changes(old_weights, new_weights, rankings)

        # Record in history
        self._record_adjustment(rankings, result)

        # Update state
        self.adjustment_count += 1
        self.last_adjustment_time = now
        self._save_state()

        return result

    def _calculate_target_weights(self, rankings: List[tuple]) -> Dict[str, float]:
        """Calculate target weights based on current strategy."""
        if self.strategy == WeightStrategy.RANK_BASED:
            return self._calc_rank_based_weights(rankings)
        elif self.strategy == WeightStrategy.SHARPE_BASED:
            return self._calc_sharpe_based_weights(rankings)
        elif self.strategy == WeightStrategy.WIN_RATE_BASED:
            return self._calc_win_rate_based_weights(rankings)
        else:  # HYBRID
            return self._calc_hybrid_weights(rankings)

    def _calc_rank_based_weights(self, rankings: List[tuple]) -> Dict[str, float]:
        """Calculate weights based on rank (40/30/20/10)."""
        weights = {}
        for rank, (name, _) in enumerate(rankings, 1):
            weights[name] = self.RANK_WEIGHTS.get(rank, self.DEFAULT_WEIGHT)
        return weights

    def _calc_sharpe_based_weights(self, rankings: List[tuple]) -> Dict[str, float]:
        """Calculate weights proportional to Sharpe ratio."""
        sharpes = {}
        for name, stats in rankings:
            sharpe = getattr(stats, 'sharpe_ratio', 0.0) or 0.0
            # Shift to positive (add 2 to handle negative Sharpes)
            sharpes[name] = max(sharpe + 2, 0.1)

        total_sharpe = sum(sharpes.values())
        weights = {name: s / total_sharpe for name, s in sharpes.items()}
        return weights

    def _calc_win_rate_based_weights(self, rankings: List[tuple]) -> Dict[str, float]:
        """Calculate weights proportional to win rates."""
        win_rates = {}
        for name, stats in rankings:
            wr = stats.win_rate
            # Minimum floor to avoid zero weights
            win_rates[name] = max(wr, 0.1)

        total_wr = sum(win_rates.values())
        weights = {name: wr / total_wr for name, wr in win_rates.items()}
        return weights

    def _calc_hybrid_weights(self, rankings: List[tuple]) -> Dict[str, float]:
        """
        Calculate hybrid weights combining multiple metrics.

        Formula:
        - 40% rank-based (proven tournament structure)
        - 30% Sharpe-based (risk-adjusted returns)
        - 20% win-rate-based (consistency)
        - 10% momentum bonus (recent performance)
        """
        rank_weights = self._calc_rank_based_weights(rankings)
        sharpe_weights = self._calc_sharpe_based_weights(rankings)
        wr_weights = self._calc_win_rate_based_weights(rankings)
        momentum_weights = self._calc_momentum_weights()

        hybrid = {}
        for name in ["A", "B", "C", "D"]:
            hybrid[name] = (
                0.40 * rank_weights.get(name, 0.25) +
                0.30 * sharpe_weights.get(name, 0.25) +
                0.20 * wr_weights.get(name, 0.25) +
                0.10 * momentum_weights.get(name, 0.25)
            )

        # Normalize to sum to 1.0
        total = sum(hybrid.values())
        return {name: w / total for name, w in hybrid.items()}

    def _calc_momentum_weights(self) -> Dict[str, float]:
        """Calculate weights based on recent performance momentum."""
        momentums = {}
        for name in ["A", "B", "C", "D"]:
            momentum = self._calculate_momentum(name)
            # Shift to positive
            momentums[name] = max(momentum + 1, 0.1)

        total = sum(momentums.values())
        return {name: m / total for name, m in momentums.items()}

    def _calculate_momentum(self, engine: str) -> float:
        """
        Calculate momentum score for an engine.

        Momentum = (recent performance - older performance) / older performance
        Positive = improving, Negative = declining
        """
        history = self.performance_history.get(engine, [])

        if len(history) < 2:
            return 0.0

        # Use last few entries for "recent" vs earlier entries
        recent = history[-2:] if len(history) >= 2 else history
        older = history[:-2] if len(history) > 2 else [history[0]]

        recent_avg = sum(recent) / len(recent) if recent else 0
        older_avg = sum(older) / len(older) if older else 0

        if older_avg == 0:
            return 0.0

        return (recent_avg - older_avg) / max(abs(older_avg), 0.01)

    def _apply_smoothing(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply exponential smoothing for gradual transitions.

        new_weight = α * target + (1-α) * current
        where α = SMOOTHING_FACTOR
        """
        smoothed = {}
        alpha = self.SMOOTHING_FACTOR

        for name in ["A", "B", "C", "D"]:
            target = target_weights.get(name, self.DEFAULT_WEIGHT)
            current = self.current_weights.get(name, self.DEFAULT_WEIGHT)
            smoothed[name] = alpha * target + (1 - alpha) * current

        return smoothed

    def _enforce_bounds(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Enforce minimum and maximum weight bounds.

        Ensures:
        - No weight below MIN_WEIGHT (5%)
        - No weight above MAX_WEIGHT (50%)
        - Total sums to 1.0
        """
        bounded = {}

        # First pass: apply bounds
        for name, weight in weights.items():
            bounded[name] = max(min(weight, self.MAX_WEIGHT), self.MIN_WEIGHT)

        # Normalize to sum to 1.0
        total = sum(bounded.values())
        return {name: w / total for name, w in bounded.items()}

    def _update_performance_history(self, rankings: List[tuple]):
        """Update performance history for momentum calculation."""
        for name, stats in rankings:
            # Use P&L percent as performance metric
            perf = stats.total_pnl_percent
            self.performance_history[name].append(perf)

            # Keep only recent window
            if len(self.performance_history[name]) > self.MOMENTUM_WINDOW:
                self.performance_history[name] = \
                    self.performance_history[name][-self.MOMENTUM_WINDOW:]

    def _should_adjust(self) -> bool:
        """Check if weight adjustment is due."""
        if self.last_adjustment_time is None:
            return True

        time_since = datetime.now(timezone.utc) - self.last_adjustment_time
        return time_since >= timedelta(hours=self.ADJUSTMENT_INTERVAL_HOURS)

    def _log_weight_changes(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        rankings: List[tuple]
    ):
        """Log weight changes for visibility."""
        logger.info("=" * 60)
        logger.info(f"WEIGHT ADJUSTMENT #{self.adjustment_count + 1}")
        logger.info(f"Strategy: {self.strategy.value}")
        logger.info("=" * 60)

        for rank, (name, stats) in enumerate(rankings, 1):
            old_w = old_weights.get(name, 0.25) * 100
            new_w = new_weights.get(name, 0.25) * 100
            change = new_w - old_w
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"

            logger.info(
                f"  #{rank} Engine {name}: {old_w:.1f}% → {new_w:.1f}% ({arrow}{abs(change):.1f}%) | "
                f"WR: {stats.win_rate:.1%} | Sharpe: {getattr(stats, 'sharpe_ratio', 0) or 0:.2f}"
            )

        logger.info("=" * 60)

    def _record_adjustment(
        self,
        rankings: List[tuple],
        result: WeightAdjustmentResult
    ):
        """Record adjustment in history."""
        snapshot = WeightSnapshot(
            timestamp=result.timestamp,
            strategy_used=self.strategy.value,
            weights={name: ew.current_weight for name, ew in result.engine_weights.items()},
            rankings={name: ew.rank for name, ew in result.engine_weights.items()},
            metrics={
                name: {
                    "sharpe": ew.sharpe_ratio,
                    "win_rate": ew.win_rate,
                    "momentum": ew.momentum_score
                }
                for name, ew in result.engine_weights.items()
            },
            adjustment_reason=result.reason
        )

        self.weight_history.append(snapshot)
        self._save_history_entry(snapshot)

    def get_current_weights(self) -> Dict[str, float]:
        """Get current engine weights."""
        return self.current_weights.copy()

    def get_weight_for_engine(self, engine: str) -> float:
        """Get current weight for specific engine."""
        return self.current_weights.get(engine, self.DEFAULT_WEIGHT)

    def set_strategy(self, strategy: WeightStrategy):
        """Set weight calculation strategy."""
        self.strategy = strategy
        logger.info(f"[WeightAdjuster] Strategy changed to: {strategy.value}")

    def get_weight_summary(self) -> Dict[str, Any]:
        """Get summary of weight adjuster state."""
        return {
            "strategy": self.strategy.value,
            "current_weights": self.current_weights,
            "adjustment_count": self.adjustment_count,
            "last_adjustment": self.last_adjustment_time.isoformat() if self.last_adjustment_time else None,
            "history_entries": len(self.weight_history)
        }

    def format_weights_display(self) -> str:
        """Format weights for dashboard display."""
        lines = ["ENGINE WEIGHTS", "=" * 30]

        # Sort by weight (descending)
        sorted_weights = sorted(
            self.current_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for engine, weight in sorted_weights:
            bar_len = int(weight * 40)  # 40 chars for 100%
            bar = "█" * bar_len + "░" * (10 - bar_len)
            lines.append(f"Engine {engine}: {bar} {weight:.1%}")

        lines.append("=" * 30)
        lines.append(f"Strategy: {self.strategy.value}")

        return "\n".join(lines)

    def _save_state(self):
        """Save current state to disk."""
        try:
            state = {
                "current_weights": self.current_weights,
                "adjustment_count": self.adjustment_count,
                "last_adjustment": self.last_adjustment_time.isoformat() if self.last_adjustment_time else None,
                "strategy": self.strategy.value,
                "performance_history": self.performance_history
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.warning(f"[WeightAdjuster] Failed to save state: {e}")

    def _load_state(self):
        """Load state from disk if exists."""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    state = json.load(f)

                self.current_weights = state.get("current_weights", self.current_weights)
                self.adjustment_count = state.get("adjustment_count", 0)

                if state.get("last_adjustment"):
                    self.last_adjustment_time = datetime.fromisoformat(state["last_adjustment"])

                strategy_str = state.get("strategy", "hybrid")
                self.strategy = WeightStrategy(strategy_str)

                self.performance_history = state.get("performance_history", self.performance_history)

                logger.info(
                    f"[WeightAdjuster] Loaded state: {self.adjustment_count} adjustments, "
                    f"strategy={self.strategy.value}"
                )

        except Exception as e:
            logger.warning(f"[WeightAdjuster] Failed to load state: {e}")

    def _save_history_entry(self, snapshot: WeightSnapshot):
        """Append snapshot to history file."""
        try:
            entry = {
                "timestamp": snapshot.timestamp.isoformat(),
                "strategy": snapshot.strategy_used,
                "weights": snapshot.weights,
                "rankings": snapshot.rankings,
                "metrics": snapshot.metrics,
                "reason": snapshot.adjustment_reason
            }

            with open(self.history_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.warning(f"[WeightAdjuster] Failed to save history: {e}")


# ==================== SINGLETON PATTERN ====================

_weight_adjuster: Optional[WeightAdjuster] = None

def get_weight_adjuster() -> WeightAdjuster:
    """Get singleton instance of WeightAdjuster."""
    global _weight_adjuster
    if _weight_adjuster is None:
        _weight_adjuster = WeightAdjuster()
    return _weight_adjuster
