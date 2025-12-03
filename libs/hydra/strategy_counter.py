"""
HYDRA 3.0 - Strategy Counter

Tracks and categorizes trading strategies used by engines.

Functions:
1. Categorize trades by strategy type
2. Track strategy usage per engine
3. Detect strategy convergence (engines using same strategies)
4. Measure strategy diversity for tournament health
5. Track strategy performance by type

Strategy Types:
- TREND_FOLLOW: Trading with the trend
- MEAN_REVERT: Fading moves, expecting reversion
- MOMENTUM: Chasing strong directional moves
- BREAKOUT: Trading range breakouts
- COUNTER_TREND: Fading exhausted trends
- SCALP: Quick in-and-out trades
- SWING: Multi-day position holds

Phase 2, Week 2 - Step 19
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
from enum import Enum
from collections import defaultdict
import json
import math


class StrategyType(Enum):
    """Trading strategy categories."""
    TREND_FOLLOW = "trend_follow"       # Trading with the trend
    MEAN_REVERT = "mean_revert"         # Fading moves
    MOMENTUM = "momentum"               # Chasing strong moves
    BREAKOUT = "breakout"               # Range breakouts
    COUNTER_TREND = "counter_trend"     # Fading exhausted trends
    SCALP = "scalp"                     # Quick trades
    SWING = "swing"                     # Multi-day holds
    UNKNOWN = "unknown"                 # Unclassified


@dataclass
class StrategyRecord:
    """Record of a single strategy usage."""
    timestamp: datetime
    engine: str
    strategy_type: StrategyType
    asset: str
    direction: str
    confidence: float
    regime: str
    reasoning: str  # LLM's reasoning for the trade
    outcome: Optional[str] = None  # win/loss after trade closes
    pnl_percent: Optional[float] = None


@dataclass
class StrategyStats:
    """Statistics for a strategy type."""
    strategy_type: StrategyType
    total_uses: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    engines_using: List[str]
    last_used: Optional[datetime]


@dataclass
class DiversityMetrics:
    """Strategy diversity metrics for tournament health."""
    timestamp: datetime
    unique_strategies: int
    total_strategies: int
    diversity_score: float  # 0-1, higher = more diverse
    dominant_strategy: Optional[StrategyType]
    dominant_percentage: float
    convergence_detected: bool  # True if engines converging on same strategy
    engine_strategies: Dict[str, StrategyType]  # Current strategy per engine


class StrategyCounter:
    """
    Strategy Counter for HYDRA 3.0.

    Tracks, categorizes, and analyzes trading strategies
    across all tournament engines.
    """

    # Keywords for strategy classification
    STRATEGY_KEYWORDS = {
        StrategyType.TREND_FOLLOW: [
            "trend", "following", "continuation", "with the move",
            "uptrend", "downtrend", "trending", "direction"
        ],
        StrategyType.MEAN_REVERT: [
            "reversion", "mean", "oversold", "overbought", "fade",
            "pullback", "correction", "reverting", "stretched"
        ],
        StrategyType.MOMENTUM: [
            "momentum", "strong move", "acceleration", "velocity",
            "explosive", "surge", "pushing", "force"
        ],
        StrategyType.BREAKOUT: [
            "breakout", "break out", "range", "resistance", "support",
            "level", "barrier", "ceiling", "floor"
        ],
        StrategyType.COUNTER_TREND: [
            "counter", "exhaustion", "reversal", "top", "bottom",
            "fading", "against", "contrary", "overextended"
        ],
        StrategyType.SCALP: [
            "scalp", "quick", "fast", "short-term", "intraday",
            "rapid", "brief", "immediate"
        ],
        StrategyType.SWING: [
            "swing", "multi-day", "position", "hold", "longer",
            "extended", "patience", "days"
        ]
    }

    # Convergence threshold (if >X% of engines use same strategy)
    CONVERGENCE_THRESHOLD = 0.75  # 75% = 3 out of 4 engines

    def __init__(self, data_dir: Optional[Path] = None):
        # Auto-detect data directory based on environment
        if data_dir is None:
            import os
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Strategy records
        self.records: List[StrategyRecord] = []

        # Per-engine strategy tracking
        self.engine_strategies: Dict[str, List[StrategyRecord]] = defaultdict(list)

        # Per-strategy stats
        self.strategy_stats: Dict[StrategyType, StrategyStats] = {}

        # Current strategy per engine (most recent)
        self.current_strategies: Dict[str, StrategyType] = {}

        # Diversity history
        self.diversity_history: List[DiversityMetrics] = []

        # Persistence
        self.records_file = self.data_dir / "strategy_records.jsonl"
        self.stats_file = self.data_dir / "strategy_stats.json"

        # Load existing data
        self._load_records()

        logger.info(f"[StrategyCounter] Initialized with {len(self.records)} records")

    def record_strategy(
        self,
        engine: str,
        asset: str,
        direction: str,
        confidence: float,
        regime: str,
        reasoning: str,
        strategy_hint: Optional[str] = None
    ) -> StrategyRecord:
        """
        Record a strategy usage from an engine's trade decision.

        Args:
            engine: Engine name (A, B, C, D)
            asset: Trading asset
            direction: LONG or SHORT
            confidence: Trade confidence
            regime: Market regime
            reasoning: LLM's reasoning for the trade
            strategy_hint: Optional explicit strategy type

        Returns:
            StrategyRecord with classified strategy
        """
        # Classify strategy from reasoning
        if strategy_hint:
            try:
                strategy_type = StrategyType(strategy_hint.lower())
            except ValueError:
                strategy_type = self._classify_strategy(reasoning)
        else:
            strategy_type = self._classify_strategy(reasoning)

        # Create record
        record = StrategyRecord(
            timestamp=datetime.now(timezone.utc),
            engine=engine,
            strategy_type=strategy_type,
            asset=asset,
            direction=direction,
            confidence=confidence,
            regime=regime,
            reasoning=reasoning[:500]  # Truncate long reasoning
        )

        # Store record
        self.records.append(record)
        self.engine_strategies[engine].append(record)
        self.current_strategies[engine] = strategy_type

        # Save to disk
        self._save_record(record)

        # Update stats
        self._update_stats()

        logger.debug(
            f"[StrategyCounter] Engine {engine}: {strategy_type.value} "
            f"({direction} {asset})"
        )

        return record

    def update_outcome(
        self,
        engine: str,
        asset: str,
        outcome: str,
        pnl_percent: float
    ):
        """
        Update the outcome of a strategy record after trade closes.

        Args:
            engine: Engine name
            asset: Trading asset
            outcome: "win" or "loss"
            pnl_percent: P&L percentage
        """
        # Find most recent matching record
        for record in reversed(self.engine_strategies[engine]):
            if record.asset == asset and record.outcome is None:
                record.outcome = outcome
                record.pnl_percent = pnl_percent
                self._update_stats()
                break

    def _classify_strategy(self, reasoning: str) -> StrategyType:
        """
        Classify strategy type from trade reasoning text.

        Uses keyword matching with scoring.
        """
        if not reasoning:
            return StrategyType.UNKNOWN

        reasoning_lower = reasoning.lower()
        scores = defaultdict(int)

        # Score each strategy type based on keyword matches
        for strategy_type, keywords in self.STRATEGY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in reasoning_lower:
                    scores[strategy_type] += 1

        if not scores:
            return StrategyType.UNKNOWN

        # Return strategy with highest score
        best_strategy = max(scores.items(), key=lambda x: x[1])
        return best_strategy[0]

    def _update_stats(self):
        """Update strategy statistics from all records."""
        stats = {}

        for strategy_type in StrategyType:
            # Get records for this strategy
            type_records = [r for r in self.records if r.strategy_type == strategy_type]

            if not type_records:
                continue

            # Calculate stats
            completed = [r for r in type_records if r.outcome is not None]
            wins = sum(1 for r in completed if r.outcome == "win")
            losses = sum(1 for r in completed if r.outcome == "loss")
            total_pnl = sum(r.pnl_percent or 0 for r in completed)

            # Engines using this strategy
            engines = list(set(r.engine for r in type_records))

            stats[strategy_type] = StrategyStats(
                strategy_type=strategy_type,
                total_uses=len(type_records),
                wins=wins,
                losses=losses,
                win_rate=wins / len(completed) if completed else 0,
                total_pnl=total_pnl,
                avg_pnl=total_pnl / len(completed) if completed else 0,
                engines_using=engines,
                last_used=type_records[-1].timestamp if type_records else None
            )

        self.strategy_stats = stats

    def get_strategy_stats(self, strategy_type: StrategyType) -> Optional[StrategyStats]:
        """Get statistics for a specific strategy type."""
        return self.strategy_stats.get(strategy_type)

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all strategy types."""
        return {
            st.value: {
                "total_uses": stats.total_uses,
                "wins": stats.wins,
                "losses": stats.losses,
                "win_rate": stats.win_rate,
                "total_pnl": stats.total_pnl,
                "avg_pnl": stats.avg_pnl,
                "engines_using": stats.engines_using
            }
            for st, stats in self.strategy_stats.items()
        }

    def get_engine_strategy_breakdown(self, engine: str) -> Dict[str, int]:
        """Get strategy type breakdown for an engine."""
        records = self.engine_strategies.get(engine, [])
        breakdown = defaultdict(int)

        for record in records:
            breakdown[record.strategy_type.value] += 1

        return dict(breakdown)

    def calculate_diversity(self) -> DiversityMetrics:
        """
        Calculate strategy diversity metrics for the tournament.

        High diversity = healthy tournament (engines exploring different strategies)
        Low diversity = unhealthy (engines converging on same approach)
        """
        now = datetime.now(timezone.utc)

        # Get current strategies for each engine
        engine_strats = {}
        for engine in ["A", "B", "C", "D"]:
            if engine in self.current_strategies:
                engine_strats[engine] = self.current_strategies[engine]

        if not engine_strats:
            return DiversityMetrics(
                timestamp=now,
                unique_strategies=0,
                total_strategies=0,
                diversity_score=0,
                dominant_strategy=None,
                dominant_percentage=0,
                convergence_detected=False,
                engine_strategies={}
            )

        # Count unique strategies
        strategy_counts = defaultdict(int)
        for strat in engine_strats.values():
            strategy_counts[strat] += 1

        unique_strategies = len(strategy_counts)
        total_strategies = len(engine_strats)

        # Find dominant strategy
        dominant = max(strategy_counts.items(), key=lambda x: x[1])
        dominant_strategy = dominant[0]
        dominant_count = dominant[1]
        dominant_percentage = dominant_count / total_strategies

        # Check for convergence
        convergence_detected = dominant_percentage >= self.CONVERGENCE_THRESHOLD

        # Calculate diversity score (Shannon entropy normalized)
        # Higher = more diverse
        if total_strategies > 1:
            entropy = 0
            for count in strategy_counts.values():
                p = count / total_strategies
                if p > 0:
                    entropy -= p * math.log2(p)

            # Normalize by max possible entropy
            max_entropy = math.log2(total_strategies)
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversity_score = 0

        metrics = DiversityMetrics(
            timestamp=now,
            unique_strategies=unique_strategies,
            total_strategies=total_strategies,
            diversity_score=diversity_score,
            dominant_strategy=dominant_strategy,
            dominant_percentage=dominant_percentage,
            convergence_detected=convergence_detected,
            engine_strategies={e: s.value for e, s in engine_strats.items()}
        )

        self.diversity_history.append(metrics)

        return metrics

    def get_convergence_alert(self) -> Optional[str]:
        """
        Check for strategy convergence and return alert message if detected.

        Returns:
            Alert message if convergence detected, None otherwise
        """
        metrics = self.calculate_diversity()

        if metrics.convergence_detected:
            engines = [
                e for e, s in self.current_strategies.items()
                if s == metrics.dominant_strategy
            ]
            return (
                f"CONVERGENCE ALERT: {len(engines)}/4 engines using "
                f"{metrics.dominant_strategy.value} strategy. "
                f"Diversity score: {metrics.diversity_score:.2f}. "
                f"Consider encouraging strategy exploration."
            )

        return None

    def get_best_strategy(self) -> Tuple[StrategyType, float]:
        """
        Get the best performing strategy by win rate.

        Returns:
            Tuple of (strategy_type, win_rate)
        """
        best = None
        best_wr = 0

        for st, stats in self.strategy_stats.items():
            if stats.wins + stats.losses >= 5:  # Min sample size
                if stats.win_rate > best_wr:
                    best = st
                    best_wr = stats.win_rate

        return (best or StrategyType.UNKNOWN, best_wr)

    def get_worst_strategy(self) -> Tuple[StrategyType, float]:
        """
        Get the worst performing strategy by win rate.

        Returns:
            Tuple of (strategy_type, win_rate)
        """
        worst = None
        worst_wr = 1.0

        for st, stats in self.strategy_stats.items():
            if stats.wins + stats.losses >= 5:  # Min sample size
                if stats.win_rate < worst_wr:
                    worst = st
                    worst_wr = stats.win_rate

        return (worst or StrategyType.UNKNOWN, worst_wr)

    def format_strategy_report(self) -> str:
        """Generate formatted strategy report for dashboard."""
        lines = ["STRATEGY USAGE REPORT", "=" * 50]

        # Per-strategy stats
        for st, stats in sorted(
            self.strategy_stats.items(),
            key=lambda x: x[1].total_uses,
            reverse=True
        ):
            if stats.total_uses == 0:
                continue

            wr_str = f"{stats.win_rate:.1%}" if stats.wins + stats.losses > 0 else "N/A"
            engines_str = ", ".join(stats.engines_using) if stats.engines_using else "None"

            lines.append(
                f"\n{st.value.upper()}:"
            )
            lines.append(f"  Uses: {stats.total_uses} | WR: {wr_str} | P&L: {stats.total_pnl:+.2f}%")
            lines.append(f"  Engines: {engines_str}")

        # Diversity metrics
        metrics = self.calculate_diversity()
        lines.append("")
        lines.append("=" * 50)
        lines.append("DIVERSITY METRICS:")
        lines.append(f"  Unique Strategies: {metrics.unique_strategies}")
        lines.append(f"  Diversity Score: {metrics.diversity_score:.2f}")
        lines.append(f"  Dominant: {metrics.dominant_strategy.value if metrics.dominant_strategy else 'None'} ({metrics.dominant_percentage:.0%})")

        if metrics.convergence_detected:
            lines.append("  ⚠️ CONVERGENCE DETECTED")

        lines.append("=" * 50)

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get strategy counter summary."""
        metrics = self.calculate_diversity()

        return {
            "total_records": len(self.records),
            "unique_strategies": metrics.unique_strategies,
            "diversity_score": metrics.diversity_score,
            "convergence_detected": metrics.convergence_detected,
            "current_strategies": {
                e: s.value for e, s in self.current_strategies.items()
            },
            "best_strategy": self.get_best_strategy()[0].value,
            "worst_strategy": self.get_worst_strategy()[0].value
        }

    def _save_record(self, record: StrategyRecord):
        """Save record to disk."""
        try:
            entry = {
                "timestamp": record.timestamp.isoformat(),
                "engine": record.engine,
                "strategy_type": record.strategy_type.value,
                "asset": record.asset,
                "direction": record.direction,
                "confidence": record.confidence,
                "regime": record.regime,
                "reasoning": record.reasoning[:200],  # Truncate for storage
                "outcome": record.outcome,
                "pnl_percent": record.pnl_percent
            }

            with open(self.records_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.warning(f"[StrategyCounter] Failed to save record: {e}")

    def _load_records(self):
        """Load records from disk."""
        try:
            if self.records_file.exists():
                with open(self.records_file) as f:
                    for line in f:
                        data = json.loads(line.strip())
                        record = StrategyRecord(
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            engine=data["engine"],
                            strategy_type=StrategyType(data["strategy_type"]),
                            asset=data["asset"],
                            direction=data["direction"],
                            confidence=data["confidence"],
                            regime=data["regime"],
                            reasoning=data["reasoning"],
                            outcome=data.get("outcome"),
                            pnl_percent=data.get("pnl_percent")
                        )
                        self.records.append(record)
                        self.engine_strategies[record.engine].append(record)

                        # Track current strategy
                        self.current_strategies[record.engine] = record.strategy_type

                # Update stats
                self._update_stats()

        except Exception as e:
            logger.warning(f"[StrategyCounter] Failed to load records: {e}")


# ==================== SINGLETON PATTERN ====================

_strategy_counter: Optional[StrategyCounter] = None

def get_strategy_counter() -> StrategyCounter:
    """Get singleton instance of StrategyCounter."""
    global _strategy_counter
    if _strategy_counter is None:
        _strategy_counter = StrategyCounter()
    return _strategy_counter
