"""
HYDRA 3.0 - Daily Improvement Tracker (MOD 11)

Tracks daily improvements and learning for each engine:
- Win rate changes
- P&L improvements
- Specialty trigger accuracy
- Learning from knowledge transfer

Daily summaries help identify which engines are improving vs stagnating.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Daily statistics for an engine."""
    date: str  # YYYY-MM-DD
    engine_id: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    pnl_pct: float
    specialty_triggers: int  # How many times specialty trigger fired
    specialty_trades: int  # How many trades from specialty
    specialty_accuracy: float  # Specialty trade win rate


@dataclass
class DailyImprovement:
    """Daily improvement metrics for an engine."""
    date: str
    engine_id: str
    win_rate_change: float  # vs yesterday
    pnl_change: float  # vs yesterday
    specialty_accuracy_change: float  # vs yesterday
    is_improving: bool  # Overall improvement signal
    momentum_score: float  # 0-1, based on trend
    notes: List[str]  # Observations


class ImprovementTracker:
    """
    Tracks daily improvements for all engines.

    Features:
    - Daily stat snapshots
    - Day-over-day comparisons
    - 7-day momentum scoring
    - Improvement/stagnation detection
    """

    _instance: Optional["ImprovementTracker"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: Optional[Path] = None):
        if self._initialized:
            return

        # Auto-detect data directory
        if data_dir is None:
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.stats_file = self.data_dir / "daily_stats.jsonl"
        self.improvements_file = self.data_dir / "daily_improvements.jsonl"

        # In-memory cache of recent stats
        self._daily_stats: Dict[str, Dict[str, DailyStats]] = {}  # date -> engine -> stats
        self._load_recent_stats()

        self._initialized = True
        logger.info("ImprovementTracker initialized")

    def _load_recent_stats(self):
        """Load last 7 days of stats from file."""
        if not self.stats_file.exists():
            return

        try:
            cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            with open(self.stats_file) as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("date", "") >= cutoff:
                        date = data["date"]
                        engine = data["engine_id"]

                        if date not in self._daily_stats:
                            self._daily_stats[date] = {}

                        self._daily_stats[date][engine] = DailyStats(
                            date=data["date"],
                            engine_id=data["engine_id"],
                            trades=data["trades"],
                            wins=data["wins"],
                            losses=data["losses"],
                            win_rate=data["win_rate"],
                            pnl_pct=data["pnl_pct"],
                            specialty_triggers=data.get("specialty_triggers", 0),
                            specialty_trades=data.get("specialty_trades", 0),
                            specialty_accuracy=data.get("specialty_accuracy", 0.0),
                        )
        except Exception as e:
            logger.warning(f"Failed to load recent stats: {e}")

    def record_daily_stats(
        self,
        engine_id: str,
        trades: int,
        wins: int,
        losses: int,
        pnl_pct: float,
        specialty_triggers: int = 0,
        specialty_trades: int = 0,
        specialty_wins: int = 0,
    ) -> DailyStats:
        """
        Record end-of-day stats for an engine.

        Args:
            engine_id: Engine ID (A, B, C, D)
            trades: Total trades today
            wins: Winning trades
            losses: Losing trades
            pnl_pct: P&L percentage
            specialty_triggers: Times specialty trigger fired
            specialty_trades: Trades from specialty signals
            specialty_wins: Wins from specialty trades
        """
        today = datetime.now().strftime("%Y-%m-%d")
        win_rate = wins / trades if trades > 0 else 0.0
        specialty_accuracy = specialty_wins / specialty_trades if specialty_trades > 0 else 0.0

        stats = DailyStats(
            date=today,
            engine_id=engine_id,
            trades=trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            pnl_pct=pnl_pct,
            specialty_triggers=specialty_triggers,
            specialty_trades=specialty_trades,
            specialty_accuracy=specialty_accuracy,
        )

        # Store in cache
        if today not in self._daily_stats:
            self._daily_stats[today] = {}
        self._daily_stats[today][engine_id] = stats

        # Persist to file
        try:
            with open(self.stats_file, 'a') as f:
                f.write(json.dumps(asdict(stats)) + '\n')
        except Exception as e:
            logger.error(f"Failed to save daily stats: {e}")

        logger.info(
            f"Daily stats recorded for Engine {engine_id}: "
            f"WR={win_rate:.1%}, P&L={pnl_pct:+.2f}%, Specialty={specialty_accuracy:.1%}"
        )

        return stats

    def calculate_improvement(self, engine_id: str) -> Optional[DailyImprovement]:
        """
        Calculate day-over-day improvement for an engine.

        Returns:
            DailyImprovement if comparison data available
        """
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Get today's and yesterday's stats
        today_stats = self._daily_stats.get(today, {}).get(engine_id)
        yesterday_stats = self._daily_stats.get(yesterday, {}).get(engine_id)

        if not today_stats:
            return None

        # Calculate changes
        if yesterday_stats:
            wr_change = today_stats.win_rate - yesterday_stats.win_rate
            pnl_change = today_stats.pnl_pct - yesterday_stats.pnl_pct
            specialty_change = today_stats.specialty_accuracy - yesterday_stats.specialty_accuracy
        else:
            # No yesterday data - use today's values as baseline
            wr_change = 0.0
            pnl_change = 0.0
            specialty_change = 0.0

        # Calculate 7-day momentum
        momentum = self._calculate_momentum(engine_id)

        # Determine if improving
        is_improving = (wr_change > 0 or pnl_change > 0) and momentum >= 0.5

        # Generate notes
        notes = []
        if wr_change > 0.05:
            notes.append(f"Win rate improved by {wr_change*100:.1f}%")
        elif wr_change < -0.05:
            notes.append(f"Win rate dropped by {abs(wr_change)*100:.1f}%")

        if pnl_change > 1.0:
            notes.append(f"P&L improved by ${pnl_change:.2f}%")
        elif pnl_change < -1.0:
            notes.append(f"P&L dropped by ${abs(pnl_change):.2f}%")

        if specialty_change > 0.1:
            notes.append("Specialty accuracy improving")
        elif specialty_change < -0.1:
            notes.append("Specialty accuracy declining")

        if momentum >= 0.7:
            notes.append("Strong positive momentum")
        elif momentum <= 0.3:
            notes.append("Warning: negative momentum")

        improvement = DailyImprovement(
            date=today,
            engine_id=engine_id,
            win_rate_change=wr_change,
            pnl_change=pnl_change,
            specialty_accuracy_change=specialty_change,
            is_improving=is_improving,
            momentum_score=momentum,
            notes=notes,
        )

        # Persist
        try:
            with open(self.improvements_file, 'a') as f:
                f.write(json.dumps(asdict(improvement)) + '\n')
        except Exception as e:
            logger.error(f"Failed to save improvement: {e}")

        return improvement

    def _calculate_momentum(self, engine_id: str) -> float:
        """
        Calculate 7-day momentum score (0-1).

        Higher = consistent improvement
        Lower = stagnation or decline
        """
        dates = sorted(self._daily_stats.keys())[-7:]  # Last 7 days

        if len(dates) < 2:
            return 0.5  # Neutral if insufficient data

        improvements = 0
        comparisons = 0

        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]

            prev_stats = self._daily_stats.get(prev_date, {}).get(engine_id)
            curr_stats = self._daily_stats.get(curr_date, {}).get(engine_id)

            if prev_stats and curr_stats:
                comparisons += 1
                if curr_stats.win_rate >= prev_stats.win_rate:
                    improvements += 0.5
                if curr_stats.pnl_pct >= prev_stats.pnl_pct:
                    improvements += 0.5

        if comparisons == 0:
            return 0.5

        return improvements / comparisons

    def get_engine_trend(self, engine_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get trend analysis for an engine.

        Returns:
            Dict with trend metrics and direction
        """
        dates = sorted(self._daily_stats.keys())[-days:]
        stats_list = []

        for date in dates:
            if engine_id in self._daily_stats.get(date, {}):
                stats_list.append(self._daily_stats[date][engine_id])

        if len(stats_list) < 2:
            return {
                "engine_id": engine_id,
                "trend": "UNKNOWN",
                "days_analyzed": len(stats_list),
                "win_rate_trend": 0.0,
                "pnl_trend": 0.0,
            }

        # Calculate trends
        first_stats = stats_list[0]
        last_stats = stats_list[-1]

        wr_trend = last_stats.win_rate - first_stats.win_rate
        pnl_trend = last_stats.pnl_pct - first_stats.pnl_pct

        # Determine overall trend
        if wr_trend > 0.05 and pnl_trend > 0:
            trend = "IMPROVING"
        elif wr_trend < -0.05 and pnl_trend < 0:
            trend = "DECLINING"
        elif abs(wr_trend) < 0.02 and abs(pnl_trend) < 0.5:
            trend = "STAGNANT"
        else:
            trend = "MIXED"

        return {
            "engine_id": engine_id,
            "trend": trend,
            "days_analyzed": len(stats_list),
            "win_rate_trend": wr_trend,
            "pnl_trend": pnl_trend,
            "first_wr": first_stats.win_rate,
            "last_wr": last_stats.win_rate,
            "first_pnl": first_stats.pnl_pct,
            "last_pnl": last_stats.pnl_pct,
        }

    def get_all_trends(self, days: int = 7) -> Dict[str, Dict]:
        """Get trends for all engines."""
        return {
            engine: self.get_engine_trend(engine, days)
            for engine in ["A", "B", "C", "D"]
        }

    def get_daily_summary(self) -> str:
        """Get human-readable daily summary."""
        today = datetime.now().strftime("%Y-%m-%d")
        lines = [f"=== Daily Improvement Summary ({today}) ==="]

        for engine in ["A", "B", "C", "D"]:
            improvement = self.calculate_improvement(engine)

            if improvement:
                status = "+" if improvement.is_improving else "-"
                lines.append(f"")
                lines.append(f"Engine {engine} [{status}] Momentum: {improvement.momentum_score:.0%}")
                lines.append(f"  WR Change: {improvement.win_rate_change:+.1%}")
                lines.append(f"  P&L Change: {improvement.pnl_change:+.2f}%")
                lines.append(f"  Specialty Change: {improvement.specialty_accuracy_change:+.1%}")

                if improvement.notes:
                    for note in improvement.notes[:2]:  # Top 2 notes
                        lines.append(f"  -> {note}")
            else:
                lines.append(f"")
                lines.append(f"Engine {engine} [?] No data for comparison")

        return "\n".join(lines)


# Singleton accessor
_tracker_instance: Optional[ImprovementTracker] = None


def get_improvement_tracker() -> ImprovementTracker:
    """Get or create the improvement tracker singleton."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ImprovementTracker()
    return _tracker_instance


# Convenience functions
def record_engine_daily_stats(
    engine_id: str,
    trades: int,
    wins: int,
    losses: int,
    pnl_pct: float,
    specialty_triggers: int = 0,
    specialty_trades: int = 0,
    specialty_wins: int = 0,
) -> DailyStats:
    """Quick record daily stats for an engine."""
    return get_improvement_tracker().record_daily_stats(
        engine_id, trades, wins, losses, pnl_pct,
        specialty_triggers, specialty_trades, specialty_wins
    )


def get_engine_improvement(engine_id: str) -> Optional[DailyImprovement]:
    """Get improvement metrics for an engine."""
    return get_improvement_tracker().calculate_improvement(engine_id)
