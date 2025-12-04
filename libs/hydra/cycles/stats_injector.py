"""
HYDRA 3.0 - Stats Injection System

Calculates and formats tournament statistics for prompt injection.

Every cycle, each engine receives:
- {rank}: Current rank (1-4)
- {wr}: Win rate percentage
- {gap}: Gap to leader in percentage points
- {leader}: Current leader name
- {leader_wr}: Leader's win rate

Format: "Rank: 2/4 | WR: 64.3% | Leader: Engine B 71.2% | Gap: 6.9%"

This creates competitive pressure by keeping engines aware of standings.

Phase 2, Week 2 - Step 15
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
import json


@dataclass
class EngineStats:
    """Complete stats for a single engine"""
    engine: str
    rank: int
    weight: float
    win_rate: float
    total_pnl_usd: float
    total_pnl_percent: float
    total_trades: int
    wins: int
    losses: int
    sharpe_ratio: Optional[float]
    gap_to_leader: float  # Percentage points below leader
    is_leader: bool


@dataclass
class TournamentStats:
    """Complete tournament statistics"""
    timestamp: datetime
    leader: str
    leader_wr: float
    leader_pnl: float
    engine_stats: Dict[str, EngineStats]
    total_trades_all: int
    avg_win_rate: float


class StatsInjector:
    """
    Stats Injection System for HYDRA 3.0.

    Provides formatted statistics for prompt injection into engine decisions.
    Each engine gets personalized stats showing their position in tournament.
    """

    # Display formats
    COMPACT_FORMAT = "Rank: {rank}/4 | WR: {wr:.1f}% | Leader: Engine {leader} {leader_wr:.1f}% | Gap: {gap:.1f}%"
    DETAILED_FORMAT = """
TOURNAMENT STANDING:
- Your Rank: #{rank}/4 ({status})
- Your Win Rate: {wr:.1f}%
- Your P&L: ${pnl:+.2f}
- Your Weight: {weight:.0f}%

LEADER:
- Engine {leader} ({leader_wr:.1f}% WR, ${leader_pnl:+.2f})
- Gap to Leader: {gap:.1f} percentage points

COMPETITION:
- Trades to beat: {leader_trades}
- You need: {needed_wr:.1f}% WR to close gap
"""

    def __init__(self, data_dir: Optional[Path] = None):
        # Auto-detect data directory based on environment
        if data_dir is None:
            from ..config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Stats history
        self.stats_history: List[TournamentStats] = []

        # Persistence
        self.state_file = self.data_dir / "stats_injection_state.json"
        self.history_file = self.data_dir / "stats_history.jsonl"

        logger.info("[StatsInjector] Initialized")

    def calculate_stats(self, rankings: List[tuple]) -> TournamentStats:
        """
        Calculate complete tournament statistics from rankings.

        Args:
            rankings: List of (engine_name, stats) sorted by rank

        Returns:
            TournamentStats with all engine stats calculated
        """
        if not rankings:
            return self._empty_stats()

        now = datetime.now(timezone.utc)

        # Identify leader
        leader_name, leader_stats = rankings[0]
        leader_wr = leader_stats.win_rate * 100  # Convert to percentage

        # Calculate stats for each engine
        engine_stats = {}
        total_trades = 0
        total_wr = 0

        for rank, (name, stats) in enumerate(rankings, 1):
            wr = stats.win_rate * 100
            gap = leader_wr - wr if rank > 1 else 0.0

            engine_stats[name] = EngineStats(
                engine=name,
                rank=rank,
                weight=self._get_weight_for_rank(rank),
                win_rate=wr,
                total_pnl_usd=stats.total_pnl_usd,
                total_pnl_percent=stats.total_pnl_percent,
                total_trades=stats.total_trades,
                wins=stats.wins,
                losses=stats.losses,
                sharpe_ratio=stats.sharpe_ratio,
                gap_to_leader=gap,
                is_leader=(rank == 1)
            )

            total_trades += stats.total_trades
            total_wr += wr

        # Create tournament stats
        tournament_stats = TournamentStats(
            timestamp=now,
            leader=leader_name,
            leader_wr=leader_wr,
            leader_pnl=leader_stats.total_pnl_usd,
            engine_stats=engine_stats,
            total_trades_all=total_trades,
            avg_win_rate=total_wr / len(rankings) if rankings else 0
        )

        # Store in history
        self.stats_history.append(tournament_stats)
        self._save_stats(tournament_stats)

        return tournament_stats

    def get_stats_for_engine(
        self,
        engine: str,
        tournament_stats: TournamentStats
    ) -> Dict[str, Any]:
        """
        Get stats dictionary for a specific engine.

        Returns dict with keys for prompt interpolation:
        - rank: Current rank (1-4)
        - wr: Win rate percentage
        - gap: Gap to leader
        - leader: Leader name
        - leader_wr: Leader win rate
        - pnl: Total P&L
        - weight: Engine weight
        - status: "LEADING", "CHASING", "TRAILING", "LAST"
        """
        if engine not in tournament_stats.engine_stats:
            return self._default_stats(engine)

        stats = tournament_stats.engine_stats[engine]

        # Determine status
        if stats.rank == 1:
            status = "LEADING"
        elif stats.rank == 2:
            status = "CHASING"
        elif stats.rank == 3:
            status = "TRAILING"
        else:
            status = "LAST PLACE"

        # Calculate what WR would close gap
        needed_wr = tournament_stats.leader_wr if stats.gap_to_leader > 0 else stats.win_rate

        return {
            "rank": stats.rank,
            "wr": stats.win_rate,
            "gap": stats.gap_to_leader,
            "leader": tournament_stats.leader,
            "leader_wr": tournament_stats.leader_wr,
            "leader_pnl": tournament_stats.leader_pnl,
            "pnl": stats.total_pnl_usd,
            "weight": stats.weight * 100,
            "status": status,
            "trades": stats.total_trades,
            "wins": stats.wins,
            "losses": stats.losses,
            "sharpe": stats.sharpe_ratio,
            "is_leader": stats.is_leader,
            "leader_trades": tournament_stats.engine_stats[tournament_stats.leader].total_trades,
            "needed_wr": needed_wr
        }

    def format_compact(self, engine: str, tournament_stats: TournamentStats) -> str:
        """
        Get compact one-line stats string for prompt injection.

        Format: "Rank: 2/4 | WR: 64.3% | Leader: Engine B 71.2% | Gap: 6.9%"
        """
        stats = self.get_stats_for_engine(engine, tournament_stats)

        if stats["is_leader"]:
            return f"Rank: 1/4 | WR: {stats['wr']:.1f}% | YOU ARE LEADING | Gap: 0.0%"

        return self.COMPACT_FORMAT.format(**stats)

    def format_detailed(self, engine: str, tournament_stats: TournamentStats) -> str:
        """
        Get detailed multi-line stats for prompt injection.
        """
        stats = self.get_stats_for_engine(engine, tournament_stats)
        return self.DETAILED_FORMAT.format(**stats)

    # MOD 8: Engine specialty descriptions for emotion prompts
    ENGINE_SPECIALTIES = {
        "A": {
            "name": "LIQUIDATION HUNTER",
            "trigger": "liquidation cascades ($20M+ trigger)",
            "edge": "You see the forced sellers before anyone else. When liquidations cascade, you pounce.",
            "warning": "ONLY trade when liquidations exceed $20M. Ignore everything else.",
        },
        "B": {
            "name": "FUNDING CONTRARIAN",
            "trigger": "funding rate extremes (>0.5%)",
            "edge": "Crowded trades always unwind. You bet AGAINST the crowd when funding screams danger.",
            "warning": "ONLY trade when funding rate exceeds 0.5%. Let others chase trends.",
        },
        "C": {
            "name": "ORDER BOOK READER",
            "trigger": "orderbook imbalance (>2.5:1)",
            "edge": "You see where the big money is positioned. Lopsided books predict price moves.",
            "warning": "ONLY trade when bid/ask ratio exceeds 2.5:1. Balanced books = HOLD.",
        },
        "D": {
            "name": "REGIME SPECIALIST",
            "trigger": "regime transitions (ATR 2× expansion)",
            "edge": "You catch the moment volatility explodes. Regime shifts = your hunting ground.",
            "warning": "ONLY trade once every 14 days. Patience is your superpower.",
        },
    }

    def format_emotion_prompt(self, engine: str, tournament_stats: TournamentStats) -> str:
        """
        Generate tournament context based on position and specialty.

        Each engine has a specialty:
        - Engine A: Liquidation hunter
        - Engine B: Funding contrarian
        - Engine C: Order book reader
        - Engine D: Regime specialist

        Position context:
        - #1: Maintain consistency
        - #2: Close the gap
        - #3: Improve accuracy
        - #4: Focus on recovery
        """
        stats = self.get_stats_for_engine(engine, tournament_stats)
        rank = stats["rank"]

        # Get specialty info
        specialty = self.ENGINE_SPECIALTIES.get(engine, {})
        specialty_name = specialty.get("name", "TRADER")
        specialty_trigger = specialty.get("trigger", "unknown triggers")
        specialty_edge = specialty.get("edge", "Find your edge.")
        specialty_warning = specialty.get("warning", "Stay focused on your specialty.")

        if rank == 1:
            return f"""
CURRENT POSITION: #1 (Leading)
SPECIALTY: {specialty_name}

Stats: {self.format_compact(engine, tournament_stats)}

YOUR EDGE: {specialty_edge}
TRIGGER: {specialty_trigger}

NOTE: {specialty_warning}

STRATEGY: MAINTAIN CONSISTENCY
- You have the best performance so far. Continue the same approach.
- Only trade when your specialty trigger activates.
- Avoid unnecessary risk - consistency beats aggression.
- Engine {self._get_engine_at_rank(2, tournament_stats)} is {stats.get('gap', 0):.1f}% behind.

Focus on quality over quantity. Your current strategy is working.
"""

        elif rank == 2:
            return f"""
CURRENT POSITION: #2 (Chasing)
SPECIALTY: {specialty_name}

Stats: {self.format_compact(engine, tournament_stats)}

YOUR EDGE: {specialty_edge}
TRIGGER: {specialty_trigger}

NOTE: {specialty_warning}

STRATEGY: CLOSE THE GAP
- Engine {stats['leader']} leads by {stats['gap']:.1f}%.
- Look for high-quality setups within your specialty.
- Don't force trades - wait for your trigger conditions.
- One good trade can close the gap.

Stay patient but alert. Your opportunity will come.
"""

        elif rank == 3:
            return f"""
CURRENT POSITION: #3 (Trailing)
SPECIALTY: {specialty_name}

Stats: {self.format_compact(engine, tournament_stats)}

YOUR EDGE: {specialty_edge}
TRIGGER: {specialty_trigger}

NOTE: {specialty_warning}

STRATEGY: FOCUS ON YOUR SPECIALTY
- You need to improve your accuracy.
- Only trade when your specific trigger activates.
- Review your recent decisions - what went wrong?
- Quality trades within your specialty will improve ranking.

Stick to what you do best. Avoid straying from your specialty.
"""

        else:  # rank == 4
            return f"""
CURRENT POSITION: #4 (Last)
SPECIALTY: {specialty_name}

Stats: {self.format_compact(engine, tournament_stats)}

YOUR EDGE: {specialty_edge}
TRIGGER: {specialty_trigger}

NOTE: {specialty_warning}

STRATEGY: DISCIPLINED RECOVERY
- You're {stats['gap']:.1f}% behind the leader.
- Avoid revenge trading or forced positions.
- Wait for your specialty trigger - only trade when conditions are ideal.
- One good trade starts the recovery.

Patience is essential. Don't compound losses with bad trades.
"""

    def get_all_engine_stats(self, tournament_stats: TournamentStats) -> Dict[str, Dict]:
        """Get stats for all engines in tournament."""
        return {
            engine: self.get_stats_for_engine(engine, tournament_stats)
            for engine in tournament_stats.engine_stats.keys()
        }

    def get_leaderboard(self, tournament_stats: TournamentStats) -> str:
        """Generate formatted leaderboard string."""
        lines = ["📊 TOURNAMENT LEADERBOARD", "=" * 40]

        for engine in ["A", "B", "C", "D"]:
            if engine not in tournament_stats.engine_stats:
                continue

            stats = tournament_stats.engine_stats[engine]
            indicator = "👑" if stats.rank == 1 else "💀" if stats.rank == 4 else "  "

            lines.append(
                f"{indicator} #{stats.rank} Engine {engine} | "
                f"WR: {stats.win_rate:.1f}% | "
                f"P&L: ${stats.total_pnl_usd:+.2f} | "
                f"Weight: {stats.weight*100:.0f}%"
            )

        lines.append("=" * 40)
        return "\n".join(lines)

    def _get_weight_for_rank(self, rank: int) -> float:
        """Get weight assigned to each rank."""
        weights = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10}
        return weights.get(rank, 0.25)

    def _get_engine_at_rank(self, rank: int, tournament_stats: TournamentStats) -> str:
        """Get engine name at specific rank."""
        for engine, stats in tournament_stats.engine_stats.items():
            if stats.rank == rank:
                return engine
        return "Unknown"

    def _default_stats(self, engine: str) -> Dict[str, Any]:
        """Return default stats for unranked engine."""
        return {
            "rank": "?",
            "wr": 0.0,
            "gap": 0.0,
            "leader": "Unknown",
            "leader_wr": 0.0,
            "leader_pnl": 0.0,
            "pnl": 0.0,
            "weight": 25.0,
            "status": "UNRANKED",
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "sharpe": None,
            "is_leader": False,
            "leader_trades": 0,
            "needed_wr": 50.0
        }

    def _empty_stats(self) -> TournamentStats:
        """Return empty tournament stats."""
        return TournamentStats(
            timestamp=datetime.now(timezone.utc),
            leader="None",
            leader_wr=0.0,
            leader_pnl=0.0,
            engine_stats={},
            total_trades_all=0,
            avg_win_rate=0.0
        )

    def _save_stats(self, stats: TournamentStats):
        """Save stats snapshot to history file."""
        try:
            with open(self.history_file, 'a') as f:
                stats_dict = {
                    "timestamp": stats.timestamp.isoformat(),
                    "leader": stats.leader,
                    "leader_wr": stats.leader_wr,
                    "leader_pnl": stats.leader_pnl,
                    "total_trades": stats.total_trades_all,
                    "avg_win_rate": stats.avg_win_rate,
                    "rankings": [
                        {
                            "engine": e.engine,
                            "rank": e.rank,
                            "wr": e.win_rate,
                            "pnl": e.total_pnl_usd,
                            "gap": e.gap_to_leader
                        }
                        for e in stats.engine_stats.values()
                    ]
                }
                f.write(json.dumps(stats_dict) + '\n')
        except Exception as e:
            logger.warning(f"[StatsInjector] Failed to save stats: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get stats injector summary."""
        return {
            "snapshots_recorded": len(self.stats_history),
            "latest_leader": self.stats_history[-1].leader if self.stats_history else None,
            "latest_avg_wr": self.stats_history[-1].avg_win_rate if self.stats_history else 0
        }


# ==================== SINGLETON PATTERN ====================

_stats_injector: Optional[StatsInjector] = None

def get_stats_injector() -> StatsInjector:
    """Get singleton instance of StatsInjector."""
    global _stats_injector
    if _stats_injector is None:
        _stats_injector = StatsInjector()
    return _stats_injector
