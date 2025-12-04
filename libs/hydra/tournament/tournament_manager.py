"""
HYDRA 3.0 Tournament Manager

Ranks engines by independent P&L (not votes).
Assigns weights: #1=40%, #2=30%, #3=20%, #4=10%
Tracks rank changes over time.

Phase 1, Week 1 - Step 10
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from libs.hydra.engine_portfolio import EnginePortfolio

logger = logging.getLogger(__name__)


@dataclass
class EngineRanking:
    """Single engine's ranking data"""
    engine: str
    rank: int
    weight: float
    win_rate: float
    total_pnl: float
    sharpe_ratio: Optional[float]
    total_trades: int
    strategies_tested: int


@dataclass
class TournamentSnapshot:
    """Snapshot of tournament rankings at a point in time"""
    timestamp: datetime
    rankings: List[EngineRanking]
    leader_name: str
    leader_wr: float

    def get_engine_rank(self, engine: str) -> Optional[EngineRanking]:
        """Get ranking for specific engine"""
        for r in self.rankings:
            if r.engine == engine:
                return r
        return None

    def get_gap_to_leader(self, engine: str) -> Optional[float]:
        """Calculate performance gap to leader for given engine"""
        eng_rank = self.get_engine_rank(engine)
        if eng_rank is None:
            return None
        return self.leader_wr - eng_rank.win_rate


class TournamentManager:
    """
    Tournament Manager for HYDRA 3.0

    Responsibilities:
    - Rank engines by independent P&L (NOT votes)
    - Assign weights: #1=40%, #2=30%, #3=20%, #4=10%
    - Track rank changes over time
    - Provide stats for emotion prompt injection
    """

    # Weight distribution by rank
    WEIGHTS = {
        1: 0.40,  # Winner gets 40% influence
        2: 0.30,
        3: 0.20,
        4: 0.10   # Worst gets 10% influence
    }

    def __init__(self, portfolios: Dict[str, EnginePortfolio]):
        """
        Initialize tournament manager

        Args:
            portfolios: Dict mapping engine name (A, B, C, D) to EnginePortfolio
        """
        self.portfolios = portfolios
        self.history: List[TournamentSnapshot] = []
        logger.info("[TournamentManager] Initialized with 4 engines")

    def calculate_rankings(self) -> TournamentSnapshot:
        """
        Calculate current engine rankings based on independent P&L

        Ranking criteria (in order of priority):
        1. Total P&L % (primary)
        2. Win rate (tiebreaker)
        3. Sharpe ratio (secondary tiebreaker)

        Returns:
            TournamentSnapshot with rankings and stats
        """
        engine_stats: List[Tuple[str, float, float, Optional[float], int, int]] = []

        # Collect stats from all portfolios
        for engine_name, portfolio in self.portfolios.items():
            stats = portfolio.get_stats()
            engine_stats.append((
                engine_name,
                stats.total_pnl_percent,  # Primary ranking metric
                stats.win_rate,           # Tiebreaker
                stats.sharpe_ratio,       # Secondary tiebreaker
                stats.total_trades,
                stats.total_strategies_tested
            ))

        # Sort by: P&L (desc), then win_rate (desc), then Sharpe (desc)
        engine_stats.sort(
            key=lambda x: (x[1], x[2], x[3] if x[3] is not None else -999),
            reverse=True
        )

        # Build rankings with assigned weights
        rankings: List[EngineRanking] = []
        for rank, (engine, pnl, wr, sharpe, trades, strategies) in enumerate(engine_stats, start=1):
            rankings.append(EngineRanking(
                engine=engine,
                rank=rank,
                weight=self.WEIGHTS[rank],
                win_rate=wr,
                total_pnl=pnl,
                sharpe_ratio=sharpe,
                total_trades=trades,
                strategies_tested=strategies
            ))

        # Identify leader
        leader = rankings[0]
        snapshot = TournamentSnapshot(
            timestamp=datetime.utcnow(),
            rankings=rankings,
            leader_name=leader.engine,
            leader_wr=leader.win_rate
        )

        # Store in history
        self.history.append(snapshot)

        logger.info(
            f"[TournamentManager] Rankings calculated: "
            f"#1={leader.engine} ({leader.win_rate:.1f}% WR, {leader.total_pnl:.2f}% P&L)"
        )

        return snapshot

    def get_current_rankings(self) -> Optional[TournamentSnapshot]:
        """Get most recent tournament rankings"""
        if not self.history:
            return None
        return self.history[-1]

    def get_engine_weight(self, engine: str) -> float:
        """
        Get current weight for engine (used in Mother AI decision making)

        Args:
            engine: Engine name (A, B, C, D)

        Returns:
            Weight between 0.10 and 0.40
        """
        snapshot = self.get_current_rankings()
        if snapshot is None:
            # Default: equal weights if no rankings yet
            return 0.25

        rank_data = snapshot.get_engine_rank(engine)
        if rank_data is None:
            logger.warning(f"[TournamentManager] Engine {engine} not found in rankings")
            return 0.25

        return rank_data.weight

    def get_stats_for_prompt(self, engine: str) -> Dict[str, any]:
        """
        Get stats for emotion prompt injection

        Returns dict with keys:
        - rank: Current rank (1-4)
        - wr: Win rate percentage
        - gap: Gap to leader in percentage points
        - leader_name: Name of current leader
        - leader_wr: Leader's win rate

        Args:
            engine: Engine name (A, B, C, D)

        Returns:
            Dict with stats for prompt injection
        """
        snapshot = self.get_current_rankings()

        if snapshot is None:
            # No data yet - return neutral stats
            return {
                "rank": "?",
                "wr": 0.0,
                "gap": 0.0,
                "leader_name": "Unknown",
                "leader_wr": 0.0
            }

        rank_data = snapshot.get_engine_rank(engine)
        if rank_data is None:
            logger.warning(f"[TournamentManager] Engine {engine} not in rankings")
            return {
                "rank": "?",
                "wr": 0.0,
                "gap": 0.0,
                "leader_name": snapshot.leader_name,
                "leader_wr": snapshot.leader_wr
            }

        gap = snapshot.get_gap_to_leader(engine)

        return {
            "rank": rank_data.rank,
            "wr": rank_data.win_rate,
            "gap": gap if gap is not None else 0.0,
            "leader_name": snapshot.leader_name,
            "leader_wr": snapshot.leader_wr
        }

    def get_winner(self) -> Optional[str]:
        """Get current tournament leader (rank #1)"""
        snapshot = self.get_current_rankings()
        if snapshot is None:
            return None
        return snapshot.leader_name

    def get_loser(self) -> Optional[str]:
        """Get current worst performer (rank #4)"""
        snapshot = self.get_current_rankings()
        if snapshot is None:
            return None

        # Find rank 4
        for r in snapshot.rankings:
            if r.rank == 4:
                return r.engine
        return None

    def print_rankings(self) -> None:
        """Print current rankings to console (for debugging/monitoring)"""
        snapshot = self.get_current_rankings()
        if snapshot is None:
            logger.info("[TournamentManager] No rankings available yet")
            return

        logger.info("=" * 60)
        logger.info("[TournamentManager] Current Tournament Rankings")
        logger.info("=" * 60)

        for r in snapshot.rankings:
            leader_indicator = "👑 LEADER" if r.rank == 1 else ""
            loser_indicator = "💀 LAST PLACE" if r.rank == 4 else ""

            logger.info(
                f"  #{r.rank} - Engine {r.engine} | "
                f"Weight: {r.weight*100:.0f}% | "
                f"WR: {r.win_rate:.1f}% | "
                f"P&L: {r.total_pnl:+.2f}% | "
                f"Trades: {r.total_trades} | "
                f"{leader_indicator}{loser_indicator}"
            )

        logger.info("=" * 60)

    def get_rank_changes(self) -> Dict[str, int]:
        """
        Calculate rank changes from previous snapshot

        Returns:
            Dict mapping engine name to rank change
            Positive = moved up, Negative = moved down
        """
        if len(self.history) < 2:
            return {}

        prev_snapshot = self.history[-2]
        curr_snapshot = self.history[-1]

        changes = {}
        for engine in ["A", "B", "C", "D"]:
            prev_rank_data = prev_snapshot.get_engine_rank(engine)
            curr_rank_data = curr_snapshot.get_engine_rank(engine)

            if prev_rank_data and curr_rank_data:
                # Rank change (negative = improved rank)
                change = prev_rank_data.rank - curr_rank_data.rank
                changes[engine] = change

        return changes
