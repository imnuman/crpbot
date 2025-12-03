"""
HYDRA 3.0 - Engine Portfolio System

Individual P&L tracking for each engine in competition mode.

Each engine maintains:
- Independent trade history
- Win rate statistics
- P&L performance
- Current rank
- Performance metrics

This enables:
- Tournament ranking based on actual P&L
- Weight adjustment (24-hour cycle)
- Breeding mechanism (4-day cycle)
- Winner teaches losers
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger
import json
from pathlib import Path


@dataclass
class EngineTrade:
    """Individual trade record for an engine"""
    trade_id: str
    engine: str  # A, B, C, or D
    asset: str
    direction: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # stop_loss, take_profit, manual
    outcome: Optional[str] = None  # win, loss
    pnl_percent: float = 0.0
    pnl_usd: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED


@dataclass
class EngineStats:
    """Performance statistics for a engine"""
    engine: str
    total_trades: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_percent: float = 0.0
    total_pnl_usd: float = 0.0
    avg_win_percent: float = 0.0
    avg_loss_percent: float = 0.0
    best_trade_percent: float = 0.0
    worst_trade_percent: float = 0.0
    sharpe_ratio: Optional[float] = None
    current_rank: int = 0  # 1 = best, 4 = worst
    weight: float = 0.25  # Default: equal weight (1/4)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EnginePortfolio:
    """
    Portfolio manager for a single engine.

    Tracks all trades, calculates performance, maintains stats.
    """

    def __init__(self, engine: str, data_dir: Path = Path("/root/crpbot/data/hydra")):
        self.engine = engine
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.trades: List[EngineTrade] = []
        self.stats = EngineStats(engine=engine)

        # Load existing trades if available
        self._load_trades()
        self._recalculate_stats()

        logger.info(f"Engine {engine} portfolio initialized (trades: {self.stats.total_trades}, WR: {self.stats.win_rate:.1%})")

    def add_trade(
        self,
        trade_id: str,
        asset: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float
    ) -> EngineTrade:
        """
        Add new trade to portfolio.
        """
        trade = EngineTrade(
            trade_id=trade_id,
            engine=self.engine,
            asset=asset,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            entry_time=datetime.now(timezone.utc),
            status="OPEN"
        )

        self.trades.append(trade)
        self._recalculate_stats()
        self._save_trades()

        logger.info(
            f"Engine {self.engine} opened {direction} {asset} @ ${entry_price:.2f} "
            f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})"
        )

        return trade

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str
    ) -> Optional[EngineTrade]:
        """
        Close existing trade and calculate P&L.
        """
        trade = self._find_trade(trade_id)
        if not trade:
            logger.error(f"Trade {trade_id} not found for engine {self.engine}")
            return None

        if trade.status == "CLOSED":
            logger.warning(f"Trade {trade_id} already closed")
            return trade

        # Calculate P&L
        if trade.direction == "BUY":
            pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:  # SELL
            pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        pnl_usd = (pnl_percent / 100) * trade.position_size

        # Update trade
        trade.exit_time = datetime.now(timezone.utc)
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pnl_percent = pnl_percent
        trade.pnl_usd = pnl_usd
        trade.outcome = "win" if pnl_percent > 0 else "loss"
        trade.status = "CLOSED"

        self._recalculate_stats()
        self._save_trades()

        logger.info(
            f"Engine {self.engine} closed {trade.direction} {trade.asset}: "
            f"{trade.outcome.upper()} {pnl_percent:+.2f}% (${pnl_usd:+.2f}) via {exit_reason}"
        )

        return trade

    def get_stats(self) -> EngineStats:
        """Get current performance stats."""
        return self.stats

    def get_open_trades(self) -> List[EngineTrade]:
        """Get all open trades."""
        return [t for t in self.trades if t.status == "OPEN"]

    def get_closed_trades(self) -> List[EngineTrade]:
        """Get all closed trades."""
        return [t for t in self.trades if t.status == "CLOSED"]

    def get_recent_trades(self, limit: int = 10) -> List[EngineTrade]:
        """Get recent trades (closed and open)."""
        return sorted(self.trades, key=lambda t: t.entry_time, reverse=True)[:limit]

    def _find_trade(self, trade_id: str) -> Optional[EngineTrade]:
        """Find trade by ID."""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    def _recalculate_stats(self):
        """Recalculate all performance statistics."""
        closed_trades = self.get_closed_trades()

        self.stats.total_trades = len(self.trades)
        self.stats.open_trades = len([t for t in self.trades if t.status == "OPEN"])
        self.stats.closed_trades = len(closed_trades)

        if closed_trades:
            wins = [t for t in closed_trades if t.outcome == "win"]
            losses = [t for t in closed_trades if t.outcome == "loss"]

            self.stats.wins = len(wins)
            self.stats.losses = len(losses)
            self.stats.win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0

            self.stats.total_pnl_percent = sum(t.pnl_percent for t in closed_trades)
            self.stats.total_pnl_usd = sum(t.pnl_usd for t in closed_trades)

            if wins:
                self.stats.avg_win_percent = sum(t.pnl_percent for t in wins) / len(wins)
                self.stats.best_trade_percent = max(t.pnl_percent for t in wins)

            if losses:
                self.stats.avg_loss_percent = sum(t.pnl_percent for t in losses) / len(losses)
                self.stats.worst_trade_percent = min(t.pnl_percent for t in losses)

            # Calculate Sharpe ratio (if enough trades)
            if len(closed_trades) >= 10:
                returns = [t.pnl_percent for t in closed_trades]
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                std_dev = variance ** 0.5

                if std_dev > 0:
                    # Annualized Sharpe (assuming daily trades)
                    self.stats.sharpe_ratio = (mean_return / std_dev) * (252 ** 0.5)
                else:
                    self.stats.sharpe_ratio = 0.0

        self.stats.last_updated = datetime.now(timezone.utc)

    def _save_trades(self):
        """Save trades to JSONL file."""
        trades_file = self.data_dir / f"engine_{self.engine}_trades.jsonl"

        with open(trades_file, 'w') as f:
            for trade in self.trades:
                trade_dict = {
                    "trade_id": trade.trade_id,
                    "engine": trade.engine,
                    "asset": trade.asset,
                    "direction": trade.direction,
                    "entry_price": trade.entry_price,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "position_size": trade.position_size,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                    "exit_price": trade.exit_price,
                    "exit_reason": trade.exit_reason,
                    "outcome": trade.outcome,
                    "pnl_percent": trade.pnl_percent,
                    "pnl_usd": trade.pnl_usd,
                    "status": trade.status
                }
                f.write(json.dumps(trade_dict) + '\n')

    def _load_trades(self):
        """Load trades from JSONL file."""
        trades_file = self.data_dir / f"engine_{self.engine}_trades.jsonl"

        if not trades_file.exists():
            return

        with open(trades_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                trade = EngineTrade(
                    trade_id=data["trade_id"],
                    engine=data["engine"],
                    asset=data["asset"],
                    direction=data["direction"],
                    entry_price=data["entry_price"],
                    stop_loss=data["stop_loss"],
                    take_profit=data["take_profit"],
                    position_size=data["position_size"],
                    entry_time=datetime.fromisoformat(data["entry_time"]),
                    exit_time=datetime.fromisoformat(data["exit_time"]) if data["exit_time"] else None,
                    exit_price=data["exit_price"],
                    exit_reason=data["exit_reason"],
                    outcome=data["outcome"],
                    pnl_percent=data["pnl_percent"],
                    pnl_usd=data["pnl_usd"],
                    status=data["status"]
                )

                self.trades.append(trade)

        logger.debug(f"Loaded {len(self.trades)} trades for engine {self.engine}")


class TournamentManager:
    """
    Manages tournament across all 4 engines.

    Responsibilities:
    - Track all engine portfolios
    - Calculate rankings
    - Adjust weights (24-hour cycle)
    - Trigger breeding (4-day cycle)
    - Facilitate winner teaches losers
    """

    def __init__(self, data_dir: Path = Path("/root/crpbot/data/hydra")):
        self.data_dir = data_dir

        # Initialize portfolios for all 4 engines
        self.portfolios: Dict[str, EnginePortfolio] = {
            "A": EnginePortfolio("A", data_dir),
            "B": EnginePortfolio("B", data_dir),
            "C": EnginePortfolio("C", data_dir),
            "D": EnginePortfolio("D", data_dir)
        }

        self.last_weight_adjustment = datetime.now(timezone.utc)
        self.last_breeding = datetime.now(timezone.utc)
        self.tournament_start = datetime.now(timezone.utc)

        logger.info("Tournament Manager initialized with 4 engines")

    def get_portfolio(self, engine: str) -> EnginePortfolio:
        """Get portfolio for specific engine."""
        if engine not in self.portfolios:
            raise ValueError(f"Invalid engine: {engine}")
        return self.portfolios[engine]

    def calculate_rankings(self) -> List[Tuple[str, EngineStats]]:
        """
        Calculate current tournament rankings.

        Ranking based on:
        1. Total P&L (USD) - primary metric
        2. Win rate - tiebreaker
        3. Sharpe ratio - secondary tiebreaker

        Returns:
            List of (engine, stats) sorted by rank (best first)
        """
        engines = []

        for name, portfolio in self.portfolios.items():
            stats = portfolio.stats
            engines.append((name, stats))

        # Sort by P&L (descending), then win rate, then Sharpe
        engines.sort(
            key=lambda x: (
                x[1].total_pnl_usd,
                x[1].win_rate,
                x[1].sharpe_ratio if x[1].sharpe_ratio else 0.0
            ),
            reverse=True
        )

        # Update ranks
        for i, (name, stats) in enumerate(engines):
            stats.current_rank = i + 1

        return engines

    def adjust_weights(self) -> Dict[str, float]:
        """
        Adjust engine weights based on performance.

        Weight formula:
        - Rank 1 (best): 40% weight
        - Rank 2: 30% weight
        - Rank 3: 20% weight
        - Rank 4 (worst): 10% weight

        This replaces "killing" - worst engine still participates but with less influence.

        Called every 24 hours.
        """
        rankings = self.calculate_rankings()

        weight_distribution = {
            1: 0.40,  # 40%
            2: 0.30,  # 30%
            3: 0.20,  # 20%
            4: 0.10   # 10%
        }

        weights = {}

        for name, stats in rankings:
            new_weight = weight_distribution[stats.current_rank]
            stats.weight = new_weight
            weights[name] = new_weight

            logger.info(
                f"Engine {name} (Rank {stats.current_rank}): "
                f"P&L: ${stats.total_pnl_usd:+.2f} ({stats.total_pnl_percent:+.2f}%), "
                f"WR: {stats.win_rate:.1%}, Weight: {new_weight:.1%}"
            )

        self.last_weight_adjustment = datetime.now(timezone.utc)
        return weights

    def should_adjust_weights(self) -> bool:
        """Check if 24 hours have passed since last weight adjustment."""
        hours_since = (datetime.now(timezone.utc) - self.last_weight_adjustment).seconds / 3600
        return hours_since >= 24

    def should_breed(self) -> bool:
        """Check if 4 days have passed since last breeding."""
        days_since = (datetime.now(timezone.utc) - self.last_breeding).days
        return days_since >= 4

    def get_breeding_candidates(self) -> Tuple[str, str]:
        """
        Get top 2 engines for breeding.

        Returns:
            (engine_1, engine_2) - top 2 performers
        """
        rankings = self.calculate_rankings()
        return rankings[0][0], rankings[1][0]

    def get_winner_teaches_losers_pairs(self) -> List[Tuple[str, str]]:
        """
        Get winner-loser pairs for knowledge transfer.

        Returns:
            [(winner, loser), ...] pairs for teaching
        """
        rankings = self.calculate_rankings()

        # Winner (rank 1) teaches all losers (ranks 2-4)
        winner = rankings[0][0]
        losers = [r[0] for r in rankings[1:]]

        return [(winner, loser) for loser in losers]

    def get_tournament_summary(self) -> Dict:
        """Get comprehensive tournament statistics."""
        rankings = self.calculate_rankings()

        summary = {
            "tournament_duration_hours": (datetime.now(timezone.utc) - self.tournament_start).seconds / 3600,
            "last_weight_adjustment": self.last_weight_adjustment.isoformat(),
            "last_breeding": self.last_breeding.isoformat(),
            "rankings": []
        }

        for name, stats in rankings:
            summary["rankings"].append({
                "engine": name,
                "rank": stats.current_rank,
                "weight": stats.weight,
                "total_trades": stats.total_trades,
                "closed_trades": stats.closed_trades,
                "win_rate": stats.win_rate,
                "total_pnl_percent": stats.total_pnl_percent,
                "total_pnl_usd": stats.total_pnl_usd,
                "sharpe_ratio": stats.sharpe_ratio
            })

        return summary

    def get_stats_for_prompt(self, engine: str) -> Dict[str, str]:
        """
        Get stats for injecting into engine prompts.

        Returns dict with placeholders:
        - {rank}: Current rank (1-4)
        - {wr}: Win rate percentage
        - {pnl}: Total P&L
        - {leader}: Name of current leader
        - {leader_pnl}: Leader's P&L
        """
        rankings = self.calculate_rankings()
        portfolio = self.get_portfolio(engine)

        leader = rankings[0][0]
        leader_stats = rankings[0][1]

        return {
            "rank": str(portfolio.stats.current_rank),
            "wr": f"{portfolio.stats.win_rate:.1%}",
            "pnl": f"${portfolio.stats.total_pnl_usd:+.2f}",
            "leader": leader,
            "leader_pnl": f"${leader_stats.total_pnl_usd:+.2f}"
        }


# ==================== SINGLETON PATTERN ====================

_tournament_manager = None

def get_tournament_manager() -> TournamentManager:
    """Get singleton instance of TournamentManager."""
    global _tournament_manager
    if _tournament_manager is None:
        _tournament_manager = TournamentManager()
    return _tournament_manager
