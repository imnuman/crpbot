"""
HYDRA 4.0 - Paper Trading Gate

Ensures strategies pass paper trading validation before live execution.
Requirements:
- Minimum 5 paper trades
- Paper win rate >= 65%
- Recent paper win rate >= 60%
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class PaperGateResult:
    """Result of paper trading gate check."""
    approved: bool
    strategy_id: str
    paper_trades: int
    paper_wr: float
    recent_paper_wr: float
    reason: str
    action: str  # "APPROVE", "NEEDS_MORE_TRADES", "WR_TOO_LOW", "RECENT_WR_LOW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "strategy_id": self.strategy_id,
            "paper_trades": self.paper_trades,
            "paper_wr": self.paper_wr,
            "recent_paper_wr": self.recent_paper_wr,
            "reason": self.reason,
            "action": self.action,
        }


class PaperTradingGate:
    """
    Gate that validates strategies have sufficient paper trading success.

    Requirements:
    - MIN_PAPER_TRADES: 5 (minimum paper trades before live)
    - MIN_PAPER_WR: 65% (overall paper win rate)
    - MIN_RECENT_WR: 60% (win rate in last 3 trades)
    """

    # Gate thresholds
    MIN_PAPER_TRADES = 5
    MIN_PAPER_WR = 0.65  # 65%
    MIN_RECENT_WR = 0.60  # 60% in last 3 trades
    RECENT_TRADE_COUNT = 3

    def __init__(self):
        """Initialize the paper trading gate."""
        logger.info("[PaperGate] Initialized")
        logger.info(f"[PaperGate] Thresholds: trades>={self.MIN_PAPER_TRADES}, WR>={self.MIN_PAPER_WR*100}%, recent>={self.MIN_RECENT_WR*100}%")

    def check_paper_requirement(
        self,
        strategy_id: str,
        paper_trades: int,
        paper_wr: float,
        paper_trade_history: Optional[List[str]] = None
    ) -> PaperGateResult:
        """
        Check if strategy meets paper trading requirements.

        Args:
            strategy_id: Strategy identifier
            paper_trades: Total number of paper trades
            paper_wr: Overall paper win rate (0-1)
            paper_trade_history: List of recent outcomes ["win", "loss", ...]

        Returns:
            PaperGateResult with approval status and details
        """
        # Calculate recent win rate
        recent_wr = self._calculate_recent_wr(paper_trade_history)

        # Check minimum trades
        if paper_trades < self.MIN_PAPER_TRADES:
            needed = self.MIN_PAPER_TRADES - paper_trades
            return PaperGateResult(
                approved=False,
                strategy_id=strategy_id,
                paper_trades=paper_trades,
                paper_wr=paper_wr,
                recent_paper_wr=recent_wr,
                reason=f"Needs {needed} more paper trades (has {paper_trades}/{self.MIN_PAPER_TRADES})",
                action="NEEDS_MORE_TRADES"
            )

        # Check overall win rate
        if paper_wr < self.MIN_PAPER_WR:
            return PaperGateResult(
                approved=False,
                strategy_id=strategy_id,
                paper_trades=paper_trades,
                paper_wr=paper_wr,
                recent_paper_wr=recent_wr,
                reason=f"Paper WR {paper_wr*100:.1f}% below minimum {self.MIN_PAPER_WR*100}%",
                action="WR_TOO_LOW"
            )

        # Check recent win rate (if enough trades)
        if paper_trade_history and len(paper_trade_history) >= self.RECENT_TRADE_COUNT:
            if recent_wr < self.MIN_RECENT_WR:
                return PaperGateResult(
                    approved=False,
                    strategy_id=strategy_id,
                    paper_trades=paper_trades,
                    paper_wr=paper_wr,
                    recent_paper_wr=recent_wr,
                    reason=f"Recent WR {recent_wr*100:.1f}% below minimum {self.MIN_RECENT_WR*100}%",
                    action="RECENT_WR_LOW"
                )

        # All checks passed
        return PaperGateResult(
            approved=True,
            strategy_id=strategy_id,
            paper_trades=paper_trades,
            paper_wr=paper_wr,
            recent_paper_wr=recent_wr,
            reason=f"Paper gate passed: {paper_trades} trades, {paper_wr*100:.1f}% WR",
            action="APPROVE"
        )

    def _calculate_recent_wr(self, trade_history: Optional[List[str]]) -> float:
        """Calculate win rate from recent trades."""
        if not trade_history:
            return 0.0

        recent = trade_history[-self.RECENT_TRADE_COUNT:]
        if not recent:
            return 0.0

        wins = sum(1 for t in recent if t == "win")
        return wins / len(recent)

    def check_strategy_from_memory(
        self,
        strategy_data: Dict[str, Any]
    ) -> PaperGateResult:
        """
        Check strategy using data from strategy memory.

        Args:
            strategy_data: Strategy dict from memory containing:
                - strategy_id
                - paper_trades
                - paper_wr
                - paper_trade_history (optional)

        Returns:
            PaperGateResult
        """
        return self.check_paper_requirement(
            strategy_id=strategy_data.get("strategy_id", "unknown"),
            paper_trades=strategy_data.get("paper_trades", 0),
            paper_wr=strategy_data.get("paper_wr", 0.0),
            paper_trade_history=strategy_data.get("paper_trade_history", [])
        )

    def get_strategies_ready_for_live(
        self,
        strategies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter strategies that are ready for live trading.

        Args:
            strategies: List of strategy dicts from memory

        Returns:
            List of approved strategies
        """
        approved = []

        for strategy in strategies:
            result = self.check_strategy_from_memory(strategy)
            if result.approved:
                approved.append(strategy)

        logger.info(f"[PaperGate] {len(approved)}/{len(strategies)} strategies approved for live")
        return approved

    def get_strategies_needing_paper(
        self,
        strategies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get strategies that still need paper trading.

        Args:
            strategies: List of strategy dicts from memory

        Returns:
            List of strategies needing more paper trades
        """
        needing_paper = []

        for strategy in strategies:
            result = self.check_strategy_from_memory(strategy)
            if not result.approved and result.action == "NEEDS_MORE_TRADES":
                needing_paper.append(strategy)

        logger.info(f"[PaperGate] {len(needing_paper)} strategies need more paper trades")
        return needing_paper


# Singleton instance
_gate_instance: Optional[PaperTradingGate] = None


def get_paper_gate() -> PaperTradingGate:
    """Get or create the paper trading gate singleton."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = PaperTradingGate()
    return _gate_instance
