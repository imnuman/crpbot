"""
HYDRA 3.0 - Strategy Memory Database

Persistent storage for each engine's winning strategies.
Enables strategy evolution instead of regenerating from scratch.

Schema:
{
    "A": {
        "BTC-USD:TRENDING": [
            {"strategy_id": "...", "win_rate": 0.65, "trades": 10, ...},
            ...
        ],
        "ETH-USD:CHOPPY": [...],
    },
    "B": {...},
    "C": {...},
    "D": {...}
}

Each engine maintains:
- 50-100 strategies per asset:regime combination
- Sorted by performance (win_rate * sqrt(trades))
- Top performers used for breeding and exploitation
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
import json
import math


class StrategyMemory:
    """
    Persistent strategy memory for all 4 engines.

    Features:
    - Store winning strategies per engine/asset/regime
    - Rank strategies by performance score
    - Provide top performers for exploitation
    - Track strategy lineage (parent strategies)
    """

    MAX_STRATEGIES_PER_KEY = 100  # Max strategies per engine:asset:regime
    MIN_STRATEGIES_FOR_EXPLOIT = 10  # Need at least 10 to start exploiting
    EXPLOIT_RATIO = 0.8  # 80% exploit, 20% explore

    # Edge Decay Detection thresholds
    DECAY_LOOKBACK_TRADES = 10  # Check last N trades for decay
    DECAY_WIN_RATE_THRESHOLD = 0.45  # Skip if recent WR < 45%

    # Regime-Strategy Validation thresholds
    MIN_REGIME_TRADES = 3  # Need at least 3 trades in regime to validate
    REGIME_WIN_RATE_THRESHOLD = 0.50  # Skip if regime WR < 50%

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            from .config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_file = self.data_dir / "strategy_database.json"

        # In-memory database: {engine: {asset:regime: [strategies]}}
        self.strategies: Dict[str, Dict[str, List[Dict]]] = {
            "A": {},
            "B": {},
            "C": {},
            "D": {}
        }

        self._load_database()
        logger.info(f"[StrategyMemory] Initialized with {self._count_strategies()} strategies")

    def _load_database(self):
        """Load strategy database from disk."""
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                    for engine in ["A", "B", "C", "D"]:
                        if engine in data:
                            self.strategies[engine] = data[engine]
                logger.info(f"[StrategyMemory] Loaded from {self.db_file}")
            except Exception as e:
                logger.error(f"[StrategyMemory] Failed to load: {e}")

    def _save_database(self):
        """Save strategy database to disk."""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.strategies, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[StrategyMemory] Failed to save: {e}")

    def _count_strategies(self) -> int:
        """Count total strategies in database."""
        total = 0
        for engine_data in self.strategies.values():
            for strategies in engine_data.values():
                total += len(strategies)
        return total

    def _calculate_score(self, strategy: Dict) -> float:
        """
        Calculate performance score for ranking.

        Score = win_rate * sqrt(trades) * (1 + avg_rr/10)

        This balances:
        - Win rate (primary metric)
        - Number of trades (confidence/significance)
        - Risk-reward ratio (quality of wins)
        """
        win_rate = strategy.get("win_rate", 0.5)
        trades = strategy.get("trades", 0)
        avg_rr = strategy.get("avg_rr", 1.0)

        if trades == 0:
            return 0.0

        return win_rate * math.sqrt(trades) * (1 + avg_rr / 10)

    def _get_key(self, asset: str, regime: str) -> str:
        """Get storage key for asset:regime combination."""
        return f"{asset}:{regime}"

    # ==================== CORE API ====================

    def add_strategy(
        self,
        engine: str,
        asset: str,
        regime: str,
        strategy: Dict,
        outcome: str = None,  # "win" or "loss"
        pnl_percent: float = 0.0,
        rr_actual: float = 0.0
    ) -> bool:
        """
        Add or update a strategy in the database.

        Args:
            engine: Engine name (A, B, C, D)
            asset: Asset symbol (e.g., "BTC-USD")
            regime: Market regime (e.g., "TRENDING")
            strategy: Strategy dict with strategy_id, direction, etc.
            outcome: Trade outcome ("win" or "loss")
            pnl_percent: P&L percentage
            rr_actual: Actual risk-reward ratio

        Returns:
            True if strategy was added/updated
        """
        if engine not in self.strategies:
            return False

        key = self._get_key(asset, regime)
        if key not in self.strategies[engine]:
            self.strategies[engine][key] = []

        strategy_id = strategy.get("strategy_id")
        if not strategy_id:
            return False

        # Check if strategy already exists
        existing = None
        for i, s in enumerate(self.strategies[engine][key]):
            if s.get("strategy_id") == strategy_id:
                existing = i
                break

        if existing is not None:
            # Update existing strategy with new trade result
            stored = self.strategies[engine][key][existing]
            stored["trades"] = stored.get("trades", 0) + 1

            if outcome == "win":
                stored["wins"] = stored.get("wins", 0) + 1
            elif outcome == "loss":
                stored["losses"] = stored.get("losses", 0) + 1

            # Update win rate
            total = stored.get("wins", 0) + stored.get("losses", 0)
            if total > 0:
                stored["win_rate"] = stored.get("wins", 0) / total

            # Update average RR
            if rr_actual > 0:
                old_rr = stored.get("avg_rr", 1.0)
                stored["avg_rr"] = (old_rr * (stored["trades"] - 1) + rr_actual) / stored["trades"]

            # Update P&L
            stored["total_pnl"] = stored.get("total_pnl", 0.0) + pnl_percent

            stored["last_used"] = datetime.now(timezone.utc).isoformat()
            stored["score"] = self._calculate_score(stored)

        else:
            # Add new strategy
            new_strategy = {
                "strategy_id": strategy_id,
                "gladiator": engine,
                "asset": asset,
                "regime": regime,
                "direction": strategy.get("direction", "HOLD"),
                "entry_rules": strategy.get("entry_rules", ""),
                "exit_rules": strategy.get("exit_rules", ""),
                "risk_per_trade": strategy.get("risk_per_trade", 0.01),
                "expected_rr": strategy.get("expected_rr", 1.5),
                "trades": 1 if outcome else 0,
                "wins": 1 if outcome == "win" else 0,
                "losses": 1 if outcome == "loss" else 0,
                "win_rate": 1.0 if outcome == "win" else (0.0 if outcome == "loss" else 0.5),
                "avg_rr": rr_actual if rr_actual > 0 else 1.0,
                "total_pnl": pnl_percent,
                "created": datetime.now(timezone.utc).isoformat(),
                "last_used": datetime.now(timezone.utc).isoformat(),
                "parent_ids": strategy.get("parent_ids", []),  # For tracking lineage
                "generation": strategy.get("generation", 0),
                "score": 0.0
            }
            new_strategy["score"] = self._calculate_score(new_strategy)

            self.strategies[engine][key].append(new_strategy)

        # Sort by score and trim to max size
        self.strategies[engine][key].sort(key=lambda x: x.get("score", 0), reverse=True)
        self.strategies[engine][key] = self.strategies[engine][key][:self.MAX_STRATEGIES_PER_KEY]

        self._save_database()
        return True

    def get_top_strategies(
        self,
        engine: str,
        asset: str,
        regime: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get top performing strategies for an engine/asset/regime.

        Args:
            engine: Engine name
            asset: Asset symbol
            regime: Market regime
            limit: Max strategies to return

        Returns:
            List of top strategies sorted by score
        """
        if engine not in self.strategies:
            return []

        key = self._get_key(asset, regime)
        strategies = self.strategies[engine].get(key, [])

        return strategies[:limit]

    def get_strategy_count(self, engine: str, asset: str, regime: str) -> int:
        """Get number of strategies for engine/asset/regime."""
        if engine not in self.strategies:
            return 0

        key = self._get_key(asset, regime)
        return len(self.strategies[engine].get(key, []))

    def should_exploit(self, engine: str, asset: str, regime: str) -> bool:
        """
        Determine if engine should exploit (use existing) or explore (generate new).

        Returns True if engine has enough strategies to exploit.
        """
        count = self.get_strategy_count(engine, asset, regime)
        return count >= self.MIN_STRATEGIES_FOR_EXPLOIT

    def select_strategy(
        self,
        engine: str,
        asset: str,
        regime: str,
        explore_probability: float = 0.2
    ) -> Optional[Dict]:
        """
        Select a strategy using 80/20 exploit/explore ratio.

        Args:
            engine: Engine name
            asset: Asset symbol
            regime: Market regime
            explore_probability: Chance to explore (generate new)

        Returns:
            Strategy dict if exploiting, None if should explore
        """
        import random

        if not self.should_exploit(engine, asset, regime):
            return None  # Must explore - not enough strategies

        if random.random() < explore_probability:
            return None  # Explore - generate new

        # Exploit - select from top strategies with weighted random
        strategies = self.get_top_strategies(engine, asset, regime, limit=20)
        if not strategies:
            return None

        # Weight by score (higher score = more likely to be selected)
        total_score = sum(s.get("score", 0.1) for s in strategies)
        if total_score == 0:
            return random.choice(strategies)

        r = random.random() * total_score
        cumulative = 0
        for s in strategies:
            cumulative += s.get("score", 0.1)
            if r <= cumulative:
                return s

        return strategies[0]  # Fallback to best

    def add_bred_strategy(
        self,
        engine: str,
        asset: str,
        regime: str,
        offspring: Dict,
        parent_ids: List[str]
    ) -> bool:
        """
        Add a bred (offspring) strategy to the database.

        Args:
            engine: Engine to receive the offspring
            asset: Asset symbol
            regime: Market regime
            offspring: Bred strategy dict
            parent_ids: List of parent strategy IDs

        Returns:
            True if added successfully
        """
        # Mark as bred with lineage
        offspring["parent_ids"] = parent_ids
        offspring["generation"] = max(
            self._get_generation(parent_id) for parent_id in parent_ids
        ) + 1 if parent_ids else 1
        offspring["is_bred"] = True

        return self.add_strategy(engine, asset, regime, offspring)

    def _get_generation(self, strategy_id: str) -> int:
        """Get generation number of a strategy."""
        for engine_data in self.strategies.values():
            for strategies in engine_data.values():
                for s in strategies:
                    if s.get("strategy_id") == strategy_id:
                        return s.get("generation", 0)
        return 0

    def get_engine_summary(self, engine: str) -> Dict:
        """Get summary stats for an engine's strategy library."""
        if engine not in self.strategies:
            return {"total": 0, "asset_regimes": 0}

        total = 0
        winning = 0
        for key, strategies in self.strategies[engine].items():
            total += len(strategies)
            winning += sum(1 for s in strategies if s.get("win_rate", 0) > 0.5)

        return {
            "total": total,
            "asset_regimes": len(self.strategies[engine]),
            "winning_strategies": winning,
            "win_ratio": winning / total if total > 0 else 0
        }

    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summary for all engines."""
        return {
            engine: self.get_engine_summary(engine)
            for engine in ["A", "B", "C", "D"]
        }

    # ==================== EDGE DECAY DETECTION ====================

    def check_edge_decay(self, strategy: Dict) -> Dict:
        """
        Check if a strategy's edge has decayed (recent performance poor).

        A strategy is considered "decayed" if its recent win rate is
        significantly below its historical performance.

        Args:
            strategy: Strategy dict with trade history

        Returns:
            {
                "is_decayed": bool,
                "recent_win_rate": float,
                "historical_win_rate": float,
                "recent_trades": int,
                "reason": str
            }
        """
        result = {
            "is_decayed": False,
            "recent_win_rate": 0.0,
            "historical_win_rate": strategy.get("win_rate", 0.5),
            "recent_trades": 0,
            "reason": ""
        }

        # Get trade history
        trades = strategy.get("trades", 0)
        recent_results = strategy.get("recent_results", [])

        # If not enough trades for decay check, assume healthy
        if trades < self.DECAY_LOOKBACK_TRADES:
            result["reason"] = f"Insufficient trades ({trades} < {self.DECAY_LOOKBACK_TRADES})"
            return result

        # Calculate recent win rate from last N trades
        recent_trades = recent_results[-self.DECAY_LOOKBACK_TRADES:]
        result["recent_trades"] = len(recent_trades)

        if not recent_trades:
            # No recent trade log, use overall stats as approximation
            # Estimate from wins/losses ratio
            wins = strategy.get("wins", 0)
            losses = strategy.get("losses", 0)
            total = wins + losses
            if total >= self.DECAY_LOOKBACK_TRADES:
                # If we have enough total trades but no recent log,
                # check if overall performance is declining
                overall_wr = wins / total if total > 0 else 0.5
                result["recent_win_rate"] = overall_wr
                if overall_wr < self.DECAY_WIN_RATE_THRESHOLD:
                    result["is_decayed"] = True
                    result["reason"] = f"Overall WR {overall_wr:.1%} < {self.DECAY_WIN_RATE_THRESHOLD:.1%}"
            return result

        # Count wins in recent trades
        recent_wins = sum(1 for t in recent_trades if t == "win")
        recent_wr = recent_wins / len(recent_trades) if recent_trades else 0.5
        result["recent_win_rate"] = recent_wr

        # Check if recent performance is below threshold
        if recent_wr < self.DECAY_WIN_RATE_THRESHOLD:
            result["is_decayed"] = True
            result["reason"] = f"Recent WR {recent_wr:.1%} < {self.DECAY_WIN_RATE_THRESHOLD:.1%} (last {len(recent_trades)} trades)"

        return result

    # ==================== REGIME-STRATEGY VALIDATION ====================

    def validate_regime_fit(self, strategy: Dict, current_regime: str) -> Dict:
        """
        Check if a strategy performs well in the current regime.

        Args:
            strategy: Strategy dict
            current_regime: Current market regime (TRENDING_UP, TRENDING_DOWN, etc.)

        Returns:
            {
                "is_valid": bool,
                "regime_win_rate": float,
                "regime_trades": int,
                "reason": str
            }
        """
        result = {
            "is_valid": True,  # Default to valid (optimistic)
            "regime_win_rate": 0.0,
            "regime_trades": 0,
            "reason": ""
        }

        # Get per-regime performance stats
        regime_stats = strategy.get("regime_performance", {})

        # If no regime-specific data, strategy is untested in this regime
        if current_regime not in regime_stats:
            result["reason"] = f"No data for regime {current_regime} - allowing exploration"
            return result

        regime_data = regime_stats[current_regime]
        trades = regime_data.get("trades", 0)
        wins = regime_data.get("wins", 0)

        result["regime_trades"] = trades

        # If not enough trades in this regime, allow exploration
        if trades < self.MIN_REGIME_TRADES:
            result["reason"] = f"Insufficient trades in {current_regime} ({trades} < {self.MIN_REGIME_TRADES})"
            return result

        # Calculate win rate in this regime
        regime_wr = wins / trades if trades > 0 else 0.5
        result["regime_win_rate"] = regime_wr

        # Check if performance in this regime is acceptable
        if regime_wr < self.REGIME_WIN_RATE_THRESHOLD:
            result["is_valid"] = False
            result["reason"] = f"Poor performance in {current_regime}: {regime_wr:.1%} < {self.REGIME_WIN_RATE_THRESHOLD:.1%}"

        return result

    def record_trade_result(
        self,
        engine: str,
        asset: str,
        regime: str,
        strategy_id: str,
        outcome: str  # "win" or "loss"
    ) -> bool:
        """
        Record a trade result and update strategy stats.

        This updates:
        - recent_results list (for decay detection)
        - regime_performance dict (for regime validation)

        Args:
            engine: Engine name (A, B, C, D)
            asset: Asset symbol
            regime: Market regime when trade was taken
            strategy_id: Strategy ID
            outcome: "win" or "loss"

        Returns:
            True if recorded successfully
        """
        if engine not in self.strategies:
            return False

        key = self._get_key(asset, regime)
        if key not in self.strategies[engine]:
            return False

        # Find the strategy
        for strategy in self.strategies[engine][key]:
            if strategy.get("strategy_id") == strategy_id:
                # Update recent results (for edge decay detection)
                if "recent_results" not in strategy:
                    strategy["recent_results"] = []
                strategy["recent_results"].append(outcome)
                # Keep only last 20 results
                strategy["recent_results"] = strategy["recent_results"][-20:]

                # Update regime-specific performance
                if "regime_performance" not in strategy:
                    strategy["regime_performance"] = {}
                if regime not in strategy["regime_performance"]:
                    strategy["regime_performance"][regime] = {"trades": 0, "wins": 0}

                strategy["regime_performance"][regime]["trades"] += 1
                if outcome == "win":
                    strategy["regime_performance"][regime]["wins"] += 1

                self._save_database()
                return True

        return False

    def select_strategy_with_validation(
        self,
        engine: str,
        asset: str,
        regime: str,
        explore_probability: float = 0.2
    ) -> Optional[Dict]:
        """
        Select a strategy with decay and regime validation.

        This is an enhanced version of select_strategy that:
        1. Checks for edge decay (skips decaying strategies)
        2. Validates regime fit (skips strategies that don't work in current regime)

        Args:
            engine: Engine name
            asset: Asset symbol
            regime: Current market regime
            explore_probability: Chance to explore (generate new)

        Returns:
            Strategy dict if valid one found, None if should explore
        """
        import random

        if not self.should_exploit(engine, asset, regime):
            logger.debug(f"[StrategyMemory] {engine}/{asset}/{regime}: Must explore (insufficient strategies)")
            return None  # Must explore

        if random.random() < explore_probability:
            logger.debug(f"[StrategyMemory] {engine}/{asset}/{regime}: Exploring (random)")
            return None  # Explore

        # Get top strategies
        strategies = self.get_top_strategies(engine, asset, regime, limit=20)
        if not strategies:
            return None

        # Filter out decayed and regime-unfit strategies
        valid_strategies = []
        for s in strategies:
            # Check decay
            decay_check = self.check_edge_decay(s)
            if decay_check["is_decayed"]:
                logger.info(f"[StrategyMemory] Skipping {s.get('strategy_id')}: {decay_check['reason']}")
                continue

            # Check regime fit
            regime_check = self.validate_regime_fit(s, regime)
            if not regime_check["is_valid"]:
                logger.info(f"[StrategyMemory] Skipping {s.get('strategy_id')}: {regime_check['reason']}")
                continue

            valid_strategies.append(s)

        if not valid_strategies:
            logger.info(f"[StrategyMemory] {engine}/{asset}/{regime}: All strategies filtered out - exploring")
            return None  # All filtered out, must explore

        # Weight by score and select
        total_score = sum(s.get("score", 0.1) for s in valid_strategies)
        if total_score == 0:
            return random.choice(valid_strategies)

        r = random.random() * total_score
        cumulative = 0
        for s in valid_strategies:
            cumulative += s.get("score", 0.1)
            if r <= cumulative:
                logger.info(f"[StrategyMemory] Selected {s.get('strategy_id')} (score: {s.get('score', 0):.2f})")
                return s

        return valid_strategies[0]


# ==================== SINGLETON PATTERN ====================

_strategy_memory: Optional[StrategyMemory] = None

def get_strategy_memory() -> StrategyMemory:
    """Get singleton instance of StrategyMemory."""
    global _strategy_memory
    if _strategy_memory is None:
        _strategy_memory = StrategyMemory()
    return _strategy_memory
