"""
HYDRA 3.0 - Tournament Manager (Layer 5)

Evolutionary competition system for strategies.

Tournament Rules:
- 24-hour elimination cycles (worst performers eliminated daily)
- 4-day breeding cycles (winners breed every 96 hours)
- Regime-specific performance tracking
- Winner teaching protocol (winners replace losers)
- Population size: 12-20 strategies per asset/regime

This is how HYDRA evolves: strategies compete, winners teach, losers die.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from loguru import logger
from dataclasses import dataclass, asdict
import statistics
import json
from pathlib import Path

from .config import HYDRA_DATA_DIR


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy in the tournament."""
    strategy_id: str
    gladiator: str
    wins: int
    losses: int
    total_trades: int
    win_rate: float
    avg_rr: float
    total_pnl_percent: float
    sharpe_ratio: float
    max_drawdown: float
    age_hours: float
    last_trade_timestamp: Optional[datetime]
    regime: str

    @property
    def fitness_score(self) -> float:
        """
        Calculate overall fitness score for ranking.

        Fitness = (Win Rate * 0.3) + (Sharpe * 0.4) + (Total PnL * 0.2) - (Max DD * 0.1)

        Emphasizes:
        - Sharpe ratio (40%) - risk-adjusted returns
        - Win rate (30%) - consistency
        - Total PnL (20%) - profitability
        - Max drawdown (10%, negative) - risk control
        """
        if self.total_trades < 3:
            # Not enough data - assign baseline fitness
            return 0.3

        # Normalize components to 0-1 scale
        wr_norm = self.win_rate  # Already 0-1
        sharpe_norm = min(max(self.sharpe_ratio / 3.0, 0), 1)  # Sharpe 3.0 = perfect
        pnl_norm = min(max(self.total_pnl_percent / 0.20, 0), 1)  # 20% gain = perfect
        dd_norm = min(max(self.max_drawdown / 0.15, 0), 1)  # 15% DD = worst

        fitness = (
            wr_norm * 0.3 +
            sharpe_norm * 0.4 +
            pnl_norm * 0.2 -
            dd_norm * 0.1
        )

        return max(fitness, 0.0)  # Never negative


class TournamentManager:
    """
    Manages evolutionary competition between strategies.

    24-Hour Cycle:
    - Track all strategy performance
    - Rank by fitness score
    - Eliminate bottom 20% (or worst performer if population < 10)
    - Log elimination reasons

    4-Day Cycle:
    - Identify top performers
    - Trigger breeding (crossover + mutation)
    - Add offspring to population
    - Cap population at 20 strategies per regime
    """

    # Tournament Configuration
    ELIMINATION_INTERVAL_HOURS = 24
    BREEDING_INTERVAL_HOURS = 96  # 4 days
    MIN_TRADES_FOR_ELIMINATION = 3  # Need 3+ trades to be evaluated
    MIN_POPULATION_SIZE = 8  # Don't eliminate below this
    MAX_POPULATION_SIZE = 20  # Cap population
    ELIMINATION_RATE = 0.20  # Eliminate bottom 20%

    # Performance Thresholds
    MIN_WIN_RATE = 0.45  # Below 45% win rate = immediate elimination
    MIN_SHARPE = -0.5  # Below -0.5 Sharpe = immediate elimination
    MAX_DRAWDOWN_THRESHOLD = 0.15  # Above 15% DD = immediate elimination

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = HYDRA_DATA_DIR
        self.data_dir = Path(data_dir)
        self.state_file = self.data_dir / "tournament_populations.json"

        self.populations: Dict[str, List[StrategyPerformance]] = {}
        # Key: "BTC-USD:TRENDING" -> List of strategies

        self.last_elimination: Dict[str, datetime] = {}
        self.last_breeding: Dict[str, datetime] = {}

        self.tournament_history: List[Dict] = []

        # Engine weights (synced from WeightAdjuster)
        self.engine_weights: Dict[str, float] = {
            "A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25
        }

        # Load existing state from disk
        self._load_state()

        logger.info(f"Tournament Manager initialized (populations: {len(self.populations)})")

    # ==================== PERSISTENCE ====================

    def _load_state(self):
        """Load tournament state from disk."""
        if not self.state_file.exists():
            logger.info("No existing tournament state found, starting fresh")
            return

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            # Restore populations
            for pop_key, strategies in data.get('populations', {}).items():
                self.populations[pop_key] = []
                for s in strategies:
                    # Convert dict back to StrategyPerformance
                    # Handle last_trade_timestamp conversion
                    last_ts = s.get('last_trade_timestamp')
                    if last_ts and isinstance(last_ts, str):
                        try:
                            s['last_trade_timestamp'] = datetime.fromisoformat(last_ts)
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Failed to parse last_trade_timestamp '{last_ts}': {e}")
                            s['last_trade_timestamp'] = None
                    perf = StrategyPerformance(**s)
                    self.populations[pop_key].append(perf)

            # Restore elimination timestamps
            for key, ts_str in data.get('last_elimination', {}).items():
                try:
                    self.last_elimination[key] = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse elimination timestamp for {key}: {e}")

            # Restore breeding timestamps
            for key, ts_str in data.get('last_breeding', {}).items():
                try:
                    self.last_breeding[key] = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse breeding timestamp for {key}: {e}")

            # Restore history (limited to last 100 entries)
            self.tournament_history = data.get('tournament_history', [])[-100:]

            # Restore engine weights
            saved_weights = data.get('engine_weights', {})
            for engine, weight in saved_weights.items():
                if engine in self.engine_weights:
                    self.engine_weights[engine] = weight

            logger.info(
                f"Loaded tournament state: {len(self.populations)} populations, "
                f"{sum(len(p) for p in self.populations.values())} total strategies"
            )

        except Exception as e:
            logger.error(f"Error loading tournament state: {e}")

    def _save_state(self):
        """Save tournament state to disk."""
        try:
            # Ensure directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Convert populations to serializable format
            populations_data = {}
            for pop_key, strategies in self.populations.items():
                populations_data[pop_key] = []
                for s in strategies:
                    s_dict = asdict(s)
                    # Convert datetime to string
                    if s_dict.get('last_trade_timestamp'):
                        s_dict['last_trade_timestamp'] = s_dict['last_trade_timestamp'].isoformat()
                    populations_data[pop_key].append(s_dict)

            # Convert datetime dicts
            last_elimination_data = {
                k: v.isoformat() for k, v in self.last_elimination.items()
            }
            last_breeding_data = {
                k: v.isoformat() for k, v in self.last_breeding.items()
            }

            # Process tournament history - convert datetime objects
            history_data = []
            for entry in self.tournament_history[-100:]:  # Limit to last 100
                entry_copy = entry.copy()
                if 'timestamp' in entry_copy and isinstance(entry_copy['timestamp'], datetime):
                    entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
                history_data.append(entry_copy)

            data = {
                'populations': populations_data,
                'last_elimination': last_elimination_data,
                'last_breeding': last_breeding_data,
                'tournament_history': history_data,
                'engine_weights': self.engine_weights,
                'saved_at': datetime.now(timezone.utc).isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved tournament state to {self.state_file}")

        except Exception as e:
            logger.error(f"Error saving tournament state: {e}")

    # ==================== WEIGHT MANAGEMENT ====================

    def update_weight(self, engine: str, weight: float) -> None:
        """
        Update weight for a specific engine.

        Called by WeightAdjuster to sync weights to tournament manager.

        Args:
            engine: Engine name (A, B, C, or D)
            weight: New weight value (0.0 to 1.0)
        """
        if engine not in self.engine_weights:
            logger.warning(f"Unknown engine: {engine}")
            return

        old_weight = self.engine_weights[engine]
        self.engine_weights[engine] = max(0.05, min(0.50, weight))  # Bound to 5-50%

        if abs(old_weight - weight) > 0.01:  # Only log significant changes
            logger.info(f"Engine {engine} weight updated: {old_weight:.2%} → {weight:.2%}")

        self._save_state()

    def get_weights(self) -> Dict[str, float]:
        """Get current engine weights."""
        return self.engine_weights.copy()

    # ==================== POPULATION MANAGEMENT ====================

    def register_strategy(
        self,
        strategy_id: str,
        gladiator: str,
        asset: str,
        regime: str
    ):
        """Register new strategy in tournament."""
        population_key = f"{asset}:{regime}"

        if population_key not in self.populations:
            self.populations[population_key] = []

        # Create initial performance record
        perf = StrategyPerformance(
            strategy_id=strategy_id,
            gladiator=gladiator,
            wins=0,
            losses=0,
            total_trades=0,
            win_rate=0.5,  # Neutral prior
            avg_rr=1.0,
            total_pnl_percent=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            age_hours=0.0,
            last_trade_timestamp=None,
            regime=regime
        )

        self.populations[population_key].append(perf)

        logger.info(
            f"Strategy {strategy_id} registered in {population_key} "
            f"(population: {len(self.populations[population_key])})"
        )

        # Persist state
        self._save_state()

    def update_strategy_performance(
        self,
        strategy_id: str,
        asset: str,
        regime: str,
        trade_result: Dict
    ):
        """
        Update strategy performance after trade closes.

        Args:
            trade_result: {
                "outcome": "win" | "loss",
                "pnl_percent": 0.02,
                "rr_actual": 1.5,
                "entry_timestamp": datetime,
                "exit_timestamp": datetime
            }
        """
        population_key = f"{asset}:{regime}"

        if population_key not in self.populations:
            logger.warning(f"Population {population_key} not found")
            return

        # Find strategy
        strategy = next(
            (s for s in self.populations[population_key] if s.strategy_id == strategy_id),
            None
        )

        if not strategy:
            logger.warning(f"Strategy {strategy_id} not found in {population_key}")
            return

        # Update metrics
        strategy.total_trades += 1
        if trade_result["outcome"] == "win":
            strategy.wins += 1
        else:
            strategy.losses += 1

        strategy.win_rate = strategy.wins / strategy.total_trades if strategy.total_trades > 0 else 0.5

        # Update PnL
        strategy.total_pnl_percent += trade_result["pnl_percent"]

        # Update R:R
        if strategy.avg_rr == 1.0:
            strategy.avg_rr = trade_result["rr_actual"]
        else:
            # Exponential moving average
            strategy.avg_rr = strategy.avg_rr * 0.8 + trade_result["rr_actual"] * 0.2

        # Update max drawdown
        if trade_result["pnl_percent"] < 0:
            potential_dd = abs(trade_result["pnl_percent"])
            strategy.max_drawdown = max(strategy.max_drawdown, potential_dd)

        strategy.last_trade_timestamp = trade_result["exit_timestamp"]

        # Calculate Sharpe ratio (simplified)
        strategy.sharpe_ratio = self._calculate_sharpe(strategy)

        logger.info(
            f"Updated {strategy_id}: {strategy.total_trades} trades, "
            f"{strategy.win_rate:.1%} WR, {strategy.sharpe_ratio:.2f} Sharpe"
        )

        # Persist state
        self._save_state()

    def _calculate_sharpe(self, strategy: StrategyPerformance) -> float:
        """
        Calculate simplified Sharpe ratio.

        Sharpe = (Avg Return - Risk Free Rate) / Std Dev of Returns

        For simplicity:
        - Assume risk-free rate = 0
        - Use win rate and R:R to estimate return distribution
        """
        if strategy.total_trades < 3:
            return 0.0

        # Expected return per trade
        avg_win = strategy.avg_rr * 0.01  # Assume 1% risk per trade
        avg_loss = -0.01

        expected_return = (strategy.win_rate * avg_win) + ((1 - strategy.win_rate) * avg_loss)

        # Estimate standard deviation (simplified)
        # Variance = p(win) * (win - E)^2 + p(loss) * (loss - E)^2
        variance = (
            strategy.win_rate * (avg_win - expected_return) ** 2 +
            (1 - strategy.win_rate) * (avg_loss - expected_return) ** 2
        )

        std_dev = variance ** 0.5

        if std_dev == 0:
            return 0.0

        sharpe = expected_return / std_dev

        return sharpe

    # ==================== ELIMINATION CYCLE (24 HOURS) ====================

    def run_elimination_cycle(
        self,
        asset: str,
        regime: str,
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Run 24-hour elimination cycle.

        Returns:
            {
                "eliminated": [strategy_ids...],
                "survivors": [strategy_ids...],
                "elimination_reasons": {strategy_id: reason},
                "population_before": 15,
                "population_after": 12
            }
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        population_key = f"{asset}:{regime}"

        # Check if it's time for elimination
        last_elim = self.last_elimination.get(population_key)
        if last_elim:
            hours_since = (current_time - last_elim).total_seconds() / 3600
            if hours_since < self.ELIMINATION_INTERVAL_HOURS:
                return {
                    "eliminated": [],
                    "reason": f"Too soon (last: {hours_since:.1f}h ago)"
                }

        if population_key not in self.populations:
            return {"eliminated": [], "reason": "Population not found"}

        population = self.populations[population_key]

        if len(population) <= self.MIN_POPULATION_SIZE:
            logger.info(f"Population {population_key} at minimum size, skipping elimination")
            return {"eliminated": [], "reason": "Population at minimum"}

        # Separate strategies: evaluated vs unevaluated
        evaluated = [s for s in population if s.total_trades >= self.MIN_TRADES_FOR_ELIMINATION]
        unevaluated = [s for s in population if s.total_trades < self.MIN_TRADES_FOR_ELIMINATION]

        if not evaluated:
            logger.info(f"No strategies with {self.MIN_TRADES_FOR_ELIMINATION}+ trades yet")
            return {"eliminated": [], "reason": "No evaluated strategies"}

        # Immediate eliminations (catastrophic failures)
        immediate_eliminations = []
        elimination_reasons = {}

        for strategy in evaluated:
            if strategy.win_rate < self.MIN_WIN_RATE:
                immediate_eliminations.append(strategy.strategy_id)
                elimination_reasons[strategy.strategy_id] = f"Win rate too low: {strategy.win_rate:.1%}"
            elif strategy.sharpe_ratio < self.MIN_SHARPE:
                immediate_eliminations.append(strategy.strategy_id)
                elimination_reasons[strategy.strategy_id] = f"Sharpe too low: {strategy.sharpe_ratio:.2f}"
            elif strategy.max_drawdown > self.MAX_DRAWDOWN_THRESHOLD:
                immediate_eliminations.append(strategy.strategy_id)
                elimination_reasons[strategy.strategy_id] = f"Max DD too high: {strategy.max_drawdown:.1%}"

        # Rank remaining by fitness
        remaining = [s for s in evaluated if s.strategy_id not in immediate_eliminations]
        ranked = sorted(remaining, key=lambda s: s.fitness_score, reverse=True)

        # Eliminate bottom 20%
        num_to_eliminate = max(1, int(len(ranked) * self.ELIMINATION_RATE))
        bottom_performers = ranked[-num_to_eliminate:]

        for strategy in bottom_performers:
            immediate_eliminations.append(strategy.strategy_id)
            elimination_reasons[strategy.strategy_id] = (
                f"Bottom {self.ELIMINATION_RATE:.0%} (fitness: {strategy.fitness_score:.2f})"
            )

        # Remove eliminated strategies
        population_before = len(population)
        self.populations[population_key] = [
            s for s in population if s.strategy_id not in immediate_eliminations
        ]
        population_after = len(self.populations[population_key])

        # Log elimination
        self.last_elimination[population_key] = current_time

        result = {
            "eliminated": immediate_eliminations,
            "survivors": [s.strategy_id for s in self.populations[population_key]],
            "elimination_reasons": elimination_reasons,
            "population_before": population_before,
            "population_after": population_after,
            "timestamp": current_time
        }

        self.tournament_history.append({
            "event": "elimination",
            "population_key": population_key,
            **result
        })

        logger.warning(
            f"Elimination cycle completed for {population_key}: "
            f"{len(immediate_eliminations)} eliminated, {population_after} remain"
        )

        # Persist state
        self._save_state()

        return result

    # ==================== BREEDING CYCLE (4 DAYS) ====================

    def run_breeding_cycle(
        self,
        asset: str,
        regime: str,
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Run 4-day breeding cycle.

        Returns:
            {
                "top_performers": [strategy_ids...],
                "breeding_pairs": [(parent1, parent2), ...],
                "offspring_count": 3,
                "population_after": 18
            }
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        population_key = f"{asset}:{regime}"

        # Check if it's time for breeding
        last_breed = self.last_breeding.get(population_key)
        if last_breed:
            hours_since = (current_time - last_breed).total_seconds() / 3600
            if hours_since < self.BREEDING_INTERVAL_HOURS:
                return {
                    "bred": False,
                    "reason": f"Too soon (last: {hours_since:.1f}h ago)"
                }

        if population_key not in self.populations:
            return {"bred": False, "reason": "Population not found"}

        population = self.populations[population_key]

        # Only breed if we have evaluated strategies
        evaluated = [s for s in population if s.total_trades >= self.MIN_TRADES_FOR_ELIMINATION]

        if len(evaluated) < 4:
            return {"bred": False, "reason": "Need 4+ evaluated strategies"}

        # Rank by fitness
        ranked = sorted(evaluated, key=lambda s: s.fitness_score, reverse=True)

        # Select top 4 as potential parents
        top_performers = ranked[:4]

        # Breeding pairs: top 2 strategies breed together
        parent1 = top_performers[0]
        parent2 = top_performers[1]

        breeding_pairs = [(parent1.strategy_id, parent2.strategy_id)]

        # Cap population
        if len(population) >= self.MAX_POPULATION_SIZE:
            logger.info(f"Population {population_key} at max size, skipping breeding")
            return {"bred": False, "reason": "Population at max"}

        # Mark breeding cycle complete
        self.last_breeding[population_key] = current_time

        result = {
            "bred": True,
            "top_performers": [s.strategy_id for s in top_performers],
            "breeding_pairs": breeding_pairs,
            "parent1_fitness": parent1.fitness_score,
            "parent2_fitness": parent2.fitness_score,
            "population_before": len(population),
            "timestamp": current_time
        }

        self.tournament_history.append({
            "event": "breeding",
            "population_key": population_key,
            **result
        })

        logger.success(
            f"Breeding cycle for {population_key}: "
            f"{parent1.strategy_id} (fitness {parent1.fitness_score:.2f}) x "
            f"{parent2.strategy_id} (fitness {parent2.fitness_score:.2f})"
        )

        # Persist state
        self._save_state()

        return result

    # ==================== ANALYSIS & REPORTING ====================

    def get_population_stats(self, asset: str, regime: str) -> Dict:
        """Get statistics for a population."""
        population_key = f"{asset}:{regime}"

        if population_key not in self.populations:
            return {"error": "Population not found"}

        population = self.populations[population_key]

        if not population:
            return {"population_size": 0}

        evaluated = [s for s in population if s.total_trades >= self.MIN_TRADES_FOR_ELIMINATION]

        if not evaluated:
            return {
                "population_size": len(population),
                "evaluated_strategies": 0,
                "unevaluated_strategies": len(population)
            }

        fitness_scores = [s.fitness_score for s in evaluated]
        win_rates = [s.win_rate for s in evaluated]
        sharpes = [s.sharpe_ratio for s in evaluated]

        return {
            "population_size": len(population),
            "evaluated_strategies": len(evaluated),
            "unevaluated_strategies": len(population) - len(evaluated),
            "avg_fitness": statistics.mean(fitness_scores),
            "max_fitness": max(fitness_scores),
            "min_fitness": min(fitness_scores),
            "avg_win_rate": statistics.mean(win_rates),
            "avg_sharpe": statistics.mean(sharpes),
            "top_strategy": max(evaluated, key=lambda s: s.fitness_score).strategy_id
        }

    def get_leaderboard(
        self,
        asset: str,
        regime: str,
        top_n: int = 10
    ) -> List[Dict]:
        """Get top N strategies by fitness."""
        population_key = f"{asset}:{regime}"

        if population_key not in self.populations:
            return []

        population = self.populations[population_key]
        evaluated = [s for s in population if s.total_trades >= self.MIN_TRADES_FOR_ELIMINATION]

        ranked = sorted(evaluated, key=lambda s: s.fitness_score, reverse=True)

        leaderboard = []
        for rank, strategy in enumerate(ranked[:top_n], 1):
            leaderboard.append({
                "rank": rank,
                "strategy_id": strategy.strategy_id,
                "gladiator": strategy.gladiator,
                "fitness_score": round(strategy.fitness_score, 3),
                "win_rate": round(strategy.win_rate, 3),
                "sharpe_ratio": round(strategy.sharpe_ratio, 2),
                "total_pnl": round(strategy.total_pnl_percent, 4),
                "total_trades": strategy.total_trades
            })

        return leaderboard


# Global singleton instance
_tournament_manager = None

def get_tournament_manager() -> TournamentManager:
    """Get global TournamentManager singleton."""
    global _tournament_manager
    if _tournament_manager is None:
        _tournament_manager = TournamentManager()
    return _tournament_manager
