"""
HYDRA 3.0 - Breeding Cycle (4-Day Evolution)

Every 4 days, the system breeds new strategies:
1. Check if winner qualifies (100+ trades, WR>60%, Sharpe>1.5)
2. Combine #1's entry logic + #2's exit logic
3. Give child strategy to #4 to test
4. If child beats parents → parents must evolve

This creates strategic diversity through genetic combination.

Phase 2, Week 2 - Step 13
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
import json
import uuid


@dataclass
class QualificationCriteria:
    """Criteria for breeding qualification"""
    min_trades: int = 100
    min_win_rate: float = 0.60  # 60%
    min_sharpe: float = 1.5


@dataclass
class BreedingEvent:
    """Record of a breeding cycle execution"""
    breeding_id: str
    timestamp: datetime
    parent1_engine: str
    parent1_pnl: float
    parent1_win_rate: float
    parent1_trades: int
    parent2_engine: str
    parent2_pnl: float
    parent2_win_rate: float
    parent2_trades: int
    offspring_id: str
    offspring_assigned_to: str
    qualification_met: bool
    cycle_number: int


@dataclass
class OffspringStrategy:
    """Offspring strategy created from breeding"""
    strategy_id: str
    parent1_engine: str
    parent2_engine: str
    entry_logic_from: str  # Which parent provided entry
    exit_logic_from: str   # Which parent provided exit
    combined_insights: List[str]
    assigned_to: str
    created_at: datetime
    performance_target: float  # Must beat this to survive
    status: str = "TESTING"  # TESTING, PROMOTED, FAILED


class BreedingCycle:
    """
    Breeding Cycle Manager - 4-day evolution mechanism.

    Every 4 days:
    1. Check if winner (#1) qualifies for breeding
    2. Extract entry logic from #1, exit logic from #2
    3. Combine into offspring strategy
    4. Assign offspring to #4 for testing
    5. If offspring beats parents → promote, else discard
    """

    BREEDING_INTERVAL_DAYS = 4

    # Qualification criteria
    MIN_TRADES_FOR_BREEDING = 100
    MIN_WIN_RATE_FOR_BREEDING = 0.60
    MIN_SHARPE_FOR_BREEDING = 1.5

    def __init__(self, data_dir: Path = Path("/root/crpbot/data/hydra")):
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Breeding history
        self.breeding_history: List[BreedingEvent] = []
        self.breeding_count = 0
        self.last_breeding_time: Optional[datetime] = None

        # Active offspring being tested
        self.active_offspring: Dict[str, OffspringStrategy] = {}

        # Persistence
        self.state_file = self.data_dir / "breeding_cycle_state.json"
        self.offspring_file = self.data_dir / "offspring_strategies.jsonl"
        self.breeding_log = self.data_dir / "breeding_history.jsonl"

        # Load existing state
        self._load_state()

        logger.info(f"[BreedingCycle] Initialized. Total breedings: {self.breeding_count}")

    def should_execute_breeding(self, rankings: List[tuple]) -> bool:
        """
        Check if breeding cycle should execute.

        Conditions:
        1. 4+ days since last breeding
        2. Winner qualifies (trades, WR, Sharpe)
        3. Rankings available

        Args:
            rankings: List of (engine_name, stats) from TournamentManager

        Returns:
            True if breeding should execute
        """
        now = datetime.now(timezone.utc)

        # Check time since last breeding
        if self.last_breeding_time is not None:
            days_since = (now - self.last_breeding_time).days
            if days_since < self.BREEDING_INTERVAL_DAYS:
                logger.debug(
                    f"[BreedingCycle] Not time yet. Days since last: {days_since}, "
                    f"Need: {self.BREEDING_INTERVAL_DAYS}"
                )
                return False

        # Check if we have rankings
        if not rankings or len(rankings) < 4:
            logger.debug("[BreedingCycle] Not enough engines for breeding")
            return False

        # Check if winner qualifies
        winner_name, winner_stats = rankings[0]
        qualifies, reason = self._check_qualification(winner_stats)

        if not qualifies:
            logger.debug(f"[BreedingCycle] Winner {winner_name} doesn't qualify: {reason}")
            return False

        logger.info("[BreedingCycle] Breeding conditions MET - ready to execute")
        return True

    def _check_qualification(self, stats) -> tuple:
        """
        Check if engine stats qualify for breeding.

        Returns:
            (qualifies: bool, reason: str)
        """
        # Check trade count
        if stats.closed_trades < self.MIN_TRADES_FOR_BREEDING:
            return False, f"Not enough trades ({stats.closed_trades} < {self.MIN_TRADES_FOR_BREEDING})"

        # Check win rate
        if stats.win_rate < self.MIN_WIN_RATE_FOR_BREEDING:
            return False, f"Win rate too low ({stats.win_rate:.1%} < {self.MIN_WIN_RATE_FOR_BREEDING:.0%})"

        # Check Sharpe ratio (if available)
        sharpe = stats.sharpe_ratio if stats.sharpe_ratio is not None else 0
        if sharpe < self.MIN_SHARPE_FOR_BREEDING:
            return False, f"Sharpe too low ({sharpe:.2f} < {self.MIN_SHARPE_FOR_BREEDING})"

        return True, "Qualifies for breeding"

    def execute_breeding(
        self,
        rankings: List[tuple],
        get_portfolio_fn,
        get_engine_fn
    ) -> Optional[BreedingEvent]:
        """
        Execute the breeding cycle.

        Args:
            rankings: List of (engine_name, stats) sorted by rank
            get_portfolio_fn: Function to get portfolio by engine name
            get_engine_fn: Function to get engine instance by name

        Returns:
            BreedingEvent if breeding executed, None otherwise
        """
        if not self.should_execute_breeding(rankings):
            return None

        self.breeding_count += 1
        now = datetime.now(timezone.utc)

        # Get parents (#1 and #2) and recipient (#4)
        parent1_name, parent1_stats = rankings[0]
        parent2_name, parent2_stats = rankings[1]
        recipient_name, recipient_stats = rankings[-1]

        logger.info(f"\n{'='*60}")
        logger.info(f"🧬 BREEDING CYCLE #{self.breeding_count} EXECUTING 🧬")
        logger.info(f"{'='*60}")
        logger.info(f"Parent #1: Engine {parent1_name} (Rank #1)")
        logger.info(f"  - P&L: ${parent1_stats.total_pnl_usd:+.2f}")
        logger.info(f"  - Win Rate: {parent1_stats.win_rate:.1%}")
        logger.info(f"  - Trades: {parent1_stats.total_trades}")
        logger.info(f"Parent #2: Engine {parent2_name} (Rank #2)")
        logger.info(f"  - P&L: ${parent2_stats.total_pnl_usd:+.2f}")
        logger.info(f"  - Win Rate: {parent2_stats.win_rate:.1%}")
        logger.info(f"  - Trades: {parent2_stats.total_trades}")
        logger.info(f"Recipient: Engine {recipient_name} (Rank #4)")
        logger.info(f"{'='*60}\n")

        # Step 1: Extract entry logic from parent #1
        parent1_portfolio = get_portfolio_fn(parent1_name)
        entry_insights = self._extract_entry_logic(parent1_name, parent1_portfolio)

        # Step 2: Extract exit logic from parent #2
        parent2_portfolio = get_portfolio_fn(parent2_name)
        exit_insights = self._extract_exit_logic(parent2_name, parent2_portfolio)

        # Step 3: Create offspring strategy
        offspring = self._create_offspring(
            parent1_name, entry_insights,
            parent2_name, exit_insights,
            recipient_name
        )

        # Step 4: Assign offspring to #4
        recipient_engine = get_engine_fn(recipient_name)
        self._assign_offspring(recipient_engine, offspring)

        # Step 5: Set performance target (must beat average of parents)
        performance_target = (parent1_stats.total_pnl_usd + parent2_stats.total_pnl_usd) / 2
        offspring.performance_target = performance_target

        # Store offspring
        self.active_offspring[offspring.strategy_id] = offspring
        self._save_offspring(offspring)

        # Create breeding event
        breeding_event = BreedingEvent(
            breeding_id=str(uuid.uuid4())[:8],
            timestamp=now,
            parent1_engine=parent1_name,
            parent1_pnl=parent1_stats.total_pnl_usd,
            parent1_win_rate=parent1_stats.win_rate,
            parent1_trades=parent1_stats.total_trades,
            parent2_engine=parent2_name,
            parent2_pnl=parent2_stats.total_pnl_usd,
            parent2_win_rate=parent2_stats.win_rate,
            parent2_trades=parent2_stats.total_trades,
            offspring_id=offspring.strategy_id,
            offspring_assigned_to=recipient_name,
            qualification_met=True,
            cycle_number=self.breeding_count
        )

        # Save event
        self.breeding_history.append(breeding_event)
        self.last_breeding_time = now
        self._save_state()
        self._log_breeding_event(breeding_event)

        logger.success(f"\n🧬 Breeding complete. Offspring {offspring.strategy_id} assigned to Engine {recipient_name}")
        logger.success(f"   Entry logic from: Engine {parent1_name}")
        logger.success(f"   Exit logic from: Engine {parent2_name}")
        logger.success(f"   Performance target: ${performance_target:+.2f}")

        return breeding_event

    def _extract_entry_logic(self, engine: str, portfolio) -> List[str]:
        """
        Extract entry logic insights from winner's successful trades.

        Focuses on:
        - Entry timing patterns
        - Direction decisions
        - Entry conditions that led to wins
        """
        insights = []

        closed_trades = portfolio.get_closed_trades()
        winning_trades = [t for t in closed_trades if t.outcome == "win"]

        if not winning_trades:
            return ["No winning entry patterns detected"]

        # Analyze entry price patterns
        avg_entry_price = sum(t.entry_price for t in winning_trades) / len(winning_trades)

        # Analyze direction success
        long_wins = [t for t in winning_trades if t.direction == "BUY"]
        short_wins = [t for t in winning_trades if t.direction == "SELL"]

        if long_wins:
            long_avg_gain = sum(t.pnl_percent for t in long_wins) / len(long_wins)
            insights.append(f"LONG entries: {len(long_wins)} wins, avg {long_avg_gain:.2f}% gain")

        if short_wins:
            short_avg_gain = sum(t.pnl_percent for t in short_wins) / len(short_wins)
            insights.append(f"SHORT entries: {len(short_wins)} wins, avg {short_avg_gain:.2f}% gain")

        # Dominant direction
        if len(long_wins) > len(short_wins) * 1.5:
            insights.append("Dominant strategy: LONG bias (favors uptrends)")
        elif len(short_wins) > len(long_wins) * 1.5:
            insights.append("Dominant strategy: SHORT bias (favors downtrends)")
        else:
            insights.append("Balanced strategy: Both directions effective")

        # Entry timing insights
        insights.append(f"Entry pattern based on {len(winning_trades)} winning trades")

        logger.info(f"[BreedingCycle] Extracted {len(insights)} entry insights from Engine {engine}")
        return insights

    def _extract_exit_logic(self, engine: str, portfolio) -> List[str]:
        """
        Extract exit logic insights from parent's trades.

        Focuses on:
        - Stop loss placement
        - Take profit placement
        - Exit timing
        - Risk/reward ratios
        """
        insights = []

        closed_trades = portfolio.get_closed_trades()
        winning_trades = [t for t in closed_trades if t.outcome == "win"]
        losing_trades = [t for t in closed_trades if t.outcome == "loss"]

        if not closed_trades:
            return ["No exit patterns detected"]

        # Analyze SL/TP ratios
        rr_ratios = []
        for t in winning_trades:
            if t.entry_price > 0 and t.stop_loss > 0 and t.take_profit > 0:
                risk = abs(t.entry_price - t.stop_loss)
                reward = abs(t.take_profit - t.entry_price)
                if risk > 0:
                    rr_ratios.append(reward / risk)

        if rr_ratios:
            avg_rr = sum(rr_ratios) / len(rr_ratios)
            insights.append(f"Average R:R ratio: {avg_rr:.2f}:1")

            if avg_rr >= 2.0:
                insights.append("High R:R strategy: Targets 2:1+ reward/risk")
            elif avg_rr >= 1.5:
                insights.append("Moderate R:R strategy: Targets 1.5:1 reward/risk")
            else:
                insights.append("Tight R:R strategy: Targets quick wins")

        # Exit reason analysis
        if winning_trades:
            tp_exits = [t for t in winning_trades if t.exit_reason == "take_profit"]
            insights.append(f"TP hit rate: {len(tp_exits)}/{len(winning_trades)} winning trades")

        if losing_trades:
            sl_exits = [t for t in losing_trades if t.exit_reason == "stop_loss"]
            insights.append(f"SL discipline: {len(sl_exits)}/{len(losing_trades)} losses via SL")

        # Average holding insights (if exit_time available)
        insights.append(f"Exit pattern based on {len(closed_trades)} total trades")

        logger.info(f"[BreedingCycle] Extracted {len(insights)} exit insights from Engine {engine}")
        return insights

    def _create_offspring(
        self,
        parent1: str,
        entry_insights: List[str],
        parent2: str,
        exit_insights: List[str],
        recipient: str
    ) -> OffspringStrategy:
        """
        Create offspring strategy combining parent insights.
        """
        now = datetime.now(timezone.utc)
        strategy_id = f"OFFSPRING_{self.breeding_count:04d}_{now.strftime('%Y%m%d')}"

        # Combine insights
        combined = []
        combined.append(f"=== ENTRY LOGIC (from Engine {parent1}) ===")
        combined.extend(entry_insights)
        combined.append(f"\n=== EXIT LOGIC (from Engine {parent2}) ===")
        combined.extend(exit_insights)
        combined.append(f"\n=== BREEDING DIRECTIVE ===")
        combined.append(f"Use {parent1}'s entry timing with {parent2}'s exit discipline")
        combined.append("This hybrid should capture entries better while protecting profits")

        offspring = OffspringStrategy(
            strategy_id=strategy_id,
            parent1_engine=parent1,
            parent2_engine=parent2,
            entry_logic_from=parent1,
            exit_logic_from=parent2,
            combined_insights=combined,
            assigned_to=recipient,
            created_at=now,
            performance_target=0.0,  # Set later
            status="TESTING"
        )

        return offspring

    def _assign_offspring(self, engine, offspring: OffspringStrategy):
        """
        Assign offspring strategy to recipient engine.

        Updates engine state with breeding instructions.
        """
        # Store offspring reference
        if hasattr(engine, 'active_offspring'):
            engine.active_offspring = offspring.strategy_id

        # Create breeding guidance
        guidance = f"""
SYSTEM NOTICE: You have been assigned a BRED OFFSPRING strategy.

Strategy ID: {offspring.strategy_id}
Entry Logic Parent: Engine {offspring.parent1_engine} (Rank #1)
Exit Logic Parent: Engine {offspring.parent2_engine} (Rank #2)

INHERITED INSIGHTS:
"""
        for insight in offspring.combined_insights:
            guidance += f"\n{insight}"

        guidance += f"""

MISSION:
- Test this hybrid strategy for the next breeding cycle (4 days)
- Performance target: ${offspring.performance_target:+.2f}
- If you beat this target, the offspring is PROMOTED
- If you fail, the offspring is DISCARDED and you return to your own strategy

FOCUS:
- Use the entry patterns from Engine {offspring.parent1_engine}
- Use the exit discipline from Engine {offspring.parent2_engine}
- Your job is to prove this combination works
"""

        # Store guidance
        if hasattr(engine, 'breeding_guidance'):
            engine.breeding_guidance = guidance

        # Mark engine as testing offspring
        if hasattr(engine, 'meta_state'):
            engine.meta_state['testing_offspring'] = True
            engine.meta_state['offspring_id'] = offspring.strategy_id
            engine.meta_state['offspring_assigned_at'] = offspring.created_at.isoformat()

        logger.info(f"[BreedingCycle] Offspring assigned to Engine {offspring.assigned_to}")

    def evaluate_offspring(self, offspring_id: str, actual_pnl: float) -> str:
        """
        Evaluate if offspring beat its target.

        Args:
            offspring_id: ID of offspring to evaluate
            actual_pnl: Actual P&L achieved during testing

        Returns:
            "PROMOTED" if successful, "FAILED" otherwise
        """
        if offspring_id not in self.active_offspring:
            logger.warning(f"[BreedingCycle] Offspring {offspring_id} not found")
            return "UNKNOWN"

        offspring = self.active_offspring[offspring_id]

        if actual_pnl >= offspring.performance_target:
            offspring.status = "PROMOTED"
            logger.success(
                f"[BreedingCycle] 🎉 Offspring {offspring_id} PROMOTED! "
                f"P&L: ${actual_pnl:+.2f} >= Target: ${offspring.performance_target:+.2f}"
            )
        else:
            offspring.status = "FAILED"
            logger.warning(
                f"[BreedingCycle] ❌ Offspring {offspring_id} FAILED. "
                f"P&L: ${actual_pnl:+.2f} < Target: ${offspring.performance_target:+.2f}"
            )

        # Update stored offspring
        self._save_offspring(offspring)

        return offspring.status

    def _save_offspring(self, offspring: OffspringStrategy):
        """Save offspring strategy to file."""
        with open(self.offspring_file, 'a') as f:
            offspring_dict = {
                "strategy_id": offspring.strategy_id,
                "parent1_engine": offspring.parent1_engine,
                "parent2_engine": offspring.parent2_engine,
                "entry_logic_from": offspring.entry_logic_from,
                "exit_logic_from": offspring.exit_logic_from,
                "combined_insights": offspring.combined_insights,
                "assigned_to": offspring.assigned_to,
                "created_at": offspring.created_at.isoformat(),
                "performance_target": offspring.performance_target,
                "status": offspring.status
            }
            f.write(json.dumps(offspring_dict) + '\n')

    def _log_breeding_event(self, event: BreedingEvent):
        """Log breeding event to history file."""
        with open(self.breeding_log, 'a') as f:
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()
            f.write(json.dumps(event_dict) + '\n')

    def _save_state(self):
        """Save breeding cycle state to disk."""
        state = {
            "breeding_count": self.breeding_count,
            "last_breeding_time": self.last_breeding_time.isoformat() if self.last_breeding_time else None,
            "active_offspring": list(self.active_offspring.keys()),
            "recent_breedings": [
                {
                    "breeding_id": b.breeding_id,
                    "timestamp": b.timestamp.isoformat(),
                    "parent1": b.parent1_engine,
                    "parent2": b.parent2_engine,
                    "offspring_id": b.offspring_id,
                    "assigned_to": b.offspring_assigned_to
                }
                for b in self.breeding_history[-10:]
            ]
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load breeding cycle state from disk."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.breeding_count = state.get("breeding_count", 0)
            last_breeding = state.get("last_breeding_time")
            if last_breeding:
                self.last_breeding_time = datetime.fromisoformat(last_breeding)

            logger.debug(f"[BreedingCycle] Loaded state: {self.breeding_count} breedings")

        except Exception as e:
            logger.warning(f"[BreedingCycle] Failed to load state: {e}")

    def get_breeding_summary(self) -> Dict[str, Any]:
        """Get summary of breeding cycle history."""
        return {
            "total_breedings": self.breeding_count,
            "last_breeding": self.last_breeding_time.isoformat() if self.last_breeding_time else None,
            "days_since_last": (
                (datetime.now(timezone.utc) - self.last_breeding_time).days
                if self.last_breeding_time else None
            ),
            "next_breeding_in_days": (
                max(0, self.BREEDING_INTERVAL_DAYS -
                    (datetime.now(timezone.utc) - self.last_breeding_time).days)
                if self.last_breeding_time else 0
            ),
            "active_offspring": len(self.active_offspring),
            "qualification_criteria": {
                "min_trades": self.MIN_TRADES_FOR_BREEDING,
                "min_win_rate": self.MIN_WIN_RATE_FOR_BREEDING,
                "min_sharpe": self.MIN_SHARPE_FOR_BREEDING
            }
        }

    def get_offspring_status(self) -> List[Dict]:
        """Get status of all active offspring."""
        return [
            {
                "strategy_id": o.strategy_id,
                "assigned_to": o.assigned_to,
                "parent1": o.parent1_engine,
                "parent2": o.parent2_engine,
                "performance_target": o.performance_target,
                "status": o.status,
                "created_at": o.created_at.isoformat()
            }
            for o in self.active_offspring.values()
        ]


# ==================== SINGLETON PATTERN ====================

_breeding_cycle: Optional[BreedingCycle] = None

def get_breeding_cycle() -> BreedingCycle:
    """Get singleton instance of BreedingCycle."""
    global _breeding_cycle
    if _breeding_cycle is None:
        _breeding_cycle = BreedingCycle()
    return _breeding_cycle
