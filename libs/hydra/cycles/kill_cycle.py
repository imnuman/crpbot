"""
HYDRA 3.0 - Kill Cycle (24-Hour Evolution)

Every 24 hours, the weakest engine (#4) is "killed":
1. Strategy completely deleted
2. Forced to learn from winner (#1)
3. Must invent NEW strategy from scratch
4. History archived to edge graveyard

This creates evolutionary pressure - adapt or die.

Phase 2, Week 2 - Step 12
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
import json
import uuid


@dataclass
class KillEvent:
    """Record of a kill cycle execution"""
    kill_id: str
    timestamp: datetime
    killed_engine: str
    killed_rank: int
    killed_pnl: float
    killed_win_rate: float
    killed_trades: int
    winner_engine: str
    winner_pnl: float
    winner_win_rate: float
    lessons_transferred: List[str]
    archived_trades: int
    cycle_number: int


@dataclass
class WinnerLesson:
    """Lesson extracted from winner's successful strategies"""
    lesson_id: str
    source_engine: str
    lesson_type: str  # entry_timing, exit_strategy, regime_detection, position_sizing
    description: str
    confidence: float
    sample_trades: List[str]  # trade IDs that demonstrated this lesson
    created_at: datetime


class KillCycle:
    """
    Kill Cycle Manager - 24-hour evolution mechanism.

    Every 24 hours:
    1. Get current rankings
    2. Identify #4 (worst P&L)
    3. Archive #4's strategy to edge graveyard
    4. DELETE #4's current strategy completely (LIVE mode only)
    5. Extract lessons from #1 (winner)
    6. Force #4 to learn from #1
    7. #4 must invent NEW strategy from scratch
    8. Log kill event for analysis

    STEP 9: Paper/Live Mode Switch
    - Paper mode: Archive trades but DON'T delete portfolio (soft kill)
    - Live mode: Full kill with portfolio deletion (hard kill)
    """

    KILL_INTERVAL_HOURS = 24
    MIN_TRADES_TO_KILL = 5  # Need at least 5 trades before killing

    # STEP 9: Paper/Live mode - default to paper for safety
    PAPER_MODE = True  # True = soft kill (no deletion), False = hard kill (full deletion)

    def __init__(self, data_dir: Optional[Path] = None):
        # Use central config for path detection
        if data_dir is None:
            from ..config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Kill history
        self.kill_history: List[KillEvent] = []
        self.kill_count = 0
        self.last_kill_time: Optional[datetime] = None

        # Persistence
        self.state_file = self.data_dir / "kill_cycle_state.json"
        self.graveyard_file = self.data_dir / "edge_graveyard.jsonl"
        self.lessons_file = self.data_dir / "winner_lessons.jsonl"

        # Load existing state
        self._load_state()

        logger.info(f"[KillCycle] Initialized. Total kills: {self.kill_count}")

    def should_execute_kill(self, rankings: List[tuple]) -> bool:
        """
        Check if kill cycle should execute.

        Conditions:
        1. 24+ hours since last kill
        2. All engines have minimum trades
        3. Rankings are available

        Args:
            rankings: List of (engine_name, stats) from TournamentManager

        Returns:
            True if kill should execute
        """
        now = datetime.now(timezone.utc)

        # Check time since last kill
        if self.last_kill_time is not None:
            hours_since = (now - self.last_kill_time).total_seconds() / 3600
            if hours_since < self.KILL_INTERVAL_HOURS:
                logger.debug(
                    f"[KillCycle] Not time yet. Hours since last: {hours_since:.1f}, "
                    f"Need: {self.KILL_INTERVAL_HOURS}"
                )
                return False

        # Check if we have rankings
        if not rankings or len(rankings) < 4:
            logger.debug("[KillCycle] Not enough engines for kill cycle")
            return False

        # Check if last place has minimum trades
        last_place = rankings[-1]
        last_stats = last_place[1]

        if last_stats.closed_trades < self.MIN_TRADES_TO_KILL:
            logger.debug(
                f"[KillCycle] Last place ({last_place[0]}) has only "
                f"{last_stats.closed_trades} closed trades (need {self.MIN_TRADES_TO_KILL})"
            )
            return False

        logger.info("[KillCycle] Kill conditions MET - ready to execute")
        return True

    def execute_kill(
        self,
        rankings: List[tuple],
        get_portfolio_fn,
        get_engine_fn
    ) -> Optional[KillEvent]:
        """
        Execute the kill cycle.

        Args:
            rankings: List of (engine_name, stats) sorted by rank
            get_portfolio_fn: Function to get portfolio by engine name
            get_engine_fn: Function to get engine instance by name

        Returns:
            KillEvent if kill executed, None otherwise
        """
        if not self.should_execute_kill(rankings):
            return None

        self.kill_count += 1
        now = datetime.now(timezone.utc)

        # Identify winner (#1) and loser (#4)
        winner_name, winner_stats = rankings[0]
        loser_name, loser_stats = rankings[-1]

        logger.warning(f"\n{'='*60}")
        logger.warning(f"💀 KILL CYCLE #{self.kill_count} EXECUTING 💀")
        logger.warning(f"{'='*60}")
        logger.warning(f"Victim: Engine {loser_name} (Rank #4)")
        logger.warning(f"  - P&L: ${loser_stats.total_pnl_usd:+.2f}")
        logger.warning(f"  - Win Rate: {loser_stats.win_rate:.1%}")
        logger.warning(f"  - Trades: {loser_stats.total_trades}")
        logger.warning(f"Winner: Engine {winner_name} (Rank #1)")
        logger.warning(f"  - P&L: ${winner_stats.total_pnl_usd:+.2f}")
        logger.warning(f"  - Win Rate: {winner_stats.win_rate:.1%}")
        logger.warning(f"{'='*60}\n")

        # Step 1: Archive loser's trades to graveyard
        loser_portfolio = get_portfolio_fn(loser_name)
        archived_count = self._archive_to_graveyard(loser_name, loser_portfolio)

        # Step 2: Extract lessons from winner
        winner_portfolio = get_portfolio_fn(winner_name)
        lessons = self._extract_winner_lessons(winner_name, winner_portfolio)

        # STEP 9: Paper/Live mode switch
        if self.PAPER_MODE:
            logger.warning(f"[KillCycle] PAPER MODE - Skipping portfolio deletion (soft kill)")
            # In paper mode, we still transfer lessons but DON'T delete the portfolio
        else:
            # Step 3: Reset loser's portfolio (DELETE strategy) - LIVE MODE ONLY
            logger.warning(f"[KillCycle] LIVE MODE - Executing full portfolio deletion (hard kill)")
            self._reset_portfolio(loser_portfolio)

        # Step 4: Inject lessons into loser
        loser_engine = get_engine_fn(loser_name)
        self._inject_lessons(loser_engine, lessons)

        # Step 5: Force loser to reinvent (update prompt/state)
        self._force_reinvention(loser_engine, winner_name, lessons)

        # Create kill event
        kill_event = KillEvent(
            kill_id=str(uuid.uuid4())[:8],
            timestamp=now,
            killed_engine=loser_name,
            killed_rank=4,
            killed_pnl=loser_stats.total_pnl_usd,
            killed_win_rate=loser_stats.win_rate,
            killed_trades=loser_stats.total_trades,
            winner_engine=winner_name,
            winner_pnl=winner_stats.total_pnl_usd,
            winner_win_rate=winner_stats.win_rate,
            lessons_transferred=[l.description for l in lessons],
            archived_trades=archived_count,
            cycle_number=self.kill_count
        )

        # Save event
        self.kill_history.append(kill_event)
        self.last_kill_time = now
        self._save_state()
        self._log_kill_event(kill_event)

        logger.success(f"\n💀 Kill cycle complete. Engine {loser_name} has been reborn.")
        logger.success(f"   Archived {archived_count} trades to graveyard")
        logger.success(f"   Transferred {len(lessons)} lessons from winner")

        return kill_event

    def _archive_to_graveyard(self, engine: str, portfolio) -> int:
        """
        Archive all trades from killed engine to edge graveyard.

        The graveyard stores failed strategies for:
        - Post-mortem analysis
        - Avoiding repeat mistakes
        - Edge counter development

        Returns:
            Number of trades archived
        """
        trades = portfolio.trades
        archived = 0

        with open(self.graveyard_file, 'a') as f:
            for trade in trades:
                graveyard_entry = {
                    "archived_at": datetime.now(timezone.utc).isoformat(),
                    "kill_cycle": self.kill_count,
                    "engine": engine,
                    "trade_id": trade.trade_id,
                    "asset": trade.asset,
                    "direction": trade.direction,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "pnl_percent": trade.pnl_percent,
                    "pnl_usd": trade.pnl_usd,
                    "outcome": trade.outcome,
                    "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                    "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                    "exit_reason": trade.exit_reason,
                    "status": trade.status
                }
                f.write(json.dumps(graveyard_entry) + '\n')
                archived += 1

        logger.info(f"[KillCycle] Archived {archived} trades to graveyard")
        return archived

    def _extract_winner_lessons(self, engine: str, portfolio) -> List[WinnerLesson]:
        """
        Extract lessons from winner's successful strategies.

        Analyzes:
        - Entry timing patterns
        - Exit strategies that worked
        - Regime detection accuracy
        - Position sizing effectiveness

        Returns:
            List of lessons to transfer
        """
        lessons = []
        now = datetime.now(timezone.utc)

        # Get winning trades
        closed_trades = portfolio.get_closed_trades()
        winning_trades = [t for t in closed_trades if t.outcome == "win"]

        if not winning_trades:
            logger.warning(f"[KillCycle] Winner {engine} has no winning trades to learn from")
            return lessons

        # Lesson 1: Entry timing
        avg_win_pnl = sum(t.pnl_percent for t in winning_trades) / len(winning_trades)
        lessons.append(WinnerLesson(
            lesson_id=str(uuid.uuid4())[:8],
            source_engine=engine,
            lesson_type="entry_timing",
            description=f"Winner achieves {avg_win_pnl:.2f}% avg on winning trades",
            confidence=min(0.9, len(winning_trades) / 20),  # Confidence scales with sample size
            sample_trades=[t.trade_id for t in winning_trades[:5]],
            created_at=now
        ))

        # Lesson 2: Win rate consistency
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        lessons.append(WinnerLesson(
            lesson_id=str(uuid.uuid4())[:8],
            source_engine=engine,
            lesson_type="consistency",
            description=f"Winner maintains {win_rate:.1%} win rate across {len(closed_trades)} trades",
            confidence=min(0.85, len(closed_trades) / 30),
            sample_trades=[t.trade_id for t in closed_trades[:5]],
            created_at=now
        ))

        # Lesson 3: Risk management (based on SL/TP ratios)
        if winning_trades:
            avg_rr = 0
            for t in winning_trades:
                if t.entry_price > 0 and t.stop_loss > 0 and t.take_profit > 0:
                    risk = abs(t.entry_price - t.stop_loss)
                    reward = abs(t.take_profit - t.entry_price)
                    if risk > 0:
                        avg_rr += reward / risk
            avg_rr = avg_rr / len(winning_trades) if winning_trades else 0

            if avg_rr > 0:
                lessons.append(WinnerLesson(
                    lesson_id=str(uuid.uuid4())[:8],
                    source_engine=engine,
                    lesson_type="risk_management",
                    description=f"Winner uses {avg_rr:.2f}:1 average reward/risk ratio",
                    confidence=0.80,
                    sample_trades=[t.trade_id for t in winning_trades[:3]],
                    created_at=now
                ))

        # Lesson 4: Direction preference
        long_wins = [t for t in winning_trades if t.direction == "BUY"]
        short_wins = [t for t in winning_trades if t.direction == "SELL"]

        if len(long_wins) > len(short_wins) * 2:
            lessons.append(WinnerLesson(
                lesson_id=str(uuid.uuid4())[:8],
                source_engine=engine,
                lesson_type="direction_bias",
                description=f"Winner favors LONG positions ({len(long_wins)}/{len(winning_trades)} wins)",
                confidence=0.70,
                sample_trades=[t.trade_id for t in long_wins[:3]],
                created_at=now
            ))
        elif len(short_wins) > len(long_wins) * 2:
            lessons.append(WinnerLesson(
                lesson_id=str(uuid.uuid4())[:8],
                source_engine=engine,
                lesson_type="direction_bias",
                description=f"Winner favors SHORT positions ({len(short_wins)}/{len(winning_trades)} wins)",
                confidence=0.70,
                sample_trades=[t.trade_id for t in short_wins[:3]],
                created_at=now
            ))

        # Save lessons to file
        self._save_lessons(lessons)

        logger.info(f"[KillCycle] Extracted {len(lessons)} lessons from winner {engine}")
        return lessons

    def _save_lessons(self, lessons: List[WinnerLesson]):
        """Save lessons to JSONL file."""
        with open(self.lessons_file, 'a') as f:
            for lesson in lessons:
                lesson_dict = {
                    "lesson_id": lesson.lesson_id,
                    "source_engine": lesson.source_engine,
                    "lesson_type": lesson.lesson_type,
                    "description": lesson.description,
                    "confidence": lesson.confidence,
                    "sample_trades": lesson.sample_trades,
                    "created_at": lesson.created_at.isoformat(),
                    "kill_cycle": self.kill_count
                }
                f.write(json.dumps(lesson_dict) + '\n')

    def _reset_portfolio(self, portfolio):
        """
        Reset portfolio - DELETE all trades and stats.

        This is the "death" - complete strategy wipe.
        """
        # Clear all trades
        portfolio.trades = []

        # Reset stats to zero
        portfolio.stats.total_trades = 0
        portfolio.stats.open_trades = 0
        portfolio.stats.closed_trades = 0
        portfolio.stats.wins = 0
        portfolio.stats.losses = 0
        portfolio.stats.win_rate = 0.0
        portfolio.stats.total_pnl_percent = 0.0
        portfolio.stats.total_pnl_usd = 0.0
        portfolio.stats.avg_win_percent = 0.0
        portfolio.stats.avg_loss_percent = 0.0
        portfolio.stats.best_trade_percent = 0.0
        portfolio.stats.worst_trade_percent = 0.0
        portfolio.stats.sharpe_ratio = None
        portfolio.stats.current_rank = 4  # Start at bottom
        portfolio.stats.weight = 0.10  # Minimum weight

        # Save empty state
        portfolio._save_trades()

        logger.info(f"[KillCycle] Portfolio for Engine {portfolio.engine} has been reset")

    def _inject_lessons(self, engine, lessons: List[WinnerLesson]):
        """
        Inject winner's lessons into killed engine.

        Updates engine's learned_lessons attribute.
        """
        if not hasattr(engine, 'learned_lessons'):
            engine.learned_lessons = []

        # Add new lessons
        for lesson in lessons:
            engine.learned_lessons.append({
                "from_kill_cycle": self.kill_count,
                "source": lesson.source_engine,
                "type": lesson.lesson_type,
                "description": lesson.description,
                "confidence": lesson.confidence,
                "applied_at": datetime.now(timezone.utc).isoformat()
            })

        logger.info(f"[KillCycle] Injected {len(lessons)} lessons into killed engine")

    def _force_reinvention(self, engine, winner_name: str, lessons: List[WinnerLesson]):
        """
        Force killed engine to reinvent with winner's knowledge.

        Updates engine state to:
        - Know it was killed
        - Have lessons from winner
        - Be motivated to create NEW strategy
        """
        # Mark engine as recently killed (for emotion prompt)
        if hasattr(engine, 'meta_state'):
            engine.meta_state['was_killed'] = True
            engine.meta_state['killed_at'] = datetime.now(timezone.utc).isoformat()
            engine.meta_state['kill_count'] = getattr(engine.meta_state, 'kill_count', 0) + 1
            engine.meta_state['learned_from'] = winner_name
            engine.meta_state['must_reinvent'] = True

        # Compose reinvention guidance
        guidance = f"""
SYSTEM NOTICE: You were ELIMINATED (Rank #4) in kill cycle #{self.kill_count}.

Your previous strategy FAILED. It has been completely deleted.

LESSONS FROM WINNER (Engine {winner_name}):
"""
        for lesson in lessons:
            guidance += f"\n- [{lesson.lesson_type}] {lesson.description}"

        guidance += """

CRITICAL DIRECTIVE:
You MUST invent a NEW strategy. Do NOT repeat your previous approach.
Use the winner's lessons as inspiration, but CREATE something original.
Prove you can adapt and survive the next kill cycle.
"""

        # Store guidance for next decision
        if hasattr(engine, 'reinvention_guidance'):
            engine.reinvention_guidance = guidance

        logger.info(f"[KillCycle] Reinvention guidance prepared for killed engine")

    def _log_kill_event(self, event: KillEvent):
        """Log kill event to kill history file."""
        kill_log_file = self.data_dir / "kill_history.jsonl"

        with open(kill_log_file, 'a') as f:
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()
            f.write(json.dumps(event_dict) + '\n')

    def _save_state(self):
        """Save kill cycle state to disk."""
        state = {
            "kill_count": self.kill_count,
            "last_kill_time": self.last_kill_time.isoformat() if self.last_kill_time else None,
            "recent_kills": [
                {
                    "kill_id": k.kill_id,
                    "timestamp": k.timestamp.isoformat(),
                    "killed_engine": k.killed_engine,
                    "winner_engine": k.winner_engine,
                    "cycle_number": k.cycle_number
                }
                for k in self.kill_history[-10:]  # Last 10 kills
            ]
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load kill cycle state from disk."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.kill_count = state.get("kill_count", 0)
            last_kill = state.get("last_kill_time")
            if last_kill:
                self.last_kill_time = datetime.fromisoformat(last_kill)

            logger.debug(f"[KillCycle] Loaded state: {self.kill_count} kills")

        except Exception as e:
            logger.warning(f"[KillCycle] Failed to load state: {e}")

    def get_kill_summary(self) -> Dict[str, Any]:
        """Get summary of kill cycle history."""
        return {
            "total_kills": self.kill_count,
            "last_kill": self.last_kill_time.isoformat() if self.last_kill_time else None,
            "hours_since_last": (
                (datetime.now(timezone.utc) - self.last_kill_time).total_seconds() / 3600
                if self.last_kill_time else None
            ),
            "next_kill_in_hours": (
                max(0, self.KILL_INTERVAL_HOURS -
                    (datetime.now(timezone.utc) - self.last_kill_time).total_seconds() / 3600)
                if self.last_kill_time else 0
            ),
            "recent_victims": [
                {"engine": k.killed_engine, "cycle": k.cycle_number}
                for k in self.kill_history[-5:]
            ],
            # STEP 9: Paper/Live mode status
            "mode": "PAPER" if self.PAPER_MODE else "LIVE",
            "hard_kill_enabled": not self.PAPER_MODE
        }

    # STEP 9: Mode switching methods
    def set_paper_mode(self):
        """Switch to paper mode (soft kills - no portfolio deletion)."""
        self.PAPER_MODE = True
        logger.warning("[KillCycle] Switched to PAPER MODE - soft kills only")

    def set_live_mode(self):
        """Switch to live mode (hard kills - full portfolio deletion)."""
        self.PAPER_MODE = False
        logger.critical("[KillCycle] Switched to LIVE MODE - hard kills enabled!")

    def is_paper_mode(self) -> bool:
        """Check if kill cycle is in paper mode."""
        return self.PAPER_MODE

    def get_graveyard_stats(self) -> Dict[str, Any]:
        """Get statistics from edge graveyard."""
        if not self.graveyard_file.exists():
            return {"total_archived": 0, "by_engine": {}}

        archived_by_engine: Dict[str, int] = {}
        total_archived = 0

        with open(self.graveyard_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                engine = entry.get("engine", "unknown")
                archived_by_engine[engine] = archived_by_engine.get(engine, 0) + 1
                total_archived += 1

        return {
            "total_archived": total_archived,
            "by_engine": archived_by_engine
        }


# ==================== SINGLETON PATTERN ====================

_kill_cycle: Optional[KillCycle] = None

def get_kill_cycle() -> KillCycle:
    """Get singleton instance of KillCycle."""
    global _kill_cycle
    if _kill_cycle is None:
        _kill_cycle = KillCycle()
    return _kill_cycle
