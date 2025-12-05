"""
HYDRA 4.0 - Nightly Scheduler

Schedules and runs nightly tasks:
- Daily evolution (mistake analysis)
- Batch strategy generation (if turbo mode)
- Rule pruning
- Stats logging
"""

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, Dict, Any, List

logger = logging.getLogger(__name__)


class NightlyScheduler:
    """
    Scheduler for nightly HYDRA 4.0 tasks.

    Tasks:
    - Midnight: Run daily evolution
    - Midnight: Generate nightly strategy batch (if FTMO prep mode)
    - Midnight: Prune old rules
    - Midnight: Log daily stats
    """

    def __init__(
        self,
        evolution_callback: Optional[Callable] = None,
        generation_callback: Optional[Callable] = None,
        stats_callback: Optional[Callable] = None,
        target_hour: int = 0  # Midnight UTC
    ):
        """
        Initialize the nightly scheduler.

        Args:
            evolution_callback: Function to call for daily evolution
            generation_callback: Function to call for batch generation
            stats_callback: Function to call for stats logging
            target_hour: Hour to run tasks (0-23, default midnight)
        """
        self.evolution_callback = evolution_callback
        self.generation_callback = generation_callback
        self.stats_callback = stats_callback
        self.target_hour = target_hour

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_run_date: Optional[str] = None

        logger.info(f"[NightlyScheduler] Initialized (target hour: {target_hour}:00 UTC)")

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            logger.warning("[NightlyScheduler] Already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[NightlyScheduler] Started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[NightlyScheduler] Stopped")

    def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                today = now.strftime("%Y-%m-%d")

                # Check if we should run
                if self._should_run(now, today):
                    logger.info(f"[NightlyScheduler] Running nightly tasks for {today}")
                    self._run_nightly_tasks()
                    self._last_run_date = today

                # Sleep for 1 minute before checking again
                time.sleep(60)

            except Exception as e:
                logger.error(f"[NightlyScheduler] Error in run loop: {e}")
                time.sleep(60)

    def _should_run(self, now: datetime, today: str) -> bool:
        """Check if nightly tasks should run."""
        # Already ran today
        if self._last_run_date == today:
            return False

        # Check if it's the target hour
        if now.hour != self.target_hour:
            return False

        return True

    def _run_nightly_tasks(self):
        """Execute all nightly tasks."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evolution": None,
            "generation": None,
            "stats": None,
        }

        # Task 1: Daily Evolution
        if self.evolution_callback:
            try:
                logger.info("[NightlyScheduler] Running daily evolution...")
                results["evolution"] = self.evolution_callback()
                logger.info(f"[NightlyScheduler] Evolution complete: {results['evolution']}")
            except Exception as e:
                logger.error(f"[NightlyScheduler] Evolution failed: {e}")
                results["evolution"] = {"error": str(e)}

        # Task 2: Batch Generation (if callback provided)
        if self.generation_callback:
            try:
                logger.info("[NightlyScheduler] Running batch generation...")
                results["generation"] = self.generation_callback()
                logger.info(f"[NightlyScheduler] Generation complete: {results['generation']}")
            except Exception as e:
                logger.error(f"[NightlyScheduler] Generation failed: {e}")
                results["generation"] = {"error": str(e)}

        # Task 3: Stats Logging
        if self.stats_callback:
            try:
                logger.info("[NightlyScheduler] Logging daily stats...")
                results["stats"] = self.stats_callback()
                logger.info(f"[NightlyScheduler] Stats logged: {results['stats']}")
            except Exception as e:
                logger.error(f"[NightlyScheduler] Stats logging failed: {e}")
                results["stats"] = {"error": str(e)}

        logger.info(f"[NightlyScheduler] All nightly tasks complete")
        return results

    def run_now(self) -> Dict[str, Any]:
        """Manually trigger nightly tasks (for testing)."""
        logger.info("[NightlyScheduler] Manual trigger - running nightly tasks now")
        self._last_run_date = None  # Reset to allow running
        return self._run_nightly_tasks()

    def get_next_run_time(self) -> datetime:
        """Get the next scheduled run time."""
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=self.target_hour, minute=0, second=0, microsecond=0)

        # If we've passed today's target hour, schedule for tomorrow
        if now.hour >= self.target_hour:
            next_run += timedelta(days=1)

        return next_run

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "target_hour": self.target_hour,
            "last_run_date": self._last_run_date,
            "next_run": self.get_next_run_time().isoformat(),
            "has_evolution_callback": self.evolution_callback is not None,
            "has_generation_callback": self.generation_callback is not None,
            "has_stats_callback": self.stats_callback is not None,
        }


def create_default_scheduler(
    paper_trader=None,
    daily_evolution=None,
    turbo_generator=None,
    turbo_tournament=None,
    turbo_config=None,
    strategy_memory=None
) -> NightlyScheduler:
    """
    Create a scheduler with default callbacks wired up.

    Args:
        paper_trader: Paper trading system
        daily_evolution: Daily evolution system
        turbo_generator: Strategy generator
        turbo_tournament: Strategy ranker
        turbo_config: Configuration
        strategy_memory: Strategy memory

    Returns:
        Configured NightlyScheduler
    """

    def evolution_callback():
        """Run daily evolution on losing trades."""
        if not daily_evolution or not paper_trader:
            return {"skipped": "missing dependencies"}

        # Get today's losing trades
        losing_trades = []
        try:
            # Get closed trades from today
            today = datetime.now(timezone.utc).date()
            for trade in paper_trader.get_closed_trades():
                if trade.get("outcome") == "loss":
                    exit_time = trade.get("exit_timestamp", "")
                    if exit_time and exit_time[:10] == str(today):
                        losing_trades.append(trade)
        except Exception as e:
            logger.error(f"Error getting losing trades: {e}")

        return daily_evolution.run_at_midnight(losing_trades)

    def generation_callback():
        """Generate nightly batch of strategies."""
        if not turbo_generator or not turbo_tournament or not turbo_config:
            return {"skipped": "missing dependencies"}

        if not turbo_config.FTMO_PREP_MODE:
            return {"skipped": "not in FTMO prep mode"}

        from libs.hydra.turbo_generator import StrategyType

        results = {
            "generated": 0,
            "ranked": 0,
            "top_strategies": [],
        }

        # Generate batch for each specialty
        batch_size = turbo_config.get_batch_size()
        all_strategies = []

        for specialty in StrategyType:
            strategies = turbo_generator.generate_batch(
                specialty=specialty,
                count=batch_size // 4,  # Split across specialties
                use_mock=True  # Use mock for nightly to save costs
            )
            all_strategies.extend(strategies)

        results["generated"] = len(all_strategies)

        # Rank strategies
        if all_strategies:
            ranked = turbo_tournament.rank_batch(all_strategies)
            results["ranked"] = len(ranked)

            # Get top strategies
            top_n = min(10, len(ranked))
            results["top_strategies"] = [
                {
                    "id": s.strategy_id,
                    "score": r.rank_score,
                    "wr": r.win_rate,
                }
                for s, r in ranked[:top_n]
            ]

            # Add to strategy memory if available
            if strategy_memory:
                for strategy, result in ranked[:100]:  # Top 100
                    try:
                        strategy_memory.add_strategy(strategy.to_dict())
                    except Exception as e:
                        logger.debug(f"Error adding strategy: {e}")

        return results

    def stats_callback():
        """Log daily statistics."""
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if paper_trader:
            try:
                stats["paper_trading"] = {
                    "open_trades": len(paper_trader.get_open_trades()),
                    "closed_today": 0,  # Would need to filter by date
                }
            except Exception:
                pass

        if daily_evolution:
            try:
                stats["prevention_rules"] = daily_evolution.get_rule_stats()
            except Exception:
                pass

        if turbo_config:
            stats["config"] = turbo_config.to_dict()

        return stats

    return NightlyScheduler(
        evolution_callback=evolution_callback if daily_evolution else None,
        generation_callback=generation_callback if turbo_generator else None,
        stats_callback=stats_callback,
        target_hour=0  # Midnight UTC
    )


# Singleton instance
_scheduler_instance: Optional[NightlyScheduler] = None


def get_nightly_scheduler() -> Optional[NightlyScheduler]:
    """Get the nightly scheduler singleton (if initialized)."""
    return _scheduler_instance


def init_nightly_scheduler(**kwargs) -> NightlyScheduler:
    """Initialize and return the nightly scheduler singleton."""
    global _scheduler_instance
    _scheduler_instance = create_default_scheduler(**kwargs)
    return _scheduler_instance
