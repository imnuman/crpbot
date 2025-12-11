#!/usr/bin/env python3
"""
HYDRA Knowledge Aggregator Service

Runs collectors on schedule and manages the knowledge base.

Usage:
    python knowledge_runner.py                  # Run with scheduler
    python knowledge_runner.py --once           # Run all collectors once
    python knowledge_runner.py --collector reddit  # Run specific collector once
    python knowledge_runner.py --stats          # Show statistics
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "/tmp/knowledge_runner.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logger.warning("apscheduler not installed - scheduled runs disabled")

from libs.knowledge.storage import get_storage
from libs.knowledge.embeddings import get_embedding_service
from libs.knowledge.query import KnowledgeQuery


# Import collectors
async def get_collectors():
    """Import and return collector runners."""
    collectors = {}

    try:
        from libs.knowledge.collectors.reddit import run_reddit_collector
        collectors["reddit"] = run_reddit_collector
    except ImportError as e:
        logger.warning(f"Reddit collector not available: {e}")

    try:
        from libs.knowledge.collectors.mql5 import run_mql5_collector
        collectors["mql5"] = run_mql5_collector
    except ImportError as e:
        logger.warning(f"MQL5 collector not available: {e}")

    try:
        from libs.knowledge.collectors.github import run_github_collector
        collectors["github"] = run_github_collector
    except ImportError as e:
        logger.warning(f"GitHub collector not available: {e}")

    try:
        from libs.knowledge.collectors.economic_calendar import run_calendar_collector
        collectors["calendar"] = run_calendar_collector
    except ImportError as e:
        logger.warning(f"Calendar collector not available: {e}")

    try:
        from libs.knowledge.collectors.tradingview import run_tradingview_collector
        collectors["tradingview"] = run_tradingview_collector
    except ImportError as e:
        logger.warning(f"TradingView collector not available: {e}")

    return collectors


# Collector schedules (cron expressions)
COLLECTOR_SCHEDULES = {
    "tradingview": "0 */6 * * *",  # Every 6 hours (Reddit replacement)
    "reddit": "0 */6 * * *",       # Every 6 hours (needs API approval)
    "mql5": "0 2 * * *",           # Daily at 02:00 UTC
    "github": "0 3 * * 6",         # Weekly on Saturday at 03:00
    "calendar": "0 0 * * *",       # Daily at 00:00 UTC
}


class KnowledgeRunner:
    """Main knowledge aggregation service."""

    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.running = False
        self.collectors = {}

    async def initialize(self):
        """Initialize the service."""
        logger.info("Initializing Knowledge Runner...")

        # Get available collectors
        self.collectors = await get_collectors()
        logger.info(f"Available collectors: {list(self.collectors.keys())}")

        # Initialize storage
        storage = get_storage()
        stats = storage.get_stats()
        logger.info(f"Knowledge base: {stats.get('total_items', 0)} items")

        # Initialize scheduler if available
        if SCHEDULER_AVAILABLE:
            self.scheduler = AsyncIOScheduler()
            self._setup_schedules()

    def _setup_schedules(self):
        """Setup scheduled jobs for each collector."""
        if not self.scheduler:
            return

        for name, cron in COLLECTOR_SCHEDULES.items():
            if name in self.collectors:
                trigger = CronTrigger.from_crontab(cron)
                self.scheduler.add_job(
                    self._run_collector,
                    trigger=trigger,
                    args=[name],
                    id=f"collector_{name}",
                    name=f"Collect from {name}",
                    replace_existing=True,
                )
                logger.info(f"Scheduled {name}: {cron}")

        # Schedule embedding processing
        self.scheduler.add_job(
            self._process_embeddings,
            trigger=CronTrigger.from_crontab("0 */2 * * *"),  # Every 2 hours
            id="process_embeddings",
            name="Process embeddings",
            replace_existing=True,
        )

    async def _run_collector(self, name: str) -> int:
        """Run a specific collector."""
        if name not in self.collectors:
            logger.error(f"Unknown collector: {name}")
            return 0

        logger.info(f"Running collector: {name}")
        start_time = datetime.now(timezone.utc)

        try:
            runner = self.collectors[name]
            count = await runner()
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"Collector {name} complete: {count} items in {elapsed:.1f}s")
            return count

        except Exception as e:
            logger.error(f"Collector {name} failed: {e}")
            return 0

    async def _process_embeddings(self):
        """Process items without embeddings."""
        logger.info("Processing pending embeddings...")

        try:
            embedding_service = get_embedding_service()
            count = embedding_service.process_pending_embeddings(batch_size=50)
            logger.info(f"Processed {count} embeddings")

        except Exception as e:
            logger.error(f"Embedding processing failed: {e}")

    async def run_once(self, collector_name: Optional[str] = None):
        """Run collectors once (no scheduling)."""
        await self.initialize()

        if collector_name:
            # Run specific collector
            await self._run_collector(collector_name)
        else:
            # Run all collectors
            for name in self.collectors:
                await self._run_collector(name)
                await asyncio.sleep(2)  # Brief pause between collectors

        # Process embeddings
        await self._process_embeddings()

    async def run_forever(self):
        """Run with scheduler."""
        await self.initialize()

        if not self.scheduler:
            logger.error("Scheduler not available - use --once flag")
            return

        self.running = True
        self.scheduler.start()

        logger.info("Knowledge Runner started. Press Ctrl+C to stop.")

        # Initial run of all collectors
        logger.info("Running initial collection...")
        for name in self.collectors:
            await self._run_collector(name)
            await asyncio.sleep(5)

        # Process initial embeddings
        await self._process_embeddings()

        # Keep running
        try:
            while self.running:
                await asyncio.sleep(60)

                # Log stats periodically
                storage = get_storage()
                stats = storage.get_stats()
                logger.info(
                    f"Stats: {stats.get('total_items', 0)} items, "
                    f"{stats.get('with_embeddings', 0)} embedded"
                )

        except asyncio.CancelledError:
            pass

        finally:
            if self.scheduler:
                self.scheduler.shutdown()
            logger.info("Knowledge Runner stopped.")

    def stop(self):
        """Stop the service."""
        self.running = False


def show_stats():
    """Display knowledge base statistics."""
    storage = get_storage()
    stats = storage.get_stats()

    print("\n=== Knowledge Base Statistics ===\n")
    print(f"Total items: {stats.get('total_items', 0)}")
    print(f"With embeddings: {stats.get('with_embeddings', 0)}")
    print(f"Economic events: {stats.get('economic_events', 0)}")
    print(f"Code files: {stats.get('code_files', 0)}")

    print("\nBy Source:")
    for source, count in stats.get("by_source", {}).items():
        print(f"  {source}: {count}")

    print("\nBy Type:")
    for content_type, count in stats.get("by_type", {}).items():
        print(f"  {content_type}: {count}")

    # Try to get embedding stats
    try:
        embedding_service = get_embedding_service()
        emb_stats = embedding_service.get_collection_stats()
        print(f"\nEmbeddings: {emb_stats.get('total_embeddings', 'N/A')}")
    except Exception:
        pass

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HYDRA Knowledge Aggregator")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run all collectors once and exit",
    )
    parser.add_argument(
        "--collector",
        type=str,
        help="Run a specific collector (reddit, mql5, github, calendar)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show knowledge base statistics",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    runner = KnowledgeRunner()

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.once or args.collector:
        asyncio.run(runner.run_once(args.collector))
    else:
        asyncio.run(runner.run_forever())


if __name__ == "__main__":
    main()
