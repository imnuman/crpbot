#!/usr/bin/env python3
"""Test database setup and operations."""
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from libs.config.config import Settings
from libs.db.auto_learning import AutoLearningSystem
from libs.db.database import init_database


def test_database():
    """Test database operations."""
    config = Settings()
    logger.info("Testing database...")

    # Initialize database
    init_database(db_url=config.db_url, create_tables=True)
    logger.info("✅ Database initialized")

    # Test auto-learning system
    auto_learning = AutoLearningSystem()
    logger.info("✅ Auto-learning system initialized")

    # Test pattern tracking
    features = {"feature1": 0.5, "feature2": 0.7}
    pattern_name = "test_pattern"

    # Record pattern result
    auto_learning.record_pattern_result(features, "win", pattern_name)
    logger.info("✅ Pattern result recorded")

    # Get pattern win rate
    win_rate, sample_count = auto_learning.get_pattern_win_rate(features, pattern_name)
    if win_rate is not None:
        logger.info(f"✅ Pattern win rate: {win_rate:.2%} (samples: {sample_count})")
    else:
        logger.info(f"✅ Pattern win rate: N/A (samples: {sample_count} < floor)")

    # Test trade recording
    signal_id = "test_signal_001"
    auto_learning.record_trade(
        signal_id=signal_id,
        pair="BTC-USD",
        tier="high",
        entry_time=datetime.utcnow(),
        entry_price=50000.0,
        tp_price=51000.0,
        sl_price=49000.0,
        rr_expected=2.0,
        mode="dryrun",
    )
    logger.info("✅ Trade recorded")

    # Update trade result
    auto_learning.update_trade_result(
        signal_id=signal_id,
        result="win",
        exit_time=datetime.utcnow(),
        exit_price=51000.0,
        r_realized=2.0,
        time_to_tp_sl_seconds=3600,
    )
    logger.info("✅ Trade result updated")

    # Get statistics
    stats = auto_learning.get_statistics(days=30, mode="dryrun")
    logger.info(f"✅ Statistics: {stats}")

    logger.info("✅ All database tests passed!")


if __name__ == "__main__":
    test_database()
