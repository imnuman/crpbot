#!/usr/bin/env python3
"""Clear all V7 signals from database to start fresh."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text
from libs.config.config import load_config

def clear_v7_signals():
    """Delete all V7 Ultimate signals from database."""
    config = load_config()
    engine = create_engine(config.db_url)

    with engine.connect() as conn:
        # Count before deletion
        result = conn.execute(text("SELECT COUNT(*) FROM signals WHERE model_version = 'v7_ultimate'"))
        count_before = result.scalar()

        print(f"Found {count_before} V7 signals in database")
        print("Deleting all V7 signals...")

        # Delete V7 signals
        conn.execute(text("DELETE FROM signals WHERE model_version = 'v7_ultimate'"))
        conn.commit()

        # Verify deletion
        result = conn.execute(text("SELECT COUNT(*) FROM signals WHERE model_version = 'v7_ultimate'"))
        count_after = result.scalar()

        print(f"✅ Deleted {count_before - count_after} signals")
        print(f"Remaining V7 signals: {count_after}")
        print("\nDatabase cleared! Dashboard will show only NEW signals from now on.")
        print("V7 Runtime is still running and will generate fresh signals every 2 minutes.")

if __name__ == "__main__":
    clear_v7_signals()
