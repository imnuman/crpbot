"""
Create missing database tables for performance tracking system.

This script adds:
1. signal_results - For paper trading performance tracking
2. theory_performance - For tracking individual theory contributions
"""

import sys
sys.path.insert(0, '/root/crpbot')

from libs.db.models import Base, SignalResult, TheoryPerformance
from sqlalchemy import create_engine
from libs.config.config import Settings

def create_missing_tables():
    """Create signal_results and theory_performance tables"""
    config = Settings()
    engine = create_engine(str(config.db_url), echo=True)

    print("=" * 70)
    print("CREATING MISSING DATABASE TABLES")
    print("=" * 70)
    print(f"Database: {config.db_url}")
    print()

    # Create only the new tables (existing tables won't be affected)
    print("Creating tables:")
    print("  - signal_results (for performance tracking)")
    print("  - theory_performance (for theory contribution tracking)")
    print()

    # This will only create tables that don't exist
    Base.metadata.create_all(engine, tables=[
        SignalResult.__table__,
        TheoryPerformance.__table__
    ])

    print()
    print("=" * 70)
    print("✅ TABLES CREATED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Restart V7 runtime to begin paper trading")
    print("2. Restart dashboard to see performance tracking")
    print("3. Monitor /tmp/v7_*.log for paper trading activity")
    print()

if __name__ == "__main__":
    create_missing_tables()
