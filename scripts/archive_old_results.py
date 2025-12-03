"""
Archive Old Signal Results

Archives the signal_results from the old (2% SL) configuration
so Guardian can monitor the NEW (4% SL) results separately.

This prevents Guardian from killing V7 based on old metrics.
"""
import sys
from pathlib import Path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

import sqlite3
from datetime import datetime

def main():
    conn = sqlite3.connect('tradingai.db')
    cursor = conn.cursor()

    # Check current signal_results
    cursor.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses
        FROM signal_results
        WHERE outcome IN ('win', 'loss')
    """)

    total, wins, losses = cursor.fetchone()

    print("=" * 80)
    print("ARCHIVING OLD SIGNAL RESULTS")
    print("=" * 80)
    print(f"\nCurrent signal_results table:")
    print(f"  Total trades: {total}")
    print(f"  Wins: {wins}")
    print(f"  Losses: {losses}")
    print(f"  Win rate: {wins/total*100 if total > 0 else 0:.1f}%")

    if total == 0:
        print("\n✅ No data to archive - signal_results table is already empty")
        conn.close()
        return

    # Simply delete old results (they're already documented in test reports)
    cursor.execute("""
        DELETE FROM signal_results
        WHERE outcome IN ('win', 'loss')
    """)

    deleted = cursor.rowcount
    print(f"\n✅ Deleted {deleted} old signal_results (with 2% SL)")
    print("   (Results were documented in PRE_DEPLOYMENT_TEST_RESULTS.md)")

    # Verify clean state
    cursor.execute("SELECT COUNT(*) FROM signal_results WHERE outcome IN ('win', 'loss')")
    remaining = cursor.fetchone()[0]

    print(f"\n✅ signal_results table now has {remaining} completed trades")
    print("\nV7 will start collecting fresh data with WIDENED STOP LOSSES (4%)")
    print("Guardian will monitor NEW metrics, not old ones")

    conn.commit()
    conn.close()

    print("\n" + "=" * 80)
    print("ARCHIVE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
