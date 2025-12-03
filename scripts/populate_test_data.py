#!/usr/bin/env python3
"""
Populate HYDRA database with test data for dashboard testing
"""

import sqlite3
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path

# Add project root to path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

from libs.hydra.config import HYDRA_DB_FILE
DB_PATH = str(HYDRA_DB_FILE)

def populate_test_data():
    """Add test data to hydra.db"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Add test strategies
    print("Adding test strategies...")
    for i in range(50):
        cursor.execute("""
            INSERT INTO strategies (
                id, strategy_hash, engine, asset, created_at,
                entry_confidence, direction, risk_reward_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"strat_{i:04d}",
            f"hash_{i:08x}",
            random.choice(["A", "B", "C", "D"]),
            random.choice(["BTC-USD", "ETH-USD", "SOL-USD"]),
            (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
            random.uniform(0.6, 0.95),
            random.choice(["BUY", "SELL"]),
            random.uniform(1.5, 3.5)
        ))

    # Add test trades with varying outcomes
    print("Adding test trades...")
    base_time = datetime.now() - timedelta(days=7)

    # Engine A: 65% win rate, good P&L
    for i in range(30):
        engine = "A"
        is_win = random.random() < 0.65
        pnl = random.uniform(10, 50) if is_win else random.uniform(-30, -10)

        entry_time = base_time + timedelta(hours=i * 6)
        exit_time = entry_time + timedelta(hours=random.randint(1, 12))

        cursor.execute("""
            INSERT INTO hydra_trades (
                trade_id, timestamp, symbol, gladiator, direction,
                entry_price, exit_price, pnl, exit_reason, exit_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{engine}_trade_{i:04d}",
            entry_time.isoformat(),
            random.choice(["BTC-USD", "ETH-USD", "SOL-USD"]),
            engine,
            random.choice(["BUY", "SELL"]),
            random.uniform(30000, 50000),
            random.uniform(30000, 50000),
            pnl,
            "SL_HIT" if not is_win else "TP_HIT",
            exit_time.isoformat()
        ))

    # Engine B: 58% win rate, moderate P&L
    for i in range(25):
        engine = "B"
        is_win = random.random() < 0.58
        pnl = random.uniform(8, 40) if is_win else random.uniform(-25, -8)

        entry_time = base_time + timedelta(hours=i * 7)
        exit_time = entry_time + timedelta(hours=random.randint(1, 10))

        cursor.execute("""
            INSERT INTO hydra_trades (
                trade_id, timestamp, symbol, gladiator, direction,
                entry_price, exit_price, pnl, exit_reason, exit_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{engine}_trade_{i:04d}",
            entry_time.isoformat(),
            random.choice(["BTC-USD", "ETH-USD", "SOL-USD"]),
            engine,
            random.choice(["BUY", "SELL"]),
            random.uniform(30000, 50000),
            random.uniform(30000, 50000),
            pnl,
            "SL_HIT" if not is_win else "TP_HIT",
            exit_time.isoformat()
        ))

    # Engine C: 48% win rate, negative P&L
    for i in range(20):
        engine = "C"
        is_win = random.random() < 0.48
        pnl = random.uniform(5, 30) if is_win else random.uniform(-35, -12)

        entry_time = base_time + timedelta(hours=i * 8)
        exit_time = entry_time + timedelta(hours=random.randint(1, 15))

        cursor.execute("""
            INSERT INTO hydra_trades (
                trade_id, timestamp, symbol, gladiator, direction,
                entry_price, exit_price, pnl, exit_reason, exit_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{engine}_trade_{i:04d}",
            entry_time.isoformat(),
            random.choice(["BTC-USD", "ETH-USD", "SOL-USD"]),
            engine,
            random.choice(["BUY", "SELL"]),
            random.uniform(30000, 50000),
            random.uniform(30000, 50000),
            pnl,
            "SL_HIT" if not is_win else "TP_HIT",
            exit_time.isoformat()
        ))

    # Engine D: 55% win rate, positive P&L
    for i in range(28):
        engine = "D"
        is_win = random.random() < 0.55
        pnl = random.uniform(12, 45) if is_win else random.uniform(-28, -10)

        entry_time = base_time + timedelta(hours=i * 6.5)
        exit_time = entry_time + timedelta(hours=random.randint(1, 11))

        cursor.execute("""
            INSERT INTO hydra_trades (
                trade_id, timestamp, symbol, gladiator, direction,
                entry_price, exit_price, pnl, exit_reason, exit_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{engine}_trade_{i:04d}",
            entry_time.isoformat(),
            random.choice(["BTC-USD", "ETH-USD", "SOL-USD"]),
            engine,
            random.choice(["BUY", "SELL"]),
            random.uniform(30000, 50000),
            random.uniform(30000, 50000),
            pnl,
            "SL_HIT" if not is_win else "TP_HIT",
            exit_time.isoformat()
        ))

    conn.commit()
    conn.close()

    print("✅ Test data populated successfully!")
    print("\nSummary:")
    print("  - 50 strategies")
    print("  - 30 trades for Engine A (65% WR)")
    print("  - 25 trades for Engine B (58% WR)")
    print("  - 20 trades for Engine C (48% WR)")
    print("  - 28 trades for Engine D (55% WR)")
    print("  - Total: 103 trades")

if __name__ == "__main__":
    populate_test_data()
