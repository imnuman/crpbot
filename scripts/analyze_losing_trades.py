#!/usr/bin/env python3
"""
Analyze Losing Trades - Find Root Cause of 0.59% Win Rate
"""
import sqlite3
from datetime import datetime

def analyze_losing_trades():
    """Comprehensive analysis of losing trades"""

    conn = sqlite3.connect('tradingai.db')
    c = conn.cursor()

    print("=" * 80)
    print("LOSING TRADE ANALYSIS - Finding Root Cause of 0.59% Win Rate")
    print("=" * 80)
    print()

    # 1. Worst 10 losing trades
    print("1. WORST 10 LOSING TRADES")
    print("-" * 80)
    c.execute("""
    SELECT
        symbol,
        direction,
        ROUND(entry_price, 2) as entry,
        ROUND(exit_price, 2) as exit,
        ROUND(pnl_percent, 2) as pnl,
        exit_reason,
        datetime(entry_time, 'localtime') as entry_time
    FROM signal_results
    ORDER BY pnl_percent ASC
    LIMIT 10
    """)

    print(f"{'Symbol':<10} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'P&L%':<8} {'Reason':<15} {'Time':<20}")
    print("-" * 80)
    for row in c.fetchall():
        print(f"{row[0]:<10} {row[1]:<5} {row[2]:<10} {row[3]:<10} {row[4]:<8} {row[5]:<15} {row[6]:<20}")
    print()

    # 2. Exit reason breakdown
    print("2. EXIT REASON BREAKDOWN")
    print("-" * 80)
    c.execute("""
    SELECT
        exit_reason,
        COUNT(*) as count,
        ROUND(AVG(pnl_percent), 2) as avg_pnl,
        ROUND(MIN(pnl_percent), 2) as worst_pnl,
        ROUND(MAX(pnl_percent), 2) as best_pnl
    FROM signal_results
    GROUP BY exit_reason
    ORDER BY count DESC
    """)

    print(f"{'Exit Reason':<20} {'Count':<10} {'Avg P&L%':<12} {'Worst%':<10} {'Best%':<10}")
    print("-" * 80)
    for row in c.fetchall():
        print(f"{row[0]:<20} {row[1]:<10} {row[2]:<12} {row[3]:<10} {row[4]:<10}")
    print()

    # 3. Performance by symbol
    print("3. PERFORMANCE BY SYMBOL")
    print("-" * 80)
    c.execute("""
    SELECT
        symbol,
        COUNT(*) as trades,
        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
        ROUND(AVG(pnl_percent), 2) as avg_pnl,
        ROUND(SUM(pnl_percent), 2) as total_pnl
    FROM signal_results
    GROUP BY symbol
    ORDER BY avg_pnl DESC
    """)

    print(f"{'Symbol':<10} {'Trades':<10} {'Wins':<8} {'Win%':<10} {'Avg P&L%':<12} {'Total P&L%':<12}")
    print("-" * 80)
    for row in c.fetchall():
        print(f"{row[0]:<10} {row[1]:<10} {row[2]:<8} {row[3]:<10} {row[4]:<12} {row[5]:<12}")
    print()

    # 4. Performance by direction
    print("4. PERFORMANCE BY DIRECTION")
    print("-" * 80)
    c.execute("""
    SELECT
        direction,
        COUNT(*) as trades,
        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
        ROUND(AVG(pnl_percent), 2) as avg_pnl
    FROM signal_results
    GROUP BY direction
    ORDER BY avg_pnl DESC
    """)

    print(f"{'Direction':<12} {'Trades':<10} {'Wins':<8} {'Win%':<10} {'Avg P&L%':<12}")
    print("-" * 80)
    for row in c.fetchall():
        print(f"{row[0]:<12} {row[1]:<10} {row[2]:<8} {row[3]:<10} {row[4]:<12}")
    print()

    # 5. Strategy comparison (if available)
    print("5. PERFORMANCE BY STRATEGY")
    print("-" * 80)
    c.execute("""
    SELECT
        s.strategy,
        COUNT(sr.id) as trades,
        SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) / COUNT(sr.id), 1) as win_rate,
        ROUND(AVG(sr.pnl_percent), 2) as avg_pnl,
        ROUND(SUM(sr.pnl_percent), 2) as total_pnl
    FROM signals s
    LEFT JOIN signal_results sr ON s.id = sr.signal_id
    WHERE sr.id IS NOT NULL
    GROUP BY s.strategy
    ORDER BY trades DESC
    """)

    print(f"{'Strategy':<35} {'Trades':<10} {'Wins':<8} {'Win%':<10} {'Avg P&L%':<12} {'Total P&L%':<12}")
    print("-" * 80)
    for row in c.fetchall():
        strategy = row[0] if row[0] else "unknown"
        print(f"{strategy:<35} {row[1]:<10} {row[2]:<8} {row[3]:<10} {row[4]:<12} {row[5]:<12}")
    print()

    # 6. Overall summary
    print("6. OVERALL SUMMARY")
    print("-" * 80)
    c.execute("""
    SELECT
        COUNT(*) as total_trades,
        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
        ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate,
        ROUND(AVG(pnl_percent), 2) as avg_pnl,
        ROUND(SUM(pnl_percent), 2) as total_pnl,
        ROUND(MIN(pnl_percent), 2) as worst_loss,
        ROUND(MAX(pnl_percent), 2) as best_win
    FROM signal_results
    """)

    row = c.fetchone()
    print(f"Total Trades:    {row[0]}")
    print(f"Wins:            {row[1]} ({row[3]:.2f}%)")
    print(f"Losses:          {row[2]}")
    print(f"Avg P&L:         {row[4]}%")
    print(f"Total P&L:       {row[5]}%")
    print(f"Worst Loss:      {row[6]}%")
    print(f"Best Win:        {row[7]}%")
    print()

    # 7. Recent trades (last 24 hours)
    print("7. RECENT TRADES (Last 24 Hours)")
    print("-" * 80)
    c.execute("""
    SELECT
        symbol,
        direction,
        ROUND(pnl_percent, 2) as pnl,
        outcome,
        exit_reason,
        datetime(entry_time, 'localtime') as time
    FROM signal_results
    WHERE entry_time > datetime('now', '-24 hours')
    ORDER BY entry_time DESC
    LIMIT 20
    """)

    print(f"{'Symbol':<10} {'Dir':<5} {'P&L%':<8} {'Outcome':<8} {'Reason':<15} {'Time':<20}")
    print("-" * 80)
    for row in c.fetchall():
        print(f"{row[0]:<10} {row[1]:<5} {row[2]:<8} {row[3]:<8} {row[4]:<15} {row[5]:<20}")
    print()

    conn.close()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    analyze_losing_trades()
