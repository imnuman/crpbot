#!/usr/bin/env python3
"""
CLI tool to manually record trade entries/exits

Usage:
    python record_trade.py entry <signal_id> <price>
    python record_trade.py exit <signal_id> <price> [reason]
    python record_trade.py stats
"""

import sys
from datetime import datetime
from pathlib import Path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

from libs.tracking.performance_tracker import PerformanceTracker

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    tracker = PerformanceTracker()
    command = sys.argv[1].lower()

    if command == 'entry':
        if len(sys.argv) < 4:
            print("Usage: record_trade.py entry <signal_id> <price>")
            sys.exit(1)

        signal_id = int(sys.argv[2])
        price = float(sys.argv[3])
        tracker.record_entry(signal_id, price, datetime.now())

    elif command == 'exit':
        if len(sys.argv) < 4:
            print("Usage: record_trade.py exit <signal_id> <price> [reason]")
            sys.exit(1)

        signal_id = int(sys.argv[2])
        price = float(sys.argv[3])
        reason = sys.argv[4] if len(sys.argv) > 4 else 'manual'
        tracker.record_exit(signal_id, price, datetime.now(), reason)

    elif command == 'stats':
        stats = tracker.get_win_rate(days=30)
        print('\n📊 Performance Stats (Last 30 Days):')
        print(f'   Total Trades: {stats["total_trades"]}')
        print(f'   Wins: {stats["wins"]} | Losses: {stats["losses"]}')
        print(f'   Win Rate: {stats["win_rate"]:.1f}%')
        print(f'   Avg Win: {stats["avg_win"]:+.2f}%')
        print(f'   Avg Loss: {stats["avg_loss"]:+.2f}%')
        print(f'   Profit Factor: {stats["profit_factor"]:.2f}')

        print('\n📈 Open Positions:')
        positions = tracker.get_open_positions()
        if positions:
            for pos in positions:
                print(f'   Signal #{pos["signal_id"]}: {pos["symbol"]} {pos["direction"]} @ ${pos["entry_price"]:,.2f}')
        else:
            print('   None')

        print('\n📜 Recent Trades:')
        trades = tracker.get_recent_trades(limit=10)
        if trades:
            for trade in trades:
                outcome_emoji = '✅' if trade['outcome'] == 'win' else '❌' if trade['outcome'] == 'loss' else '➖'
                print(f'   {outcome_emoji} {trade["symbol"]} {trade["direction"]}: {trade["pnl_percent"]:+.2f}% ({trade["hold_duration_minutes"]}m)')
        else:
            print('   No closed trades yet')

    else:
        print(f'Unknown command: {command}')
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()
