#!/usr/bin/env python3
"""
HYDRA 3.0 Terminal Monitor

TRUE terminal application (like htop/top) that outputs to stdout.
Can be served via ttyd for web viewing.

Usage:
  # Direct terminal:
  python3 hydra_monitor.py

  # Via ttyd (web terminal):
  ttyd -p 8080 python3 hydra_monitor.py
"""

import os
import sys
import time
import sqlite3
from datetime import datetime
from typing import Dict, List, Any

# ANSI escape codes for terminal formatting
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'

    # Bright colors
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_CYAN = '\033[96m'

# Terminal control sequences
def clear_screen():
    """Clear the terminal screen."""
    print('\033[2J', end='')

def move_cursor(row: int, col: int):
    """Move cursor to specified position."""
    print(f'\033[{row};{col}H', end='')

def hide_cursor():
    """Hide the cursor."""
    print('\033[?25l', end='')

def show_cursor():
    """Show the cursor."""
    print('\033[?25h', end='')


class HydraMonitor:
    """Terminal monitor for HYDRA 3.0 system."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            from libs.hydra.config import HYDRA_DB_FILE
            db_path = str(HYDRA_DB_FILE)
        self.db_path = db_path
        self.refresh_interval = 5  # seconds

    def get_engine_stats(self) -> List[Dict]:
        """Get performance stats for all engines."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = []
        for engine in ["A", "B", "C", "D"]:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl_usd) as total_pnl
                FROM hydra_trades
                WHERE gladiator = ? AND exit_time IS NOT NULL
            """, (engine,))

            result = cursor.fetchone()
            total = result[0] or 0
            wins = result[1] or 0
            win_rate = (wins / total * 100) if total > 0 else 0.0

            stats.append({
                "engine": engine,
                "total_trades": total,
                "wins": wins,
                "losses": result[2] or 0,
                "win_rate": win_rate,
                "total_pnl": result[3] or 0.0
            })

        conn.close()

        # Sort by P&L for rankings
        stats_sorted = sorted(stats, key=lambda x: x["total_pnl"], reverse=True)

        # Add rank
        for rank, s in enumerate(stats_sorted, 1):
            s["rank"] = rank

        return stats_sorted

    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                timestamp,
                symbol,
                gladiator,
                direction,
                pnl_usd,
                exit_reason
            FROM hydra_trades
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT ?
        """, (limit,))

        trades = []
        for row in cursor.fetchall():
            trades.append({
                "timestamp": row[0],
                "symbol": row[1],
                "engine": row[2],
                "direction": row[3],
                "pnl": row[4],
                "reason": row[5]
            })

        conn.close()
        return trades

    def get_system_health(self) -> Dict:
        """Get system health metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM strategies")
        total_strategies = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM hydra_trades WHERE exit_time IS NULL")
        active_trades = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM hydra_trades
            WHERE timestamp > datetime('now', '-24 hours')
        """)
        trades_24h = cursor.fetchone()[0]

        conn.close()

        return {
            "total_strategies": total_strategies,
            "active_trades": active_trades,
            "trades_24h": trades_24h
        }

    def print_header(self):
        """Print dashboard header."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"{Colors.BG_GREEN}{Colors.BLACK}{Colors.BOLD}")
        print(f" HYDRA 3.0 - Tournament Dashboard {' ' * 30} {now} ")
        print(f"{Colors.RESET}")
        print()

    def print_engine_rankings(self, stats: List[Dict]):
        """Print engine performance rankings."""
        print(f"{Colors.BOLD}{Colors.CYAN}ENGINE PERFORMANCE{Colors.RESET}")
        print(f"{Colors.BOLD}─{Colors.RESET}" * 80)

        # Header
        print(f"{Colors.BOLD}{'Rank':<6} {'Engine':<10} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Status':<10}{Colors.RESET}")
        print(f"{Colors.BOLD}─{Colors.RESET}" * 80)

        # Weight distribution
        weights = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10}

        for s in stats:
            rank = s['rank']
            engine = s['engine']
            trades = s['total_trades']
            win_rate = s['win_rate']
            pnl = s['total_pnl']
            weight = weights.get(rank, 0.10)

            # Color coding
            pnl_color = Colors.BRIGHT_GREEN if pnl > 0 else Colors.BRIGHT_RED
            rank_color = Colors.BRIGHT_YELLOW if rank == 1 else Colors.WHITE

            rank_str = f"#{rank}"
            engine_str = f"Engine {engine}"
            trades_str = f"{trades}"
            wr_str = f"{win_rate:.1f}%"
            pnl_str = f"${pnl:+.2f}"
            status_str = f"Weight: {weight:.0%}"

            print(f"{rank_color}{rank_str:<6}{Colors.RESET} "
                  f"{engine_str:<10} "
                  f"{trades_str:<8} "
                  f"{wr_str:<10} "
                  f"{pnl_color}{pnl_str:<12}{Colors.RESET} "
                  f"{status_str:<10}")

        print()

    def print_system_health(self, health: Dict):
        """Print system health metrics."""
        print(f"{Colors.BOLD}{Colors.MAGENTA}SYSTEM HEALTH{Colors.RESET}")
        print(f"{Colors.BOLD}─{Colors.RESET}" * 80)

        print(f"  Total Strategies: {Colors.BRIGHT_CYAN}{health['total_strategies']}{Colors.RESET}")
        print(f"  Active Trades:    {Colors.BRIGHT_YELLOW}{health['active_trades']}{Colors.RESET}")
        print(f"  Trades (24h):     {Colors.BRIGHT_GREEN}{health['trades_24h']}{Colors.RESET}")
        print()

    def print_recent_trades(self, trades: List[Dict]):
        """Print recent trades."""
        print(f"{Colors.BOLD}{Colors.BLUE}RECENT TRADES{Colors.RESET}")
        print(f"{Colors.BOLD}─{Colors.RESET}" * 80)

        # Header
        print(f"{Colors.BOLD}{'Time':<12} {'Engine':<8} {'Symbol':<10} {'Dir':<6} {'P&L':<12} {'Reason':<20}{Colors.RESET}")
        print(f"{Colors.BOLD}─{Colors.RESET}" * 80)

        for trade in trades[:10]:
            # Format timestamp
            try:
                ts = datetime.fromisoformat(trade['timestamp'])
                time_str = ts.strftime("%H:%M:%S")
            except:
                time_str = "N/A"

            engine = trade['engine']
            symbol = trade['symbol']
            direction = trade['direction']
            pnl = trade['pnl']
            reason = (trade['reason'] or "N/A")[:18]

            # Color coding
            pnl_color = Colors.BRIGHT_GREEN if pnl > 0 else Colors.BRIGHT_RED
            dir_color = Colors.GREEN if direction == "BUY" else Colors.RED

            print(f"{time_str:<12} "
                  f"Eng {engine:<5} "
                  f"{symbol:<10} "
                  f"{dir_color}{direction:<6}{Colors.RESET} "
                  f"{pnl_color}${pnl:+8.2f}{Colors.RESET} "
                  f"{reason:<20}")

        print()

    def print_footer(self):
        """Print dashboard footer."""
        print(f"{Colors.BOLD}─{Colors.RESET}" * 80)
        print(f"{Colors.YELLOW}Press Ctrl+C to exit | Auto-refresh every {self.refresh_interval}s{Colors.RESET}")

    def render_dashboard(self):
        """Render the complete dashboard."""
        # Get data
        stats = self.get_engine_stats()
        health = self.get_system_health()
        trades = self.get_recent_trades(10)

        # Clear screen and move to top
        clear_screen()
        move_cursor(1, 1)

        # Print all sections
        self.print_header()
        self.print_engine_rankings(stats)
        self.print_system_health(health)
        self.print_recent_trades(trades)
        self.print_footer()

        # Flush stdout to ensure immediate display
        sys.stdout.flush()

    def run(self):
        """Run the terminal monitor with auto-refresh."""
        try:
            # Hide cursor for cleaner display
            hide_cursor()

            while True:
                self.render_dashboard()
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            # Clean exit on Ctrl+C
            show_cursor()
            clear_screen()
            print(f"\n{Colors.YELLOW}HYDRA Monitor stopped by user{Colors.RESET}\n")
        except Exception as e:
            # Clean exit on error
            show_cursor()
            clear_screen()
            print(f"\n{Colors.RED}Error: {e}{Colors.RESET}\n")


def main():
    """Main entry point."""
    monitor = HydraMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
