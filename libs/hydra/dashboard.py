"""
HYDRA 3.0 - Terminal Dashboard

Real-time terminal dashboard showing:
- Engine rankings and performance
- Exploration stats (curiosity/focus balance)
- Safety status (circuit breakers, guardian)
- Internet search log
- Win rate chart (ASCII using plotext if available)

Usage:
    python -m libs.hydra.dashboard
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ASCII colors (ANSI escape codes)
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_box(title: str, content: str, width: int = 60) -> str:
    """Create ASCII box around content."""
    lines = content.split('\n')
    box_lines = []

    # Top border
    box_lines.append(f"╔{'═' * (width - 2)}╗")
    box_lines.append(f"║ {BOLD}{title}{RESET}" + " " * (width - len(title) - 4) + " ║")
    box_lines.append(f"╠{'═' * (width - 2)}╣")

    # Content
    for line in lines:
        padded = line[:width - 4].ljust(width - 4)
        box_lines.append(f"║ {padded} ║")

    # Bottom border
    box_lines.append(f"╚{'═' * (width - 2)}╝")

    return '\n'.join(box_lines)


class HydraDashboard:
    """
    Terminal dashboard for HYDRA 3.0.

    Displays live system status in the terminal.
    """

    def __init__(self):
        self.refresh_interval = 5  # seconds
        self.running = False

    def _get_rankings_section(self) -> str:
        """Get engine rankings display."""
        try:
            from .engine_portfolio import get_tournament_manager
            manager = get_tournament_manager()
            rankings = manager.calculate_rankings()

            if not rankings:
                return "No rankings data available"

            lines = []
            for i, (name, stats) in enumerate(rankings):
                rank = i + 1
                if rank == 1:
                    indicator = f"{GREEN}👑{RESET}"
                elif rank == 4:
                    indicator = f"{RED}💀{RESET}"
                else:
                    indicator = "  "

                wr = stats.win_rate * 100
                pnl = stats.total_pnl_usd

                # Color code based on performance
                if pnl > 0:
                    pnl_color = GREEN
                elif pnl < 0:
                    pnl_color = RED
                else:
                    pnl_color = RESET

                lines.append(
                    f"{indicator} #{rank} Engine {name} | "
                    f"WR: {wr:5.1f}% | "
                    f"P&L: {pnl_color}${pnl:+8.2f}{RESET} | "
                    f"Trades: {stats.total_trades:3d}"
                )

            return '\n'.join(lines)

        except Exception as e:
            return f"Error loading rankings: {e}"

    def _get_exploration_stats(self) -> str:
        """Get exploration/curiosity stats."""
        try:
            from .improvement_tracker import get_improvement_tracker
            tracker = get_improvement_tracker()
            trends = tracker.get_all_trends()

            lines = []
            for engine, trend in trends.items():
                status = trend.get("trend", "UNKNOWN")

                if status == "IMPROVING":
                    color = GREEN
                    icon = "↑"
                elif status == "DECLINING":
                    color = RED
                    icon = "↓"
                elif status == "STAGNANT":
                    color = YELLOW
                    icon = "→"
                else:
                    color = RESET
                    icon = "?"

                wr_trend = trend.get("win_rate_trend", 0) * 100
                lines.append(f"Engine {engine}: {color}{icon} {status}{RESET} (WR: {wr_trend:+.1f}%)")

            return '\n'.join(lines) if lines else "No exploration data"

        except Exception as e:
            return f"Error: {e}"

    def _get_safety_status(self) -> str:
        """Get safety/guardian status."""
        lines = []

        # Guardian status
        try:
            from .guardian import get_guardian
            guardian = get_guardian()
            status = guardian.get_status()

            if status.get("emergency_shutdown_active"):
                lines.append(f"{RED}⚠️  EMERGENCY SHUTDOWN ACTIVE{RESET}")
                lines.append(f"   Until: {status.get('emergency_shutdown_until', 'unknown')}")
            elif not status.get("trading_allowed"):
                lines.append(f"{YELLOW}⚠️  Trading Paused{RESET}")
            else:
                lines.append(f"{GREEN}✓ Guardian: NORMAL{RESET}")

            # Circuit breaker
            losses = status.get("consecutive_losses", 0)
            if losses >= 3:
                lines.append(f"{YELLOW}⚠️  Circuit Breaker: {losses} losses (50% size){RESET}")
            else:
                lines.append(f"   Consecutive losses: {losses}")

            # Drawdown
            dd = status.get("current_drawdown_percent", 0)
            if dd > 5:
                lines.append(f"{RED}⚠️  Drawdown: {dd:.1f}%{RESET}")
            elif dd > 3:
                lines.append(f"{YELLOW}   Drawdown: {dd:.1f}%{RESET}")
            else:
                lines.append(f"   Drawdown: {dd:.1f}%")

        except Exception as e:
            lines.append(f"Guardian status unavailable: {e}")

        # Mother AI status
        try:
            from .mother_ai import get_mother_ai
            mother = get_mother_ai()
            health = mother.get_health_status()

            if health.get("is_frozen"):
                lines.append(f"{RED}🧊 MOTHER AI FROZEN{RESET}")
                lines.append(f"   Reason: {health.get('failure_reason', 'unknown')[:30]}")
            elif health.get("is_healthy"):
                lines.append(f"{GREEN}✓ Mother AI: HEALTHY{RESET}")
            else:
                lines.append(f"{YELLOW}⚠️  Mother AI: DEGRADED{RESET}")

        except Exception as e:
            lines.append(f"Mother AI: {YELLOW}Not initialized{RESET}")

        return '\n'.join(lines)

    def _get_search_log(self) -> str:
        """Get recent internet search results."""
        try:
            from .data_feeds import get_historical_storage, DataType
            storage = get_historical_storage()
            records = storage.query(data_type=DataType.SEARCH_RESULT, hours_back=24, limit=5)

            if not records:
                return "No recent searches"

            lines = []
            for record in records[:5]:
                data = record.data
                query = data.get("query", "unknown")[:30]
                count = data.get("result_count", 0)
                ts = record.timestamp.strftime("%H:%M")
                lines.append(f"[{ts}] {query} ({count} results)")

            return '\n'.join(lines)

        except Exception as e:
            return f"Search log unavailable: {e}"

    def _get_wr_chart(self) -> str:
        """Get ASCII win rate chart using plotext."""
        try:
            import plotext as plt
        except ImportError:
            return "plotext not installed (pip install plotext)"

        try:
            from .data_feeds import get_historical_storage, DataType
            storage = get_historical_storage()

            # Get engine signals for last 24h
            lines = []

            for engine in ["A", "B", "C", "D"]:
                records = storage.query(
                    data_type=DataType.ENGINE_SIGNAL,
                    symbol=engine,
                    hours_back=24,
                    limit=20
                )

                if records:
                    wins = sum(1 for r in records if r.data.get("won", False))
                    wr = wins / len(records) * 100 if records else 0
                    bar_len = int(wr / 5)  # Scale to ~20 chars max
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    lines.append(f"Engine {engine}: [{bar}] {wr:.0f}%")
                else:
                    lines.append(f"Engine {engine}: [░░░░░░░░░░░░░░░░░░░░] 0%")

            return '\n'.join(lines)

        except Exception as e:
            return f"Chart error: {e}"

    def _render_dashboard(self):
        """Render full dashboard."""
        clear_screen()

        # Header
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}")
        print(f"{BOLD}{CYAN}                    HYDRA 3.0 DASHBOARD{RESET}")
        print(f"{BOLD}{CYAN}                    {now}{RESET}")
        print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}\n")

        # Rankings (STEP 28)
        print(format_box("ENGINE RANKINGS", self._get_rankings_section()))
        print()

        # Side by side: Exploration + Safety
        exploration = self._get_exploration_stats()
        safety = self._get_safety_status()

        print(f"{BOLD}EXPLORATION STATS{RESET}          {BOLD}SAFETY STATUS{RESET}")
        print("─" * 30 + "  " + "─" * 30)

        exp_lines = exploration.split('\n')
        safety_lines = safety.split('\n')
        max_lines = max(len(exp_lines), len(safety_lines))

        for i in range(max_lines):
            exp = exp_lines[i] if i < len(exp_lines) else ""
            saf = safety_lines[i] if i < len(safety_lines) else ""
            print(f"{exp:<30}  {saf}")

        print()

        # Search log (STEP 31)
        print(format_box("INTERNET SEARCH LOG", self._get_search_log(), width=65))
        print()

        # WR Chart (STEP 32)
        print(format_box("WIN RATE (24h)", self._get_wr_chart(), width=65))

        # Footer
        print(f"\n{YELLOW}Press Ctrl+C to exit | Refresh: {self.refresh_interval}s{RESET}")

    def run(self):
        """Run dashboard in loop."""
        self.running = True

        try:
            while self.running:
                self._render_dashboard()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print(f"\n{GREEN}Dashboard stopped.{RESET}")

    def render_once(self) -> str:
        """Render dashboard once and return as string."""
        lines = []
        lines.append("=" * 60)
        lines.append("HYDRA 3.0 STATUS")
        lines.append("=" * 60)
        lines.append("")
        lines.append("ENGINE RANKINGS:")
        lines.append(self._get_rankings_section())
        lines.append("")
        lines.append("SAFETY STATUS:")
        lines.append(self._get_safety_status())
        lines.append("")
        lines.append("EXPLORATION:")
        lines.append(self._get_exploration_stats())
        return '\n'.join(lines)


def main():
    """Main entry point."""
    dashboard = HydraDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
