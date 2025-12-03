#!/usr/bin/env python3
"""
HYDRA 4.0 - Simple Terminal Dashboard

Pure terminal output with ANSI colors.
No dependencies other than standard library + HYDRA libs.

Run: python scripts/hydra_simple_dashboard.py
Web: ttyd -p 7681 python scripts/hydra_simple_dashboard.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
DIM = "\033[2m"


def clear():
    os.system('clear')


def header():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{BOLD}{CYAN}")
    print("=" * 60)
    print(f"           HYDRA 4.0 LIVE DASHBOARD")
    print(f"              {now}")
    print("=" * 60)
    print(f"{RESET}")


def engine_rankings():
    print(f"\n{BOLD}ENGINE RANKINGS{RESET}")
    print("-" * 50)

    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()

        specialties = {"A": "Liquidation", "B": "Funding", "C": "Orderbook", "D": "Regime"}

        for i, (name, stats) in enumerate(rankings):
            rank = i + 1
            if rank == 1:
                icon = f"{GREEN}#1{RESET}"
            elif rank == 4:
                icon = f"{RED}#4{RESET}"
            else:
                icon = f"#{rank}"

            wr = stats.win_rate * 100
            pnl = stats.total_pnl_usd
            pnl_color = GREEN if pnl >= 0 else RED

            print(f"  {icon}  Engine {name:<2} | {specialties.get(name, '?'):12} | "
                  f"WR: {wr:5.1f}% | PnL: {pnl_color}${pnl:+8.2f}{RESET} | "
                  f"Trades: {stats.total_trades:3d}")

    except Exception as e:
        print(f"  {RED}Error: {e}{RESET}")


def safety_status():
    print(f"\n{BOLD}SAFETY STATUS{RESET}")
    print("-" * 50)

    # Guardian
    try:
        from libs.hydra.guardian import get_guardian
        guardian = get_guardian()
        status = guardian.get_status()

        if status.get("emergency_shutdown_active"):
            print(f"  {RED}EMERGENCY SHUTDOWN ACTIVE{RESET}")
        elif not status.get("trading_allowed"):
            print(f"  {YELLOW}Trading Paused{RESET}")
        else:
            print(f"  {GREEN}Guardian: NORMAL{RESET}")

        losses = status.get("consecutive_losses", 0)
        if losses >= 3:
            print(f"  {YELLOW}Circuit Breaker: {losses} losses (50% size){RESET}")
        else:
            print(f"  Consecutive Losses: {losses}/3")

        dd = status.get("current_drawdown_percent", 0)
        dd_color = RED if dd > 5 else YELLOW if dd > 3 else RESET
        print(f"  {dd_color}Drawdown: {dd:.1f}%{RESET}")

    except Exception as e:
        print(f"  {DIM}Guardian: {e}{RESET}")

    # Mother AI
    try:
        from libs.hydra.mother_ai import get_mother_ai
        mother = get_mother_ai()
        health = mother.get_health_status()

        if health.get("is_frozen"):
            print(f"  {RED}MOTHER AI: FROZEN{RESET}")
        elif health.get("is_healthy"):
            print(f"  {GREEN}Mother AI: OK{RESET}")
        else:
            print(f"  {YELLOW}Mother AI: DEGRADED{RESET}")
    except:
        print(f"  {DIM}Mother AI: Not loaded{RESET}")


def improvement_trends():
    print(f"\n{BOLD}ENGINE TRENDS (7-day){RESET}")
    print("-" * 50)

    try:
        from libs.hydra.improvement_tracker import get_improvement_tracker
        tracker = get_improvement_tracker()
        trends = tracker.get_all_trends()

        for engine, trend in trends.items():
            status = trend.get("trend", "UNKNOWN")

            if status == "IMPROVING":
                icon = f"{GREEN}+{RESET}"
            elif status == "DECLINING":
                icon = f"{RED}-{RESET}"
            elif status == "STAGNANT":
                icon = f"{YELLOW}={RESET}"
            else:
                icon = "?"

            wr_trend = trend.get("win_rate_trend", 0) * 100
            print(f"  Engine {engine}: [{icon}] {status:10} (WR: {wr_trend:+5.1f}%)")

    except:
        print(f"  {DIM}No trend data{RESET}")


def data_feeds():
    print(f"\n{BOLD}DATA FEEDS{RESET}")
    print("-" * 50)

    try:
        from libs.hydra.data_feeds import get_historical_storage
        storage = get_historical_storage()
        stats = storage.get_stats()

        print(f"  Records: {stats['total_records']:,}")
        print(f"  DB Size: {stats['db_size_mb']:.1f} MB")
        print(f"  Retention: 30 days")

        if stats.get('by_type'):
            for dtype, count in list(stats['by_type'].items())[:4]:
                print(f"    {dtype}: {count:,}")
    except Exception as e:
        print(f"  {DIM}Data feeds: {e}{RESET}")


def footer():
    print(f"\n{DIM}Ctrl+C to exit | Refresh: 5s{RESET}")
    print("-" * 60)


def main():
    """Run dashboard loop."""
    try:
        while True:
            clear()
            header()
            engine_rankings()
            safety_status()
            improvement_trends()
            data_feeds()
            footer()
            time.sleep(5)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Dashboard stopped.{RESET}")


if __name__ == "__main__":
    main()
