#!/usr/bin/env python3
"""
HYDRA 4.0 - Live Terminal Dashboard

Pure terminal dashboard with live updates using Rich library.
Run with: python scripts/hydra_dashboard.py
Share via web: ttyd -p 7681 python scripts/hydra_dashboard.py

Press Ctrl+C to exit.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import BarColumn, Progress, TextColumn
    from rich import box
except ImportError:
    print("Installing rich...")
    os.system("pip install rich")
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import BarColumn, Progress, TextColumn
    from rich import box

console = Console()

# Track uptime
START_TIME = datetime.now()


def get_engine_rankings():
    """Get engine rankings table."""
    table = Table(title="ENGINE RANKINGS", box=box.ROUNDED)
    table.add_column("Rank", style="cyan", justify="center", width=6)
    table.add_column("Engine", style="white", justify="center", width=8)
    table.add_column("Specialty", style="yellow", width=20)
    table.add_column("WR", style="green", justify="right", width=8)
    table.add_column("P&L", justify="right", width=12)
    table.add_column("Trades", justify="right", width=8)

    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()

        specialties = {
            "A": "Liquidation",
            "B": "Funding",
            "C": "Orderbook",
            "D": "Regime"
        }

        for i, (name, stats) in enumerate(rankings):
            rank = i + 1
            rank_str = f"#{rank}" if rank > 1 else "👑 #1"
            if rank == 4:
                rank_str = "💀 #4"

            wr = f"{stats.win_rate * 100:.1f}%"
            pnl = stats.total_pnl_usd
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_str = f"[{pnl_style}]${pnl:+.2f}[/]"

            table.add_row(
                rank_str,
                f"Engine {name}",
                specialties.get(name, "Unknown"),
                wr,
                pnl_str,
                str(stats.total_trades)
            )
    except Exception as e:
        table.add_row("?", "-", f"Error: {str(e)[:20]}", "-", "-", "-")

    return table


def get_safety_status():
    """Get safety status panel."""
    lines = []

    # Guardian status
    try:
        from libs.hydra.guardian import get_guardian
        guardian = get_guardian()
        status = guardian.get_status()

        if status.get("emergency_shutdown_active"):
            lines.append("[red]⚠️  EMERGENCY SHUTDOWN[/red]")
        elif not status.get("trading_allowed"):
            lines.append("[yellow]⚠️  Trading Paused[/yellow]")
        else:
            lines.append("[green]✓ Guardian: ACTIVE[/green]")

        losses = status.get("consecutive_losses", 0)
        if losses >= 3:
            lines.append(f"[yellow]Circuit: {losses} losses (50%)[/yellow]")
        else:
            lines.append(f"  Losses: {losses}/3")

        dd = status.get("current_drawdown_percent", 0)
        dd_color = "red" if dd > 5 else "yellow" if dd > 3 else "white"
        lines.append(f"[{dd_color}]  Drawdown: {dd:.1f}%[/]")

    except Exception as e:
        lines.append(f"Guardian: {str(e)[:30]}")

    # Mother AI status
    try:
        from libs.hydra.mother_ai import get_mother_ai
        mother = get_mother_ai()
        health = mother.get_health_status()

        if health.get("is_frozen"):
            lines.append("[red]🧊 MOTHER AI FROZEN[/red]")
        elif health.get("is_healthy"):
            lines.append("[green]✓ Mother AI: OK[/green]")
        else:
            lines.append("[yellow]⚠️  Mother AI: WARN[/yellow]")
    except:
        lines.append("[dim]Mother AI: Not loaded[/dim]")

    return Panel("\n".join(lines), title="SAFETY STATUS", box=box.ROUNDED)


def get_improvement_panel():
    """Get engine improvement panel."""
    lines = []

    try:
        from libs.hydra.improvement_tracker import get_improvement_tracker
        tracker = get_improvement_tracker()
        trends = tracker.get_all_trends()

        for engine, trend in trends.items():
            status = trend.get("trend", "UNKNOWN")
            if status == "IMPROVING":
                icon = "[green]↑[/green]"
            elif status == "DECLINING":
                icon = "[red]↓[/red]"
            elif status == "STAGNANT":
                icon = "[yellow]→[/yellow]"
            else:
                icon = "[dim]?[/dim]"

            wr_trend = trend.get("win_rate_trend", 0) * 100
            lines.append(f"Engine {engine}: {icon} {status} ({wr_trend:+.1f}%)")
    except:
        lines.append("[dim]No improvement data[/dim]")

    return Panel("\n".join(lines) if lines else "Loading...", title="TRENDS", box=box.ROUNDED)


def get_data_feeds_panel():
    """Get data feeds status."""
    lines = []

    try:
        from libs.hydra.data_feeds import get_historical_storage
        storage = get_historical_storage()
        stats = storage.get_stats()

        lines.append(f"Records: {stats['total_records']:,}")
        lines.append(f"DB Size: {stats['db_size_mb']:.1f} MB")

        if stats.get('by_type'):
            for dtype, count in list(stats['by_type'].items())[:3]:
                lines.append(f"  {dtype}: {count:,}")
    except Exception as e:
        lines.append(f"Error: {str(e)[:30]}")

    return Panel("\n".join(lines) if lines else "Loading...", title="DATA FEEDS", box=box.ROUNDED)


def get_engine_d_status():
    """Get Engine D special rules status."""
    lines = []

    try:
        from libs.hydra.engine_d_rules import get_engine_d_controller
        controller = get_engine_d_controller()
        state = controller.state

        if state.can_activate:
            lines.append("[green]Ready to activate[/green]")
        else:
            lines.append(f"[yellow]Cooldown: {state.days_until_available} days[/yellow]")

        expectancy = state.expectancy
        exp_color = "green" if expectancy >= 0 else "red"
        lines.append(f"[{exp_color}]Expectancy: {expectancy:+.2f}%[/]")

        lines.append(f"Activations: {state.total_activations}")
        lines.append(f"Win Rate: {state.win_rate:.0%}")
    except Exception as e:
        lines.append(f"[dim]Engine D: {str(e)[:20]}[/dim]")

    return Panel("\n".join(lines) if lines else "Loading...", title="ENGINE D (REGIME)", box=box.ROUNDED)


def get_validation_status():
    """Get trade validation status."""
    lines = []

    try:
        from libs.hydra.trade_validator import get_trade_validator
        validator = get_trade_validator()
        stats = validator.get_stats()

        lines.append(f"Validated: {stats.get('total_validated', 0)}")
        lines.append(f"Rejected: {stats.get('total_rejected', 0)}")

        if stats.get('rejection_reasons'):
            lines.append("")
            for reason, count in list(stats['rejection_reasons'].items())[:2]:
                lines.append(f"  {reason}: {count}")
    except:
        lines.append("[dim]Validator: Not loaded[/dim]")

    return Panel("\n".join(lines) if lines else "Loading...", title="VALIDATION", box=box.ROUNDED)


def get_header():
    """Get dashboard header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uptime = datetime.now() - START_TIME
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    return Panel(
        f"[bold cyan]HYDRA 4.0 DASHBOARD[/bold cyan]\n"
        f"{now}  |  Uptime: {uptime_str}",
        box=box.DOUBLE,
        style="cyan"
    )


def make_layout():
    """Create the dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="main", ratio=1),
        Layout(name="bottom", size=10),
        Layout(name="footer", size=3)
    )

    layout["main"].split_row(
        Layout(name="rankings", ratio=2),
        Layout(name="sidebar", ratio=1)
    )

    layout["sidebar"].split_column(
        Layout(name="safety"),
        Layout(name="trends")
    )

    layout["bottom"].split_row(
        Layout(name="engine_d"),
        Layout(name="validation"),
        Layout(name="feeds")
    )

    return layout


def update_dashboard(layout):
    """Update all dashboard components."""
    layout["header"].update(get_header())
    layout["rankings"].update(Panel(get_engine_rankings(), box=box.ROUNDED))
    layout["safety"].update(get_safety_status())
    layout["trends"].update(get_improvement_panel())
    layout["engine_d"].update(get_engine_d_status())
    layout["validation"].update(get_validation_status())
    layout["feeds"].update(get_data_feeds_panel())
    layout["footer"].update(
        Panel("[dim]Ctrl+C to exit | Auto-refresh: 5s | Web: ttyd -p 7681[/dim]", box=box.SIMPLE)
    )
    return layout


def main():
    """Run the live dashboard."""
    console.clear()

    layout = make_layout()

    try:
        with Live(layout, refresh_per_second=0.2, screen=True) as live:
            while True:
                update_dashboard(layout)
                time.sleep(5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


if __name__ == "__main__":
    main()
