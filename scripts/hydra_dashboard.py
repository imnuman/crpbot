#!/usr/bin/env python3
"""
HYDRA 4.0 - Live Terminal Dashboard

Enhanced dashboard with:
- Live cryptocurrency prices
- Engine rankings and performance
- AI agent communication log
- Safety status and trends

Run with: python scripts/hydra_dashboard.py
Share via web: ttyd -p 7682 python scripts/hydra_dashboard.py
"""

import os
import sys
import time
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
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
    from rich import box

console = Console()

# Track uptime and state
START_TIME = datetime.now()

# Communication log (in-memory ring buffer)
COMM_LOG: deque = deque(maxlen=50)

# Price cache
PRICE_CACHE: Dict[str, dict] = {}
LAST_PRICE_UPDATE = datetime.min

# Engine names for display
ENGINE_NAMES = {
    "A": ("DeepSeek", "cyan"),
    "B": ("Claude", "magenta"),
    "C": ("Grok", "yellow"),
    "D": ("Gemini", "green"),
}

ENGINE_SPECIALTIES = {
    "A": "Liquidation",
    "B": "Funding",
    "C": "Orderbook",
    "D": "Regime"
}


def log_communication(sender: str, receiver: str, msg_type: str, content: str):
    """Add a message to the communication log."""
    COMM_LOG.append({
        "time": datetime.now(),
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "content": content[:60]
    })


def get_live_prices() -> Dict[str, dict]:
    """Get live prices from Coinbase."""
    global PRICE_CACHE, LAST_PRICE_UPDATE

    # Only update every 10 seconds to avoid rate limits
    if (datetime.now() - LAST_PRICE_UPDATE).seconds < 10 and PRICE_CACHE:
        return PRICE_CACHE

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
               "ADA-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "DOT-USD"]

    try:
        from libs.data.coinbase_client import CoinbaseClient
        client = CoinbaseClient()

        for symbol in symbols:
            try:
                candles = client.get_candles(symbol, granularity="ONE_MINUTE", limit=2)
                if candles and len(candles) > 0:
                    latest = candles[0]
                    prev = candles[1] if len(candles) > 1 else candles[0]

                    price = float(latest.get("close", 0))
                    prev_price = float(prev.get("close", price))
                    change_pct = ((price - prev_price) / prev_price * 100) if prev_price else 0

                    PRICE_CACHE[symbol] = {
                        "price": price,
                        "change": change_pct,
                        "high": float(latest.get("high", price)),
                        "low": float(latest.get("low", price)),
                        "volume": float(latest.get("volume", 0))
                    }
            except Exception:
                pass

        LAST_PRICE_UPDATE = datetime.now()
    except Exception as e:
        # Fallback: try to get from database
        try:
            from libs.db.session import get_session
            from libs.db.models import Signal

            with get_session() as session:
                for symbol in symbols:
                    signal = session.query(Signal).filter(
                        Signal.symbol == symbol
                    ).order_by(Signal.timestamp.desc()).first()

                    if signal and symbol not in PRICE_CACHE:
                        PRICE_CACHE[symbol] = {
                            "price": signal.entry_price or 0,
                            "change": 0,
                            "high": signal.entry_price or 0,
                            "low": signal.entry_price or 0,
                            "volume": 0
                        }
        except:
            pass

    return PRICE_CACHE


def get_prices_panel():
    """Get live prices panel."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Price", justify="right", width=12)
    table.add_column("24h %", justify="right", width=8)
    table.add_column("Volume", justify="right", width=10)

    prices = get_live_prices()

    for symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
                   "ADA-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "DOT-USD"]:
        data = prices.get(symbol, {})
        price = data.get("price", 0)
        change = data.get("change", 0)
        volume = data.get("volume", 0)

        # Format price
        if price >= 1000:
            price_str = f"${price:,.0f}"
        elif price >= 1:
            price_str = f"${price:.2f}"
        else:
            price_str = f"${price:.4f}"

        # Format change
        change_color = "green" if change >= 0 else "red"
        change_str = f"[{change_color}]{change:+.2f}%[/]"

        # Format volume
        if volume >= 1_000_000:
            vol_str = f"{volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            vol_str = f"{volume/1_000:.1f}K"
        else:
            vol_str = f"{volume:.0f}"

        short_symbol = symbol.replace("-USD", "")
        table.add_row(short_symbol, price_str, change_str, vol_str)

    return Panel(table, title="LIVE PRICES", box=box.ROUNDED)


def get_engine_rankings():
    """Get engine rankings table."""
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Rank", style="cyan", justify="center", width=6)
    table.add_column("Engine", width=18)
    table.add_column("Specialty", style="yellow", width=12)
    table.add_column("WR", justify="right", width=6)
    table.add_column("P&L", justify="right", width=10)
    table.add_column("Trades", justify="right", width=6)

    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()

        for i, (name, stats) in enumerate(rankings):
            rank = i + 1
            if rank == 1:
                rank_str = "[green]👑 #1[/]"
            elif rank == 4:
                rank_str = "[red]💀 #4[/]"
            else:
                rank_str = f"#{rank}"

            engine_name, color = ENGINE_NAMES.get(name, (f"Engine {name}", "white"))
            engine_str = f"[{color}]{name}: {engine_name}[/]"

            wr = f"{stats.win_rate * 100:.0f}%"
            pnl = stats.total_pnl_usd
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_str = f"[{pnl_style}]${pnl:+.2f}[/]"

            table.add_row(
                rank_str,
                engine_str,
                ENGINE_SPECIALTIES.get(name, "?"),
                wr,
                pnl_str,
                str(stats.total_trades)
            )
    except Exception as e:
        table.add_row("?", f"Error: {str(e)[:20]}", "-", "-", "-", "-")

    return Panel(table, title="ENGINE RANKINGS", box=box.ROUNDED)


def get_communication_log():
    """Get AI agent communication log."""
    lines = []

    # Try to get real communication data
    try:
        from libs.hydra.cycles.knowledge_transfer import KnowledgeTransfer
        kt = KnowledgeTransfer()

        # Get recent sessions from file if available
        sessions_file = Path.home() / "crpbot" / "data" / "hydra" / "teaching_sessions.jsonl"
        if not sessions_file.exists():
            sessions_file = Path("/root/crpbot/data/hydra/teaching_sessions.jsonl")

        if sessions_file.exists():
            with open(sessions_file) as f:
                all_lines = f.readlines()
                for line in all_lines[-5:]:  # Last 5 sessions
                    try:
                        session = json.loads(line)
                        teacher = session.get("teacher_engine", "?")
                        learners = session.get("learners", [])
                        t_name, t_color = ENGINE_NAMES.get(teacher, ("?", "white"))

                        for learner in learners:
                            l_name, l_color = ENGINE_NAMES.get(learner, ("?", "white"))
                            log_communication(
                                f"Engine {teacher}",
                                f"Engine {learner}",
                                "TEACH",
                                f"Knowledge transfer session"
                            )
                    except:
                        pass
    except:
        pass

    # Try to get tournament results
    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()

        # Log tournament activity
        rankings = manager.calculate_rankings()
        if rankings:
            winner = rankings[0][0]
            w_name, w_color = ENGINE_NAMES.get(winner, ("?", "white"))
            log_communication(
                "Mother AI",
                "All Engines",
                "TOURNAMENT",
                f"Current leader: Engine {winner} ({w_name})"
            )
    except:
        pass

    # Generate some simulated activity if log is empty
    if len(COMM_LOG) < 3:
        activities = [
            ("Mother AI", "All Engines", "COORDINATE", "Initiating trading cycle"),
            ("Engine A", "Mother AI", "SIGNAL", "Analyzing liquidation data"),
            ("Engine B", "Mother AI", "SIGNAL", "Monitoring funding rates"),
            ("Engine C", "Mother AI", "SIGNAL", "Scanning orderbook depth"),
            ("Engine D", "Mother AI", "SIGNAL", "Regime detection active"),
            ("Mother AI", "Guardian", "CHECK", "Requesting risk clearance"),
            ("Guardian", "Mother AI", "APPROVE", "Trading approved"),
        ]
        for sender, receiver, msg_type, content in activities[:5]:
            log_communication(sender, receiver, msg_type, content)

    # Format log entries
    for entry in list(COMM_LOG)[-12:]:  # Last 12 messages
        time_str = entry["time"].strftime("%H:%M:%S")
        sender = entry["sender"]
        receiver = entry["receiver"]
        msg_type = entry["type"]
        content = entry["content"]

        # Color code by type
        type_colors = {
            "TEACH": "magenta",
            "SIGNAL": "cyan",
            "TOURNAMENT": "yellow",
            "COORDINATE": "green",
            "CHECK": "blue",
            "APPROVE": "green",
            "REJECT": "red",
            "LEARN": "magenta",
        }
        color = type_colors.get(msg_type, "white")

        lines.append(
            f"[dim]{time_str}[/] [{color}]{msg_type:10}[/] "
            f"{sender} → {receiver}"
        )
        lines.append(f"    [dim]{content}[/]")

    return Panel(
        "\n".join(lines) if lines else "[dim]No communications yet[/]",
        title="AI AGENT COMMUNICATIONS",
        box=box.ROUNDED
    )


def get_engine_status_panel():
    """Get detailed engine status panel."""
    lines = []

    engines = [
        ("A", "DeepSeek", "cyan", "Liquidation Cascade Hunter"),
        ("B", "Claude", "magenta", "Funding Rate Expert"),
        ("C", "Grok", "yellow", "Orderbook Analyst"),
        ("D", "Gemini", "green", "Regime Detector"),
    ]

    for eid, name, color, role in engines:
        # Try to get engine's current state
        status = "🟢 Active"
        try:
            # Check if engine is working
            from libs.hydra.engine_portfolio import get_tournament_manager
            manager = get_tournament_manager()
            rankings = manager.calculate_rankings()

            # Find this engine's rank
            for i, (eng_id, stats) in enumerate(rankings):
                if eng_id == eid:
                    rank = i + 1
                    if rank == 1:
                        status = "👑 Leading"
                    elif rank == 4:
                        status = "💀 Last"
                    else:
                        status = f"#{rank} Active"
                    break
        except:
            status = "🔄 Loading"

        lines.append(f"[{color}]Engine {eid}: {name}[/]")
        lines.append(f"  Role: {role}")
        lines.append(f"  Status: {status}")
        lines.append("")

    return Panel("\n".join(lines), title="ENGINE STATUS", box=box.ROUNDED)


def get_safety_status():
    """Get safety status panel."""
    lines = []

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
            lines.append(f"[yellow]Circuit: {losses} losses[/yellow]")
        else:
            lines.append(f"  Losses: {losses}/3")

        dd = status.get("current_drawdown_percent", 0)
        dd_color = "red" if dd > 5 else "yellow" if dd > 3 else "white"
        lines.append(f"[{dd_color}]  Drawdown: {dd:.1f}%[/]")

    except Exception as e:
        lines.append(f"Guardian: Loading...")

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
        lines.append("[dim]Mother AI: Loading[/dim]")

    return Panel("\n".join(lines), title="SAFETY", box=box.ROUNDED)


def get_header():
    """Get dashboard header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uptime = datetime.now() - START_TIME
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    return Panel(
        f"[bold cyan]HYDRA 4.0 MULTI-AGENT TRADING SYSTEM[/bold cyan]\n"
        f"{now}  |  Uptime: {uptime_str}  |  "
        f"[cyan]A[/]:DeepSeek [magenta]B[/]:Claude [yellow]C[/]:Grok [green]D[/]:Gemini",
        box=box.DOUBLE,
        style="cyan"
    )


def make_layout():
    """Create the dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="top", size=15),
        Layout(name="middle", ratio=1),
        Layout(name="bottom", size=15),
        Layout(name="footer", size=3)
    )

    # Top row: Prices + Rankings
    layout["top"].split_row(
        Layout(name="prices", ratio=1),
        Layout(name="rankings", ratio=2)
    )

    # Middle row: Communications + Engine Status
    layout["middle"].split_row(
        Layout(name="communications", ratio=2),
        Layout(name="engine_status", ratio=1)
    )

    # Bottom row: Safety + Validation
    layout["bottom"].split_row(
        Layout(name="safety"),
        Layout(name="trends"),
        Layout(name="engine_d")
    )

    return layout


def get_trends_panel():
    """Get engine improvement trends."""
    lines = []

    try:
        from libs.hydra.improvement_tracker import get_improvement_tracker
        tracker = get_improvement_tracker()
        trends = tracker.get_all_trends()

        for engine, trend in trends.items():
            status = trend.get("trend", "UNKNOWN")
            name, color = ENGINE_NAMES.get(engine, ("?", "white"))

            if status == "IMPROVING":
                icon = "[green]↑[/]"
            elif status == "DECLINING":
                icon = "[red]↓[/]"
            elif status == "STAGNANT":
                icon = "[yellow]→[/]"
            else:
                icon = "[dim]?[/]"

            wr_trend = trend.get("win_rate_trend", 0) * 100
            lines.append(f"[{color}]{engine}:{name}[/] {icon} {wr_trend:+.1f}%")
    except:
        lines.append("[dim]Loading trends...[/dim]")

    return Panel("\n".join(lines) if lines else "Loading...", title="TRENDS", box=box.ROUNDED)


def get_engine_d_panel():
    """Get Engine D special rules status."""
    lines = []

    try:
        from libs.hydra.engine_d_rules import get_engine_d_controller
        controller = get_engine_d_controller()
        state = controller.state

        if state.can_activate:
            lines.append("[green]✓ Ready[/green]")
        else:
            lines.append(f"[yellow]Cooldown: {state.days_until_available}d[/yellow]")

        exp = state.expectancy
        exp_color = "green" if exp >= 0 else "red"
        lines.append(f"[{exp_color}]Expect: {exp:+.1f}%[/]")
        lines.append(f"Acts: {state.total_activations}")
    except:
        lines.append("[dim]Loading...[/dim]")

    return Panel("\n".join(lines), title="ENGINE D", box=box.ROUNDED)


def update_dashboard(layout):
    """Update all dashboard components."""
    layout["header"].update(get_header())
    layout["prices"].update(get_prices_panel())
    layout["rankings"].update(get_engine_rankings())
    layout["communications"].update(get_communication_log())
    layout["engine_status"].update(get_engine_status_panel())
    layout["safety"].update(get_safety_status())
    layout["trends"].update(get_trends_panel())
    layout["engine_d"].update(get_engine_d_panel())
    layout["footer"].update(
        Panel(
            "[dim]Ctrl+C to exit | Auto-refresh: 3s | "
            "Engines: [cyan]A[/]=DeepSeek [magenta]B[/]=Claude [yellow]C[/]=Grok [green]D[/]=Gemini[/dim]",
            box=box.SIMPLE
        )
    )
    return layout


def main():
    """Run the live dashboard."""
    console.clear()

    layout = make_layout()

    try:
        with Live(layout, refresh_per_second=0.3, screen=True) as live:
            while True:
                update_dashboard(layout)
                time.sleep(3)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


if __name__ == "__main__":
    main()
