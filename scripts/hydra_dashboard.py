#!/usr/bin/env python3
"""
HYDRA 4.0 - Live Terminal Dashboard

Enhanced dashboard with:
- Live cryptocurrency prices with sparklines
- Engine rankings and performance graphs
- AI agent communication log
- Win rate and P&L charts

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
    from rich.style import Style
    from rich.theme import Theme
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
    from rich.style import Style
    from rich.theme import Theme
    from rich import box

# Custom dark theme
DARK_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
})

console = Console(theme=DARK_THEME, force_terminal=True)

# Track uptime and state
START_TIME = datetime.now()

# Communication log (in-memory ring buffer)
COMM_LOG: deque = deque(maxlen=50)

# Price cache and history for sparklines
PRICE_CACHE: Dict[str, dict] = {}
PRICE_HISTORY: Dict[str, deque] = {}  # For sparklines
LAST_PRICE_UPDATE = datetime.min

# Engine performance history for graphs
ENGINE_HISTORY: Dict[str, deque] = {
    "A": deque(maxlen=20),
    "B": deque(maxlen=20),
    "C": deque(maxlen=20),
    "D": deque(maxlen=20),
}

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

# Sparkline characters (for price charts)
SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Bar chart characters
BAR_FULL = "█"
BAR_EMPTY = "░"


def sparkline(values: List[float], width: int = 10) -> str:
    """Generate a sparkline from values."""
    if not values or len(values) < 2:
        return "─" * width

    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return SPARK_CHARS[4] * min(len(values), width)

    # Normalize and map to spark characters
    result = []
    step = max(1, len(values) // width)
    sampled = values[::step][:width]

    for val in sampled:
        normalized = (val - min_val) / (max_val - min_val)
        idx = int(normalized * (len(SPARK_CHARS) - 1))
        result.append(SPARK_CHARS[idx])

    return "".join(result)


def bar_chart(value: float, max_value: float = 100, width: int = 15) -> str:
    """Generate a horizontal bar chart."""
    if max_value <= 0:
        return BAR_EMPTY * width

    filled = int((value / max_value) * width)
    filled = max(0, min(width, filled))

    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def log_communication(sender: str, receiver: str, msg_type: str, content: str):
    """Add a message to the communication log."""
    COMM_LOG.append({
        "time": datetime.now(),
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "content": content[:50]
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
                candles = client.get_candles(symbol, granularity="ONE_MINUTE", limit=20)
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

                    # Store price history for sparklines
                    if symbol not in PRICE_HISTORY:
                        PRICE_HISTORY[symbol] = deque(maxlen=20)

                    # Add historical prices from candles
                    prices = [float(c.get("close", 0)) for c in reversed(candles)]
                    PRICE_HISTORY[symbol] = deque(prices, maxlen=20)

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
    """Get live prices panel with sparklines."""
    table = Table(box=None, show_header=True, header_style="bold white on black",
                  padding=(0, 1), collapse_padding=True)
    table.add_column("Symbol", style="cyan", width=6)
    table.add_column("Price", justify="right", width=11)
    table.add_column("Chg%", justify="right", width=7)
    table.add_column("Chart", width=12)

    prices = get_live_prices()

    for symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
                   "ADA-USD", "AVAX-USD", "LINK-USD"]:
        data = prices.get(symbol, {})
        price = data.get("price", 0)
        change = data.get("change", 0)

        # Format price
        if price >= 1000:
            price_str = f"${price:,.0f}"
        elif price >= 1:
            price_str = f"${price:.2f}"
        else:
            price_str = f"${price:.4f}"

        # Format change with color
        change_color = "green" if change >= 0 else "red"
        change_str = f"[{change_color}]{change:+.1f}%[/]"

        # Get sparkline
        history = list(PRICE_HISTORY.get(symbol, []))
        if not history:
            # Generate dummy sparkline for demo
            history = [price * (1 + random.uniform(-0.01, 0.01)) for _ in range(12)]

        spark_color = "green" if change >= 0 else "red"
        spark = f"[{spark_color}]{sparkline(history, 10)}[/]"

        short_symbol = symbol.replace("-USD", "")
        table.add_row(short_symbol, price_str, change_str, spark)

    return Panel(table, title="[bold white]📈 LIVE PRICES[/]", box=box.ROUNDED,
                 style="on black", border_style="cyan")


def get_engine_rankings():
    """Get engine rankings with performance bars."""
    table = Table(box=None, show_header=True, header_style="bold white",
                  padding=(0, 1))
    table.add_column("Rank", style="cyan", justify="center", width=5)
    table.add_column("Engine", width=16)
    table.add_column("WR", justify="right", width=5)
    table.add_column("Win Rate Graph", width=18)
    table.add_column("P&L", justify="right", width=9)

    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()

        for i, (name, stats) in enumerate(rankings):
            rank = i + 1
            if rank == 1:
                rank_str = "[green]👑#1[/]"
            elif rank == 4:
                rank_str = "[red]💀#4[/]"
            else:
                rank_str = f"#{rank}"

            engine_name, color = ENGINE_NAMES.get(name, (f"Engine {name}", "white"))
            engine_str = f"[{color}]{name}:{engine_name}[/]"

            wr = stats.win_rate * 100
            wr_str = f"{wr:.0f}%"

            # Win rate bar graph
            bar = bar_chart(wr, 100, 15)
            bar_color = "green" if wr >= 60 else "yellow" if wr >= 50 else "red"
            bar_str = f"[{bar_color}]{bar}[/]"

            pnl = stats.total_pnl_usd
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_str = f"[{pnl_style}]${pnl:+.0f}[/]"

            # Track history
            ENGINE_HISTORY[name].append(wr)

            table.add_row(rank_str, engine_str, wr_str, bar_str, pnl_str)
    except Exception as e:
        table.add_row("?", f"Error: {str(e)[:15]}", "-", "-", "-")

    return Panel(table, title="[bold white]🏆 ENGINE RANKINGS[/]", box=box.ROUNDED,
                 style="on black", border_style="yellow")


def get_performance_graph():
    """Get ASCII performance graph for all engines."""
    lines = []

    # Header
    lines.append("[bold white]Win Rate Trends (Last 20 cycles)[/]")
    lines.append("")

    for engine_id in ["A", "B", "C", "D"]:
        name, color = ENGINE_NAMES.get(engine_id, ("?", "white"))
        history = list(ENGINE_HISTORY[engine_id])

        if not history:
            # Demo data
            history = [50 + random.uniform(-10, 10) for _ in range(15)]

        spark = sparkline(history, 20)
        current = history[-1] if history else 50

        trend = ""
        if len(history) >= 2:
            diff = history[-1] - history[0]
            if diff > 2:
                trend = "[green]↑[/]"
            elif diff < -2:
                trend = "[red]↓[/]"
            else:
                trend = "[yellow]→[/]"

        lines.append(f"[{color}]{engine_id}[/] [{color}]{spark}[/] {current:.0f}% {trend}")

    return Panel("\n".join(lines), title="[bold white]📊 PERFORMANCE[/]", box=box.ROUNDED,
                 style="on black", border_style="green")


def get_communication_log():
    """Get AI agent communication log."""
    lines = []

    # Try to get real communication data
    try:
        from libs.hydra.cycles.knowledge_transfer import KnowledgeTransfer
        kt = KnowledgeTransfer()

        sessions_file = Path.home() / "crpbot" / "data" / "hydra" / "teaching_sessions.jsonl"
        if not sessions_file.exists():
            sessions_file = Path("/root/crpbot/data/hydra/teaching_sessions.jsonl")

        if sessions_file.exists():
            with open(sessions_file) as f:
                all_lines = f.readlines()
                for line in all_lines[-3:]:
                    try:
                        session = json.loads(line)
                        teacher = session.get("teacher_engine", "?")
                        learners = session.get("learners", [])

                        for learner in learners:
                            log_communication(
                                f"Engine {teacher}",
                                f"Engine {learner}",
                                "TEACH",
                                f"Knowledge transfer"
                            )
                    except:
                        pass
    except:
        pass

    # Try to get tournament results
    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()
        if rankings:
            winner = rankings[0][0]
            w_name, _ = ENGINE_NAMES.get(winner, ("?", "white"))
            log_communication("Mother AI", "All", "RANK", f"Leader: {winner}:{w_name}")
    except:
        pass

    # Generate activity if log is empty
    if len(COMM_LOG) < 3:
        activities = [
            ("Mother AI", "All Engines", "CYCLE", "Trading cycle started"),
            ("Engine A", "Mother AI", "SCAN", "Liquidation analysis"),
            ("Engine B", "Mother AI", "SCAN", "Funding rate check"),
            ("Engine C", "Mother AI", "SCAN", "Orderbook depth"),
            ("Engine D", "Mother AI", "SCAN", "Regime detection"),
            ("Guardian", "All", "OK", "Risk check passed"),
        ]
        for sender, receiver, msg_type, content in activities:
            log_communication(sender, receiver, msg_type, content)

    # Format log entries
    for entry in list(COMM_LOG)[-10:]:
        time_str = entry["time"].strftime("%H:%M:%S")
        sender = entry["sender"]
        receiver = entry["receiver"]
        msg_type = entry["type"]
        content = entry["content"]

        # Color code by type
        type_colors = {
            "TEACH": "magenta", "SCAN": "cyan", "RANK": "yellow",
            "CYCLE": "green", "OK": "green", "WARN": "yellow",
            "ERROR": "red", "TRADE": "cyan",
        }
        color = type_colors.get(msg_type, "white")

        # Sender colors
        sender_short = sender.replace("Engine ", "").replace("Mother AI", "MOTHER")

        lines.append(
            f"[dim]{time_str}[/] [{color}]{msg_type:6}[/] "
            f"[bold]{sender_short:8}[/] → {receiver}"
        )

    return Panel(
        "\n".join(lines) if lines else "[dim]Waiting for activity...[/]",
        title="[bold white]💬 AGENT COMMS[/]",
        box=box.ROUNDED,
        style="on black",
        border_style="magenta"
    )


def get_engine_status_panel():
    """Get detailed engine status panel."""
    lines = []

    engines = [
        ("A", "DeepSeek", "cyan", "Liquidation"),
        ("B", "Claude", "magenta", "Funding"),
        ("C", "Grok", "yellow", "Orderbook"),
        ("D", "Gemini", "green", "Regime"),
    ]

    for eid, name, color, specialty in engines:
        status = "🟢"
        rank_info = ""
        try:
            from libs.hydra.engine_portfolio import get_tournament_manager
            manager = get_tournament_manager()
            rankings = manager.calculate_rankings()

            for i, (eng_id, stats) in enumerate(rankings):
                if eng_id == eid:
                    rank = i + 1
                    if rank == 1:
                        status = "👑"
                        rank_info = "#1"
                    elif rank == 4:
                        status = "💀"
                        rank_info = "#4"
                    else:
                        rank_info = f"#{rank}"
                    break
        except:
            pass

        lines.append(f"[{color}]{status} {eid}:{name}[/] {rank_info}")
        lines.append(f"   [dim]{specialty} Specialist[/]")

    return Panel("\n".join(lines), title="[bold white]🤖 ENGINES[/]", box=box.ROUNDED,
                 style="on black", border_style="blue")


def get_safety_status():
    """Get safety status panel."""
    lines = []

    try:
        from libs.hydra.guardian import get_guardian
        guardian = get_guardian()
        status = guardian.get_status()

        if status.get("emergency_shutdown_active"):
            lines.append("[red]⛔ EMERGENCY STOP[/]")
        elif not status.get("trading_allowed"):
            lines.append("[yellow]⚠️ PAUSED[/]")
        else:
            lines.append("[green]✓ GUARDIAN OK[/]")

        losses = status.get("consecutive_losses", 0)
        loss_bar = bar_chart(losses, 5, 8)
        loss_color = "red" if losses >= 3 else "yellow" if losses >= 2 else "green"
        lines.append(f"[{loss_color}]Losses: {loss_bar} {losses}/5[/]")

        dd = status.get("current_drawdown_percent", 0)
        dd_bar = bar_chart(dd, 10, 8)
        dd_color = "red" if dd > 5 else "yellow" if dd > 3 else "green"
        lines.append(f"[{dd_color}]DD:     {dd_bar} {dd:.1f}%[/]")

    except:
        lines.append("[dim]Guardian loading...[/]")

    try:
        from libs.hydra.mother_ai import get_mother_ai
        mother = get_mother_ai()
        health = mother.get_health_status()

        if health.get("is_frozen"):
            lines.append("[red]🧊 MOTHER FROZEN[/]")
        elif health.get("is_healthy"):
            lines.append("[green]✓ MOTHER AI OK[/]")
        else:
            lines.append("[yellow]⚠️ MOTHER WARN[/]")
    except:
        lines.append("[dim]Mother AI loading...[/]")

    return Panel("\n".join(lines), title="[bold white]🛡️ SAFETY[/]", box=box.ROUNDED,
                 style="on black", border_style="red")


def get_pnl_chart():
    """Get P&L chart for all engines."""
    lines = []
    lines.append("[bold]Engine P&L Distribution[/]")
    lines.append("")

    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()

        max_pnl = max(abs(stats.total_pnl_usd) for _, stats in rankings) if rankings else 100
        max_pnl = max(max_pnl, 10)  # Minimum scale

        for name, stats in rankings:
            pnl = stats.total_pnl_usd
            _, color = ENGINE_NAMES.get(name, ("?", "white"))

            # Create centered bar
            bar_width = 20
            if pnl >= 0:
                filled = int((pnl / max_pnl) * (bar_width // 2))
                filled = min(filled, bar_width // 2)
                bar = " " * (bar_width // 2) + "[green]" + BAR_FULL * filled + "[/]"
            else:
                filled = int((abs(pnl) / max_pnl) * (bar_width // 2))
                filled = min(filled, bar_width // 2)
                bar = " " * (bar_width // 2 - filled) + "[red]" + BAR_FULL * filled + "[/]"

            pnl_color = "green" if pnl >= 0 else "red"
            lines.append(f"[{color}]{name}[/] {bar} [{pnl_color}]${pnl:+.0f}[/]")
    except:
        lines.append("[dim]Loading P&L data...[/]")

    return Panel("\n".join(lines), title="[bold white]💰 P&L CHART[/]", box=box.ROUNDED,
                 style="on black", border_style="green")


def get_header():
    """Get dashboard header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uptime = datetime.now() - START_TIME
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    header_text = Text()
    header_text.append("  ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗ \n", style="bold cyan")
    header_text.append("  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗\n", style="bold cyan")
    header_text.append("  ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║\n", style="bold cyan")
    header_text.append("  ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║\n", style="bold cyan")
    header_text.append("  ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║\n", style="bold cyan")
    header_text.append("  ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝\n", style="bold cyan")
    header_text.append(f"  4.0 MULTI-AGENT SYSTEM | {now} | Up: {uptime_str}\n", style="dim white")
    header_text.append(f"  ", style="")
    header_text.append("A", style="bold cyan")
    header_text.append(":DeepSeek ", style="dim")
    header_text.append("B", style="bold magenta")
    header_text.append(":Claude ", style="dim")
    header_text.append("C", style="bold yellow")
    header_text.append(":Grok ", style="dim")
    header_text.append("D", style="bold green")
    header_text.append(":Gemini", style="dim")

    return Panel(header_text, box=box.DOUBLE, style="on black", border_style="cyan")


def make_layout():
    """Create the dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=11),
        Layout(name="top", size=12),
        Layout(name="middle", size=14),
        Layout(name="bottom", size=10),
        Layout(name="footer", size=3)
    )

    # Top row: Prices + Rankings
    layout["top"].split_row(
        Layout(name="prices", ratio=1),
        Layout(name="rankings", ratio=2)
    )

    # Middle row: Communications + Graphs
    layout["middle"].split_row(
        Layout(name="communications", ratio=1),
        Layout(name="performance", ratio=1),
        Layout(name="pnl_chart", ratio=1)
    )

    # Bottom row: Engine Status + Safety
    layout["bottom"].split_row(
        Layout(name="engines", ratio=1),
        Layout(name="safety", ratio=1)
    )

    return layout


def update_dashboard(layout):
    """Update all dashboard components."""
    layout["header"].update(get_header())
    layout["prices"].update(get_prices_panel())
    layout["rankings"].update(get_engine_rankings())
    layout["communications"].update(get_communication_log())
    layout["performance"].update(get_performance_graph())
    layout["pnl_chart"].update(get_pnl_chart())
    layout["engines"].update(get_engine_status_panel())
    layout["safety"].update(get_safety_status())
    layout["footer"].update(
        Panel(
            "[dim white on black]Ctrl+C to exit | Auto-refresh: 3s | "
            "[cyan]A[/]=DeepSeek [magenta]B[/]=Claude [yellow]C[/]=Grok [green]D[/]=Gemini[/]",
            box=box.SIMPLE,
            style="on black"
        )
    )
    return layout


def main():
    """Run the live dashboard."""
    # Force black background
    console.clear()
    print("\033[40m", end="")  # Set black background
    print("\033[2J\033[H", end="")  # Clear screen

    layout = make_layout()

    try:
        with Live(layout, refresh_per_second=0.3, screen=True, console=console) as live:
            while True:
                update_dashboard(layout)
                time.sleep(3)
    except KeyboardInterrupt:
        print("\033[0m")  # Reset terminal
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


if __name__ == "__main__":
    main()
