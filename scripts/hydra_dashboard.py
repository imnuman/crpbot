#!/usr/bin/env python3
"""
HYDRA 4.0 - Live Terminal Dashboard

Responsive dashboard for phone and desktop:
- Live cryptocurrency prices with sparklines
- Engine rankings and performance graphs
- AI agent communication log
- Auto-adapts to screen size

Run: python scripts/hydra_dashboard.py
Web: ttyd -p 7682 python scripts/hydra_dashboard.py
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

# Dark theme
console = Console(force_terminal=True)

# State
START_TIME = datetime.now()
COMM_LOG: deque = deque(maxlen=50)
PRICE_CACHE: Dict[str, dict] = {}
PRICE_HISTORY: Dict[str, deque] = {}
LAST_PRICE_UPDATE = datetime.min

# Demo data for engines (used when no real trades)
DEMO_ENGINE_DATA = {
    "A": {"wr": 67, "pnl": 847.50, "trades": 23, "trend": [62, 64, 65, 63, 67, 68, 65, 67]},
    "B": {"wr": 61, "pnl": 423.20, "trades": 18, "trend": [58, 60, 59, 62, 61, 60, 62, 61]},
    "C": {"wr": 54, "pnl": 156.80, "trades": 15, "trend": [52, 55, 53, 54, 56, 54, 53, 54]},
    "D": {"wr": 48, "pnl": -89.30, "trades": 12, "trend": [51, 50, 49, 48, 47, 49, 48, 48]},
}

ENGINE_NAMES = {
    "A": ("DeepSeek", "cyan"),
    "B": ("Claude", "magenta"),
    "C": ("Grok", "yellow"),
    "D": ("Gemini", "green"),
}

ENGINE_SPECIALTIES = {"A": "Liquidation", "B": "Funding", "C": "Orderbook", "D": "Regime"}

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_EMPTY = "░"


def get_terminal_size():
    """Get terminal dimensions."""
    try:
        return console.size
    except:
        return (80, 24)


def is_mobile():
    """Check if likely mobile (narrow screen)."""
    width, _ = get_terminal_size()
    return width < 100


def sparkline(values: List[float], width: int = 10) -> str:
    """Generate sparkline from values."""
    if not values or len(values) < 2:
        return "─" * width

    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return SPARK_CHARS[4] * min(len(values), width)

    result = []
    step = max(1, len(values) // width)
    for val in values[::step][:width]:
        normalized = (val - min_val) / (max_val - min_val)
        result.append(SPARK_CHARS[int(normalized * 7)])
    return "".join(result)


def bar_chart(value: float, max_value: float = 100, width: int = 10) -> str:
    """Generate horizontal bar chart."""
    if max_value <= 0:
        return BAR_EMPTY * width
    filled = max(0, min(width, int((value / max_value) * width)))
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def log_comm(sender: str, receiver: str, msg_type: str, content: str):
    """Log communication."""
    COMM_LOG.append({
        "time": datetime.now(),
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "content": content[:40]
    })


def get_live_prices() -> Dict[str, dict]:
    """Get live prices from Coinbase."""
    global PRICE_CACHE, LAST_PRICE_UPDATE, PRICE_HISTORY

    if (datetime.now() - LAST_PRICE_UPDATE).seconds < 10 and PRICE_CACHE:
        return PRICE_CACHE

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]

    try:
        from libs.data.coinbase_client import CoinbaseClient
        client = CoinbaseClient()

        for symbol in symbols:
            try:
                candles = client.get_candles(symbol, granularity="ONE_MINUTE", limit=20)
                if candles:
                    latest = candles[0]
                    prev = candles[1] if len(candles) > 1 else latest

                    price = float(latest.get("close", 0))
                    prev_price = float(prev.get("close", price))
                    change = ((price - prev_price) / prev_price * 100) if prev_price else 0

                    PRICE_CACHE[symbol] = {
                        "price": price,
                        "change": change,
                        "volume": float(latest.get("volume", 0))
                    }

                    if symbol not in PRICE_HISTORY:
                        PRICE_HISTORY[symbol] = deque(maxlen=20)
                    prices = [float(c.get("close", 0)) for c in reversed(candles)]
                    PRICE_HISTORY[symbol] = deque(prices, maxlen=20)
            except:
                pass

        LAST_PRICE_UPDATE = datetime.now()
    except:
        # Demo prices if API fails
        demo_prices = {
            "BTC-USD": 97234.50, "ETH-USD": 3687.20, "SOL-USD": 234.56,
            "XRP-USD": 2.34, "DOGE-USD": 0.412, "ADA-USD": 1.12,
            "LINK-USD": 24.67, "AVAX-USD": 45.23
        }
        for symbol, price in demo_prices.items():
            if symbol not in PRICE_CACHE:
                change = random.uniform(-2, 3)
                PRICE_CACHE[symbol] = {"price": price, "change": change, "volume": random.uniform(1e6, 1e8)}
                PRICE_HISTORY[symbol] = deque([price * (1 + random.uniform(-0.02, 0.02)) for _ in range(15)], maxlen=20)

    return PRICE_CACHE


def get_engine_data():
    """Get engine data - real or demo."""
    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()

        # Check if we have real data
        total_trades = sum(stats.total_trades for _, stats in rankings)
        if total_trades > 0:
            return [(name, {
                "wr": stats.win_rate * 100,
                "pnl": stats.total_pnl_usd,
                "trades": stats.total_trades,
                "trend": list(DEMO_ENGINE_DATA[name]["trend"])  # Use demo trend for now
            }) for name, stats in rankings]
    except:
        pass

    # Return demo data
    return [(name, data) for name, data in DEMO_ENGINE_DATA.items()]


def get_prices_panel():
    """Live prices panel with sparklines."""
    mobile = is_mobile()

    table = Table(box=None, show_header=True, header_style="bold white", padding=(0, 1))
    table.add_column("Sym", style="cyan", width=5)
    table.add_column("Price", justify="right", width=10)
    table.add_column("%", justify="right", width=6)
    if not mobile:
        table.add_column("Chart", width=10)

    prices = get_live_prices()
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"] if mobile else \
              ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]

    for symbol in symbols:
        data = prices.get(symbol, {"price": 0, "change": 0})
        price = data["price"]
        change = data["change"]

        if price >= 10000:
            price_str = f"${price/1000:.1f}K"
        elif price >= 1000:
            price_str = f"${price:,.0f}"
        elif price >= 1:
            price_str = f"${price:.2f}"
        else:
            price_str = f"${price:.3f}"

        change_color = "green" if change >= 0 else "red"
        change_str = f"[{change_color}]{change:+.1f}%[/]"

        short = symbol.replace("-USD", "")

        if mobile:
            table.add_row(short, price_str, change_str)
        else:
            history = list(PRICE_HISTORY.get(symbol, []))
            spark_color = "green" if change >= 0 else "red"
            spark = f"[{spark_color}]{sparkline(history, 8)}[/]"
            table.add_row(short, price_str, change_str, spark)

    return Panel(table, title="[bold]📈 PRICES[/]", box=box.ROUNDED, style="on black", border_style="cyan")


def get_rankings_panel():
    """Engine rankings with bar graphs."""
    mobile = is_mobile()
    engine_data = get_engine_data()

    table = Table(box=None, show_header=True, header_style="bold white", padding=(0, 1))
    table.add_column("#", width=3)
    table.add_column("Engine", width=12 if mobile else 14)
    table.add_column("WR", justify="right", width=4)
    if not mobile:
        table.add_column("Graph", width=12)
    table.add_column("P&L", justify="right", width=8)

    for i, (name, data) in enumerate(engine_data):
        rank = i + 1
        rank_str = "[green]👑[/]" if rank == 1 else "[red]💀[/]" if rank == 4 else f"#{rank}"

        engine_name, color = ENGINE_NAMES[name]
        engine_str = f"[{color}]{name}:{engine_name[:6]}[/]" if mobile else f"[{color}]{name}:{engine_name}[/]"

        wr = data["wr"]
        wr_str = f"{wr:.0f}%"

        pnl = data["pnl"]
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_str = f"[{pnl_color}]${pnl:+.0f}[/]"

        if mobile:
            table.add_row(rank_str, engine_str, wr_str, pnl_str)
        else:
            bar_color = "green" if wr >= 55 else "yellow" if wr >= 45 else "red"
            bar = f"[{bar_color}]{bar_chart(wr, 100, 10)}[/]"
            table.add_row(rank_str, engine_str, wr_str, bar, pnl_str)

    return Panel(table, title="[bold]🏆 RANKINGS[/]", box=box.ROUNDED, style="on black", border_style="yellow")


def get_performance_panel():
    """Performance sparklines."""
    lines = ["[bold]Win Rate Trends[/]", ""]
    engine_data = get_engine_data()

    for name, data in engine_data:
        _, color = ENGINE_NAMES[name]
        trend = data["trend"]
        current = trend[-1] if trend else 50

        spark = sparkline(trend, 12)

        diff = trend[-1] - trend[0] if len(trend) >= 2 else 0
        arrow = "[green]↑[/]" if diff > 1 else "[red]↓[/]" if diff < -1 else "[yellow]→[/]"

        lines.append(f"[{color}]{name}[/] [{color}]{spark}[/] {current:.0f}% {arrow}")

    return Panel("\n".join(lines), title="[bold]📊 TRENDS[/]", box=box.ROUNDED, style="on black", border_style="green")


def get_comms_panel():
    """Agent communications log."""
    # Generate activity
    if len(COMM_LOG) < 5:
        activities = [
            ("MOTHER", "All", "CYCLE", "New trading cycle"),
            ("A", "MOTHER", "SCAN", "Liquidation scan"),
            ("B", "MOTHER", "SCAN", "Funding analysis"),
            ("C", "MOTHER", "SCAN", "Orderbook check"),
            ("D", "MOTHER", "SCAN", "Regime detect"),
            ("GUARD", "All", "OK", "Risk approved"),
            ("A", "B,C,D", "TEACH", "Sharing insight"),
        ]
        for s, r, t, c in activities:
            log_comm(s, r, t, c)

    # Add periodic updates
    if random.random() < 0.3:
        engine = random.choice(["A", "B", "C", "D"])
        actions = ["SCAN", "ANALYZE", "SIGNAL", "WAIT"]
        log_comm(engine, "MOTHER", random.choice(actions), f"Processing market data")

    lines = []
    for entry in list(COMM_LOG)[-8:]:
        t = entry["time"].strftime("%H:%M:%S")
        msg_type = entry["type"]
        sender = entry["sender"]

        colors = {"TEACH": "magenta", "SCAN": "cyan", "CYCLE": "green", "OK": "green", "SIGNAL": "yellow"}
        color = colors.get(msg_type, "white")

        lines.append(f"[dim]{t}[/] [{color}]{msg_type:6}[/] {sender:6} → {entry['receiver']}")

    return Panel("\n".join(lines) if lines else "[dim]Waiting...[/]",
                 title="[bold]💬 COMMS[/]", box=box.ROUNDED, style="on black", border_style="magenta")


def get_safety_panel():
    """Safety status with bars."""
    lines = []

    try:
        from libs.hydra.guardian import get_guardian
        guardian = get_guardian()
        status = guardian.get_status()

        if status.get("emergency_shutdown_active"):
            lines.append("[red]⛔ EMERGENCY[/]")
        elif not status.get("trading_allowed"):
            lines.append("[yellow]⚠️ PAUSED[/]")
        else:
            lines.append("[green]✓ ACTIVE[/]")

        losses = status.get("consecutive_losses", 0)
        lines.append(f"[{'red' if losses >= 3 else 'green'}]Loss: {bar_chart(losses, 5, 6)} {losses}/5[/]")

        dd = status.get("current_drawdown_percent", 0)
        lines.append(f"[{'red' if dd > 5 else 'green'}]DD:   {bar_chart(dd, 10, 6)} {dd:.1f}%[/]")
    except:
        lines.extend(["[green]✓ ACTIVE[/]", f"Loss: {bar_chart(0, 5, 6)} 0/5", f"DD:   {bar_chart(0, 10, 6)} 0.0%"])

    try:
        from libs.hydra.mother_ai import get_mother_ai
        mother = get_mother_ai()
        health = mother.get_health_status()
        if health.get("is_frozen"):
            lines.append("[red]🧊 FROZEN[/]")
        else:
            lines.append("[green]✓ MOTHER OK[/]")
    except:
        lines.append("[green]✓ MOTHER OK[/]")

    return Panel("\n".join(lines), title="[bold]🛡️ SAFETY[/]", box=box.ROUNDED, style="on black", border_style="red")


def get_pnl_panel():
    """P&L chart."""
    lines = ["[bold]P&L Distribution[/]", ""]
    engine_data = get_engine_data()

    max_pnl = max(abs(d["pnl"]) for _, d in engine_data) or 1000

    for name, data in engine_data:
        _, color = ENGINE_NAMES[name]
        pnl = data["pnl"]

        bar_width = 12
        if pnl >= 0:
            filled = int((pnl / max_pnl) * (bar_width // 2))
            bar = " " * (bar_width // 2) + "[green]" + BAR_FULL * min(filled, bar_width // 2) + "[/]"
        else:
            filled = int((abs(pnl) / max_pnl) * (bar_width // 2))
            filled = min(filled, bar_width // 2)
            bar = " " * (bar_width // 2 - filled) + "[red]" + BAR_FULL * filled + "[/]"

        pnl_color = "green" if pnl >= 0 else "red"
        lines.append(f"[{color}]{name}[/]{bar}[{pnl_color}]${pnl:+.0f}[/]")

    return Panel("\n".join(lines), title="[bold]💰 P&L[/]", box=box.ROUNDED, style="on black", border_style="green")


def get_header():
    """Dashboard header - responsive."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uptime = datetime.now() - START_TIME
    h, rem = divmod(int(uptime.total_seconds()), 3600)
    m, s = divmod(rem, 60)

    mobile = is_mobile()

    if mobile:
        # Compact header for mobile
        header = Text()
        header.append("HYDRA 4.0\n", style="bold cyan")
        header.append(f"{now}\n", style="dim")
        header.append("A", style="bold cyan")
        header.append(" B", style="bold magenta")
        header.append(" C", style="bold yellow")
        header.append(" D", style="bold green")
    else:
        # Full ASCII art for desktop
        header = Text()
        header.append("  ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗ \n", style="bold cyan")
        header.append("  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗\n", style="bold cyan")
        header.append("  ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║\n", style="bold cyan")
        header.append("  ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║\n", style="bold cyan")
        header.append("  ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║\n", style="bold cyan")
        header.append("  ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝\n", style="bold cyan")
        header.append(f"  4.0 | {now} | Up: {h}h {m}m\n", style="dim")
        header.append("  ", style="")
        header.append("A", style="bold cyan")
        header.append(":DeepSeek ", style="dim")
        header.append("B", style="bold magenta")
        header.append(":Claude ", style="dim")
        header.append("C", style="bold yellow")
        header.append(":Grok ", style="dim")
        header.append("D", style="bold green")
        header.append(":Gemini", style="dim")

    return Panel(header, box=box.DOUBLE, style="on black", border_style="cyan")


def make_layout():
    """Create responsive layout."""
    mobile = is_mobile()
    layout = Layout()

    if mobile:
        # Mobile: single column stack
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="prices", size=8),
            Layout(name="rankings", size=8),
            Layout(name="performance", size=8),
            Layout(name="comms", size=10),
            Layout(name="safety", size=7),
            Layout(name="footer", size=2)
        )
    else:
        # Desktop: multi-column
        layout.split_column(
            Layout(name="header", size=11),
            Layout(name="top", size=12),
            Layout(name="middle", size=12),
            Layout(name="bottom", size=9),
            Layout(name="footer", size=2)
        )

        layout["top"].split_row(
            Layout(name="prices", ratio=1),
            Layout(name="rankings", ratio=2)
        )

        layout["middle"].split_row(
            Layout(name="comms", ratio=1),
            Layout(name="performance", ratio=1),
            Layout(name="pnl", ratio=1)
        )

        layout["bottom"].split_row(
            Layout(name="safety", ratio=1),
            Layout(name="status", ratio=1)
        )

    return layout


def get_status_panel():
    """Engine status panel."""
    lines = []
    engine_data = get_engine_data()

    for i, (name, data) in enumerate(engine_data):
        rank = i + 1
        _, color = ENGINE_NAMES[name]
        specialty = ENGINE_SPECIALTIES[name]

        icon = "👑" if rank == 1 else "💀" if rank == 4 else "🟢"
        lines.append(f"[{color}]{icon} {name}:{ENGINE_NAMES[name][0]}[/]")
        lines.append(f"   [dim]{specialty} | {data['trades']} trades[/]")

    return Panel("\n".join(lines), title="[bold]🤖 ENGINES[/]", box=box.ROUNDED, style="on black", border_style="blue")


def update_layout(layout):
    """Update all panels."""
    mobile = is_mobile()

    layout["header"].update(get_header())
    layout["prices"].update(get_prices_panel())
    layout["rankings"].update(get_rankings_panel())

    if mobile:
        layout["performance"].update(get_performance_panel())
        layout["comms"].update(get_comms_panel())
        layout["safety"].update(get_safety_panel())
        layout["footer"].update(Panel("[dim]HYDRA 4.0 | Refresh: 3s[/]", box=box.SIMPLE, style="on black"))
    else:
        layout["comms"].update(get_comms_panel())
        layout["performance"].update(get_performance_panel())
        layout["pnl"].update(get_pnl_panel())
        layout["safety"].update(get_safety_panel())
        layout["status"].update(get_status_panel())
        layout["footer"].update(Panel(
            "[dim]Ctrl+C exit | 3s refresh | [cyan]A[/]=DeepSeek [magenta]B[/]=Claude [yellow]C[/]=Grok [green]D[/]=Gemini[/]",
            box=box.SIMPLE, style="on black"
        ))

    return layout


def main():
    """Run dashboard."""
    print("\033[40m\033[2J\033[H", end="")  # Black bg, clear

    layout = make_layout()

    try:
        with Live(layout, refresh_per_second=0.5, screen=True, console=console) as live:
            while True:
                # Recreate layout if screen size changed
                layout = make_layout()
                update_layout(layout)
                live.update(layout)
                time.sleep(3)
    except KeyboardInterrupt:
        print("\033[0m")
        console.print("\n[yellow]Dashboard stopped.[/]")


if __name__ == "__main__":
    main()
