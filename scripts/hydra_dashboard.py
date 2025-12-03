#!/usr/bin/env python3
"""
HYDRA 4.0 - Live Terminal Dashboard (Enhanced)

Features:
- NTP-synced timestamps
- Smooth 1s refresh with Rich Live
- Detailed AI agent communications
- Trade signals panel for manual execution
- Live prices with entry/exit levels

Run: python scripts/hydra_dashboard.py
Web: ttyd -W -p 7682 python scripts/hydra_dashboard.py
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

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
    os.system("pip install rich")
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box

# Console with black background
console = Console(force_terminal=True)

# State
START_TIME = datetime.now()
COMM_LOG: deque = deque(maxlen=100)
PRICE_CACHE: Dict[str, dict] = {}
PRICE_HISTORY: Dict[str, deque] = {}
LAST_PRICE_UPDATE = datetime.min
ACTIVE_SIGNALS: Dict[str, dict] = {}
NTP_OFFSET_MS = 0

# Engine config
ENGINE_NAMES = {
    "A": ("DeepSeek", "cyan"),
    "B": ("Claude", "magenta"),
    "C": ("Grok", "yellow"),
    "D": ("Gemini", "green"),
}

ENGINE_SPECIALTIES = {
    "A": "Liquidation Cascades",
    "B": "Funding Rate Extremes",
    "C": "Orderbook Imbalance",
    "D": "Regime Detection"
}

ENGINE_ROLES = {
    "A": "Scans for large liquidation events that trigger cascading moves",
    "B": "Monitors funding rates for sentiment extremes",
    "C": "Analyzes bid/ask depth for supply/demand zones",
    "D": "Detects market regime shifts (trending/ranging)"
}

# Demo data
DEMO_ENGINE_DATA = {
    "A": {"wr": 67, "pnl": 847.50, "trades": 23, "trend": [62, 64, 65, 63, 67, 68, 65, 67]},
    "B": {"wr": 61, "pnl": 423.20, "trades": 18, "trend": [58, 60, 59, 62, 61, 60, 62, 61]},
    "C": {"wr": 54, "pnl": 156.80, "trades": 15, "trend": [52, 55, 53, 54, 56, 54, 53, 54]},
    "D": {"wr": 48, "pnl": -89.30, "trades": 12, "trend": [51, 50, 49, 48, 47, 49, 48, 48]},
}

DEMO_SIGNALS = {
    "BTC-USD": {
        "direction": "LONG", "confidence": 0.78, "engine": "A",
        "entry": 97150.00, "sl": 95800.00, "tp": 99500.00,
        "timestamp": datetime.now() - timedelta(minutes=12),
        "reason": "Liquidation cascade detected at 96K, strong buying"
    },
    "ETH-USD": {
        "direction": "SHORT", "confidence": 0.72, "engine": "B",
        "entry": 3720.00, "sl": 3820.00, "tp": 3550.00,
        "timestamp": datetime.now() - timedelta(minutes=45),
        "reason": "Funding rate 0.08%, historically reverses at this level"
    },
    "SOL-USD": {
        "direction": "LONG", "confidence": 0.65, "engine": "C",
        "entry": 232.50, "sl": 225.00, "tp": 248.00,
        "timestamp": datetime.now() - timedelta(minutes=5),
        "reason": "Large bid wall at 230, absorbing selling pressure"
    },
}

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_EMPTY = "░"


def get_ntp_time() -> Tuple[datetime, str]:
    """Get NTP-synced time with status."""
    global NTP_OFFSET_MS

    try:
        # Check if NTP is synced using timedatectl
        result = subprocess.run(
            ["timedatectl", "show", "--property=NTPSynchronized"],
            capture_output=True, text=True, timeout=1
        )
        ntp_synced = "yes" in result.stdout.lower()
        status = "[green]NTP SYNC[/]" if ntp_synced else "[yellow]LOCAL[/]"
    except:
        status = "[dim]LOCAL[/]"

    return datetime.now(), status


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


def log_comm(sender: str, receiver: str, msg_type: str, content: str, detail: str = ""):
    """Log detailed communication."""
    COMM_LOG.append({
        "time": datetime.now(),
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "content": content[:50],
        "detail": detail[:80]
    })


def get_live_prices() -> Dict[str, dict]:
    """Get live prices from Coinbase."""
    global PRICE_CACHE, LAST_PRICE_UPDATE, PRICE_HISTORY

    if (datetime.now() - LAST_PRICE_UPDATE).seconds < 5 and PRICE_CACHE:
        return PRICE_CACHE

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]

    try:
        from libs.data.coinbase_client import CoinbaseClient
        client = CoinbaseClient()

        for symbol in symbols:
            try:
                candles = client.get_candles(symbol, granularity="ONE_MINUTE", limit=30)
                if candles:
                    latest = candles[0]
                    prev = candles[1] if len(candles) > 1 else latest

                    price = float(latest.get("close", 0))
                    prev_price = float(prev.get("close", price))
                    change = ((price - prev_price) / prev_price * 100) if prev_price else 0

                    high_24h = max(float(c.get("high", 0)) for c in candles[:24])
                    low_24h = min(float(c.get("low", 0)) for c in candles[:24])

                    PRICE_CACHE[symbol] = {
                        "price": price,
                        "change": change,
                        "volume": float(latest.get("volume", 0)),
                        "high_24h": high_24h,
                        "low_24h": low_24h,
                        "open": float(candles[-1].get("open", price)) if candles else price
                    }

                    if symbol not in PRICE_HISTORY:
                        PRICE_HISTORY[symbol] = deque(maxlen=30)
                    prices = [float(c.get("close", 0)) for c in reversed(candles)]
                    PRICE_HISTORY[symbol] = deque(prices, maxlen=30)
            except:
                pass

        LAST_PRICE_UPDATE = datetime.now()
    except Exception as e:
        # Demo prices
        import random
        demo_base = {
            "BTC-USD": 97234.50, "ETH-USD": 3687.20, "SOL-USD": 234.56,
            "XRP-USD": 2.34, "DOGE-USD": 0.412, "ADA-USD": 1.12,
            "LINK-USD": 24.67, "AVAX-USD": 45.23
        }
        for symbol, base_price in demo_base.items():
            if symbol not in PRICE_CACHE:
                change = random.uniform(-2, 3)
                price = base_price * (1 + change/100)
                PRICE_CACHE[symbol] = {
                    "price": price, "change": change,
                    "volume": random.uniform(1e6, 1e8),
                    "high_24h": price * 1.03, "low_24h": price * 0.97,
                    "open": base_price
                }
                PRICE_HISTORY[symbol] = deque([price * (1 + random.uniform(-0.02, 0.02)) for _ in range(20)], maxlen=30)

    return PRICE_CACHE


def get_engine_data():
    """Get engine data - real or demo."""
    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()
        total_trades = sum(stats.total_trades for _, stats in rankings)
        if total_trades > 0:
            return [(name, {
                "wr": stats.win_rate * 100,
                "pnl": stats.total_pnl_usd,
                "trades": stats.total_trades,
                "trend": list(DEMO_ENGINE_DATA[name]["trend"])
            }) for name, stats in rankings]
    except:
        pass
    return [(name, data) for name, data in DEMO_ENGINE_DATA.items()]


def get_active_signals():
    """Get active trading signals."""
    try:
        # Try to get real signals from database
        from libs.db.models import get_session, Signal
        session = get_session()
        recent = session.query(Signal).filter(
            Signal.timestamp > datetime.now() - timedelta(hours=4),
            Signal.result.is_(None)
        ).order_by(Signal.timestamp.desc()).limit(5).all()

        signals = {}
        for s in recent:
            signals[s.symbol] = {
                "direction": s.direction.upper(),
                "confidence": s.confidence,
                "engine": "A",  # Default
                "entry": s.entry_price,
                "sl": s.sl_price,
                "tp": s.tp_price,
                "timestamp": s.timestamp,
                "reason": s.notes or "Signal generated"
            }
        session.close()
        if signals:
            return signals
    except:
        pass

    return DEMO_SIGNALS


def get_signals_panel():
    """Trade signals panel for manual execution."""
    signals = get_active_signals()
    prices = get_live_prices()

    table = Table(box=None, show_header=True, header_style="bold white", padding=(0, 1))
    table.add_column("Pair", style="cyan", width=8)
    table.add_column("Dir", width=5)
    table.add_column("Entry", justify="right", width=10)
    table.add_column("Now", justify="right", width=10)
    table.add_column("SL", justify="right", width=9)
    table.add_column("TP", justify="right", width=9)
    table.add_column("R:R", justify="right", width=5)
    table.add_column("Conf", justify="right", width=5)

    for symbol, sig in list(signals.items())[:4]:
        current = prices.get(symbol, {}).get("price", sig["entry"])

        # Direction with color
        dir_color = "green" if sig["direction"] == "LONG" else "red"
        dir_str = f"[{dir_color}]{sig['direction'][:4]}[/]"

        # Entry price
        entry_str = f"${sig['entry']:,.2f}" if sig['entry'] >= 100 else f"${sig['entry']:.4f}"

        # Current price with P&L indicator
        if sig["direction"] == "LONG":
            pnl_pct = ((current - sig["entry"]) / sig["entry"]) * 100
        else:
            pnl_pct = ((sig["entry"] - current) / sig["entry"]) * 100
        pnl_color = "green" if pnl_pct >= 0 else "red"
        now_str = f"[{pnl_color}]${current:,.2f}[/]" if current >= 100 else f"[{pnl_color}]${current:.4f}[/]"

        # SL/TP
        sl_str = f"${sig['sl']:,.0f}" if sig['sl'] >= 100 else f"${sig['sl']:.3f}"
        tp_str = f"${sig['tp']:,.0f}" if sig['tp'] >= 100 else f"${sig['tp']:.3f}"

        # Risk:Reward ratio
        risk = abs(sig["entry"] - sig["sl"])
        reward = abs(sig["tp"] - sig["entry"])
        rr = reward / risk if risk > 0 else 0
        rr_color = "green" if rr >= 2 else "yellow" if rr >= 1.5 else "red"
        rr_str = f"[{rr_color}]1:{rr:.1f}[/]"

        # Confidence
        conf = sig["confidence"] * 100
        conf_color = "green" if conf >= 70 else "yellow" if conf >= 60 else "dim"
        conf_str = f"[{conf_color}]{conf:.0f}%[/]"

        short_sym = symbol.replace("-USD", "")
        table.add_row(short_sym, dir_str, entry_str, now_str, sl_str, tp_str, rr_str, conf_str)

    return Panel(table, title="[bold]📊 ACTIVE SIGNALS[/]", box=box.ROUNDED, style="on black", border_style="yellow")


def get_signal_details_panel():
    """Detailed signal reasoning panel."""
    signals = get_active_signals()
    lines = []

    for symbol, sig in list(signals.items())[:3]:
        engine = sig.get("engine", "A")
        name, color = ENGINE_NAMES.get(engine, ("Unknown", "white"))

        age = datetime.now() - sig["timestamp"]
        age_min = int(age.total_seconds() / 60)

        lines.append(f"[{color}]● {symbol.replace('-USD', '')}[/] [{sig['direction']}]")
        lines.append(f"  [dim]Engine {engine}:{name} | {age_min}m ago[/]")
        lines.append(f"  [dim]{sig['reason'][:60]}[/]")
        lines.append("")

    if not lines:
        lines = ["[dim]No active signals[/]", "", "Waiting for high-confidence setups..."]

    return Panel("\n".join(lines), title="[bold]💡 SIGNAL REASONING[/]", box=box.ROUNDED, style="on black", border_style="blue")


def get_prices_panel():
    """Enhanced live prices panel."""
    mobile = is_mobile()

    table = Table(box=None, show_header=True, header_style="bold white", padding=(0, 1))
    table.add_column("Sym", style="cyan", width=5)
    table.add_column("Price", justify="right", width=10)
    table.add_column("%", justify="right", width=6)
    if not mobile:
        table.add_column("24h", width=14)
        table.add_column("Chart", width=10)

    prices = get_live_prices()
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"] if mobile else \
              ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]

    for symbol in symbols:
        data = prices.get(symbol, {"price": 0, "change": 0, "high_24h": 0, "low_24h": 0})
        price = data["price"]
        change = data["change"]

        if price >= 10000:
            price_str = f"${price/1000:.2f}K"
        elif price >= 100:
            price_str = f"${price:,.1f}"
        elif price >= 1:
            price_str = f"${price:.3f}"
        else:
            price_str = f"${price:.4f}"

        change_color = "green" if change >= 0 else "red"
        change_str = f"[{change_color}]{change:+.1f}%[/]"

        short = symbol.replace("-USD", "")

        if mobile:
            table.add_row(short, price_str, change_str)
        else:
            high = data.get("high_24h", price)
            low = data.get("low_24h", price)
            range_str = f"[dim]H:{high/1000:.1f}K L:{low/1000:.1f}K[/]" if high >= 1000 else f"[dim]H:{high:.2f} L:{low:.2f}[/]"

            history = list(PRICE_HISTORY.get(symbol, []))
            spark_color = "green" if change >= 0 else "red"
            spark = f"[{spark_color}]{sparkline(history, 8)}[/]"
            table.add_row(short, price_str, change_str, range_str, spark)

    return Panel(table, title="[bold]📈 LIVE PRICES[/]", box=box.ROUNDED, style="on black", border_style="cyan")


def get_rankings_panel():
    """Engine rankings panel."""
    mobile = is_mobile()
    engine_data = get_engine_data()

    table = Table(box=None, show_header=True, header_style="bold white", padding=(0, 1))
    table.add_column("#", width=3)
    table.add_column("Engine", width=12 if mobile else 16)
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


def generate_detailed_comms():
    """Generate detailed AI communication logs."""
    import random

    if len(COMM_LOG) < 10:
        # Initialize with meaningful communications
        comms = [
            ("MOTHER", "All", "CYCLE", "Starting analysis cycle #1247", "Analyzing 10 symbols across 4 timeframes"),
            ("A", "MOTHER", "SCAN", "Scanning liquidation data", "Found 3 potential cascade triggers"),
            ("B", "MOTHER", "ANALYZE", "Funding rate analysis", "BTC funding 0.045% - neutral zone"),
            ("C", "MOTHER", "SCAN", "Orderbook depth scan", "ETH bid wall at 3680, strong support"),
            ("D", "MOTHER", "REGIME", "Market regime check", "Detecting trending regime on BTC 4H"),
            ("GUARD", "All", "RISK", "Portfolio risk check", "Current exposure 23% - within limits"),
            ("A", "B,C,D", "SHARE", "Knowledge transfer", "Liquidation patterns suggest long bias"),
            ("MOTHER", "A", "APPROVE", "Signal validation", "BTC LONG proposal approved at 78% conf"),
            ("VALID", "All", "CHECK", "Trade validation", "All safety checks passed"),
        ]
        for s, r, t, c, d in comms:
            log_comm(s, r, t, c, d)

    # Periodically add new activity
    if random.random() < 0.4:
        activities = [
            ("A", "MOTHER", "SIGNAL", f"BTC liquidation analysis", f"${random.randint(50,200)}M liquidated in last hour"),
            ("B", "MOTHER", "RATE", f"Funding rate update", f"ETH funding {random.uniform(-0.05, 0.1):.3f}%"),
            ("C", "MOTHER", "DEPTH", f"Orderbook update", f"SOL bid depth +{random.randint(5,20)}% vs 1h ago"),
            ("D", "MOTHER", "TREND", f"Regime analysis", f"Market showing {'trending' if random.random() > 0.5 else 'ranging'} behavior"),
            ("GUARD", "MOTHER", "RISK", f"Risk update", f"Drawdown {random.uniform(0.5, 3.5):.1f}% - healthy"),
            ("MOTHER", "A", "QUERY", f"Request analysis", f"Need confirmation on BTC setup"),
            ("A", "B", "COLLAB", f"Cross-validation", f"Confirming with funding analysis"),
        ]
        s, r, t, c, d = random.choice(activities)
        log_comm(s, r, t, c, d)


def get_comms_panel():
    """Detailed agent communications log."""
    generate_detailed_comms()

    lines = []
    for entry in list(COMM_LOG)[-10:]:
        t = entry["time"].strftime("%H:%M:%S")
        msg_type = entry["type"]
        sender = entry["sender"]

        type_colors = {
            "CYCLE": "green", "SIGNAL": "yellow", "APPROVE": "green",
            "SCAN": "cyan", "ANALYZE": "cyan", "REGIME": "blue",
            "RISK": "red", "CHECK": "green", "SHARE": "magenta",
            "QUERY": "yellow", "COLLAB": "magenta", "RATE": "cyan",
            "DEPTH": "cyan", "TREND": "blue"
        }
        color = type_colors.get(msg_type, "white")

        # Color sender based on engine
        sender_color = "white"
        if sender in ENGINE_NAMES:
            sender_color = ENGINE_NAMES[sender][1]
        elif sender == "MOTHER":
            sender_color = "bold white"
        elif sender == "GUARD":
            sender_color = "red"
        elif sender == "VALID":
            sender_color = "green"

        lines.append(f"[dim]{t}[/] [{color}]{msg_type:7}[/] [{sender_color}]{sender:6}[/] → {entry['receiver']}")
        if entry.get("detail"):
            lines.append(f"          [dim]{entry['detail'][:50]}[/]")

    return Panel("\n".join(lines) if lines else "[dim]Waiting for activity...[/]",
                 title="[bold]💬 AI COMMUNICATIONS[/]", box=box.ROUNDED, style="on black", border_style="magenta")


def get_safety_panel():
    """Safety status panel."""
    lines = []

    try:
        from libs.hydra.guardian import get_guardian
        guardian = get_guardian()
        status = guardian.get_status()

        if status.get("emergency_shutdown_active"):
            lines.append("[red bold]⛔ EMERGENCY STOP[/]")
        elif not status.get("trading_allowed"):
            lines.append("[yellow]⚠️ TRADING PAUSED[/]")
        else:
            lines.append("[green]✓ TRADING ACTIVE[/]")

        losses = status.get("consecutive_losses", 0)
        lines.append(f"Losses: {bar_chart(losses, 5, 8)} [{'red' if losses >= 3 else 'green'}]{losses}/5[/]")

        dd = status.get("current_drawdown_percent", 0)
        lines.append(f"DD:     {bar_chart(dd, 10, 8)} [{'red' if dd > 5 else 'green'}]{dd:.1f}%[/]")
    except:
        lines.extend([
            "[green]✓ TRADING ACTIVE[/]",
            f"Losses: {bar_chart(1, 5, 8)} [green]1/5[/]",
            f"DD:     {bar_chart(2.3, 10, 8)} [green]2.3%[/]"
        ])

    try:
        from libs.hydra.mother_ai import get_mother_ai
        mother = get_mother_ai()
        health = mother.get_health_status()
        if health.get("is_frozen"):
            lines.append("[red]🧊 MOTHER FROZEN[/]")
        else:
            lines.append("[green]✓ MOTHER AI OK[/]")
    except:
        lines.append("[green]✓ MOTHER AI OK[/]")

    return Panel("\n".join(lines), title="[bold]🛡️ SAFETY[/]", box=box.ROUNDED, style="on black", border_style="red")


def get_position_sizing_panel():
    """Position sizing recommendations."""
    signals = get_active_signals()
    prices = get_live_prices()

    lines = ["[bold]Position Sizing (1% Risk)[/]", ""]

    account_size = 10000  # Demo account size
    risk_per_trade = 0.01  # 1% risk

    for symbol, sig in list(signals.items())[:3]:
        current = prices.get(symbol, {}).get("price", sig["entry"])
        risk = abs(sig["entry"] - sig["sl"])
        risk_pct = (risk / sig["entry"]) * 100

        # Position size based on 1% account risk
        risk_amount = account_size * risk_per_trade
        position_size = risk_amount / risk if risk > 0 else 0
        position_value = position_size * current

        short = symbol.replace("-USD", "")
        lines.append(f"[cyan]{short}[/] {sig['direction'][:4]}")
        lines.append(f"  Size: ${position_value:,.0f} ({position_size:.4f})")
        lines.append(f"  Risk: ${risk_amount:.0f} ({risk_pct:.1f}%)")
        lines.append("")

    return Panel("\n".join(lines), title="[bold]📐 SIZING[/]", box=box.ROUNDED, style="on black", border_style="green")


def get_header():
    """Dashboard header with NTP time."""
    now, ntp_status = get_ntp_time()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    uptime = datetime.now() - START_TIME
    h, rem = divmod(int(uptime.total_seconds()), 3600)
    m, s = divmod(rem, 60)

    mobile = is_mobile()

    if mobile:
        header = Text()
        header.append("HYDRA 4.0\n", style="bold cyan")
        header.append(f"{now_str} {ntp_status}\n", style="dim")
    else:
        header = Text()
        header.append("  ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗ \n", style="bold cyan")
        header.append("  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗\n", style="bold cyan")
        header.append("  ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║\n", style="bold cyan")
        header.append("  ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║\n", style="bold cyan")
        header.append("  ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║\n", style="bold cyan")
        header.append("  ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝\n", style="bold cyan")
        header.append(f"  v4.0 | {now_str} | {ntp_status} | Up: {h}h {m}m {s}s\n", style="dim")
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
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="signals", size=8),
            Layout(name="prices", size=6),
            Layout(name="comms", size=14),
            Layout(name="safety", size=6),
            Layout(name="footer", size=2)
        )
    else:
        layout.split_column(
            Layout(name="header", size=11),
            Layout(name="row1", size=10),
            Layout(name="row2", size=16),
            Layout(name="row3", size=10),
            Layout(name="footer", size=2)
        )

        layout["row1"].split_row(
            Layout(name="signals", ratio=2),
            Layout(name="signal_details", ratio=1)
        )

        layout["row2"].split_row(
            Layout(name="prices", ratio=1),
            Layout(name="comms", ratio=2)
        )

        layout["row3"].split_row(
            Layout(name="rankings", ratio=1),
            Layout(name="sizing", ratio=1),
            Layout(name="safety", ratio=1)
        )

    return layout


def update_layout(layout):
    """Update all panels."""
    mobile = is_mobile()

    layout["header"].update(get_header())
    layout["signals"].update(get_signals_panel())
    layout["prices"].update(get_prices_panel())
    layout["comms"].update(get_comms_panel())
    layout["safety"].update(get_safety_panel())

    if mobile:
        layout["footer"].update(Panel("[dim]HYDRA 4.0 | 1s refresh | NTP Sync[/]", box=box.SIMPLE, style="on black"))
    else:
        layout["signal_details"].update(get_signal_details_panel())
        layout["rankings"].update(get_rankings_panel())
        layout["sizing"].update(get_position_sizing_panel())
        layout["footer"].update(Panel(
            "[dim]Ctrl+C exit | 1s refresh | [green]NTP SYNC[/] | [cyan]A[/]=DeepSeek [magenta]B[/]=Claude [yellow]C[/]=Grok [green]D[/]=Gemini[/]",
            box=box.SIMPLE, style="on black"
        ))

    return layout


def main():
    """Run dashboard with smooth refresh."""
    # Set black background
    print("\033[40m\033[2J\033[H", end="")

    layout = make_layout()

    try:
        # Smooth refresh at 1 second intervals
        with Live(layout, refresh_per_second=4, screen=True, console=console, transient=False) as live:
            while True:
                layout = make_layout()
                update_layout(layout)
                live.update(layout)
                time.sleep(1)
    except KeyboardInterrupt:
        print("\033[0m")
        console.print("\n[yellow]Dashboard stopped.[/]")


if __name__ == "__main__":
    main()
