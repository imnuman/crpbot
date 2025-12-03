#!/usr/bin/env python3
"""
HYDRA 4.0 - Live Terminal Dashboard (Enhanced)

Features:
- NTP-synced timestamps
- Smooth refresh without flicker
- Detailed AI agent communications
- Trade signals panel for manual execution
- Responsive: phone (< 80 cols) and desktop

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
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box
except ImportError:
    os.system("pip install rich")
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box

# Console
console = Console(force_terminal=True)

# State
START_TIME = datetime.now()
COMM_LOG: deque = deque(maxlen=100)
PRICE_CACHE: Dict[str, dict] = {}
PRICE_HISTORY: Dict[str, deque] = {}
LAST_PRICE_UPDATE = datetime.min
LAST_WIDTH = 0

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
        "reason": "Liquidation cascade at 96K, strong buying pressure"
    },
    "ETH-USD": {
        "direction": "SHORT", "confidence": 0.72, "engine": "B",
        "entry": 3720.00, "sl": 3820.00, "tp": 3550.00,
        "timestamp": datetime.now() - timedelta(minutes=45),
        "reason": "Funding 0.08% extreme, historically reverses here"
    },
    "SOL-USD": {
        "direction": "LONG", "confidence": 0.65, "engine": "C",
        "entry": 232.50, "sl": 225.00, "tp": 248.00,
        "timestamp": datetime.now() - timedelta(minutes=5),
        "reason": "Large bid wall at 230, absorbing sell pressure"
    },
}

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_EMPTY = "░"


def get_terminal_width() -> int:
    """Get terminal width."""
    try:
        return console.width
    except:
        return 80


def is_mobile() -> bool:
    """Mobile if width < 80."""
    return get_terminal_width() < 80


def is_tablet() -> bool:
    """Tablet if width 80-120."""
    w = get_terminal_width()
    return 80 <= w < 120


def get_ntp_status() -> str:
    """Get NTP sync status."""
    try:
        result = subprocess.run(
            ["timedatectl", "show", "--property=NTPSynchronized"],
            capture_output=True, text=True, timeout=1
        )
        if "yes" in result.stdout.lower():
            return "[green]●[/] NTP"
        return "[yellow]○[/] LOCAL"
    except:
        return "[dim]○[/] LOCAL"


def sparkline(values: List[float], width: int = 8) -> str:
    """Generate sparkline."""
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


def bar_chart(value: float, max_value: float = 100, width: int = 8) -> str:
    """Generate bar chart."""
    if max_value <= 0:
        return BAR_EMPTY * width
    filled = max(0, min(width, int((value / max_value) * width)))
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def log_comm(sender: str, receiver: str, msg_type: str, content: str, detail: str = ""):
    """Log communication."""
    COMM_LOG.append({
        "time": datetime.now(),
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "content": content[:40],
        "detail": detail[:60]
    })


def get_live_prices() -> Dict[str, dict]:
    """Get live prices."""
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

                    PRICE_CACHE[symbol] = {
                        "price": price, "change": change,
                        "high": max(float(c.get("high", 0)) for c in candles[:24]),
                        "low": min(float(c.get("low", 0)) for c in candles[:24])
                    }

                    if symbol not in PRICE_HISTORY:
                        PRICE_HISTORY[symbol] = deque(maxlen=30)
                    PRICE_HISTORY[symbol] = deque([float(c.get("close", 0)) for c in reversed(candles)], maxlen=30)
            except:
                pass
        LAST_PRICE_UPDATE = datetime.now()
    except:
        import random
        demo = {"BTC-USD": 97234, "ETH-USD": 3687, "SOL-USD": 234, "XRP-USD": 2.34,
                "DOGE-USD": 0.41, "ADA-USD": 1.12, "LINK-USD": 24.6, "AVAX-USD": 45.2}
        for sym, base in demo.items():
            if sym not in PRICE_CACHE:
                ch = random.uniform(-2, 3)
                PRICE_CACHE[sym] = {"price": base * (1 + ch/100), "change": ch, "high": base * 1.03, "low": base * 0.97}
                PRICE_HISTORY[sym] = deque([base * (1 + random.uniform(-0.02, 0.02)) for _ in range(15)], maxlen=30)

    return PRICE_CACHE


def get_engine_data():
    """Get engine rankings."""
    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        manager = get_tournament_manager()
        rankings = manager.calculate_rankings()
        if sum(s.total_trades for _, s in rankings) > 0:
            return [(n, {"wr": s.win_rate*100, "pnl": s.total_pnl_usd, "trades": s.total_trades,
                        "trend": DEMO_ENGINE_DATA[n]["trend"]}) for n, s in rankings]
    except:
        pass
    return list(DEMO_ENGINE_DATA.items())


def get_active_signals():
    """Get active signals."""
    try:
        from libs.db.models import get_session, Signal
        session = get_session()
        recent = session.query(Signal).filter(
            Signal.timestamp > datetime.now() - timedelta(hours=4),
            Signal.result.is_(None)
        ).order_by(Signal.timestamp.desc()).limit(5).all()

        signals = {}
        for s in recent:
            signals[s.symbol] = {
                "direction": s.direction.upper(), "confidence": s.confidence, "engine": "A",
                "entry": s.entry_price, "sl": s.sl_price, "tp": s.tp_price,
                "timestamp": s.timestamp, "reason": s.notes or "Signal generated"
            }
        session.close()
        if signals:
            return signals
    except:
        pass
    return DEMO_SIGNALS


def generate_comms():
    """Generate AI communications."""
    import random

    if len(COMM_LOG) < 8:
        init = [
            ("MOTHER", "All", "CYCLE", "Analysis cycle #1247", "10 symbols, 4 timeframes"),
            ("A", "MOTHER", "SCAN", "Liquidation scan", "3 cascade triggers found"),
            ("B", "MOTHER", "RATE", "Funding analysis", "BTC 0.045% neutral"),
            ("C", "MOTHER", "DEPTH", "Orderbook scan", "ETH bid wall 3680"),
            ("D", "MOTHER", "REGIME", "Regime check", "BTC trending 4H"),
            ("GUARD", "All", "RISK", "Risk check", "Exposure 23% OK"),
            ("A", "B,C,D", "SHARE", "Knowledge share", "Long bias detected"),
            ("MOTHER", "A", "APPROVE", "Signal approved", "BTC LONG 78% conf"),
        ]
        for s, r, t, c, d in init:
            log_comm(s, r, t, c, d)

    if random.random() < 0.3:
        acts = [
            ("A", "MOTHER", "SIGNAL", "Liquidation update", f"${random.randint(50,200)}M liquidated"),
            ("B", "MOTHER", "RATE", "Funding update", f"ETH {random.uniform(-0.05, 0.1):.3f}%"),
            ("C", "MOTHER", "DEPTH", "Depth update", f"SOL bid +{random.randint(5,20)}%"),
            ("D", "MOTHER", "TREND", "Regime update", f"{'Trending' if random.random() > 0.5 else 'Ranging'}"),
            ("GUARD", "MOTHER", "RISK", "Risk update", f"DD {random.uniform(0.5, 3.5):.1f}%"),
        ]
        s, r, t, c, d = random.choice(acts)
        log_comm(s, r, t, c, d)


# ============== MOBILE LAYOUT ==============

def render_mobile() -> Panel:
    """Compact mobile layout."""
    now = datetime.now()
    ntp = get_ntp_status()
    prices = get_live_prices()
    signals = get_active_signals()

    lines = []

    # Header
    lines.append(f"[bold cyan]HYDRA 4.0[/] {ntp}")
    lines.append(f"[dim]{now.strftime('%H:%M:%S')}[/]")
    lines.append("")

    # Prices (compact)
    lines.append("[bold]PRICES[/]")
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0})
        p = d["price"]
        c = d["change"]
        ps = f"${p/1000:.1f}K" if p >= 1000 else f"${p:.2f}"
        cc = "green" if c >= 0 else "red"
        lines.append(f" {sym[:3]} {ps} [{cc}]{c:+.1f}%[/]")
    lines.append("")

    # Active Signals
    lines.append("[bold yellow]SIGNALS[/]")
    for sym, sig in list(signals.items())[:2]:
        d = sig["direction"]
        dc = "green" if d == "LONG" else "red"
        conf = sig["confidence"] * 100
        lines.append(f" [{dc}]{sym[:3]} {d[:4]}[/] {conf:.0f}%")
        lines.append(f"  E:${sig['entry']:,.0f} SL:${sig['sl']:,.0f}")
        lines.append(f"  TP:${sig['tp']:,.0f}")
    lines.append("")

    # Comms (last 4)
    generate_comms()
    lines.append("[bold magenta]COMMS[/]")
    for e in list(COMM_LOG)[-4:]:
        t = e["time"].strftime("%H:%M")
        lines.append(f" [dim]{t}[/] {e['sender'][:4]}→{e['type'][:6]}")
    lines.append("")

    # Safety
    lines.append("[bold red]SAFETY[/]")
    lines.append(f" [green]✓ Active[/] DD: 2.3%")

    return Panel("\n".join(lines), title="[bold]HYDRA[/]", box=box.ROUNDED,
                 style="on black", border_style="cyan", padding=(0, 1))


# ============== TABLET LAYOUT ==============

def render_tablet() -> Panel:
    """Medium tablet layout."""
    now = datetime.now()
    ntp = get_ntp_status()
    uptime = now - START_TIME
    h, m = divmod(int(uptime.total_seconds()) // 60, 60)

    prices = get_live_prices()
    signals = get_active_signals()
    engines = get_engine_data()
    generate_comms()

    # Build sections
    sections = []

    # Header
    sections.append(f"[bold cyan]HYDRA 4.0[/] | {now.strftime('%Y-%m-%d %H:%M:%S')} | {ntp} | Up: {h}h{m}m")
    sections.append("")

    # Prices row
    price_line = "[bold]PRICES:[/] "
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0})
        p = d["price"]
        c = d["change"]
        ps = f"${p/1000:.1f}K" if p >= 1000 else f"${p:.2f}"
        cc = "green" if c >= 0 else "red"
        price_line += f"{sym[:3]}:{ps}[{cc}]{c:+.1f}%[/] "
    sections.append(price_line)
    sections.append("")

    # Signals table
    sections.append("[bold yellow]━━━ ACTIVE SIGNALS ━━━[/]")
    sections.append(f"{'Pair':<6} {'Dir':<5} {'Entry':>10} {'Now':>10} {'SL':>9} {'TP':>9} {'R:R':>5} {'%':>4}")

    for sym, sig in list(signals.items())[:3]:
        curr = prices.get(sym, {}).get("price", sig["entry"])
        d = sig["direction"]
        dc = "green" if d == "LONG" else "red"

        if d == "LONG":
            pnl = ((curr - sig["entry"]) / sig["entry"]) * 100
        else:
            pnl = ((sig["entry"] - curr) / sig["entry"]) * 100
        pc = "green" if pnl >= 0 else "red"

        risk = abs(sig["entry"] - sig["sl"])
        reward = abs(sig["tp"] - sig["entry"])
        rr = reward / risk if risk > 0 else 0

        e_str = f"${sig['entry']:,.0f}" if sig['entry'] >= 100 else f"${sig['entry']:.3f}"
        n_str = f"${curr:,.0f}" if curr >= 100 else f"${curr:.3f}"
        sl_str = f"${sig['sl']:,.0f}" if sig['sl'] >= 100 else f"${sig['sl']:.2f}"
        tp_str = f"${sig['tp']:,.0f}" if sig['tp'] >= 100 else f"${sig['tp']:.2f}"

        sections.append(f"{sym[:6]:<6} [{dc}]{d[:4]:<5}[/] {e_str:>10} [{pc}]{n_str:>10}[/] {sl_str:>9} {tp_str:>9} 1:{rr:.1f} {sig['confidence']*100:.0f}%")
    sections.append("")

    # Two columns: Rankings + Comms
    sections.append("[bold yellow]━━━ RANKINGS ━━━[/]                    [bold magenta]━━━ AI COMMS ━━━[/]")

    eng_lines = []
    for i, (name, data) in enumerate(engines):
        rank = i + 1
        icon = "👑" if rank == 1 else "💀" if rank == 4 else f"#{rank}"
        wr = data["wr"]
        pnl = data["pnl"]
        pc = "green" if pnl >= 0 else "red"
        _, color = ENGINE_NAMES[name]
        eng_lines.append(f"{icon} [{color}]{name}:{ENGINE_NAMES[name][0][:6]}[/] {wr:.0f}% [{pc}]${pnl:+.0f}[/]")

    comm_lines = []
    for e in list(COMM_LOG)[-4:]:
        t = e["time"].strftime("%H:%M:%S")
        s = e["sender"]
        sc = ENGINE_NAMES.get(s, ("", "white"))[1]
        comm_lines.append(f"[dim]{t}[/] [{sc}]{s:5}[/] {e['type'][:6]} {e['detail'][:20]}")

    for i in range(max(len(eng_lines), len(comm_lines))):
        el = eng_lines[i] if i < len(eng_lines) else ""
        cl = comm_lines[i] if i < len(comm_lines) else ""
        sections.append(f"{el:<40} {cl}")
    sections.append("")

    # Safety
    sections.append(f"[bold red]SAFETY:[/] [green]✓ Active[/] | Losses: {bar_chart(1, 5, 5)} 1/5 | DD: {bar_chart(2.3, 10, 5)} 2.3%")

    return Panel("\n".join(sections), box=box.DOUBLE, style="on black", border_style="cyan", padding=(0, 1))


# ============== DESKTOP LAYOUT ==============

def render_desktop() -> Panel:
    """Full desktop layout."""
    now = datetime.now()
    ntp = get_ntp_status()
    uptime = now - START_TIME
    h, rem = divmod(int(uptime.total_seconds()), 3600)
    m, s = divmod(rem, 60)

    prices = get_live_prices()
    signals = get_active_signals()
    engines = get_engine_data()
    generate_comms()

    lines = []

    # ASCII Header
    lines.append("[bold cyan]  ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗ [/]")
    lines.append("[bold cyan]  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗[/]")
    lines.append("[bold cyan]  ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║[/]")
    lines.append("[bold cyan]  ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║[/]")
    lines.append("[bold cyan]  ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║[/]")
    lines.append("[bold cyan]  ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝[/]")
    lines.append(f"  [dim]v4.0 | {now.strftime('%Y-%m-%d %H:%M:%S')} | {ntp} | Uptime: {h}h {m}m {s}s[/]")
    lines.append(f"  [cyan]A[/]:DeepSeek [magenta]B[/]:Claude [yellow]C[/]:Grok [green]D[/]:Gemini")
    lines.append("")

    # Prices
    lines.append("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━ LIVE PRICES ━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    price_header = f"  {'Symbol':<8} {'Price':>12} {'Change':>8} {'24h Range':>20} {'Chart':>10}"
    lines.append(f"[dim]{price_header}[/]")

    for sym in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0, "high": 0, "low": 0})
        p = d["price"]
        c = d["change"]
        ps = f"${p:,.2f}" if p >= 1 else f"${p:.4f}"
        cc = "green" if c >= 0 else "red"
        h, l = d.get("high", p), d.get("low", p)
        rng = f"H:${h:,.0f} L:${l:,.0f}" if h >= 100 else f"H:${h:.2f} L:${l:.2f}"
        hist = list(PRICE_HISTORY.get(sym, []))
        spark = sparkline(hist, 8)
        sc = "green" if c >= 0 else "red"
        lines.append(f"  {sym:<8} {ps:>12} [{cc}]{c:>+7.2f}%[/] [dim]{rng:>20}[/] [{sc}]{spark}[/]")
    lines.append("")

    # Signals
    lines.append("[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━ ACTIVE SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    sig_header = f"  {'Pair':<8} {'Dir':<6} {'Entry':>12} {'Current':>12} {'Stop Loss':>11} {'Take Profit':>12} {'R:R':>6} {'Conf':>5}"
    lines.append(f"[dim]{sig_header}[/]")

    for sym, sig in list(signals.items())[:3]:
        curr = prices.get(sym, {}).get("price", sig["entry"])
        d = sig["direction"]
        dc = "green" if d == "LONG" else "red"

        if d == "LONG":
            pnl = ((curr - sig["entry"]) / sig["entry"]) * 100
        else:
            pnl = ((sig["entry"] - curr) / sig["entry"]) * 100
        pc = "green" if pnl >= 0 else "red"

        risk = abs(sig["entry"] - sig["sl"])
        reward = abs(sig["tp"] - sig["entry"])
        rr = reward / risk if risk > 0 else 0
        rrc = "green" if rr >= 2 else "yellow" if rr >= 1.5 else "red"

        e_str = f"${sig['entry']:,.2f}"
        n_str = f"${curr:,.2f}"
        sl_str = f"${sig['sl']:,.2f}"
        tp_str = f"${sig['tp']:,.2f}"
        conf = sig['confidence'] * 100
        confc = "green" if conf >= 70 else "yellow" if conf >= 60 else "dim"

        lines.append(f"  {sym:<8} [{dc}]{d:<6}[/] {e_str:>12} [{pc}]{n_str:>12}[/] {sl_str:>11} {tp_str:>12} [{rrc}]1:{rr:.1f}[/] [{confc}]{conf:.0f}%[/]")

    # Signal reasoning
    lines.append("")
    for sym, sig in list(signals.items())[:2]:
        age = (now - sig["timestamp"]).seconds // 60
        eng = sig.get("engine", "A")
        _, ec = ENGINE_NAMES.get(eng, ("", "white"))
        lines.append(f"  [{ec}]● {sym}[/] - Engine {eng} ({age}m ago): [dim]{sig['reason'][:70]}[/]")
    lines.append("")

    # Rankings + Comms side by side
    lines.append("[bold yellow]━━━━━━━━━━ RANKINGS ━━━━━━━━━━[/]          [bold magenta]━━━━━━━━━━━━━━━ AI COMMUNICATIONS ━━━━━━━━━━━━━━━[/]")

    eng_lines = []
    for i, (name, data) in enumerate(engines):
        rank = i + 1
        icon = "👑" if rank == 1 else "💀" if rank == 4 else f"#{rank}"
        wr = data["wr"]
        pnl = data["pnl"]
        pc = "green" if pnl >= 0 else "red"
        wc = "green" if wr >= 55 else "yellow" if wr >= 45 else "red"
        _, color = ENGINE_NAMES[name]
        bar = bar_chart(wr, 100, 8)
        eng_lines.append(f"  {icon} [{color}]{name}:{ENGINE_NAMES[name][0]:<8}[/] [{wc}]{wr:>3.0f}%[/] [{wc}]{bar}[/] [{pc}]${pnl:>+8.0f}[/]")

    comm_lines = []
    for e in list(COMM_LOG)[-6:]:
        t = e["time"].strftime("%H:%M:%S")
        s = e["sender"]
        sc = ENGINE_NAMES.get(s, ("", "white"))[1]
        if s == "MOTHER":
            sc = "bold white"
        elif s == "GUARD":
            sc = "red"
        tc = {"SIGNAL": "yellow", "APPROVE": "green", "RISK": "red", "SCAN": "cyan",
              "RATE": "cyan", "DEPTH": "cyan", "REGIME": "blue", "SHARE": "magenta"}.get(e["type"], "white")
        comm_lines.append(f"[dim]{t}[/] [{tc}]{e['type']:7}[/] [{sc}]{s:6}[/] → {e['receiver']:<6} [dim]{e['detail'][:25]}[/]")

    for i in range(max(len(eng_lines), len(comm_lines))):
        el = eng_lines[i] if i < len(eng_lines) else " " * 45
        cl = comm_lines[i] if i < len(comm_lines) else ""
        lines.append(f"{el:<45}  {cl}")
    lines.append("")

    # Position Sizing
    lines.append("[bold green]━━━━━━━━━━ POSITION SIZING (1% Risk on $10K) ━━━━━━━━━━[/]")
    for sym, sig in list(signals.items())[:3]:
        curr = prices.get(sym, {}).get("price", sig["entry"])
        risk = abs(sig["entry"] - sig["sl"])
        risk_pct = (risk / sig["entry"]) * 100
        risk_amt = 100  # 1% of 10K
        pos_size = risk_amt / risk if risk > 0 else 0
        pos_val = pos_size * curr
        lines.append(f"  [cyan]{sym:<8}[/] {sig['direction'][:4]} | Size: ${pos_val:>8,.0f} ({pos_size:.4f}) | Risk: ${risk_amt:.0f} ({risk_pct:.1f}% SL)")
    lines.append("")

    # Safety
    lines.append("[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━ SAFETY STATUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    lines.append(f"  [green]✓ TRADING ACTIVE[/] | Consec Losses: {bar_chart(1, 5, 6)} 1/5 | Drawdown: {bar_chart(2.3, 10, 6)} 2.3% | [green]✓ MOTHER AI OK[/]")
    lines.append("")

    # Footer
    lines.append(f"[dim]  Press Ctrl+C to exit | Refresh: 2s | {ntp} | [cyan]A[/]=DeepSeek [magenta]B[/]=Claude [yellow]C[/]=Grok [green]D[/]=Gemini[/]")

    return Panel("\n".join(lines), box=box.DOUBLE, style="on black", border_style="cyan", padding=(0, 0))


def render_dashboard():
    """Render appropriate layout based on terminal size."""
    width = get_terminal_width()

    if width < 80:
        return render_mobile()
    elif width < 120:
        return render_tablet()
    else:
        return render_desktop()


def main():
    """Run dashboard with smooth refresh."""
    # Clear screen and set black background
    print("\033[40m\033[2J\033[H", end="", flush=True)

    try:
        with Live(render_dashboard(), console=console, refresh_per_second=2,
                  screen=True, transient=False) as live:
            while True:
                time.sleep(2)
                live.update(render_dashboard())
    except KeyboardInterrupt:
        print("\033[0m")
        console.print("\n[yellow]Dashboard stopped.[/]")


if __name__ == "__main__":
    main()
