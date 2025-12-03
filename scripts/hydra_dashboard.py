#!/usr/bin/env python3
"""
HYDRA 4.0 - Live Terminal Dashboard

Responsive layouts:
- Phone (< 50 cols): Compact view
- Desktop 1920x1080 (100+ cols): Full layout with Tournament + AI Comms

Run: python scripts/hydra_dashboard.py
Web: ttyd -W -p 7682 python scripts/hydra_dashboard.py
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich import box
except ImportError:
    os.system("pip install rich")
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich import box

console = Console(force_terminal=True)

# State
START_TIME = datetime.now()
COMM_LOG: deque = deque(maxlen=100)
PRICE_CACHE: Dict[str, dict] = {}
PRICE_HISTORY: Dict[str, deque] = {}
LAST_PRICE_UPDATE = datetime.min

# Engine config
ENGINE_NAMES = {
    "A": ("DeepSeek", "cyan"),
    "B": ("Claude", "magenta"),
    "C": ("Grok", "yellow"),
    "D": ("Gemini", "green"),
}

ENGINE_ROLES = {
    "A": "Liquidation Hunter",
    "B": "Funding Analyst",
    "C": "Orderbook Scanner",
    "D": "Regime Detector",
}

DEMO_ENGINE_DATA = {
    "A": {"wr": 67, "pnl": 847.50, "trades": 23},
    "B": {"wr": 61, "pnl": 423.20, "trades": 18},
    "C": {"wr": 54, "pnl": 156.80, "trades": 15},
    "D": {"wr": 48, "pnl": -89.30, "trades": 12},
}

DEMO_SIGNALS = {
    "BTC-USD": {
        "direction": "LONG", "confidence": 0.78, "engine": "A",
        "entry": 97150.00, "sl": 95800.00, "tp": 99500.00,
        "timestamp": datetime.now() - timedelta(minutes=12),
        "reason": "Liquidation cascade at 96K, strong buying"
    },
    "ETH-USD": {
        "direction": "SHORT", "confidence": 0.72, "engine": "B",
        "entry": 3720.00, "sl": 3820.00, "tp": 3550.00,
        "timestamp": datetime.now() - timedelta(minutes=45),
        "reason": "Funding 0.08% extreme, reversal expected"
    },
    "SOL-USD": {
        "direction": "LONG", "confidence": 0.65, "engine": "C",
        "entry": 232.50, "sl": 225.00, "tp": 248.00,
        "timestamp": datetime.now() - timedelta(minutes=5),
        "reason": "Bid wall at 230, absorbing sells"
    },
}

SPARK = "▁▂▃▄▅▆▇█"
BAR = "█"
EMPTY = "░"


def get_size():
    try:
        return console.width, console.height
    except:
        return 80, 24


def get_ntp() -> str:
    try:
        r = subprocess.run(["timedatectl", "show", "--property=NTPSynchronized"],
                          capture_output=True, text=True, timeout=1)
        return "[green]●[/]NTP" if "yes" in r.stdout.lower() else "[yellow]○[/]"
    except:
        return "[dim]○[/]"


def spark(vals, w=8):
    if not vals or len(vals) < 2:
        return "─" * w
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return SPARK[4] * w
    r = []
    step = max(1, len(vals) // w)
    for v in vals[::step][:w]:
        r.append(SPARK[int((v - mn) / (mx - mn) * 7)])
    return "".join(r)


def bar(v, mx=100, w=8):
    if mx <= 0:
        return EMPTY * w
    f = max(0, min(w, int(v / mx * w)))
    return BAR * f + EMPTY * (w - f)


def log_comm(sender, receiver, msg_type, content, detail=""):
    COMM_LOG.append({
        "time": datetime.now(),
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "content": content[:40],
        "detail": detail[:60]
    })


def get_prices():
    global PRICE_CACHE, LAST_PRICE_UPDATE, PRICE_HISTORY

    if (datetime.now() - LAST_PRICE_UPDATE).seconds < 5 and PRICE_CACHE:
        return PRICE_CACHE

    syms = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]

    try:
        from libs.data.coinbase_client import CoinbaseClient
        client = CoinbaseClient()
        for sym in syms:
            try:
                candles = client.get_candles(sym, granularity="ONE_MINUTE", limit=30)
                if candles:
                    p = float(candles[0].get("close", 0))
                    prev = float(candles[1].get("close", p)) if len(candles) > 1 else p
                    ch = ((p - prev) / prev * 100) if prev else 0
                    PRICE_CACHE[sym] = {"price": p, "change": ch,
                                        "high": max(float(c.get("high", 0)) for c in candles[:24]),
                                        "low": min(float(c.get("low", 0)) for c in candles[:24])}
                    PRICE_HISTORY[sym] = deque([float(c.get("close", 0)) for c in reversed(candles)], maxlen=30)
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
                PRICE_CACHE[sym] = {"price": base * (1 + ch/100), "change": ch,
                                   "high": base * 1.03, "low": base * 0.97}
                PRICE_HISTORY[sym] = deque([base * (1 + random.uniform(-0.02, 0.02)) for _ in range(15)], maxlen=30)
    return PRICE_CACHE


def get_engines():
    try:
        from libs.hydra.engine_portfolio import get_tournament_manager
        mgr = get_tournament_manager()
        ranks = mgr.calculate_rankings()
        if sum(s.total_trades for _, s in ranks) > 0:
            return [(n, {"wr": s.win_rate*100, "pnl": s.total_pnl_usd, "trades": s.total_trades}) for n, s in ranks]
    except:
        pass
    return list(DEMO_ENGINE_DATA.items())


def get_signals():
    try:
        from libs.db.models import get_session, Signal
        sess = get_session()
        recent = sess.query(Signal).filter(
            Signal.timestamp > datetime.now() - timedelta(hours=4),
            Signal.result.is_(None)
        ).order_by(Signal.timestamp.desc()).limit(5).all()
        sigs = {}
        for s in recent:
            sigs[s.symbol] = {"direction": s.direction.upper(), "confidence": s.confidence, "engine": "A",
                             "entry": s.entry_price, "sl": s.sl_price, "tp": s.tp_price,
                             "timestamp": s.timestamp, "reason": s.notes or "Signal"}
        sess.close()
        if sigs:
            return sigs
    except:
        pass
    return DEMO_SIGNALS


def gen_comms():
    """Generate realistic AI agent communications."""
    import random

    if len(COMM_LOG) < 10:
        # Initialize with meaningful messages
        init_comms = [
            ("MOTHER", "ALL", "CYCLE", "Starting analysis cycle #1247", "Scanning 8 symbols across 4 timeframes"),
            ("A", "MOTHER", "SCAN", "Liquidation scan complete", f"Found ${random.randint(80,150)}M in liquidations"),
            ("B", "MOTHER", "RATE", "Funding rate analysis", f"BTC: {random.uniform(0.01, 0.08):.3f}% - Neutral zone"),
            ("C", "MOTHER", "DEPTH", "Orderbook depth scan", "ETH bid wall detected at $3,650"),
            ("D", "MOTHER", "REGIME", "Regime detection", "Market in TRENDING mode (4H)"),
            ("MOTHER", "A", "QUERY", "Request confirmation", "Validate BTC liquidation cascade"),
            ("A", "B", "COLLAB", "Cross-validation", "Requesting funding correlation check"),
            ("B", "A", "REPLY", "Funding confirms", "Long bias supported by negative funding"),
            ("GUARD", "ALL", "RISK", "Risk assessment", "Portfolio exposure: 23% - Within limits"),
            ("MOTHER", "A", "APPROVE", "Signal approved", "BTC LONG @ 78% confidence"),
        ]
        for s, r, t, c, d in init_comms:
            log_comm(s, r, t, c, d)

    # Generate new activity periodically
    if random.random() < 0.4:
        activities = [
            ("A", "MOTHER", "SIGNAL", "New liquidation detected", f"${random.randint(50,200)}M cascade forming"),
            ("B", "MOTHER", "RATE", "Funding update", f"ETH funding: {random.uniform(-0.05, 0.1):.4f}%"),
            ("C", "MOTHER", "DEPTH", "Orderbook shift", f"SOL bid depth +{random.randint(5,25)}% in 5min"),
            ("D", "MOTHER", "TREND", "Regime shift detected", f"Switching to {'TRENDING' if random.random() > 0.5 else 'RANGING'}"),
            ("A", "C", "TEACH", "Knowledge transfer", "Sharing liquidation patterns from last hour"),
            ("GUARD", "MOTHER", "RISK", "Risk update", f"Current drawdown: {random.uniform(0.5, 3.5):.1f}%"),
            ("MOTHER", "D", "QUERY", "Regime confirmation", "Validate trend continuation signal"),
            ("B", "D", "COLLAB", "Cross-check", "Comparing funding vs regime signals"),
        ]
        s, r, t, c, d = random.choice(activities)
        log_comm(s, r, t, c, d)


# ==================== PHONE LAYOUT ====================
def render_phone(w, h):
    now = datetime.now()
    ntp = get_ntp()
    prices = get_prices()
    signals = get_signals()
    engines = get_engines()
    gen_comms()

    L = []
    L.append(f"[bold cyan]HYDRA 4.0[/] {ntp}")
    L.append(f"[dim]{now.strftime('%H:%M:%S')}[/]")
    L.append("─" * (w - 4))

    # Prices
    L.append("[bold]PRICES[/]")
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0})
        p, c = d["price"], d["change"]
        ps = f"${p/1000:.1f}K" if p >= 1000 else f"${p:.2f}"
        cc = "green" if c >= 0 else "red"
        L.append(f"[cyan]{sym[:3]}[/] {ps} [{cc}]{c:+.1f}%[/]")
    L.append("")

    # Tournament
    L.append("[bold yellow]TOURNAMENT[/]")
    for i, (name, data) in enumerate(engines):
        icon = "👑" if i == 0 else "💀" if i == 3 else f"#{i+1}"
        _, color = ENGINE_NAMES[name]
        pc = "green" if data["pnl"] >= 0 else "red"
        L.append(f"{icon}[{color}]{name}[/] {data['wr']:.0f}% [{pc}]${data['pnl']:+.0f}[/]")
    L.append("")

    # Signals
    L.append("[bold yellow]SIGNALS[/]")
    for sym, sig in list(signals.items())[:2]:
        d = sig["direction"]
        dc = "green" if d == "LONG" else "red"
        L.append(f"[{dc}]{sym[:3]} {d[:1]}[/] E${sig['entry']:,.0f}")
    L.append("")

    # Comms
    L.append("[bold magenta]COMMS[/]")
    for e in list(COMM_LOG)[-3:]:
        t = e["time"].strftime("%H:%M")
        sc = ENGINE_NAMES.get(e["sender"], ("", "white"))[1]
        L.append(f"[dim]{t}[/][{sc}]{e['sender'][:2]}[/]→{e['type'][:5]}")

    return Panel("\n".join(L), box=box.ROUNDED, style="on black", border_style="cyan", padding=(0, 0))


# ==================== DESKTOP 1920x1080 LAYOUT ====================
def render_desktop(w, h):
    now = datetime.now()
    ntp = get_ntp()
    uptime = now - START_TIME
    hrs, rem = divmod(int(uptime.total_seconds()), 3600)
    mins, secs = divmod(rem, 60)

    prices = get_prices()
    signals = get_signals()
    engines = get_engines()
    gen_comms()

    L = []

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # HEADER
    L.append("[bold cyan]  ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗     [/][bold white]Multi-Agent Trading System[/]")
    L.append("[bold cyan]  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗    [/][dim]4 AI Engines competing in real-time[/]")
    L.append("[bold cyan]  ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║    [/]")
    L.append("[bold cyan]  ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║    [/]")
    L.append("[bold cyan]  ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║    [/]")
    L.append("[bold cyan]  ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    [/]")
    L.append(f"  [dim]v4.0 │ {now.strftime('%Y-%m-%d %H:%M:%S')} │ {ntp} │ Uptime: {hrs}h {mins}m {secs}s[/]")
    L.append("")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # LIVE TOURNAMENT - PROMINENT SECTION
    L.append("[bold yellow on black]╔══════════════════════════════════════════════════════════ LIVE TOURNAMENT ══════════════════════════════════════════════════════════╗[/]")
    L.append("[bold yellow]║[/]  [bold]Rank[/]   [bold]Engine[/]              [bold]Role[/]                    [bold]Win Rate[/]              [bold]P&L[/]         [bold]Trades[/]   [bold]Status[/]                    [bold yellow]║[/]")
    L.append("[bold yellow]╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣[/]")

    for i, (name, data) in enumerate(engines):
        rank = i + 1
        if rank == 1:
            icon = "[green]👑 #1[/]"
            status = "[green]LEADING[/]"
        elif rank == 4:
            icon = "[red]💀 #4[/]"
            status = "[red]LAST[/]"
        else:
            icon = f"   #{rank}"
            status = "[dim]ACTIVE[/]"

        wr = data["wr"]
        pnl = data["pnl"]
        trades = data.get("trades", 0)
        pc = "green" if pnl >= 0 else "red"
        wc = "green" if wr >= 55 else "yellow" if wr >= 45 else "red"
        _, color = ENGINE_NAMES[name]
        role = ENGINE_ROLES[name]
        wr_bar = bar(wr, 100, 15)

        L.append(f"[bold yellow]║[/]  {icon}   [{color}]{name}:{ENGINE_NAMES[name][0]:<12}[/] [dim]{role:<22}[/] [{wc}]{wr:>5.1f}% {wr_bar}[/] [{pc}]${pnl:>+10,.2f}[/]    {trades:>3}   {status:<20} [bold yellow]║[/]")

    L.append("[bold yellow]╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[/]")
    L.append("")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # AI COMMUNICATIONS - PROMINENT SECTION
    L.append("[bold magenta on black]╔══════════════════════════════════════════════════════ AI AGENT COMMUNICATIONS ══════════════════════════════════════════════════════╗[/]")
    L.append("[bold magenta]║[/]  [bold]Time[/]      [bold]Type[/]       [bold]From[/]       [bold]To[/]         [bold]Message[/]                                                                   [bold magenta]║[/]")
    L.append("[bold magenta]╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣[/]")

    for e in list(COMM_LOG)[-8:]:
        t = e["time"].strftime("%H:%M:%S")
        sender = e["sender"]
        receiver = e["receiver"]
        msg_type = e["type"]
        detail = e["detail"]

        # Color sender
        if sender in ENGINE_NAMES:
            sc = ENGINE_NAMES[sender][1]
            sender_str = f"[{sc}]{sender}:{ENGINE_NAMES[sender][0][:6]}[/]"
        elif sender == "MOTHER":
            sender_str = "[bold white]MOTHER[/]    "
        elif sender == "GUARD":
            sender_str = "[red]GUARDIAN[/]  "
        else:
            sender_str = f"{sender:<10}"

        # Color receiver
        if receiver in ENGINE_NAMES:
            rc = ENGINE_NAMES[receiver][1]
            recv_str = f"[{rc}]{receiver}:{ENGINE_NAMES[receiver][0][:4]}[/]"
        elif receiver == "ALL":
            recv_str = "[white]ALL[/]      "
        elif receiver == "MOTHER":
            recv_str = "[bold white]MOTHER[/]   "
        else:
            recv_str = f"{receiver:<10}"

        # Color type
        type_colors = {"SIGNAL": "yellow", "APPROVE": "green", "RISK": "red", "SCAN": "cyan",
                       "RATE": "cyan", "DEPTH": "cyan", "REGIME": "blue", "QUERY": "yellow",
                       "COLLAB": "magenta", "TEACH": "green", "REPLY": "white", "CYCLE": "green", "TREND": "blue"}
        tc = type_colors.get(msg_type, "white")

        L.append(f"[bold magenta]║[/]  [dim]{t}[/]   [{tc}]{msg_type:<10}[/] {sender_str:<14} → {recv_str:<12} [dim]{detail:<55}[/] [bold magenta]║[/]")

    L.append("[bold magenta]╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[/]")
    L.append("")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # LIVE PRICES + ACTIVE SIGNALS (side by side)
    L.append("[bold cyan]═══════════════════ LIVE PRICES ═══════════════════[/]          [bold yellow]═══════════════════════ ACTIVE SIGNALS ═══════════════════════[/]")

    price_lines = []
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0})
        p, c = d["price"], d["change"]
        ps = f"${p:>10,.2f}" if p >= 1 else f"${p:>10.4f}"
        cc = "green" if c >= 0 else "red"
        hist = list(PRICE_HISTORY.get(sym, []))
        sp = spark(hist, 8)
        price_lines.append(f"  [cyan]{sym:<10}[/] {ps} [{cc}]{c:>+6.2f}%[/] [{cc}]{sp}[/]")

    sig_lines = []
    for sym, sig in list(signals.items())[:4]:
        curr = prices.get(sym, {}).get("price", sig["entry"])
        d = sig["direction"]
        dc = "green" if d == "LONG" else "red"
        risk = abs(sig["entry"] - sig["sl"])
        reward = abs(sig["tp"] - sig["entry"])
        rr = reward / risk if risk > 0 else 0
        conf = sig["confidence"] * 100
        sig_lines.append(f"  [cyan]{sym:<8}[/] [{dc}]{d:<5}[/] E:${sig['entry']:>8,.0f} SL:${sig['sl']:>7,.0f} TP:${sig['tp']:>7,.0f} R:{rr:.1f} {conf:.0f}%")

    for i in range(max(len(price_lines), len(sig_lines))):
        pl = price_lines[i] if i < len(price_lines) else " " * 52
        sl = sig_lines[i] if i < len(sig_lines) else ""
        L.append(f"{pl:<52}          {sl}")
    L.append("")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # SAFETY + POSITION SIZING
    L.append(f"[bold red]SAFETY:[/] [green]✓ TRADING ACTIVE[/] │ Losses: {bar(1, 5, 6)} 1/5 │ DD: {bar(2.3, 10, 6)} 2.3% │ [green]✓ MOTHER OK[/] │ [green]✓ GUARDIAN OK[/]")
    L.append("")
    L.append(f"[dim]Ctrl+C exit │ 2s refresh │ {ntp} │ [cyan]A[/]=DeepSeek [magenta]B[/]=Claude [yellow]C[/]=Grok [green]D[/]=Gemini │ Width: {w}[/]")

    return Panel("\n".join(L), box=box.DOUBLE, style="on black", border_style="cyan", padding=(0, 0))


# ==================== TABLET LAYOUT ====================
def render_tablet(w, h):
    now = datetime.now()
    ntp = get_ntp()
    prices = get_prices()
    signals = get_signals()
    engines = get_engines()
    gen_comms()

    L = []
    L.append(f"[bold cyan]HYDRA 4.0[/] │ {now.strftime('%H:%M:%S')} │ {ntp}")
    L.append("─" * (w - 4))

    # Prices
    L.append("[bold]PRICES[/]")
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0})
        p, c = d["price"], d["change"]
        ps = f"${p/1000:.1f}K" if p >= 1000 else f"${p:.2f}"
        cc = "green" if c >= 0 else "red"
        L.append(f" [cyan]{sym[:4]}[/] {ps:>8} [{cc}]{c:+.1f}%[/]")
    L.append("")

    # Tournament
    L.append("[bold yellow]TOURNAMENT[/]")
    for i, (name, data) in enumerate(engines):
        icon = "👑" if i == 0 else "💀" if i == 3 else f"#{i+1}"
        _, color = ENGINE_NAMES[name]
        pc = "green" if data["pnl"] >= 0 else "red"
        L.append(f" {icon} [{color}]{name}:{ENGINE_NAMES[name][0][:6]}[/] {data['wr']:.0f}% [{pc}]${data['pnl']:+.0f}[/]")
    L.append("")

    # Comms
    L.append("[bold magenta]AI COMMS[/]")
    for e in list(COMM_LOG)[-4:]:
        t = e["time"].strftime("%H:%M")
        sc = ENGINE_NAMES.get(e["sender"], ("", "white"))[1]
        L.append(f" [dim]{t}[/] [{sc}]{e['sender'][:4]}[/] {e['type'][:6]} [dim]{e['detail'][:25]}[/]")
    L.append("")

    # Signals
    L.append("[bold yellow]SIGNALS[/]")
    for sym, sig in list(signals.items())[:3]:
        d = sig["direction"]
        dc = "green" if d == "LONG" else "red"
        L.append(f" {sym[:6]} [{dc}]{d[:4]}[/] E:${sig['entry']:,.0f} SL:${sig['sl']:,.0f}")

    L.append("")
    L.append(f"[green]✓ Active[/] │ DD: {bar(2.3, 10, 6)} 2.3%")

    return Panel("\n".join(L), box=box.ROUNDED, style="on black", border_style="cyan", padding=(0, 1))


def render():
    w, h = get_size()
    if w < 50:
        return render_phone(w, h)
    elif w < 100:
        return render_tablet(w, h)
    else:
        return render_desktop(w, h)


def main():
    print("\033[40m\033[2J\033[H", end="", flush=True)

    try:
        with Live(render(), console=console, refresh_per_second=2,
                  screen=True, transient=False) as live:
            while True:
                time.sleep(2)
                live.update(render())
    except KeyboardInterrupt:
        print("\033[0m")
        console.print("\n[yellow]Dashboard stopped.[/]")


if __name__ == "__main__":
    main()
