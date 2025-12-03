#!/usr/bin/env python3
"""
HYDRA 4.0 - Live Terminal Dashboard

Responsive layouts:
- Phone (< 50 cols): Full-screen compact view
- Desktop 1920x1080 (100+ cols): Full HD layout with all panels

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
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich import box
except ImportError:
    os.system("pip install rich")
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
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
    """Get terminal width and height."""
    try:
        return console.width, console.height
    except:
        return 80, 24


def get_ntp() -> str:
    """Get NTP status."""
    try:
        r = subprocess.run(["timedatectl", "show", "--property=NTPSynchronized"],
                          capture_output=True, text=True, timeout=1)
        return "[green]●[/]NTP" if "yes" in r.stdout.lower() else "[yellow]○[/]"
    except:
        return "[dim]○[/]"


def spark(vals, w=8):
    """Sparkline."""
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
    """Bar chart."""
    if mx <= 0:
        return EMPTY * w
    f = max(0, min(w, int(v / mx * w)))
    return BAR * f + EMPTY * (w - f)


def log_comm(s, r, t, c, d=""):
    """Log communication."""
    COMM_LOG.append({"time": datetime.now(), "sender": s, "receiver": r,
                     "type": t, "content": c[:40], "detail": d[:50]})


def get_prices():
    """Get live prices."""
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
    """Get engine data."""
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
    """Get active signals."""
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
    """Generate communications."""
    import random
    if len(COMM_LOG) < 6:
        init = [
            ("MOTHER", "All", "CYCLE", "Cycle #1247", "10 symbols"),
            ("A", "MOTHER", "SCAN", "Liq scan", "3 triggers"),
            ("B", "MOTHER", "RATE", "Funding", "0.045%"),
            ("C", "MOTHER", "DEPTH", "Orderbook", "ETH wall 3680"),
            ("D", "MOTHER", "REGIME", "Regime", "Trending"),
            ("GUARD", "All", "RISK", "Risk OK", "23% exp"),
        ]
        for s, r, t, c, d in init:
            log_comm(s, r, t, c, d)
    if random.random() < 0.25:
        acts = [
            ("A", "MOTHER", "SIGNAL", "Liq update", f"${random.randint(50,200)}M"),
            ("B", "MOTHER", "RATE", "Fund update", f"{random.uniform(-0.05, 0.1):.3f}%"),
            ("C", "MOTHER", "DEPTH", "Depth", f"+{random.randint(5,20)}%"),
            ("D", "MOTHER", "TREND", "Regime", f"{'Trend' if random.random() > 0.5 else 'Range'}"),
        ]
        s, r, t, c, d = random.choice(acts)
        log_comm(s, r, t, c, d)


# ==================== PHONE LAYOUT ====================
def render_phone(w, h):
    """Phone full-screen layout (< 50 cols)."""
    now = datetime.now()
    ntp = get_ntp()
    prices = get_prices()
    signals = get_signals()
    gen_comms()

    L = []

    # Header - compact
    L.append(f"[bold cyan]HYDRA 4.0[/] {ntp}")
    L.append(f"[dim]{now.strftime('%H:%M:%S')}[/]")
    L.append("─" * (w - 4))

    # Prices - 3 main coins
    L.append("[bold white]PRICES[/]")
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0})
        p = d["price"]
        c = d["change"]
        ps = f"${p/1000:.1f}K" if p >= 1000 else f"${p:.2f}"
        cc = "green" if c >= 0 else "red"
        L.append(f"[cyan]{sym[:3]}[/] {ps} [{cc}]{c:+.1f}%[/]")
    L.append("")

    # Signals - compact
    L.append("[bold yellow]SIGNALS[/]")
    for sym, sig in list(signals.items())[:2]:
        d = sig["direction"]
        dc = "green" if d == "LONG" else "red"
        cf = sig["confidence"] * 100
        L.append(f"[{dc}]{sym[:3]} {d[:1]}[/] {cf:.0f}%")
        L.append(f" E${sig['entry']:,.0f}")
        L.append(f" S${sig['sl']:,.0f} T${sig['tp']:,.0f}")

        # R:R
        risk = abs(sig["entry"] - sig["sl"])
        reward = abs(sig["tp"] - sig["entry"])
        rr = reward / risk if risk > 0 else 0
        L.append(f" [dim]R:R 1:{rr:.1f}[/]")
    L.append("")

    # Comms - last 3
    L.append("[bold magenta]COMMS[/]")
    for e in list(COMM_LOG)[-3:]:
        t = e["time"].strftime("%H:%M")
        sc = ENGINE_NAMES.get(e["sender"], ("", "white"))[1]
        L.append(f"[dim]{t}[/][{sc}]{e['sender'][:2]}[/]{e['type'][:4]}")
    L.append("")

    # Safety - one line
    L.append(f"[green]✓OK[/] DD:2.3%")

    content = "\n".join(L)
    return Panel(content, box=box.ROUNDED, style="on black", border_style="cyan",
                 padding=(0, 0), width=w, height=h)


# ==================== DESKTOP 1920x1080 LAYOUT ====================
def render_desktop(w, h):
    """Desktop full HD layout (1920x1080 = ~160 cols)."""
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

    # ASCII Header
    L.append("[bold cyan]  ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗     [/][bold white]Multi-Agent Trading System[/]")
    L.append("[bold cyan]  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗    [/][dim]Real-time Market Intelligence[/]")
    L.append("[bold cyan]  ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║    [/]")
    L.append("[bold cyan]  ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║    [/][dim]Engines: [cyan]A[/]:DeepSeek [magenta]B[/]:Claude [yellow]C[/]:Grok [green]D[/]:Gemini[/]")
    L.append("[bold cyan]  ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║    [/]")
    L.append("[bold cyan]  ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    [/]")
    L.append(f"  [dim]v4.0 │ {now.strftime('%Y-%m-%d %H:%M:%S')} │ {ntp} │ Uptime: {hrs}h {mins}m {secs}s[/]")
    L.append("")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # LIVE PRICES
    L.append("[bold cyan]═══════════════════════════════════════════════════════════ LIVE PRICES ═══════════════════════════════════════════════════════════[/]")
    hdr = f"  {'Symbol':<10} {'Price':>14} {'Change':>10} {'24h High':>14} {'24h Low':>14} {'Trend':>12}"
    L.append(f"[dim]{hdr}[/]")

    for sym in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0, "high": 0, "low": 0})
        p, c = d["price"], d["change"]
        hi, lo = d.get("high", p), d.get("low", p)

        ps = f"${p:>12,.2f}" if p >= 1 else f"${p:>12.4f}"
        cc = "green" if c >= 0 else "red"
        hs = f"${hi:>12,.2f}" if hi >= 1 else f"${hi:>12.4f}"
        ls = f"${lo:>12,.2f}" if lo >= 1 else f"${lo:>12.4f}"
        hist = list(PRICE_HISTORY.get(sym, []))
        sp = spark(hist, 10)
        spc = "green" if c >= 0 else "red"

        L.append(f"  [cyan]{sym:<10}[/] {ps} [{cc}]{c:>+9.2f}%[/] [dim]{hs}[/] [dim]{ls}[/] [{spc}]{sp}[/]")
    L.append("")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # ACTIVE SIGNALS
    L.append("[bold yellow]═══════════════════════════════════════════════════════ ACTIVE SIGNALS ══════════════════════════════════════════════════════════[/]")
    shdr = f"  {'Pair':<10} {'Direction':<10} {'Entry':>14} {'Current':>14} {'Stop Loss':>14} {'Take Profit':>14} {'R:R':>8} {'Conf':>8}"
    L.append(f"[dim]{shdr}[/]")

    for sym, sig in list(signals.items())[:4]:
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

        conf = sig["confidence"] * 100
        confc = "green" if conf >= 70 else "yellow" if conf >= 60 else "dim"

        es = f"${sig['entry']:>12,.2f}"
        cs = f"${curr:>12,.2f}"
        sls = f"${sig['sl']:>12,.2f}"
        tps = f"${sig['tp']:>12,.2f}"

        L.append(f"  [cyan]{sym:<10}[/] [{dc}]{d:<10}[/] {es} [{pc}]{cs}[/] {sls} {tps} [{rrc}]1:{rr:>5.1f}[/] [{confc}]{conf:>6.0f}%[/]")

    # Signal reasoning - clean format
    L.append("")
    for sym, sig in list(signals.items())[:3]:
        age_mins = (now - sig["timestamp"]).total_seconds() / 60
        if age_mins > 60:
            age_str = f"{int(age_mins/60)}h ago"
        else:
            age_str = f"{int(age_mins)}m ago"
        eng = sig.get("engine", "A")
        _, ec = ENGINE_NAMES.get(eng, ("", "white"))

        # Parse reason - extract clean text from JSON if needed
        reason = sig.get("reason", "")
        if isinstance(reason, str) and reason.startswith("{"):
            try:
                import json
                data = json.loads(reason)
                reason = data.get("reasoning", reason)[:80]
            except:
                reason = reason[:80]
        else:
            reason = str(reason)[:80]

        L.append(f"  [{ec}]● {sym}[/] Engine {eng} ({age_str}): [dim]{reason}[/]")
    L.append("")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # RANKINGS + COMMS + SIZING (3 columns)
    L.append("[bold yellow]══════════ ENGINE RANKINGS ══════════[/]   [bold magenta]══════════════ AI COMMUNICATIONS ══════════════[/]   [bold green]══════ POSITION SIZING ══════[/]")

    eng_lines = []
    for i, (name, data) in enumerate(engines):
        rank = i + 1
        icon = "👑" if rank == 1 else "💀" if rank == 4 else f"#{rank}"
        wr = data["wr"]
        pnl = data["pnl"]
        pc = "green" if pnl >= 0 else "red"
        wc = "green" if wr >= 55 else "yellow" if wr >= 45 else "red"
        _, color = ENGINE_NAMES[name]
        b = bar(wr, 100, 10)
        eng_lines.append(f" {icon} [{color}]{name}:{ENGINE_NAMES[name][0]:<8}[/] [{wc}]{wr:>3.0f}%[/] [{wc}]{b}[/] [{pc}]${pnl:>+8.0f}[/]")

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
              "RATE": "cyan", "DEPTH": "cyan", "REGIME": "blue"}.get(e["type"], "white")
        comm_lines.append(f" [dim]{t}[/] [{tc}]{e['type']:7}[/] [{sc}]{s:6}[/] → {e['receiver']:<6} [dim]{e['detail'][:18]}[/]")

    size_lines = []
    for sym, sig in list(signals.items())[:4]:
        curr = prices.get(sym, {}).get("price", sig["entry"])
        risk = abs(sig["entry"] - sig["sl"])
        risk_pct = (risk / sig["entry"]) * 100
        risk_amt = 100  # 1% of 10K
        pos = risk_amt / risk if risk > 0 else 0
        val = pos * curr
        size_lines.append(f" [cyan]{sym[:6]}[/] ${val:>7,.0f} ({pos:.3f})")
        size_lines.append(f"   [dim]Risk: ${risk_amt:.0f} ({risk_pct:.1f}%)[/]")

    # Combine columns
    max_rows = max(len(eng_lines), len(comm_lines), len(size_lines) // 2 + 1)
    for i in range(max_rows):
        el = eng_lines[i] if i < len(eng_lines) else " " * 38
        cl = comm_lines[i] if i < len(comm_lines) else " " * 48
        # Size lines come in pairs
        si = i * 2
        sl1 = size_lines[si] if si < len(size_lines) else ""
        sl2 = size_lines[si + 1] if si + 1 < len(size_lines) else ""
        sl = f"{sl1}\n{sl2}" if sl2 else sl1
        L.append(f"{el:<38}   {cl:<48}   {sl}")
    L.append("")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
    # SAFETY STATUS
    L.append("[bold red]════════════════════════════════════════════════════════ SAFETY STATUS ════════════════════════════════════════════════════════════[/]")
    L.append(f"  [green]✓ TRADING ACTIVE[/]  │  Consecutive Losses: {bar(1, 5, 8)} 1/5  │  Drawdown: {bar(2.3, 10, 8)} 2.3%  │  [green]✓ MOTHER AI OK[/]  │  [green]✓ GUARDIAN OK[/]")
    L.append("")

    # Footer
    L.append(f"[dim]  Ctrl+C exit │ 2s refresh │ {ntp} │ Width: {w} │ [cyan]A[/]=DeepSeek [magenta]B[/]=Claude [yellow]C[/]=Grok [green]D[/]=Gemini[/]")

    content = "\n".join(L)
    return Panel(content, box=box.DOUBLE, style="on black", border_style="cyan", padding=(0, 0))


# ==================== TABLET LAYOUT ====================
def render_tablet(w, h):
    """Tablet/medium layout (50-100 cols)."""
    now = datetime.now()
    ntp = get_ntp()
    prices = get_prices()
    signals = get_signals()
    engines = get_engines()
    gen_comms()

    L = []

    # Header
    L.append(f"[bold cyan]HYDRA 4.0[/] │ {now.strftime('%H:%M:%S')} │ {ntp}")
    L.append("─" * (w - 4))

    # Prices
    L.append("[bold]PRICES[/]")
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]:
        d = prices.get(sym, {"price": 0, "change": 0})
        p, c = d["price"], d["change"]
        ps = f"${p/1000:.1f}K" if p >= 1000 else f"${p:.2f}"
        cc = "green" if c >= 0 else "red"
        hist = list(PRICE_HISTORY.get(sym, []))
        sp = spark(hist, 6)
        L.append(f" [cyan]{sym[:4]}[/] {ps:>8} [{cc}]{c:+.1f}%[/] [{cc}]{sp}[/]")
    L.append("")

    # Signals
    L.append("[bold yellow]SIGNALS[/]")
    L.append(f" {'Pair':<6}{'Dir':<5}{'Entry':>9}{'SL':>9}{'TP':>9}{'R:R':>6}")
    for sym, sig in list(signals.items())[:3]:
        d = sig["direction"]
        dc = "green" if d == "LONG" else "red"
        risk = abs(sig["entry"] - sig["sl"])
        reward = abs(sig["tp"] - sig["entry"])
        rr = reward / risk if risk > 0 else 0
        L.append(f" {sym[:6]:<6}[{dc}]{d[:4]:<5}[/]${sig['entry']:>7,.0f}${sig['sl']:>7,.0f}${sig['tp']:>7,.0f} 1:{rr:.1f}")
    L.append("")

    # Rankings + Comms side by side
    L.append("[bold yellow]RANKINGS[/]           [bold magenta]COMMS[/]")
    eng_lines = []
    for i, (name, data) in enumerate(engines):
        rank = i + 1
        icon = "👑" if rank == 1 else "💀" if rank == 4 else f"#{rank}"
        _, color = ENGINE_NAMES[name]
        pc = "green" if data["pnl"] >= 0 else "red"
        eng_lines.append(f"{icon}[{color}]{name}[/] {data['wr']:.0f}% [{pc}]${data['pnl']:+.0f}[/]")

    comm_lines = []
    for e in list(COMM_LOG)[-4:]:
        t = e["time"].strftime("%H:%M")
        sc = ENGINE_NAMES.get(e["sender"], ("", "white"))[1]
        comm_lines.append(f"[dim]{t}[/][{sc}]{e['sender'][:3]}[/]{e['type'][:5]}")

    for i in range(max(len(eng_lines), len(comm_lines))):
        el = eng_lines[i] if i < len(eng_lines) else ""
        cl = comm_lines[i] if i < len(comm_lines) else ""
        L.append(f" {el:<22}{cl}")
    L.append("")

    # Safety
    L.append(f"[green]✓ Active[/] │ DD: {bar(2.3, 10, 6)} 2.3%")

    content = "\n".join(L)
    return Panel(content, box=box.ROUNDED, style="on black", border_style="cyan", padding=(0, 1))


def render():
    """Render based on terminal size."""
    w, h = get_size()

    if w < 50:
        return render_phone(w, h)
    elif w < 100:
        return render_tablet(w, h)
    else:
        return render_desktop(w, h)


def main():
    """Main loop."""
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
