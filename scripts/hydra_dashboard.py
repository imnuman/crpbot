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
LAST_ENGINE_UPDATE = datetime.min
LAST_SIGNAL_UPDATE = datetime.min
ENGINE_CACHE: List = []
SIGNAL_CACHE: Dict = {}
SAFETY_CACHE: Dict = {}
LAST_SAFETY_UPDATE = datetime.min
PULSE_COUNTER = 0
FEAR_GREED_CACHE: Dict = {}
LAST_FG_UPDATE = datetime.min

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

# No demo data - using 100% real data only

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


def get_pulse() -> str:
    """Returns an animated pulse indicator to show live refresh."""
    global PULSE_COUNTER
    PULSE_COUNTER += 1
    pulses = ["[green]◉[/]", "[cyan]◎[/]", "[blue]◉[/]", "[cyan]◎[/]"]
    return pulses[PULSE_COUNTER % len(pulses)]


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
    """Fetch REAL prices from Coinbase API."""
    global PRICE_CACHE, LAST_PRICE_UPDATE, PRICE_HISTORY

    # Refresh every 2 seconds
    elapsed = (datetime.now() - LAST_PRICE_UPDATE).total_seconds()
    if elapsed < 2 and PRICE_CACHE:
        return PRICE_CACHE

    syms = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]

    try:
        from libs.data.coinbase_client import get_coinbase_client
        client = get_coinbase_client()

        for sym in syms:
            try:
                df = client.fetch_klines(sym, interval="1m", limit=30)
                if df is not None and len(df) > 0:
                    p = float(df.iloc[-1]["close"])
                    prev = float(df.iloc[-2]["close"]) if len(df) > 1 else p
                    ch = ((p - prev) / prev * 100) if prev else 0
                    PRICE_CACHE[sym] = {
                        "price": p,
                        "change": ch,
                        "high": float(df["high"].max()),
                        "low": float(df["low"].min()),
                        "updated": datetime.now(),
                        "source": "LIVE"
                    }
                    PRICE_HISTORY[sym] = deque(df["close"].tolist(), maxlen=30)
            except Exception as e:
                # Keep old price on error
                if sym not in PRICE_CACHE:
                    PRICE_CACHE[sym] = {"price": 0, "change": 0, "source": "ERROR", "updated": datetime.now()}
        LAST_PRICE_UPDATE = datetime.now()
    except Exception as e:
        pass  # Keep existing cache on connection error
    return PRICE_CACHE


def get_engines():
    """Fetch REAL engine rankings - forces fresh data each time."""
    global ENGINE_CACHE, LAST_ENGINE_UPDATE

    elapsed = (datetime.now() - LAST_ENGINE_UPDATE).total_seconds()
    if elapsed < 5 and ENGINE_CACHE:
        return ENGINE_CACHE

    try:
        # Reset the singleton to force fresh data load
        import libs.hydra.engine_portfolio as ep
        ep._tournament_manager = None  # Force recreation

        mgr = ep.get_tournament_manager()
        ranks = mgr.calculate_rankings()

        # Check if we have real data
        has_data = any(s.total_trades > 0 for _, s in ranks)

        ENGINE_CACHE = [(n, {
            "wr": s.win_rate * 100,
            "pnl": s.total_pnl_usd,
            "trades": s.total_trades,
            "source": "HYDRA" if has_data else "NO_DATA"
        }) for n, s in ranks]
        LAST_ENGINE_UPDATE = datetime.now()
        return ENGINE_CACHE
    except Exception as e:
        # Return error state
        ENGINE_CACHE = [
            ("A", {"wr": 0, "pnl": 0, "trades": 0, "source": "ERROR", "error": str(e)[:30]}),
            ("B", {"wr": 0, "pnl": 0, "trades": 0, "source": "ERROR"}),
            ("C", {"wr": 0, "pnl": 0, "trades": 0, "source": "ERROR"}),
            ("D", {"wr": 0, "pnl": 0, "trades": 0, "source": "ERROR"}),
        ]
        LAST_ENGINE_UPDATE = datetime.now()
        return ENGINE_CACHE


def get_signals():
    """Fetch REAL signals from database only."""
    global SIGNAL_CACHE, LAST_SIGNAL_UPDATE

    elapsed = (datetime.now() - LAST_SIGNAL_UPDATE).total_seconds()
    if elapsed < 2 and SIGNAL_CACHE:
        return SIGNAL_CACHE

    try:
        from sqlalchemy import create_engine, text
        from pathlib import Path

        # Try local database first
        db_path = Path(__file__).parent.parent / "tradingai.db"
        if db_path.exists():
            engine = create_engine(f"sqlite:///{db_path}")
            with engine.connect() as conn:
                # Use raw SQL to avoid schema mismatches
                result = conn.execute(text("""
                    SELECT symbol, direction, confidence, entry_price, sl_price, tp_price,
                           timestamp, notes
                    FROM signals
                    WHERE timestamp > datetime('now', '-4 hours')
                      AND result IS NULL
                    ORDER BY timestamp DESC
                    LIMIT 5
                """))
                sigs = {}
                for row in result:
                    sigs[row[0]] = {
                        "direction": row[1].upper() if row[1] else "HOLD",
                        "confidence": row[2] or 0,
                        "engine": "V7",
                        "entry": row[3] or 0,
                        "sl": row[4] or 0,
                        "tp": row[5] or 0,
                        "timestamp": datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
                        "reason": row[7] or "Signal from V7",
                        "source": "DATABASE"
                    }
                if sigs:
                    SIGNAL_CACHE = sigs
                    LAST_SIGNAL_UPDATE = datetime.now()
                    return SIGNAL_CACHE
    except Exception as e:
        pass

    # No signals found - show empty state
    if not SIGNAL_CACHE:
        SIGNAL_CACHE = {"NO_SIGNALS": {
            "direction": "WAIT",
            "confidence": 0,
            "engine": "-",
            "entry": 0, "sl": 0, "tp": 0,
            "timestamp": datetime.now(),
            "reason": "No active signals",
            "source": "EMPTY"
        }}
    LAST_SIGNAL_UPDATE = datetime.now()
    return SIGNAL_CACHE


def get_safety():
    """Fetch REAL safety status from Guardian/Mother AI."""
    global SAFETY_CACHE, LAST_SAFETY_UPDATE

    elapsed = (datetime.now() - LAST_SAFETY_UPDATE).total_seconds()
    if elapsed < 2 and SAFETY_CACHE:
        return SAFETY_CACHE

    safety = {
        "trading_active": None,
        "consecutive_losses": 0,
        "drawdown": 0.0,
        "mother_ok": None,
        "guardian_ok": None,
        "emergency": False,
        "source": "UNKNOWN"
    }

    # Try to fetch from Guardian
    try:
        from libs.hydra.guardian import get_guardian
        guardian = get_guardian()
        status = guardian.get_status()
        safety["trading_active"] = status.get("trading_allowed", True)
        safety["consecutive_losses"] = status.get("consecutive_losses", 0)
        safety["drawdown"] = status.get("current_drawdown_percent", 0)
        safety["emergency"] = status.get("emergency_shutdown_active", False)
        safety["guardian_ok"] = not safety["emergency"]
        safety["source"] = "GUARDIAN"
    except Exception as e:
        safety["guardian_ok"] = None
        safety["guardian_error"] = str(e)[:30]

    # Try to fetch from Mother AI
    try:
        from libs.hydra.mother_ai import get_mother_ai
        mother = get_mother_ai()
        health = mother.get_health_status()
        safety["mother_ok"] = health.get("is_healthy", True) and not health.get("is_frozen", False)
    except Exception as e:
        safety["mother_ok"] = None
        safety["mother_error"] = str(e)[:30]

    SAFETY_CACHE = safety
    LAST_SAFETY_UPDATE = datetime.now()
    return SAFETY_CACHE


def get_fear_greed():
    """Fetch Fear & Greed Index."""
    global FEAR_GREED_CACHE, LAST_FG_UPDATE

    # Only refresh every 60 seconds (it doesn't change often)
    elapsed = (datetime.now() - LAST_FG_UPDATE).total_seconds()
    if elapsed < 60 and FEAR_GREED_CACHE:
        return FEAR_GREED_CACHE

    try:
        from libs.data.fear_greed_client import FearGreedClient
        fg = FearGreedClient()
        data = fg.get_current_index()
        if data:
            FEAR_GREED_CACHE = {
                "value": data.get("value", 0),
                "classification": data.get("classification", "Unknown"),
                "signal": data.get("signal", "hold"),
                "source": "ALTERNATIVE.ME"
            }
            LAST_FG_UPDATE = datetime.now()
            return FEAR_GREED_CACHE
    except Exception as e:
        pass

    # Return cached or empty
    if not FEAR_GREED_CACHE:
        FEAR_GREED_CACHE = {"value": 0, "classification": "N/A", "signal": "unknown", "source": "ERROR"}
    return FEAR_GREED_CACHE


def get_real_comms():
    """Try to fetch real communication logs from HYDRA system."""
    try:
        from libs.hydra.mother_ai import get_mother_ai
        mother = get_mother_ai()
        if hasattr(mother, 'get_recent_logs'):
            return mother.get_recent_logs(limit=20)
    except:
        pass

    try:
        # Try to read from log files
        from pathlib import Path
        import re
        log_files = list(Path("/tmp").glob("hydra_*.log"))
        if log_files:
            log_file = sorted(log_files, key=lambda x: x.stat().st_mtime)[-1]
            with open(log_file) as f:
                lines = f.readlines()[-50:]
                for line in lines:
                    # Parse log lines
                    if "Engine" in line or "MOTHER" in line or "Guardian" in line:
                        log_comm("SYSTEM", "LOG", "INFO", line[:40], line[40:100])
    except:
        pass
    return None


def gen_comms():
    """Fetch real communications or show system status messages."""

    # Try to get real comms first
    real_comms = get_real_comms()
    if real_comms:
        for comm in real_comms:
            log_comm(comm.get("sender", "SYS"), comm.get("receiver", "ALL"),
                     comm.get("type", "LOG"), comm.get("message", "")[:40],
                     comm.get("detail", "")[:60])
        return

    # Show real system status messages based on actual data
    prices = PRICE_CACHE
    safety = SAFETY_CACHE

    if len(COMM_LOG) < 5:
        # Initialize with system status
        log_comm("SYSTEM", "ALL", "INIT", "Dashboard started", f"Monitoring {len(prices)} symbols")

    # Add real data status messages
    if prices:
        btc = prices.get("BTC-USD", {})
        if btc.get("source") == "COINBASE":
            log_comm("DATA", "DASH", "PRICE", f"BTC: ${btc.get('price', 0):,.0f}",
                     f"Change: {btc.get('change', 0):+.2f}%")

    if safety.get("source") == "GUARDIAN":
        log_comm("GUARD", "DASH", "STATUS",
                 f"DD: {safety.get('drawdown', 0):.1f}%",
                 f"Losses: {safety.get('consecutive_losses', 0)}")


# ==================== PHONE LAYOUT ====================
def render_phone(w, h):
    now = datetime.now()
    ntp = get_ntp()
    prices = get_prices()
    signals = get_signals()
    engines = get_engines()
    gen_comms()

    L = []
    pulse = get_pulse()
    L.append(f"[bold cyan]HYDRA 4.0[/] {pulse} {ntp}")
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
    pulse = get_pulse()
    # Show data freshness
    price_age = (now - LAST_PRICE_UPDATE).total_seconds() if LAST_PRICE_UPDATE != datetime.min else 999
    price_status = "[green]●[/]" if price_age < 5 else "[yellow]○[/]" if price_age < 30 else "[red]✗[/]"
    L.append(f"  [dim]v4.0 │ {now.strftime('%Y-%m-%d %H:%M:%S')} │ {pulse} LIVE │ {ntp} │ Prices:{price_status} │ Uptime: {hrs}h {mins}m {secs}s[/]")
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
        src = d.get("source", "?")
        ps = f"${p:>10,.2f}" if p >= 1 else f"${p:>10.4f}"
        cc = "green" if c >= 0 else "red"
        hist = list(PRICE_HISTORY.get(sym, []))
        sp = spark(hist, 8)
        # Show source indicator
        src_ind = "[green]●[/]" if src == "LIVE" else "[yellow]○[/]" if src == "STALE" else "[red]✗[/]"
        price_lines.append(f"  {src_ind}[cyan]{sym:<9}[/] {ps} [{cc}]{c:>+6.2f}%[/] [{cc}]{sp}[/]")

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
    # SAFETY + POSITION SIZING (live data)
    safety = get_safety()
    losses = safety["consecutive_losses"]
    dd = safety["drawdown"]

    # Handle None states
    if safety["trading_active"] is None:
        trading_status = "[yellow]? UNKNOWN[/]"
    elif safety["trading_active"]:
        trading_status = "[green]✓ TRADING[/]"
    else:
        trading_status = "[red]⚠ PAUSED[/]"

    if safety["mother_ok"] is None:
        mother_status = "[yellow]? MOTHER[/]"
    elif safety["mother_ok"]:
        mother_status = "[green]✓ MOTHER[/]"
    else:
        mother_status = "[red]⚠ MOTHER[/]"

    if safety["guardian_ok"] is None:
        guardian_status = "[yellow]? GUARD[/]"
    elif safety["guardian_ok"]:
        guardian_status = "[green]✓ GUARD[/]"
    else:
        guardian_status = "[red]⚠ GUARD[/]"

    # Fear & Greed Index
    fg = get_fear_greed()
    fg_val = fg.get("value", 0)
    fg_class = fg.get("classification", "?")
    fg_signal = fg.get("signal", "?")
    if fg_val <= 25:
        fg_color = "red"
    elif fg_val <= 45:
        fg_color = "yellow"
    elif fg_val <= 55:
        fg_color = "white"
    elif fg_val <= 75:
        fg_color = "green"
    else:
        fg_color = "bold green"

    src = safety.get("source", "?")
    L.append(f"[bold red]SAFETY:[/] {trading_status} │ Losses: {bar(losses, 5, 6)} {losses}/5 │ DD: {bar(dd, 10, 6)} {dd:.1f}% │ {mother_status} │ {guardian_status} │ [{fg_color}]F&G:{fg_val} {fg_class}[/] │ [dim]src:{src}[/]")
    L.append("")

    # Last update timestamps
    price_age = (now - LAST_PRICE_UPDATE).total_seconds() if LAST_PRICE_UPDATE != datetime.min else 999
    engine_age = (now - LAST_ENGINE_UPDATE).total_seconds() if LAST_ENGINE_UPDATE != datetime.min else 999
    price_ind = "[green]●[/]" if price_age < 5 else "[yellow]●[/]" if price_age < 10 else "[red]●[/]"
    engine_ind = "[green]●[/]" if engine_age < 5 else "[yellow]●[/]" if engine_age < 10 else "[red]●[/]"

    L.append(f"[dim]Ctrl+C exit │ 2s refresh │ {ntp} │ Prices {price_ind}{price_age:.0f}s │ Engines {engine_ind}{engine_age:.0f}s │ [cyan]A[/]=DeepSeek [magenta]B[/]=Claude [yellow]C[/]=Grok [green]D[/]=Gemini │ w:{w}[/]")

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
    pulse = get_pulse()
    L.append(f"[bold cyan]HYDRA 4.0[/] {pulse} │ {now.strftime('%H:%M:%S')} │ {ntp}")
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
    safety = get_safety()
    status = "[green]✓ Active[/]" if safety["trading_active"] else "[red]⚠ Paused[/]"
    L.append(f"{status} │ DD: {bar(safety['drawdown'], 10, 6)} {safety['drawdown']:.1f}%")

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
        # Refresh 4 times per second for smooth updates
        with Live(render(), console=console, refresh_per_second=4,
                  screen=True, transient=False) as live:
            while True:
                # Short sleep, let Rich handle actual refresh rate
                time.sleep(0.5)
                live.update(render())
    except KeyboardInterrupt:
        print("\033[0m")
        console.print("\n[yellow]Dashboard stopped.[/]")


if __name__ == "__main__":
    main()
