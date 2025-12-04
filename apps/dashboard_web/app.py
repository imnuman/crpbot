"""
HYDRA 4.0 Web Dashboard
Flask-based web dashboard for monitoring HYDRA trading system
"""
import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__,
    template_folder=str(PROJECT_ROOT / "apps" / "dashboard_web" / "templates"),
    static_folder=str(PROJECT_ROOT / "apps" / "dashboard_web" / "static")
)

# Database path
DB_PATH = PROJECT_ROOT / "tradingai.db"


def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_prices():
    """Fetch live prices from Coinbase."""
    prices = {}
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]

    try:
        from libs.data.coinbase_client import get_coinbase_client
        client = get_coinbase_client()

        for sym in symbols:
            try:
                df = client.fetch_klines(sym, interval="1m", limit=30)
                if df is not None and len(df) > 0:
                    price = float(df.iloc[-1]["close"])
                    prev = float(df.iloc[-2]["close"]) if len(df) > 1 else price
                    change = ((price - prev) / prev * 100) if prev else 0
                    prices[sym] = {
                        "price": price,
                        "change": round(change, 2),
                        "high_24h": float(df["high"].max()),
                        "low_24h": float(df["low"].min())
                    }
            except Exception:
                prices[sym] = {"price": 0, "change": 0, "error": True}
    except Exception as e:
        for sym in symbols:
            prices[sym] = {"price": 0, "change": 0, "error": True}

    return prices


def get_tournament_data():
    """Get tournament leaderboard data."""
    leaderboard = []

    try:
        data_dir = PROJECT_ROOT / "data" / "hydra"
        leaderboard_file = data_dir / "tournament_leaderboard.json"

        if leaderboard_file.exists():
            with open(leaderboard_file) as f:
                data = json.load(f)
                leaderboard = data.get("rankings", [])
    except Exception:
        pass

    # Default if no data
    if not leaderboard:
        leaderboard = [
            {"gladiator": "A", "name": "DeepSeek", "elo": 1500, "wins": 0, "losses": 0, "win_rate": 50.0},
            {"gladiator": "B", "name": "Claude", "elo": 1500, "wins": 0, "losses": 0, "win_rate": 50.0},
            {"gladiator": "C", "name": "Grok", "elo": 1500, "wins": 0, "losses": 0, "win_rate": 50.0},
            {"gladiator": "D", "name": "Gemini", "elo": 1500, "wins": 0, "losses": 0, "win_rate": 50.0},
        ]

    return leaderboard


def get_recent_signals(limit=20):
    """Get recent signals from database."""
    signals = []

    try:
        conn = get_db_connection()
        cursor = conn.execute("""
            SELECT timestamp, symbol, direction, confidence, entry_price,
                   stop_loss, take_profit, signal_variant
            FROM signals
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        for row in cursor:
            signals.append({
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "direction": row["direction"],
                "confidence": row["confidence"],
                "entry_price": row["entry_price"],
                "stop_loss": row["stop_loss"],
                "take_profit": row["take_profit"],
                "variant": row["signal_variant"]
            })
        conn.close()
    except Exception:
        pass

    return signals


def get_paper_trades():
    """Get paper trading results."""
    trades = []

    try:
        conn = get_db_connection()
        cursor = conn.execute("""
            SELECT s.timestamp, s.symbol, s.direction, s.confidence,
                   sr.outcome, sr.pnl_percent, sr.exit_price, sr.exit_timestamp
            FROM signals s
            LEFT JOIN signal_results sr ON s.id = sr.signal_id
            WHERE sr.outcome IS NOT NULL
            ORDER BY s.timestamp DESC
            LIMIT 50
        """)

        for row in cursor:
            trades.append({
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "direction": row["direction"],
                "confidence": row["confidence"],
                "outcome": row["outcome"],
                "pnl_percent": row["pnl_percent"],
                "exit_timestamp": row["exit_timestamp"]
            })
        conn.close()
    except Exception:
        pass

    return trades


def get_system_stats():
    """Get system statistics."""
    stats = {
        "total_signals": 0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0,
        "total_pnl": 0,
        "signals_today": 0
    }

    try:
        conn = get_db_connection()

        # Total signals
        cursor = conn.execute("SELECT COUNT(*) FROM signals")
        stats["total_signals"] = cursor.fetchone()[0]

        # Signals today
        cursor = conn.execute("""
            SELECT COUNT(*) FROM signals
            WHERE timestamp > datetime('now', '-24 hours')
        """)
        stats["signals_today"] = cursor.fetchone()[0]

        # Paper trade results
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl_percent), 0) as total_pnl
            FROM signal_results
            WHERE outcome IS NOT NULL
        """)
        row = cursor.fetchone()
        stats["total_trades"] = row[0] or 0
        stats["wins"] = row[1] or 0
        stats["losses"] = row[2] or 0
        stats["total_pnl"] = round(row[3] or 0, 2)

        if stats["total_trades"] > 0:
            stats["win_rate"] = round(stats["wins"] / stats["total_trades"] * 100, 1)

        conn.close()
    except Exception:
        pass

    return stats


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/prices')
def api_prices():
    """API endpoint for live prices."""
    return jsonify(get_prices())


@app.route('/api/tournament')
def api_tournament():
    """API endpoint for tournament data."""
    return jsonify(get_tournament_data())


@app.route('/api/signals')
def api_signals():
    """API endpoint for recent signals."""
    return jsonify(get_recent_signals())


@app.route('/api/trades')
def api_trades():
    """API endpoint for paper trades."""
    return jsonify(get_paper_trades())


@app.route('/api/stats')
def api_stats():
    """API endpoint for system stats."""
    return jsonify(get_system_stats())


@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0"
    })


if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    templates_dir = PROJECT_ROOT / "apps" / "dashboard_web" / "templates"
    static_dir = PROJECT_ROOT / "apps" / "dashboard_web" / "static"
    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    app.run(host='0.0.0.0', port=8080, debug=False)
