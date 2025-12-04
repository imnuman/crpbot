#!/usr/bin/env python3
"""
HYDRA Terminal Dashboard Web Server

Serves the terminal dashboard as HTML/JSON for web viewing.
Lightweight Flask server that converts terminal output to web format.
"""

from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

from libs.hydra.config import HYDRA_DB_FILE
DB_PATH = str(HYDRA_DB_FILE)

def get_engine_stats() -> Dict[str, Any]:
    """Get performance stats for all engines."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    stats = {}
    for engine in ["A", "B", "C", "D"]:
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl_usd) as total_pnl
            FROM hydra_trades
            WHERE gladiator = ? AND exit_time IS NOT NULL
        """, (engine,))

        result = cursor.fetchone()
        total = result[0] or 0
        wins = result[1] or 0
        win_rate = (wins / total * 100) if total > 0 else 0.0

        stats[engine] = {
            "total_trades": total,
            "wins": wins,
            "losses": result[2] or 0,
            "win_rate": win_rate,
            "total_pnl": result[3] or 0.0
        }

    conn.close()
    return stats

def get_recent_trades(limit: int = 20) -> List[Dict]:
    """Get recent trades across all engines."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            trade_id,
            timestamp,
            symbol,
            gladiator,
            direction,
            entry_price,
            exit_price,
            pnl_usd,
            exit_reason,
            exit_time
        FROM hydra_trades
        WHERE exit_time IS NOT NULL
        ORDER BY exit_time DESC
        LIMIT ?
    """, (limit,))

    trades = []
    for row in cursor.fetchall():
        trades.append({
            "trade_id": row[0],
            "timestamp": row[1],
            "symbol": row[2],
            "engine": row[3],
            "direction": row[4],
            "entry_price": row[5],
            "exit_price": row[6],
            "pnl": row[7],
            "exit_reason": row[8],
            "exit_time": row[9]
        })

    conn.close()
    return trades

def get_system_health() -> Dict[str, Any]:
    """Get system health metrics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total strategies generated
    cursor.execute("SELECT COUNT(*) FROM strategies")
    total_strategies = cursor.fetchone()[0]

    # Active trades
    cursor.execute("SELECT COUNT(*) FROM hydra_trades WHERE exit_time IS NULL")
    active_trades = cursor.fetchone()[0]

    # Trades last 24h
    cursor.execute("""
        SELECT COUNT(*) FROM hydra_trades
        WHERE timestamp > datetime('now', '-24 hours')
    """)
    trades_24h = cursor.fetchone()[0]

    conn.close()

    return {
        "total_strategies": total_strategies,
        "active_trades": active_trades,
        "trades_24h": trades_24h,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.route("/")
def index():
    """Main dashboard page."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>HYDRA 3.0 Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #00ff00;
            border-bottom: 2px solid #00ff00;
            padding-bottom: 10px;
            margin-bottom: 20px;
            text-shadow: 0 0 10px #00ff00;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: #1a1a1a;
            border: 1px solid #00ff00;
            border-radius: 5px;
            padding: 15px;
        }
        .panel h2 {
            color: #00ffff;
            margin-bottom: 15px;
            font-size: 18px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            background: #2a2a2a;
            color: #00ffff;
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #00ff00;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #333;
        }
        .positive { color: #00ff00; }
        .negative { color: #ff0000; }
        .rank { color: #ffff00; font-weight: bold; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .stat-box {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #00ff00;
        }
        .stat-label { font-size: 12px; color: #888; }
        .stat-value { font-size: 24px; color: #00ff00; font-weight: bold; }
        .update-time {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 12px;
        }
        @media (max-width: 1024px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
    <script>
        // Auto-refresh every 5 seconds
        function refreshData() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => updateDashboard(data));
        }
        function updateDashboard(data) {
            // Update engine stats
            const engineTable = document.getElementById('engine-stats');
            if (engineTable && data.engines) {
                let html = '<tr><th>Engine</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th><th>Rank</th></tr>';
                Object.entries(data.engines).forEach(([engine, stats], index) => {
                    const pnlClass = stats.total_pnl >= 0 ? 'positive' : 'negative';
                    html += `<tr>
                        <td>Engine ${engine}</td>
                        <td>${stats.total_trades}</td>
                        <td>${stats.win_rate.toFixed(1)}%</td>
                        <td class="${pnlClass}">$${stats.total_pnl.toFixed(2)}</td>
                        <td class="rank">#${index + 1}</td>
                    </tr>`;
                });
                engineTable.innerHTML = html;
            }

            // Update system health
            if (data.health) {
                document.getElementById('total-strategies').textContent = data.health.total_strategies;
                document.getElementById('active-trades').textContent = data.health.active_trades;
                document.getElementById('trades-24h').textContent = data.health.trades_24h;
            }

            // Update recent trades
            const tradesTable = document.getElementById('recent-trades');
            if (tradesTable && data.trades) {
                let html = '<tr><th>Time</th><th>Engine</th><th>Symbol</th><th>Direction</th><th>P&L</th></tr>';
                data.trades.slice(0, 10).forEach(trade => {
                    const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
                    const time = new Date(trade.timestamp).toLocaleTimeString();
                    html += `<tr>
                        <td>${time}</td>
                        <td>${trade.engine}</td>
                        <td>${trade.symbol}</td>
                        <td>${trade.direction}</td>
                        <td class="${pnlClass}">$${trade.pnl.toFixed(2)}</td>
                    </tr>`;
                });
                tradesTable.innerHTML = html;
            }

            // Update timestamp
            document.getElementById('last-update').textContent =
                `Last Updated: ${new Date().toLocaleString()}`;
        }

        // Initial load and auto-refresh
        window.addEventListener('load', () => {
            refreshData();
            setInterval(refreshData, 5000);  // Refresh every 5 seconds
        });
    </script>
</head>
<body>
    <div class="container">
        <h1>HYDRA 3.0 - Terminal Dashboard</h1>

        <div class="grid">
            <!-- Engine Performance -->
            <div class="panel">
                <h2>Engine Performance</h2>
                <table id="engine-stats">
                    <tr><td colspan="5">Loading...</td></tr>
                </table>
            </div>

            <!-- System Health -->
            <div class="panel">
                <h2>System Health</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">Total Strategies</div>
                        <div class="stat-value" id="total-strategies">-</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Active Trades</div>
                        <div class="stat-value" id="active-trades">-</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Trades (24h)</div>
                        <div class="stat-value" id="trades-24h">-</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recent Trades -->
        <div class="panel">
            <h2>Recent Trades</h2>
            <table id="recent-trades">
                <tr><td>Loading...</td></tr>
            </table>
        </div>

        <div class="update-time" id="last-update">Initializing...</div>
    </div>
</body>
</html>
    """
    return render_template_string(html)

@app.route("/api/stats")
def api_stats():
    """API endpoint for dashboard statistics."""
    try:
        stats = get_engine_stats()
        trades = get_recent_trades(20)
        health = get_system_health()

        # Rank engines by P&L
        ranked_engines = dict(sorted(stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True))

        return jsonify({
            "engines": ranked_engines,
            "trades": trades,
            "health": health,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

def main():
    """Run web server."""
    print("Starting HYDRA Terminal Dashboard Web Server...")
    print("Access at: http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)

if __name__ == "__main__":
    main()
