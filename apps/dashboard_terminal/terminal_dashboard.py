#!/usr/bin/env python3
"""
HYDRA 3.0 Terminal Dashboard

Real-time monitoring dashboard for HYDRA trading system.
Can be run in terminal or hosted as web interface via ttyd.

Features:
- Live P&L tracking per engine
- Trade history
- Win rate statistics
- Engine rankings
- System health metrics
"""

import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

class HydraDashboard:
    """Terminal dashboard for HYDRA system."""

    def __init__(self, db_path: str = "/root/crpbot/data/hydra/hydra.db"):
        self.db_path = db_path
        self.refresh_interval = 5  # seconds

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get performance stats for all engines."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}
        for engine in ["A", "B", "C", "D"]:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN pnl_usd > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(pnl_usd) as total_pnl,
                    AVG(pnl_usd) as avg_pnl
                FROM hydra_trades
                WHERE gladiator = ? AND exit_time IS NOT NULL
            """, (engine,))

            result = cursor.fetchone()
            stats[engine] = {
                "total_trades": result[0] or 0,
                "wins": result[1] or 0,
                "losses": result[2] or 0,
                "win_rate": result[3] or 0.0,
                "total_pnl": result[4] or 0.0,
                "avg_pnl": result[5] or 0.0
            }

        conn.close()
        return stats

    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades across all engines."""
        conn = sqlite3.connect(self.db_path)
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
                exit_reason
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
                "pnl_usd": row[7],
                "exit_reason": row[8]
            })

        conn.close()
        return trades

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        conn = sqlite3.connect(self.db_path)
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
            "uptime": "N/A"  # TODO: Track runtime uptime
        }

    def create_engine_table(self, stats: Dict[str, Any]) -> Table:
        """Create engine performance table."""
        table = Table(title="Engine Performance", box=box.ROUNDED, title_style="bold cyan")

        table.add_column("Engine", justify="center", style="cyan")
        table.add_column("Trades", justify="right")
        table.add_column("Wins", justify="right", style="green")
        table.add_column("Losses", justify="right", style="red")
        table.add_column("Win Rate", justify="right")
        table.add_column("Total P&L", justify="right")
        table.add_column("Avg P&L", justify="right")
        table.add_column("Rank", justify="center", style="yellow")

        # Sort engines by total P&L for ranking
        ranked_engines = sorted(stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True)

        for rank, (engine, data) in enumerate(ranked_engines, 1):
            win_rate_str = f"{data['win_rate']:.1%}"
            pnl_style = "green" if data['total_pnl'] > 0 else "red"

            table.add_row(
                f"Engine {engine}",
                str(data['total_trades']),
                str(data['wins']),
                str(data['losses']),
                win_rate_str,
                f"[{pnl_style}]${data['total_pnl']:.2f}[/{pnl_style}]",
                f"[{pnl_style}]${data['avg_pnl']:.2f}[/{pnl_style}]",
                f"#{rank}"
            )

        return table

    def create_trades_table(self, trades: List[Dict]) -> Table:
        """Create recent trades table."""
        table = Table(title="Recent Trades", box=box.ROUNDED, title_style="bold magenta")

        table.add_column("Time", style="dim")
        table.add_column("Engine", justify="center")
        table.add_column("Symbol")
        table.add_column("Direction", justify="center")
        table.add_column("Entry", justify="right")
        table.add_column("Exit", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Exit Reason")

        for trade in trades[:10]:  # Show last 10 trades
            pnl_style = "green" if trade['pnl_usd'] > 0 else "red"
            direction_style = "green" if trade['direction'] == "BUY" else "red"

            # Format timestamp
            try:
                ts = datetime.fromisoformat(trade['timestamp'])
                time_str = ts.strftime("%H:%M:%S")
            except:
                time_str = "N/A"

            table.add_row(
                time_str,
                trade['engine'],
                trade['symbol'],
                f"[{direction_style}]{trade['direction']}[/{direction_style}]",
                f"${trade['entry_price']:.2f}",
                f"${trade['exit_price']:.2f}" if trade['exit_price'] else "Open",
                f"[{pnl_style}]${trade['pnl_usd']:.2f}[/{pnl_style}]",
                trade['exit_reason'] or "N/A"
            )

        return table

    def create_system_panel(self, health: Dict[str, Any]) -> Panel:
        """Create system health panel."""
        content = Text()
        content.append(f"Total Strategies: ", style="bold")
        content.append(f"{health['total_strategies']}\n", style="cyan")
        content.append(f"Active Trades: ", style="bold")
        content.append(f"{health['active_trades']}\n", style="yellow")
        content.append(f"Trades (24h): ", style="bold")
        content.append(f"{health['trades_24h']}\n", style="green")

        return Panel(content, title="System Health", border_style="green")

    def create_dashboard_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        # Get data
        stats = self.get_engine_stats()
        trades = self.get_recent_trades(10)
        health = self.get_system_health()

        # Create components
        header = Panel(
            Text("HYDRA 3.0 Dashboard", justify="center", style="bold white on blue"),
            subtitle=f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        engine_table = self.create_engine_table(stats)
        trades_table = self.create_trades_table(trades)
        system_panel = self.create_system_panel(health)

        # Build layout
        layout.split_column(
            Layout(header, size=3),
            Layout(name="body")
        )

        layout["body"].split_row(
            Layout(engine_table, name="engines"),
            Layout(name="right")
        )

        layout["right"].split_column(
            Layout(system_panel, size=8),
            Layout(trades_table)
        )

        return layout

    def run(self, refresh_interval: int = 5):
        """Run dashboard with live updates."""
        console.clear()
        console.print("[bold green]Starting HYDRA Dashboard...[/bold green]")
        time.sleep(1)

        with Live(self.create_dashboard_layout(), console=console, refresh_per_second=0.2) as live:
            while True:
                try:
                    time.sleep(refresh_interval)
                    live.update(self.create_dashboard_layout())
                except KeyboardInterrupt:
                    console.print("\n[yellow]Dashboard stopped by user[/yellow]")
                    break
                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]")
                    time.sleep(5)

def main():
    """Main entry point."""
    dashboard = HydraDashboard()
    dashboard.run(refresh_interval=5)

if __name__ == "__main__":
    main()
