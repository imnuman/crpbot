"""
HYDRA 3.0 Trading Dashboard - Reflex Version
Real-time WebSocket-based dashboard for HYDRA multi-agent monitoring
"""

import reflex as rx
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import sqlite3
from pathlib import Path
import sys

# Add libs to path for portfolio imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from libs.hydra.engine_portfolio import get_tournament_manager


class HydraState(rx.State):
    """Main state for HYDRA dashboard with real-time updates"""

    # Tournament Rankings
    tournament_rankings: List[Dict[str, Any]] = []

    # Gladiator stats (legacy - kept for compatibility)
    gladiator_a_strategies: int = 0
    gladiator_b_approvals: int = 0
    gladiator_b_rejections: int = 0
    gladiator_c_backtests: int = 0
    gladiator_d_syntheses: int = 0

    # Paper trading stats
    total_trades: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    win_rate: float = 0.0
    total_pnl_percent: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Recent trades
    recent_trades: List[Dict[str, Any]] = []

    # Consensus stats
    unanimous_count: int = 0
    strong_count: int = 0
    weak_count: int = 0
    no_consensus_count: int = 0

    # Status
    last_update: str = "Never"
    hydra_running: bool = False

    def load_data(self):
        """Load data from HYDRA's paper_trades.jsonl and tournament portfolios"""
        try:
            # Load tournament rankings from portfolio system
            try:
                manager = get_tournament_manager()
                tournament_summary = manager.get_tournament_summary()
                self.tournament_rankings = tournament_summary.get("rankings", [])
            except Exception as e:
                print(f"Error loading tournament data: {e}")
                self.tournament_rankings = []

            # Load paper trades (legacy)
            trades_file = Path("/root/crpbot/data/hydra/paper_trades.jsonl")
            if trades_file.exists():
                trades = []
                with open(trades_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            trades.append(json.loads(line))

                # Calculate stats
                self.total_trades = len(trades)
                self.open_trades = sum(1 for t in trades if t.get('status') == 'OPEN')
                self.closed_trades = sum(1 for t in trades if t.get('status') == 'CLOSED')

                closed = [t for t in trades if t.get('outcome') in ['win', 'loss']]
                if closed:
                    wins = [t for t in closed if t.get('outcome') == 'win']
                    losses = [t for t in closed if t.get('outcome') == 'loss']

                    self.win_rate = (len(wins) / len(closed)) * 100 if closed else 0.0

                    if wins:
                        self.avg_win = sum(t.get('pnl_percent', 0) for t in wins) / len(wins)
                    if losses:
                        self.avg_loss = sum(t.get('pnl_percent', 0) for t in losses) / len(losses)

                    self.total_pnl_percent = sum(t.get('pnl_percent', 0) for t in closed)

                # Recent trades (last 10)
                self.recent_trades = trades[-10:] if trades else []

                # Count gladiator actions from paper trades
                self.gladiator_a_strategies = sum(1 for t in trades if t.get('gladiator') == 'A')
                self.gladiator_b_approvals = sum(1 for t in trades if t.get('gladiator') == 'B')
                self.gladiator_c_backtests = sum(1 for t in trades if t.get('gladiator') == 'C')
                self.gladiator_d_syntheses = sum(1 for t in trades if t.get('gladiator') == 'D')

            self.last_update = datetime.now().strftime("%H:%M:%S")

            # Check if HYDRA is running
            import subprocess
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            self.hydra_running = 'hydra_runtime.py' in result.stdout

        except Exception as e:
            print(f"Error loading HYDRA data: {e}")


def engine_card(name: str, role: str, provider: str, count, color: str) -> rx.Component:
    """Card showing gladiator stats"""
    return rx.card(
        rx.vstack(
            rx.heading(f"Gladiator {name}", size="4"),
            rx.text(provider, color="gray", size="2"),
            rx.text(role, size="1", color="gray"),
            rx.divider(),
            rx.hstack(
                rx.text("Actions:", weight="bold"),
                rx.badge(count.to_string(), color_scheme=color),
                spacing="2",
            ),
            align="start",
            spacing="2",
        ),
        width="100%",
    )


def stats_card(title: str, value, subtitle = None) -> rx.Component:
    """Card showing a stat"""
    content = [
        rx.text(title, size="2", color="gray"),
        rx.heading(value, size="6"),
    ]
    if subtitle is not None:
        content.append(rx.text(subtitle, size="1", color="gray"))

    return rx.card(
        rx.vstack(
            *content,
            align="start",
            spacing="1",
        ),
        width="100%",
    )


def trade_row(trade: Dict[str, Any]) -> rx.Component:
    """Row showing a trade"""
    return rx.table.row(
        rx.table.cell(trade["asset"]),
        rx.table.cell(trade["direction"]),
        rx.table.cell("$" + trade["entry_price"].to_string()),
        rx.table.cell(
            rx.badge(
                trade["status"],
                color_scheme=rx.cond(trade["status"] == "OPEN", "green", "gray"),
            )
        ),
        rx.table.cell(trade["pnl_percent"].to_string() + "%"),
    )


def tournament_ranking_row(ranking: Dict[str, Any]) -> rx.Component:
    """Row showing a gladiator's tournament ranking"""
    # Color based on rank
    rank_colors = {1: "gold", 2: "cyan", 3: "orange", 4: "gray"}
    color = rank_colors.get(ranking["rank"], "gray")

    return rx.table.row(
        rx.table.cell(
            rx.badge(f"#{ranking['rank']}", color_scheme=color, variant="solid")
        ),
        rx.table.cell(f"Gladiator {ranking['gladiator']}"),
        rx.table.cell(f"{ranking['weight'] * 100:.0f}%"),
        rx.table.cell(ranking["total_trades"]),
        rx.table.cell(f"{ranking['win_rate'] * 100:.1f}%"),
        rx.table.cell(
            rx.text(
                f"${ranking['total_pnl_usd']:+.2f}",
                color=rx.cond(ranking["total_pnl_usd"] >= 0, "green", "red"),
                weight="bold"
            )
        ),
        rx.table.cell(
            f"{ranking['sharpe_ratio']:.2f}" if ranking.get("sharpe_ratio") else "N/A"
        ),
    )


def index() -> rx.Component:
    """Main dashboard page"""
    return rx.container(
        rx.vstack(
            # Header
            rx.heading("HYDRA 3.0 Dashboard", size="8"),
            rx.hstack(
                rx.badge(
                    "Live",
                    color_scheme="green",
                    variant="solid",
                ),
                rx.text(f"Last Update: {HydraState.last_update}", size="2", color="gray"),
                rx.button("Refresh", on_click=HydraState.load_data, size="1"),
                spacing="3",
            ),

            # Tournament Rankings
            rx.heading("Tournament Rankings", size="6", margin_top="4"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Rank"),
                        rx.table.column_header_cell("Gladiator"),
                        rx.table.column_header_cell("Weight"),
                        rx.table.column_header_cell("Trades"),
                        rx.table.column_header_cell("Win Rate"),
                        rx.table.column_header_cell("Total P&L"),
                        rx.table.column_header_cell("Sharpe"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(HydraState.tournament_rankings, tournament_ranking_row)
                ),
                width="100%",
            ),

            # Gladiators Grid
            rx.heading("Gladiators", size="6", margin_top="4"),
            rx.grid(
                engine_card("A", "Structural Edge", "DeepSeek", HydraState.gladiator_a_strategies, "blue"),
                engine_card("B", "Logic Validator", "Claude", HydraState.gladiator_b_approvals, "purple"),
                engine_card("C", "Fast Backtester", "Grok", HydraState.gladiator_c_backtests, "orange"),
                engine_card("D", "Synthesizer", "Gemini", HydraState.gladiator_d_syntheses, "green"),
                columns="4",
                spacing="4",
                width="100%",
            ),

            # Performance Stats
            rx.heading("Performance", size="6", margin_top="6"),
            rx.grid(
                stats_card("Total Trades", HydraState.total_trades.to_string()),
                stats_card("Open Positions", HydraState.open_trades.to_string(), "Active"),
                stats_card("Win Rate", HydraState.win_rate.to_string() + "%", HydraState.closed_trades.to_string() + " closed"),
                stats_card("Total P&L", HydraState.total_pnl_percent.to_string() + "%"),
                columns="4",
                spacing="4",
                width="100%",
            ),

            # Recent Trades Table
            rx.heading("Recent Trades", size="6", margin_top="6"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Asset"),
                        rx.table.column_header_cell("Direction"),
                        rx.table.column_header_cell("Entry Price"),
                        rx.table.column_header_cell("Status"),
                        rx.table.column_header_cell("P&L"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(HydraState.recent_trades, trade_row)
                ),
                width="100%",
            ),

            spacing="4",
            padding="4",
        ),
        max_width="1400px",
        on_mount=HydraState.load_data,
    )


# App configuration
app = rx.App()
app.add_page(index)
