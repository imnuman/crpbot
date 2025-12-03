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

from .chat_page import chat_interface


class HydraState(rx.State):
    """Main state for HYDRA dashboard with real-time updates"""

    # Gladiator stats
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

    # Auto-refresh control
    _refresh_active: bool = False

    def _load_mother_ai_data(self):
        """Internal method to load Mother AI state file"""
        try:
            mother_ai_state = Path("/root/crpbot/data/hydra/mother_ai_state.json")

            if not mother_ai_state.exists():
                return False

            with open(mother_ai_state, 'r') as f:
                state = json.load(f)

            gladiators = state.get("gladiators", {})

            # Set gladiator action counts
            self.gladiator_a_strategies = gladiators.get("A", {}).get("total_trades", 0)
            self.gladiator_b_approvals = gladiators.get("B", {}).get("total_trades", 0)
            self.gladiator_c_backtests = gladiators.get("C", {}).get("total_trades", 0)
            self.gladiator_d_syntheses = gladiators.get("D", {}).get("total_trades", 0)

            # Calculate aggregate stats
            self.total_trades = sum(g.get("total_trades", 0) for g in gladiators.values())
            self.open_trades = sum(g.get("open_trades", 0) for g in gladiators.values())
            self.closed_trades = sum(g.get("closed_trades", 0) for g in gladiators.values())

            # Win rate
            if self.closed_trades > 0:
                total_wins = sum(g.get("wins", 0) for g in gladiators.values())
                self.win_rate = (total_wins / self.closed_trades) * 100
            else:
                self.win_rate = 0.0

            # P&L
            self.total_pnl_percent = sum(g.get("total_pnl_percent", 0) for g in gladiators.values())

            # Empty trades for now
            self.recent_trades = []

            return True

        except Exception as e:
            print(f"[DASHBOARD] ERROR loading Mother AI data: {e}")
            return False

    def load_data(self):
        """Public method to reload data - called by Refresh button"""
        self._load_mother_ai_data()
        self.last_update = datetime.now().strftime("%H:%M:%S")

        # Check if Mother AI is running
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        self.hydra_running = ('mother_ai_runtime.py' in result.stdout or
                            'hydra_runtime.py' in result.stdout)

    def __init__(self, *args, **kwargs):
        """Initialize state and load initial data"""
        super().__init__(*args, **kwargs)
        # Load data immediately on state creation
        self._load_mother_ai_data()
        self.last_update = datetime.now().strftime("%H:%M:%S")

        # Check if Mother AI is running
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        self.hydra_running = ('mother_ai_runtime.py' in result.stdout or
                            'hydra_runtime.py' in result.stdout)


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


def index() -> rx.Component:
    """Main dashboard page"""
    return rx.fragment(
        # Auto-refresh script - reloads page every 30 seconds
        rx.script("""
            setInterval(function() {
                window.location.reload();
            }, 30000);  // 30 seconds
        """),

        rx.container(
            rx.vstack(
                # Navigation
                rx.hstack(
                    rx.link(
                        rx.button("Dashboard", variant="soft", color_scheme="blue"),
                        href="/",
                    ),
                    rx.link(
                        rx.button("Chat", variant="soft", color_scheme="purple"),
                        href="/chat",
                    ),
                    spacing="2",
                    margin_bottom="4",
                ),

                # Header
                rx.heading("HYDRA 3.0 Dashboard", size="8"),
                rx.hstack(
                    rx.badge(
                        "Live Auto-Refresh (30s)",
                        color_scheme="green",
                        variant="solid",
                    ),
                    rx.text(f"Last Update: {HydraState.last_update}", size="2", color="gray"),
                    rx.button("Refresh Now", on_click=HydraState.load_data, size="1"),
                    spacing="3",
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
    )
)


# App configuration
app = rx.App()
app.add_page(index, route="/", title="HYDRA Dashboard")
app.add_page(chat_interface, route="/chat", title="Chat with Gladiators")
