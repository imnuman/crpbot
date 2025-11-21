"""
V7 Ultimate Trading Dashboard - Reflex Version
Real-time WebSocket-based dashboard for V7 signal monitoring
"""

import reflex as rx
from datetime import datetime, timedelta
from typing import List, Dict, Any, TypedDict
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
import json


class Position(TypedDict):
    signal_id: int
    symbol: str
    direction: str
    entry_price: float
    entry_timestamp: str
    entry_time: str  # Formatted EST timestamp


class Trade(TypedDict):
    signal_id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_percent: float
    outcome: str
    entry_timestamp: str
    exit_timestamp: str
    entry_time: str  # Formatted EST timestamp
    exit_time: str   # Formatted EST timestamp
    hold_duration_minutes: int

# Import your existing models
import sys
sys.path.insert(0, '/root/crpbot')
from libs.db.models import Signal
from libs.config.config import Settings
from libs.tracking.performance_tracker import PerformanceTracker

# Initialize settings
config = Settings()


class V7State(rx.State):
    """Main state for V7 dashboard with real-time updates"""

    # Signal data
    signals: List[Dict[str, Any]] = []
    signal_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    avg_confidence: float = 0.0

    # Market prices
    btc_price: float = 0.0
    eth_price: float = 0.0
    sol_price: float = 0.0

    # Cost tracking
    daily_cost: float = 0.0
    monthly_cost: float = 0.0

    # Status
    last_update: str = "Never"
    is_loading: bool = False

    # Performance tracking data
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    open_positions: List[Position] = []
    recent_trades: List[Trade] = []

    # A/B Test data
    full_math_total: int = 0
    full_math_wins: int = 0
    full_math_losses: int = 0
    full_math_win_rate: float = 0.0
    full_math_avg_pnl: float = 0.0
    full_math_profit_factor: float = 0.0
    full_math_trades: List[Trade] = []

    deepseek_only_total: int = 0
    deepseek_only_wins: int = 0
    deepseek_only_losses: int = 0
    deepseek_only_win_rate: float = 0.0
    deepseek_only_avg_pnl: float = 0.0
    deepseek_only_profit_factor: float = 0.0
    deepseek_only_trades: List[Trade] = []

    # Class-level engine (shared across all instances to prevent leaks)
    _engine = None
    _Session = None

    @classmethod
    def get_session(cls):
        """Get a database session with proper connection pooling"""
        if cls._engine is None:
            cls._engine = create_engine(
                str(config.db_url),
                pool_size=5,  # Limit connection pool
                max_overflow=10,
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_pre_ping=True  # Verify connections before use
            )
            cls._Session = sessionmaker(bind=cls._engine)
        return cls._Session()

    def fetch_signals(self):
        """Fetch latest V7 signals from database"""
        self.is_loading = True

        session = self.get_session()
        try:
            # Fetch last 2 hours of signals
            since = datetime.now() - timedelta(hours=2)
            signals_query = session.query(Signal).filter(
                Signal.timestamp >= since,
                Signal.model_version == 'v7_ultimate'
            ).order_by(desc(Signal.timestamp)).limit(30).all()

            # Process signals
            signals_data = []
            buy_count = sell_count = hold_count = 0
            total_confidence = 0.0

            for s in signals_query:
                # Parse notes for reasoning
                reasoning = 'N/A'
                if s.notes:
                    try:
                        data = json.loads(s.notes)
                        reasoning = data.get('reasoning', 'N/A')
                        # Limit reasoning to first sentence or 300 chars
                        if len(reasoning) > 300:
                            reasoning = reasoning[:300] + '...'
                    except:
                        reasoning = s.notes[:300] if s.notes else 'N/A'

                # Count by direction
                direction = s.direction.lower()
                if direction == 'buy' or direction == 'long':
                    buy_count += 1
                elif direction == 'sell' or direction == 'short':
                    sell_count += 1
                else:
                    hold_count += 1

                total_confidence += (s.confidence or 0.0)

                signals_data.append({
                    'timestamp': s.timestamp.strftime('%b %d %H:%M EST'),
                    'symbol': s.symbol,
                    'direction': s.direction.upper(),
                    'confidence': f"{(s.confidence or 0.0) * 100:.1f}%",
                    'confidence_num': (s.confidence or 0.0) * 100,
                    'entry_price': f"${s.entry_price:,.2f}" if s.entry_price else 'N/A',
                    'reasoning': reasoning,
                    'tier': s.tier or 'low'
                })

            avg_conf = (total_confidence / len(signals_query) * 100) if signals_query else 0.0

            # Update state (triggers WebSocket push to frontend)
            self.signals = signals_data
            self.signal_count = len(signals_query)
            self.buy_count = buy_count
            self.sell_count = sell_count
            self.hold_count = hold_count
            self.avg_confidence = avg_conf
            self.last_update = datetime.now().strftime('%H:%M:%S EST')
            self.is_loading = False

        except Exception as e:
            print(f"Error fetching signals: {e}")
            import traceback
            print(traceback.format_exc())
            self.is_loading = False
        finally:
            # CRITICAL: Always close session to prevent connection leaks
            session.close()

    def fetch_market_prices(self):
        """Fetch latest market prices"""
        session = self.get_session()
        try:
            # Get most recent signals to extract prices
            for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                latest = session.query(Signal).filter(
                    Signal.symbol == symbol
                ).order_by(desc(Signal.timestamp)).first()

                if latest and latest.entry_price:
                    if symbol == 'BTC-USD':
                        self.btc_price = latest.entry_price
                    elif symbol == 'ETH-USD':
                        self.eth_price = latest.entry_price
                    elif symbol == 'SOL-USD':
                        self.sol_price = latest.entry_price

        except Exception as e:
            print(f"Error fetching prices: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            # CRITICAL: Always close session to prevent connection leaks
            session.close()

    def fetch_performance(self):
        """Fetch performance tracking data"""
        from libs.utils.timezone import format_timestamp_est
        from dateutil import parser

        tracker = PerformanceTracker()

        try:
            # Get performance stats
            stats = tracker.get_win_rate(days=30)
            self.total_trades = stats['total_trades']
            self.wins = stats['wins']
            self.losses = stats['losses']
            self.win_rate = stats['win_rate']
            self.avg_win = stats['avg_win']
            self.avg_loss = stats['avg_loss']
            self.profit_factor = stats['profit_factor']

            # Get open positions and format timestamps
            positions = tracker.get_open_positions()
            for pos in positions:
                if pos.get('entry_timestamp'):
                    try:
                        pos['entry_time'] = format_timestamp_est(parser.parse(pos['entry_timestamp']))
                    except:
                        pos['entry_time'] = 'N/A'
                else:
                    pos['entry_time'] = 'N/A'
            self.open_positions = positions

            # Get recent trades and format timestamps
            trades = tracker.get_recent_trades(limit=10)
            for trade in trades:
                if trade.get('entry_timestamp'):
                    try:
                        trade['entry_time'] = format_timestamp_est(parser.parse(trade['entry_timestamp']))
                    except:
                        trade['entry_time'] = 'N/A'
                else:
                    trade['entry_time'] = 'N/A'

                if trade.get('exit_timestamp'):
                    try:
                        trade['exit_time'] = format_timestamp_est(parser.parse(trade['exit_timestamp']))
                    except:
                        trade['exit_time'] = 'N/A'
                else:
                    trade['exit_time'] = 'N/A'
            self.recent_trades = trades

        except Exception as e:
            print(f"Error fetching performance: {e}")
            import traceback
            print(traceback.format_exc())

    def fetch_ab_test_data(self):
        """Fetch A/B test comparison data"""
        from libs.utils.timezone import format_timestamp_est
        from dateutil import parser

        tracker = PerformanceTracker()

        try:
            # Get data for v7_full_math strategy
            full_math_stats = tracker.get_win_rate(days=30, strategy="v7_full_math")
            self.full_math_total = full_math_stats['total_trades']
            self.full_math_wins = full_math_stats['wins']
            self.full_math_losses = full_math_stats['losses']
            self.full_math_win_rate = full_math_stats['win_rate']
            self.full_math_avg_pnl = full_math_stats['avg_pnl']
            self.full_math_profit_factor = full_math_stats['profit_factor']

            # Get recent trades for v7_full_math
            full_math_trades = tracker.get_recent_trades(limit=10, strategy="v7_full_math")
            for trade in full_math_trades:
                if trade.get('entry_timestamp'):
                    try:
                        trade['entry_time'] = format_timestamp_est(parser.parse(trade['entry_timestamp']))
                    except:
                        trade['entry_time'] = 'N/A'
                else:
                    trade['entry_time'] = 'N/A'

                if trade.get('exit_timestamp'):
                    try:
                        trade['exit_time'] = format_timestamp_est(parser.parse(trade['exit_timestamp']))
                    except:
                        trade['exit_time'] = 'N/A'
                else:
                    trade['exit_time'] = 'N/A'
            self.full_math_trades = full_math_trades

            # Get data for v7_deepseek_only strategy
            deepseek_only_stats = tracker.get_win_rate(days=30, strategy="v7_deepseek_only")
            self.deepseek_only_total = deepseek_only_stats['total_trades']
            self.deepseek_only_wins = deepseek_only_stats['wins']
            self.deepseek_only_losses = deepseek_only_stats['losses']
            self.deepseek_only_win_rate = deepseek_only_stats['win_rate']
            self.deepseek_only_avg_pnl = deepseek_only_stats['avg_pnl']
            self.deepseek_only_profit_factor = deepseek_only_stats['profit_factor']

            # Get recent trades for v7_deepseek_only
            deepseek_only_trades = tracker.get_recent_trades(limit=10, strategy="v7_deepseek_only")
            for trade in deepseek_only_trades:
                if trade.get('entry_timestamp'):
                    try:
                        trade['entry_time'] = format_timestamp_est(parser.parse(trade['entry_timestamp']))
                    except:
                        trade['entry_time'] = 'N/A'
                else:
                    trade['entry_time'] = 'N/A'

                if trade.get('exit_timestamp'):
                    try:
                        trade['exit_time'] = format_timestamp_est(parser.parse(trade['exit_timestamp']))
                    except:
                        trade['exit_time'] = 'N/A'
                else:
                    trade['exit_time'] = 'N/A'
            self.deepseek_only_trades = deepseek_only_trades

        except Exception as e:
            print(f"Error fetching A/B test data: {e}")
            import traceback
            print(traceback.format_exc())

    def on_load(self):
        """Called when page loads - fetch data"""
        self.fetch_signals()
        self.fetch_market_prices()
        self.fetch_performance()


def signal_card(signal: Dict[str, Any]) -> rx.Component:
    """Render a single signal card"""
    # Color based on direction
    direction_color = rx.cond(
        signal['direction'] == 'BUY',
        'green',
        rx.cond(signal['direction'] == 'SELL', 'red', 'gray')
    )

    return rx.box(
        rx.hstack(
            # Timestamp
            rx.text(signal['timestamp'], size="2", color='gray'),
            # Symbol
            rx.badge(signal['symbol'], color_scheme='blue'),
            # Direction
            rx.badge(signal['direction'], color_scheme=direction_color),
            # Confidence
            rx.badge(signal['confidence'], color_scheme='purple'),
            # Entry Price
            rx.text(signal['entry_price'], size="2", weight='bold'),
            spacing="3",
        ),
        rx.text(
            signal['reasoning'],
            size="1",
            color='gray',
            style={'margin_top': '8px'}
        ),
        padding="3",
        border_radius="md",
        border="1px solid",
        border_color='gray.200',
        margin_bottom="2",
    )


def index() -> rx.Component:
    """Main dashboard page"""
    return rx.container(
        # Header
        rx.heading("V7 Ultimate Trading Dashboard", size="8", margin_bottom="4"),
        rx.hstack(
            rx.text(
                f"Real-time signal monitoring with WebSocket updates",
                size="3",
                color='gray',
            ),
            rx.hstack(
                rx.link(
                    rx.button("View Performance", size="2"),
                    href="/performance",
                ),
                rx.link(
                    rx.button("A/B Test Results", size="2", color_scheme='orange'),
                    href="/ab-test",
                ),
                spacing="2",
            ),
            justify="between",
            align_items="center",
            margin_bottom="6",
        ),

        # Stats Cards
        rx.hstack(
            rx.card(
                rx.vstack(
                    rx.text("Total Signals", size="2", color='gray'),
                    rx.heading(V7State.signal_count, size="7"),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("BUY Signals", size="2", color='gray'),
                    rx.heading(V7State.buy_count, size="7", color='green'),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("SELL Signals", size="2", color='gray'),
                    rx.heading(V7State.sell_count, size="7", color='red'),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("HOLD Signals", size="2", color='gray'),
                    rx.heading(V7State.hold_count, size="7", color='gray'),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("Avg Confidence", size="2", color='gray'),
                    rx.heading(f"{V7State.avg_confidence:.1f}%", size="7", color='purple'),
                    align_items="start",
                ),
            ),
            spacing="4",
            margin_bottom="6",
        ),

        # Market Prices
        rx.card(
            rx.heading("Live Market Prices", size="5", margin_bottom="3"),
            rx.hstack(
                rx.vstack(
                    rx.text("BTC", size="2", color='gray'),
                    rx.heading(f"${V7State.btc_price:,.2f}", size="6"),
                    align_items="start",
                ),
                rx.vstack(
                    rx.text("ETH", size="2", color='gray'),
                    rx.heading(f"${V7State.eth_price:,.2f}", size="6"),
                    align_items="start",
                ),
                rx.vstack(
                    rx.text("SOL", size="2", color='gray'),
                    rx.heading(f"${V7State.sol_price:,.2f}", size="6"),
                    align_items="start",
                ),
                spacing="8",
            ),
            margin_bottom="6",
        ),

        # Signals Section
        rx.box(
            rx.hstack(
                rx.heading("Recent Signals (Last 2 Hours)", size="5"),
                rx.button(
                    "Refresh",
                    on_click=V7State.fetch_signals,
                    loading=V7State.is_loading,
                    size="2",
                ),
                rx.text(f"Last update: {V7State.last_update}", size="2", color='gray'),
                justify="between",
                align_items="center",
                margin_bottom="4",
            ),

            # Signal cards
            rx.cond(
                V7State.signals.length() > 0,
                rx.foreach(V7State.signals, signal_card),
                rx.text("No signals in the last 2 hours", size="3", color='gray'),
            ),
        ),

        max_width="1200px",
        padding="4",
        on_mount=V7State.on_load,
    )


def trade_row(trade: Trade) -> rx.Component:
    """Render a single trade row"""
    outcome_color = rx.cond(
        trade['outcome'] == 'win',
        'green',
        rx.cond(trade['outcome'] == 'loss', 'red', 'gray')
    )

    outcome_icon = rx.cond(
        trade['outcome'] == 'win',
        '✅',
        rx.cond(trade['outcome'] == 'loss', '❌', '➖')
    )

    return rx.table.row(
        rx.table.cell(outcome_icon),
        rx.table.cell(trade['symbol']),
        rx.table.cell(rx.badge(trade['direction'], color_scheme=outcome_color)),
        rx.table.cell(f"${trade['entry_price']:,.2f}"),
        rx.table.cell(trade['entry_time'], size="1"),
        rx.table.cell(f"${trade['exit_price']:,.2f}"),
        rx.table.cell(trade['exit_time'], size="1"),
        rx.table.cell(
            rx.text(f"{trade['pnl_percent']:.2f}%", color=outcome_color),
            weight='bold'
        ),
        rx.table.cell(f"{trade['hold_duration_minutes']}m"),
    )


def position_row(position: Position) -> rx.Component:
    """Render a single open position row"""
    direction_color = rx.cond(
        position['direction'] == 'long',
        'green',
        'red'
    )

    return rx.table.row(
        rx.table.cell(f"#{position['signal_id']}"),
        rx.table.cell(position['symbol']),
        rx.table.cell(rx.badge(position['direction'], color_scheme=direction_color)),
        rx.table.cell(f"${position['entry_price']:,.2f}"),
        rx.table.cell(position['entry_time'], size="2"),
    )


def performance() -> rx.Component:
    """Performance tracking page"""
    return rx.container(
        # Header
        rx.heading("V7 Performance Tracking", size="8", margin_bottom="4"),
        rx.hstack(
            rx.text(
                "Real-time trading performance metrics (Last 30 Days)",
                size="3",
                color='gray',
            ),
            rx.hstack(
                rx.link(
                    rx.button("Back to Signals", size="2"),
                    href="/",
                ),
                rx.link(
                    rx.button("A/B Test Results", size="2", color_scheme='orange'),
                    href="/ab-test",
                ),
                spacing="2",
            ),
            justify="between",
            align_items="center",
            margin_bottom="6",
        ),

        # Performance Stats Cards
        rx.hstack(
            rx.card(
                rx.vstack(
                    rx.text("Total Trades", size="2", color='gray'),
                    rx.heading(V7State.total_trades, size="7"),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("Win Rate", size="2", color='gray'),
                    rx.heading(f"{V7State.win_rate:.1f}%", size="7", color='green'),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("Wins / Losses", size="2", color='gray'),
                    rx.heading(f"{V7State.wins} / {V7State.losses}", size="7"),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("Avg Win", size="2", color='gray'),
                    rx.heading(f"{V7State.avg_win:.2f}%", size="7", color='green'),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("Avg Loss", size="2", color='gray'),
                    rx.heading(f"{V7State.avg_loss:.2f}%", size="7", color='red'),
                    align_items="start",
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("Profit Factor", size="2", color='gray'),
                    rx.heading(f"{V7State.profit_factor:.2f}", size="7", color='purple'),
                    align_items="start",
                ),
            ),
            spacing="4",
            margin_bottom="6",
        ),

        # Open Positions Section
        rx.card(
            rx.hstack(
                rx.heading("Open Positions", size="5"),
                rx.button(
                    "Refresh",
                    on_click=V7State.fetch_performance,
                    size="2",
                ),
                justify="between",
                align_items="center",
                margin_bottom="4",
            ),

            rx.cond(
                V7State.open_positions.length() > 0,
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Signal"),
                            rx.table.column_header_cell("Symbol"),
                            rx.table.column_header_cell("Direction"),
                            rx.table.column_header_cell("Entry Price"),
                            rx.table.column_header_cell("Entry Time"),
                        ),
                    ),
                    rx.table.body(
                        rx.foreach(V7State.open_positions, position_row)
                    ),
                ),
                rx.text("No open positions", size="3", color='gray'),
            ),
            margin_bottom="6",
        ),

        # Recent Trades Section
        rx.card(
            rx.heading("Recent Trades", size="5", margin_bottom="4"),

            rx.cond(
                V7State.recent_trades.length() > 0,
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell(""),
                            rx.table.column_header_cell("Symbol"),
                            rx.table.column_header_cell("Direction"),
                            rx.table.column_header_cell("Entry $"),
                            rx.table.column_header_cell("Entry Time"),
                            rx.table.column_header_cell("Exit $"),
                            rx.table.column_header_cell("Exit Time"),
                            rx.table.column_header_cell("P&L"),
                            rx.table.column_header_cell("Duration"),
                        ),
                    ),
                    rx.table.body(
                        rx.foreach(V7State.recent_trades, trade_row)
                    ),
                ),
                rx.text("No closed trades yet", size="3", color='gray'),
            ),
        ),

        max_width="1400px",
        padding="4",
        on_mount=V7State.on_load,
    )


def ab_test_comparison() -> rx.Component:
    """A/B Test Comparison Page - Side-by-side comparison of strategies"""
    return rx.container(
        # Header
        rx.heading("A/B Test: Strategy Comparison", size="8", margin_bottom="4"),
        rx.hstack(
            rx.text(
                "Comparing v7_full_math (WITH mathematical theories) vs v7_deepseek_only (NO math theories)",
                size="3",
                color='gray',
            ),
            rx.link(
                rx.button("Back to Performance", size="2"),
                href="/performance",
            ),
            justify="between",
            align_items="center",
            margin_bottom="6",
        ),

        # Side-by-side comparison grid
        rx.grid(
            # Left side: v7_full_math
            rx.box(
                rx.card(
                    rx.heading("v7_full_math", size="6", color='blue', margin_bottom="4"),
                    rx.text("DeepSeek WITH mathematical theories", size="2", color='gray', margin_bottom="4"),

                    # Stats
                    rx.vstack(
                        rx.hstack(
                            rx.box(
                                rx.text("Total Trades", size="2", color='gray'),
                                rx.heading(V7State.full_math_total, size="6"),
                                padding="3",
                                border_radius="md",
                                border="1px solid",
                                border_color='blue.200',
                                width="100%",
                            ),
                            rx.box(
                                rx.text("Win Rate", size="2", color='gray'),
                                rx.heading(f"{V7State.full_math_win_rate:.1f}%", size="6", color='green'),
                                padding="3",
                                border_radius="md",
                                border="1px solid",
                                border_color='blue.200',
                                width="100%",
                            ),
                            spacing="3",
                        ),
                        rx.hstack(
                            rx.box(
                                rx.text("Wins / Losses", size="2", color='gray'),
                                rx.heading(f"{V7State.full_math_wins} / {V7State.full_math_losses}", size="6"),
                                padding="3",
                                border_radius="md",
                                border="1px solid",
                                border_color='blue.200',
                                width="100%",
                            ),
                            rx.box(
                                rx.text("Avg P&L", size="2", color='gray'),
                                rx.heading(f"{V7State.full_math_avg_pnl:.2f}%", size="6", color='purple'),
                                padding="3",
                                border_radius="md",
                                border="1px solid",
                                border_color='blue.200',
                                width="100%",
                            ),
                            spacing="3",
                        ),
                        rx.box(
                            rx.text("Profit Factor", size="2", color='gray'),
                            rx.heading(f"{V7State.full_math_profit_factor:.2f}", size="6", color='blue'),
                            padding="3",
                            border_radius="md",
                            border="1px solid",
                            border_color='blue.200',
                            width="100%",
                        ),
                        spacing="4",
                    ),
                ),

                # Recent trades for full_math
                rx.card(
                    rx.heading("Recent Trades", size="5", margin_bottom="4"),
                    rx.cond(
                        V7State.full_math_trades.length() > 0,
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    rx.table.column_header_cell(""),
                                    rx.table.column_header_cell("Symbol"),
                                    rx.table.column_header_cell("P&L"),
                                    rx.table.column_header_cell("Exit Time"),
                                ),
                            ),
                            rx.table.body(
                                rx.foreach(V7State.full_math_trades, lambda t: rx.table.row(
                                    rx.table.cell(
                                        rx.cond(
                                            t['outcome'] == 'win',
                                            '✅',
                                            rx.cond(t['outcome'] == 'loss', '❌', '➖')
                                        )
                                    ),
                                    rx.table.cell(t['symbol']),
                                    rx.table.cell(
                                        rx.text(
                                            f"{t['pnl_percent']:.2f}%",
                                            color=rx.cond(
                                                t['outcome'] == 'win',
                                                'green',
                                                rx.cond(t['outcome'] == 'loss', 'red', 'gray')
                                            ),
                                        ),
                                        weight='bold'
                                    ),
                                    rx.table.cell(t['exit_time'], size="1"),
                                ))
                            ),
                        ),
                        rx.text("No trades yet for this strategy", size="3", color='gray'),
                    ),
                    margin_top="4",
                ),
            ),

            # Right side: v7_deepseek_only
            rx.box(
                rx.card(
                    rx.heading("v7_deepseek_only", size="6", color='orange', margin_bottom="4"),
                    rx.text("DeepSeek WITHOUT mathematical theories (minimal prompt)", size="2", color='gray', margin_bottom="4"),

                    # Stats
                    rx.vstack(
                        rx.hstack(
                            rx.box(
                                rx.text("Total Trades", size="2", color='gray'),
                                rx.heading(V7State.deepseek_only_total, size="6"),
                                padding="3",
                                border_radius="md",
                                border="1px solid",
                                border_color='orange.200',
                                width="100%",
                            ),
                            rx.box(
                                rx.text("Win Rate", size="2", color='gray'),
                                rx.heading(f"{V7State.deepseek_only_win_rate:.1f}%", size="6", color='green'),
                                padding="3",
                                border_radius="md",
                                border="1px solid",
                                border_color='orange.200',
                                width="100%",
                            ),
                            spacing="3",
                        ),
                        rx.hstack(
                            rx.box(
                                rx.text("Wins / Losses", size="2", color='gray'),
                                rx.heading(f"{V7State.deepseek_only_wins} / {V7State.deepseek_only_losses}", size="6"),
                                padding="3",
                                border_radius="md",
                                border="1px solid",
                                border_color='orange.200',
                                width="100%",
                            ),
                            rx.box(
                                rx.text("Avg P&L", size="2", color='gray'),
                                rx.heading(f"{V7State.deepseek_only_avg_pnl:.2f}%", size="6", color='purple'),
                                padding="3",
                                border_radius="md",
                                border="1px solid",
                                border_color='orange.200',
                                width="100%",
                            ),
                            spacing="3",
                        ),
                        rx.box(
                            rx.text("Profit Factor", size="2", color='gray'),
                            rx.heading(f"{V7State.deepseek_only_profit_factor:.2f}", size="6", color='orange'),
                            padding="3",
                            border_radius="md",
                            border="1px solid",
                            border_color='orange.200',
                            width="100%",
                        ),
                        spacing="4",
                    ),
                ),

                # Recent trades for deepseek_only
                rx.card(
                    rx.heading("Recent Trades", size="5", margin_bottom="4"),
                    rx.cond(
                        V7State.deepseek_only_trades.length() > 0,
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    rx.table.column_header_cell(""),
                                    rx.table.column_header_cell("Symbol"),
                                    rx.table.column_header_cell("P&L"),
                                    rx.table.column_header_cell("Exit Time"),
                                ),
                            ),
                            rx.table.body(
                                rx.foreach(V7State.deepseek_only_trades, lambda t: rx.table.row(
                                    rx.table.cell(
                                        rx.cond(
                                            t['outcome'] == 'win',
                                            '✅',
                                            rx.cond(t['outcome'] == 'loss', '❌', '➖')
                                        )
                                    ),
                                    rx.table.cell(t['symbol']),
                                    rx.table.cell(
                                        rx.text(
                                            f"{t['pnl_percent']:.2f}%",
                                            color=rx.cond(
                                                t['outcome'] == 'win',
                                                'green',
                                                rx.cond(t['outcome'] == 'loss', 'red', 'gray')
                                            ),
                                        ),
                                        weight='bold'
                                    ),
                                    rx.table.cell(t['exit_time'], size="1"),
                                ))
                            ),
                        ),
                        rx.text("No trades yet for this strategy", size="3", color='gray'),
                    ),
                    margin_top="4",
                ),
            ),

            columns="2",
            spacing="4",
            margin_bottom="6",
        ),

        # Refresh button
        rx.center(
            rx.button(
                "Refresh Data",
                on_click=V7State.fetch_ab_test_data,
                size="3",
            ),
        ),

        max_width="1600px",
        padding="4",
        on_mount=V7State.fetch_ab_test_data,
    )


# Create the Reflex app
app = rx.App()
app.add_page(index, route="/", title="V7 Ultimate Dashboard")
app.add_page(performance, route="/performance", title="Performance Tracking")
app.add_page(ab_test_comparison, route="/ab-test", title="A/B Test Comparison")
