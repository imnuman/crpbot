"""
V7 Ultimate Trading Dashboard - Reflex Version
Real-time WebSocket-based dashboard for V7 signal monitoring
"""

import reflex as rx
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
import json
import pytz

# Import your existing models
import sys
from pathlib import Path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

from libs.db.models import Signal
from libs.config.config import Settings

# Initialize settings
app_config = Settings()

# Timezones
UTC = pytz.UTC
EST = pytz.timezone('America/New_York')


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

    def _get_session(self):
        """Get database session"""
        engine = create_engine(str(app_config.db_url))
        Session = sessionmaker(bind=engine)
        return Session()

    def fetch_signals(self):
        """Fetch latest V7 signals from database"""
        self.is_loading = True

        try:
            session = self._get_session()

            # Fetch last 2 hours of signals (use naive datetime for SQLite)
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

                # Convert timestamp to EST for display
                # Timestamps in DB are naive UTC (stored without timezone info)
                if s.timestamp.tzinfo is None:
                    # Naive datetime - treat as UTC, then convert to EST
                    ts_utc = UTC.localize(s.timestamp)
                    ts_est = ts_utc.astimezone(EST)
                else:
                    # Already aware - just convert to EST
                    ts_est = s.timestamp.astimezone(EST)

                signals_data.append({
                    'timestamp': ts_est.strftime('%b %d %H:%M EST'),
                    'symbol': s.symbol,
                    'direction': s.direction.upper(),
                    'confidence': f"{(s.confidence or 0.0) * 100:.1f}%",
                    'confidence_num': (s.confidence or 0.0) * 100,
                    'entry_price': f"${s.entry_price:,.2f}" if s.entry_price else 'N/A',
                    'reasoning': reasoning,
                    'tier': s.tier or 'low'
                })

            avg_conf = (total_confidence / len(signals_query) * 100) if signals_query else 0.0

            session.close()

            # Update state (triggers WebSocket push to frontend)
            self.signals = signals_data
            self.signal_count = len(signals_query)
            self.buy_count = buy_count
            self.sell_count = sell_count
            self.hold_count = hold_count
            self.avg_confidence = avg_conf
            # Get current time in EST
            now_est = datetime.now(EST)
            self.last_update = now_est.strftime('%H:%M:%S EST')
            self.is_loading = False

        except Exception as e:
            print(f"Error fetching signals: {e}")
            self.is_loading = False

    def fetch_market_prices(self):
        """Fetch live market prices from Coinbase API"""
        try:
            from coinbase.rest import RESTClient
            import os

            # Get Coinbase API credentials
            api_key = os.getenv('COINBASE_API_KEY_NAME')
            api_secret = os.getenv('COINBASE_API_PRIVATE_KEY')

            if not api_key or not api_secret:
                print("Coinbase API credentials not configured")
                return

            client = RESTClient(api_key=api_key, api_secret=api_secret)

            # Fetch live prices for each symbol
            for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                try:
                    product = client.get_product(symbol)
                    price = float(product['price'])

                    if symbol == 'BTC-USD':
                        self.btc_price = price
                    elif symbol == 'ETH-USD':
                        self.eth_price = price
                    elif symbol == 'SOL-USD':
                        self.sol_price = price
                except Exception as symbol_error:
                    print(f"Error fetching {symbol}: {symbol_error}")

        except Exception as e:
            print(f"Error fetching market prices: {e}")

    def on_load(self):
        """Called when page loads - trigger data fetch"""
        # Call methods directly to update state
        self.fetch_signals()
        self.fetch_market_prices()


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
        rx.text(
            f"Real-time signal monitoring with WebSocket updates",
            size="3",
            color='gray',
            margin_bottom="6"
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


# Create the Reflex app
app = rx.App()
app.add_page(index, route="/", title="V7 Ultimate Dashboard")
