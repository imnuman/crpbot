"""
Coinbase Advanced Trade WebSocket Client
Real-time ticker and order book data for crypto markets

This client uses Coinbase WebSocket API (free, no authentication) to fetch:
- Real-time ticker updates (price, volume, best bid/ask)
- Level 2 order book snapshots
- Trade flow analysis (buy/sell pressure)
- Spread and liquidity metrics
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class CoinbaseWebSocketClient:
    """Client for Coinbase WebSocket (real-time ticker + order book)"""

    # Reconnection settings
    MAX_RECONNECT_ATTEMPTS = 10
    BASE_RECONNECT_DELAY = 2  # seconds
    MAX_RECONNECT_DELAY = 300  # 5 minutes max

    def __init__(self, symbols: List[str] = None):
        """
        Initialize Coinbase WebSocket client

        Args:
            symbols: List of trading pairs (e.g., ['BTC-USD', 'ETH-USD', 'SOL-USD'])
        """
        self.symbols = symbols or ['BTC-USD', 'ETH-USD', 'SOL-USD']
        self.ws_url = "wss://ws-feed.exchange.coinbase.com"

        # Internal state
        self.websocket = None
        self.running = False
        self.connected = False

        # Reconnection state
        self._reconnect_attempts = 0
        self._last_message_time = None

        # Data storage (last 100 ticks per symbol)
        self.tickers: Dict[str, deque] = {symbol: deque(maxlen=100) for symbol in self.symbols}
        self.order_books: Dict[str, Dict] = {}

        # Callbacks for real-time data
        self.on_ticker_callback: Optional[Callable] = None
        self.on_orderbook_callback: Optional[Callable] = None
        self.on_reconnect_callback: Optional[Callable] = None

        logger.info(f"CoinbaseWebSocket initialized for {len(self.symbols)} symbols")

    async def connect(self):
        """Establish WebSocket connection and subscribe to channels"""
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5
            )
            self.running = True
            self.connected = True
            self._last_message_time = datetime.now()
            logger.info(f"Connected to {self.ws_url}")

            # Subscribe to ticker and level2 channels
            subscribe_message = {
                "type": "subscribe",
                "product_ids": self.symbols,
                "channels": [
                    "ticker",      # Real-time ticker updates
                    "level2"       # Order book snapshots
                ]
            }

            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to channels: ticker, level2 for {self.symbols}")

            # Reset reconnection counter on successful connection
            self._reconnect_attempts = 0

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.running = False
            self.connected = False
            raise

    async def listen(self):
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                self._last_message_time = datetime.now()
                await self._handle_message(json.loads(message))
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self.connected = False
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            self.connected = False

    async def _reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection successful, False if max attempts exceeded
        """
        while self._reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1

            # Calculate delay with exponential backoff (capped)
            delay = min(
                self.BASE_RECONNECT_DELAY * (2 ** (self._reconnect_attempts - 1)),
                self.MAX_RECONNECT_DELAY
            )

            logger.warning(
                f"[WebSocket] Reconnection attempt {self._reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS} "
                f"in {delay}s..."
            )
            await asyncio.sleep(delay)

            try:
                # Close existing connection if any
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except Exception:
                        pass

                # Attempt new connection
                await self.connect()

                logger.info(f"[WebSocket] Reconnected successfully after {self._reconnect_attempts} attempts")

                # Execute callback if set
                if self.on_reconnect_callback:
                    await self.on_reconnect_callback()

                return True

            except Exception as e:
                logger.error(f"[WebSocket] Reconnection attempt {self._reconnect_attempts} failed: {e}")

        logger.critical(
            f"[WebSocket] Failed to reconnect after {self.MAX_RECONNECT_ATTEMPTS} attempts. "
            "Data feed is OFFLINE. Manual intervention required."
        )
        self.running = False
        return False

    async def _handle_message(self, data: dict):
        """Process incoming WebSocket message"""
        msg_type = data.get('type')

        if msg_type == 'ticker':
            await self._handle_ticker(data)
        elif msg_type == 'snapshot':
            await self._handle_snapshot(data)
        elif msg_type == 'l2update':
            await self._handle_l2update(data)
        elif msg_type == 'subscriptions':
            logger.debug(f"Subscription confirmed: {data.get('channels')}")
        elif msg_type == 'error':
            logger.error(f"WebSocket error: {data.get('message')}")

    async def _handle_ticker(self, data: dict):
        """Handle ticker update message"""
        product_id = data.get('product_id')
        if product_id not in self.symbols:
            return

        ticker = {
            'symbol': product_id,
            'price': float(data.get('price', 0)),
            'volume_24h': float(data.get('volume_24h', 0)),
            'best_bid': float(data.get('best_bid', 0)),
            'best_ask': float(data.get('best_ask', 0)),
            'last_size': float(data.get('last_size', 0)),
            'side': data.get('side'),  # 'buy' or 'sell' for last trade
            'time': data.get('time'),
            'timestamp': datetime.now()
        }

        # Calculate spread metrics
        if ticker['best_bid'] > 0 and ticker['best_ask'] > 0:
            ticker['spread'] = ticker['best_ask'] - ticker['best_bid']
            ticker['spread_pct'] = (ticker['spread'] / ticker['price']) * 100
        else:
            ticker['spread'] = 0.0
            ticker['spread_pct'] = 0.0

        # Store ticker
        self.tickers[product_id].append(ticker)

        # Execute callback
        if self.on_ticker_callback:
            await self.on_ticker_callback(ticker)

        logger.debug(
            f"{product_id} ticker: ${ticker['price']:.2f}, "
            f"Spread: ${ticker['spread']:.2f} ({ticker['spread_pct']:.3f}%), "
            f"Side: {ticker['side']}"
        )

    async def _handle_snapshot(self, data: dict):
        """Handle order book snapshot message"""
        product_id = data.get('product_id')
        if product_id not in self.symbols:
            return

        # Store order book
        self.order_books[product_id] = {
            'bids': [[float(price), float(size)] for price, size in data.get('bids', [])[:10]],  # Top 10
            'asks': [[float(price), float(size)] for price, size in data.get('asks', [])[:10]],  # Top 10
            'timestamp': datetime.now()
        }

        # Execute callback
        if self.on_orderbook_callback:
            await self.on_orderbook_callback(product_id, self.order_books[product_id])

        logger.debug(f"{product_id} order book snapshot: {len(self.order_books[product_id]['bids'])} bids, {len(self.order_books[product_id]['asks'])} asks")

    async def _handle_l2update(self, data: dict):
        """Handle order book update message"""
        product_id = data.get('product_id')
        if product_id not in self.symbols or product_id not in self.order_books:
            return

        # Update order book (simplified - just log, full implementation would update bid/ask levels)
        changes = data.get('changes', [])
        logger.debug(f"{product_id} order book update: {len(changes)} changes")

    async def disconnect(self):
        """Close WebSocket connection"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket disconnected")

    def get_latest_ticker(self, symbol: str) -> Optional[Dict]:
        """Get latest ticker for a symbol"""
        if symbol in self.tickers and len(self.tickers[symbol]) > 0:
            return self.tickers[symbol][-1]
        return None

    def get_order_flow_metrics(self, symbol: str, window: int = 20) -> Dict:
        """
        Calculate order flow metrics from recent tickers

        Args:
            symbol: Trading symbol (BTC-USD, ETH-USD, SOL-USD)
            window: Number of recent tickers to analyze

        Returns:
            {
                'buy_pressure': float (0-1),
                'sell_pressure': float (0-1),
                'imbalance': float (-1 to +1, negative = sell pressure, positive = buy pressure),
                'avg_spread_pct': float,
                'whale_activity': bool (large trade detected)
            }
        """
        if symbol not in self.tickers or len(self.tickers[symbol]) == 0:
            return {
                'buy_pressure': 0.5,
                'sell_pressure': 0.5,
                'imbalance': 0.0,
                'avg_spread_pct': 0.0,
                'whale_activity': False
            }

        recent_tickers = list(self.tickers[symbol])[-window:]

        buy_count = sum(1 for t in recent_tickers if t.get('side') == 'buy')
        sell_count = sum(1 for t in recent_tickers if t.get('side') == 'sell')
        total_count = buy_count + sell_count

        if total_count == 0:
            buy_pressure = 0.5
            sell_pressure = 0.5
            imbalance = 0.0
        else:
            buy_pressure = buy_count / total_count
            sell_pressure = sell_count / total_count
            imbalance = (buy_count - sell_count) / total_count

        # Average spread
        spreads = [t['spread_pct'] for t in recent_tickers if t.get('spread_pct', 0) > 0]
        avg_spread_pct = sum(spreads) / len(spreads) if spreads else 0.0

        # Whale activity detection (last trade > 10 BTC/ETH or $100k for SOL)
        latest = recent_tickers[-1]
        whale_threshold = {
            'BTC-USD': 10.0,   # 10 BTC
            'ETH-USD': 10.0,   # 10 ETH
            'SOL-USD': 100000.0 / latest['price']  # $100k worth
        }
        whale_activity = latest.get('last_size', 0) > whale_threshold.get(symbol, 10.0)

        return {
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'imbalance': imbalance,
            'avg_spread_pct': avg_spread_pct,
            'whale_activity': whale_activity
        }

    def get_order_book_depth(self, symbol: str) -> Dict:
        """
        Calculate order book depth metrics

        Returns:
            {
                'bid_depth': float (total volume on bid side, top 10 levels),
                'ask_depth': float (total volume on ask side, top 10 levels),
                'depth_imbalance': float (-1 to +1, negative = more asks, positive = more bids),
                'best_bid': float,
                'best_ask': float
            }
        """
        if symbol not in self.order_books:
            return {
                'bid_depth': 0.0,
                'ask_depth': 0.0,
                'depth_imbalance': 0.0,
                'best_bid': 0.0,
                'best_ask': 0.0
            }

        book = self.order_books[symbol]

        bid_depth = sum(size for price, size in book['bids'])
        ask_depth = sum(size for price, size in book['asks'])
        total_depth = bid_depth + ask_depth

        depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        best_bid = book['bids'][0][0] if book['bids'] else 0.0
        best_ask = book['asks'][0][0] if book['asks'] else 0.0

        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'depth_imbalance': depth_imbalance,
            'best_bid': best_bid,
            'best_ask': best_ask
        }

    async def start(self):
        """Start WebSocket client (async main loop)"""
        await self.connect()
        await self.listen()

    async def run_forever(self):
        """
        Run WebSocket client with auto-reconnect and exponential backoff.

        This method will:
        1. Attempt initial connection
        2. Listen for messages
        3. On disconnect, attempt reconnection with exponential backoff
        4. Continue until MAX_RECONNECT_ATTEMPTS exceeded or manual stop
        """
        self.running = True
        logger.info("[WebSocket] Starting run_forever loop...")

        while self.running:
            try:
                if not self.connected:
                    await self.connect()

                await self.listen()

                # listen() returned - connection was lost
                if self.running and not self.connected:
                    logger.warning("[WebSocket] Connection lost, attempting reconnect...")
                    success = await self._reconnect()
                    if not success:
                        logger.critical("[WebSocket] Reconnection failed, exiting run_forever")
                        break

            except Exception as e:
                logger.error(f"[WebSocket] Error in run_forever: {e}")

                if self.running:
                    success = await self._reconnect()
                    if not success:
                        logger.critical("[WebSocket] Reconnection failed, exiting run_forever")
                        break

        logger.info("[WebSocket] run_forever loop ended")

    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected and receiving data."""
        if not self.connected or not self.websocket:
            return False

        # Check if we've received a message in the last 60 seconds
        if self._last_message_time:
            time_since_last = (datetime.now() - self._last_message_time).total_seconds()
            if time_since_last > 60:
                logger.warning(f"[WebSocket] No message received in {time_since_last:.0f}s - connection may be stale")
                return False

        return True


# Convenience function for V7 runtime
async def get_coinbase_realtime_data(symbol: str, duration: int = 30) -> Dict:
    """
    Get real-time Coinbase data for a symbol (runs for specified duration)

    Args:
        symbol: Trading symbol (BTC-USD, ETH-USD, SOL-USD)
        duration: Duration in seconds to collect data

    Returns:
        {
            'order_flow': {buy_pressure, sell_pressure, imbalance, spread, whale_activity},
            'order_book': {bid_depth, ask_depth, depth_imbalance, best_bid, best_ask},
            'latest_ticker': {price, volume, spread, etc.}
        }
    """
    client = CoinbaseWebSocketClient(symbols=[symbol])

    try:
        # Start connection
        await client.connect()

        # Listen for specified duration
        listen_task = asyncio.create_task(client.listen())
        await asyncio.sleep(duration)

        # Stop listening
        listen_task.cancel()
        await client.disconnect()

        # Collect metrics
        return {
            'order_flow': client.get_order_flow_metrics(symbol),
            'order_book': client.get_order_book_depth(symbol),
            'latest_ticker': client.get_latest_ticker(symbol),
            'timestamp': datetime.now()
        }

    except Exception as e:
        logger.error(f"Failed to get real-time data: {e}")
        await client.disconnect()
        return {
            'order_flow': {},
            'order_book': {},
            'latest_ticker': None,
            'timestamp': datetime.now(),
            'error': str(e)
        }


if __name__ == "__main__":
    # Test the WebSocket client
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Coinbase WebSocket Client - Test Run (30 seconds)")
    print("=" * 80)

    async def test_websocket():
        """Test WebSocket client for 30 seconds"""
        client = CoinbaseWebSocketClient(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'])

        # Set up callbacks for real-time display
        async def on_ticker(ticker):
            print(f"\nTICKER: {ticker['symbol']} @ ${ticker['price']:,.2f} | Spread: {ticker['spread_pct']:.3f}% | Side: {ticker['side']}")

        async def on_orderbook(symbol, book):
            print(f"ORDER BOOK: {symbol} | Bids: {len(book['bids'])}, Asks: {len(book['asks'])}")

        client.on_ticker_callback = on_ticker
        client.on_orderbook_callback = on_orderbook

        # Connect and listen for 30 seconds
        await client.connect()
        listen_task = asyncio.create_task(client.listen())
        await asyncio.sleep(30)

        # Stop and analyze
        listen_task.cancel()
        await client.disconnect()

        print("\n" + "=" * 80)
        print("30-Second Analysis Summary")
        print("=" * 80)

        for symbol in client.symbols:
            print(f"\n{symbol}:")

            # Latest ticker
            ticker = client.get_latest_ticker(symbol)
            if ticker:
                print(f"  Latest Price:      ${ticker['price']:,.2f}")
                print(f"  24h Volume:        ${ticker['volume_24h']/1e9:.2f}B")
                print(f"  Spread:            ${ticker['spread']:.2f} ({ticker['spread_pct']:.3f}%)")

            # Order flow
            flow = client.get_order_flow_metrics(symbol)
            print(f"\n  Order Flow (last 20 ticks):")
            print(f"    Buy Pressure:    {flow['buy_pressure']:.1%}")
            print(f"    Sell Pressure:   {flow['sell_pressure']:.1%}")
            print(f"    Imbalance:       {flow['imbalance']:+.2f}")
            print(f"    Avg Spread:      {flow['avg_spread_pct']:.3f}%")
            print(f"    Whale Activity:  {'YES' if flow['whale_activity'] else 'NO'}")

            # Order book
            book_depth = client.get_order_book_depth(symbol)
            print(f"\n  Order Book Depth (top 10 levels):")
            print(f"    Bid Depth:       {book_depth['bid_depth']:.2f} {symbol.split('-')[0]}")
            print(f"    Ask Depth:       {book_depth['ask_depth']:.2f} {symbol.split('-')[0]}")
            print(f"    Depth Imbalance: {book_depth['depth_imbalance']:+.2f}")

        print("\n" + "=" * 80)
        print("Test complete!")
        print("=" * 80)

    # Run test
    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
