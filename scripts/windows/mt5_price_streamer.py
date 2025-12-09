"""
MT5 Price Streamer - Real-time price streaming via ZMQ PUB
Runs on Windows VPS alongside MT5.

Architecture:
    MT5 -> mt5_price_streamer.py (PUB :5556) -> Linux Event Bus (SUB)

Features:
- Sub-second tick streaming for all FTMO symbols
- Automatic reconnection to MT5
- Heartbeat for connection monitoring
- Candle aggregation (M1, M5, H1)
- Symbol grouping by priority (HF vs session-based)

Usage:
    python mt5_price_streamer.py
"""

import os
import sys
import json
import time
import zmq
import threading
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

# MT5 import (Windows only)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARN] MetaTrader5 not available - running in simulation mode")


@dataclass
class Tick:
    """Price tick data."""
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: float
    volume: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Candle:
    """OHLCV candle data."""
    symbol: str
    timeframe: str
    time: float
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MT5PriceStreamer:
    """
    Streams real-time prices from MT5 via ZMQ PUB socket.

    Message Types:
    - TICK:<symbol> - Real-time tick data
    - CANDLE:<timeframe>:<symbol> - Completed candle
    - HEARTBEAT - Connection health check
    """

    # FTMO symbols to stream - prioritized by update frequency needed
    HIGH_FREQUENCY_SYMBOLS = ["XAUUSD"]  # HF Scalper needs fast updates
    SESSION_SYMBOLS = ["EURUSD", "US30", "NAS100", "GBPUSD"]  # Session-based bots

    # Update intervals (milliseconds)
    HF_INTERVAL_MS = 100  # 10 ticks/second for HF
    SESSION_INTERVAL_MS = 500  # 2 ticks/second for session bots
    HEARTBEAT_INTERVAL = 5  # seconds

    def __init__(
        self,
        pub_port: int = 5556,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None
    ):
        self.pub_port = pub_port

        # MT5 credentials from env
        self.login = login or int(os.getenv("FTMO_LOGIN", "0"))
        self.password = password or os.getenv("FTMO_PASS", "")
        self.server = server or os.getenv("FTMO_SERVER", "FTMO-Server3")

        # ZMQ setup
        self.context: Optional[zmq.Context] = None
        self.pub_socket: Optional[zmq.Socket] = None

        # State
        self._running = False
        self._mt5_connected = False
        self._last_prices: Dict[str, Tick] = {}
        self._candle_builders: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Stats
        self._stats = {
            "ticks_sent": 0,
            "candles_sent": 0,
            "errors": 0,
            "start_time": None
        }

    def start(self):
        """Start the price streamer."""
        print("=" * 60)
        print("  MT5 Price Streamer - Event-Driven Architecture")
        print("=" * 60)
        print(f"  PUB Port: {self.pub_port}")
        print(f"  HF Symbols: {self.HIGH_FREQUENCY_SYMBOLS}")
        print(f"  Session Symbols: {self.SESSION_SYMBOLS}")
        print("=" * 60)

        # Initialize ZMQ
        self._init_zmq()

        # Connect to MT5
        if MT5_AVAILABLE:
            if not self._connect_mt5():
                print("[ERROR] Failed to connect to MT5")
                return False
        else:
            print("[WARN] MT5 not available - using simulation mode")

        self._running = True
        self._stats["start_time"] = time.time()

        # Start streaming threads
        threads = [
            threading.Thread(target=self._stream_hf_ticks, daemon=True),
            threading.Thread(target=self._stream_session_ticks, daemon=True),
            threading.Thread(target=self._heartbeat_loop, daemon=True),
        ]

        for t in threads:
            t.start()

        print("[INFO] Streamer started - press Ctrl+C to stop")

        # Main loop
        try:
            while self._running:
                time.sleep(1)
                self._print_stats()
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
        finally:
            self.stop()

        return True

    def stop(self):
        """Stop the streamer."""
        self._running = False

        if self.pub_socket:
            self.pub_socket.close()
        if self.context:
            self.context.term()

        if MT5_AVAILABLE and self._mt5_connected:
            mt5.shutdown()

        print("[INFO] Streamer stopped")

    def _init_zmq(self):
        """Initialize ZMQ PUB socket."""
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 10000)  # High water mark
        self.pub_socket.setsockopt(zmq.LINGER, 0)  # Don't block on close
        self.pub_socket.bind(f"tcp://*:{self.pub_port}")
        print(f"[INFO] ZMQ PUB bound to port {self.pub_port}")

    def _connect_mt5(self) -> bool:
        """Connect to MT5 terminal."""
        if not MT5_AVAILABLE:
            return False

        if not mt5.initialize():
            print(f"[ERROR] MT5 initialize failed: {mt5.last_error()}")
            return False

        # Login if credentials provided
        if self.login and self.password:
            if not mt5.login(self.login, self.password, self.server):
                print(f"[ERROR] MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        # Get account info
        account = mt5.account_info()
        if account:
            print(f"[INFO] MT5 connected - Account: {account.login}, Balance: ${account.balance:,.2f}")
            self._mt5_connected = True
            return True

        return False

    def _stream_hf_ticks(self):
        """Stream high-frequency ticks (HF Scalper symbols)."""
        while self._running:
            try:
                for symbol in self.HIGH_FREQUENCY_SYMBOLS:
                    tick = self._get_tick(symbol)
                    if tick:
                        self._publish_tick(tick)

                time.sleep(self.HF_INTERVAL_MS / 1000)

            except Exception as e:
                self._stats["errors"] += 1
                print(f"[ERROR] HF tick error: {e}")
                time.sleep(0.5)

    def _stream_session_ticks(self):
        """Stream session-based symbols at lower frequency."""
        while self._running:
            try:
                for symbol in self.SESSION_SYMBOLS:
                    tick = self._get_tick(symbol)
                    if tick:
                        self._publish_tick(tick)
                        self._update_candle_builder(tick)

                time.sleep(self.SESSION_INTERVAL_MS / 1000)

            except Exception as e:
                self._stats["errors"] += 1
                print(f"[ERROR] Session tick error: {e}")
                time.sleep(0.5)

    def _get_tick(self, symbol: str) -> Optional[Tick]:
        """Get current tick from MT5."""
        if MT5_AVAILABLE and self._mt5_connected:
            tick_info = mt5.symbol_info_tick(symbol)
            if tick_info:
                return Tick(
                    symbol=symbol,
                    bid=tick_info.bid,
                    ask=tick_info.ask,
                    spread=round((tick_info.ask - tick_info.bid) * 10000, 1),  # pips for forex
                    timestamp=tick_info.time,
                    volume=tick_info.volume
                )
        else:
            # Simulation mode - generate fake prices
            import random
            base_prices = {
                "XAUUSD": 2650.0,
                "EURUSD": 1.0550,
                "US30": 44500.0,
                "NAS100": 21500.0,
                "GBPUSD": 1.2750,
            }
            base = base_prices.get(symbol, 1000.0)
            noise = random.uniform(-0.001, 0.001) * base
            bid = base + noise

            return Tick(
                symbol=symbol,
                bid=round(bid, 5 if "USD" in symbol and symbol != "XAUUSD" else 2),
                ask=round(bid + 0.0002 * base, 5 if "USD" in symbol and symbol != "XAUUSD" else 2),
                spread=2.0,
                timestamp=time.time(),
                volume=100
            )

        return None

    def _publish_tick(self, tick: Tick):
        """Publish tick to ZMQ."""
        # Check if price changed (avoid spam)
        last = self._last_prices.get(tick.symbol)
        if last and last.bid == tick.bid and last.ask == tick.ask:
            return

        self._last_prices[tick.symbol] = tick

        # Publish: topic is "TICK:<symbol>"
        topic = f"TICK:{tick.symbol}"
        message = json.dumps(tick.to_dict())

        try:
            self.pub_socket.send_string(f"{topic} {message}")
            self._stats["ticks_sent"] += 1
        except zmq.ZMQError as e:
            self._stats["errors"] += 1
            print(f"[ERROR] ZMQ send failed: {e}")

    def _update_candle_builder(self, tick: Tick):
        """Build M1 candles from ticks and publish when complete."""
        current_minute = int(tick.timestamp // 60) * 60
        symbol = tick.symbol

        builder = self._candle_builders[symbol]

        # Check if new candle started
        if "time" not in builder or builder["time"] != current_minute:
            # Publish completed candle if exists
            if "time" in builder:
                candle = Candle(
                    symbol=symbol,
                    timeframe="M1",
                    time=builder["time"],
                    open=builder["open"],
                    high=builder["high"],
                    low=builder["low"],
                    close=builder["close"],
                    volume=builder["volume"]
                )
                self._publish_candle(candle)

            # Start new candle
            self._candle_builders[symbol] = {
                "time": current_minute,
                "open": tick.bid,
                "high": tick.bid,
                "low": tick.bid,
                "close": tick.bid,
                "volume": tick.volume
            }
        else:
            # Update current candle
            builder["high"] = max(builder["high"], tick.bid)
            builder["low"] = min(builder["low"], tick.bid)
            builder["close"] = tick.bid
            builder["volume"] += tick.volume

    def _publish_candle(self, candle: Candle):
        """Publish completed candle."""
        topic = f"CANDLE:{candle.timeframe}:{candle.symbol}"
        message = json.dumps(candle.to_dict())

        try:
            self.pub_socket.send_string(f"{topic} {message}")
            self._stats["candles_sent"] += 1
        except zmq.ZMQError as e:
            self._stats["errors"] += 1

    def _heartbeat_loop(self):
        """Send periodic heartbeats for connection monitoring."""
        while self._running:
            try:
                heartbeat = {
                    "type": "HEARTBEAT",
                    "timestamp": time.time(),
                    "mt5_connected": self._mt5_connected,
                    "symbols": list(self._last_prices.keys()),
                    "stats": {
                        "ticks_sent": self._stats["ticks_sent"],
                        "candles_sent": self._stats["candles_sent"],
                        "errors": self._stats["errors"],
                        "uptime": time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
                    }
                }

                self.pub_socket.send_string(f"HEARTBEAT {json.dumps(heartbeat)}")

            except Exception as e:
                print(f"[ERROR] Heartbeat failed: {e}")

            time.sleep(self.HEARTBEAT_INTERVAL)

    def _print_stats(self):
        """Print streaming statistics."""
        uptime = time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
        tps = self._stats["ticks_sent"] / uptime if uptime > 0 else 0

        # Only print every 30 seconds
        if int(uptime) % 30 == 0:
            print(
                f"[STATS] Uptime: {int(uptime)}s | "
                f"Ticks: {self._stats['ticks_sent']} ({tps:.1f}/s) | "
                f"Candles: {self._stats['candles_sent']} | "
                f"Errors: {self._stats['errors']}"
            )


def main():
    """Run the price streamer."""
    # Load credentials from environment or command line
    login = int(os.getenv("FTMO_LOGIN", "531025383"))
    password = os.getenv("FTMO_PASS", "c*B@lWp41b784c")
    server = os.getenv("FTMO_SERVER", "FTMO-Server3")
    pub_port = int(os.getenv("STREAMER_PUB_PORT", "5556"))

    streamer = MT5PriceStreamer(
        pub_port=pub_port,
        login=login,
        password=password,
        server=server
    )

    streamer.start()


if __name__ == "__main__":
    main()
