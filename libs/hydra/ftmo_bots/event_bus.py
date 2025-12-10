"""
FTMO Event Bus - Real-time price event dispatcher

Architecture:
    Windows VPS (PUB :5556) -> SSH Tunnel -> Linux Event Bus (SUB) -> Bots

Features:
- ZMQ SUB socket for receiving price ticks
- Event dispatcher to registered bot handlers
- Prometheus metrics for monitoring
- SSH tunnel management for secure connection
- Automatic reconnection with exponential backoff
- Tick buffering for burst handling

Usage:
    from libs.hydra.ftmo_bots.event_bus import FTMOEventBus, TickEvent

    bus = FTMOEventBus()
    bus.subscribe("XAUUSD", my_bot.on_tick)
    bus.start()
"""

import os
import json
import time
import zmq
import threading
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
from loguru import logger

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class TickEvent:
    """Price tick event."""
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: float
    volume: int = 0

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TickEvent":
        return cls(
            symbol=data["symbol"],
            bid=data["bid"],
            ask=data["ask"],
            spread=data["spread"],
            timestamp=data["timestamp"],
            volume=data.get("volume", 0)
        )


@dataclass
class CandleEvent:
    """Completed candle event."""
    symbol: str
    timeframe: str
    time: float
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandleEvent":
        return cls(
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            time=data["time"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"]
        )


@dataclass
class HeartbeatEvent:
    """Heartbeat from streamer."""
    timestamp: float
    mt5_connected: bool
    symbols: List[str]
    stats: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeartbeatEvent":
        return cls(
            timestamp=data["timestamp"],
            mt5_connected=data["mt5_connected"],
            symbols=data["symbols"],
            stats=data.get("stats", {})
        )


# Type aliases for callbacks
TickHandler = Callable[[TickEvent], None]
CandleHandler = Callable[[CandleEvent], None]
HeartbeatHandler = Callable[[HeartbeatEvent], None]


class SSHTunnel:
    """SSH tunnel for ZMQ connection to Windows VPS."""

    def __init__(
        self,
        remote_host: str = "45.82.167.195",
        remote_port: int = 5556,
        local_port: int = 5556,
        username: str = "trader",
        password: str = "80B#^yOr2b5s"
    ):
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.username = username
        self.password = password
        self._process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Start SSH tunnel."""
        if self._process and self._process.poll() is None:
            return True

        cmd = [
            "sshpass", "-p", self.password,
            "ssh", "-N", "-L",
            f"{self.local_port}:127.0.0.1:{self.remote_port}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            f"{self.username}@{self.remote_host}"
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)  # Wait for tunnel to establish

            if self._process.poll() is None:
                logger.info(f"[EventBus] SSH tunnel started: localhost:{self.local_port} -> {self.remote_host}:{self.remote_port}")
                return True
            else:
                logger.error("[EventBus] SSH tunnel failed to start")
                return False

        except Exception as e:
            logger.error(f"[EventBus] SSH tunnel error: {e}")
            return False

    def stop(self):
        """Stop SSH tunnel."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
            logger.info("[EventBus] SSH tunnel stopped")

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None


class FTMOEventBus:
    """
    Event bus for real-time FTMO price streaming.

    Subscribes to ZMQ PUB socket and dispatches events to registered handlers.
    """

    # Default Windows VPS configuration
    DEFAULT_HOST = os.getenv("STREAMER_HOST", "45.82.167.195")
    DEFAULT_PUB_PORT = int(os.getenv("STREAMER_PUB_PORT", "5556"))

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_ssh_tunnel: bool = True
    ):
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PUB_PORT
        self.use_ssh_tunnel = use_ssh_tunnel

        # ZMQ setup
        self.context: Optional[zmq.Context] = None
        self.sub_socket: Optional[zmq.Socket] = None
        self._tunnel: Optional[SSHTunnel] = None

        # Event handlers
        self._tick_handlers: Dict[str, List[TickHandler]] = defaultdict(list)
        self._candle_handlers: Dict[str, List[CandleHandler]] = defaultdict(list)
        self._heartbeat_handlers: List[HeartbeatHandler] = []

        # State
        self._running = False
        self._connected = False
        self._last_heartbeat: Optional[float] = None
        self._last_tick_time: Dict[str, float] = {}

        # Tick buffer for burst handling
        self._tick_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Stats
        self._stats = {
            "ticks_received": 0,
            "candles_received": 0,
            "heartbeats_received": 0,
            "handler_errors": 0,
            "reconnects": 0,
            "start_time": None
        }

        # Reconnection limits to prevent infinite loops
        self._consecutive_failures = 0
        self.MAX_RECONNECT_ATTEMPTS = 10  # Stop after 10 consecutive failures

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Thread lock
        self._lock = threading.Lock()

        logger.info(f"[EventBus] Initialized (host={self.host}, port={self.port}, tunnel={use_ssh_tunnel})")

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics - use shared FTMOMetrics singleton."""
        # Use the shared metrics singleton to avoid duplicate registration
        from .metrics import get_ftmo_metrics
        self._shared_metrics = get_ftmo_metrics()
        # No need to create separate metrics - use the singleton

    def subscribe_tick(self, symbol: str, handler: TickHandler):
        """Subscribe to tick events for a symbol."""
        with self._lock:
            self._tick_handlers[symbol].append(handler)
            logger.info(f"[EventBus] Subscribed tick handler for {symbol}")

            # Metrics handled by shared singleton

    def subscribe_candle(self, symbol: str, handler: CandleHandler):
        """Subscribe to candle events for a symbol."""
        with self._lock:
            self._candle_handlers[symbol].append(handler)
            logger.info(f"[EventBus] Subscribed candle handler for {symbol}")

    def subscribe_heartbeat(self, handler: HeartbeatHandler):
        """Subscribe to heartbeat events."""
        with self._lock:
            self._heartbeat_handlers.append(handler)

    def subscribe_all_ticks(self, handler: TickHandler):
        """Subscribe to all tick events (any symbol)."""
        with self._lock:
            self._tick_handlers["*"].append(handler)
            logger.info("[EventBus] Subscribed to all ticks")

    def start(self) -> bool:
        """Start the event bus."""
        if self._running:
            return True

        logger.info("[EventBus] Starting event bus...")

        # Start SSH tunnel if needed
        if self.use_ssh_tunnel:
            self._tunnel = SSHTunnel(
                remote_host=self.host,
                remote_port=self.port,
                local_port=self.port
            )
            if not self._tunnel.start():
                logger.error("[EventBus] Failed to start SSH tunnel")
                return False

        # Connect ZMQ
        if not self._connect():
            return False

        self._running = True
        self._stats["start_time"] = time.time()

        # Start receiver thread
        self._receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receiver_thread.start()

        # Start health monitor thread
        self._health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self._health_thread.start()

        logger.info("[EventBus] Event bus started")
        return True

    def stop(self):
        """Stop the event bus."""
        with self._lock:
            self._running = False

            if self.sub_socket:
                try:
                    self.sub_socket.close()
                except Exception:
                    pass
                self.sub_socket = None

            if self.context:
                try:
                    self.context.term()
                except Exception:
                    pass
                self.context = None

            if self._tunnel:
                self._tunnel.stop()

        logger.info("[EventBus] Event bus stopped")

    def _connect(self) -> bool:
        """Connect to ZMQ PUB socket."""
        try:
            self.context = zmq.Context()
            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
            self.sub_socket.setsockopt(zmq.RCVHWM, 10000)

            # Subscribe to all topics
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "TICK:")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "CANDLE:")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "HEARTBEAT")

            # Connect (to tunnel or direct)
            connect_addr = f"tcp://127.0.0.1:{self.port}" if self.use_ssh_tunnel else f"tcp://{self.host}:{self.port}"
            self.sub_socket.connect(connect_addr)

            self._connected = True
            logger.info(f"[EventBus] Connected to {connect_addr}")

            if PROMETHEUS_AVAILABLE:
                self._shared_metrics.set_connection_status("event_bus", True)

            return True

        except Exception as e:
            logger.error(f"[EventBus] Connection failed: {e}")
            self._connected = False

            if PROMETHEUS_AVAILABLE:
                self._shared_metrics.set_connection_status("event_bus", False)

            return False

    def _receive_loop(self):
        """Main receive loop for ZMQ messages."""
        backoff = 1

        while self._running:
            try:
                message = self.sub_socket.recv_string()
                backoff = 1  # Reset backoff on successful receive
                self._reset_failure_counter()  # Reset failure counter on successful receive

                # Parse topic and payload
                parts = message.split(" ", 1)
                if len(parts) != 2:
                    continue

                topic, payload = parts
                data = json.loads(payload)

                # Route by topic type
                if topic.startswith("TICK:"):
                    # Safe split with bounds check to prevent IndexError
                    tick_parts = topic.split(":")
                    if len(tick_parts) >= 2:
                        symbol = tick_parts[1]
                        self._handle_tick(symbol, data)
                    else:
                        logger.warning(f"[EventBus] Malformed TICK topic: {topic}")

                elif topic.startswith("CANDLE:"):
                    # Safe split with bounds check to prevent IndexError
                    candle_parts = topic.split(":")
                    if len(candle_parts) >= 3:
                        timeframe = candle_parts[1]
                        symbol = candle_parts[2]
                        self._handle_candle(symbol, timeframe, data)
                    else:
                        logger.warning(f"[EventBus] Malformed CANDLE topic: {topic}")

                elif topic == "HEARTBEAT":
                    self._handle_heartbeat(data)

            except zmq.Again:
                # Timeout - check connection health
                if self._last_heartbeat:
                    age = time.time() - self._last_heartbeat
                    if age > 30:
                        logger.warning(f"[EventBus] No heartbeat for {age:.1f}s")

            except zmq.ZMQError as e:
                logger.error(f"[EventBus] ZMQ error: {e}")
                if not self._reconnect(backoff):
                    break  # Max retries exceeded, stop the loop
                backoff = min(backoff * 2, 60)

            except Exception as e:
                logger.error(f"[EventBus] Receive error: {e}")
                self._stats["handler_errors"] += 1

    def _handle_tick(self, symbol: str, data: Dict[str, Any]):
        """Handle incoming tick event."""
        try:
            event = TickEvent.from_dict(data)
            self._stats["ticks_received"] += 1
            self._last_tick_time[symbol] = time.time()

            # Store in buffer
            self._tick_buffer[symbol].append(event)

            # Calculate and record lag
            if PROMETHEUS_AVAILABLE:
                lag = time.time() - event.timestamp
                self._shared_metrics.record_tick(symbol, event.bid, event.ask, lag)

            # Dispatch to symbol-specific handlers
            handlers = self._tick_handlers.get(symbol, []) + self._tick_handlers.get("*", [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"[EventBus] Tick handler error for {symbol}: {e}")
                    self._stats["handler_errors"] += 1
                    if PROMETHEUS_AVAILABLE:
                        self._shared_metrics.record_handler_error("tick", "event_bus")

        except Exception as e:
            logger.error(f"[EventBus] Tick parse error: {e}")

    def _handle_candle(self, symbol: str, timeframe: str, data: Dict[str, Any]):
        """Handle incoming candle event."""
        try:
            event = CandleEvent.from_dict(data)
            self._stats["candles_received"] += 1

            handlers = self._candle_handlers.get(symbol, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"[EventBus] Candle handler error for {symbol}: {e}")
                    self._stats["handler_errors"] += 1

        except Exception as e:
            logger.error(f"[EventBus] Candle parse error: {e}")

    def _handle_heartbeat(self, data: Dict[str, Any]):
        """Handle incoming heartbeat."""
        try:
            event = HeartbeatEvent.from_dict(data)
            self._stats["heartbeats_received"] += 1
            self._last_heartbeat = time.time()

            if PROMETHEUS_AVAILABLE:
                pass  # Heartbeat age handled internally

            for handler in self._heartbeat_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"[EventBus] Heartbeat handler error: {e}")

        except Exception as e:
            logger.error(f"[EventBus] Heartbeat parse error: {e}")

    def _health_monitor(self):
        """Monitor connection health and reconnect if needed."""
        while self._running:
            try:
                time.sleep(10)

                # Update heartbeat age metric
                if self._last_heartbeat and PROMETHEUS_AVAILABLE:
                    age = time.time() - self._last_heartbeat
                    pass  # Heartbeat age tracked internally

                # Check tunnel health
                if self.use_ssh_tunnel and self._tunnel and not self._tunnel.is_alive:
                    logger.warning("[EventBus] SSH tunnel died, reconnecting...")
                    self._reconnect(1)

            except Exception as e:
                logger.error(f"[EventBus] Health monitor error: {e}")

    def _reconnect(self, delay: float) -> bool:
        """Reconnect after failure. Returns False if max retries exceeded."""
        self._consecutive_failures += 1

        if self._consecutive_failures > self.MAX_RECONNECT_ATTEMPTS:
            logger.error(
                f"[EventBus] Max reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) exceeded. "
                "Stopping event bus to prevent infinite loop. Manual restart required."
            )
            self._running = False
            return False

        logger.info(
            f"[EventBus] Reconnecting in {delay}s... "
            f"(attempt {self._consecutive_failures}/{self.MAX_RECONNECT_ATTEMPTS})"
        )
        time.sleep(delay)

        with self._lock:
            self._stats["reconnects"] += 1

            # Close existing connection
            if self.sub_socket:
                try:
                    self.sub_socket.close()
                except Exception:
                    pass
                self.sub_socket = None

            if self.context:
                try:
                    self.context.term()
                except Exception:
                    pass
                self.context = None

            # Restart tunnel if needed
            if self.use_ssh_tunnel and self._tunnel:
                self._tunnel.stop()
                self._tunnel.start()

            # Reconnect
            self._connect()
        return True

    def _reset_failure_counter(self):
        """Reset consecutive failure counter after successful operation."""
        if self._consecutive_failures > 0:
            logger.debug(f"[EventBus] Resetting failure counter (was {self._consecutive_failures})")
            self._consecutive_failures = 0

    def get_latest_tick(self, symbol: str) -> Optional[TickEvent]:
        """Get most recent tick for a symbol."""
        buffer = self._tick_buffer.get(symbol)
        if buffer and len(buffer) > 0:
            return buffer[-1]
        return None

    def get_tick_history(self, symbol: str, count: int = 100) -> List[TickEvent]:
        """Get recent tick history for a symbol."""
        buffer = self._tick_buffer.get(symbol)
        if buffer:
            return list(buffer)[-count:]
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        uptime = time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
        tps = self._stats["ticks_received"] / uptime if uptime > 0 else 0

        return {
            "connected": self._connected,
            "uptime_seconds": uptime,
            "ticks_received": self._stats["ticks_received"],
            "ticks_per_second": round(tps, 2),
            "candles_received": self._stats["candles_received"],
            "heartbeats_received": self._stats["heartbeats_received"],
            "handler_errors": self._stats["handler_errors"],
            "reconnects": self._stats["reconnects"],
            "subscribed_symbols": list(self._tick_handlers.keys()),
            "last_heartbeat_age": time.time() - self._last_heartbeat if self._last_heartbeat else None,
            "last_tick_times": {
                symbol: time.time() - ts
                for symbol, ts in self._last_tick_time.items()
            }
        }

    @property
    def is_connected(self) -> bool:
        """Check if connected to price stream."""
        return self._connected and self._running


# Singleton instance
_event_bus: Optional[FTMOEventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> FTMOEventBus:
    """Get singleton event bus instance."""
    global _event_bus
    if _event_bus is None:
        with _event_bus_lock:
            if _event_bus is None:
                _event_bus = FTMOEventBus()
    return _event_bus
