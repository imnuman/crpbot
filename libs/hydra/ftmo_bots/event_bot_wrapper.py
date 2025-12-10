"""
Event-Driven Bot Wrapper - Wraps existing FTMO bots with tick handlers

Instead of rewriting all bots, this wrapper:
- Buffers incoming ticks
- Triggers bot analysis on configurable intervals or tick counts
- Handles session windows for session-based bots
- Manages candle building from ticks

Usage:
    from libs.hydra.ftmo_bots.event_bot_wrapper import EventBotWrapper

    wrapper = EventBotWrapper(
        bot=get_gold_london_bot(),
        trigger_mode="session",  # or "tick_count" or "interval"
    )
    event_bus.subscribe_tick("XAUUSD", wrapper.on_tick)
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from loguru import logger

from .event_bus import TickEvent, CandleEvent
from .base_ftmo_bot import BaseFTMOBot, TradeSignal
from .metrics import get_ftmo_metrics


@dataclass
class TriggerConfig:
    """Configuration for when bot analysis is triggered."""
    mode: str = "hybrid"  # "session", "tick_count", "interval", "hybrid"

    # Tick count trigger
    tick_count: int = 100  # Analyze after N ticks

    # Interval trigger
    interval_seconds: float = 60.0  # Analyze every N seconds

    # Session windows (UTC hours)
    session_windows: List[tuple] = field(default_factory=list)

    # Debounce (prevent rapid-fire analysis)
    min_analysis_gap: float = 5.0  # Minimum seconds between analyses


@dataclass
class TickBuffer:
    """Buffer for accumulating ticks."""
    ticks: deque = field(default_factory=lambda: deque(maxlen=1000))
    candles_m1: deque = field(default_factory=lambda: deque(maxlen=100))
    candles_m5: deque = field(default_factory=lambda: deque(maxlen=100))
    last_candle_time: Optional[float] = None

    def add_tick(self, tick: TickEvent):
        self.ticks.append(tick)

    def get_recent_ticks(self, count: int = 100) -> List[TickEvent]:
        return list(self.ticks)[-count:]

    def get_latest_price(self) -> Optional[float]:
        if self.ticks:
            return self.ticks[-1].bid
        return None


class CandleBuilder:
    """Builds candles from tick stream."""

    def __init__(self, timeframe_minutes: int = 1):
        self.timeframe_minutes = timeframe_minutes
        self._current_candle: Optional[Dict[str, float]] = None
        self._completed_candles: deque = deque(maxlen=500)

    def process_tick(self, tick: TickEvent) -> Optional[CandleEvent]:
        """Process a tick and return completed candle if any."""
        candle_time = self._get_candle_time(tick.timestamp)

        if self._current_candle is None:
            self._start_new_candle(tick, candle_time)
            return None

        if self._current_candle["time"] != candle_time:
            # Complete current candle and start new one
            completed = self._complete_candle(tick.symbol)
            self._start_new_candle(tick, candle_time)
            return completed

        # Update current candle
        self._current_candle["high"] = max(self._current_candle["high"], tick.bid)
        self._current_candle["low"] = min(self._current_candle["low"], tick.bid)
        self._current_candle["close"] = tick.bid
        self._current_candle["volume"] += tick.volume

        return None

    def _get_candle_time(self, timestamp: float) -> float:
        """Get candle start time for a timestamp."""
        interval = self.timeframe_minutes * 60
        return (int(timestamp) // interval) * interval

    def _start_new_candle(self, tick: TickEvent, candle_time: float):
        """Start a new candle."""
        self._current_candle = {
            "symbol": tick.symbol,
            "time": candle_time,
            "open": tick.bid,
            "high": tick.bid,
            "low": tick.bid,
            "close": tick.bid,
            "volume": tick.volume
        }

    def _complete_candle(self, symbol: str) -> CandleEvent:
        """Complete current candle and return as event."""
        candle = CandleEvent(
            symbol=symbol,
            timeframe=f"M{self.timeframe_minutes}",
            time=self._current_candle["time"],
            open=self._current_candle["open"],
            high=self._current_candle["high"],
            low=self._current_candle["low"],
            close=self._current_candle["close"],
            volume=self._current_candle["volume"]
        )
        self._completed_candles.append(candle)
        return candle

    def get_candles(self, count: int = 100) -> List[Dict[str, float]]:
        """Get recent completed candles as dicts (for bot compatibility)."""
        return [
            {
                "time": c.time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            }
            for c in list(self._completed_candles)[-count:]
        ]


class EventBotWrapper:
    """
    Wraps existing FTMO bots for event-driven operation.

    Handles:
    - Tick buffering
    - Candle building
    - Analysis triggering based on configurable rules
    - Session window management
    - Trade signal forwarding
    """

    def __init__(
        self,
        bot: BaseFTMOBot,
        trigger_config: Optional[TriggerConfig] = None,
        on_signal: Optional[Callable[[TradeSignal], None]] = None
    ):
        self.bot = bot
        self.config = trigger_config or self._default_trigger_config()
        self.on_signal_callback = on_signal

        # Tick buffer
        self._buffer = TickBuffer()

        # Candle builders
        self._candle_builder_m1 = CandleBuilder(1)
        self._candle_builder_m5 = CandleBuilder(5)

        # State
        self._tick_count = 0
        self._last_analysis_time: float = 0
        self._last_tick_time: float = 0
        self._running = True

        # Metrics
        self._metrics = get_ftmo_metrics()

        # Lock for thread safety
        self._lock = threading.Lock()

        logger.info(
            f"[EventWrapper] Wrapped {bot.config.bot_name} | "
            f"symbol={bot.config.symbol}, trigger={self.config.mode}"
        )

    def _default_trigger_config(self) -> TriggerConfig:
        """Get default trigger config based on bot type."""
        bot_name = self.bot.config.bot_name.lower()

        if "scalper" in bot_name or "hf" in bot_name:
            # High-frequency: analyze on every N ticks
            return TriggerConfig(
                mode="tick_count",
                tick_count=50,
                min_analysis_gap=2.0
            )

        elif "london" in bot_name:
            # London session: 07:00-09:00 UTC
            return TriggerConfig(
                mode="session",
                session_windows=[(7, 9)],
                interval_seconds=30.0,
                min_analysis_gap=30.0
            )

        elif "ny" in bot_name:
            # NY session: 14:00-16:00 UTC
            return TriggerConfig(
                mode="session",
                session_windows=[(14, 16)],
                interval_seconds=30.0,
                min_analysis_gap=30.0
            )

        elif "orb" in bot_name:
            # Opening Range Breakout: 14:30-15:30 UTC
            return TriggerConfig(
                mode="session",
                session_windows=[(14, 16)],
                interval_seconds=30.0,
                min_analysis_gap=30.0
            )

        elif "gap" in bot_name:
            # Gap trading: 13:30-14:30 UTC
            return TriggerConfig(
                mode="session",
                session_windows=[(13, 15)],
                interval_seconds=30.0,
                min_analysis_gap=30.0
            )

        else:
            # Default: hybrid mode
            return TriggerConfig(
                mode="hybrid",
                tick_count=100,
                interval_seconds=60.0,
                min_analysis_gap=10.0
            )

    def on_tick(self, tick: TickEvent):
        """Handle incoming tick event."""
        if not self._running:
            return

        with self._lock:
            # Buffer the tick
            self._buffer.add_tick(tick)
            self._tick_count += 1
            self._last_tick_time = time.time()

            # Build candles
            m1_candle = self._candle_builder_m1.process_tick(tick)
            m5_candle = self._candle_builder_m5.process_tick(tick)

            if m1_candle:
                self._buffer.candles_m1.append(m1_candle)

            # Check if we should analyze
            if self._should_analyze():
                self._run_analysis()

    def on_candle(self, candle: CandleEvent):
        """Handle incoming candle event (alternative to building from ticks)."""
        with self._lock:
            if candle.timeframe == "M1":
                self._buffer.candles_m1.append(candle)
            elif candle.timeframe == "M5":
                self._buffer.candles_m5.append(candle)

    def _should_analyze(self) -> bool:
        """Check if we should trigger bot analysis."""
        now = time.time()

        # Always respect minimum gap
        if now - self._last_analysis_time < self.config.min_analysis_gap:
            return False

        mode = self.config.mode

        if mode == "tick_count":
            return self._tick_count >= self.config.tick_count

        elif mode == "interval":
            return now - self._last_analysis_time >= self.config.interval_seconds

        elif mode == "session":
            if not self._in_session_window():
                return False
            return now - self._last_analysis_time >= self.config.interval_seconds

        elif mode == "hybrid":
            # Either tick count OR interval in session
            if self._tick_count >= self.config.tick_count:
                return True
            if self._in_session_window():
                return now - self._last_analysis_time >= self.config.interval_seconds
            return False

        return False

    def _in_session_window(self) -> bool:
        """Check if current time is within a trading session window."""
        if not self.config.session_windows:
            return True

        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour

        for start_hour, end_hour in self.config.session_windows:
            if start_hour <= current_hour < end_hour:
                return True

        return False

    def _run_analysis(self):
        """Run bot analysis cycle."""
        try:
            start_time = time.time()

            # Build market data from buffers
            candles = self._candle_builder_m1.get_candles(500)

            if len(candles) < 10:
                logger.debug(f"[EventWrapper] {self.bot.config.bot_name}: Not enough candles ({len(candles)})")
                return

            market_data = {"candles": candles}

            # Run bot cycle
            signal = self.bot.run_cycle(market_data)

            # Reset counters
            self._tick_count = 0
            self._last_analysis_time = time.time()

            # Record metrics
            processing_time = time.time() - start_time
            self._metrics.record_event_bus_tick(processing_time)

            if signal:
                logger.info(
                    f"[EventWrapper] {self.bot.config.bot_name} SIGNAL: "
                    f"{signal.direction} {signal.symbol} @ {signal.entry_price}"
                )

                self._metrics.record_signal(
                    self.bot.config.bot_name,
                    signal.symbol,
                    signal.direction
                )

                # Forward signal
                if self.on_signal_callback:
                    self.on_signal_callback(signal)

        except Exception as e:
            logger.error(f"[EventWrapper] {self.bot.config.bot_name} analysis error: {e}")
            self._metrics.record_handler_error("analysis", self.bot.config.bot_name)

    def get_current_price(self) -> Optional[float]:
        """Get current price from buffer."""
        return self._buffer.get_latest_price()

    def get_status(self) -> Dict[str, Any]:
        """Get wrapper status."""
        return {
            "bot_name": self.bot.config.bot_name,
            "symbol": self.bot.config.symbol,
            "trigger_mode": self.config.mode,
            "tick_count": self._tick_count,
            "buffer_size": len(self._buffer.ticks),
            "candles_m1": len(self._buffer.candles_m1),
            "last_analysis_age": time.time() - self._last_analysis_time if self._last_analysis_time else None,
            "in_session": self._in_session_window(),
            "running": self._running
        }

    def stop(self):
        """Stop the wrapper."""
        self._running = False


class MultiSymbolBotWrapper(EventBotWrapper):
    """
    Special wrapper for bots that analyze multiple symbols (like HFScalper).

    Maintains separate buffers per symbol and triggers analysis when any
    symbol has enough data.
    """

    def __init__(
        self,
        bot: BaseFTMOBot,
        symbols: List[str],
        trigger_config: Optional[TriggerConfig] = None,
        on_signal: Optional[Callable[[TradeSignal], None]] = None
    ):
        super().__init__(bot, trigger_config, on_signal)

        self.symbols = symbols
        self._symbol_buffers: Dict[str, TickBuffer] = {
            symbol: TickBuffer() for symbol in symbols
        }
        self._symbol_candle_builders: Dict[str, CandleBuilder] = {
            symbol: CandleBuilder(1) for symbol in symbols
        }
        self._symbol_tick_counts: Dict[str, int] = {symbol: 0 for symbol in symbols}

        logger.info(f"[MultiSymbolWrapper] {bot.config.bot_name} tracking {len(symbols)} symbols: {symbols}")

    def on_tick(self, tick: TickEvent):
        """Handle tick for any tracked symbol."""
        if not self._running:
            return

        if tick.symbol not in self.symbols:
            return

        with self._lock:
            # Buffer for this symbol
            self._symbol_buffers[tick.symbol].add_tick(tick)
            self._symbol_tick_counts[tick.symbol] += 1
            self._tick_count += 1  # Also increment base counter for status reporting
            self._last_tick_time = time.time()

            # Build candles
            builder = self._symbol_candle_builders[tick.symbol]
            candle = builder.process_tick(tick)

            if candle:
                self._symbol_buffers[tick.symbol].candles_m1.append(candle)

            # Check if we should analyze
            if self._should_analyze_multi():
                self._run_analysis_multi()

    def _should_analyze_multi(self) -> bool:
        """Check if any symbol triggers analysis."""
        now = time.time()

        if now - self._last_analysis_time < self.config.min_analysis_gap:
            return False

        # Check if any symbol has enough ticks
        for symbol, count in self._symbol_tick_counts.items():
            if count >= self.config.tick_count:
                return True

        return False

    def _run_analysis_multi(self):
        """Run analysis across all symbols."""
        try:
            start_time = time.time()

            # Build market data with candles for all symbols
            all_candles = {}
            for symbol in self.symbols:
                builder = self._symbol_candle_builders[symbol]
                candles = builder.get_candles(100)
                if len(candles) >= 10:
                    all_candles[symbol] = candles

            if not all_candles:
                return

            # For multi-symbol bots, pass candles in expected format: {symbol}_candles
            market_data = {
                "candles_by_symbol": all_candles,
            }
            # Add {symbol}_candles keys for HFScalper compatibility
            for symbol, candles in all_candles.items():
                market_data[f"{symbol}_candles"] = candles
            # Keep generic candles for backwards compatibility
            market_data["candles"] = list(all_candles.values())[0] if all_candles else []

            # Run bot cycle
            signal = self.bot.run_cycle(market_data)

            # Reset counters
            for symbol in self.symbols:
                self._symbol_tick_counts[symbol] = 0
            self._tick_count = 0  # Also reset base counter
            self._last_analysis_time = time.time()

            processing_time = time.time() - start_time
            self._metrics.record_event_bus_tick(processing_time)

            if signal:
                logger.info(
                    f"[MultiSymbolWrapper] {self.bot.config.bot_name} SIGNAL: "
                    f"{signal.direction} {signal.symbol} @ {signal.entry_price}"
                )

                self._metrics.record_signal(
                    self.bot.config.bot_name,
                    signal.symbol,
                    signal.direction
                )

                if self.on_signal_callback:
                    self.on_signal_callback(signal)

        except Exception as e:
            logger.error(f"[MultiSymbolWrapper] {self.bot.config.bot_name} analysis error: {e}")
            self._metrics.record_handler_error("analysis", self.bot.config.bot_name)

    def get_status(self) -> Dict[str, Any]:
        """Get wrapper status."""
        base_status = super().get_status()
        base_status.update({
            "symbols": self.symbols,
            "tick_counts_by_symbol": self._symbol_tick_counts.copy(),
            "buffer_sizes_by_symbol": {
                symbol: len(buf.ticks) for symbol, buf in self._symbol_buffers.items()
            }
        })
        return base_status
