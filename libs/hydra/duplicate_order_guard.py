"""
HYDRA Duplicate Order Prevention Guard

Prevents duplicate or conflicting orders that could:
- Exceed position limits
- Create unexpected risk exposure
- Violate FTMO rules
- Waste trading capital

Features:
- Cooldown period between orders for same symbol
- Checks for existing open positions
- Prevents opposite direction orders (creates net flat)
- Rate limiting per engine
- Idempotency keys for order requests

Usage:
    guard = get_duplicate_guard()
    if guard.can_place_order(symbol, direction, engine):
        # Place order
        guard.record_order(symbol, direction, engine)
"""

import os
import hashlib
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class OrderRecord:
    """Record of a placed order."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    engine: str
    timestamp: datetime
    order_id: str


@dataclass
class DuplicateGuardConfig:
    """Configuration for duplicate order prevention."""
    # Cooldown in seconds between orders for same symbol+engine
    order_cooldown_seconds: int = 300  # 5 minutes

    # Maximum orders per symbol per hour (across all engines)
    max_orders_per_symbol_per_hour: int = 4

    # Maximum orders per engine per hour
    max_orders_per_engine_per_hour: int = 10

    # Block opposite direction if position exists (per engine)
    block_opposite_direction: bool = True

    # Block opposite direction across ALL engines (prevents conflicting positions)
    block_opposite_direction_global: bool = True  # NEW: Prevents BUY+SELL on same symbol

    # Block same direction if position exists (no pyramiding)
    block_same_direction_existing: bool = False

    # Idempotency window (orders with same key within window are blocked)
    idempotency_window_seconds: int = 60


class DuplicateOrderGuard:
    """
    Prevents duplicate and conflicting orders.

    Thread-safe guard that tracks recent orders and open positions
    to prevent problematic order placement.
    """

    def __init__(self, config: Optional[DuplicateGuardConfig] = None):
        """
        Initialize duplicate order guard.

        Args:
            config: Guard configuration (uses defaults if not provided)
        """
        self.config = config or DuplicateGuardConfig()

        # Recent orders: {symbol: {engine: [OrderRecord, ...]}}
        self._recent_orders: Dict[str, Dict[str, list]] = {}

        # Open positions: {symbol: {engine: direction}}
        self._open_positions: Dict[str, Dict[str, str]] = {}

        # Idempotency keys: {key: timestamp}
        self._idempotency_keys: Dict[str, datetime] = {}

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            f"DuplicateOrderGuard initialized "
            f"(cooldown: {self.config.order_cooldown_seconds}s, "
            f"max/symbol/hour: {self.config.max_orders_per_symbol_per_hour})"
        )

    def _cleanup_old_records(self):
        """Remove expired records."""
        now = datetime.now(timezone.utc)
        cutoff_orders = now - timedelta(hours=1)
        cutoff_idempotency = now - timedelta(seconds=self.config.idempotency_window_seconds)

        # Clean up old orders
        for symbol in list(self._recent_orders.keys()):
            for engine in list(self._recent_orders[symbol].keys()):
                self._recent_orders[symbol][engine] = [
                    order for order in self._recent_orders[symbol][engine]
                    if order.timestamp > cutoff_orders
                ]
                # Remove empty engine entries
                if not self._recent_orders[symbol][engine]:
                    del self._recent_orders[symbol][engine]
            # Remove empty symbol entries
            if not self._recent_orders[symbol]:
                del self._recent_orders[symbol]

        # Clean up expired idempotency keys
        expired_keys = [
            key for key, ts in self._idempotency_keys.items()
            if ts < cutoff_idempotency
        ]
        for key in expired_keys:
            del self._idempotency_keys[key]

    def _generate_idempotency_key(
        self,
        symbol: str,
        direction: str,
        engine: str,
        entry_price: Optional[float] = None
    ) -> str:
        """Generate idempotency key for order request."""
        key_parts = [symbol, direction, engine]
        if entry_price:
            # Round price to avoid float precision issues
            key_parts.append(f"{entry_price:.2f}")

        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _get_orders_in_cooldown(self, symbol: str, engine: str) -> list:
        """Get orders within cooldown period for symbol+engine."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.config.order_cooldown_seconds)

        if symbol not in self._recent_orders:
            return []
        if engine not in self._recent_orders[symbol]:
            return []

        return [
            order for order in self._recent_orders[symbol][engine]
            if order.timestamp > cutoff
        ]

    def _get_orders_per_symbol_hour(self, symbol: str) -> int:
        """Get total orders for symbol in last hour (all engines)."""
        if symbol not in self._recent_orders:
            return 0

        total = 0
        for engine_orders in self._recent_orders[symbol].values():
            total += len(engine_orders)
        return total

    def _get_orders_per_engine_hour(self, engine: str) -> int:
        """Get total orders for engine in last hour (all symbols)."""
        total = 0
        for symbol_data in self._recent_orders.values():
            if engine in symbol_data:
                total += len(symbol_data[engine])
        return total

    def can_place_order(
        self,
        symbol: str,
        direction: str,
        engine: str,
        entry_price: Optional[float] = None,
        idempotency_key: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if order can be placed.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            direction: Order direction ("BUY" or "SELL")
            engine: Engine name (e.g., "A", "B", "C", "D")
            entry_price: Optional entry price for idempotency
            idempotency_key: Optional explicit idempotency key

        Returns:
            Tuple of (can_place, reason)
        """
        with self._lock:
            self._cleanup_old_records()

            # Check idempotency key
            if not idempotency_key:
                idempotency_key = self._generate_idempotency_key(symbol, direction, engine, entry_price)

            if idempotency_key in self._idempotency_keys:
                return False, f"Duplicate order (idempotency key: {idempotency_key})"

            # Check cooldown for symbol+engine
            cooldown_orders = self._get_orders_in_cooldown(symbol, engine)
            if cooldown_orders:
                last_order = cooldown_orders[-1]
                remaining = self.config.order_cooldown_seconds - (
                    datetime.now(timezone.utc) - last_order.timestamp
                ).total_seconds()
                return False, f"Cooldown active for {symbol} on Engine {engine} ({remaining:.0f}s remaining)"

            # Check rate limit per symbol
            symbol_orders = self._get_orders_per_symbol_hour(symbol)
            if symbol_orders >= self.config.max_orders_per_symbol_per_hour:
                return False, f"Rate limit reached for {symbol} ({symbol_orders}/{self.config.max_orders_per_symbol_per_hour} per hour)"

            # Check rate limit per engine
            engine_orders = self._get_orders_per_engine_hour(engine)
            if engine_orders >= self.config.max_orders_per_engine_per_hour:
                return False, f"Rate limit reached for Engine {engine} ({engine_orders}/{self.config.max_orders_per_engine_per_hour} per hour)"

            # Check existing position for THIS engine
            if symbol in self._open_positions and engine in self._open_positions[symbol]:
                existing_direction = self._open_positions[symbol][engine]

                if direction == existing_direction:
                    if self.config.block_same_direction_existing:
                        return False, f"Existing {existing_direction} position for {symbol} on Engine {engine} (no pyramiding)"
                else:
                    if self.config.block_opposite_direction:
                        return False, f"Existing {existing_direction} position for {symbol} on Engine {engine} (opposite direction blocked)"

            # Check existing position across ALL engines (prevents conflicting positions)
            if self.config.block_opposite_direction_global and symbol in self._open_positions:
                for other_engine, other_direction in self._open_positions[symbol].items():
                    if other_engine != engine and other_direction != direction:
                        return False, f"Conflicting {other_direction} position exists for {symbol} on Engine {other_engine} (global opposite blocked)"

            return True, "Order allowed"

    def record_order(
        self,
        symbol: str,
        direction: str,
        engine: str,
        order_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        idempotency_key: Optional[str] = None
    ):
        """
        Record a placed order.

        Args:
            symbol: Trading symbol
            direction: Order direction
            engine: Engine name
            order_id: Optional order ID
            entry_price: Optional entry price
            idempotency_key: Optional explicit idempotency key
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Generate order ID if not provided
            if not order_id:
                order_id = f"{symbol}_{engine}_{int(now.timestamp())}"

            # Record order
            order = OrderRecord(
                symbol=symbol,
                direction=direction,
                engine=engine,
                timestamp=now,
                order_id=order_id
            )

            if symbol not in self._recent_orders:
                self._recent_orders[symbol] = {}
            if engine not in self._recent_orders[symbol]:
                self._recent_orders[symbol][engine] = []

            self._recent_orders[symbol][engine].append(order)

            # Record idempotency key
            if not idempotency_key:
                idempotency_key = self._generate_idempotency_key(symbol, direction, engine, entry_price)
            self._idempotency_keys[idempotency_key] = now

            # Record open position
            if symbol not in self._open_positions:
                self._open_positions[symbol] = {}
            self._open_positions[symbol][engine] = direction

            logger.debug(
                f"Order recorded: {symbol} {direction} (Engine {engine}, ID: {order_id})"
            )

    def record_position_closed(self, symbol: str, engine: str):
        """
        Record that a position was closed.

        Args:
            symbol: Trading symbol
            engine: Engine name
        """
        with self._lock:
            if symbol in self._open_positions and engine in self._open_positions[symbol]:
                del self._open_positions[symbol][engine]
                if not self._open_positions[symbol]:
                    del self._open_positions[symbol]

                logger.debug(f"Position closed: {symbol} (Engine {engine})")

    def get_open_positions(self, engine: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """
        Get current open positions.

        Args:
            engine: Optional engine filter

        Returns:
            Dict of {symbol: {engine: direction}}
        """
        with self._lock:
            if engine:
                return {
                    symbol: {e: d for e, d in engines.items() if e == engine}
                    for symbol, engines in self._open_positions.items()
                    if engine in engines
                }
            return dict(self._open_positions)

    def get_recent_orders_count(self, symbol: Optional[str] = None, engine: Optional[str] = None) -> int:
        """
        Get count of recent orders.

        Args:
            symbol: Optional symbol filter
            engine: Optional engine filter

        Returns:
            Count of orders in last hour
        """
        with self._lock:
            self._cleanup_old_records()

            total = 0
            for s, engines in self._recent_orders.items():
                if symbol and s != symbol:
                    continue
                for e, orders in engines.items():
                    if engine and e != engine:
                        continue
                    total += len(orders)
            return total

    def clear_all(self):
        """Clear all records (for testing)."""
        with self._lock:
            self._recent_orders.clear()
            self._open_positions.clear()
            self._idempotency_keys.clear()
            logger.info("DuplicateOrderGuard cleared")


# Global singleton
_duplicate_guard: Optional[DuplicateOrderGuard] = None


def get_duplicate_guard() -> DuplicateOrderGuard:
    """Get or create global duplicate order guard singleton."""
    global _duplicate_guard
    if _duplicate_guard is None:
        _duplicate_guard = DuplicateOrderGuard()
    return _duplicate_guard


def sync_guard_with_paper_trades(paper_trades_path: str = "/app/data/hydra/paper_trades.jsonl"):
    """
    Sync duplicate guard with existing open positions from paper trades.

    Call this at startup to ensure guard knows about existing positions.
    """
    import json
    import os

    guard = get_duplicate_guard()

    # Try alternative paths
    paths_to_try = [
        paper_trades_path,
        "/root/crpbot/data/hydra/paper_trades.jsonl",
        "data/hydra/paper_trades.jsonl"
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            paper_trades_path = path
            break
    else:
        logger.warning("Paper trades file not found, guard not synced")
        return

    try:
        with open(paper_trades_path) as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    if trade.get('status') == 'OPEN':
                        symbol = trade.get('asset', trade.get('symbol'))
                        engine = trade.get('gladiator', trade.get('engine', 'X'))
                        direction = trade.get('direction')

                        if symbol and direction:
                            with guard._lock:
                                if symbol not in guard._open_positions:
                                    guard._open_positions[symbol] = {}
                                guard._open_positions[symbol][engine] = direction
                except json.JSONDecodeError:
                    continue

        total_positions = sum(len(engines) for engines in guard._open_positions.values())
        logger.info(f"DuplicateGuard synced: {total_positions} open positions loaded")

    except Exception as e:
        logger.warning(f"Failed to sync guard with paper trades: {e}")
