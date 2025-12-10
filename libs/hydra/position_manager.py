"""
Position Manager - Dynamic SL/TP Management

Manages open positions with:
- Breakeven stops (move SL to entry when profitable)
- Pyramiding (add to winners, never losers)
- Partial profit-taking (scale out at targets)
- Trailing stops (lock in profits)

CRITICAL SAFETY RULE:
    NEVER widen stops on losing trades!
    Only move SL in favorable direction.

Usage:
    from libs.hydra.position_manager import get_position_manager

    pm = get_position_manager()
    pm.register_position(trade_id, symbol, direction, entry, sl, tp, size)

    # On each price update
    action = pm.update_position(trade_id, current_price)
    if action:
        # Apply action (modify_sl, partial_close, pyramid, etc.)
"""

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger


# ==================== ENUMS ====================

class ActionType(Enum):
    """Types of position management actions."""
    MODIFY_SL = "modify_sl"
    MODIFY_TP = "modify_tp"
    PARTIAL_CLOSE = "partial_close"
    PYRAMID = "pyramid"
    TRAIL_UPDATE = "trail_update"


# ==================== DATA CLASSES ====================

@dataclass
class ManagementConfig:
    """Position management configuration."""
    # Breakeven
    breakeven_enabled: bool = True
    breakeven_trigger_percent: float = 0.005   # +0.5% to trigger BE
    breakeven_buffer_pips: float = 2.0          # Lock profit of 2 pips

    # Pyramiding
    pyramid_enabled: bool = True
    pyramid_trigger_percent: float = 0.005      # +0.5% profit to add
    pyramid_size_percent: float = 0.25          # Add 25% of original
    pyramid_max_adds: int = 2
    pyramid_move_sl_to_be: bool = True          # Move SL to BE on pyramid
    pyramid_require_momentum: bool = True       # Only pyramid with momentum

    # Partial profit taking
    partial_enabled: bool = True
    partial_levels: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1.0, 0.25),   # At 1R profit, take 25%
        (2.0, 0.25),   # At 2R profit, take 25%
        # Remaining 50% trails
    ])

    # Trailing stop
    trail_enabled: bool = True
    trail_trigger_r: float = 1.5               # Start trailing at 1.5R
    trail_distance_percent: float = 0.003      # Trail 0.3% behind
    trail_step_pips: float = 5.0               # Minimum step


@dataclass
class PositionState:
    """Current state of a managed position."""
    trade_id: str
    symbol: str
    direction: str  # "BUY" or "SELL"

    # Original levels
    entry_price: float
    original_sl: float
    original_tp: float
    original_size: float

    # Current levels (can be modified)
    current_sl: float
    current_tp: float
    current_size: float

    # Management state
    breakeven_triggered: bool = False
    pyramid_count: int = 0
    pyramid_entries: List[Dict] = field(default_factory=list)
    partial_takes: List[Dict] = field(default_factory=list)
    trail_stop_active: bool = False

    # Performance tracking
    max_favorable_excursion: float = 0.0  # Best unrealized P&L
    max_adverse_excursion: float = 0.0    # Worst unrealized P&L
    last_price: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "original_sl": self.original_sl,
            "original_tp": self.original_tp,
            "original_size": self.original_size,
            "current_sl": self.current_sl,
            "current_tp": self.current_tp,
            "current_size": self.current_size,
            "breakeven_triggered": self.breakeven_triggered,
            "pyramid_count": self.pyramid_count,
            "pyramid_entries": self.pyramid_entries,
            "partial_takes": self.partial_takes,
            "trail_stop_active": self.trail_stop_active,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "last_price": self.last_price,
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PositionState":
        state = cls(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            direction=data["direction"],
            entry_price=data["entry_price"],
            original_sl=data["original_sl"],
            original_tp=data["original_tp"],
            original_size=data["original_size"],
            current_sl=data["current_sl"],
            current_tp=data["current_tp"],
            current_size=data["current_size"],
        )
        state.breakeven_triggered = data.get("breakeven_triggered", False)
        state.pyramid_count = data.get("pyramid_count", 0)
        state.pyramid_entries = data.get("pyramid_entries", [])
        state.partial_takes = data.get("partial_takes", [])
        state.trail_stop_active = data.get("trail_stop_active", False)
        state.max_favorable_excursion = data.get("max_favorable_excursion", 0.0)
        state.max_adverse_excursion = data.get("max_adverse_excursion", 0.0)
        state.last_price = data.get("last_price", 0.0)

        if "created_at" in data:
            state.created_at = datetime.fromisoformat(data["created_at"])
        if "last_update" in data:
            state.last_update = datetime.fromisoformat(data["last_update"])

        return state


@dataclass
class ManagementAction:
    """Action to take on a position."""
    action_type: ActionType
    trade_id: str
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    partial_percent: Optional[float] = None
    pyramid_size: Optional[float] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)  # Additional details for callers

    @property
    def action(self) -> str:
        """Backward compatibility: return action type as string."""
        return self.action_type.value


# ==================== MAIN CLASS ====================

class PositionManager:
    """
    Manages open positions with dynamic SL/TP.

    CRITICAL: NEVER widens stops on losing trades.
    Only moves SL in favorable direction.

    Features:
    1. Breakeven stops - Lock in risk-free at +0.5%
    2. Pyramiding - Add to winners (max 2 adds)
    3. Partial profit-taking - Scale out at 1R, 2R
    4. Trailing stops - Lock in profits as price moves
    """

    PERSISTENCE_FILE = "data/hydra/position_manager_state.json"

    def __init__(self, config: Optional[ManagementConfig] = None, data_dir: str = "data/hydra"):
        """Initialize Position Manager."""
        self.config = config or ManagementConfig()
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._positions: Dict[str, PositionState] = {}
        self._lock = threading.Lock()

        # Load persisted state
        self._load_state()

        logger.info(
            f"[PositionManager] Initialized with {len(self._positions)} positions "
            f"(BE={self.config.breakeven_enabled}, "
            f"Pyramid={self.config.pyramid_enabled}, "
            f"Partial={self.config.partial_enabled}, "
            f"Trail={self.config.trail_enabled})"
        )

    # ==================== PUBLIC METHODS ====================

    def register_position(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
    ) -> PositionState:
        """
        Register a new position for management.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            position_size: Position size in USD

        Returns:
            PositionState for the new position
        """
        state = PositionState(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction.upper(),
            entry_price=entry_price,
            original_sl=stop_loss,
            original_tp=take_profit,
            original_size=position_size,
            current_sl=stop_loss,
            current_tp=take_profit,
            current_size=position_size,
            last_price=entry_price,
        )

        with self._lock:
            self._positions[trade_id] = state

        self._save_state()

        logger.info(
            f"[PositionManager] Registered: {trade_id} "
            f"{direction} {symbol} @ {entry_price:.2f} "
            f"(SL={stop_loss:.2f}, TP={take_profit:.2f})"
        )

        return state

    def update_position(
        self,
        trade_id: str,
        current_price: float,
        candles: Optional[List[Dict]] = None,
    ) -> Optional[ManagementAction]:
        """
        Update position and check for management actions.

        Args:
            trade_id: Trade identifier
            current_price: Current market price
            candles: Optional candle data for momentum check

        Returns:
            ManagementAction if action needed, None otherwise
        """
        with self._lock:
            state = self._positions.get(trade_id)
            if not state:
                return None

            # Update last price and track excursions
            state.last_price = current_price
            state.last_update = datetime.now(timezone.utc)

            # Calculate current P&L percent
            pnl_percent = self._calc_pnl_percent(state, current_price)

            # Track max favorable/adverse excursion
            if pnl_percent > state.max_favorable_excursion:
                state.max_favorable_excursion = pnl_percent
            if pnl_percent < state.max_adverse_excursion:
                state.max_adverse_excursion = pnl_percent

        # Check management actions in order of priority
        # 1. Breakeven (safest, highest priority)
        if self.config.breakeven_enabled:
            action = self._check_breakeven(state, current_price)
            if action:
                self._save_state()
                return action

        # 2. Trailing stop (protect profits)
        if self.config.trail_enabled:
            action = self._check_trail_stop(state, current_price)
            if action:
                self._save_state()
                return action

        # 3. Partial profit taking
        if self.config.partial_enabled:
            action = self._check_partial_take(state, current_price)
            if action:
                self._save_state()
                return action

        # 4. Pyramiding (riskiest, lowest priority)
        if self.config.pyramid_enabled:
            action = self._check_pyramid(state, current_price, candles)
            if action:
                self._save_state()
                return action

        return None

    def unregister_position(self, trade_id: str) -> None:
        """Remove position from management (when closed)."""
        with self._lock:
            if trade_id in self._positions:
                del self._positions[trade_id]
                logger.debug(f"[PositionManager] Unregistered: {trade_id}")

        self._save_state()

    def clear_position(self, trade_id: str) -> None:
        """Alias for unregister_position (for API compatibility)."""
        self.unregister_position(trade_id)

    def get_position_state(self, trade_id: str) -> Optional[PositionState]:
        """Get current state of a position."""
        with self._lock:
            return self._positions.get(trade_id)

    def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all managed positions."""
        with self._lock:
            return dict(self._positions)

    # ==================== MANAGEMENT CHECKS ====================

    def _check_breakeven(
        self,
        state: PositionState,
        current_price: float,
    ) -> Optional[ManagementAction]:
        """
        Check if breakeven stop should be triggered.

        Breakeven moves SL to entry when position is profitable by trigger amount.
        """
        if state.breakeven_triggered:
            return None  # Already triggered

        pnl_percent = self._calc_pnl_percent(state, current_price)

        if pnl_percent >= self.config.breakeven_trigger_percent:
            # Calculate new SL (entry + small buffer)
            if state.direction == "BUY":
                new_sl = state.entry_price + self.config.breakeven_buffer_pips
            else:  # SELL
                new_sl = state.entry_price - self.config.breakeven_buffer_pips

            # SAFETY: Only move SL if it's better than current
            if not self._is_better_sl(state, new_sl):
                return None

            # Update state
            with self._lock:
                state.breakeven_triggered = True
                state.current_sl = new_sl

            logger.info(
                f"[PositionManager] BREAKEVEN triggered: {state.trade_id} "
                f"SL moved to {new_sl:.2f} (was {state.original_sl:.2f})"
            )

            return ManagementAction(
                action_type=ActionType.MODIFY_SL,
                trade_id=state.trade_id,
                new_sl=new_sl,
                reason=f"Breakeven triggered at {pnl_percent:.1%} profit",
            )

        return None

    def _check_trail_stop(
        self,
        state: PositionState,
        current_price: float,
    ) -> Optional[ManagementAction]:
        """
        Check if trailing stop should update.

        Trail starts at trail_trigger_r and follows price at trail_distance_percent.
        """
        # Calculate current R multiple
        current_r = self._calc_current_r(state, current_price)

        # Check if trail should be active
        if current_r >= self.config.trail_trigger_r:
            if not state.trail_stop_active:
                with self._lock:
                    state.trail_stop_active = True
                logger.info(f"[PositionManager] Trail activated: {state.trade_id} at {current_r:.2f}R")

        if not state.trail_stop_active:
            return None

        # Calculate trail stop level
        if state.direction == "BUY":
            trail_sl = current_price * (1 - self.config.trail_distance_percent)
        else:  # SELL
            trail_sl = current_price * (1 + self.config.trail_distance_percent)

        # SAFETY: Only update if better than current SL
        if not self._is_better_sl(state, trail_sl):
            return None

        # Check minimum step size
        sl_change = abs(trail_sl - state.current_sl)
        if sl_change < self.config.trail_step_pips:
            return None

        # Update state
        with self._lock:
            old_sl = state.current_sl
            state.current_sl = trail_sl

        logger.info(
            f"[PositionManager] TRAIL updated: {state.trade_id} "
            f"SL {old_sl:.2f} → {trail_sl:.2f}"
        )

        return ManagementAction(
            action_type=ActionType.TRAIL_UPDATE,
            trade_id=state.trade_id,
            new_sl=trail_sl,
            reason=f"Trail stop at {current_r:.2f}R",
        )

    def _check_partial_take(
        self,
        state: PositionState,
        current_price: float,
    ) -> Optional[ManagementAction]:
        """
        Check if partial profit should be taken.

        Takes partial at defined R levels.
        """
        current_r = self._calc_current_r(state, current_price)

        # Check each partial level
        for level_r, level_percent in self.config.partial_levels:
            # Skip if already taken at this level
            already_taken = any(
                p.get("level_r") == level_r for p in state.partial_takes
            )
            if already_taken:
                continue

            # Check if level reached
            if current_r >= level_r:
                partial_size = state.current_size * level_percent

                # Record partial take
                with self._lock:
                    state.partial_takes.append({
                        "level_r": level_r,
                        "percent": level_percent,
                        "size": partial_size,
                        "price": current_price,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    state.current_size -= partial_size

                logger.info(
                    f"[PositionManager] PARTIAL take: {state.trade_id} "
                    f"{level_percent:.0%} at {level_r}R (size: {partial_size:.2f})"
                )

                return ManagementAction(
                    action_type=ActionType.PARTIAL_CLOSE,
                    trade_id=state.trade_id,
                    partial_percent=level_percent,
                    reason=f"Partial take at {level_r}R",
                    details={"close_percent": level_percent, "level_r": level_r},
                )

        return None

    def _check_pyramid(
        self,
        state: PositionState,
        current_price: float,
        candles: Optional[List[Dict]] = None,
    ) -> Optional[ManagementAction]:
        """
        Check if pyramid add should be triggered.

        Requires:
        - Position in profit by trigger amount
        - Momentum confirmation (if enabled)
        - Not exceeded max pyramid adds
        """
        # Check pyramid count
        if state.pyramid_count >= self.config.pyramid_max_adds:
            return None

        pnl_percent = self._calc_pnl_percent(state, current_price)

        # Need profit to pyramid
        min_profit = self.config.pyramid_trigger_percent * (state.pyramid_count + 1)
        if pnl_percent < min_profit:
            return None

        # Check momentum if required
        if self.config.pyramid_require_momentum and candles:
            has_momentum = self._check_momentum(state.direction, candles)
            if not has_momentum:
                return None

        # Calculate pyramid size
        pyramid_size = state.original_size * self.config.pyramid_size_percent

        # Record pyramid
        with self._lock:
            state.pyramid_count += 1
            state.pyramid_entries.append({
                "price": current_price,
                "size": pyramid_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            state.current_size += pyramid_size

            # Move to breakeven if configured
            if self.config.pyramid_move_sl_to_be and not state.breakeven_triggered:
                state.breakeven_triggered = True
                if state.direction == "BUY":
                    state.current_sl = state.entry_price + self.config.breakeven_buffer_pips
                else:
                    state.current_sl = state.entry_price - self.config.breakeven_buffer_pips

        logger.info(
            f"[PositionManager] PYRAMID add #{state.pyramid_count}: {state.trade_id} "
            f"+{pyramid_size:.2f} at {current_price:.2f}"
        )

        return ManagementAction(
            action_type=ActionType.PYRAMID,
            trade_id=state.trade_id,
            pyramid_size=pyramid_size,
            new_sl=state.current_sl if self.config.pyramid_move_sl_to_be else None,
            reason=f"Pyramid #{state.pyramid_count} at {pnl_percent:.1%} profit",
            details={"add_percent": self.config.pyramid_size_percent, "pyramid_count": state.pyramid_count},
        )

    # ==================== HELPER METHODS ====================

    def _calc_pnl_percent(self, state: PositionState, current_price: float) -> float:
        """Calculate current P&L percentage."""
        if state.entry_price <= 0:
            return 0.0

        if state.direction == "BUY":
            return (current_price - state.entry_price) / state.entry_price
        else:  # SELL
            return (state.entry_price - current_price) / state.entry_price

    def _calc_current_r(self, state: PositionState, current_price: float) -> float:
        """Calculate current R multiple (profit in terms of risk)."""
        risk = abs(state.entry_price - state.original_sl)
        if risk <= 0:
            return 0.0

        if state.direction == "BUY":
            profit = current_price - state.entry_price
        else:
            profit = state.entry_price - current_price

        return profit / risk

    def _is_better_sl(self, state: PositionState, new_sl: float) -> bool:
        """
        Check if new SL is better (closer to TP) than current.

        CRITICAL: This prevents widening stops on losers!
        """
        if state.direction == "BUY":
            # For longs, higher SL is better
            return new_sl > state.current_sl
        else:  # SELL
            # For shorts, lower SL is better
            return new_sl < state.current_sl

    def _check_momentum(self, direction: str, candles: List[Dict]) -> bool:
        """Check if price momentum supports the direction."""
        if not candles or len(candles) < 5:
            return True  # No data, allow by default

        closes = [c.get("close", 0) for c in candles[-5:] if c.get("close")]
        if len(closes) < 5:
            return True

        # Check if recent candles support direction
        momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0

        if direction == "BUY":
            return momentum > 0  # Price going up
        else:  # SELL
            return momentum < 0  # Price going down

    # ==================== PERSISTENCE ====================

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        state_file = self._data_dir / "position_manager_state.json"

        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    self._positions = {
                        tid: PositionState.from_dict(pdata)
                        for tid, pdata in data.items()
                    }
                logger.info(f"[PositionManager] Loaded {len(self._positions)} positions from disk")
            except Exception as e:
                logger.warning(f"[PositionManager] Failed to load state: {e}")
                self._positions = {}
        else:
            self._positions = {}

    def _save_state(self) -> None:
        """Save state to disk (atomic write)."""
        state_file = self._data_dir / "position_manager_state.json"
        temp_file = self._data_dir / "position_manager_state.tmp"

        try:
            with self._lock:
                data = {
                    tid: state.to_dict()
                    for tid, state in self._positions.items()
                }

            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            temp_file.rename(state_file)

        except Exception as e:
            logger.error(f"[PositionManager] Failed to save state: {e}")


# ==================== SINGLETON ACCESSOR ====================

_instance: Optional[PositionManager] = None
_instance_lock = threading.Lock()


def get_position_manager(
    config: Optional[ManagementConfig] = None,
    data_dir: str = "data/hydra"
) -> PositionManager:
    """
    Get the singleton instance of PositionManager.

    Args:
        config: Optional configuration (only used on first call)
        data_dir: Data directory for persistence

    Returns:
        PositionManager instance
    """
    global _instance

    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PositionManager(config=config, data_dir=data_dir)

    return _instance


def reset_position_manager() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    with _instance_lock:
        _instance = None


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage
    pm = get_position_manager()

    # Register a position
    state = pm.register_position(
        trade_id="test_123",
        symbol="BTC-USD",
        direction="BUY",
        entry_price=97000,
        stop_loss=96000,  # 1000 risk
        take_profit=100000,  # 3000 target
        position_size=1000,
    )

    print(f"Registered position: {state.trade_id}")
    print(f"Entry: {state.entry_price}, SL: {state.current_sl}, TP: {state.current_tp}")

    # Simulate price movement
    prices = [97500, 98000, 98500, 99000, 99500]  # Winning trade

    for price in prices:
        action = pm.update_position("test_123", price)
        if action:
            print(f"\nAction: {action.action}")
            print(f"  Reason: {action.reason}")
            if action.new_sl:
                print(f"  New SL: {action.new_sl}")

    # Show final state
    final_state = pm.get_position_state("test_123")
    if final_state:
        print(f"\nFinal State:")
        print(f"  Current SL: {final_state.current_sl}")
        print(f"  Breakeven: {final_state.breakeven_triggered}")
        print(f"  Trail Active: {final_state.trail_stop_active}")
        print(f"  Pyramids: {final_state.pyramid_count}")
        print(f"  Partials: {len(final_state.partial_takes)}")
