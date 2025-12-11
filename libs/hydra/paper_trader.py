"""
HYDRA 3.0 - Paper Trading System

Simulates trades with $0 risk to validate strategy performance.

Paper Trade Lifecycle:
1. Signal generated → Create paper trade entry
2. Monitor price → Check for SL/TP hit
3. Exit detected → Close trade, record result
4. Update tournament → Feed performance back
5. Learn from losses → Extract lessons

This runs continuously in the background, monitoring all open paper trades.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path
import json
import os
import tempfile
import shutil
import threading


@dataclass
class PaperTrade:
    """
    A single paper trade (simulated).

    Tracks entry, SL, TP, and monitors for exit.
    """
    trade_id: str
    asset: str
    regime: str
    strategy_id: str
    gladiator: str  # Engine ID: "A", "B", "C", or "D" (historical name, same as engine_id)

    # Entry details
    direction: str  # "BUY" | "SELL" (final direction after any inversion)
    entry_price: float
    entry_timestamp: datetime
    position_size_usd: float

    # Exit targets
    stop_loss: float
    take_profit: float

    # Counter-Trade Intelligence tracking (must be after required fields)
    original_direction: Optional[str] = None  # What AI originally said (before inversion)
    counter_trade_inverted: bool = False  # Was this trade inverted by Counter-Trade Intelligence?

    # Certainty Engine tracking
    certainty_score: float = 0.0  # Entry certainty (0.0 to 1.0)

    # Dynamic Position Management (SL/TP can change)
    current_sl: Optional[float] = None      # Dynamic SL (may differ from original)
    current_tp: Optional[float] = None      # Dynamic TP (may differ from original)
    breakeven_triggered: bool = False       # Has SL moved to breakeven?
    trail_active: bool = False              # Is trailing stop active?
    pyramid_count: int = 0                  # Number of pyramid adds
    partial_close_count: int = 0            # Number of partial closes

    @property
    def effective_sl(self) -> float:
        """Get current effective stop loss (dynamic or original)."""
        return self.current_sl if self.current_sl is not None else self.stop_loss

    @property
    def effective_tp(self) -> float:
        """Get current effective take profit (dynamic or original)."""
        return self.current_tp if self.current_tp is not None else self.take_profit

    @property
    def engine(self) -> str:
        """Alias for gladiator - returns engine ID (A, B, C, D)."""
        return self.gladiator

    # Exit details (filled when closed)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "stop_loss" | "take_profit" | "manual"

    # Performance
    pnl_usd: float = 0.0
    pnl_percent: float = 0.0
    outcome: Optional[str] = None  # "win" | "loss"
    rr_actual: float = 0.0

    # Metadata
    status: str = "OPEN"  # "OPEN" | "CLOSED"
    slippage_est: float = 0.0005  # 0.05% estimated slippage

    def to_dict(self) -> Dict:
        """Convert to dict for storage."""
        return {
            "trade_id": self.trade_id,
            "asset": self.asset,
            "regime": self.regime,
            "strategy_id": self.strategy_id,
            "gladiator": self.gladiator,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "position_size_usd": self.position_size_usd,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "exit_reason": self.exit_reason,
            "pnl_usd": self.pnl_usd,
            "pnl_percent": self.pnl_percent,
            "outcome": self.outcome,
            "rr_actual": self.rr_actual,
            "status": self.status,
            "slippage_est": self.slippage_est,
            "original_direction": self.original_direction,
            "counter_trade_inverted": self.counter_trade_inverted,
            "certainty_score": self.certainty_score,
            "current_sl": self.current_sl,
            "current_tp": self.current_tp,
            "breakeven_triggered": self.breakeven_triggered,
            "trail_active": self.trail_active,
            "pyramid_count": self.pyramid_count,
            "partial_close_count": self.partial_close_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperTrade':
        """Load from dict."""
        trade = cls(
            trade_id=data["trade_id"],
            asset=data["asset"],
            regime=data["regime"],
            strategy_id=data["strategy_id"],
            gladiator=data["gladiator"],
            direction=data["direction"],
            entry_price=data["entry_price"],
            entry_timestamp=datetime.fromisoformat(data["entry_timestamp"]),
            position_size_usd=data["position_size_usd"],
            stop_loss=data["stop_loss"],
            take_profit=data["take_profit"]
        )

        trade.exit_price = data.get("exit_price")
        trade.exit_timestamp = datetime.fromisoformat(data["exit_timestamp"]) if data.get("exit_timestamp") else None
        trade.exit_reason = data.get("exit_reason")
        trade.pnl_usd = data.get("pnl_usd", 0.0)
        trade.pnl_percent = data.get("pnl_percent", 0.0)
        trade.outcome = data.get("outcome")
        trade.rr_actual = data.get("rr_actual", 0.0)
        trade.status = data.get("status", "OPEN")
        trade.slippage_est = data.get("slippage_est", 0.0005)

        # Counter-Trade Intelligence fields (backward compatible)
        trade.original_direction = data.get("original_direction", data["direction"])
        trade.counter_trade_inverted = data.get("counter_trade_inverted", False)

        # Certainty Engine field
        trade.certainty_score = data.get("certainty_score", 0.0)

        # Position Management fields (backward compatible)
        trade.current_sl = data.get("current_sl")
        trade.current_tp = data.get("current_tp")
        trade.breakeven_triggered = data.get("breakeven_triggered", False)
        trade.trail_active = data.get("trail_active", False)
        trade.pyramid_count = data.get("pyramid_count", 0)
        trade.partial_close_count = data.get("partial_close_count", 0)

        return trade


class PaperTradingSystem:
    """
    Manages all paper trades for HYDRA.

    Responsibilities:
    - Create paper trades from signals
    - Monitor open trades for SL/TP hits
    - Close trades and calculate P&L
    - Feed results back to tournament
    - Extract lessons from losses
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from .config import PAPER_TRADES_FILE
            storage_path = PAPER_TRADES_FILE

        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup path for recovery
        self.backup_path = Path(str(storage_path) + ".backup")

        # In-memory tracking
        self.open_trades: Dict[str, PaperTrade] = {}
        self.closed_trades: List[PaperTrade] = []

        # Statistics
        self.total_trades = 0
        self.wins = 0
        self.losses = 0

        # Thread safety for file operations
        self._file_lock = threading.Lock()

        # Load existing trades
        self._load_trades()

        # Sync duplicate guard with open positions
        try:
            from .duplicate_order_guard import sync_guard_with_paper_trades
            sync_guard_with_paper_trades(str(self.storage_path))
        except Exception as e:
            logger.warning(f"Failed to sync duplicate guard: {e}")

        logger.info(
            f"Paper Trading System initialized "
            f"({len(self.open_trades)} open, {len(self.closed_trades)} closed)"
        )

    # ==================== TRADE CREATION ====================

    def create_paper_trade(
        self,
        asset: str,
        regime: str,
        strategy_id: str,
        gladiator: str,
        signal: Dict,
        market_data: Optional[Dict] = None
    ) -> Optional[PaperTrade]:
        """
        Create new paper trade from signal.

        Args:
            signal: {
                "action": "BUY" | "SELL",
                "entry_price": 97500,
                "stop_loss_pct": 0.015,
                "take_profit_pct": 0.025,
                "position_size_usd": 100,
                "consensus_level": "STRONG"
            }
            market_data: Optional dict with price data for trend filtering

        Returns:
            PaperTrade if created, None if blocked by filters
        """
        direction = signal["action"]
        entry_price = signal["entry_price"]

        # SELL FILTER: Block SELL/SHORT trades until short detection improves
        # LESSON LEARNED: BUY=100% WR (13/13), SELL=27% WR (8/31)
        ALLOW_SHORT_TRADES = False  # Set to True when short detection is fixed
        ALLOW_FILTERED_SHORTS = True  # Allow shorts with strict overbought filters (2025-12-11)

        if direction.upper() in ("SELL", "SHORT"):
            if ALLOW_SHORT_TRADES:
                pass  # Shorts fully enabled
            elif ALLOW_FILTERED_SHORTS:
                # Check for RSI and z-score in market_data or signal
                rsi = signal.get("rsi") or (market_data.get("rsi") if market_data else None)
                zscore = signal.get("zscore") or (market_data.get("zscore") if market_data else None)

                # Thresholds for filtered shorts
                RSI_THRESHOLD = 70  # Overbought
                ZSCORE_THRESHOLD = 2.0  # 2+ std devs above mean

                if rsi is not None and zscore is not None:
                    if rsi >= RSI_THRESHOLD and zscore >= ZSCORE_THRESHOLD:
                        logger.info(f"[FilteredShort] ALLOWED: {direction} {asset} (RSI={rsi:.1f}, z={zscore:.2f})")
                        # Reduce position size by 50% for filtered shorts
                        signal["position_size_usd"] = signal.get("position_size_usd", 100) * 0.5
                    else:
                        logger.warning(f"[FilteredShort] BLOCKED: {direction} {asset} - RSI={rsi:.1f}<{RSI_THRESHOLD} or z={zscore:.2f}<{ZSCORE_THRESHOLD}")
                        return None
                else:
                    logger.warning(f"[FilteredShort] BLOCKED: {direction} {asset} - No RSI/z-score data (need RSI>={RSI_THRESHOLD}, z>={ZSCORE_THRESHOLD})")
                    return None
            else:
                logger.warning(f"[SellFilter] Trade blocked: {direction} {asset} - SELL trades disabled (historical WR: 27%)")
                return None

        # Check hybrid mode - log if engine is validated
        engine_validated = False
        try:
            from .engine_specialization import get_specialty_validator
            validator = get_specialty_validator()
            engine_validated = validator.is_engine_live(gladiator)
            if engine_validated:
                logger.info(f"[HybridMode] Engine {gladiator} - VALIDATED (live-ready)")
            else:
                logger.debug(f"[HybridMode] Engine {gladiator} - paper-only (collecting data)")
        except Exception as e:
            logger.debug(f"Hybrid mode check failed: {e}")

        # Check duplicate guard (prevents conflicting positions)
        try:
            from .duplicate_order_guard import get_duplicate_guard
            guard = get_duplicate_guard()
            can_place, reason = guard.can_place_order(asset, direction, gladiator, entry_price)
            if not can_place:
                logger.warning(f"[DuplicateGuard] Trade blocked: {reason}")
                return None
        except Exception as e:
            logger.warning(f"Duplicate guard check failed: {e}")

        # Check trend filter (prevents counter-trend trades)
        if market_data:
            try:
                from .trend_filter import get_trend_filter
                trend_filter = get_trend_filter()
                can_trade, reason = trend_filter.check_trend_alignment(asset, direction, market_data)
                if not can_trade:
                    logger.warning(f"[TrendFilter] Trade blocked: {reason}")
                    return None
            except Exception as e:
                logger.warning(f"Trend filter check failed: {e}")

        # Counter-Trade Intelligence: Check if we should INVERT the signal
        original_direction = direction
        counter_trade_inverted = False

        # Calculate current RSI for CTI and Certainty Engine
        current_rsi = 50.0
        candles = []
        if market_data:
            candles = market_data.get("candles") or market_data.get(asset, [])
            if candles and len(candles) >= 14:
                # Simple RSI approximation from recent closes
                closes = [c.get("close", 0) for c in candles[-15:] if c.get("close")]
                if len(closes) >= 2:
                    gains = sum(max(0, closes[i] - closes[i-1]) for i in range(1, len(closes)))
                    losses = sum(max(0, closes[i-1] - closes[i]) for i in range(1, len(closes)))
                    if losses > 0:
                        rs = gains / losses
                        current_rsi = 100 - (100 / (1 + rs))

        try:
            from .counter_trade_intelligence import get_counter_trade_intelligence

            cti = get_counter_trade_intelligence()

            # Get current session
            session = cti.get_current_session()

            # Check if we should invert
            final_direction, meta = cti.get_smart_direction(
                original_direction=direction,
                engine=gladiator,
                symbol=asset,
                regime=regime,
                rsi=current_rsi,
                session=session,
            )

            if meta.get("inverted"):
                direction = final_direction
                counter_trade_inverted = True
                logger.info(
                    f"[CounterTrade] INVERTED {original_direction} → {direction} "
                    f"(Historical WR={meta.get('historical_wr', 0):.0%}, "
                    f"Inverted WR={meta.get('inverted_wr', 0):.0%})"
                )
            else:
                logger.debug(
                    f"[CounterTrade] No inversion needed: {meta.get('reason', 'N/A')}"
                )
        except Exception as e:
            logger.warning(f"Counter-Trade Intelligence check failed: {e}")

        # Certainty Engine: Only take trades with 80%+ certainty
        certainty_enabled = os.environ.get("CERTAINTY_ENABLED", "true").lower() == "true"
        certainty_score = 0.0

        if certainty_enabled:
            try:
                from .certainty_engine import get_certainty_engine

                certainty_engine = get_certainty_engine()
                certainty_result = certainty_engine.calculate_certainty(
                    symbol=asset,
                    direction=direction,
                    candles=candles,
                    current_price=entry_price,
                )

                certainty_score = certainty_result.total_score

                if not certainty_result.should_trade:
                    logger.warning(
                        f"[CertaintyGate] Trade blocked: {certainty_score:.0%} < "
                        f"{certainty_engine.CERTAINTY_THRESHOLD:.0%} threshold "
                        f"({certainty_result.reason})"
                    )
                    return None

                logger.info(
                    f"[CertaintyGate] Trade approved: {certainty_score:.0%} "
                    f"(T:{certainty_result.factors.technical_confluence:.0%}, "
                    f"S:{certainty_result.factors.market_structure:.0%}, "
                    f"M:{certainty_result.factors.sentiment_order_flow:.0%}, "
                    f"X:{certainty_result.factors.time_factors:.0%})"
                )
            except Exception as e:
                logger.warning(f"Certainty Engine check failed: {e}")

        trade_id = f"{asset}_{int(datetime.now(timezone.utc).timestamp())}"
        sl_pct = signal.get("stop_loss_pct", 0.015)
        tp_pct = signal.get("take_profit_pct", 0.025)
        position_size = signal.get("position_size_usd", 100)

        # Calculate SL/TP levels
        if direction == "BUY":
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)

        trade = PaperTrade(
            trade_id=trade_id,
            asset=asset,
            regime=regime,
            strategy_id=strategy_id,
            gladiator=gladiator,
            direction=direction,
            entry_price=entry_price,
            entry_timestamp=datetime.now(timezone.utc),
            position_size_usd=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            original_direction=original_direction,
            counter_trade_inverted=counter_trade_inverted,
            certainty_score=certainty_score
        )

        self.open_trades[trade_id] = trade
        self.total_trades += 1

        # Record in duplicate guard
        try:
            from .duplicate_order_guard import get_duplicate_guard
            guard = get_duplicate_guard()
            guard.record_order(asset, direction, gladiator, trade_id, entry_price)
        except Exception as e:
            logger.warning(f"Failed to record order in guard: {e}")

        # Register with Position Manager for dynamic SL/TP management
        position_mgmt_enabled = os.environ.get("POSITION_MANAGEMENT_ENABLED", "true").lower() == "true"
        if position_mgmt_enabled:
            try:
                from .position_manager import get_position_manager
                pm = get_position_manager()
                pm.register_position(
                    trade_id=trade_id,
                    symbol=asset,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                )
                logger.debug(f"[PositionManager] Registered trade {trade_id}")
            except Exception as e:
                logger.warning(f"Failed to register with Position Manager: {e}")

        # Log trade creation (include inversion info if applicable)
        inversion_info = f" [INVERTED from {original_direction}]" if counter_trade_inverted else ""
        logger.success(
            f"Paper trade created: {direction} {asset} @ {entry_price:.2f} "
            f"(SL: {stop_loss:.2f}, TP: {take_profit:.2f}, size: ${position_size:.0f}){inversion_info}"
        )

        self._save_trades()

        return trade

    # ==================== TRADE MONITORING ====================

    def check_open_trades(self, market_data: Dict[str, List[Dict]]):
        """
        Check all open trades for SL/TP hits, edge validity, and position management.

        Enhanced with:
        - Edge Monitor: Exit if edge is lost (<2/5 factors valid)
        - Position Manager: Dynamic SL (breakeven, trail), partials, pyramids

        Args:
            market_data: {
                "BTC-USD": [{"high": 98000, "low": 97000, "close": 97500, ...}, ...],
                "ETH-USD": [...],
                ...
            }
        """
        if not self.open_trades:
            return

        trades_to_close = []

        # Check feature flags
        edge_monitor_enabled = os.environ.get("EDGE_MONITOR_ENABLED", "true").lower() == "true"
        position_mgmt_enabled = os.environ.get("POSITION_MANAGEMENT_ENABLED", "true").lower() == "true"

        for trade_id, trade in self.open_trades.items():
            asset_data = market_data.get(trade.asset)

            if not asset_data or len(asset_data) == 0:
                continue

            # Get latest candle
            latest = asset_data[-1]
            high = latest["high"]
            low = latest["low"]
            close = latest["close"]

            # 1. Edge Monitor: Check if edge is still valid
            if edge_monitor_enabled:
                try:
                    from .edge_monitor import get_edge_monitor, TradeContext, EdgeAction

                    # Calculate RSI from candles
                    current_rsi = 50.0
                    if len(asset_data) >= 14:
                        closes = [c.get("close", 0) for c in asset_data[-15:] if c.get("close")]
                        if len(closes) >= 2:
                            gains = sum(max(0, closes[i] - closes[i-1]) for i in range(1, len(closes)))
                            losses = sum(max(0, closes[i-1] - closes[i]) for i in range(1, len(closes)))
                            if losses > 0:
                                rs = gains / losses
                                current_rsi = 100 - (100 / (1 + rs))

                    context = TradeContext(
                        trade_id=trade_id,
                        symbol=trade.asset,
                        direction=trade.direction,
                        entry_price=trade.entry_price,
                        current_price=close,
                        entry_time=trade.entry_timestamp,
                        rsi=current_rsi,
                    )

                    edge_state = get_edge_monitor().check_edge(context, asset_data)

                    if edge_state.action == EdgeAction.EXIT:
                        # Edge lost - exit early to protect profits/cut losses
                        logger.warning(
                            f"[EdgeMonitor] Edge lost for {trade_id}: {edge_state.reason}"
                        )
                        trades_to_close.append((trade_id, close, "edge_invalidated"))
                        continue

                except Exception as e:
                    logger.warning(f"Edge monitor check failed for {trade_id}: {e}")

            # 2. Position Manager: Update dynamic SL/TP
            if position_mgmt_enabled:
                try:
                    from .position_manager import get_position_manager, ActionType

                    pm = get_position_manager()
                    action = pm.update_position(trade_id, close, asset_data)

                    if action:
                        if action.action_type in (ActionType.MODIFY_SL, ActionType.TRAIL_UPDATE):
                            # Update dynamic SL (both breakeven and trail modify SL)
                            trade.current_sl = action.new_sl
                            if action.reason and "breakeven" in action.reason.lower():
                                trade.breakeven_triggered = True
                            if action.action_type == ActionType.TRAIL_UPDATE or (action.reason and "trail" in action.reason.lower()):
                                trade.trail_active = True
                            logger.info(
                                f"[PositionManager] {trade_id}: SL moved to {action.new_sl:.2f} "
                                f"({action.reason})"
                            )

                        elif action.action_type == ActionType.PARTIAL_CLOSE:
                            # Partial close - reduce position size
                            close_pct = action.details.get("close_percent", 0.25)
                            trade.position_size_usd *= (1 - close_pct)
                            trade.partial_close_count += 1
                            logger.info(
                                f"[PositionManager] {trade_id}: Partial close {close_pct:.0%} "
                                f"(remaining: ${trade.position_size_usd:.0f})"
                            )

                        elif action.action_type == ActionType.PYRAMID:
                            # Pyramid - add to position (for paper trading, just track it)
                            add_pct = action.details.get("add_percent", 0.25)
                            trade.position_size_usd *= (1 + add_pct)
                            trade.pyramid_count += 1
                            logger.info(
                                f"[PositionManager] {trade_id}: Pyramid +{add_pct:.0%} "
                                f"(new size: ${trade.position_size_usd:.0f})"
                            )

                except Exception as e:
                    logger.warning(f"Position manager update failed for {trade_id}: {e}")

            # 3. Check for SL/TP hit using EFFECTIVE values (dynamic or original)
            effective_sl = trade.effective_sl
            effective_tp = trade.effective_tp

            exit_price = None
            exit_reason = None

            if trade.direction == "BUY":
                # Check stop loss (price went below SL)
                if low <= effective_sl:
                    exit_price = effective_sl
                    exit_reason = "stop_loss"
                # Check take profit (price went above TP)
                elif high >= effective_tp:
                    exit_price = effective_tp
                    exit_reason = "take_profit"

            else:  # SELL
                # Check stop loss (price went above SL)
                if high >= effective_sl:
                    exit_price = effective_sl
                    exit_reason = "stop_loss"
                # Check take profit (price went below TP)
                elif low <= effective_tp:
                    exit_price = effective_tp
                    exit_reason = "take_profit"

            if exit_price and exit_reason:
                trades_to_close.append((trade_id, exit_price, exit_reason))

        # Close trades
        for trade_id, exit_price, exit_reason in trades_to_close:
            self.close_paper_trade(trade_id, exit_price, exit_reason)

    def close_paper_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str
    ) -> Optional[PaperTrade]:
        """
        Close paper trade and calculate P&L.

        Returns:
            Closed trade with results
        """
        if trade_id not in self.open_trades:
            logger.warning(f"Trade {trade_id} not found in open trades")
            return None

        trade = self.open_trades[trade_id]

        # Validate entry price to prevent division by zero
        if trade.entry_price <= 0:
            logger.error(f"Invalid entry price {trade.entry_price} for trade {trade_id}")
            del self.open_trades[trade_id]
            return None

        # Apply slippage ONLY to SL (market order)
        # TP orders are limit orders and execute at exact price
        if exit_reason == "stop_loss":
            # Slippage works against you on SL (market order)
            if trade.direction == "BUY":
                exit_price = exit_price * (1 - trade.slippage_est)
            else:  # SELL
                exit_price = exit_price * (1 + trade.slippage_est)
        # TP: No slippage - limit orders fill at exact price

        # Calculate P&L (entry_price validated above)
        if trade.direction == "BUY":
            pnl_percent = (exit_price - trade.entry_price) / trade.entry_price
        else:  # SELL
            pnl_percent = (trade.entry_price - exit_price) / trade.entry_price

        pnl_usd = trade.position_size_usd * pnl_percent

        # Determine outcome
        outcome = "win" if pnl_percent > 0 else "loss"

        # Calculate actual R:R with safe division
        if trade.direction == "BUY":
            risk = trade.entry_price - trade.stop_loss
            reward = exit_price - trade.entry_price
        else:  # SELL
            risk = trade.stop_loss - trade.entry_price
            reward = trade.entry_price - exit_price

        # Safe R:R calculation - avoid division by zero or tiny numbers
        rr_actual = abs(reward / risk) if abs(risk) > 0.0001 else 0.0

        # Update trade
        trade.exit_price = exit_price
        trade.exit_timestamp = datetime.now(timezone.utc)
        trade.exit_reason = exit_reason
        trade.pnl_usd = pnl_usd
        trade.pnl_percent = pnl_percent
        trade.outcome = outcome
        trade.rr_actual = rr_actual
        trade.status = "CLOSED"

        # Move to closed trades
        del self.open_trades[trade_id]
        self.closed_trades.append(trade)

        # Notify duplicate guard that position is closed
        try:
            from .duplicate_order_guard import get_duplicate_guard
            guard = get_duplicate_guard()
            guard.record_position_closed(trade.asset, trade.gladiator)
        except Exception as e:
            logger.warning(f"Failed to notify guard of position close: {e}")

        # Clear Position Manager state
        try:
            from .position_manager import get_position_manager
            get_position_manager().clear_position(trade_id)
        except Exception as e:
            logger.debug(f"Failed to clear position manager state: {e}")

        # Clear Edge Monitor state
        try:
            from .edge_monitor import get_edge_monitor
            get_edge_monitor().clear_trade(trade_id)
        except Exception as e:
            logger.debug(f"Failed to clear edge monitor state: {e}")

        # Update statistics
        if outcome == "win":
            self.wins += 1
        else:
            self.losses += 1

        # Record outcome for Counter-Trade Intelligence learning
        # IMPORTANT: Record the ORIGINAL direction (what AI said), not the final direction
        try:
            from .counter_trade_intelligence import get_counter_trade_intelligence

            cti = get_counter_trade_intelligence()

            # Use original direction for recording (what AI originally predicted)
            record_direction = trade.original_direction or trade.direction

            # Get RSI zone approximation (default to neutral)
            rsi_value = 50.0  # We don't have exact RSI at entry, use neutral

            # Get session from entry timestamp
            entry_hour = trade.entry_timestamp.hour
            if 0 <= entry_hour < 8:
                session = "asia"
            elif 8 <= entry_hour < 13:
                session = "london"
            elif 13 <= entry_hour < 17:
                session = "overlap"
            elif 17 <= entry_hour < 22:
                session = "new_york"
            else:
                session = "off_hours"

            # Record the outcome (using original direction, not inverted)
            cti.record_outcome(
                engine=trade.gladiator,
                direction=record_direction,
                symbol=trade.asset,
                regime=trade.regime,
                rsi=rsi_value,
                session=session,
                won=(outcome == "win"),
                pnl_percent=pnl_percent,
            )

            logger.debug(
                f"[CounterTrade] Recorded outcome: {trade.gladiator} {record_direction} "
                f"{trade.asset} {trade.regime} → {outcome}"
            )
        except Exception as e:
            logger.warning(f"Failed to record Counter-Trade outcome: {e}")

        logger.info(
            f"Paper trade closed: {trade.asset} {trade.direction} "
            f"({outcome.upper()}, {exit_reason}) - "
            f"P&L: {pnl_percent:+.2%} (${pnl_usd:+.2f}), R:R: {rr_actual:.2f}"
        )

        self._save_trades()

        return trade

    # ==================== STATISTICS & REPORTING ====================

    def get_closed_trades(self) -> List[Dict]:
        """
        Get all closed trades as list of dicts.

        Returns:
            List of closed trade dictionaries
        """
        return [
            {
                "trade_id": t.trade_id,
                "asset": t.asset,
                "regime": t.regime,
                "strategy_id": t.strategy_id,
                "gladiator": t.gladiator,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "position_size_usd": t.position_size_usd,
                "entry_timestamp": t.entry_timestamp.isoformat() if t.entry_timestamp else None,
                "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                "exit_reason": t.exit_reason,
                "pnl_usd": t.pnl_usd,
                "pnl_percent": t.pnl_percent,
                "outcome": t.outcome,
                "rr_actual": t.rr_actual,
                "status": t.status
            }
            for t in self.closed_trades
        ]

    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics."""
        if self.total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl_usd": 0.0,
                "total_pnl_percent": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_rr": 0.0,
                "sharpe_ratio": 0.0,
                "open_trades": 0
            }

        win_rate = self.wins / self.total_trades

        # Calculate average win/loss
        wins = [t.pnl_percent for t in self.closed_trades if t.outcome == "win"]
        losses = [t.pnl_percent for t in self.closed_trades if t.outcome == "loss"]

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # Total P&L
        total_pnl_usd = sum(t.pnl_usd for t in self.closed_trades)
        total_pnl_percent = sum(t.pnl_percent for t in self.closed_trades)

        # Average R:R
        rrs = [t.rr_actual for t in self.closed_trades if t.rr_actual > 0]
        avg_rr = sum(rrs) / len(rrs) if rrs else 0.0

        # Sharpe ratio (simplified)
        if len(self.closed_trades) > 1:
            returns = [t.pnl_percent for t in self.closed_trades]
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            sharpe = avg_return / std_dev if std_dev > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "total_pnl_usd": total_pnl_usd,
            "total_pnl_percent": total_pnl_percent,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_rr": avg_rr,
            "sharpe_ratio": sharpe,
            "open_trades": len(self.open_trades)
        }

    def get_stats_by_asset(self, asset: str) -> Dict:
        """Get statistics for specific asset."""
        asset_trades = [t for t in self.closed_trades if t.asset == asset]

        if not asset_trades:
            return {
                "asset": asset,
                "total_trades": 0,
                "win_rate": 0.0
            }

        wins = sum(1 for t in asset_trades if t.outcome == "win")

        return {
            "asset": asset,
            "total_trades": len(asset_trades),
            "wins": wins,
            "losses": len(asset_trades) - wins,
            "win_rate": wins / len(asset_trades),
            "total_pnl_percent": sum(t.pnl_percent for t in asset_trades),
            "avg_pnl_percent": sum(t.pnl_percent for t in asset_trades) / len(asset_trades)
        }

    def get_stats_by_regime(self, regime: str) -> Dict:
        """Get statistics for specific regime."""
        regime_trades = [t for t in self.closed_trades if t.regime == regime]

        if not regime_trades:
            return {
                "regime": regime,
                "total_trades": 0,
                "win_rate": 0.0
            }

        wins = sum(1 for t in regime_trades if t.outcome == "win")

        return {
            "regime": regime,
            "total_trades": len(regime_trades),
            "wins": wins,
            "losses": len(regime_trades) - wins,
            "win_rate": wins / len(regime_trades),
            "total_pnl_percent": sum(t.pnl_percent for t in regime_trades),
            "avg_pnl_percent": sum(t.pnl_percent for t in regime_trades) / len(regime_trades)
        }

    def get_stats_by_strategy(self, strategy_id: str) -> Dict:
        """Get statistics for specific strategy."""
        strategy_trades = [t for t in self.closed_trades if t.strategy_id == strategy_id]

        if not strategy_trades:
            return {
                "strategy_id": strategy_id,
                "total_trades": 0,
                "win_rate": 0.0
            }

        wins = sum(1 for t in strategy_trades if t.outcome == "win")

        return {
            "strategy_id": strategy_id,
            "total_trades": len(strategy_trades),
            "wins": wins,
            "losses": len(strategy_trades) - wins,
            "win_rate": wins / len(strategy_trades),
            "total_pnl_percent": sum(t.pnl_percent for t in strategy_trades),
            "avg_pnl_percent": sum(t.pnl_percent for t in strategy_trades) / len(strategy_trades),
            "avg_rr": sum(t.rr_actual for t in strategy_trades) / len(strategy_trades)
        }

    # ==================== PERSISTENCE ====================

    def _save_trades(self):
        """
        Save trades to JSONL file using atomic write pattern.

        This ensures data integrity even if process crashes mid-write:
        1. Write to temporary file
        2. Sync to disk (fsync)
        3. Atomically rename temp -> target
        4. Create backup copy for recovery
        """
        with self._file_lock:
            try:
                # Get directory of storage path
                storage_dir = self.storage_path.parent

                # Create temp file in same directory (required for atomic rename)
                fd, temp_path = tempfile.mkstemp(
                    dir=storage_dir,
                    prefix=".paper_trades_",
                    suffix=".tmp"
                )

                try:
                    # Write all trades to temp file
                    with os.fdopen(fd, 'w') as f:
                        # Save closed trades
                        for trade in self.closed_trades:
                            f.write(json.dumps(trade.to_dict()) + "\n")

                        # Save open trades
                        for trade in self.open_trades.values():
                            f.write(json.dumps(trade.to_dict()) + "\n")

                        # Flush Python buffers
                        f.flush()
                        # Sync to disk (ensure data hits storage)
                        os.fsync(f.fileno())

                    # Create backup of current file before replacing
                    if self.storage_path.exists():
                        shutil.copy2(self.storage_path, self.backup_path)

                    # Atomic rename (POSIX guarantees atomicity)
                    shutil.move(temp_path, self.storage_path)

                    logger.debug(
                        f"Saved {len(self.closed_trades)} closed + "
                        f"{len(self.open_trades)} open trades (atomic)"
                    )

                except Exception as e:
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise

            except Exception as e:
                logger.error(f"Failed to save trades atomically: {e}")
                # Fall back to non-atomic write as last resort
                self._save_trades_fallback()

    def _save_trades_fallback(self):
        """
        Fallback non-atomic save for when atomic save fails.

        Only used as last resort - better to lose in-progress data
        than crash entirely.
        """
        try:
            with open(self.storage_path, "w") as f:
                for trade in self.closed_trades:
                    f.write(json.dumps(trade.to_dict()) + "\n")
                for trade in self.open_trades.values():
                    f.write(json.dumps(trade.to_dict()) + "\n")
            logger.warning("Used fallback (non-atomic) save")
        except Exception as e:
            logger.error(f"Fallback save also failed: {e}")

    def _load_trades(self):
        """
        Load trades from JSONL file with backup recovery.

        If main file is corrupted, attempts to recover from backup.
        """
        loaded_from_backup = False

        # Try main file first
        if self.storage_path.exists():
            try:
                self._load_trades_from_file(self.storage_path)
                return
            except Exception as e:
                logger.error(f"Failed to load from main file: {e}")

                # Try backup if main file failed
                if self.backup_path.exists():
                    logger.warning("Attempting recovery from backup...")
                    try:
                        self._load_trades_from_file(self.backup_path)
                        loaded_from_backup = True
                        logger.success("Recovered trades from backup!")
                    except Exception as backup_e:
                        logger.error(f"Backup recovery also failed: {backup_e}")
        elif self.backup_path.exists():
            # Main file doesn't exist but backup does
            logger.warning("Main file missing, loading from backup...")
            try:
                self._load_trades_from_file(self.backup_path)
                loaded_from_backup = True
            except Exception as e:
                logger.error(f"Failed to load backup: {e}")

        if loaded_from_backup:
            # Restore main file from backup
            try:
                shutil.copy2(self.backup_path, self.storage_path)
                logger.info("Restored main file from backup")
            except Exception as e:
                logger.error(f"Failed to restore main file: {e}")

        logger.info(
            f"Loaded {len(self.closed_trades)} closed trades, "
            f"{len(self.open_trades)} open trades"
        )

    def _load_trades_from_file(self, file_path: Path):
        """
        Load trades from specific file path.

        Args:
            file_path: Path to JSONL file
        """
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        trade = PaperTrade.from_dict(data)

                        if trade.status == "OPEN":
                            self.open_trades[trade.trade_id] = trade
                        else:
                            self.closed_trades.append(trade)

                            # Update statistics
                            self.total_trades += 1
                            if trade.outcome == "win":
                                self.wins += 1
                            else:
                                self.losses += 1

                    except Exception as e:
                        logger.error(f"Failed to load trade: {e}")


# Global singleton instance
_paper_trader = None

def get_paper_trader() -> PaperTradingSystem:
    """Get global PaperTradingSystem singleton."""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTradingSystem()
    return _paper_trader
