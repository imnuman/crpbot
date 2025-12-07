"""
HYDRA Live Trading Executor

Handles live trade execution with safety checks:
- Pre-trade FTMO validation
- Duplicate order prevention
- Position sizing by risk
- SL/TP management
- Trade logging for audit

Usage:
    executor = get_live_executor()
    result = executor.execute_signal(signal_data)
"""

import os
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .broker_interface import (
    BrokerInterface,
    OrderResult,
    OrderSide,
    PositionInfo,
)
from .mt5_broker import MT5Broker, get_mt5_broker, HYDRA_MAGIC_NUMBER

# Import HYDRA components
from libs.hydra.guardian import get_guardian
from libs.hydra.duplicate_order_guard import get_duplicate_guard
from libs.notifications.alert_manager import get_alert_manager, AlertSeverity


class ExecutionMode(Enum):
    """Trade execution mode."""
    PAPER = "paper"  # Simulation only
    LIVE = "live"  # Real money trading
    SHADOW = "shadow"  # Real prices, simulated execution (for testing)


@dataclass
class ExecutionConfig:
    """Configuration for live executor."""
    mode: ExecutionMode = ExecutionMode.PAPER
    default_risk_percent: float = 0.01  # 1% risk per trade
    max_risk_percent: float = 0.02  # 2% max risk per trade
    max_daily_trades: int = 20
    max_open_positions: int = 5
    slippage_tolerance_pips: float = 3.0
    require_guardian_approval: bool = True
    send_trade_alerts: bool = True
    log_all_executions: bool = True


@dataclass
class ExecutionResult:
    """Result of trade execution attempt."""
    success: bool
    trade_id: str = ""
    order_result: Optional[OrderResult] = None
    rejection_reason: str = ""
    guardian_check: Dict[str, Any] = field(default_factory=dict)
    duplicate_check: Tuple[bool, str] = field(default_factory=lambda: (True, ""))
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "success": self.success,
            "trade_id": self.trade_id,
            "rejection_reason": self.rejection_reason,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "order_ticket": self.order_result.ticket if self.order_result else 0,
            "fill_price": self.order_result.fill_price if self.order_result else 0,
            "slippage_pips": self.order_result.slippage_pips if self.order_result else 0,
        }


class LiveExecutor:
    """
    Live trading executor for HYDRA.

    Handles all aspects of live trade execution:
    - Pre-trade validation (Guardian, duplicate check)
    - Position sizing
    - Order execution via MT5
    - Post-trade monitoring
    - Alert notifications
    """

    def __init__(
        self,
        broker: Optional[BrokerInterface] = None,
        config: Optional[ExecutionConfig] = None
    ):
        """
        Initialize live executor.

        Args:
            broker: Broker instance (defaults to MT5)
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()

        # Initialize broker based on mode
        if self.config.mode == ExecutionMode.LIVE:
            self.broker = broker or get_mt5_broker()
            if self.broker is None:
                raise RuntimeError("MT5 broker not available for LIVE mode")
        else:
            self.broker = broker  # Can be None for paper mode

        # Initialize HYDRA components
        self.guardian = get_guardian()
        self.duplicate_guard = get_duplicate_guard()
        self.alert_manager = get_alert_manager()

        # Track daily executions
        self._daily_trades: List[Dict[str, Any]] = []
        self._daily_reset_time: Optional[datetime] = None

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._total_trades = 0
        self._successful_trades = 0
        self._rejected_trades = 0
        self._total_slippage_pips = 0.0

        logger.info(
            f"LiveExecutor initialized (mode: {self.config.mode.value}, "
            f"risk: {self.config.default_risk_percent*100:.1f}%)"
        )

    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics at midnight UTC."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self._daily_reset_time is None or self._daily_reset_time < today_start:
            self._daily_trades = []
            self._daily_reset_time = today_start
            logger.debug("Daily trade stats reset")

    def _generate_trade_id(self, symbol: str, engine: str) -> str:
        """Generate unique trade ID."""
        timestamp = int(time.time() * 1000)
        return f"HYDRA_{engine}_{symbol}_{timestamp}"

    def _convert_symbol_to_mt5(self, symbol: str) -> str:
        """
        Convert HYDRA symbol format to MT5 format.

        Examples:
            BTC-USD -> BTCUSD
            ETH-USD -> ETHUSD
        """
        return symbol.replace("-", "")

    def _calculate_position_size(
        self,
        symbol: str,
        stop_loss_pct: float,
        risk_percent: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk management.

        Args:
            symbol: Trading symbol
            stop_loss_pct: Stop loss percentage (e.g., 0.015 = 1.5%)
            risk_percent: Risk percentage (defaults to config)

        Returns:
            Lot size
        """
        if self.broker is None:
            return 0.01  # Minimum lot for paper trading

        risk_pct = risk_percent or self.config.default_risk_percent
        risk_pct = min(risk_pct, self.config.max_risk_percent)

        account = self.broker.get_account_info()
        if account is None:
            logger.warning("Could not get account info, using minimum lot")
            return 0.01

        risk_amount = account.equity * risk_pct

        # Convert SL percent to pips (approximate)
        # For crypto: 1 pip = 0.01% typically
        stop_loss_pips = stop_loss_pct * 10000 / 100  # Convert to approximate pips

        mt5_symbol = self._convert_symbol_to_mt5(symbol)
        lot_size = self.broker.calculate_lot_size(
            symbol=mt5_symbol,
            risk_amount=risk_amount,
            stop_loss_pips=stop_loss_pips
        )

        logger.debug(
            f"Position size: {lot_size} lots "
            f"(risk: ${risk_amount:.2f}, SL: {stop_loss_pips:.0f} pips)"
        )

        return lot_size

    def execute_signal(
        self,
        symbol: str,
        direction: str,  # "BUY" or "SELL"
        entry_price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        engine: str,
        strategy_id: str,
        confidence: float = 0.5,
        position_size_modifier: float = 1.0
    ) -> ExecutionResult:
        """
        Execute a trading signal.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            direction: "BUY" or "SELL"
            entry_price: Target entry price
            stop_loss_pct: Stop loss as percentage (e.g., 0.015)
            take_profit_pct: Take profit as percentage (e.g., 0.025)
            engine: Engine name (A/B/C/D)
            strategy_id: Strategy identifier
            confidence: Signal confidence (0-1)
            position_size_modifier: Position size multiplier (0-1)

        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        trade_id = self._generate_trade_id(symbol, engine)

        logger.info(
            f"[{trade_id}] Executing signal: {direction} {symbol} @ {entry_price:.2f} "
            f"(SL: {stop_loss_pct*100:.2f}%, TP: {take_profit_pct*100:.2f}%)"
        )

        with self._lock:
            self._reset_daily_stats_if_needed()
            self._total_trades += 1

            # ========== PRE-TRADE CHECKS ==========

            # 1. Daily trade limit check
            if len(self._daily_trades) >= self.config.max_daily_trades:
                reason = f"Daily trade limit reached ({self.config.max_daily_trades})"
                logger.warning(f"[{trade_id}] REJECTED: {reason}")
                self._rejected_trades += 1
                return ExecutionResult(
                    success=False,
                    trade_id=trade_id,
                    rejection_reason=reason,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            # 2. Duplicate order check
            can_place, guard_reason = self.duplicate_guard.can_place_order(
                symbol=symbol,
                direction=direction,
                engine=engine,
                entry_price=entry_price
            )
            if not can_place:
                logger.warning(f"[{trade_id}] REJECTED by duplicate guard: {guard_reason}")
                self._rejected_trades += 1
                return ExecutionResult(
                    success=False,
                    trade_id=trade_id,
                    rejection_reason=f"Duplicate guard: {guard_reason}",
                    duplicate_check=(can_place, guard_reason),
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            # 3. Guardian approval (if required)
            guardian_check = {"approved": True, "rejection_reason": ""}
            if self.config.require_guardian_approval:
                # Calculate SL price
                if direction == "BUY":
                    sl_price = entry_price * (1 - stop_loss_pct)
                else:
                    sl_price = entry_price * (1 + stop_loss_pct)

                # Get asset type
                asset_type = "standard"  # Default for crypto

                guardian_check = self.guardian.validate_trade(
                    asset=symbol,
                    asset_type=asset_type,
                    direction="LONG" if direction == "BUY" else "SHORT",
                    position_size_usd=100,  # Will be calculated properly below
                    entry_price=entry_price,
                    sl_price=sl_price,
                    regime="UNKNOWN",  # Could be passed in signal
                    current_positions=[],
                    strategy_correlations=None
                )

                if not guardian_check["approved"]:
                    reason = guardian_check.get("rejection_reason", "Guardian rejected")
                    logger.warning(f"[{trade_id}] REJECTED by Guardian: {reason}")
                    self._rejected_trades += 1

                    # Send alert for critical rejections
                    if "FTMO" in reason.upper() or "LOSS" in reason.upper():
                        self.alert_manager.send_warning(
                            f"Trade rejected by Guardian: {reason}",
                            "guardian_rejection"
                        )

                    return ExecutionResult(
                        success=False,
                        trade_id=trade_id,
                        rejection_reason=reason,
                        guardian_check=guardian_check,
                        execution_time_ms=(time.time() - start_time) * 1000
                    )

            # 4. Open position limit check
            if self.broker and self.config.mode == ExecutionMode.LIVE:
                positions = self.broker.get_positions(magic_number=HYDRA_MAGIC_NUMBER)
                if len(positions) >= self.config.max_open_positions:
                    reason = f"Max open positions reached ({self.config.max_open_positions})"
                    logger.warning(f"[{trade_id}] REJECTED: {reason}")
                    self._rejected_trades += 1
                    return ExecutionResult(
                        success=False,
                        trade_id=trade_id,
                        rejection_reason=reason,
                        execution_time_ms=(time.time() - start_time) * 1000
                    )

            # ========== EXECUTE TRADE ==========

            if self.config.mode == ExecutionMode.PAPER:
                # Paper trading - just log
                logger.info(f"[{trade_id}] PAPER TRADE executed (no real order)")
                self._successful_trades += 1
                self._daily_trades.append({
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "direction": direction,
                    "timestamp": datetime.now(timezone.utc)
                })

                # Record with duplicate guard
                self.duplicate_guard.record_order(
                    symbol=symbol,
                    direction=direction,
                    engine=engine,
                    entry_price=entry_price
                )

                return ExecutionResult(
                    success=True,
                    trade_id=trade_id,
                    guardian_check=guardian_check,
                    duplicate_check=(can_place, guard_reason),
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            elif self.config.mode == ExecutionMode.LIVE:
                if self.broker is None:
                    return ExecutionResult(
                        success=False,
                        trade_id=trade_id,
                        rejection_reason="Broker not available",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )

                # Calculate position size
                lot_size = self._calculate_position_size(symbol, stop_loss_pct)
                lot_size *= position_size_modifier  # Apply modifier

                # Convert symbol to MT5 format
                mt5_symbol = self._convert_symbol_to_mt5(symbol)

                # Calculate SL/TP prices
                if direction == "BUY":
                    sl_price = entry_price * (1 - stop_loss_pct)
                    tp_price = entry_price * (1 + take_profit_pct)
                    side = OrderSide.BUY
                else:
                    sl_price = entry_price * (1 + stop_loss_pct)
                    tp_price = entry_price * (1 - take_profit_pct)
                    side = OrderSide.SELL

                # Place order
                order_result = self.broker.place_market_order(
                    symbol=mt5_symbol,
                    side=side,
                    volume=lot_size,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    comment=f"HYDRA_{engine}_{strategy_id[:8]}",
                    magic_number=HYDRA_MAGIC_NUMBER
                )

                execution_time_ms = (time.time() - start_time) * 1000

                if order_result.success:
                    self._successful_trades += 1
                    self._total_slippage_pips += order_result.slippage_pips

                    self._daily_trades.append({
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "direction": direction,
                        "ticket": order_result.ticket,
                        "fill_price": order_result.fill_price,
                        "timestamp": datetime.now(timezone.utc)
                    })

                    # Record with duplicate guard
                    self.duplicate_guard.record_order(
                        symbol=symbol,
                        direction=direction,
                        engine=engine,
                        entry_price=order_result.fill_price
                    )

                    logger.success(
                        f"[{trade_id}] ORDER FILLED: {direction} {lot_size} {mt5_symbol} "
                        f"@ {order_result.fill_price:.2f} "
                        f"(slip: {order_result.slippage_pips:.1f} pips, "
                        f"time: {execution_time_ms:.0f}ms)"
                    )

                    # Send trade alert
                    if self.config.send_trade_alerts:
                        self.alert_manager.send_trade_executed(
                            symbol=symbol,
                            direction=direction,
                            entry_price=order_result.fill_price,
                            stop_loss=sl_price,
                            take_profit=tp_price,
                            engine=engine
                        )

                    return ExecutionResult(
                        success=True,
                        trade_id=trade_id,
                        order_result=order_result,
                        guardian_check=guardian_check,
                        duplicate_check=(can_place, guard_reason),
                        execution_time_ms=execution_time_ms
                    )

                else:
                    self._rejected_trades += 1
                    logger.error(
                        f"[{trade_id}] ORDER FAILED: {order_result.error_message}"
                    )

                    return ExecutionResult(
                        success=False,
                        trade_id=trade_id,
                        order_result=order_result,
                        rejection_reason=order_result.error_message,
                        guardian_check=guardian_check,
                        execution_time_ms=execution_time_ms
                    )

            else:  # SHADOW mode
                logger.info(f"[{trade_id}] SHADOW TRADE (simulated)")
                return ExecutionResult(
                    success=True,
                    trade_id=trade_id,
                    guardian_check=guardian_check,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

    def close_position_by_signal(
        self,
        symbol: str,
        engine: str,
        close_reason: str = "signal"
    ) -> ExecutionResult:
        """
        Close a position based on signal.

        Args:
            symbol: Trading symbol
            engine: Engine that owns the position
            close_reason: Reason for closing

        Returns:
            ExecutionResult
        """
        if self.broker is None or self.config.mode != ExecutionMode.LIVE:
            logger.info(f"Close signal for {symbol} (engine {engine}) - paper/shadow mode")
            self.duplicate_guard.record_position_closed(symbol, engine)
            return ExecutionResult(success=True, trade_id=f"CLOSE_{symbol}_{engine}")

        mt5_symbol = self._convert_symbol_to_mt5(symbol)
        positions = self.broker.get_positions(
            symbol=mt5_symbol,
            magic_number=HYDRA_MAGIC_NUMBER
        )

        if not positions:
            logger.warning(f"No position found for {symbol}")
            return ExecutionResult(
                success=False,
                rejection_reason="Position not found"
            )

        # Close first matching position
        position = positions[0]
        result = self.broker.close_position(position.ticket)

        if result.success:
            self.duplicate_guard.record_position_closed(symbol, engine)

            # Send alert
            if self.config.send_trade_alerts:
                pnl_pct = position.profit_percent
                outcome = "win" if pnl_pct > 0 else "loss"
                self.alert_manager.send_trade_closed(
                    symbol=symbol,
                    direction=position.side.value.upper(),
                    outcome=outcome,
                    pnl_usd=position.profit,
                    pnl_pct=pnl_pct,
                    engine=engine
                )

        return ExecutionResult(
            success=result.success,
            trade_id=f"CLOSE_{position.ticket}",
            order_result=result
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            win_rate = (
                self._successful_trades / self._total_trades * 100
                if self._total_trades > 0 else 0
            )
            avg_slippage = (
                self._total_slippage_pips / self._successful_trades
                if self._successful_trades > 0 else 0
            )

            return {
                "total_trades": self._total_trades,
                "successful_trades": self._successful_trades,
                "rejected_trades": self._rejected_trades,
                "success_rate": win_rate,
                "avg_slippage_pips": avg_slippage,
                "daily_trades_today": len(self._daily_trades),
                "mode": self.config.mode.value,
            }


# Global singleton
_live_executor: Optional[LiveExecutor] = None


def get_live_executor(
    mode: Optional[ExecutionMode] = None
) -> LiveExecutor:
    """
    Get or create global live executor singleton.

    Args:
        mode: Execution mode (defaults to PAPER)

    Returns:
        LiveExecutor instance
    """
    global _live_executor

    if _live_executor is None:
        # Determine mode from environment
        env_mode = os.getenv("EXECUTION_MODE", "paper").lower()
        if mode is None:
            if env_mode == "live":
                mode = ExecutionMode.LIVE
            elif env_mode == "shadow":
                mode = ExecutionMode.SHADOW
            else:
                mode = ExecutionMode.PAPER

        config = ExecutionConfig(mode=mode)
        _live_executor = LiveExecutor(config=config)

    return _live_executor
