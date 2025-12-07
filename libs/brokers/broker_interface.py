"""
Abstract Broker Interface for HYDRA Trading System.

Defines the contract all broker implementations must follow.
This enables broker-agnostic trading logic in the runtime.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (direction)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderResult:
    """Result of placing an order."""
    success: bool
    order_id: str = ""
    ticket: int = 0  # MT5 ticket number
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    requested_volume: float = 0.0
    filled_volume: float = 0.0
    requested_price: float = 0.0
    fill_price: float = 0.0
    slippage_pips: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    error_code: int = 0
    error_message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def slippage_percent(self) -> float:
        """Calculate slippage as percentage."""
        if self.requested_price > 0:
            return abs(self.fill_price - self.requested_price) / self.requested_price * 100
        return 0.0


@dataclass
class PositionInfo:
    """Information about an open position."""
    ticket: int
    symbol: str
    side: OrderSide
    volume: float
    open_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    profit: float
    profit_percent: float
    swap: float
    commission: float
    magic_number: int  # For identifying HYDRA trades
    comment: str
    open_time: datetime
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def pnl_pips(self) -> float:
        """P&L in pips (simplified)."""
        if self.side == OrderSide.BUY:
            return (self.current_price - self.open_price) * 10000
        else:
            return (self.open_price - self.current_price) * 10000


@dataclass
class AccountInfo:
    """Broker account information."""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float  # Margin level percentage
    profit: float
    leverage: int
    currency: str
    name: str
    server: str
    login: int
    trade_allowed: bool
    expert_allowed: bool
    connected: bool


class BrokerInterface(ABC):
    """
    Abstract interface for broker implementations.

    All broker implementations (MT5, paper trading, etc.)
    must implement this interface for compatibility with HYDRA runtime.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to broker.

        Returns:
            True if connected
        """
        pass

    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account information.

        Returns:
            AccountInfo or None if not available
        """
        pass

    @abstractmethod
    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        magic_number: int = 0
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            side: BUY or SELL
            volume: Position size in lots
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            comment: Order comment (for identification)
            magic_number: Magic number (for identifying HYDRA trades)

        Returns:
            OrderResult with execution details
        """
        pass

    @abstractmethod
    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        magic_number: int = 0,
        expiration: Optional[datetime] = None
    ) -> OrderResult:
        """
        Place a limit order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            volume: Position size in lots
            price: Limit price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            comment: Order comment
            magic_number: Magic number
            expiration: Order expiration time (optional)

        Returns:
            OrderResult with execution details
        """
        pass

    @abstractmethod
    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> OrderResult:
        """
        Modify an open position's SL/TP.

        Args:
            ticket: Position ticket number
            stop_loss: New stop loss price (None = keep current)
            take_profit: New take profit price (None = keep current)

        Returns:
            OrderResult with modification details
        """
        pass

    @abstractmethod
    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None
    ) -> OrderResult:
        """
        Close an open position.

        Args:
            ticket: Position ticket number
            volume: Volume to close (None = close all)

        Returns:
            OrderResult with close details
        """
        pass

    @abstractmethod
    def close_all_positions(
        self,
        symbol: Optional[str] = None,
        magic_number: Optional[int] = None
    ) -> List[OrderResult]:
        """
        Close all open positions.

        Args:
            symbol: Only close positions for this symbol (optional)
            magic_number: Only close positions with this magic number (optional)

        Returns:
            List of OrderResult for each closed position
        """
        pass

    @abstractmethod
    def get_positions(
        self,
        symbol: Optional[str] = None,
        magic_number: Optional[int] = None
    ) -> List[PositionInfo]:
        """
        Get all open positions.

        Args:
            symbol: Filter by symbol (optional)
            magic_number: Filter by magic number (optional)

        Returns:
            List of PositionInfo
        """
        pass

    @abstractmethod
    def get_position(self, ticket: int) -> Optional[PositionInfo]:
        """
        Get a specific position by ticket.

        Args:
            ticket: Position ticket number

        Returns:
            PositionInfo or None if not found
        """
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information (spread, tick size, etc.).

        Args:
            symbol: Trading symbol

        Returns:
            Dict with symbol info or None
        """
        pass

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current tick (bid/ask) for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with 'bid' and 'ask' or None
        """
        pass

    @abstractmethod
    def calculate_lot_size(
        self,
        symbol: str,
        risk_amount: float,
        stop_loss_pips: float
    ) -> float:
        """
        Calculate lot size based on risk amount and stop loss.

        Args:
            symbol: Trading symbol
            risk_amount: Amount to risk in account currency
            stop_loss_pips: Stop loss distance in pips

        Returns:
            Lot size
        """
        pass
