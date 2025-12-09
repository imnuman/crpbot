"""
MT5 (MetaTrader 5) Broker Integration for HYDRA Trading System.

Provides real live trading via MetaTrader 5 Python API.
Designed for FTMO Challenge compliance.

Features:
- Auto-reconnect on connection loss
- Thread-safe operations
- FTMO-compliant order management
- Detailed execution logging
- Slippage tracking

Usage:
    broker = MT5Broker(
        login=12345678,
        password="your_password",
        server="FTMO-Demo"
    )
    broker.connect()
    result = broker.place_market_order("BTCUSD", OrderSide.BUY, 0.01)

Environment Variables:
    FTMO_LOGIN: MT5 login ID
    FTMO_PASS: MT5 password
    FTMO_SERVER: MT5 server (e.g., "FTMO-Demo")
    MT5_PATH: Path to MT5 terminal (optional, for Linux Wine)
"""

import os
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from .broker_interface import (
    BrokerInterface,
    OrderResult,
    OrderStatus,
    OrderType,
    OrderSide,
    PositionInfo,
    AccountInfo,
)

# Try to import MT5, handle gracefully if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning(
        "MetaTrader5 module not available. "
        "Install with: pip install MetaTrader5 (Windows only) "
        "or use Wine/Docker on Linux."
    )


# HYDRA Magic Number - identifies our trades in MT5
HYDRA_MAGIC_NUMBER = 20241207  # YYYYMMDD format


@dataclass
class MT5Config:
    """Configuration for MT5 broker connection."""
    login: int
    password: str
    server: str
    path: str = ""  # Optional MT5 terminal path
    timeout: int = 60000  # Connection timeout in ms
    retry_attempts: int = 3
    retry_delay: float = 2.0
    auto_reconnect: bool = True
    reconnect_interval: float = 30.0
    magic_number: int = HYDRA_MAGIC_NUMBER


class MT5Broker(BrokerInterface):
    """
    MetaTrader 5 broker implementation.

    Provides live trading capabilities via MT5 Python API.
    Designed for FTMO Challenge compliance with safety features.
    """

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        config: Optional[MT5Config] = None
    ):
        """
        Initialize MT5 broker.

        Args:
            login: MT5 login ID (or FTMO_LOGIN env var)
            password: MT5 password (or FTMO_PASS env var)
            server: MT5 server (or FTMO_SERVER env var)
            path: Optional path to MT5 terminal
            config: Optional MT5Config object (overrides other params)
        """
        if not MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 module not available. "
                "This broker requires Windows or Wine on Linux."
            )

        if config:
            self.config = config
        else:
            self.config = MT5Config(
                login=login or int(os.getenv("FTMO_LOGIN", "0")),
                password=password or os.getenv("FTMO_PASS", ""),
                server=server or os.getenv("FTMO_SERVER", "FTMO-Demo"),
                path=path or os.getenv("MT5_PATH", ""),
            )

        # Validate credentials
        if not self.config.login or not self.config.password:
            raise ValueError(
                "MT5 credentials required. Set FTMO_LOGIN and FTMO_PASS env vars "
                "or pass login/password parameters."
            )

        # Connection state
        self._connected = False
        self._last_connect_time: Optional[datetime] = None
        self._connect_lock = threading.Lock()

        # Auto-reconnect thread
        self._reconnect_running = False
        self._reconnect_thread: Optional[threading.Thread] = None

        # Symbol info cache
        self._symbol_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)

        logger.info(
            f"MT5Broker initialized (login: {self.config.login}, "
            f"server: {self.config.server})"
        )

    def connect(self) -> bool:
        """
        Connect to MT5 terminal.

        Returns:
            True if connection successful
        """
        with self._connect_lock:
            if self._connected:
                return True

            for attempt in range(self.config.retry_attempts):
                try:
                    # Initialize MT5
                    if self.config.path:
                        init_result = mt5.initialize(
                            path=self.config.path,
                            login=self.config.login,
                            password=self.config.password,
                            server=self.config.server,
                            timeout=self.config.timeout
                        )
                    else:
                        init_result = mt5.initialize(
                            login=self.config.login,
                            password=self.config.password,
                            server=self.config.server,
                            timeout=self.config.timeout
                        )

                    if not init_result:
                        error = mt5.last_error()
                        logger.warning(
                            f"MT5 initialization failed (attempt {attempt + 1}): "
                            f"Error {error}"
                        )
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue

                    # Verify login
                    account = mt5.account_info()
                    if account is None:
                        logger.warning("MT5 connected but account info unavailable")
                        mt5.shutdown()
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue

                    self._connected = True
                    self._last_connect_time = datetime.now(timezone.utc)

                    logger.success(
                        f"MT5 connected successfully "
                        f"(account: {account.login}, balance: ${account.balance:,.2f})"
                    )

                    # Start auto-reconnect if enabled
                    if self.config.auto_reconnect:
                        self._start_reconnect_monitor()

                    return True

                except Exception as e:
                    logger.error(f"MT5 connection error (attempt {attempt + 1}): {e}")
                    time.sleep(self.config.retry_delay * (attempt + 1))

            logger.error(
                f"MT5 connection failed after {self.config.retry_attempts} attempts"
            )
            return False

    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        self._stop_reconnect_monitor()

        with self._connect_lock:
            if self._connected:
                mt5.shutdown()
                self._connected = False
                logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        if not self._connected:
            return False

        # Verify connection is still valid
        try:
            account = mt5.account_info()
            return account is not None
        except Exception:
            self._connected = False
            return False

    def _ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if needed."""
        if self.is_connected():
            return True

        logger.warning("MT5 connection lost, attempting reconnect...")
        return self.connect()

    def _start_reconnect_monitor(self):
        """Start background thread to monitor connection."""
        if self._reconnect_running:
            return

        self._reconnect_running = True
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="MT5ReconnectMonitor"
        )
        self._reconnect_thread.start()
        logger.debug("MT5 reconnect monitor started")

    def _stop_reconnect_monitor(self):
        """Stop reconnect monitor thread."""
        self._reconnect_running = False
        if self._reconnect_thread:
            self._reconnect_thread.join(timeout=5)
            self._reconnect_thread = None

    def _reconnect_loop(self):
        """Background loop to monitor and restore connection."""
        while self._reconnect_running:
            try:
                if not self.is_connected():
                    logger.warning("MT5 connection lost, reconnecting...")
                    self.connect()
            except Exception as e:
                logger.error(f"Error in reconnect loop: {e}")

            time.sleep(self.config.reconnect_interval)

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information."""
        if not self._ensure_connected():
            return None

        try:
            account = mt5.account_info()
            if account is None:
                return None

            return AccountInfo(
                balance=account.balance,
                equity=account.equity,
                margin=account.margin,
                free_margin=account.margin_free,
                margin_level=account.margin_level or 0,
                profit=account.profit,
                leverage=account.leverage,
                currency=account.currency,
                name=account.name,
                server=account.server,
                login=account.login,
                trade_allowed=account.trade_allowed,
                expert_allowed=account.trade_expert,
                connected=True
            )
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    def _validate_order_params(
        self,
        symbol: str,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        price: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate order parameters before sending to MT5.

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        # Validate symbol (alphanumeric with limited special chars)
        if not symbol or not isinstance(symbol, str):
            return False, "Symbol is required"
        if len(symbol) > 20:
            return False, "Symbol name too long (max 20 chars)"
        if not all(c.isalnum() or c in '.-_/' for c in symbol):
            return False, "Invalid symbol format"

        # Validate volume
        if not isinstance(volume, (int, float)) or volume <= 0:
            return False, "Volume must be positive"
        if volume > 100:  # Reasonable max lot size
            return False, "Volume exceeds maximum (100 lots)"

        # Validate prices (if provided)
        if price is not None and (not isinstance(price, (int, float)) or price <= 0):
            return False, "Price must be positive"
        if stop_loss is not None and (not isinstance(stop_loss, (int, float)) or stop_loss <= 0):
            return False, "Stop loss must be positive"
        if take_profit is not None and (not isinstance(take_profit, (int, float)) or take_profit <= 0):
            return False, "Take profit must be positive"

        return True, ""

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
        """Place a market order."""
        # Validate input parameters
        is_valid, error_msg = self._validate_order_params(symbol, volume, stop_loss, take_profit)
        if not is_valid:
            logger.warning(f"Order validation failed: {error_msg}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error_message=f"Validation: {error_msg}"
            )

        if not self._ensure_connected():
            return OrderResult(
                success=False,
                error_message="Not connected to MT5"
            )

        try:
            # Get current tick for price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    error_message=f"Could not get tick for {symbol}"
                )

            # Determine price and order type
            if side == OrderSide.BUY:
                price = tick.ask
                order_type = mt5.ORDER_TYPE_BUY
            else:
                price = tick.bid
                order_type = mt5.ORDER_TYPE_SELL

            # Build request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 20,  # 2 pips slippage allowed
                "magic": magic_number or self.config.magic_number,
                "comment": comment or f"HYDRA_{side.value.upper()}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Add SL/TP if provided
            if stop_loss:
                request["sl"] = stop_loss
            if take_profit:
                request["tp"] = take_profit

            # Send order
            logger.info(
                f"Placing market order: {side.value} {volume} {symbol} @ {price}"
            )
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side,
                    requested_price=price,
                    requested_volume=volume,
                    error_code=error[0] if error else 0,
                    error_message=f"Order send failed: {error}"
                )

            # Parse result
            success = result.retcode == mt5.TRADE_RETCODE_DONE

            order_result = OrderResult(
                success=success,
                order_id=str(result.order),
                ticket=result.deal if success else 0,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                requested_volume=volume,
                filled_volume=result.volume if success else 0,
                requested_price=price,
                fill_price=result.price if success else 0,
                slippage_pips=abs(result.price - price) * 10000 if success and result.price else 0,
                status=OrderStatus.FILLED if success else OrderStatus.REJECTED,
                error_code=result.retcode,
                error_message="" if success else f"Error {result.retcode}: {result.comment}",
                raw_response={
                    "retcode": result.retcode,
                    "deal": result.deal,
                    "order": result.order,
                    "volume": result.volume,
                    "price": result.price,
                    "bid": result.bid,
                    "ask": result.ask,
                    "comment": result.comment,
                }
            )

            if success:
                logger.success(
                    f"Order filled: {side.value} {result.volume} {symbol} @ {result.price} "
                    f"(slip: {order_result.slippage_pips:.1f} pips)"
                )
            else:
                logger.error(
                    f"Order rejected: {result.retcode} - {result.comment}"
                )

            return order_result

        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error_message=str(e)
            )

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
        """Place a limit order."""
        # Validate input parameters
        is_valid, error_msg = self._validate_order_params(symbol, volume, stop_loss, take_profit, price)
        if not is_valid:
            logger.warning(f"Limit order validation failed: {error_msg}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error_message=f"Validation: {error_msg}"
            )

        if not self._ensure_connected():
            return OrderResult(
                success=False,
                error_message="Not connected to MT5"
            )

        try:
            # Determine order type
            if side == OrderSide.BUY:
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                order_type = mt5.ORDER_TYPE_SELL_LIMIT

            # Build request
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "magic": magic_number or self.config.magic_number,
                "comment": comment or f"HYDRA_{side.value.upper()}_LIMIT",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }

            if stop_loss:
                request["sl"] = stop_loss
            if take_profit:
                request["tp"] = take_profit
            if expiration:
                request["expiration"] = int(expiration.timestamp())
                request["type_time"] = mt5.ORDER_TIME_SPECIFIED

            # Send order
            logger.info(
                f"Placing limit order: {side.value} {volume} {symbol} @ {price}"
            )
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side,
                    error_message=f"Order send failed: {error}"
                )

            success = result.retcode == mt5.TRADE_RETCODE_DONE

            return OrderResult(
                success=success,
                order_id=str(result.order),
                ticket=result.order if success else 0,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                requested_volume=volume,
                requested_price=price,
                status=OrderStatus.PENDING if success else OrderStatus.REJECTED,
                error_code=result.retcode,
                error_message="" if success else f"Error {result.retcode}: {result.comment}",
            )

        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error_message=str(e)
            )

    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> OrderResult:
        """Modify an open position's SL/TP."""
        if not self._ensure_connected():
            return OrderResult(
                success=False,
                error_message="Not connected to MT5"
            )

        try:
            # Get current position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return OrderResult(
                    success=False,
                    ticket=ticket,
                    error_message=f"Position {ticket} not found"
                )

            pos = position[0]

            # Build modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": pos.symbol,
                "sl": stop_loss if stop_loss is not None else pos.sl,
                "tp": take_profit if take_profit is not None else pos.tp,
            }

            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                return OrderResult(
                    success=False,
                    ticket=ticket,
                    error_message=f"Modification failed: {error}"
                )

            success = result.retcode == mt5.TRADE_RETCODE_DONE

            if success:
                logger.info(
                    f"Position {ticket} modified: SL={request['sl']}, TP={request['tp']}"
                )

            return OrderResult(
                success=success,
                ticket=ticket,
                symbol=pos.symbol,
                error_code=result.retcode,
                error_message="" if success else f"Error {result.retcode}: {result.comment}",
            )

        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return OrderResult(
                success=False,
                ticket=ticket,
                error_message=str(e)
            )

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None
    ) -> OrderResult:
        """Close an open position."""
        if not self._ensure_connected():
            return OrderResult(
                success=False,
                error_message="Not connected to MT5"
            )

        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return OrderResult(
                    success=False,
                    ticket=ticket,
                    error_message=f"Position {ticket} not found"
                )

            pos = position[0]
            close_volume = volume if volume else pos.volume

            # Get tick for price
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return OrderResult(
                    success=False,
                    ticket=ticket,
                    error_message=f"Could not get tick for {pos.symbol}"
                )

            # Determine close price
            if pos.type == mt5.ORDER_TYPE_BUY:
                price = tick.bid
                close_type = mt5.ORDER_TYPE_SELL
            else:
                price = tick.ask
                close_type = mt5.ORDER_TYPE_BUY

            # Build close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": pos.symbol,
                "volume": close_volume,
                "type": close_type,
                "price": price,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "HYDRA_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            logger.info(
                f"Closing position {ticket}: {close_volume} {pos.symbol} @ {price}"
            )
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                return OrderResult(
                    success=False,
                    ticket=ticket,
                    error_message=f"Close failed: {error}"
                )

            success = result.retcode == mt5.TRADE_RETCODE_DONE

            if success:
                logger.success(
                    f"Position {ticket} closed: {close_volume} {pos.symbol} @ {result.price}"
                )

            return OrderResult(
                success=success,
                order_id=str(result.order),
                ticket=result.deal if success else ticket,
                symbol=pos.symbol,
                side=OrderSide.SELL if pos.type == mt5.ORDER_TYPE_BUY else OrderSide.BUY,
                filled_volume=result.volume if success else 0,
                fill_price=result.price if success else 0,
                status=OrderStatus.FILLED if success else OrderStatus.REJECTED,
                error_code=result.retcode,
                error_message="" if success else f"Error {result.retcode}: {result.comment}",
            )

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return OrderResult(
                success=False,
                ticket=ticket,
                error_message=str(e)
            )

    def close_all_positions(
        self,
        symbol: Optional[str] = None,
        magic_number: Optional[int] = None
    ) -> List[OrderResult]:
        """Close all open positions."""
        positions = self.get_positions(symbol=symbol, magic_number=magic_number)
        results = []

        for pos in positions:
            result = self.close_position(pos.ticket)
            results.append(result)

        return results

    def get_positions(
        self,
        symbol: Optional[str] = None,
        magic_number: Optional[int] = None
    ) -> List[PositionInfo]:
        """Get all open positions."""
        if not self._ensure_connected():
            return []

        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                return []

            result = []
            for pos in positions:
                # Filter by magic number if specified
                if magic_number is not None and pos.magic != magic_number:
                    continue

                # Get current tick for profit calculation
                tick = mt5.symbol_info_tick(pos.symbol)
                current_price = 0
                if tick:
                    current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

                # Calculate profit percent
                profit_pct = 0
                if pos.price_open > 0:
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        profit_pct = (current_price - pos.price_open) / pos.price_open * 100
                    else:
                        profit_pct = (pos.price_open - current_price) / pos.price_open * 100

                result.append(PositionInfo(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    side=OrderSide.BUY if pos.type == mt5.ORDER_TYPE_BUY else OrderSide.SELL,
                    volume=pos.volume,
                    open_price=pos.price_open,
                    current_price=current_price,
                    stop_loss=pos.sl,
                    take_profit=pos.tp,
                    profit=pos.profit,
                    profit_percent=profit_pct,
                    swap=pos.swap,
                    commission=pos.commission if hasattr(pos, 'commission') else 0,
                    magic_number=pos.magic,
                    comment=pos.comment,
                    open_time=datetime.fromtimestamp(pos.time, tz=timezone.utc),
                    raw_data={
                        "ticket": pos.ticket,
                        "time": pos.time,
                        "type": pos.type,
                        "magic": pos.magic,
                        "identifier": pos.identifier,
                        "reason": pos.reason,
                    }
                ))

            return result

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_position(self, ticket: int) -> Optional[PositionInfo]:
        """Get a specific position by ticket."""
        positions = self.get_positions()
        for pos in positions:
            if pos.ticket == ticket:
                return pos
        return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information."""
        # Check cache
        if symbol in self._symbol_cache:
            if self._cache_expiry.get(symbol, datetime.min) > datetime.now(timezone.utc):
                return self._symbol_cache[symbol]

        if not self._ensure_connected():
            return None

        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None

            # Select symbol if not visible
            if not info.visible:
                mt5.symbol_select(symbol, True)
                info = mt5.symbol_info(symbol)

            result = {
                "name": info.name,
                "description": info.description,
                "point": info.point,
                "digits": info.digits,
                "trade_contract_size": info.trade_contract_size,
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "volume_step": info.volume_step,
                "spread": info.spread,
                "spread_float": info.spread_float,
                "bid": info.bid,
                "ask": info.ask,
                "trade_mode": info.trade_mode,
                "trade_stops_level": info.trade_stops_level,
                "trade_freeze_level": info.trade_freeze_level,
            }

            # Cache result
            self._symbol_cache[symbol] = result
            self._cache_expiry[symbol] = datetime.now(timezone.utc) + self._cache_ttl

            return result

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def get_tick(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current tick (bid/ask) for symbol."""
        if not self._ensure_connected():
            return None

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None

            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "time": tick.time,
            }

        except Exception as e:
            logger.error(f"Error getting tick for {symbol}: {e}")
            return None

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
            Lot size (rounded to broker's volume step)
        """
        if not self._ensure_connected():
            return 0.01  # Minimum lot

        try:
            info = self.get_symbol_info(symbol)
            if info is None:
                return 0.01

            account = self.get_account_info()
            if account is None:
                return 0.01

            # Get pip value
            point = info["point"]
            contract_size = info["trade_contract_size"]
            digits = info["digits"]

            # Pip value = contract_size * point * (10 for 5-digit brokers)
            pip_multiplier = 10 if digits in [3, 5] else 1
            pip_value = contract_size * point * pip_multiplier

            # Calculate lot size: risk_amount / (stop_loss_pips * pip_value)
            if stop_loss_pips > 0 and pip_value > 0:
                lot_size = risk_amount / (stop_loss_pips * pip_value)
            else:
                lot_size = 0.01

            # Round to volume step
            volume_step = info["volume_step"]
            volume_min = info["volume_min"]
            volume_max = info["volume_max"]

            lot_size = round(lot_size / volume_step) * volume_step
            lot_size = max(volume_min, min(volume_max, lot_size))

            return lot_size

        except Exception as e:
            logger.error(f"Error calculating lot size: {e}")
            return 0.01


# Global singleton
_mt5_broker: Optional[MT5Broker] = None


def get_mt5_broker() -> Optional[MT5Broker]:
    """
    Get or create global MT5 broker singleton.

    Returns:
        MT5Broker instance or None if MT5 not available
    """
    global _mt5_broker

    if not MT5_AVAILABLE:
        logger.warning("MT5 not available, returning None")
        return None

    if _mt5_broker is None:
        try:
            _mt5_broker = MT5Broker()
        except Exception as e:
            logger.error(f"Failed to create MT5Broker: {e}")
            return None

    return _mt5_broker
