"""
Broker integrations for HYDRA trading system.

Provides broker-agnostic interface for live trading:
- MT5 (MetaTrader 5) - Primary for FTMO
- Paper Trading - For testing

Usage:
    from libs.brokers import get_broker
    broker = get_broker()  # Returns MT5 or Paper based on config
    broker.place_order(symbol="BTCUSD", direction="BUY", ...)

    # Or use live executor for full integration:
    from libs.brokers import get_live_executor
    executor = get_live_executor()
    result = executor.execute_signal(symbol, direction, ...)
"""

from .broker_interface import (
    BrokerInterface,
    OrderResult,
    OrderStatus,
    OrderType,
    OrderSide,
    PositionInfo,
    AccountInfo,
)
from .mt5_broker import MT5Broker, get_mt5_broker, HYDRA_MAGIC_NUMBER
from .live_executor import (
    LiveExecutor,
    ExecutionMode,
    ExecutionConfig,
    ExecutionResult,
    get_live_executor,
)

__all__ = [
    # Interface
    'BrokerInterface',
    'OrderResult',
    'OrderStatus',
    'OrderType',
    'OrderSide',
    'PositionInfo',
    'AccountInfo',
    # MT5
    'MT5Broker',
    'get_mt5_broker',
    'HYDRA_MAGIC_NUMBER',
    # Live Executor
    'LiveExecutor',
    'ExecutionMode',
    'ExecutionConfig',
    'ExecutionResult',
    'get_live_executor',
]
