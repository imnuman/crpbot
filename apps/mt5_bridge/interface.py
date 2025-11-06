"""Abstract interface for MT5/FTMO bridge with mock implementation."""
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger


class MT5BridgeInterface(ABC):
    """Abstract interface for MT5/FTMO data access."""

    @abstractmethod
    def connect(self, login: str, password: str, server: str) -> bool:
        """Connect to FTMO account."""
        pass

    @abstractmethod
    def get_historical_data(
        self, symbol: str, timeframe: str, bars: int
    ) -> list[dict[str, Any]]:
        """Get historical OHLCV data."""
        pass

    @abstractmethod
    def get_spread(self, symbol: str) -> float:
        """Get current spread in bps."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from FTMO account."""
        pass


class MockMT5Bridge(MT5BridgeInterface):
    """Mock implementation for development/testing."""

    def connect(self, login: str, password: str, server: str) -> bool:
        """Mock connection."""
        logger.info(f"Mock: Connecting to {server} as {login}")
        return True

    def get_historical_data(
        self, symbol: str, timeframe: str, bars: int
    ) -> list[dict[str, Any]]:
        """Return empty mock data."""
        logger.info(f"Mock: Getting {bars} bars of {symbol} {timeframe}")
        return []

    def get_spread(self, symbol: str) -> float:
        """Return mock spread."""
        return 12.0  # 12 bps

    def disconnect(self) -> None:
        """Mock disconnect."""
        logger.info("Mock: Disconnected")


# Default bridge instance (will be replaced with real implementation in Phase 2)
bridge: MT5BridgeInterface = MockMT5Bridge()

