"""Abstract data provider interface for cryptocurrency exchanges."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger


class DataProviderInterface(ABC):
    """Abstract interface for cryptocurrency data providers."""

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candle) data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD', 'BTCUSDT')
            interval: Time interval (e.g., '1m', '5m', '1h', '1d')
            start_time: Start datetime (optional)
            end_time: End datetime (optional)
            limit: Maximum number of candles (optional)

        Returns:
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """Get list of available trading pairs."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if API connection works."""
        pass


class MockDataProvider(DataProviderInterface):
    """Mock data provider for testing."""

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return empty mock data."""
        logger.info(f"Mock: Fetching {symbol} {interval} data")
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    def get_available_symbols(self) -> list[str]:
        """Return mock symbols."""
        return ["BTC-USD", "ETH-USD", "BNB-USD"]

    def test_connection(self) -> bool:
        """Mock connection test."""
        return True

