"""YFinance-based data provider for cryptocurrency data (completely free, no auth)."""
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from libs.data.provider import DataProviderInterface


class YFinanceDataProvider(DataProviderInterface):
    """YFinance data provider - completely free, no authentication needed."""

    # Symbol mapping from our format to yfinance format
    SYMBOL_MAP = {
        "BTC-USD": "BTC-USD",
        "ETH-USD": "ETH-USD",
        "BNB-USD": "BNB-USD",
        "SOL-USD": "SOL-USD",
        "ADA-USD": "ADA-USD",
    }

    def __init__(self):
        """Initialize YFinance data provider."""
        logger.info("YFinance data provider initialized (free, no authentication)")

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data using YFinance.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            interval: Timeframe (1m, 5m, 15m, 1h, 1d, etc.)
            start_time: Start datetime
            end_time: End datetime
            limit: Max candles (not used, yfinance fetches all)

        Returns:
            DataFrame with [timestamp, open, high, low, close, volume]
        """
        # Map symbol to yfinance format
        yf_symbol = self.SYMBOL_MAP.get(symbol, symbol)

        # Convert interval to yfinance format
        # yfinance: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",  # Approximate
            "1d": "1d",
        }
        yf_interval = interval_map.get(interval, "1m")

        try:
            # Create ticker object
            ticker = yf.Ticker(yf_symbol)

            # Determine period or start/end
            if start_time and end_time:
                df = ticker.history(
                    start=start_time,
                    end=end_time,
                    interval=yf_interval,
                )
            elif start_time:
                df = ticker.history(
                    start=start_time,
                    interval=yf_interval,
                )
            else:
                # Default: last 7 days
                df = ticker.history(period="7d", interval=yf_interval)

            if df.empty:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            # Reset index to get timestamp as column
            df = df.reset_index()

            # Rename columns to match our format
            df = df.rename(
                columns={
                    "Date": "timestamp",
                    "Datetime": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Select only required columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Ensure timestamp is timezone-aware UTC
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

            # Apply limit if specified
            if limit:
                df = df.tail(limit)

            logger.info(f"Fetched {len(df)} candles for {symbol} ({interval})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

    def get_available_symbols(self) -> list[str]:
        """Get list of available trading pairs."""
        return list(self.SYMBOL_MAP.keys())

    def test_connection(self) -> bool:
        """Test if yfinance works."""
        try:
            ticker = yf.Ticker("BTC-USD")
            info = ticker.history(period="1d", interval="1d")
            if not info.empty:
                logger.info("✅ YFinance connection successful")
                return True
            else:
                logger.error("❌ YFinance returned empty data")
                return False
        except Exception as e:
            logger.error(f"❌ YFinance connection failed: {e}")
            return False
