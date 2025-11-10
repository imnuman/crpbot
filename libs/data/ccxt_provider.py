"""CCXT-based data provider for free cryptocurrency data (no auth required)."""
from datetime import datetime, timezone
from typing import Any

import ccxt
import pandas as pd
from loguru import logger

from libs.data.provider import DataProviderInterface


class CCXTDataProvider(DataProviderInterface):
    """CCXT data provider using public endpoints (no authentication needed)."""

    def __init__(self, exchange: str = "binance"):
        """
        Initialize CCXT data provider.

        Args:
            exchange: Exchange name (binance, coinbase, kraken, etc.)
                     Default: binance (most reliable for historical data)
        """
        self.exchange_name = exchange

        # Initialize exchange (public API, no auth needed)
        exchange_class = getattr(ccxt, exchange)
        self.exchange = exchange_class({
            'enableRateLimit': True,  # Respect rate limits
        })

        logger.info(f"CCXT data provider initialized for {exchange} (public API)")

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data using CCXT.

        Args:
            symbol: Trading pair (e.g., 'BTC/USD', 'BTC/USDT')
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Start datetime
            end_time: End datetime
            limit: Max candles to fetch

        Returns:
            DataFrame with [timestamp, open, high, low, close, volume]
        """
        # Convert symbol format (BTC-USD -> BTC/USDT for Binance)
        if self.exchange_name == "binance" and "USD" in symbol and "/" not in symbol:
            symbol = symbol.replace("-", "/").replace("/USD", "/USDT")
        elif "/" not in symbol:
            symbol = symbol.replace("-", "/")

        # Convert interval format if needed
        timeframe = interval

        try:
            # Fetch OHLCV data
            if start_time:
                since = int(start_time.timestamp() * 1000)  # Convert to milliseconds
            else:
                since = None

            # Fetch data in chunks if needed (exchanges have limits)
            all_ohlcv = []
            current_since = since
            max_limit = limit or 1000

            while True:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=max_limit,
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Check if we should continue fetching
                if end_time:
                    last_timestamp = ohlcv[-1][0]
                    if last_timestamp >= int(end_time.timestamp() * 1000):
                        break

                # If we got less than requested, we're done
                if len(ohlcv) < max_limit:
                    break

                # Update since for next batch
                current_since = ohlcv[-1][0] + 1

                # Safety limit: max 100k candles
                if len(all_ohlcv) >= 100000:
                    logger.warning(f"Reached safety limit of 100k candles for {symbol}")
                    break

            # Convert to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            # Filter by end_time if specified
            if end_time:
                df = df[df['timestamp'] <= end_time]

            # Apply limit if specified
            if limit:
                df = df.head(limit)

            logger.info(f"Fetched {len(df)} candles for {symbol} ({interval})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

    def get_available_symbols(self) -> list[str]:
        """Get list of available trading pairs."""
        try:
            markets = self.exchange.load_markets()
            # Return symbols in our format (BTC-USD)
            symbols = []
            for symbol in markets.keys():
                if 'USD' in symbol or 'USDT' in symbol:
                    normalized = symbol.replace('/', '-').replace('USDT', 'USD')
                    symbols.append(normalized)
            return symbols[:100]  # Return first 100 for performance
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return ["BTC-USD", "ETH-USD", "BNB-USD"]

    def test_connection(self) -> bool:
        """Test if exchange connection works."""
        try:
            self.exchange.fetch_ticker('BTC/USDT')
            logger.info(f"✅ Connection to {self.exchange_name} successful")
            return True
        except Exception as e:
            logger.error(f"❌ Connection to {self.exchange_name} failed: {e}")
            return False
