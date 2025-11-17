"""Real-time market data fetcher for production runtime."""
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from coinbase.rest import RESTClient
from loguru import logger

from libs.config.config import Settings


class MarketDataFetcher:
    """Fetches live market data from Coinbase."""

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or Settings()
        self.cache: dict[str, pd.DataFrame] = {}
        self.cache_duration = timedelta(minutes=1)
        self.last_fetch: dict[str, datetime] = {}

        # Initialize Coinbase REST client with official SDK
        try:
            self.client = RESTClient(
                api_key=self.config.effective_api_key,
                api_secret=self.config.effective_api_secret
            )
            logger.info("✅ Coinbase REST client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Coinbase client: {e}")
            self.client = None

    def fetch_latest_candles(self, symbol: str, num_candles: int = 100) -> pd.DataFrame:
        """Fetch latest N candles from Coinbase.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            num_candles: Number of candles to fetch
            
        Returns:
            DataFrame with ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Check cache (but only if it has enough rows)
        if symbol in self.cache and symbol in self.last_fetch:
            age = datetime.now() - self.last_fetch[symbol]
            cached_rows = len(self.cache[symbol])
            if age < self.cache_duration and cached_rows >= num_candles:
                logger.debug(f"Using cached data for {symbol} (age: {age.total_seconds():.1f}s, rows: {cached_rows})")
                return self.cache[symbol].iloc[:num_candles].copy()
            elif age < self.cache_duration and cached_rows < num_candles:
                logger.debug(f"Cache has insufficient data for {symbol} ({cached_rows} < {num_candles}), fetching fresh data")

        if not self.client:
            logger.error("Coinbase client not initialized")
            return pd.DataFrame()

        logger.info(f"Fetching latest {num_candles} candles for {symbol}")

        try:
            # Use official Coinbase SDK to get candles
            response = self.client.get_candles(
                product_id=symbol,
                start=None,  # Latest candles
                end=None,
                granularity="ONE_MINUTE"
            )

            if not response or not response.candles:
                logger.error(f"No candle data returned for {symbol}")
                return pd.DataFrame()

            # Convert SDK response to DataFrame
            candles_data = []
            for candle in response.candles[:num_candles]:  # Take only requested number
                candles_data.append({
                    'timestamp': pd.to_datetime(int(candle['start']), unit='s', utc=True),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })

            df = pd.DataFrame(candles_data)
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"✅ Fetched {len(df)} candles for {symbol}")

            self.cache[symbol] = df.copy()
            self.last_fetch[symbol] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return pd.DataFrame()


_fetcher_instance: Optional[MarketDataFetcher] = None


def get_data_fetcher(config: Optional[Settings] = None) -> MarketDataFetcher:
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = MarketDataFetcher(config=config)
    return _fetcher_instance
