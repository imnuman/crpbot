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

    def fetch_historical_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: str = "ONE_MINUTE"
    ) -> pd.DataFrame:
        """
        Fetch historical candles for backtesting (handles Coinbase 350 candle limit via batching).

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            start: Start datetime (UTC)
            end: End datetime (UTC)
            granularity: Candle granularity ("ONE_MINUTE", "FIVE_MINUTE", etc.)

        Returns:
            DataFrame with ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        if not self.client:
            logger.error("Coinbase client not initialized")
            return pd.DataFrame()

        logger.info(f"Fetching historical candles for {symbol} from {start} to {end}")

        # Coinbase API limit: 350 candles per request
        # For 1-minute candles: 350 minutes = ~5.8 hours per batch
        candle_limit = 350
        granularity_seconds = 60  # 1 minute (adjust if using other granularities)

        all_candles = []

        try:
            current_start = start
            while current_start < end:
                # Calculate batch end time (350 candles from current_start)
                batch_end = current_start + timedelta(seconds=candle_limit * granularity_seconds)
                if batch_end > end:
                    batch_end = end

                # Convert to Unix timestamp
                start_unix = int(current_start.timestamp())
                end_unix = int(batch_end.timestamp())

                logger.debug(f"Fetching batch: {current_start} to {batch_end}")

                # Fetch batch
                response = self.client.get_candles(
                    product_id=symbol,
                    start=start_unix,
                    end=end_unix,
                    granularity=granularity
                )

                if response and response.candles:
                    # Convert SDK response to dict
                    for candle in response.candles:
                        all_candles.append({
                            'timestamp': pd.to_datetime(int(candle['start']), unit='s', utc=True),
                            'open': float(candle['open']),
                            'high': float(candle['high']),
                            'low': float(candle['low']),
                            'close': float(candle['close']),
                            'volume': float(candle['volume'])
                        })

                # Move to next batch
                current_start = batch_end

            if not all_candles:
                logger.error(f"No historical candle data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_candles)
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

            logger.info(f"✅ Fetched {len(df)} historical candles for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical candles for {symbol}: {e}")
            return pd.DataFrame()


_fetcher_instance: Optional[MarketDataFetcher] = None


def get_data_fetcher(config: Optional[Settings] = None) -> MarketDataFetcher:
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = MarketDataFetcher(config=config)
    return _fetcher_instance
