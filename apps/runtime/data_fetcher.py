"""Real-time market data fetcher for production runtime."""
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from libs.config.config import Settings


class MarketDataFetcher:
    """Fetches live market data from Coinbase."""

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or Settings()
        self.cache: dict[str, pd.DataFrame] = {}
        self.cache_duration = timedelta(minutes=1)
        self.last_fetch: dict[str, datetime] = {}

    def fetch_latest_candles(self, symbol: str, num_candles: int = 100) -> pd.DataFrame:
        """Fetch latest N candles from Coinbase.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            num_candles: Number of candles to fetch
            
        Returns:
            DataFrame with ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Check cache
        if symbol in self.cache and symbol in self.last_fetch:
            age = datetime.now() - self.last_fetch[symbol]
            if age < self.cache_duration:
                logger.debug(f"Using cached data for {symbol} (age: {age.total_seconds():.1f}s)")
                return self.cache[symbol].copy()

        logger.info(f"Fetching latest {num_candles} candles for {symbol}")

        try:
            url = f"https://api.coinbase.com/api/v3/brokerage/products/{symbol}/candles"
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=num_candles)

            params = {
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "granularity": "ONE_MINUTE"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data or 'candles' not in data:
                logger.error(f"No candle data returned for {symbol}")
                return pd.DataFrame()

            candles = data['candles']
            if not candles:
                logger.error(f"Empty candles array for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(candles)
            df = df.rename(columns={'start': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

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
