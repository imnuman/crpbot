"""Synthetic data provider for testing and development (when network is restricted)."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from loguru import logger

from libs.data.provider import DataProviderInterface


class SyntheticDataProvider(DataProviderInterface):
    """Generate realistic synthetic cryptocurrency data."""

    def __init__(self, seed: int = 42):
        """
        Initialize synthetic data provider.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        logger.info(f"Synthetic data provider initialized (seed={seed})")

        # Base prices for different coins (realistic as of 2020-2025)
        self.base_prices = {
            "BTC-USD": 40000,
            "ETH-USD": 2500,
            "BNB-USD": 300,
        }

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data with realistic properties.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            interval: Timeframe (1m, 5m, 15m, 1h, 1d)
            start_time: Start datetime
            end_time: End datetime
            limit: Max candles

        Returns:
            DataFrame with [timestamp, open, high, low, close, volume]
        """
        # Get base price
        base_price = self.base_prices.get(symbol, 1000)

        # Determine time delta for interval
        interval_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }.get(interval, 1)

        # Calculate number of candles
        if start_time and end_time:
            time_diff = end_time - start_time
            n_candles = int(time_diff.total_seconds() / 60 / interval_minutes)
        elif limit:
            n_candles = limit
        else:
            # Default: 30 days of data
            n_candles = int((30 * 24 * 60) / interval_minutes)

        # Limit to reasonable size
        n_candles = min(n_candles, 1000000)  # Max 1M candles

        # Generate timestamps
        if end_time:
            end = end_time
        else:
            end = datetime.now(timezone.utc)

        if start_time:
            start = start_time
        else:
            start = end - timedelta(minutes=n_candles * interval_minutes)

        timestamps = pd.date_range(
            start=start,
            end=end,
            periods=n_candles,
            tz=timezone.utc
        )

        # Generate realistic price movement using geometric brownian motion
        # Parameters tuned for crypto volatility
        dt = interval_minutes / (60 * 24 * 365)  # Time step in years
        mu = 0.05  # Drift (5% annual return)
        sigma = 1.5  # Volatility (150% annual for crypto)

        # Generate returns
        returns = np.random.normal(
            mu * dt,
            sigma * np.sqrt(dt),
            n_candles
        )

        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))

        # Add realistic noise and trends
        # Add hourly/daily cycles
        cycle_hours = np.sin(2 * np.pi * np.arange(n_candles) / (24 * 60 / interval_minutes))
        cycle_daily = np.sin(2 * np.pi * np.arange(n_candles) / (7 * 24 * 60 / interval_minutes))
        prices = prices * (1 + 0.02 * cycle_hours + 0.03 * cycle_daily)

        # Generate OHLCV data
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            # Random intrabar movement
            bar_volatility = sigma * np.sqrt(dt) * 0.3  # Intrabar vol
            open_price = close * (1 + np.random.normal(0, bar_volatility))
            high_price = max(open_price, close) * (1 + abs(np.random.normal(0, bar_volatility)))
            low_price = min(open_price, close) * (1 - abs(np.random.normal(0, bar_volatility)))

            # Volume (random with realistic distribution)
            volume = np.random.lognormal(15, 2)  # Log-normal distribution

            data.append({
                "timestamp": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close,
                "volume": volume,
            })

        df = pd.DataFrame(data)

        logger.info(f"Generated {len(df)} synthetic candles for {symbol} ({interval})")
        logger.info(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols."""
        return list(self.base_prices.keys())

    def test_connection(self) -> bool:
        """Test synthetic data generation."""
        try:
            df = self.fetch_klines("BTC-USD", "1h", limit=10)
            if not df.empty:
                logger.info("✅ Synthetic data generation successful")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Synthetic data generation failed: {e}")
            return False
