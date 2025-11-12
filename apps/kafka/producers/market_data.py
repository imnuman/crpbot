"""Market data ingester - produces candles to Kafka topics."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from apps.kafka.config.topics import MARKET_CANDLES_1M, get_topic_name
from apps.kafka.producers.base import KafkaProducerBase
from apps.providers.coinbase_provider import CoinbaseProvider


class MarketDataIngester:
    """Ingests market data from Coinbase and produces to Kafka."""

    def __init__(
        self,
        symbols: list[str],
        intervals: list[str] = ["1m"],
        bootstrap_servers: str = "localhost:9092",
        provider_config: Optional[dict] = None,
    ):
        """Initialize market data ingester.

        Args:
            symbols: List of symbols to ingest (e.g., ["BTC-USD", "ETH-USD"])
            intervals: List of intervals (currently only 1m supported)
            bootstrap_servers: Kafka broker addresses
            provider_config: Optional provider configuration
        """
        self.symbols = symbols
        self.intervals = intervals
        self.bootstrap_servers = bootstrap_servers

        # Initialize Kafka producer
        self.producer = KafkaProducerBase(bootstrap_servers=bootstrap_servers)

        # Initialize Coinbase provider
        self.provider = CoinbaseProvider(**(provider_config or {}))

        # Track last candle timestamps to avoid duplicates
        self.last_candle_times: dict[str, dict[str, datetime]] = {
            symbol: {} for symbol in symbols
        }

        logger.info(
            f"Initialized MarketDataIngester: symbols={symbols}, intervals={intervals}"
        )

    async def start(self) -> None:
        """Start ingesting market data."""
        logger.info("Starting market data ingestion...")

        # Start WebSocket connection for real-time data
        await self._ingest_realtime()

    async def _ingest_realtime(self) -> None:
        """Ingest real-time market data via WebSocket."""
        # For now, use polling until WebSocket is implemented
        logger.info("Starting polling-based ingestion (1-minute intervals)")

        try:
            while True:
                for symbol in self.symbols:
                    await self._fetch_and_produce_candle(symbol, "1m")

                # Wait until next minute boundary
                now = datetime.now(timezone.utc)
                seconds_until_next_minute = 60 - now.second
                await asyncio.sleep(seconds_until_next_minute)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.stop()

    async def _fetch_and_produce_candle(self, symbol: str, interval: str) -> None:
        """Fetch latest candle and produce to Kafka.

        Args:
            symbol: Symbol to fetch (e.g., "BTC-USD")
            interval: Interval (e.g., "1m")
        """
        try:
            # Fetch latest candles
            df = self.provider.get_candles(
                symbol=symbol,
                interval=interval,
                limit=2,  # Get last 2 candles
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol} {interval}")
                return

            # Get the completed candle (second to last)
            if len(df) >= 2:
                candle = df.iloc[-2]
            else:
                # If only 1 candle, use it (might be incomplete)
                candle = df.iloc[-1]

            candle_time = candle["timestamp"]

            # Check if we've already produced this candle
            last_time = self.last_candle_times.get(symbol, {}).get(interval)
            if last_time and candle_time <= last_time:
                logger.debug(
                    f"Skipping duplicate candle for {symbol} {interval} @ {candle_time}"
                )
                return

            # Prepare message
            message = {
                "timestamp": candle_time.isoformat(),
                "symbol": symbol,
                "interval": interval,
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volume"]),
                "source": "coinbase",
            }

            # Get topic name
            topic = get_topic_name(MARKET_CANDLES_1M, symbol=symbol)

            # Produce to Kafka
            self.producer.produce(
                topic=topic,
                value=message,
                key=symbol,
                headers={"interval": interval, "source": "coinbase"},
            )

            # Update last candle time
            if symbol not in self.last_candle_times:
                self.last_candle_times[symbol] = {}
            self.last_candle_times[symbol][interval] = candle_time

            logger.info(
                f"Produced candle: {symbol} {interval} @ {candle_time} -> {topic}"
            )

        except Exception as e:
            logger.error(f"Error fetching/producing candle for {symbol} {interval}: {e}")

    def stop(self) -> None:
        """Stop the ingester and close connections."""
        logger.info("Stopping market data ingester")
        self.producer.close()


async def main():
    """Main entry point for market data ingester."""
    # Configuration
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    intervals = ["1m"]
    bootstrap_servers = "localhost:9092"

    # Create and start ingester
    ingester = MarketDataIngester(
        symbols=symbols,
        intervals=intervals,
        bootstrap_servers=bootstrap_servers,
    )

    try:
        await ingester.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        ingester.stop()


if __name__ == "__main__":
    asyncio.run(main())
