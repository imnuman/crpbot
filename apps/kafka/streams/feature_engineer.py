"""Feature engineering stream processor for Kafka.

Consumes: market.candles.{symbol}.1m
Produces: features.aggregated.{symbol}

Maintains a sliding window of candles to calculate features in real-time.
"""

from collections import deque
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from apps.kafka.config.topics import (
    FEATURES_AGGREGATED,
    MARKET_CANDLES_1M,
    get_topic_name,
)
from apps.kafka.consumers.base import KafkaConsumerBase
from apps.kafka.producers.base import KafkaProducerBase
from apps.trainer.features import engineer_features
from apps.trainer.multi_tf_features import engineer_multi_tf_features


class FeatureEngineeringStream:
    """Real-time feature engineering stream processor."""

    def __init__(
        self,
        symbols: list[str],
        bootstrap_servers: str = "localhost:9092",
        window_size: int = 500,  # Number of candles to maintain for feature calculation
        enable_multi_tf: bool = True,
    ):
        """Initialize feature engineering stream.

        Args:
            symbols: List of symbols to process (e.g., ["BTC-USD", "ETH-USD"])
            bootstrap_servers: Kafka broker addresses
            window_size: Number of historical candles to maintain
            enable_multi_tf: Enable multi-timeframe features (requires buffering)
        """
        self.symbols = symbols
        self.bootstrap_servers = bootstrap_servers
        self.window_size = window_size
        self.enable_multi_tf = enable_multi_tf

        # Initialize Kafka consumer and producer
        input_topics = [
            get_topic_name(MARKET_CANDLES_1M, symbol=symbol) for symbol in symbols
        ]

        self.consumer = KafkaConsumerBase(
            topics=input_topics,
            group_id="feature-engineering",
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",  # Only process new candles
        )

        self.producer = KafkaProducerBase(bootstrap_servers=bootstrap_servers)

        # Maintain sliding window buffer for each symbol
        # Each buffer is a deque of candle dicts
        self.buffers: dict[str, deque] = {
            symbol: deque(maxlen=window_size) for symbol in symbols
        }

        # Track feature calculation stats
        self.stats = {
            "messages_processed": 0,
            "features_produced": 0,
            "errors": 0,
        }

        logger.info(
            f"Initialized FeatureEngineeringStream: symbols={symbols}, window_size={window_size}"
        )

    def start(self) -> None:
        """Start processing candles and producing features."""
        logger.info("Starting feature engineering stream...")

        try:
            self.consumer.consume(
                message_handler=self._process_candle,
                poll_timeout=1.0,
            )
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.stop()

    def _process_candle(self, candle: dict, metadata: dict) -> None:
        """Process a single candle and produce features.

        Args:
            candle: Candle data from Kafka
            metadata: Message metadata (topic, partition, offset, etc.)
        """
        try:
            symbol = candle.get("symbol")
            if not symbol or symbol not in self.symbols:
                logger.warning(f"Unknown symbol: {symbol}")
                return

            # Add candle to buffer
            self.buffers[symbol].append(candle)
            self.stats["messages_processed"] += 1

            # Check if we have enough candles to calculate features
            buffer_size = len(self.buffers[symbol])
            min_required = 100  # Minimum candles needed for technical indicators

            if buffer_size < min_required:
                logger.debug(
                    f"Buffer not ready for {symbol}: {buffer_size}/{min_required} candles"
                )
                return

            # Convert buffer to DataFrame
            df = self._buffer_to_dataframe(symbol)

            # Calculate features
            features_dict = self._calculate_features(df, symbol)

            if features_dict:
                # Produce features to Kafka
                self._produce_features(symbol, features_dict)
                self.stats["features_produced"] += 1

                # Log progress every 10 features
                if self.stats["features_produced"] % 10 == 0:
                    logger.info(
                        f"Feature engineering stats: {self.stats['features_produced']} features produced, "
                        f"{self.stats['messages_processed']} candles processed, "
                        f"{self.stats['errors']} errors"
                    )

        except Exception as e:
            logger.error(f"Error processing candle: {e}")
            self.stats["errors"] += 1

    def _buffer_to_dataframe(self, symbol: str) -> pd.DataFrame:
        """Convert candle buffer to DataFrame.

        Args:
            symbol: Symbol to convert

        Returns:
            DataFrame with OHLCV data
        """
        buffer = list(self.buffers[symbol])

        df = pd.DataFrame(buffer)

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure correct column order and types
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def _calculate_features(self, df: pd.DataFrame, symbol: str) -> Optional[dict]:
        """Calculate features from candle DataFrame.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being processed

        Returns:
            Dictionary of features for the latest candle
        """
        try:
            # For now, use basic feature engineering without multi-TF
            # Multi-TF requires buffering multiple timeframes which is more complex
            df_with_features = engineer_features(df)

            # Get features for the latest candle
            latest = df_with_features.iloc[-1]

            # Convert to dict, excluding timestamp
            features = latest.to_dict()

            # Remove timestamp (already in metadata)
            features.pop("timestamp", None)

            return features

        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            return None

    def _produce_features(self, symbol: str, features: dict) -> None:
        """Produce features to Kafka.

        Args:
            symbol: Symbol
            features: Feature dictionary
        """
        # Get output topic
        topic = get_topic_name(FEATURES_AGGREGATED, symbol=symbol)

        # Add metadata
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "features": features,
            "version": "v2.0",
            "processor": "feature_engineer",
        }

        # Produce to Kafka
        self.producer.produce(
            topic=topic,
            value=message,
            key=symbol,
            headers={"symbol": symbol, "processor": "feature_engineer"},
        )

        logger.debug(f"Produced features for {symbol} to {topic}")

    def stop(self) -> None:
        """Stop the stream processor."""
        logger.info(
            f"Stopping feature engineering stream. Final stats: {self.stats}"
        )
        self.consumer.close()
        self.producer.close()


def main():
    """Main entry point for feature engineering stream."""
    # Configuration
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    bootstrap_servers = "localhost:9092"
    window_size = 500  # Keep last 500 candles for each symbol

    # Create and start stream processor
    stream = FeatureEngineeringStream(
        symbols=symbols,
        bootstrap_servers=bootstrap_servers,
        window_size=window_size,
        enable_multi_tf=False,  # Disabled for now (complex to implement in streaming)
    )

    try:
        stream.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stream.stop()


if __name__ == "__main__":
    main()
