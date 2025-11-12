"""Kafka topic configurations for crpbot."""
from dataclasses import dataclass


@dataclass
class TopicConfig:
    """Kafka topic configuration."""

    name: str
    partitions: int
    replication_factor: int
    retention_ms: int
    compression_type: str = "lz4"


# Market Data Topics
MARKET_CANDLES_1M = TopicConfig(
    name="market.candles.{symbol}.1m",
    partitions=3,
    replication_factor=1,  # 3 in production
    retention_ms=7 * 24 * 60 * 60 * 1000,  # 7 days
)

MARKET_CANDLES_5M = TopicConfig(
    name="market.candles.{symbol}.5m",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

MARKET_CANDLES_15M = TopicConfig(
    name="market.candles.{symbol}.15m",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

MARKET_CANDLES_1H = TopicConfig(
    name="market.candles.{symbol}.1h",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

# Feature Topics
FEATURES_RAW = TopicConfig(
    name="features.raw.{symbol}",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

FEATURES_TECHNICAL = TopicConfig(
    name="features.technical.{symbol}",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

FEATURES_MULTI_TF = TopicConfig(
    name="features.multi_tf.{symbol}",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

FEATURES_AGGREGATED = TopicConfig(
    name="features.aggregated.{symbol}",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

# Prediction Topics
PREDICTIONS_LSTM = TopicConfig(
    name="predictions.lstm.{symbol}",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

PREDICTIONS_TRANSFORMER = TopicConfig(
    name="predictions.transformer",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

PREDICTIONS_ENSEMBLE = TopicConfig(
    name="predictions.ensemble.{symbol}",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

# Signal Topics
SIGNALS_RAW = TopicConfig(
    name="signals.raw",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

SIGNALS_CALIBRATED = TopicConfig(
    name="signals.calibrated",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

SIGNALS_FILTERED = TopicConfig(
    name="signals.filtered",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

SIGNALS_FINAL = TopicConfig(
    name="signals.final",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

# Trade Topics
TRADES_ORDERS_PENDING = TopicConfig(
    name="trades.orders.pending",
    partitions=3,
    replication_factor=1,
    retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days (audit trail)
)

TRADES_ORDERS_CONFIRMED = TopicConfig(
    name="trades.orders.confirmed",
    partitions=3,
    replication_factor=1,
    retention_ms=30 * 24 * 60 * 60 * 1000,
)

TRADES_EXECUTIONS = TopicConfig(
    name="trades.executions",
    partitions=3,
    replication_factor=1,
    retention_ms=30 * 24 * 60 * 60 * 1000,
)

TRADES_PNL = TopicConfig(
    name="trades.pnl",
    partitions=3,
    replication_factor=1,
    retention_ms=30 * 24 * 60 * 60 * 1000,
)

# Metrics Topics
METRICS_PERFORMANCE = TopicConfig(
    name="metrics.performance.realtime",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

METRICS_MODEL_DRIFT = TopicConfig(
    name="metrics.model.drift",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

METRICS_LATENCY = TopicConfig(
    name="metrics.latency",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

METRICS_ALERTS = TopicConfig(
    name="metrics.alerts",
    partitions=3,
    replication_factor=1,
    retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days
)

# Sentiment Topics (Optional V2+)
SENTIMENT_NEWS = TopicConfig(
    name="sentiment.news.{symbol}",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

SENTIMENT_SOCIAL = TopicConfig(
    name="sentiment.social.{symbol}",
    partitions=3,
    replication_factor=1,
    retention_ms=7 * 24 * 60 * 60 * 1000,
)

# Dead Letter Queue
DLQ = TopicConfig(
    name="dlq.{component}",
    partitions=1,
    replication_factor=1,
    retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days for debugging
)


# Helper functions
def get_all_topic_configs() -> list[TopicConfig]:
    """Get all topic configurations."""
    return [
        MARKET_CANDLES_1M,
        MARKET_CANDLES_5M,
        MARKET_CANDLES_15M,
        MARKET_CANDLES_1H,
        FEATURES_RAW,
        FEATURES_TECHNICAL,
        FEATURES_MULTI_TF,
        FEATURES_AGGREGATED,
        PREDICTIONS_LSTM,
        PREDICTIONS_TRANSFORMER,
        PREDICTIONS_ENSEMBLE,
        SIGNALS_RAW,
        SIGNALS_CALIBRATED,
        SIGNALS_FILTERED,
        SIGNALS_FINAL,
        TRADES_ORDERS_PENDING,
        TRADES_ORDERS_CONFIRMED,
        TRADES_EXECUTIONS,
        TRADES_PNL,
        METRICS_PERFORMANCE,
        METRICS_MODEL_DRIFT,
        METRICS_LATENCY,
        METRICS_ALERTS,
        SENTIMENT_NEWS,
        SENTIMENT_SOCIAL,
        DLQ,
    ]


def get_topic_name(config: TopicConfig, **kwargs) -> str:
    """Get formatted topic name with substitutions."""
    return config.name.format(**kwargs)
