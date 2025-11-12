"""Base Kafka producer for crpbot."""

import json
from typing import Any, Callable, Optional

from confluent_kafka import Producer
from loguru import logger


class KafkaProducerBase:
    """Base class for Kafka producers."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        compression_type: str = "lz4",
        enable_idempotence: bool = True,
        acks: str = "all",
        max_in_flight_requests: int = 5,
        **kwargs,
    ):
        """Initialize Kafka producer.

        Args:
            bootstrap_servers: Kafka broker addresses
            compression_type: Compression algorithm (lz4, snappy, gzip, none)
            enable_idempotence: Enable idempotent producer
            acks: Acknowledgment mode (all, 1, 0)
            max_in_flight_requests: Max unacknowledged requests
            **kwargs: Additional producer configuration
        """
        self.config = {
            "bootstrap.servers": bootstrap_servers,
            "compression.type": compression_type,
            "enable.idempotence": enable_idempotence,
            "acks": acks,
            "max.in.flight.requests.per.connection": max_in_flight_requests,
            **kwargs,
        }

        self.producer = Producer(self.config)
        logger.info(f"Initialized Kafka producer: {bootstrap_servers}")

    def produce(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """Produce a message to a Kafka topic.

        Args:
            topic: Topic name
            value: Message value (will be JSON-serialized)
            key: Message key (optional)
            headers: Message headers (optional)
            callback: Delivery report callback (optional)
        """
        try:
            # Serialize value to JSON
            value_bytes = json.dumps(value).encode("utf-8")

            # Serialize key if provided
            key_bytes = key.encode("utf-8") if key else None

            # Convert headers to list of tuples if provided
            headers_list = None
            if headers:
                headers_list = [(k, v.encode("utf-8")) for k, v in headers.items()]

            # Produce message
            self.producer.produce(
                topic=topic,
                value=value_bytes,
                key=key_bytes,
                headers=headers_list,
                callback=callback or self._default_delivery_report,
            )

            # Poll for delivery reports (non-blocking)
            self.producer.poll(0)

        except BufferError:
            logger.warning(f"Producer queue full, waiting for messages to be delivered")
            self.producer.flush()
            # Retry
            self.produce(topic, value, key, headers, callback)

        except Exception as e:
            logger.error(f"Error producing message to {topic}: {e}")
            raise

    def produce_batch(
        self,
        topic: str,
        messages: list[dict[str, Any]],
        key_fn: Optional[Callable[[dict], str]] = None,
    ) -> None:
        """Produce a batch of messages to a Kafka topic.

        Args:
            topic: Topic name
            messages: List of message values
            key_fn: Function to extract key from message (optional)
        """
        for msg in messages:
            key = key_fn(msg) if key_fn else None
            self.produce(topic, msg, key=key)

        # Flush all pending messages
        self.flush()

    def flush(self, timeout: float = 10.0) -> int:
        """Wait for all messages to be delivered.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Number of messages still in queue
        """
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            logger.warning(f"Failed to flush {remaining} messages within {timeout}s")
        return remaining

    def close(self) -> None:
        """Close the producer and flush all pending messages."""
        logger.info("Closing Kafka producer")
        self.flush()
        # Producer doesn't have explicit close method, flush is sufficient

    def _default_delivery_report(self, err, msg):
        """Default delivery report callback.

        Args:
            err: Error if delivery failed
            msg: Message that was delivered or failed
        """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(
                f"Message delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}"
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
