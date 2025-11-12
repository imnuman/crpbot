"""Base Kafka consumer for crpbot."""

import json
import signal
import sys
from typing import Any, Callable, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException
from loguru import logger


class KafkaConsumerBase:
    """Base class for Kafka consumers."""

    def __init__(
        self,
        topics: list[str],
        group_id: str,
        bootstrap_servers: str = "localhost:9092",
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
        max_poll_interval_ms: int = 300000,  # 5 minutes
        session_timeout_ms: int = 30000,  # 30 seconds
        **kwargs,
    ):
        """Initialize Kafka consumer.

        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            bootstrap_servers: Kafka broker addresses
            auto_offset_reset: Where to start reading (earliest, latest, none)
            enable_auto_commit: Enable automatic offset commits
            max_poll_interval_ms: Max time between polls before rebalance
            session_timeout_ms: Session timeout before rebalance
            **kwargs: Additional consumer configuration
        """
        self.topics = topics
        self.group_id = group_id
        self.running = False

        self.config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": enable_auto_commit,
            "max.poll.interval.ms": max_poll_interval_ms,
            "session.timeout.ms": session_timeout_ms,
            **kwargs,
        }

        self.consumer = Consumer(self.config)
        self.consumer.subscribe(topics)

        logger.info(
            f"Initialized Kafka consumer: group={group_id}, topics={topics}, servers={bootstrap_servers}"
        )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def consume(
        self,
        message_handler: Callable[[dict[str, Any]], None],
        poll_timeout: float = 1.0,
        max_messages: Optional[int] = None,
    ) -> None:
        """Start consuming messages.

        Args:
            message_handler: Function to process each message
            poll_timeout: Timeout for polling in seconds
            max_messages: Maximum messages to consume (None = infinite)
        """
        self.running = True
        message_count = 0

        logger.info(f"Starting consumer loop for topics: {self.topics}")

        try:
            while self.running:
                # Check max_messages limit
                if max_messages is not None and message_count >= max_messages:
                    logger.info(f"Reached max_messages limit: {max_messages}")
                    break

                # Poll for messages
                msg = self.consumer.poll(timeout=poll_timeout)

                if msg is None:
                    continue

                if msg.error():
                    self._handle_error(msg.error())
                    continue

                try:
                    # Deserialize message
                    value = json.loads(msg.value().decode("utf-8"))
                    key = msg.key().decode("utf-8") if msg.key() else None

                    # Extract headers
                    headers = {}
                    if msg.headers():
                        headers = {k: v.decode("utf-8") for k, v in msg.headers()}

                    # Create message metadata
                    metadata = {
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset(),
                        "timestamp": msg.timestamp()[1],  # (type, timestamp_ms)
                        "key": key,
                        "headers": headers,
                    }

                    # Process message
                    logger.debug(
                        f"Processing message from {metadata['topic']} [{metadata['partition']}] @ {metadata['offset']}"
                    )
                    message_handler(value, metadata)

                    # Manually commit offset if auto-commit is disabled
                    if not self.config.get("enable.auto.commit", False):
                        self.consumer.commit(asynchronous=False)

                    message_count += 1

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                    self._handle_failed_message(msg, e)

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self._handle_failed_message(msg, e)

        except KafkaException as e:
            logger.error(f"Kafka exception: {e}")
            raise

        finally:
            self.close()
            logger.info(f"Consumer stopped after processing {message_count} messages")

    def consume_batch(
        self,
        batch_handler: Callable[[list[dict[str, Any]]], None],
        batch_size: int = 100,
        batch_timeout: float = 5.0,
        max_batches: Optional[int] = None,
    ) -> None:
        """Consume messages in batches.

        Args:
            batch_handler: Function to process each batch
            batch_size: Maximum messages per batch
            batch_timeout: Maximum time to wait for batch in seconds
            max_batches: Maximum batches to consume (None = infinite)
        """
        self.running = True
        batch_count = 0

        logger.info(
            f"Starting batch consumer: batch_size={batch_size}, timeout={batch_timeout}s"
        )

        try:
            while self.running:
                # Check max_batches limit
                if max_batches is not None and batch_count >= max_batches:
                    logger.info(f"Reached max_batches limit: {max_batches}")
                    break

                # Consume batch
                messages = self.consumer.consume(
                    num_messages=batch_size, timeout=batch_timeout
                )

                if not messages:
                    continue

                # Process batch
                batch = []
                last_msg = None

                for msg in messages:
                    if msg.error():
                        self._handle_error(msg.error())
                        continue

                    try:
                        value = json.loads(msg.value().decode("utf-8"))
                        batch.append(value)
                        last_msg = msg

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode message: {e}")
                        self._handle_failed_message(msg, e)

                if batch:
                    logger.debug(f"Processing batch of {len(batch)} messages")
                    batch_handler(batch)

                    # Commit last message offset
                    if last_msg and not self.config.get("enable.auto.commit", False):
                        self.consumer.commit(message=last_msg, asynchronous=False)

                    batch_count += 1

        except KafkaException as e:
            logger.error(f"Kafka exception: {e}")
            raise

        finally:
            self.close()
            logger.info(f"Batch consumer stopped after processing {batch_count} batches")

    def seek_to_beginning(self) -> None:
        """Seek to the beginning of all assigned partitions."""
        self.consumer.seek_to_beginning()
        logger.info("Seeked to beginning of all partitions")

    def seek_to_end(self) -> None:
        """Seek to the end of all assigned partitions."""
        self.consumer.seek_to_end()
        logger.info("Seeked to end of all partitions")

    def close(self) -> None:
        """Close the consumer."""
        logger.info("Closing Kafka consumer")
        self.running = False
        self.consumer.close()

    def _handle_error(self, error: KafkaError) -> None:
        """Handle Kafka errors.

        Args:
            error: Kafka error
        """
        if error.code() == KafkaError._PARTITION_EOF:
            # End of partition - not an error
            logger.debug(f"Reached end of partition: {error}")
        elif error.code() == KafkaError._ALL_BROKERS_DOWN:
            logger.critical("All brokers are down!")
            self.running = False
        else:
            logger.error(f"Kafka error: {error}")

    def _handle_failed_message(self, msg, error: Exception) -> None:
        """Handle failed message processing.

        Args:
            msg: Failed message
            error: Exception that occurred
        """
        # Log error with message details
        logger.error(
            f"Failed to process message from {msg.topic()} [{msg.partition()}] @ {msg.offset()}: {error}"
        )

        # TODO: Send to dead letter queue
        # dlq_topic = f"dlq.{self.group_id}"
        # self.producer.produce(dlq_topic, msg.value(), msg.key())

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
