"""Signal aggregator consumer for ensemble predictions.

Consumes: predictions.lstm.{symbol}, predictions.transformer
Produces: signals.raw, signals.calibrated

Aggregates predictions from multiple models and applies confidence calibration.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

from apps.kafka.config.topics import (
    PREDICTIONS_LSTM,
    SIGNALS_CALIBRATED,
    SIGNALS_RAW,
    get_topic_name,
)
from apps.kafka.consumers.base import KafkaConsumerBase
from apps.kafka.producers.base import KafkaProducerBase


class SignalAggregator:
    """Aggregates predictions from multiple models into trading signals."""

    def __init__(
        self,
        symbols: list[str],
        bootstrap_servers: str = "localhost:9092",
        ensemble_weights: Optional[dict[str, float]] = None,
        confidence_threshold: float = 0.6,
        signal_timeout_seconds: int = 300,  # 5 minutes
    ):
        """Initialize signal aggregator.

        Args:
            symbols: List of symbols to process
            bootstrap_servers: Kafka broker addresses
            ensemble_weights: Weights for each model type (e.g., {"lstm": 0.6, "transformer": 0.4})
            confidence_threshold: Minimum confidence to generate signal
            signal_timeout_seconds: Time to wait for all model predictions before aggregating
        """
        self.symbols = symbols
        self.bootstrap_servers = bootstrap_servers
        self.ensemble_weights = ensemble_weights or {"lstm": 1.0}  # Default: LSTM only
        self.confidence_threshold = confidence_threshold
        self.signal_timeout_seconds = signal_timeout_seconds

        # Initialize Kafka consumer and producer
        input_topics = [
            get_topic_name(PREDICTIONS_LSTM, symbol=symbol) for symbol in symbols
        ]
        # TODO: Add transformer predictions when implemented
        # input_topics.append(get_topic_name(PREDICTIONS_TRANSFORMER, symbol=symbol))

        self.consumer = KafkaConsumerBase(
            topics=input_topics,
            group_id="signal-aggregator",
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",
        )

        self.producer = KafkaProducerBase(bootstrap_servers=bootstrap_servers)

        # Buffer predictions awaiting ensemble
        # Structure: {symbol: {model_type: prediction}}
        self.prediction_buffer: dict[str, dict[str, dict]] = defaultdict(dict)
        self.buffer_timestamps: dict[str, datetime] = {}

        # Track signal stats
        self.stats = {
            "predictions_consumed": 0,
            "signals_produced": 0,
            "filtered_signals": 0,
            "errors": 0,
        }

        logger.info(
            f"Initialized SignalAggregator: symbols={symbols}, "
            f"ensemble_weights={ensemble_weights}"
        )

    def start(self) -> None:
        """Start consuming predictions and producing signals."""
        logger.info("Starting signal aggregator...")

        try:
            self.consumer.consume(
                message_handler=self._process_prediction,
                poll_timeout=1.0,
            )
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.stop()

    def _process_prediction(self, message: dict, metadata: dict) -> None:
        """Process a prediction and aggregate signals.

        Args:
            message: Prediction message from Kafka
            metadata: Message metadata
        """
        try:
            symbol = message.get("symbol")
            model_type = message.get("model_type")
            prediction = message.get("prediction")

            if not symbol or not model_type or not prediction:
                logger.warning("Invalid message format")
                return

            # Add prediction to buffer
            self.prediction_buffer[symbol][model_type] = prediction
            self.buffer_timestamps[symbol] = datetime.utcnow()
            self.stats["predictions_consumed"] += 1

            # Check if we have all predictions for ensemble
            if self._has_complete_ensemble(symbol):
                # Aggregate predictions into signal
                signal = self._aggregate_predictions(symbol)

                if signal:
                    # Produce raw signal
                    self._produce_signal(symbol, signal, "raw")

                    # Apply confidence calibration
                    calibrated_signal = self._calibrate_confidence(signal)

                    # Produce calibrated signal
                    self._produce_signal(symbol, calibrated_signal, "calibrated")

                    self.stats["signals_produced"] += 1

                    # Log progress
                    if self.stats["signals_produced"] % 10 == 0:
                        logger.info(
                            f"Signal aggregation stats: {self.stats['signals_produced']} signals produced, "
                            f"{self.stats['predictions_consumed']} predictions consumed"
                        )

                # Clear buffer for this symbol
                self.prediction_buffer[symbol].clear()
                self.buffer_timestamps.pop(symbol, None)

            else:
                # Check if buffer is stale (timeout exceeded)
                self._check_stale_buffers()

        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            self.stats["errors"] += 1

    def _has_complete_ensemble(self, symbol: str) -> bool:
        """Check if we have predictions from all required models.

        Args:
            symbol: Symbol to check

        Returns:
            True if all models have provided predictions
        """
        buffer = self.prediction_buffer[symbol]
        required_models = set(self.ensemble_weights.keys())
        available_models = set(buffer.keys())

        return required_models.issubset(available_models)

    def _check_stale_buffers(self) -> None:
        """Check for stale prediction buffers and force aggregation."""
        now = datetime.utcnow()

        for symbol, timestamp in list(self.buffer_timestamps.items()):
            age_seconds = (now - timestamp).total_seconds()

            if age_seconds > self.signal_timeout_seconds:
                logger.warning(
                    f"Prediction buffer for {symbol} is stale ({age_seconds:.0f}s). "
                    f"Forcing aggregation with available predictions."
                )

                # Aggregate with available predictions
                signal = self._aggregate_predictions(symbol)

                if signal:
                    self._produce_signal(symbol, signal, "raw")
                    calibrated_signal = self._calibrate_confidence(signal)
                    self._produce_signal(symbol, calibrated_signal, "calibrated")
                    self.stats["signals_produced"] += 1

                # Clear buffer
                self.prediction_buffer[symbol].clear()
                self.buffer_timestamps.pop(symbol, None)

    def _aggregate_predictions(self, symbol: str) -> Optional[dict]:
        """Aggregate predictions from multiple models.

        Args:
            symbol: Symbol to aggregate

        Returns:
            Aggregated signal dictionary
        """
        try:
            buffer = self.prediction_buffer[symbol]

            if not buffer:
                return None

            # Weighted average of probabilities
            weighted_prob = 0.0
            total_weight = 0.0

            for model_type, prediction in buffer.items():
                weight = self.ensemble_weights.get(model_type, 0.0)
                prob = prediction.get("probability", 0.5)
                weighted_prob += prob * weight
                total_weight += weight

            if total_weight == 0:
                return None

            ensemble_prob = weighted_prob / total_weight

            # Determine direction
            direction = 1 if ensemble_prob > 0.5 else -1

            # Calculate confidence (distance from 0.5, scaled to 0-1)
            confidence = abs(ensemble_prob - 0.5) * 2

            # Aggregate inference times
            inference_times = [
                pred.get("inference_time_ms", 0) for pred in buffer.values()
            ]
            avg_inference_time = np.mean(inference_times) if inference_times else 0

            return {
                "direction": direction,
                "probability": float(ensemble_prob),
                "confidence": float(confidence),
                "ensemble_weights": self.ensemble_weights,
                "models_used": list(buffer.keys()),
                "avg_inference_time_ms": float(avg_inference_time),
            }

        except Exception as e:
            logger.error(f"Error aggregating predictions for {symbol}: {e}")
            return None

    def _calibrate_confidence(self, signal: dict) -> dict:
        """Apply confidence calibration to signal.

        Args:
            signal: Raw signal dictionary

        Returns:
            Calibrated signal dictionary
        """
        # Simple confidence calibration (can be improved with isotonic regression)
        # For now, just apply a threshold filter

        calibrated = signal.copy()

        # If confidence is below threshold, set to neutral
        if calibrated["confidence"] < self.confidence_threshold:
            calibrated["direction"] = 0  # Neutral
            calibrated["filtered"] = True
            self.stats["filtered_signals"] += 1
        else:
            calibrated["filtered"] = False

        return calibrated

    def _produce_signal(self, symbol: str, signal: dict, signal_type: str) -> None:
        """Produce signal to Kafka.

        Args:
            symbol: Symbol
            signal: Signal dictionary
            signal_type: 'raw' or 'calibrated'
        """
        # Get output topic
        if signal_type == "raw":
            topic = get_topic_name(SIGNALS_RAW)
        elif signal_type == "calibrated":
            topic = get_topic_name(SIGNALS_CALIBRATED)
        else:
            logger.error(f"Unknown signal type: {signal_type}")
            return

        # Create message
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "signal": signal,
            "signal_type": signal_type,
        }

        # Produce to Kafka
        self.producer.produce(
            topic=topic,
            value=message,
            key=symbol,
            headers={"symbol": symbol, "signal_type": signal_type},
        )

        logger.debug(
            f"Produced {signal_type} signal for {symbol}: direction={signal['direction']}, "
            f"confidence={signal['confidence']:.2f}"
        )

    def stop(self) -> None:
        """Stop the signal aggregator."""
        logger.info(f"Stopping signal aggregator. Final stats: {self.stats}")
        self.consumer.close()
        self.producer.close()


def main():
    """Main entry point for signal aggregator."""
    # Configuration
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    bootstrap_servers = "localhost:9092"
    ensemble_weights = {
        "lstm": 1.0,  # Only LSTM for now
        # "transformer": 0.4,  # Add when transformer is implemented
    }
    confidence_threshold = 0.6

    # Create and start aggregator
    aggregator = SignalAggregator(
        symbols=symbols,
        bootstrap_servers=bootstrap_servers,
        ensemble_weights=ensemble_weights,
        confidence_threshold=confidence_threshold,
    )

    try:
        aggregator.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        aggregator.stop()


if __name__ == "__main__":
    main()
