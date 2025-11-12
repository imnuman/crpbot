"""Model inference consumer for real-time predictions.

Consumes: features.aggregated.{symbol}
Produces: predictions.lstm.{symbol}

Loads trained LSTM models and runs real-time inference on incoming features.
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from apps.kafka.config.topics import (
    FEATURES_AGGREGATED,
    PREDICTIONS_LSTM,
    get_topic_name,
)
from apps.kafka.consumers.base import KafkaConsumerBase
from apps.kafka.producers.base import KafkaProducerBase
from apps.trainer.models.lstm import LSTMDirectionModel


class ModelInferenceWorker:
    """Real-time model inference worker."""

    def __init__(
        self,
        symbols: list[str],
        model_dir: str = "models/production",
        bootstrap_servers: str = "localhost:9092",
        sequence_length: int = 60,
        device: str = "cpu",
    ):
        """Initialize model inference worker.

        Args:
            symbols: List of symbols to process (e.g., ["BTC-USD", "ETH-USD"])
            model_dir: Directory containing trained models
            bootstrap_servers: Kafka broker addresses
            sequence_length: Sequence length for LSTM (must match training)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.symbols = symbols
        self.model_dir = Path(model_dir)
        self.bootstrap_servers = bootstrap_servers
        self.sequence_length = sequence_length
        self.device = torch.device(device)

        # Initialize Kafka consumer and producer
        input_topics = [
            get_topic_name(FEATURES_AGGREGATED, symbol=symbol) for symbol in symbols
        ]

        self.consumer = KafkaConsumerBase(
            topics=input_topics,
            group_id="model-inference-lstm",
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",
        )

        self.producer = KafkaProducerBase(bootstrap_servers=bootstrap_servers)

        # Load models for each symbol
        self.models: dict[str, torch.nn.Module] = {}
        self.model_metadata: dict[str, dict] = {}
        self._load_models()

        # Maintain sequence buffer for each symbol
        self.buffers: dict[str, deque] = {
            symbol: deque(maxlen=sequence_length) for symbol in symbols
        }

        # Track inference stats
        self.stats = {
            "features_consumed": 0,
            "predictions_produced": 0,
            "errors": 0,
            "inference_times_ms": [],
        }

        logger.info(
            f"Initialized ModelInferenceWorker: symbols={symbols}, device={device}"
        )

    def _load_models(self) -> None:
        """Load trained models for all symbols."""
        for symbol in self.symbols:
            try:
                # Look for latest model checkpoint
                model_pattern = f"lstm_{symbol.replace('-', '_')}_*.pt"
                model_files = list(self.model_dir.glob(model_pattern))

                if not model_files:
                    logger.warning(
                        f"No model found for {symbol} in {self.model_dir}. "
                        f"Skipping inference for this symbol."
                    )
                    continue

                # Get the latest model (by modification time)
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

                # Load model checkpoint
                checkpoint = torch.load(latest_model, map_location=self.device)

                # Load metadata
                metadata = checkpoint.get("metadata", {})
                self.model_metadata[symbol] = metadata

                # Extract model config
                input_size = metadata.get("input_size")
                hidden_size = metadata.get("hidden_size", 64)
                num_layers = metadata.get("num_layers", 2)
                dropout = metadata.get("dropout", 0.2)
                bidirectional = metadata.get("bidirectional", False)

                if not input_size:
                    logger.error(
                        f"No input_size in metadata for {symbol}. Cannot load model."
                    )
                    continue

                # Create model instance
                model = LSTMDirectionModel(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )

                # Load weights
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()

                self.models[symbol] = model

                logger.info(
                    f"Loaded model for {symbol}: {latest_model.name} "
                    f"(input_size={input_size}, version={metadata.get('version', 'unknown')})"
                )

            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {e}")

    def start(self) -> None:
        """Start consuming features and producing predictions."""
        logger.info("Starting model inference worker...")

        try:
            self.consumer.consume(
                message_handler=self._process_features,
                poll_timeout=1.0,
            )
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.stop()

    def _process_features(self, message: dict, metadata: dict) -> None:
        """Process features and run model inference.

        Args:
            message: Feature message from Kafka
            metadata: Message metadata
        """
        try:
            symbol = message.get("symbol")
            features_dict = message.get("features")

            if not symbol or not features_dict:
                logger.warning("Invalid message format")
                return

            if symbol not in self.models:
                logger.debug(f"No model loaded for {symbol}, skipping")
                return

            # Add features to buffer
            self.buffers[symbol].append(features_dict)
            self.stats["features_consumed"] += 1

            # Check if we have enough features for a sequence
            if len(self.buffers[symbol]) < self.sequence_length:
                logger.debug(
                    f"Buffer not ready for {symbol}: {len(self.buffers[symbol])}/{self.sequence_length}"
                )
                return

            # Run inference
            prediction = self._run_inference(symbol)

            if prediction:
                # Produce prediction to Kafka
                self._produce_prediction(symbol, prediction)
                self.stats["predictions_produced"] += 1

                # Log progress every 10 predictions
                if self.stats["predictions_produced"] % 10 == 0:
                    avg_time = np.mean(self.stats["inference_times_ms"][-100:])
                    logger.info(
                        f"Inference stats: {self.stats['predictions_produced']} predictions, "
                        f"{self.stats['features_consumed']} features consumed, "
                        f"avg inference time: {avg_time:.2f}ms"
                    )

        except Exception as e:
            logger.error(f"Error processing features: {e}")
            self.stats["errors"] += 1

    def _run_inference(self, symbol: str) -> Optional[dict]:
        """Run model inference for a symbol.

        Args:
            symbol: Symbol to run inference for

        Returns:
            Prediction dictionary or None if inference failed
        """
        try:
            start_time = datetime.utcnow()

            # Get model and metadata
            model = self.models[symbol]
            metadata = self.model_metadata[symbol]

            # Convert buffer to numpy array
            feature_names = metadata.get("feature_names", [])
            buffer_list = list(self.buffers[symbol])

            # Extract features in correct order
            sequence = []
            for features_dict in buffer_list:
                # Extract only the features the model was trained on
                feature_vector = [
                    features_dict.get(name, 0.0) for name in feature_names
                ]
                sequence.append(feature_vector)

            sequence_array = np.array(sequence, dtype=np.float32)

            # Convert to tensor
            x = torch.from_numpy(sequence_array).unsqueeze(0).to(self.device)  # (1, seq_len, features)

            # Run inference
            with torch.no_grad():
                output = model(x)

            # Extract prediction
            probability = output.item()  # Probability of upward movement
            direction = 1 if probability > 0.5 else -1

            # Calculate confidence (distance from 0.5)
            confidence = abs(probability - 0.5) * 2  # Scale to 0-1

            # Calculate inference time
            inference_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats["inference_times_ms"].append(inference_time_ms)

            return {
                "direction": direction,
                "probability": float(probability),
                "confidence": float(confidence),
                "inference_time_ms": float(inference_time_ms),
            }

        except Exception as e:
            logger.error(f"Error running inference for {symbol}: {e}")
            return None

    def _produce_prediction(self, symbol: str, prediction: dict) -> None:
        """Produce prediction to Kafka.

        Args:
            symbol: Symbol
            prediction: Prediction dictionary
        """
        # Get output topic
        topic = get_topic_name(PREDICTIONS_LSTM, symbol=symbol)

        # Get model metadata
        metadata = self.model_metadata.get(symbol, {})

        # Create message
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "model_type": "lstm",
            "model_version": metadata.get("version", "unknown"),
            "prediction": prediction,
        }

        # Produce to Kafka
        self.producer.produce(
            topic=topic,
            value=message,
            key=symbol,
            headers={"symbol": symbol, "model_type": "lstm"},
        )

        logger.debug(
            f"Produced prediction for {symbol}: direction={prediction['direction']}, "
            f"confidence={prediction['confidence']:.2f}"
        )

    def stop(self) -> None:
        """Stop the inference worker."""
        logger.info(f"Stopping model inference worker. Final stats: {self.stats}")
        self.consumer.close()
        self.producer.close()


def main():
    """Main entry point for model inference worker."""
    # Configuration
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    model_dir = "models/production"
    bootstrap_servers = "localhost:9092"
    sequence_length = 60
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create and start worker
    worker = ModelInferenceWorker(
        symbols=symbols,
        model_dir=model_dir,
        bootstrap_servers=bootstrap_servers,
        sequence_length=sequence_length,
        device=device,
    )

    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        worker.stop()


if __name__ == "__main__":
    main()
