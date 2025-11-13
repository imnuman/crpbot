#!/usr/bin/env python3
"""Evaluate a trained model with backtest and promotion gates."""
import argparse
from pathlib import Path

import torch

from loguru import logger

from apps.trainer.data_pipeline import create_walk_forward_splits, load_features_from_parquet
from apps.trainer.features import normalize_features
from apps.trainer.models.lstm import LSTMDirectionModel
from apps.trainer.models.transformer import TransformerTrendModel
from apps.trainer.train.dataset import TradingDataset
from apps.trainer.eval.evaluator import ModelEvaluator
from apps.trainer.eval.tracking import ExperimentTracker
from libs.data.quality import validate_feature_quality
from libs.rl_env.execution_model import ExecutionModel


def evaluate_model(
    model_path: str,
    symbol: str,
    interval: str = "1m",
    model_type: str = "lstm",
    confidence_threshold: float = 0.5,
    min_accuracy: float = 0.68,
    max_calibration_error: float = 0.05,
) -> bool:
    """
    Evaluate a model and check promotion gates.

    Args:
        model_path: Path to model file
        symbol: Trading pair symbol
        interval: Time interval
        model_type: Type of model ('lstm' or 'transformer')
        confidence_threshold: Confidence threshold for signals
        min_accuracy: Minimum accuracy threshold
        max_calibration_error: Maximum calibration error threshold

    Returns:
        True if model passes promotion gates, False otherwise
    """
    logger.info(f"Evaluating {model_type} model for {symbol}")

    # Load features
    df = load_features_from_parquet(symbol=symbol, interval=interval, version="latest")

    # Validate feature quality
    quality_report = validate_feature_quality(df)
    if not quality_report.is_valid:
        logger.error("Feature quality validation failed!")
        return False

    # Get feature columns
    exclude_cols = ["timestamp", "open", "high", "low", "close", "volume", "session", "volatility_regime"]
    feature_columns = [col for col in df.columns if col not in exclude_cols]

    # Create test split
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    train_end = df_sorted["timestamp"].quantile(0.7)
    val_end = df_sorted["timestamp"].quantile(0.85)

    _, _, test_df = create_walk_forward_splits(df_sorted, train_end, val_end)

    # Normalize features
    test_df, _ = normalize_features(test_df, feature_columns=feature_columns, method="standard")

    # Create test dataset
    if model_type == "lstm":
        test_dataset = TradingDataset(
            test_df,
            feature_columns=feature_columns,
            sequence_length=60,
            horizon=15,
            prediction_type="direction",
            return_metadata=True,  # Enable metadata for evaluation (timestamps, prices)
        )
    else:
        test_dataset = TradingDataset(
            test_df,
            feature_columns=feature_columns,
            sequence_length=100,
            horizon=15,
            prediction_type="trend",
            return_metadata=True,  # Enable metadata for evaluation (timestamps, prices)
        )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if model_type == "lstm":
        model = LSTMDirectionModel(input_size=len(feature_columns))
    else:
        model = TransformerTrendModel(input_size=len(feature_columns))

    # Handle both checkpoint formats:
    # - Local training: dict with "model_state_dict" key
    # - GPU/Colab: direct state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Direct state_dict (GPU/Colab format)
        model.load_state_dict(checkpoint)
    model.to(device)

    # Evaluate
    execution_model = ExecutionModel()
    evaluator = ModelEvaluator(execution_model=execution_model)

    metrics, detailed_results = evaluator.evaluate_model(
        model=model,
        test_dataset=test_dataset,
        symbol=symbol,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    logger.info(f"\n{metrics}")

    # Check promotion gates
    passed, failures = evaluator.check_promotion_gates(
        metrics, symbol=symbol, min_accuracy=min_accuracy, max_calibration_error=max_calibration_error
    )

    # Register in tracking
    tracker = ExperimentTracker()
    tracker.register_model(
        model_path=model_path,
        model_type=model_type.upper(),
        symbol=symbol,
        metrics={
            "win_rate": metrics.win_rate,
            "total_pnl": metrics.total_pnl,
            "max_drawdown": metrics.max_drawdown,
            "calibration_error": metrics.calibration_error,
            "brier_score": metrics.brier_score,
        },
    )

    if passed:
        logger.info("✅ Model passes all promotion gates!")
    else:
        logger.warning("❌ Model does not pass promotion gates")

    return passed


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--symbol", required=True, help="Trading pair (e.g., BTC-USD)")
    parser.add_argument("--interval", default="1m", help="Time interval")
    parser.add_argument("--model-type", choices=["lstm", "transformer"], default="lstm", help="Model type")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--min-accuracy", type=float, default=0.68, help="Minimum accuracy threshold")
    parser.add_argument(
        "--max-calibration-error", type=float, default=0.05, help="Maximum calibration error threshold"
    )

    args = parser.parse_args()

    passed = evaluate_model(
        model_path=args.model,
        symbol=args.symbol,
        interval=args.interval,
        model_type=args.model_type,
        confidence_threshold=args.confidence_threshold,
        min_accuracy=args.min_accuracy,
        max_calibration_error=args.max_calibration_error,
    )

    exit(0 if passed else 1)


if __name__ == "__main__":
    main()

