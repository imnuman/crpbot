"""Training script for LSTM model."""
import argparse
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from apps.trainer.data_pipeline import (
    create_walk_forward_splits,
    load_features_from_parquet,
)
from apps.trainer.features import normalize_features
from apps.trainer.models.lstm import LSTMDirectionModel
from apps.trainer.train.dataset import TradingDataset
from apps.trainer.train.trainer import train_model
from libs.data.quality import validate_feature_quality


def train_lstm_for_coin(
    symbol: str,
    interval: str = "1m",
    epochs: int = 10,
    batch_size: int = 32,
    sequence_length: int = 60,
    horizon: int = 15,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    device: torch.device | None = None,
    data_dir: Path | str = "data/features",
    models_dir: Path | str = "models",
    use_normalized: bool = True,
) -> dict[str, any]:
    """
    Train LSTM model for a specific coin.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC-USD')
        interval: Time interval (e.g., '1m', '1h')
        epochs: Number of training epochs
        batch_size: Batch size for training
        sequence_length: Input sequence length
        horizon: Prediction horizon (time steps ahead)
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        learning_rate: Learning rate
        device: Device to train on
        data_dir: Directory with feature files
        models_dir: Directory to save models
        use_normalized: Whether to use normalized features

    Returns:
        Dictionary with training results
    """
    logger.info(f"Training LSTM for {symbol} ({interval})")
    logger.info(f"  Sequence length: {sequence_length}, Horizon: {horizon}")

    # Load features
    try:
        df = load_features_from_parquet(symbol=symbol, interval=interval, version="latest")
        logger.info(f"Loaded {len(df)} rows of features")
    except FileNotFoundError:
        logger.error(f"Feature file not found for {symbol} {interval}")
        logger.error("Please run feature engineering first: python scripts/engineer_features.py")
        raise

    # Validate feature quality
    logger.info("Validating feature quality...")
    quality_report = validate_feature_quality(df)
    if not quality_report.is_valid:
        logger.error("Feature quality validation failed!")
        logger.error(quality_report)
        raise ValueError("Feature quality checks failed")
    logger.info("✅ Feature quality validated")

    # Get feature columns (exclude OHLCV + timestamp)
    exclude_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "session",
        "volatility_regime",
    ]
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Using {len(feature_columns)} features: {feature_columns[:5]}...")

    # Create walk-forward splits
    logger.info("Creating walk-forward splits...")
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    train_end = df_sorted["timestamp"].quantile(0.7)
    val_end = df_sorted["timestamp"].quantile(0.85)

    train_df, val_df, test_df = create_walk_forward_splits(df_sorted, train_end, val_end)
    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val: {len(val_df)} rows")
    logger.info(f"  Test: {len(test_df)} rows")

    # Normalize features (fit on train, apply to val/test)
    if use_normalized:
        logger.info("Normalizing features...")
        train_df, norm_params = normalize_features(
            train_df, feature_columns=feature_columns, method="standard"
        )
        val_df, _ = normalize_features(
            val_df, feature_columns=feature_columns, method="standard", fit_data=train_df
        )
        test_df, _ = normalize_features(
            test_df, feature_columns=feature_columns, method="standard", fit_data=train_df
        )
        logger.info("✅ Features normalized")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TradingDataset(
        train_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
        prediction_type="direction",
    )
    val_dataset = TradingDataset(
        val_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
        prediction_type="direction",
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    logger.info("Creating LSTM model...")
    model = LSTMDirectionModel(
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=models_dir,
        model_name=f"lstm_{symbol.replace('-', '_')}_{interval}",
    )

    logger.info(f"✅ Training complete for {symbol}!")
    logger.info(f"  Best validation accuracy: {results['best_val_accuracy']:.4f}")

    return results


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train LSTM model for cryptocurrency trading")
    parser.add_argument("--symbol", required=True, help="Trading pair (e.g., BTC-USD)")
    parser.add_argument("--interval", default="1m", help="Time interval (default: 1m)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=60, help="Input sequence length")
    parser.add_argument("--horizon", type=int, default=15, help="Prediction horizon (time steps)")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-normalize", action="store_true", help="Disable feature normalization")
    parser.add_argument("--data-dir", default="data/features", help="Directory with feature files")
    parser.add_argument("--models-dir", default="models", help="Directory to save models")

    args = parser.parse_args()

    train_lstm_for_coin(
        symbol=args.symbol,
        interval=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        use_normalized=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
