"""Training script for Transformer model."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from loguru import logger

from apps.trainer.data_pipeline import (
    create_walk_forward_splits,
    load_features_from_parquet,
)
from apps.trainer.features import engineer_features, normalize_features
from apps.trainer.models.transformer import TransformerTrendModel
from apps.trainer.train.dataset import TradingDataset
from apps.trainer.train.trainer import train_model
from libs.data.quality import validate_feature_quality


def train_transformer_multi_coin(
    symbols: list[str],
    interval: str = "1m",
    epochs: int = 10,
    batch_size: int = 16,
    sequence_length: int = 100,
    horizon: int = 15,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
    learning_rate: float = 0.0001,
    device: torch.device | None = None,
    data_dir: Path | str = "data/features",
    models_dir: Path | str = "models",
    use_normalized: bool = True,
) -> dict[str, any]:
    """
    Train Transformer model on multiple coins.

    Args:
        symbols: List of trading pair symbols (e.g., ['BTC-USD', 'ETH-USD'])
        interval: Time interval (e.g., '1m', '1h')
        epochs: Number of training epochs
        batch_size: Batch size for training (smaller for Transformer due to memory)
        sequence_length: Input sequence length
        horizon: Prediction horizon (time steps ahead)
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        learning_rate: Learning rate
        device: Device to train on
        data_dir: Directory with feature files
        models_dir: Directory to save models
        use_normalized: Whether to use normalized features

    Returns:
        Dictionary with training results
    """
    logger.info(f"Training Transformer for {symbols} ({interval})")
    logger.info(f"  Sequence length: {sequence_length}, Horizon: {horizon}")
    logger.info(f"  Transformer: d_model={d_model}, nhead={nhead}, layers={num_layers}")

    # Load and combine features from all symbols
    all_dataframes = []
    feature_columns = None

    for symbol in symbols:
        try:
            df = load_features_from_parquet(symbol=symbol, interval=interval, version="latest")
            logger.info(f"Loaded {len(df)} rows for {symbol}")

            # Validate feature quality
            quality_report = validate_feature_quality(df)
            if not quality_report.is_valid:
                logger.warning(f"Feature quality validation failed for {symbol}")
                logger.warning(quality_report)
                continue

            all_dataframes.append(df)
        except FileNotFoundError:
            logger.error(f"Feature file not found for {symbol} {interval}")
            logger.error("Please run feature engineering first: python scripts/engineer_features.py")
            raise

    if not all_dataframes:
        raise ValueError("No valid data found for any symbol")

    # Combine all dataframes
    df = pd.concat(all_dataframes, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Combined dataset: {len(df)} rows from {len(symbols)} symbols")

    # Get feature columns (exclude OHLCV + timestamp + symbol-specific)
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
    train_end = df["timestamp"].quantile(0.7)
    val_end = df["timestamp"].quantile(0.85)

    train_df, val_df, test_df = create_walk_forward_splits(df, train_end, val_end)
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
        prediction_type="trend",  # Transformer predicts trend strength
    )
    val_dataset = TradingDataset(
        val_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
        prediction_type="trend",
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    logger.info("Creating Transformer model...")
    model = TransformerTrendModel(
        input_size=len(feature_columns),
        d_model=d_model,
        nhead=nhead,
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
        model_name=f"transformer_multi_{'_'.join([s.replace('-', '_') for s in symbols])}_{interval}",
    )

    logger.info(f"✅ Training complete for Transformer!")
    logger.info(f"  Best validation accuracy: {results['best_val_accuracy']:.4f}")

    return results


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Transformer model for cryptocurrency trading")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC-USD", "ETH-USD"],
        help="Trading pairs (default: BTC-USD ETH-USD)",
    )
    parser.add_argument("--interval", default="1m", help="Time interval (default: 1m)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (smaller for Transformer)")
    parser.add_argument("--sequence-length", type=int, default=100, help="Input sequence length")
    parser.add_argument("--horizon", type=int, default=15, help="Prediction horizon (time steps)")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--no-normalize", action="store_true", help="Disable feature normalization")
    parser.add_argument("--data-dir", default="data/features", help="Directory with feature files")
    parser.add_argument("--models-dir", default="models", help="Directory to save models")

    args = parser.parse_args()

    # Import pandas here to avoid circular import
    import pandas as pd
    globals()["pd"] = pd

    train_transformer_multi_coin(
        symbols=args.symbols,
        interval=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        use_normalized=not args.no_normalize,
    )


if __name__ == "__main__":
    main()

