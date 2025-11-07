#!/usr/bin/env python3
"""Script to engineer features from raw OHLCV data."""
import argparse
from pathlib import Path

from loguru import logger

from apps.trainer.data_pipeline import (
    clean_and_validate_data,
    load_data,
    load_features_from_parquet,
    save_features_to_parquet,
)
from apps.trainer.features import engineer_features, normalize_features


def main():
    """Engineer features from raw data."""
    parser = argparse.ArgumentParser(description="Engineer features from raw OHLCV data")
    parser.add_argument("--input", required=True, help="Input parquet file with raw OHLCV data")
    parser.add_argument(
        "--output", default=None, help="Output directory (default: data/features)"
    )
    parser.add_argument("--symbol", default=None, help="Symbol (auto-detected from filename if not provided)")
    parser.add_argument("--interval", default=None, help="Interval (auto-detected from filename if not provided)")
    parser.add_argument("--version", default=None, help="Version string (default: date-based)")
    parser.add_argument("--normalize", action="store_true", help="Normalize features")
    parser.add_argument("--normalize-method", default="standard", choices=["standard", "minmax", "robust"], help="Normalization method")
    parser.add_argument("--no-session", action="store_true", help="Skip session features")
    parser.add_argument("--no-technical", action="store_true", help="Skip technical indicators")
    parser.add_argument("--no-spread", action="store_true", help="Skip spread features")
    parser.add_argument("--no-volume", action="store_true", help="Skip volume features")
    parser.add_argument("--no-volatility", action="store_true", help="Skip volatility regime")

    args = parser.parse_args()

    # Load raw data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading raw data from {input_path}")
    df = load_data(input_path)

    if df.empty:
        logger.error("Input DataFrame is empty!")
        return

    # Auto-detect symbol and interval from filename if not provided
    symbol = args.symbol
    interval = args.interval
    if not symbol or not interval:
        # Try to extract from filename: {symbol}_{interval}_... or test_{symbol}_{interval}_...
        filename = input_path.stem
        parts = filename.split("_")
        if len(parts) >= 2:
            if parts[0] == "test":
                # test_BTC-USD_1h_7d.parquet
                symbol = symbol or parts[1] if len(parts) > 1 else None
                interval = interval or parts[2] if len(parts) > 2 else None
            else:
                # BTC-USD_1m_2024-01-01_2024-01-01.parquet
                symbol = symbol or parts[0]
                interval = interval or parts[1] if len(parts) > 1 else None

    if not symbol or not interval:
        logger.error("Could not auto-detect symbol/interval. Please provide --symbol and --interval")
        return

    logger.info(f"Processing: {symbol} {interval}")

    # Engineer features
    df_features = engineer_features(
        df,
        add_session_features_flag=not args.no_session,
        add_technical_indicators_flag=not args.no_technical,
        add_spread_features_flag=not args.no_spread,
        add_volume_features_flag=not args.no_volume,
        add_volatility_regime_flag=not args.no_volatility,
    )

    # Normalize if requested
    if args.normalize:
        logger.info(f"Normalizing features using {args.normalize_method} method")
        df_features, norm_params = normalize_features(df_features, method=args.normalize_method)
        logger.info(f"Normalization parameters: {len(norm_params)} features normalized")

    # Save features
    output_dir = Path(args.output) if args.output else Path("data/features")
    feature_file = save_features_to_parquet(
        df_features, symbol=symbol, interval=interval, version=args.version, base_dir=output_dir
    )

    logger.info(f"✅ Features engineered and saved to {feature_file}")
    logger.info(f"   Total features: {len(df_features.columns)}")
    logger.info(f"   Rows: {len(df_features)}")
    logger.info(f"   Date range: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")


if __name__ == "__main__":
    main()

