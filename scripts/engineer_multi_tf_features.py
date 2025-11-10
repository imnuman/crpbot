#!/usr/bin/env python3
"""Engineer features with multi-timeframe support for Phase 3.5."""
import argparse
from pathlib import Path

from loguru import logger

from apps.trainer.features import engineer_features
from apps.trainer.multi_tf_features import engineer_multi_tf_features


def main():
    """Engineer features with multi-TF support."""
    parser = argparse.ArgumentParser(description="Engineer multi-TF features")
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., BTC-USD)",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="1m,5m,15m,1h",
        help="Comma-separated list of intervals (default: 1m,5m,15m,1h)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory with raw data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features",
        help="Output directory for engineered features",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-11-10",
        help="Start date for data",
    )

    args = parser.parse_args()
    intervals = args.intervals.split(",")

    logger.info("=" * 80)
    logger.info(f"Multi-TF Feature Engineering: {args.symbol}")
    logger.info("=" * 80)
    logger.info(f"  Intervals: {intervals}")
    logger.info(f"  Data dir: {args.data_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info("")

    # Step 1: Engineer multi-TF features
    logger.info("Step 1: Engineering multi-TF features...")
    df = engineer_multi_tf_features(
        symbol=args.symbol,
        intervals=intervals,
        data_dir=args.data_dir,
        start_date=args.start_date,
    )

    logger.info(f"Multi-TF features: {len(df)} rows, {len(df.columns)} columns")

    # Step 2: Engineer standard V1 features on top of multi-TF
    logger.info("")
    logger.info("Step 2: Engineering standard features on multi-TF base...")
    df_with_features = engineer_features(df)

    logger.info(f"Final features: {len(df_with_features)} rows, {len(df_with_features.columns)} columns")

    # Step 3: Save to output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interval_str = intervals[0]  # Use base interval for filename
    output_file = output_dir / f"features_{args.symbol}_{interval_str}_latest.parquet"

    logger.info("")
    logger.info(f"Saving features to: {output_file}")
    df_with_features.to_parquet(output_file, index=False)

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ Feature engineering complete!")
    logger.info("=" * 80)
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Rows: {len(df_with_features):,}")
    logger.info(f"  Columns: {len(df_with_features.columns)}")
    logger.info(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Log feature categories
    multi_tf_cols = [c for c in df_with_features.columns if any(x in c for x in ["5m_", "15m_", "1h_"])]
    v1_cols = [c for c in df_with_features.columns if c not in multi_tf_cols and c not in ["timestamp", "open", "high", "low", "close", "volume"]]

    logger.info("")
    logger.info("Feature breakdown:")
    logger.info(f"  Multi-TF features: {len(multi_tf_cols)}")
    logger.info(f"  V1 features: {len(v1_cols)}")
    logger.info(f"  OHLCV: 6")
    logger.info(f"  Total: {len(df_with_features.columns)}")


if __name__ == "__main__":
    main()
