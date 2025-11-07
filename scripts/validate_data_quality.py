#!/usr/bin/env python3
"""Script to validate data quality for raw and engineered features."""
import argparse
from pathlib import Path

from loguru import logger

from apps.trainer.data_pipeline import load_data, load_features_from_parquet
from apps.trainer.features import engineer_features
from libs.data.quality import validate_data_quality, validate_feature_quality


def main():
    """Validate data quality."""
    parser = argparse.ArgumentParser(description="Validate data quality")
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet file (raw data or features)",
    )
    parser.add_argument(
        "--type",
        choices=["raw", "features"],
        default="raw",
        help="Type of data: raw or features",
    )
    parser.add_argument(
        "--interval",
        default="1m",
        help="Expected interval (for completeness check)",
    )
    parser.add_argument(
        "--split-timestamp",
        default=None,
        help="Split timestamp for leakage check (ISO format, e.g., '2024-01-01T00:00:00Z')",
    )
    parser.add_argument(
        "--no-leakage",
        action="store_true",
        help="Skip leakage check",
    )
    parser.add_argument(
        "--no-completeness",
        action="store_true",
        help="Skip completeness check",
    )
    parser.add_argument(
        "--engineer-features",
        action="store_true",
        help="Engineer features and validate them too (for raw data)",
    )

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from {input_path}")
    if args.type == "features":
        df = load_features_from_parquet(filepath=input_path)
    else:
        df = load_data(input_path)

    if df.empty:
        logger.error("DataFrame is empty!")
        return

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Parse split timestamp if provided
    split_timestamp = None
    if args.split_timestamp:
        from datetime import datetime
        split_timestamp = datetime.fromisoformat(args.split_timestamp.replace("Z", "+00:00"))

    # Validate raw data
    logger.info("\n" + "=" * 60)
    logger.info("Validating Raw Data Quality")
    logger.info("=" * 60)

    report = validate_data_quality(
        df=df,
        timestamp_col="timestamp",
        interval=args.interval,
        split_timestamp=split_timestamp,
        check_leakage=not args.no_leakage,
        check_completeness=not args.no_completeness,
        check_missing=True,
        check_types=True,
        check_ranges=True,
    )

    logger.info(f"\n{report}")

    if not report.is_valid:
        logger.error("Data quality validation FAILED!")
        return 1

    # If raw data and engineer_features flag, also validate features
    if args.type == "raw" and args.engineer_features:
        logger.info("\n" + "=" * 60)
        logger.info("Engineering Features and Validating Quality")
        logger.info("=" * 60)

        df_features = engineer_features(df)
        logger.info(f"Engineered {len(df_features.columns)} columns")

        feature_report = validate_feature_quality(
            df=df_features,
            timestamp_col="timestamp",
            split_timestamp=split_timestamp,
        )

        logger.info(f"\n{feature_report}")

        if not feature_report.is_valid:
            logger.error("Feature quality validation FAILED!")
            return 1

    logger.info("\n✅ Data quality validation PASSED!")
    return 0


if __name__ == "__main__":
    exit(main())

