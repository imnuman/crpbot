"""Comprehensive data quality checks and validation."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from apps.trainer.features import engineer_features


@dataclass
class DataQualityCheck:
    """Result of a single data quality check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"

    def __str__(self) -> str:
        """String representation."""
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} [{self.severity.upper()}]: {self.name} - {self.message}"


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""

    checks: list[DataQualityCheck]
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int

    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=" * 60,
            "Data Quality Report",
            "=" * 60,
            f"Total checks: {self.total_checks}",
            f"Passed: {self.passed_checks}",
            f"Failed: {self.failed_checks}",
            f"Warnings: {self.warnings}",
            "",
        ]

        # Group by severity
        errors = [c for c in self.checks if not c.passed and c.severity == "error"]
        warnings_list = [c for c in self.checks if (not c.passed and c.severity == "warning") or c.severity == "warning"]

        if errors:
            lines.append("❌ ERRORS:")
            for check in errors:
                lines.append(f"  {check}")
            lines.append("")

        if warnings_list:
            lines.append("⚠️  WARNINGS:")
            for check in warnings_list:
                lines.append(f"  {check}")
            lines.append("")

        if self.passed_checks > 0:
            lines.append("✅ PASSED CHECKS:")
            passed = [c for c in self.checks if c.passed and c.severity == "info"]
            for check in passed[:10]:  # Show first 10
                lines.append(f"  {check}")
            if len(passed) > 10:
                lines.append(f"  ... and {len(passed) - 10} more")

        lines.append("=" * 60)

        return "\n".join(lines)

    @property
    def is_valid(self) -> bool:
        """Check if data quality is acceptable (no errors)."""
        return self.failed_checks == 0 or all(
            c.severity != "error" for c in self.checks if not c.passed
        )


def check_data_leakage(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    timestamp_col: str = "timestamp",
    split_timestamp: datetime | None = None,
) -> DataQualityCheck:
    """
    Check for data leakage (features using future data).

    Args:
        df: DataFrame with features
        feature_columns: List of feature columns to check (if None, auto-detect)
        timestamp_col: Name of timestamp column
        split_timestamp: If provided, check that features before split don't use data after split

    Returns:
        DataQualityCheck result
    """
    if df.empty:
        return DataQualityCheck(
            name="Leakage Detection",
            passed=False,
            message="DataFrame is empty",
            severity="error",
        )

    # Check timestamp ordering
    if not df[timestamp_col].is_monotonic_increasing:
        return DataQualityCheck(
            name="Leakage Detection",
            passed=False,
            message="Timestamps are not sorted - potential leakage risk!",
            severity="error",
        )

    # Auto-detect feature columns if not provided
    if feature_columns is None:
        exclude_cols = [timestamp_col, "open", "high", "low", "close", "volume"]
        feature_columns = [col for col in df.columns if col not in exclude_cols]

    if not feature_columns:
        return DataQualityCheck(
            name="Leakage Detection",
            passed=False,
            message="No feature columns found to check",
            severity="warning",
        )

    # Check for temporal leakage (if split timestamp provided)
    if split_timestamp is not None:
        train_df = df[df[timestamp_col] < split_timestamp]
        test_df = df[df[timestamp_col] >= split_timestamp]

        if not train_df.empty and not test_df.empty:
            # Check if any train features correlate with future test data
            # This is a simplified check - full check would require feature engineering inspection
            for col in feature_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    train_mean = train_df[col].mean()
                    test_mean = test_df[col].mean()
                    
                    # If feature distributions are very similar, it might indicate leakage
                    # This is a heuristic - not definitive
                    if abs(train_mean - test_mean) < 1e-10 and train_df[col].std() > 0:
                        return DataQualityCheck(
                            name="Leakage Detection",
                            passed=False,
                            message=f"Feature '{col}' has suspiciously similar distribution in train/test - potential leakage",
                            severity="warning",
                        )

    # Basic check passed
    return DataQualityCheck(
        name="Leakage Detection",
        passed=True,
        message=f"Checked {len(feature_columns)} features - no obvious leakage detected",
        severity="info",
    )


def check_data_completeness(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    interval: str = "1m",
    max_gap_minutes: int = 60,
    min_completeness_pct: float = 95.0,
) -> DataQualityCheck:
    """
    Check data completeness (missing periods, gaps).

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        interval: Expected interval (e.g., '1m', '1h')
        max_gap_minutes: Maximum allowed gap in minutes
        min_completeness_pct: Minimum completeness percentage

    Returns:
        DataQualityCheck result
    """
    if df.empty:
        return DataQualityCheck(
            name="Data Completeness",
            passed=False,
            message="DataFrame is empty",
            severity="error",
        )

    # Calculate expected interval
    interval_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    expected_interval = interval_map.get(interval.lower(), timedelta(minutes=1))

    # Calculate date range
    start_date = df[timestamp_col].min()
    end_date = df[timestamp_col].max()
    date_range = end_date - start_date

    # Calculate expected number of rows
    expected_rows = int(date_range / expected_interval) + 1
    actual_rows = len(df)

    # Calculate completeness
    completeness_pct = (actual_rows / expected_rows * 100) if expected_rows > 0 else 0

    # Check for large gaps
    df_sorted = df.sort_values(timestamp_col).copy()
    df_sorted["time_diff"] = df_sorted[timestamp_col].diff()
    large_gaps = df_sorted[df_sorted["time_diff"] > timedelta(minutes=max_gap_minutes)]

    if completeness_pct < min_completeness_pct:
        return DataQualityCheck(
            name="Data Completeness",
            passed=False,
            message=f"Completeness {completeness_pct:.2f}% < {min_completeness_pct}% (expected {expected_rows} rows, got {actual_rows})",
            severity="error",
        )

    if len(large_gaps) > 0:
        return DataQualityCheck(
            name="Data Completeness",
            passed=False,
            message=f"Found {len(large_gaps)} gaps > {max_gap_minutes} minutes",
            severity="warning",
        )

    return DataQualityCheck(
        name="Data Completeness",
        passed=True,
        message=f"Completeness: {completeness_pct:.2f}% ({actual_rows}/{expected_rows} rows)",
        severity="info",
    )


def check_missing_values(
    df: pd.DataFrame, feature_columns: list[str] | None = None, max_missing_pct: float = 5.0
) -> list[DataQualityCheck]:
    """
    Check for missing values in features.

    Args:
        df: DataFrame with features
        feature_columns: List of feature columns to check (if None, check all)
        max_missing_pct: Maximum allowed missing percentage per column

    Returns:
        List of DataQualityCheck results
    """
    checks = []

    if df.empty:
        checks.append(
            DataQualityCheck(
                name="Missing Values",
                passed=False,
                message="DataFrame is empty",
                severity="error",
            )
        )
        return checks

    if feature_columns is None:
        feature_columns = df.columns.tolist()

    total_missing = 0
    problematic_cols = []

    for col in feature_columns:
        if col not in df.columns:
            continue

        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        total_missing += missing_count

        if missing_pct > max_missing_pct:
            problematic_cols.append((col, missing_pct, missing_count))
            checks.append(
                DataQualityCheck(
                    name=f"Missing Values: {col}",
                    passed=False,
                    message=f"{missing_pct:.2f}% missing ({missing_count}/{len(df)} rows)",
                    severity="error" if missing_pct > 20 else "warning",
                )
            )

    if not problematic_cols:
        checks.append(
            DataQualityCheck(
                name="Missing Values",
                passed=True,
                message=f"No missing values found (checked {len(feature_columns)} columns)",
                severity="info",
            )
        )
    else:
        if total_missing == 0:
            # All missing values were handled
            checks.insert(
                0,
                DataQualityCheck(
                    name="Missing Values",
                    passed=True,
                    message=f"Missing values detected but handled (total: {total_missing})",
                    severity="info",
                ),
            )

    return checks


def check_data_types(df: pd.DataFrame) -> list[DataQualityCheck]:
    """
    Check data types are correct.

    Args:
        df: DataFrame to check

    Returns:
        List of DataQualityCheck results
    """
    checks = []

    if df.empty:
        checks.append(
            DataQualityCheck(
                name="Data Types",
                passed=False,
                message="DataFrame is empty",
                severity="error",
            )
        )
        return checks

    # Check timestamp column
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            checks.append(
                DataQualityCheck(
                    name="Data Types: timestamp",
                    passed=False,
                    message="Timestamp column is not datetime type",
                    severity="error",
                )
            )
        else:
            checks.append(
                DataQualityCheck(
                    name="Data Types: timestamp",
                    passed=True,
                    message="Timestamp column is datetime type",
                    severity="info",
                )
            )

    # Check OHLCV columns
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                checks.append(
                    DataQualityCheck(
                        name=f"Data Types: {col}",
                        passed=False,
                        message=f"{col} column is not numeric type",
                        severity="error",
                    )
                )
            else:
                checks.append(
                    DataQualityCheck(
                        name=f"Data Types: {col}",
                        passed=True,
                        message=f"{col} column is numeric type",
                        severity="info",
                    )
                )

    return checks


def check_data_ranges(df: pd.DataFrame) -> list[DataQualityCheck]:
    """
    Check data ranges are reasonable (no negative prices, high > low, etc.).

    Args:
        df: DataFrame with OHLCV data

    Returns:
        List of DataQualityCheck results
    """
    checks = []

    if df.empty:
        return checks

    # Check OHLCV logic
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
        ).sum()

        if invalid_ohlc > 0:
            checks.append(
                DataQualityCheck(
                    name="Data Ranges: OHLC",
                    passed=False,
                    message=f"Found {invalid_ohlc} rows with invalid OHLC (high < low or negative prices)",
                    severity="error",
                )
            )
        else:
            checks.append(
                DataQualityCheck(
                    name="Data Ranges: OHLC",
                    passed=True,
                    message="All OHLC values are valid",
                    severity="info",
                )
            )

    # Check volume
    if "volume" in df.columns:
        negative_volume = (df["volume"] < 0).sum()
        if negative_volume > 0:
            checks.append(
                DataQualityCheck(
                    name="Data Ranges: Volume",
                    passed=False,
                    message=f"Found {negative_volume} rows with negative volume",
                    severity="error",
                )
            )
        else:
            checks.append(
                DataQualityCheck(
                    name="Data Ranges: Volume",
                    passed=True,
                    message="All volume values are non-negative",
                    severity="info",
                )
            )

    return checks


def validate_data_quality(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    timestamp_col: str = "timestamp",
    interval: str = "1m",
    split_timestamp: datetime | None = None,
    check_leakage: bool = True,
    check_completeness: bool = True,
    check_missing: bool = True,
    check_types: bool = True,
    check_ranges: bool = True,
) -> DataQualityReport:
    """
    Run comprehensive data quality checks.

    Args:
        df: DataFrame to validate
        feature_columns: List of feature columns to check
        timestamp_col: Name of timestamp column
        interval: Expected data interval
        split_timestamp: Timestamp for train/test split (for leakage check)
        check_leakage: Whether to check for data leakage
        check_completeness: Whether to check data completeness
        check_missing: Whether to check for missing values
        check_types: Whether to check data types
        check_ranges: Whether to check data ranges

    Returns:
        DataQualityReport with all checks
    """
    checks = []

    # Leakage check
    if check_leakage:
        leakage_check = check_data_leakage(df, feature_columns, timestamp_col, split_timestamp)
        checks.append(leakage_check)

    # Completeness check
    if check_completeness:
        completeness_check = check_data_completeness(df, timestamp_col, interval)
        checks.append(completeness_check)

    # Missing values check
    if check_missing:
        missing_checks = check_missing_values(df, feature_columns)
        checks.extend(missing_checks)

    # Data types check
    if check_types:
        type_checks = check_data_types(df)
        checks.extend(type_checks)

    # Data ranges check
    if check_ranges:
        range_checks = check_data_ranges(df)
        checks.extend(range_checks)

    # Calculate statistics
    total_checks = len(checks)
    passed_checks = sum(1 for c in checks if c.passed)
    failed_checks = total_checks - passed_checks
    warnings = sum(1 for c in checks if c.severity == "warning")

    return DataQualityReport(
        checks=checks,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        warnings=warnings,
    )


def validate_feature_quality(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    timestamp_col: str = "timestamp",
    split_timestamp: datetime | None = None,
) -> DataQualityReport:
    """
    Validate quality of engineered features.

    Args:
        df: DataFrame with engineered features
        feature_columns: List of feature columns to check
        timestamp_col: Name of timestamp column
        split_timestamp: Timestamp for train/test split (for leakage check)

    Returns:
        DataQualityReport with feature-specific checks
    """
    # Auto-detect feature columns
    if feature_columns is None:
        exclude_cols = [timestamp_col, "open", "high", "low", "close", "volume"]
        feature_columns = [col for col in df.columns if col not in exclude_cols]

    # Run validation
    return validate_data_quality(
        df=df,
        feature_columns=feature_columns,
        timestamp_col=timestamp_col,
        interval="1m",  # Features are typically on same interval as raw data
        split_timestamp=split_timestamp,
        check_leakage=True,
        check_completeness=False,  # Completeness already checked on raw data
        check_missing=True,
        check_types=True,
        check_ranges=False,  # Features may have different ranges
    )

