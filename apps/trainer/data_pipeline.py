"""Data pipeline for fetching, cleaning, and preparing cryptocurrency data."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from libs.config.config import Settings
from libs.data.provider import create_data_provider

# Interval mapping for pandas frequency
interval_map = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",  # Changed from "1H" to avoid deprecation warning
    "4h": "4h",  # Changed from "4H"
    "1d": "1D",
}


def fetch_historical_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime | None = None,
    interval: str = "1m",
    config: Settings | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from configured data provider.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC-USD', 'ETH-USD')
        start_date: Start date for data collection
        end_date: End date (default: current date)
        interval: Time interval (default: '1m')
        config: Settings object (if None, loads from env)

    Returns:
        DataFrame with OHLCV data
    """
    if config is None:
        config = Settings()

    if end_date is None:
        end_date = datetime.now(timezone=start_date.tzinfo)

    logger.info(
        f"Fetching {symbol} data from {start_date} to {end_date} " f"(interval: {interval})"
    )

    # Create data provider
    # For Coinbase, use JWT authentication with API key name and private key
    if config.data_provider == "coinbase":
        provider = create_data_provider(
            provider=config.data_provider,
            api_key_name=config.coinbase_api_key_name or config.effective_api_key,
            private_key=config.coinbase_api_private_key or config.effective_api_secret,
        )
    else:
        provider = create_data_provider(
            provider=config.data_provider,
            api_key=config.effective_api_key,
            api_secret=config.effective_api_secret,
            api_passphrase=config.effective_api_passphrase,
        )

    # Test connection
    if not provider.test_connection():
        raise ConnectionError(f"Failed to connect to {config.data_provider} API")

    # Fetch data in chunks (Coinbase limit is 300 candles per request)
    all_data = []
    current_start = start_date
    max_candles_per_request = 300

    # Map interval to timedelta for chunking
    interval_delta_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    interval_delta = interval_delta_map.get(interval.lower(), timedelta(minutes=1))

    while current_start < end_date:
        # Calculate chunk end (max 300 candles)
        chunk_end = current_start + (interval_delta * max_candles_per_request)
        if chunk_end > end_date:
            chunk_end = end_date

        logger.debug(f"Fetching chunk: {current_start} to {chunk_end}")

        chunk_data = provider.fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=chunk_end,
            limit=max_candles_per_request,
        )

        if not chunk_data.empty:
            all_data.append(chunk_data)
            # Move to next chunk (start from last timestamp + interval)
            current_start = chunk_data["timestamp"].max() + interval_delta
        else:
            # No more data available
            break

    if not all_data:
        logger.warning(f"No data fetched for {symbol}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Combine all chunks
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Fetched {len(df)} candles for {symbol}")
    return df


@dataclass
class DataQualityReport:
    """Data quality report with statistics."""

    initial_rows: int
    final_rows: int
    duplicates_removed: int
    invalid_price_removed: int
    invalid_ohlc_removed: int
    outliers_removed: int
    missing_values_filled: int
    gaps_filled: int
    completeness_pct: float
    date_range: tuple[datetime, datetime]
    missing_periods: list[tuple[datetime, datetime]]

    def __str__(self) -> str:
        """Generate string summary."""
        return f"""
Data Quality Report:
  Initial rows: {self.initial_rows}
  Final rows: {self.final_rows}
  Removed: {self.duplicates_removed} duplicates, {self.invalid_price_removed} invalid prices,
           {self.invalid_ohlc_removed} invalid OHLC, {self.outliers_removed} outliers
  Filled: {self.missing_values_filled} missing values, {self.gaps_filled} gaps
  Completeness: {self.completeness_pct:.2f}%
  Date range: {self.date_range[0]} to {self.date_range[1]}
  Missing periods: {len(self.missing_periods)} gaps
"""


def _get_interval_timedelta(interval: str) -> timedelta:
    """Convert interval string to timedelta."""
    interval_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    return interval_map.get(interval.lower(), timedelta(minutes=1))


def detect_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.

    Args:
        df: DataFrame with OHLCV data
        z_threshold: Z-score threshold (default: 3.0)

    Returns:
        Boolean Series indicating outliers
    """
    outlier_mask = pd.Series(False, index=df.index)

    # Check price changes (returns)
    if len(df) > 1:
        returns = df["close"].pct_change()
        # Avoid division by zero
        if returns.std() > 0:
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            outlier_mask |= z_scores > z_threshold

        # Check volume spikes
        volume_mean = df["volume"].mean()
        volume_std = df["volume"].std()
        if volume_std > 0:
            volume_z = np.abs((df["volume"] - volume_mean) / volume_std)
            outlier_mask |= volume_z > z_threshold

        # Check extreme price movements (high-low range as % of price)
        price_range_pct = ((df["high"] - df["low"]) / df["close"]) * 100
        range_mean = price_range_pct.mean()
        range_std = price_range_pct.std()
        if range_std > 0:
            range_z = np.abs((price_range_pct - range_mean) / range_std)
            outlier_mask |= range_z > z_threshold

    return outlier_mask


def validate_data_completeness(
    df: pd.DataFrame, interval: str = "1m", max_gap_minutes: int = 60
) -> tuple[float, list[tuple[datetime, datetime]]]:
    """
    Validate data completeness and detect missing periods.

    Args:
        df: DataFrame with timestamp column
        interval: Expected interval (e.g., '1m', '5m', '1h')
        max_gap_minutes: Maximum acceptable gap in minutes before flagging

    Returns:
        Tuple of (completeness_percentage, list of missing periods)
    """
    if df.empty:
        return 0.0, []

    df_sorted = df.sort_values("timestamp")
    start_time = df_sorted["timestamp"].min()
    end_time = df_sorted["timestamp"].max()
    expected_interval = _get_interval_timedelta(interval)

    # Generate expected timestamps
    freq_str = interval_map.get(interval.lower(), "1min")
    expected_timestamps = pd.date_range(start=start_time, end=end_time, freq=freq_str)
    expected_count = len(expected_timestamps)

    # Find missing timestamps
    actual_timestamps = set(df_sorted["timestamp"])
    missing_timestamps = set(expected_timestamps) - actual_timestamps

    # Calculate completeness
    actual_count = len(df_sorted)
    completeness = (actual_count / expected_count * 100) if expected_count > 0 else 0.0

    # Identify missing periods (gaps larger than max_gap_minutes)
    missing_periods = []
    if missing_timestamps:
        missing_sorted = sorted(missing_timestamps)
        gap_start = None
        for ts in missing_sorted:
            if gap_start is None:
                gap_start = ts
            elif (ts - gap_start) > timedelta(minutes=max_gap_minutes):
                # End of gap
                missing_periods.append((gap_start, ts - expected_interval))
                gap_start = ts
        if gap_start:
            missing_periods.append((gap_start, missing_sorted[-1]))

    return completeness, missing_periods


def clean_and_validate_data(
    df: pd.DataFrame,
    interval: str = "1m",
    remove_outliers: bool = True,
    outlier_z_threshold: float = 3.0,
    fill_gaps: bool = True,
    max_gap_minutes: int = 60,
    generate_report: bool = True,
) -> tuple[pd.DataFrame, DataQualityReport | None]:
    """
    Clean and validate OHLCV data with comprehensive checks.

    Args:
        df: Raw OHLCV DataFrame
        interval: Expected time interval (default: '1m')
        remove_outliers: Whether to remove detected outliers (default: True)
        outlier_z_threshold: Z-score threshold for outlier detection (default: 3.0)
        fill_gaps: Whether to fill small gaps (default: True)
        max_gap_minutes: Maximum gap size to fill (default: 60 minutes)
        generate_report: Whether to generate quality report (default: True)

    Returns:
        Tuple of (cleaned DataFrame, quality report or None)
    """
    logger.info("Cleaning and validating data...")
    initial_len = len(df)
    report = DataQualityReport(
        initial_rows=initial_len,
        final_rows=0,
        duplicates_removed=0,
        invalid_price_removed=0,
        invalid_ohlc_removed=0,
        outliers_removed=0,
        missing_values_filled=0,
        gaps_filled=0,
        completeness_pct=0.0,
        date_range=(datetime.min, datetime.max),
        missing_periods=[],
    )

    if df.empty:
        logger.warning("DataFrame is empty, returning as-is")
        report.final_rows = 0
        return df, report if generate_report else None

    # Validate required columns
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    report.duplicates_removed = initial_len - len(df)

    # Remove rows with invalid prices (NaN, zero, negative)
    before = len(df)
    df = df[
        (df["open"].notna())
        & (df["high"].notna())
        & (df["low"].notna())
        & (df["close"].notna())
        & (df["volume"].notna())
        & (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ].copy()
    report.invalid_price_removed = before - len(df)

    # Validate OHLC logic (high >= low, high >= open, high >= close, etc.)
    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    if invalid_ohlc.any():
        invalid_count = invalid_ohlc.sum()
        logger.warning(f"Found {invalid_count} rows with invalid OHLC logic, removing")
        df = df[~invalid_ohlc].copy()
        report.invalid_ohlc_removed = invalid_count

    # Detect and optionally remove outliers
    if remove_outliers and len(df) > 10:  # Need enough data for outlier detection
        outlier_mask = detect_outliers(df, z_threshold=outlier_z_threshold)
        if outlier_mask.any():
            outlier_count = outlier_mask.sum()
            logger.warning(f"Detected {outlier_count} outliers, removing")
            df = df[~outlier_mask].copy()
            report.outliers_removed = outlier_count

    # Fill missing values (forward fill for small gaps)
    before_fill = df.isna().sum().sum()
    df = df.ffill().bfill()  # Forward then backward fill (replaces deprecated fillna)
    after_fill = df.isna().sum().sum()
    report.missing_values_filled = max(0, before_fill - after_fill)

    # Fill timestamp gaps if requested
    if fill_gaps and len(df) > 1:
        df_indexed = df.set_index("timestamp").sort_index()
        freq_str = interval_map.get(interval.lower(), "1min")
        rows_before = len(df_indexed)
        df_indexed = df_indexed.asfreq(freq_str, method="ffill")
        rows_after = len(df_indexed)
        df = df_indexed.reset_index()
        report.gaps_filled = max(0, rows_after - rows_before)

    # Validate completeness
    completeness, missing_periods = validate_data_completeness(
        df, interval=interval, max_gap_minutes=max_gap_minutes
    )
    report.completeness_pct = completeness
    report.missing_periods = missing_periods
    report.date_range = (df["timestamp"].min(), df["timestamp"].max())
    report.final_rows = len(df)

    logger.info(f"Data cleaning complete: {len(df)} valid candles")
    logger.info(f"Completeness: {completeness:.2f}%")
    if missing_periods:
        logger.warning(f"Found {len(missing_periods)} missing periods")

    return df, report if generate_report else None


def create_walk_forward_splits(
    df: pd.DataFrame, train_end: datetime, val_end: datetime
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create walk-forward train/validation/test splits.

    Args:
        df: Full dataset
        train_end: End date for training set (start is first date in df)
        val_end: End date for validation set (test starts after this)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Creating walk-forward splits: train until {train_end}, val until {val_end}")

    train_df = df[df["timestamp"] <= train_end].copy()
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)].copy()
    test_df = df[df["timestamp"] > val_end].copy()

    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df


def save_data(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save DataFrame to parquet file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved data to {filepath}")


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded data from {filepath}: {len(df)} rows")
    return df


def save_features_to_parquet(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    version: str | None = None,
    base_dir: str | Path = "data/features",
) -> Path:
    """
    Save engineered features to versioned parquet file.

    Args:
        df: DataFrame with engineered features
        symbol: Trading pair symbol (e.g., 'BTC-USD')
        interval: Time interval (e.g., '1m', '1h')
        version: Version string (if None, uses date-based version)
        base_dir: Base directory for feature storage

    Returns:
        Path to saved file
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate version if not provided
    if version is None:
        from datetime import datetime

        version = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create filename: features_{symbol}_{interval}_{version}.parquet
    filename = f"features_{symbol}_{interval}_{version}.parquet"
    filepath = base_dir / filename

    df.to_parquet(filepath, index=False)
    logger.info(f"Saved features to {filepath} ({len(df)} rows, {len(df.columns)} columns)")

    # Update symlink to latest version
    latest_link = base_dir / f"features_{symbol}_{interval}_latest.parquet"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(filepath.name)

    return filepath


def load_features_from_parquet(
    filepath: str | Path | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    version: str = "latest",
    base_dir: str | Path = "data/features",
) -> pd.DataFrame:
    """
    Load engineered features from parquet file.

    Args:
        filepath: Direct path to file (if provided, other params ignored)
        symbol: Trading pair symbol (e.g., 'BTC-USD')
        interval: Time interval (e.g., '1m', '1h')
        version: Version string or 'latest' for symlink
        base_dir: Base directory for feature storage

    Returns:
        DataFrame with engineered features
    """
    base_dir = Path(base_dir)

    if filepath:
        # Direct path provided
        filepath = Path(filepath)
    else:
        # Build path from symbol, interval, version
        if symbol is None or interval is None:
            raise ValueError("Either filepath or both symbol and interval must be provided")

        if version == "latest":
            filename = f"features_{symbol}_{interval}_latest.parquet"
        else:
            filename = f"features_{symbol}_{interval}_{version}.parquet"

        filepath = base_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    df = pd.read_parquet(filepath)
    logger.info(f"Loaded features from {filepath}: {len(df)} rows, {len(df.columns)} columns")
    return df


def detect_leakage(
    df: pd.DataFrame, feature_columns: list[str], timestamp_col: str = "timestamp"
) -> bool:
    """
    Detect data leakage by checking if any features use future data.

    This is a basic check - it assumes features are derived from OHLCV data.
    More sophisticated checks would require inspecting feature engineering code.

    Args:
        df: DataFrame with features
        feature_columns: List of feature column names to check
        timestamp_col: Name of timestamp column

    Returns:
        True if leakage detected, False otherwise
    """
    # This is a placeholder - actual leakage detection requires:
    # 1. Inspecting feature engineering code
    # 2. Checking if features use future candles (T+1, T+2, etc.)
    # 3. Validating that all features are computed from T and earlier only

    logger.warning(
        "Leakage detection is a basic check. "
        "Ensure feature engineering code only uses past data (T and earlier)."
    )

    # Basic check: ensure timestamp is sorted
    if not df[timestamp_col].is_monotonic_increasing:
        logger.error("Timestamps are not sorted - potential leakage risk!")
        return True

    # TODO: Add more sophisticated checks:
    # - Check if features correlate too highly with future returns
    # - Validate feature engineering functions
    # - Check for look-ahead bias in technical indicators

    return False
