"""Multi-timeframe feature engineering for Phase 3.5 (V2)."""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from apps.trainer.data_pipeline import load_data


def load_multi_tf_data(
    symbol: str,
    intervals: List[str] = ["1m", "5m", "15m", "1h"],
    data_dir: Path | str = "data/raw",
    start_date: str = "2023-11-10",
) -> Dict[str, pd.DataFrame]:
    """
    Load data from multiple timeframes.

    Args:
        symbol: Trading pair (e.g., 'BTC-USD')
        intervals: List of timeframe intervals (default: ['1m', '5m', '15m', '1h'])
        data_dir: Directory containing raw data files
        start_date: Start date for data loading

    Returns:
        Dictionary mapping interval -> DataFrame
    """
    data_dir = Path(data_dir)
    multi_tf_data = {}

    for interval in intervals:
        # Find the data file (pattern: SYMBOL_INTERVAL_STARTDATE_*.parquet)
        pattern = f"{symbol}_{interval}_{start_date}_*.parquet"
        files = list(data_dir.glob(pattern))

        if not files:
            logger.warning(f"No data file found for {symbol} {interval} (pattern: {pattern})")
            continue

        if len(files) > 1:
            logger.warning(f"Multiple files found for {symbol} {interval}, using latest: {files[-1]}")

        file_path = files[-1]
        logger.info(f"Loading {interval} data from {file_path}")

        df = load_data(file_path)
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol} {interval}")
            continue

        multi_tf_data[interval] = df

    if not multi_tf_data:
        raise ValueError(f"No data loaded for {symbol}")

    logger.info(f"Loaded {len(multi_tf_data)} timeframes for {symbol}: {list(multi_tf_data.keys())}")
    return multi_tf_data


def resample_to_base_tf(
    base_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
    higher_tf_interval: str,
    feature_prefix: str = "",
) -> pd.DataFrame:
    """
    Resample higher timeframe data to align with base timeframe.

    Uses forward-fill to propagate higher TF values to all base TF candles within each higher TF period.

    Args:
        base_df: Base timeframe DataFrame (e.g., 1m)
        higher_tf_df: Higher timeframe DataFrame (e.g., 5m, 15m, 1h)
        higher_tf_interval: Interval string (e.g., '5m', '15m', '1h')
        feature_prefix: Prefix for resampled features (e.g., '5m_', '15m_')

    Returns:
        Base DataFrame with resampled higher TF features added
    """
    base_df = base_df.copy()

    # Ensure timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(base_df["timestamp"]):
        base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
    if not pd.api.types.is_datetime64_any_dtype(higher_tf_df["timestamp"]):
        higher_tf_df["timestamp"] = pd.to_datetime(higher_tf_df["timestamp"])

    # Set timestamp as index for merging
    base_df_indexed = base_df.set_index("timestamp")
    higher_tf_indexed = higher_tf_df.set_index("timestamp")

    # Select columns to resample (exclude timestamp, keep OHLCV and technical indicators)
    resample_cols = ["open", "high", "low", "close", "volume"]

    # Merge on timestamp with forward fill (asof merge)
    # This propagates higher TF values to all base TF candles within that period
    for col in resample_cols:
        if col in higher_tf_indexed.columns:
            # Use merge_asof for time-based alignment
            merged = pd.merge_asof(
                base_df_indexed.reset_index()[['timestamp']],
                higher_tf_indexed.reset_index()[['timestamp', col]],
                on='timestamp',
                direction='backward',  # Use the most recent higher TF value
            )
            base_df[f"{feature_prefix}{col}"] = merged[col].values

    logger.debug(f"Resampled {higher_tf_interval} features to base TF with prefix '{feature_prefix}'")
    return base_df


def calculate_cross_tf_alignment(
    df: pd.DataFrame,
    intervals: List[str] = ["1m", "5m", "15m", "1h"],
) -> pd.DataFrame:
    """
    Calculate cross-timeframe alignment score.

    Alignment measures whether price trends agree across multiple timeframes.
    Higher alignment = stronger trend across timeframes.

    Args:
        df: DataFrame with multi-TF features (e.g., '1m_close', '5m_close', etc.)
        intervals: List of intervals to check alignment for

    Returns:
        DataFrame with alignment features added
    """
    df = df.copy()

    # Calculate price changes for each timeframe
    price_changes = {}
    for interval in intervals:
        close_col = f"{interval}_close" if interval != "1m" else "close"
        if close_col in df.columns:
            # Calculate percentage change over lookback period
            price_changes[interval] = df[close_col].pct_change(periods=5)

    if len(price_changes) < 2:
        logger.warning("Not enough timeframes for alignment calculation")
        df["tf_alignment_score"] = 0.5  # Neutral
        return df

    # Calculate alignment score: proportion of timeframes moving in same direction
    alignment_scores = []
    for i in range(len(df)):
        changes = [price_changes[tf].iloc[i] for tf in price_changes if not pd.isna(price_changes[tf].iloc[i])]
        if len(changes) < 2:
            alignment_scores.append(0.5)  # Neutral when insufficient data
            continue

        # Count timeframes with positive/negative changes
        positive = sum(1 for c in changes if c > 0)
        negative = sum(1 for c in changes if c < 0)
        total = len(changes)

        # Alignment = max(positive, negative) / total
        # 1.0 = all agree, 0.5 = neutral, 0.0 = complete disagreement
        alignment = max(positive, negative) / total
        alignment_scores.append(alignment)

    df["tf_alignment_score"] = alignment_scores

    # Calculate alignment direction (-1, 0, 1)
    df["tf_alignment_direction"] = df["tf_alignment_score"].apply(
        lambda x: 1 if x > 0.66 else (-1 if x < 0.34 else 0)
    )

    # Calculate alignment strength (0-1, higher = stronger agreement)
    df["tf_alignment_strength"] = df["tf_alignment_score"].apply(
        lambda x: abs(x - 0.5) * 2  # Map [0.5, 1.0] -> [0.0, 1.0]
    )

    logger.info(
        f"Cross-TF alignment calculated: "
        f"mean={df['tf_alignment_score'].mean():.3f}, "
        f"high_alignment={sum(df['tf_alignment_strength'] > 0.5) / len(df) * 100:.1f}%"
    )

    return df


def calculate_volatility_regime_features(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Calculate volatility regime classification features.

    Uses ATR percentile to classify current volatility as low/medium/high.

    Args:
        df: DataFrame with OHLC data
        lookback: Lookback period for ATR calculation (default: 20)

    Returns:
        DataFrame with volatility regime features added
    """
    df = df.copy()

    # Calculate ATR if not already present
    if "atr" not in df.columns:
        # Simple ATR calculation
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=lookback).mean()

    # Calculate rolling ATR percentile
    df["atr_percentile"] = df["atr"].rolling(window=lookback * 5).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x) if len(x) > 0 else 0.5
    )

    # Classify volatility regime
    df["volatility_regime"] = pd.cut(
        df["atr_percentile"],
        bins=[0, 0.33, 0.67, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    # One-hot encode volatility regime
    df["volatility_low"] = (df["volatility_regime"] == "low").astype(int)
    df["volatility_medium"] = (df["volatility_regime"] == "medium").astype(int)
    df["volatility_high"] = (df["volatility_regime"] == "high").astype(int)

    logger.info(
        f"Volatility regime distribution: "
        f"low={df['volatility_low'].sum()}, "
        f"medium={df['volatility_medium'].sum()}, "
        f"high={df['volatility_high'].sum()}"
    )

    return df


def engineer_multi_tf_features(
    symbol: str,
    intervals: List[str] = ["1m", "5m", "15m", "1h"],
    data_dir: Path | str = "data/raw",
    start_date: str = "2023-11-10",
) -> pd.DataFrame:
    """
    Engineer features using multiple timeframes (Phase 3.5 / V2).

    This function:
    1. Loads data from all specified timeframes
    2. Resamples higher TF data to align with base TF (1m)
    3. Calculates cross-TF alignment scores
    4. Calculates volatility regime features

    Args:
        symbol: Trading pair (e.g., 'BTC-USD')
        intervals: List of timeframes to use (default: ['1m', '5m', '15m', '1h'])
        data_dir: Directory with raw data files
        start_date: Start date for data

    Returns:
        DataFrame with multi-TF features engineered
    """
    logger.info(f"Engineering multi-TF features for {symbol}")
    logger.info(f"  Timeframes: {intervals}")

    # Load data from all timeframes
    multi_tf_data = load_multi_tf_data(symbol, intervals, data_dir, start_date)

    # Use 1m as base timeframe
    if "1m" not in multi_tf_data:
        raise ValueError("Base timeframe '1m' not found in loaded data")

    df = multi_tf_data["1m"].copy()
    logger.info(f"Base timeframe (1m): {len(df)} rows")

    # Resample higher timeframes to base TF
    for interval in intervals:
        if interval == "1m":
            continue  # Skip base timeframe

        if interval in multi_tf_data:
            logger.info(f"Resampling {interval} to base TF...")
            df = resample_to_base_tf(
                df,
                multi_tf_data[interval],
                interval,
                feature_prefix=f"{interval}_",
            )

    # Calculate cross-TF alignment
    logger.info("Calculating cross-TF alignment...")
    df = calculate_cross_tf_alignment(df, intervals)

    # Calculate volatility regime features
    logger.info("Calculating volatility regime features...")
    df = calculate_volatility_regime_features(df)

    # Log feature summary
    multi_tf_features = [col for col in df.columns if any(tf in col for tf in ["5m_", "15m_", "1h_", "tf_", "volatility_"])]
    logger.info(f"✅ Multi-TF features engineered: {len(multi_tf_features)} new features")
    logger.info(f"  Sample features: {multi_tf_features[:10]}")

    return df
