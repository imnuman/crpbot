"""Feature engineering for cryptocurrency trading data."""
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import ta
except ImportError:
    logger.warning("ta library not installed. Install with: pip install ta")
    ta = None


def get_trading_session(timestamp: datetime) -> str:
    """
    Determine trading session based on timestamp (UTC).

    Sessions:
    - Tokyo (Asia): 00:00-08:00 UTC
    - London (EU): 08:00-16:00 UTC
    - New York (US): 16:00-00:00 UTC

    Args:
        timestamp: Datetime object (UTC)

    Returns:
        Session name: 'tokyo', 'london', 'new_york'
    """
    hour = timestamp.hour
    if 0 <= hour < 8:
        return "tokyo"
    elif 8 <= hour < 16:
        return "london"
    else:  # 16 <= hour < 24
        return "new_york"


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session-related features to DataFrame.

    Features added:
    - session: Trading session (tokyo, london, new_york)
    - session_tokyo: Binary indicator for Tokyo session
    - session_london: Binary indicator for London session
    - session_new_york: Binary indicator for New York session
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - is_weekend: Binary indicator for weekend (Saturday/Sunday)

    Args:
        df: DataFrame with 'timestamp' column (UTC)

    Returns:
        DataFrame with session features added
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Add session
    df["session"] = df["timestamp"].apply(get_trading_session)

    # One-hot encode sessions
    df["session_tokyo"] = (df["session"] == "tokyo").astype(int)
    df["session_london"] = (df["session"] == "london").astype(int)
    df["session_new_york"] = (df["session"] == "new_york").astype(int)

    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Weekend indicator
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    logger.debug(f"Added session features: {df[['session', 'day_of_week', 'is_weekend']].head()}")

    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame with OHLC columns
        period: ATR period (default: 14)

    Returns:
        Series with ATR values
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR as moving average of TR
    atr = tr.rolling(window=period, min_periods=1).mean()

    return atr


def calculate_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate spread-related features.

    Features added:
    - spread: high - low (absolute)
    - spread_pct: (high - low) / close * 100 (percentage)
    - spread_atr_ratio: spread / ATR (normalized by volatility)

    Args:
        df: DataFrame with OHLC columns

    Returns:
        DataFrame with spread features added
    """
    df = df.copy()

    # Absolute spread
    df["spread"] = df["high"] - df["low"]

    # Percentage spread
    df["spread_pct"] = (df["spread"] / df["close"]) * 100

    # Calculate ATR for normalization
    atr = calculate_atr(df)
    df["atr"] = atr

    # Spread normalized by ATR
    df["spread_atr_ratio"] = df["spread"] / (atr + 1e-8)  # Avoid division by zero

    return df


def calculate_volume_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculate volume-related features.

    Features added:
    - volume_ma: Moving average of volume
    - volume_ratio: volume / volume_ma (current vs average)
    - volume_trend: Trend in volume (positive/negative slope)

    Args:
        df: DataFrame with 'volume' column
        period: Period for moving average (default: 20)

    Returns:
        DataFrame with volume features added
    """
    df = df.copy()

    # Volume moving average
    df["volume_ma"] = df["volume"].rolling(window=period, min_periods=1).mean()

    # Volume ratio (current vs average)
    df["volume_ratio"] = df["volume"] / (df["volume_ma"] + 1e-8)

    # Volume trend (slope over last N periods)
    df["volume_trend"] = (
        df["volume"]
        .rolling(window=period, min_periods=1)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    )

    return df


def calculate_volatility_regime(
    df: pd.DataFrame, window: int = 20, low_percentile: float = 33.0, high_percentile: float = 67.0
) -> pd.Series:
    """
    Calculate volatility regime (high/medium/low).

    Uses rolling ATR to determine volatility regime:
    - Low: ATR < low_percentile
    - Medium: low_percentile <= ATR < high_percentile
    - High: ATR >= high_percentile

    Args:
        df: DataFrame with OHLC columns
        window: Window for rolling calculation (default: 20)
        low_percentile: Low percentile threshold (default: 33.0)
        high_percentile: High percentile threshold (default: 67.0)

    Returns:
        Series with volatility regime: 'low', 'medium', 'high'
    """
    atr = calculate_atr(df)

    # Calculate rolling percentiles
    low_thresh = atr.rolling(window=window, min_periods=1).quantile(low_percentile / 100)
    high_thresh = atr.rolling(window=window, min_periods=1).quantile(high_percentile / 100)

    # Determine regime based on ATR relative to percentiles
    regime = pd.Series("medium", index=df.index, dtype="object")
    regime[atr < low_thresh] = "low"
    regime[atr >= high_thresh] = "high"

    return regime


def add_technical_indicators(df: pd.DataFrame, use_ta: bool = True) -> pd.DataFrame:
    """
    Add technical indicators using ta library or custom implementations.

    Indicators added:
    - ATR (Average True Range)
    - RSI (Relative Strength Index) - if ta available
    - MACD (Moving Average Convergence Divergence) - if ta available
    - Bollinger Bands - if ta available
    - Moving averages (SMA 7, 14, 21, 50)

    Args:
        df: DataFrame with OHLCV columns
        use_ta: Whether to use ta library (default: True)

    Returns:
        DataFrame with technical indicators added
    """
    df = df.copy()

    # ATR (custom implementation)
    df["atr"] = calculate_atr(df)

    # Moving averages
    for period in [7, 14, 21, 50]:
        df[f"sma_{period}"] = df["close"].rolling(window=period, min_periods=1).mean()
        df[f"price_sma_{period}_ratio"] = df["close"] / (df[f"sma_{period}"] + 1e-8)

    # Use ta library if available
    if use_ta and ta is not None:
        try:
            # RSI
            df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()

            # MACD
            macd = ta.trend.MACD(close=df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close=df["close"])
            df["bb_high"] = bollinger.bollinger_hband()
            df["bb_low"] = bollinger.bollinger_lband()
            df["bb_width"] = (df["bb_high"] - df["bb_low"]) / (df["close"] + 1e-8)
            df["bb_position"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"] + 1e-8)
        except Exception as e:
            logger.warning(f"Error calculating ta indicators: {e}")
    else:
        logger.debug("ta library not available, using custom indicators only")

    return df


def engineer_features(
    df: pd.DataFrame,
    add_session_features_flag: bool = True,
    add_technical_indicators_flag: bool = True,
    add_spread_features_flag: bool = True,
    add_volume_features_flag: bool = True,
    add_volatility_regime_flag: bool = True,
) -> pd.DataFrame:
    """
    Engineer all features for trading data.

    Args:
        df: DataFrame with OHLCV data and timestamp
        add_session_features_flag: Whether to add session features
        add_technical_indicators_flag: Whether to add technical indicators
        add_spread_features_flag: Whether to add spread features
        add_volume_features_flag: Whether to add volume features
        add_volatility_regime_flag: Whether to add volatility regime

    Returns:
        DataFrame with all engineered features
    """
    logger.info("Engineering features...")

    if df.empty:
        logger.warning("DataFrame is empty, returning as-is")
        return df

    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by timestamp to ensure proper feature calculation
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add session features (critical for trading)
    if add_session_features_flag:
        df = add_session_features(df)

    # Add spread features
    if add_spread_features_flag:
        df = calculate_spread_features(df)

    # Add volume features
    if add_volume_features_flag:
        df = calculate_volume_features(df)

    # Add technical indicators
    if add_technical_indicators_flag:
        df = add_technical_indicators(df)

    # Add volatility regime
    if add_volatility_regime_flag:
        df["volatility_regime"] = calculate_volatility_regime(df)
        # One-hot encode volatility regime
        df["volatility_low"] = (df["volatility_regime"] == "low").astype(int)
        df["volatility_medium"] = (df["volatility_regime"] == "medium").astype(int)
        df["volatility_high"] = (df["volatility_regime"] == "high").astype(int)

    # Handle NaN values (forward fill, then backward fill)
    initial_nans = df.isna().sum().sum()
    df = df.ffill().bfill()
    final_nans = df.isna().sum().sum()

    if initial_nans > 0:
        logger.info(f"Filled {initial_nans - final_nans} NaN values in features")

    # Log feature count
    feature_cols = [
        col
        for col in df.columns
        if col not in ["timestamp", "open", "high", "low", "close", "volume"]
    ]
    logger.info(f"Engineered {len(feature_cols)} features")
    logger.debug(f"Feature columns: {feature_cols[:10]}...")  # Show first 10

    return df


def normalize_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    method: str = "standard",
    fit_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Normalize features using specified method.

    Methods:
    - 'standard': (x - mean) / std
    - 'minmax': (x - min) / (max - min)
    - 'robust': (x - median) / IQR

    Args:
        df: DataFrame with features to normalize
        feature_columns: List of columns to normalize (if None, auto-detect)
        method: Normalization method (default: 'standard')
        fit_data: Data to use for fitting normalization params (if None, use df)

    Returns:
        Tuple of (normalized DataFrame, normalization parameters dict)
    """
    if df.empty:
        return df, {}

    df = df.copy()

    # Auto-detect feature columns if not provided
    if feature_columns is None:
        exclude_cols = ["timestamp", "session", "volatility_regime"]
        feature_columns = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not feature_columns:
        logger.warning("No feature columns found for normalization")
        return df, {}

    # Use fit_data for calculating normalization params if provided
    fit_df = fit_data if fit_data is not None else df

    normalization_params = {}

    for col in feature_columns:
        if col not in df.columns:
            continue

        values = fit_df[col].dropna()
        if len(values) == 0:
            continue

        if method == "standard":
            mean = values.mean()
            std = values.std()
            if std > 1e-8:
                df[col] = (df[col] - mean) / std
                normalization_params[col] = {"method": "standard", "mean": mean, "std": std}
        elif method == "minmax":
            min_val = values.min()
            max_val = values.max()
            if max_val - min_val > 1e-8:
                df[col] = (df[col] - min_val) / (max_val - min_val)
                normalization_params[col] = {"method": "minmax", "min": min_val, "max": max_val}
        elif method == "robust":
            median = values.median()
            q75 = values.quantile(0.75)
            q25 = values.quantile(0.25)
            iqr = q75 - q25
            if iqr > 1e-8:
                df[col] = (df[col] - median) / iqr
                normalization_params[col] = {"method": "robust", "median": median, "iqr": iqr}

    logger.info(f"Normalized {len(normalization_params)} features using {method} method")

    return df, normalization_params
