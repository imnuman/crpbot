"""Complete feature engineering for runtime with all training features."""
import pandas as pd
from loguru import logger

from apps.runtime.data_fetcher import MarketDataFetcher
from apps.runtime.multi_tf_fetcher import (
    fetch_multi_tf_data,
    align_multi_tf_to_base,
    add_tf_alignment_features
)
from apps.trainer.features import engineer_features


def engineer_runtime_features(
    df: pd.DataFrame,
    symbol: str,
    data_fetcher: MarketDataFetcher,
    include_multi_tf: bool = True,
    include_coingecko: bool = False
) -> pd.DataFrame:
    """
    Engineer all features for runtime prediction matching training features.

    Args:
        df: Base 1m OHLCV DataFrame
        symbol: Trading pair (e.g., 'BTC-USD')
        data_fetcher: Market data fetcher instance
        include_multi_tf: Include multi-timeframe features
        include_coingecko: Include CoinGecko fundamental features (requires API)

    Returns:
        DataFrame with all features engineered
    """
    # Step 1: Base technical indicators (existing function)
    logger.debug("Engineering base technical indicators...")
    df_features = engineer_features(df)

    if len(df_features) < 60:
        logger.warning(f"Insufficient data after base feature engineering: {len(df_features)} rows")
        return df_features

    # Step 2: Multi-timeframe features
    # V6 models require multi-TF features for all symbols
    # V5 FIXED ETH model was trained without them, but V6 models all use multi-TF
    skip_multi_tf_symbols = []  # Empty - all V6 models use multi-TF features

    if include_multi_tf and symbol not in skip_multi_tf_symbols:
        try:
            logger.debug("Fetching multi-timeframe data...")
            multi_tf_data = fetch_multi_tf_data(
                data_fetcher=data_fetcher,
                symbol=symbol,
                intervals=["5m", "15m", "1h"],
                num_candles=200
            )

            if multi_tf_data:
                logger.debug("Aligning multi-TF data to base timeframe...")
                df_features = align_multi_tf_to_base(df_features, multi_tf_data)

                logger.debug("Adding TF alignment features...")
                df_features = add_tf_alignment_features(df_features)

                # Add multi-TF technical indicators
                df_features = add_multi_tf_indicators(df_features)
            else:
                logger.warning("No multi-TF data fetched, adding placeholder features")
                df_features = add_multi_tf_placeholders(df_features)

        except Exception as e:
            logger.error(f"Failed to add multi-TF features: {e}")
            df_features = add_multi_tf_placeholders(df_features)

    # Step 3: CoinGecko features (placeholder for now)
    if include_coingecko:
        try:
            logger.debug("Adding CoinGecko features...")
            df_features = add_coingecko_features(df_features, symbol)
        except Exception as e:
            logger.error(f"Failed to add CoinGecko features: {e}")
            df_features = add_coingecko_placeholders(df_features)
    else:
        # Add placeholders to match training feature count
        df_features = add_coingecko_placeholders(df_features)

    # Step 4: Convert categorical columns to numeric (required for V5 FIXED models)
    df_features = convert_categoricals_to_numeric(df_features)

    # Step 5: Final cleanup - fill any remaining NaN values
    numeric_cols = df_features.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    df_features[numeric_cols] = df_features[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    logger.info(f"Total features engineered: {len(df_features.columns)}")
    logger.debug(f"Runtime columns: {sorted(df_features.columns.tolist())}")
    return df_features


def add_multi_tf_indicators(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Add technical indicators from higher timeframes.

    Args:
        df: DataFrame with features
        symbol: Trading symbol (used to determine which features to add)

    Returns:
        DataFrame with multi-TF indicators added
    """
    try:
        # 1h RSI
        if '1h_close' in df.columns:
            df['rsi_1h'] = calculate_rsi(df['1h_close'], period=14)
        else:
            df['rsi_1h'] = 50.0  # Neutral

        # 1h MACD
        if '1h_close' in df.columns:
            macd_df = calculate_macd(df['1h_close'])
            df['macd_1h'] = macd_df['macd']
            df['macd_signal_1h'] = macd_df['signal']
            df['macd_hist_1h'] = macd_df['hist']
        else:
            df['macd_1h'] = 0.0
            df['macd_signal_1h'] = 0.0
            df['macd_hist_1h'] = 0.0

        # 1h Bollinger Bands
        if '1h_close' in df.columns:
            bb_df = calculate_bollinger_bands(df['1h_close'])
            df['bb_upper_1h'] = bb_df['upper']
            df['bb_lower_1h'] = bb_df['lower']
            df['bb_position_1h'] = (df['1h_close'] - bb_df['lower']) / (bb_df['upper'] - bb_df['lower'] + 1e-10)
        else:
            df['bb_upper_1h'] = df['close'] * 1.02
            df['bb_lower_1h'] = df['close'] * 0.98
            df['bb_position_1h'] = 0.5

        # 1h Volume indicators
        if '1h_volume' in df.columns:
            df['volume_1h_ma'] = df['1h_volume'].rolling(window=20, min_periods=1).mean()
            df['volume_1h_ratio'] = df['1h_volume'] / (df['volume_1h_ma'] + 1e-10)
        else:
            df['volume_1h_ma'] = df['volume']
            df['volume_1h_ratio'] = 1.0

        # Price change percentages
        if '1h_close' in df.columns:
            df['price_1h_change_pct'] = df['1h_close'].pct_change()
        else:
            df['price_1h_change_pct'] = 0.0

        df['price_4h_change_pct'] = df['close'].pct_change(periods=240)  # 4h in 1m candles
        df['price_24h_change_pct'] = df['close'].pct_change(periods=1440)  # 24h in 1m candles

        # ATR percentile (only for BTC/SOL, not ETH)
        if symbol != "ETH-USD":
            if 'atr' in df.columns:
                df['atr_percentile'] = df['atr'].rolling(window=100, min_periods=1).apply(
                    lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50.0
                )
            else:
                df['atr_percentile'] = 50.0  # Neutral

    except Exception as e:
        logger.error(f"Error adding multi-TF indicators: {e}")

    return df


def add_multi_tf_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Add placeholder multi-TF features when real data unavailable."""
    # Multi-TF OHLCV
    for tf in ['5m', '15m', '1h']:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[f'{tf}_{col}'] = df[col]

    # TF alignment
    df['tf_alignment_score'] = 0.5
    df['tf_alignment_direction'] = 0
    df['tf_alignment_strength'] = 0.5

    # ATR percentile and volatility regime
    df['atr_percentile'] = 50.0
    df['volatility_regime'] = 1  # Medium
    df['volatility_low'] = 0
    df['volatility_medium'] = 1
    df['volatility_high'] = 0

    # Multi-TF indicators
    df['rsi_1h'] = 50.0
    df['macd_1h'] = 0.0
    df['macd_signal_1h'] = 0.0
    df['macd_hist_1h'] = 0.0
    df['bb_upper_1h'] = df['close'] * 1.02
    df['bb_lower_1h'] = df['close'] * 0.98
    df['bb_position_1h'] = 0.5
    df['volume_1h_ma'] = df['volume']
    df['volume_1h_ratio'] = 1.0
    df['price_1h_change_pct'] = 0.0
    df['price_4h_change_pct'] = 0.0
    df['price_24h_change_pct'] = 0.0

    return df


def add_coingecko_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Add placeholder CoinGecko features when API unavailable."""
    # Use neutral/default values that won't skew predictions
    df['ath_date'] = 0  # Days since ATH (placeholder)
    df['market_cap_change_pct'] = 0.0
    df['volume_change_pct'] = 0.0
    df['price_change_pct'] = 0.0
    df['ath_distance_pct'] = -50.0  # Assume 50% below ATH
    df['market_cap_7d_ma'] = 0.0
    df['market_cap_30d_ma'] = 0.0
    df['market_cap_change_7d_pct'] = 0.0
    df['market_cap_trend'] = 0.0
    df['volume_7d_ma'] = df['volume'].rolling(window=7*1440, min_periods=1).mean()
    df['volume_change_7d_pct'] = 0.0

    return df


def add_coingecko_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add CoinGecko fundamental features using Premium API.

    Fetches real-time market data (market cap, ATH, volume) from CoinGecko
    with 5-minute caching to avoid rate limiting.

    Falls back to placeholders if API key not available or fetch fails.
    """
    from apps.runtime.coingecko_fetcher import CoinGeckoFetcher
    import os

    api_key = os.getenv('COINGECKO_API_KEY')

    # If no API key, use placeholders
    if not api_key:
        logger.warning("⚠️  No CoinGecko API key - using placeholders")
        return add_coingecko_placeholders(df)

    # Fetch real data
    try:
        fetcher = CoinGeckoFetcher(api_key)
        features = fetcher.get_features(symbol)

        # Add features to DataFrame (broadcast scalar values to all rows)
        for feature_name, value in features.items():
            df[feature_name] = value

        logger.info(f"✅ Added CoinGecko features for {symbol} "
                   f"(ath_distance: {features['ath_distance_pct']:.1f}%, "
                   f"price_change_24h: {features['price_change_pct']:.2f}%)")

        # Still calculate volume_7d_ma from Coinbase data
        df['volume_7d_ma'] = df['volume'].rolling(window=7*1440, min_periods=1).mean()

        return df

    except Exception as e:
        logger.error(f"Failed to fetch CoinGecko data: {e}")
        logger.warning("Falling back to placeholders")
        return add_coingecko_placeholders(df)


# Helper functions for technical indicators
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """Calculate MACD."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame({'macd': macd, 'signal': macd_signal, 'hist': macd_hist})


def calculate_bollinger_bands(series: pd.Series, period=20, std_dev=2) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    sma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return pd.DataFrame({'upper': upper, 'lower': lower, 'middle': sma})


def convert_categoricals_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns to numeric codes to match V5 FIXED model training.

    The V5 FIXED models were trained with categorical columns converted to numeric:
    - session: tokyo=0, london=1, new_york=2
    - volatility_regime: low=0, medium=1, high=2
    - ath_date: converted to 0 (placeholder)

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with categorical columns converted to numeric
    """
    # Convert session to numeric
    if 'session' in df.columns:
        session_map = {'tokyo': 0, 'london': 1, 'new_york': 2}
        df['session'] = df['session'].map(session_map).fillna(0).astype(int)

    # Convert volatility_regime to numeric
    if 'volatility_regime' in df.columns:
        volatility_map = {'low': 0, 'medium': 1, 'high': 2}
        df['volatility_regime'] = df['volatility_regime'].map(volatility_map).fillna(1).astype(int)

    # Convert ath_date to numeric (use 0 as placeholder)
    if 'ath_date' in df.columns:
        df['ath_date'] = df['ath_date'].fillna(0).astype(int) if df['ath_date'].dtype != 'int64' else df['ath_date']

    return df
