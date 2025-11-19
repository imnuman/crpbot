"""Amazon Q Feature Engineering - 72 Features.

This module replicates the exact 72 features used by Amazon Q for V6 Enhanced training.
Achieves 70% accuracy with FNN model.
"""
import pandas as pd
import numpy as np
from loguru import logger


def engineer_amazon_q_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the exact 72 features used by Amazon Q for V6 Enhanced training.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)

    Returns:
        DataFrame with 72 engineered features
    """
    logger.info("Engineering Amazon Q's 72 features...")

    df = df.copy()

    # === 1. Simple ratios (3 features) ===
    df['close_open_ratio'] = df['close'] / (df['open'] + 1e-10)
    df['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)

    # === 2. Returns and lagged returns (7 features) ===
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_lag_1'] = df['returns'].shift(1)
    df['returns_lag_2'] = df['returns'].shift(2)
    df['returns_lag_3'] = df['returns'].shift(3)
    df['returns_lag_5'] = df['returns'].shift(5)

    # === 3. Volume features (6 features) ===
    df['volume_lag_1'] = df['volume'].shift(1)
    df['volume_lag_2'] = df['volume'].shift(2)
    df['volume_lag_3'] = df['volume'].shift(3)
    df['volume_lag_5'] = df['volume'].shift(5)
    df['volume_price_trend'] = df['volume'] * df['returns']

    # === 4. Simple Moving Averages (5 features) ===
    for period in [5, 10, 20, 50, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()

    # === 5. Exponential Moving Averages (5 features) ===
    for period in [5, 10, 20, 50, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # === 6. Price to SMA ratios (5 features) ===
    for period in [5, 10, 20, 50, 200]:
        df[f'price_to_sma_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-10)

    # === 7. Price to EMA ratios (5 features) ===
    for period in [5, 10, 20, 50, 200]:
        df[f'price_to_ema_{period}'] = df['close'] / (df[f'ema_{period}'] + 1e-10)

    # === 8. RSI (3 features) ===
    for period in [14, 21, 30]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # === 9. MACD - two variants (6 features) ===
    # MACD 12/26
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_12_26'] = ema12 - ema26
    df['macd_signal_12_26'] = df['macd_12_26'].ewm(span=9, adjust=False).mean()
    df['macd_histogram_12_26'] = df['macd_12_26'] - df['macd_signal_12_26']

    # MACD 5/35
    ema5 = df['close'].ewm(span=5, adjust=False).mean()
    ema35 = df['close'].ewm(span=35, adjust=False).mean()
    df['macd_5_35'] = ema5 - ema35
    df['macd_signal_5_35'] = df['macd_5_35'].ewm(span=9, adjust=False).mean()
    df['macd_histogram_5_35'] = df['macd_5_35'] - df['macd_signal_5_35']

    # === 10. Bollinger Bands - two periods (6 features) ===
    for period in [20, 50]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + (std * 2)
        df[f'bb_lower_{period}'] = sma - (std * 2)
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (
            df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10
        )

    # === 11. ATR (1 feature) ===
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()

    # === 12. Stochastic Oscillator (4 features) ===
    for period in [14, 21]:
        lowest_low = df['low'].rolling(period).min()
        highest_high = df['high'].rolling(period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()

    # === 13. Williams %R (2 features) ===
    for period in [14, 21]:
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        df[f'williams_r_{period}'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)

    # === 14. Momentum (4 features) ===
    for period in [5, 10, 20, 50]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

    # === 15. Rate of Change (4 features) ===
    for period in [5, 10, 20, 50]:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                               (df['close'].shift(period) + 1e-10)) * 100

    # === 16. Price Channels (6 features) ===
    for period in [20, 50]:
        df[f'price_channel_high_{period}'] = df['high'].rolling(period).max()
        df[f'price_channel_low_{period}'] = df['low'].rolling(period).min()
        df[f'price_channel_position_{period}'] = (
            (df['close'] - df[f'price_channel_low_{period}']) /
            (df[f'price_channel_high_{period}'] - df[f'price_channel_low_{period}'] + 1e-10)
        )

    # === 17. Volatility (2 features) ===
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_50'] = df['returns'].rolling(50).std()

    # Fill NaN values
    df = df.ffill().bfill().fillna(0)

    # Select only the 72 features (in exact order used in training)
    feature_columns = [
        'atr_14',
        'bb_lower_20', 'bb_lower_50',
        'bb_position_20', 'bb_position_50',
        'bb_upper_20', 'bb_upper_50',
        'close_open_ratio',
        'ema_10', 'ema_20', 'ema_200', 'ema_5', 'ema_50',
        'high_low_ratio',
        'log_returns',
        'macd_12_26', 'macd_5_35',
        'macd_histogram_12_26', 'macd_histogram_5_35',
        'macd_signal_12_26', 'macd_signal_5_35',
        'momentum_10', 'momentum_20', 'momentum_5', 'momentum_50',
        'price_channel_high_20', 'price_channel_high_50',
        'price_channel_low_20', 'price_channel_low_50',
        'price_channel_position_20', 'price_channel_position_50',
        'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_200', 'price_to_ema_5', 'price_to_ema_50',
        'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_200', 'price_to_sma_5', 'price_to_sma_50',
        'returns',
        'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
        'roc_10', 'roc_20', 'roc_5', 'roc_50',
        'rsi_14', 'rsi_21', 'rsi_30',
        'sma_10', 'sma_20', 'sma_200', 'sma_5', 'sma_50',
        'stoch_d_14', 'stoch_d_21',
        'stoch_k_14', 'stoch_k_21',
        'volatility_20', 'volatility_50',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
        'volume_price_trend',
        'volume_ratio',
        'williams_r_14', 'williams_r_21'
    ]

    # Keep original OHLCV + timestamp + 72 features
    result_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + feature_columns
    result_columns = [c for c in result_columns if c in df.columns]

    logger.info(f"✅ Engineered {len(feature_columns)} Amazon Q features")

    return df[result_columns]
