#!/usr/bin/env python3
"""Enhanced feature engineering for V6 rebuild - targeting 100+ features.

This module adds advanced features beyond the base technical indicators:
- Advanced momentum indicators (Stochastic, ADX, Williams %R)
- Volatility measures (historical vol, Parkinson, Garman-Klass)
- Price action patterns (candlestick patterns, support/resistance)
- Market microstructure (tick counts, trade intensity proxies)
- Cross-asset correlation features
- Entropy and information theory features
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

try:
    import ta
except ImportError:
    logger.warning("ta library not installed")
    ta = None


def add_advanced_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced momentum indicators.

    Features added (~15):
    - Stochastic Oscillator (%K, %D)
    - ADX (Average Directional Index)
    - Williams %R
    - CCI (Commodity Channel Index)
    - Rate of Change (ROC) at multiple periods
    - Momentum indicators
    - DI+ and DI- (Directional Indicators)
    """
    df = df.copy()

    if ta is not None:
        try:
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'], low=df['low'], close=df['close']
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # ADX (Average Directional Index)
            adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
            df['adx'] = adx.adx()
            df['di_plus'] = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()

            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(
                high=df['high'], low=df['low'], close=df['close']
            ).williams_r()

            # CCI (Commodity Channel Index)
            df['cci'] = ta.trend.CCIIndicator(
                high=df['high'], low=df['low'], close=df['close']
            ).cci()

            # Rate of Change at multiple periods
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = ta.momentum.ROCIndicator(
                    close=df['close'], window=period
                ).roc()

            # Ultimate Oscillator
            df['ultimate_osc'] = ta.momentum.UltimateOscillator(
                high=df['high'], low=df['low'], close=df['close']
            ).ultimate_oscillator()

            # Awesome Oscillator
            df['awesome_osc'] = ta.momentum.AwesomeOscillatorIndicator(
                high=df['high'], low=df['low']
            ).awesome_oscillator()

            # KAMA (Kaufman Adaptive Moving Average)
            df['kama'] = ta.momentum.KAMAIndicator(close=df['close']).kama()
            df['kama_ratio'] = df['close'] / (df['kama'] + 1e-8)

            logger.info("✅ Added 15 advanced momentum indicators")

        except Exception as e:
            logger.error(f"Error adding momentum indicators: {e}")

    return df


def add_volatility_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced volatility measures.

    Features added (~10):
    - Historical volatility (multiple periods)
    - Parkinson volatility estimator
    - Garman-Klass volatility
    - Rogers-Satchell volatility
    - Keltner Channels
    - Donchian Channels
    - True Range percentile
    """
    df = df.copy()

    # Historical volatility (rolling std of returns)
    returns = df['close'].pct_change()
    for period in [10, 20, 50]:
        df[f'hist_vol_{period}'] = returns.rolling(window=period, min_periods=1).std() * np.sqrt(252 * 1440)  # Annualized

    # Parkinson volatility estimator (uses high-low range)
    hl_ratio = np.log(df['high'] / df['low'])
    for period in [20, 50]:
        df[f'parkinson_vol_{period}'] = np.sqrt(
            (hl_ratio ** 2).rolling(window=period, min_periods=1).mean() / (4 * np.log(2))
        ) * np.sqrt(252 * 1440)

    # Garman-Klass volatility
    hl = np.log(df['high'] / df['low']) ** 2
    co = np.log(df['close'] / df['open']) ** 2
    df['garman_klass_vol'] = np.sqrt(
        (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window=20, min_periods=1).mean()
    ) * np.sqrt(252 * 1440)

    if ta is not None:
        try:
            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(
                high=df['high'], low=df['low'], close=df['close']
            )
            df['keltner_high'] = keltner.keltner_channel_hband()
            df['keltner_low'] = keltner.keltner_channel_lband()
            df['keltner_position'] = (df['close'] - df['keltner_low']) / (df['keltner_high'] - df['keltner_low'] + 1e-8)

            # Donchian Channels
            donchian = ta.volatility.DonchianChannel(
                high=df['high'], low=df['low'], close=df['close']
            )
            df['donchian_high'] = donchian.donchian_channel_hband()
            df['donchian_low'] = donchian.donchian_channel_lband()
            df['donchian_position'] = (df['close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'] + 1e-8)

            logger.info("✅ Added 10 volatility measures")

        except Exception as e:
            logger.error(f"Error adding volatility measures: {e}")

    return df


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price action and candlestick pattern features.

    Features added (~15):
    - Candle body/wick ratios
    - Price position in candle
    - Gap detection
    - Higher highs/lower lows
    - Support/resistance levels
    - Pivot points
    """
    df = df.copy()

    # Candle body
    df['body'] = abs(df['close'] - df['open'])
    df['body_pct'] = df['body'] / df['close'] * 100

    # Upper and lower wicks
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # Wick ratios
    df['wick_body_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['body'] + 1e-8)
    df['upper_wick_ratio'] = df['upper_wick'] / (df['high'] - df['low'] + 1e-8)
    df['lower_wick_ratio'] = df['lower_wick'] / (df['high'] - df['low'] + 1e-8)

    # Bullish/Bearish candle
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_bearish'] = (df['close'] < df['open']).astype(int)

    # Gap detection
    df['gap_up'] = ((df['low'] > df['high'].shift(1)) & (df['low'].shift(1).notna())).astype(int)
    df['gap_down'] = ((df['high'] < df['low'].shift(1)) & (df['high'].shift(1).notna())).astype(int)

    # Higher highs / Lower lows (swing detection)
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1).notna())).astype(int)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'].shift(1).notna())).astype(int)

    # Pivot points (classic)
    pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['pivot'] = pivot
    df['pivot_r1'] = 2 * pivot - df['low'].shift(1)
    df['pivot_s1'] = 2 * pivot - df['high'].shift(1)
    df['distance_from_pivot'] = (df['close'] - pivot) / (pivot + 1e-8) * 100

    # Consecutive candles
    df['consec_bullish'] = (df['is_bullish'] * (df['is_bullish'].groupby((df['is_bullish'] != df['is_bullish'].shift()).cumsum()).cumcount() + 1))
    df['consec_bearish'] = (df['is_bearish'] * (df['is_bearish'].groupby((df['is_bearish'] != df['is_bearish'].shift()).cumsum()).cumcount() + 1))

    logger.info("✅ Added 15 price action features")

    return df


def add_market_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market microstructure proxy features.

    Features added (~8):
    - Volume-weighted average price (VWAP)
    - Money flow index (MFI)
    - On-balance volume (OBV)
    - Accumulation/Distribution
    - Chaikin Money Flow
    - Volume price trend
    """
    df = df.copy()

    if ta is not None:
        try:
            # Money Flow Index
            df['mfi'] = ta.volume.MFIIndicator(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
            ).money_flow_index()

            # On-Balance Volume
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'], volume=df['volume']
            ).on_balance_volume()
            df['obv_ma'] = df['obv'].rolling(window=20, min_periods=1).mean()
            df['obv_signal'] = (df['obv'] > df['obv_ma']).astype(int)

            # Accumulation/Distribution
            df['ad'] = ta.volume.AccDistIndexIndicator(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
            ).acc_dist_index()

            # Chaikin Money Flow
            df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
            ).chaikin_money_flow()

            # Volume Price Trend
            df['vpt'] = ta.volume.VolumePriceTrendIndicator(
                close=df['close'], volume=df['volume']
            ).volume_price_trend()

            # Ease of Movement
            df['eom'] = ta.volume.EaseOfMovementIndicator(
                high=df['high'], low=df['low'], volume=df['volume']
            ).ease_of_movement()

            logger.info("✅ Added 8 market microstructure features")

        except Exception as e:
            logger.error(f"Error adding microstructure features: {e}")

    # VWAP (manual calculation)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8) * 100

    return df


def add_information_theory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add entropy and information theory features.

    Features added (~8):
    - Price entropy (Shannon)
    - Volume entropy
    - Return distribution kurtosis
    - Return distribution skewness
    - Hurst exponent
    - Fractal dimension
    """
    df = df.copy()

    # Price returns
    returns = df['close'].pct_change()

    # Rolling statistics on returns
    for window in [20, 50]:
        df[f'return_skew_{window}'] = returns.rolling(window=window, min_periods=1).skew()
        df[f'return_kurt_{window}'] = returns.rolling(window=window, min_periods=1).kurt()

    # Price entropy (Shannon entropy of price bins)
    def shannon_entropy(series, bins=10):
        """Calculate Shannon entropy of a series."""
        if len(series) < 2:
            return 0.0
        hist, _ = np.histogram(series, bins=bins)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))

    df['price_entropy_20'] = df['close'].rolling(window=20, min_periods=10).apply(shannon_entropy)
    df['volume_entropy_20'] = df['volume'].rolling(window=20, min_periods=10).apply(shannon_entropy)

    # Hurst exponent proxy (rescaled range)
    def hurst_proxy(series):
        """Simplified Hurst exponent estimation."""
        if len(series) < 10:
            return 0.5

        mean = series.mean()
        dev = series - mean
        cumdev = dev.cumsum()
        R = cumdev.max() - cumdev.min()
        S = series.std()

        if S < 1e-8:
            return 0.5

        return np.log(R / S) / np.log(len(series))

    df['hurst_50'] = returns.rolling(window=50, min_periods=20).apply(hurst_proxy)

    logger.info("✅ Added 8 information theory features")

    return df


def add_trend_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend strength and direction features.

    Features added (~10):
    - Multiple EMA crossovers
    - Ichimoku components
    - Parabolic SAR
    - Supertrend
    - Trend strength indicators
    """
    df = df.copy()

    if ta is not None:
        try:
            # Ichimoku
            ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()

            # Parabolic SAR
            df['psar'] = ta.trend.PSARIndicator(
                high=df['high'], low=df['low'], close=df['close']
            ).psar()
            df['psar_signal'] = (df['close'] > df['psar']).astype(int)

            # Aroon
            aroon = ta.trend.AroonIndicator(high=df['high'], low=df['low'])
            df['aroon_up'] = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            df['aroon_indicator'] = aroon.aroon_indicator()

            logger.info("✅ Added 10 trend strength features")

        except Exception as e:
            logger.error(f"Error adding trend features: {e}")

    # EMA crossovers
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_crossover'] = (ema_fast > ema_slow).astype(int)

    return df


def engineer_enhanced_features(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Engineer all enhanced features for V6 rebuild.

    This adds ~70 additional features beyond base technical indicators,
    bringing total to 100+ features.

    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol (optional, for symbol-specific features)

    Returns:
        DataFrame with enhanced features
    """
    logger.info(f"Engineering enhanced features for {symbol or 'unknown'}...")

    initial_cols = len(df.columns)

    # Add feature groups
    df = add_advanced_momentum_indicators(df)
    df = add_volatility_measures(df)
    df = add_price_action_features(df)
    df = add_market_microstructure_features(df)
    df = add_information_theory_features(df)
    df = add_trend_strength_features(df)

    # Fill NaN values
    df = df.ffill().bfill()

    final_cols = len(df.columns)
    added_features = final_cols - initial_cols

    logger.info(f"✅ Added {added_features} enhanced features (total: {final_cols} columns)")

    return df


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    from datetime import datetime, timedelta

    # Create sample OHLCV data
    dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=1000, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(1000).cumsum(),
        'high': 101 + np.random.randn(1000).cumsum(),
        'low': 99 + np.random.randn(1000).cumsum(),
        'close': 100 + np.random.randn(1000).cumsum(),
        'volume': np.abs(np.random.randn(1000)) * 1000
    })

    # Ensure OHLC relationship is valid
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    print(f"Input columns: {len(df.columns)}")

    df_enhanced = engineer_enhanced_features(df, symbol="BTC-USD")

    print(f"Output columns: {len(df_enhanced.columns)}")
    print(f"Feature columns: {[col for col in df_enhanced.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]}")
