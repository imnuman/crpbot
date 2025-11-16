"""Regenerate source features from existing raw data with multi-TF pipeline.

Uses existing raw OHLCV data and applies the runtime feature engineering pipeline.
"""
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from apps.trainer.features import engineer_features


def add_multi_tf_features_offline(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add multi-TF features without fetching external data.

    Resamples the base 1m data to create 5m, 15m, and 1h timeframes.
    """
    logger.info("Adding multi-TF features (offline resampling)...")

    # Ensure timestamp is datetime and sorted
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Set timestamp as index for resampling
    df_indexed = df.set_index('timestamp')

    # Resample to higher timeframes
    for tf, rule in [('5m', '5T'), ('15m', '15T'), ('1h', '1H')]:
        logger.debug(f"Resampling to {tf}...")

        # Resample OHLCV
        resampled = df_indexed.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Forward fill to align with base timeframe
        resampled_ffill = resampled.reindex(df_indexed.index, method='ffill')

        # Add to main dataframe
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[f'{tf}_{col}'] = resampled_ffill[col].values

    # Add TF alignment features
    logger.debug("Adding TF alignment features...")

    # Simple alignment score based on trend agreement
    df['tf_alignment_score'] = 0.5  # Neutral default
    df['tf_alignment_direction'] = 0  # Neutral
    df['tf_alignment_strength'] = 0.5  # Medium

    # Calculate actual alignment if we have enough data
    if len(df) >= 60:
        # Check if trends align across timeframes
        close_1m_trend = (df['close'] > df['close'].shift(15)).astype(int)
        close_5m_trend = (df['5m_close'] > df['5m_close'].shift(15)).astype(int)
        close_15m_trend = (df['15m_close'] > df['15m_close'].shift(15)).astype(int)
        close_1h_trend = (df['1h_close'] > df['1h_close'].shift(60)).astype(int)

        # Alignment score (0-1)
        alignment_sum = close_1m_trend + close_5m_trend + close_15m_trend + close_1h_trend
        df['tf_alignment_score'] = alignment_sum / 4.0

        # Direction (-1, 0, 1)
        df['tf_alignment_direction'] = (df['tf_alignment_score'] > 0.6).astype(int) - (df['tf_alignment_score'] < 0.4).astype(int)

        # Strength (0-1)
        df['tf_alignment_strength'] = abs(df['tf_alignment_score'] - 0.5) * 2

    # Add multi-TF technical indicators (only for BTC/SOL, not ETH)
    if symbol != "ETH-USD":
        logger.debug("Adding multi-TF indicators...")

        # 1h RSI
        from apps.runtime.runtime_features import calculate_rsi, calculate_macd, calculate_bollinger_bands

        df['rsi_1h'] = calculate_rsi(df['1h_close'], period=14)

        # 1h MACD
        macd_df = calculate_macd(df['1h_close'])
        df['macd_1h'] = macd_df['macd']
        df['macd_signal_1h'] = macd_df['signal']
        df['macd_hist_1h'] = macd_df['hist']

        # 1h Bollinger Bands
        bb_df = calculate_bollinger_bands(df['1h_close'])
        df['bb_upper_1h'] = bb_df['upper']
        df['bb_lower_1h'] = bb_df['lower']
        df['bb_position_1h'] = (df['1h_close'] - bb_df['lower']) / (bb_df['upper'] - bb_df['lower'] + 1e-10)

        # 1h Volume indicators
        df['volume_1h_ma'] = df['1h_volume'].rolling(window=20, min_periods=1).mean()
        df['volume_1h_ratio'] = df['1h_volume'] / (df['volume_1h_ma'] + 1e-10)

        # Price change percentages
        df['price_1h_change_pct'] = df['1h_close'].pct_change()
        df['price_4h_change_pct'] = df['close'].pct_change(periods=240)  # 4h in 1m candles
        df['price_24h_change_pct'] = df['close'].pct_change(periods=1440)  # 24h in 1m candles

        # ATR percentile
        if 'atr' in df.columns:
            df['atr_percentile'] = df['atr'].rolling(window=100, min_periods=1).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50.0
            )
        else:
            df['atr_percentile'] = 50.0
    else:
        # ETH: Add 1h indicators without multi-TF features
        logger.debug("Adding 1h indicators for ETH (no multi-TF)...")

        from apps.runtime.runtime_features import calculate_macd

        # Create synthetic 1h close from 1m data
        df_indexed = df.set_index('timestamp')
        close_1h = df_indexed['close'].resample('1H').last().reindex(df_indexed.index, method='ffill')

        # 1h MACD
        macd_df = calculate_macd(close_1h)
        df['macd_1h'] = macd_df['macd'].values
        df['macd_signal_1h'] = macd_df['signal'].values
        df['macd_hist_1h'] = macd_df['hist'].values

        # 1h BB
        from apps.runtime.runtime_features import calculate_bollinger_bands
        bb_df = calculate_bollinger_bands(close_1h)
        df['bb_upper_1h'] = bb_df['upper'].values
        df['bb_lower_1h'] = bb_df['lower'].values
        df['bb_position_1h'] = ((close_1h - bb_df['lower']) / (bb_df['upper'] - bb_df['lower'] + 1e-10)).values

        # Volume
        volume_1h = df_indexed['volume'].resample('1H').sum().reindex(df_indexed.index, method='ffill')
        volume_1h_ma = volume_1h.rolling(window=20, min_periods=1).mean()
        df['volume_1h_ma'] = volume_1h_ma.values
        df['volume_1h_ratio'] = (volume_1h / (volume_1h_ma + 1e-10)).values

        # Price changes
        df['price_1h_change_pct'] = close_1h.pct_change().values
        df['price_4h_change_pct'] = df['close'].pct_change(periods=240)
        df['price_24h_change_pct'] = df['close'].pct_change(periods=1440)

    # Add CoinGecko placeholders
    logger.debug("Adding CoinGecko placeholders...")
    df['ath_date'] = 0
    df['ath_distance_pct'] = -50.0
    df['market_cap_change_pct'] = 0.0
    df['market_cap_7d_ma'] = 0.0
    df['market_cap_30d_ma'] = 0.0
    df['market_cap_change_7d_pct'] = 0.0
    df['market_cap_trend'] = 0.0
    df['volume_7d_ma'] = df['volume'].rolling(window=7*1440, min_periods=1).mean()
    df['volume_change_7d_pct'] = 0.0
    df['volume_change_pct'] = 0.0

    logger.info(f"✅ Multi-TF features added: {len(df.columns)} total columns")

    return df


def regenerate_features_for_symbol(
    symbol: str,
    raw_data_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/features")
):
    """Regenerate features from raw data."""
    logger.info(f"{'='*60}")
    logger.info(f"Regenerating features for {symbol}")
    logger.info(f"{'='*60}")

    # Find latest raw data file
    raw_files = list(raw_data_dir.glob(f"{symbol}_1m_*.parquet"))
    if not raw_files:
        logger.error(f"No raw data found for {symbol}")
        return False

    latest_raw = max(raw_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading raw data: {latest_raw.name}")

    df_raw = pd.read_parquet(latest_raw)
    logger.info(f"Loaded {len(df_raw):,} rows")

    if len(df_raw) < 200:
        logger.error(f"Insufficient data: {len(df_raw)} rows")
        return False

    # Step 1: Engineer base features
    logger.info("Engineering base technical indicators...")
    df_features = engineer_features(df_raw)
    logger.info(f"Base features: {len(df_features.columns)} columns")

    # Step 2: Add multi-TF features
    df_features = add_multi_tf_features_offline(df_features, symbol)

    # Step 3: Convert categoricals to numeric
    logger.info("Converting categoricals to numeric...")
    session_map = {'tokyo': 0, 'london': 1, 'new_york': 2}
    df_features['session'] = df_features['session'].map(session_map).fillna(0).astype(int)

    volatility_map = {'low': 0, 'medium': 1, 'high': 2}
    df_features['volatility_regime'] = df_features['volatility_regime'].map(volatility_map).fillna(1).astype(int)

    # Step 4: Fill NaNs
    logger.info("Filling NaN values...")
    numeric_cols = df_features.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    df_features[numeric_cols] = df_features[numeric_cols].ffill().bfill().fillna(0)

    # Verify feature count
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'session', 'volatility_regime']
    feature_cols = [c for c in df_features.columns if c not in exclude]
    numeric_features = [c for c in feature_cols if df_features[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    logger.info(f"Feature summary:")
    logger.info(f"  Total columns: {len(df_features.columns)}")
    logger.info(f"  After exclusions: {len(feature_cols)}")
    logger.info(f"  Numeric features: {len(numeric_features)}")

    expected = {'BTC-USD': 73, 'ETH-USD': 54, 'SOL-USD': 73}
    if symbol in expected:
        if len(numeric_features) == expected[symbol]:
            logger.success(f"✅ Feature count matches expectation: {expected[symbol]}")
        else:
            logger.warning(f"⚠️  Feature count mismatch: expected {expected[symbol]}, got {len(numeric_features)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"features_{symbol}_1m_{date_str}.parquet"

    logger.info(f"Saving to {output_file.name}...")
    df_features.to_parquet(output_file, index=False)

    # Update symlink
    latest_symlink = output_dir / f"features_{symbol}_1m_latest.parquet"
    if latest_symlink.exists() or latest_symlink.is_symlink():
        latest_symlink.unlink()
    latest_symlink.symlink_to(output_file.name)

    logger.success(f"✅ Saved {len(df_features):,} rows")
    logger.success(f"✅ Updated symlink: {latest_symlink.name}")

    return True


def main():
    """Main entry point."""
    logger.info("Regenerating features from raw data with multi-TF pipeline...")

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    for symbol in symbols:
        success = regenerate_features_for_symbol(symbol)
        if not success:
            logger.error(f"Failed for {symbol}")
            sys.exit(1)
        logger.info("")

    logger.success("✅ All features regenerated!")
    logger.info("\nNext: Retrain models with new features")


if __name__ == "__main__":
    main()
