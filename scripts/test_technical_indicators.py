"""
Test Technical Indicators Library on Historical Data

Tests the 50+ technical indicators on 2 years of collected data.
"""
import sys
sys.path.insert(0, '/root/crpbot')

import pandas as pd
import logging
from pathlib import Path
from libs.features.technical_indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_technical_indicators():
    """Test technical indicators on historical BTC data"""

    print("=" * 70)
    print("TECHNICAL INDICATORS TEST")
    print("=" * 70)

    # Load historical data
    data_path = Path('data/historical/BTC_USD_3600s_730d.parquet')

    if not data_path.exists():
        logger.error(f"Historical data not found: {data_path}")
        logger.error("Run scripts/collect_historical_data.py first")
        return None

    logger.info(f"Loading BTC historical data from {data_path}")
    df = pd.read_parquet(data_path)

    logger.info(f"Loaded {len(df)} candles")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Columns: {list(df.columns)}")

    # Initialize feature engineering
    ti = TechnicalIndicators()

    # Add all indicators
    print("\n" + "=" * 70)
    print("ADDING TECHNICAL INDICATORS")
    print("=" * 70 + "\n")

    df_with_features = ti.add_all_indicators(df.copy())

    # Summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df_with_features.columns if c not in original_cols]

    print(f"\nOriginal columns: {len(original_cols)}")
    print(f"Feature columns:  {len(feature_cols)}")
    print(f"Total columns:    {len(df_with_features.columns)}")

    print(f"\n📊 Feature Categories:")

    # Count by category
    momentum_features = [c for c in feature_cols if any(x in c for x in ['rsi', 'macd', 'stoch', 'williams', 'roc', 'cmo'])]
    volatility_features = [c for c in feature_cols if any(x in c for x in ['atr', 'bb_', 'kc_', 'dc_'])]
    trend_features = [c for c in feature_cols if any(x in c for x in ['adx', 'di', 'supertrend', 'trix'])]
    volume_features = [c for c in feature_cols if any(x in c for x in ['obv', 'vwap', 'mfi', 'ad_', 'cmf'])]
    statistical_features = [c for c in feature_cols if any(x in c for x in ['z_score', 'percentile', 'lr_slope'])]

    print(f"  Momentum:     {len(momentum_features)} features")
    print(f"  Volatility:   {len(volatility_features)} features")
    print(f"  Trend:        {len(trend_features)} features")
    print(f"  Volume:       {len(volume_features)} features")
    print(f"  Statistical:  {len(statistical_features)} features")

    # Show sample values
    print(f"\n📈 Sample Feature Values (last row):")
    print(f"  Close price:   ${df_with_features['close'].iloc[-1]:,.2f}")
    print(f"  RSI (14):      {df_with_features['rsi_14'].iloc[-1]:.2f}")
    print(f"  MACD:          {df_with_features['macd'].iloc[-1]:.2f}")
    print(f"  ATR (14):      {df_with_features['atr_14'].iloc[-1]:.2f}")
    print(f"  ADX:           {df_with_features['adx'].iloc[-1]:.2f}")
    print(f"  Z-Score:       {df_with_features['z_score_20'].iloc[-1]:.2f}")

    # Check for null values
    print(f"\n🔍 Data Quality:")
    null_counts = df_with_features[feature_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if len(cols_with_nulls) > 0:
        print(f"  ⚠️  Columns with nulls: {len(cols_with_nulls)}")
        for col, count in list(cols_with_nulls.items())[:5]:
            pct = (count / len(df_with_features)) * 100
            print(f"    {col}: {count} ({pct:.1f}%)")
        print(f"  Note: Initial nulls are normal due to indicator warmup periods")
    else:
        print(f"  ✅ No null values in features")

    # Save enriched data
    output_path = Path('data/historical/BTC_USD_3600s_730d_features.parquet')
    df_with_features.to_parquet(output_path, compression='snappy', index=False)
    file_size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"\n💾 Saved enriched data:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Rows: {len(df_with_features):,}")
    print(f"  Cols: {len(df_with_features.columns)}")

    # Feature statistics
    print(f"\n📊 Feature Statistics:")
    print(f"\nMomentum Features:")
    for col in momentum_features[:5]:
        mean = df_with_features[col].mean()
        std = df_with_features[col].std()
        print(f"  {col:20s} μ={mean:8.2f}  σ={std:8.2f}")

    print(f"\nVolatility Features:")
    for col in volatility_features[:5]:
        mean = df_with_features[col].mean()
        std = df_with_features[col].std()
        print(f"  {col:20s} μ={mean:8.2f}  σ={std:8.2f}")

    print("\n" + "=" * 70)
    print("✅ Technical Indicators Test Complete!")
    print("=" * 70)

    return df_with_features


if __name__ == "__main__":
    df = test_technical_indicators()
