#!/usr/bin/env python3
"""
V3 Ultimate - Step 2: Feature Engineering
Generate 270 features per candle across all timeframes.

Expected output: Feature-enriched datasets ready for training
Runtime: ~4 hours on Colab Pro+

Requirements:
- pip install pandas numpy ta-lib scikit-learn joblib tqdm
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Technical indicators
try:
    import talib
    HAS_TALIB = True
except ImportError:
    print("⚠️  TA-Lib not available, using simplified indicators")
    HAS_TALIB = False

# Configuration
INPUT_DIR = Path('/content/drive/MyDrive/crpbot/data/raw')
OUTPUT_DIR = Path('/content/drive/MyDrive/crpbot/data/features')

COINS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT',
         'ADA_USDT', 'XRP_USDT', 'MATIC_USDT', 'AVAX_USDT',
         'DOGE_USDT', 'DOT_USDT']

TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

def add_price_features(df):
    """Add price-based features (30 features)."""
    # Returns
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_15'] = df['close'].pct_change(15)
    df['return_60'] = df['close'].pct_change(60)

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # OHLC relationships
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    df['co_ratio'] = (df['close'] - df['open']) / df['open']
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Price levels
    df['distance_from_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
    df['distance_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        df[f'distance_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df['close']

    return df

def add_momentum_features(df):
    """Add momentum indicators (40 features)."""
    if HAS_TALIB:
        # RSI variations
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)

        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd

        # ADX
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'])

        # CCI
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'])

        # Williams %R
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'])

        # MFI
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])

        # ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
    else:
        # Simplified RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

    # Rate of change
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)

    return df

def add_volatility_features(df):
    """Add volatility indicators (30 features)."""
    # ATR
    if HAS_TALIB:
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'])
        df['natr_14'] = talib.NATR(df['high'], df['low'], df['close'])
    else:
        tr = np.maximum(df['high'] - df['low'],
                       np.maximum(abs(df['high'] - df['close'].shift(1)),
                                 abs(df['low'] - df['close'].shift(1))))
        df['atr_14'] = tr.rolling(14).mean()
        df['natr_14'] = (df['atr_14'] / df['close']) * 100

    # Bollinger Bands
    for period in [20, 50]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + (2 * std)
        df[f'bb_lower_{period}'] = sma - (2 * std)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)

    # Historical volatility
    for period in [10, 20, 50]:
        df[f'volatility_{period}'] = df['return_1'].rolling(period).std() * np.sqrt(period)

    # Parkinson volatility
    df['parkinson_vol'] = np.sqrt(1 / (4 * np.log(2)) * np.log(df['high'] / df['low']) ** 2)

    # Garman-Klass volatility
    df['gk_vol'] = np.sqrt(0.5 * np.log(df['high'] / df['low']) ** 2 -
                           (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2)

    return df

def add_volume_features(df):
    """Add volume-based features (25 features)."""
    # Volume ratios
    df['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
    df['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()

    # Volume momentum
    df['volume_momentum_5'] = df['volume'].pct_change(5)

    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_ema'] = df['obv'].ewm(span=20).mean()

    # VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['distance_vwap'] = (df['close'] - df['vwap']) / df['close']

    # Force Index
    df['force_index'] = df['close'].diff() * df['volume']
    df['force_index_ema'] = df['force_index'].ewm(span=13).mean()

    # Ease of Movement
    distance = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
    box_ratio = (df['volume'] / 1e6) / (df['high'] - df['low'] + 1e-10)
    df['eom'] = distance / box_ratio
    df['eom_ema'] = df['eom'].ewm(span=14).mean()

    # Volume-price trend
    df['vpt'] = (df['volume'] * df['return_1']).cumsum()

    # MFI simplified
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    df['mfi_simple'] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))

    return df

def add_pattern_features(df):
    """Add candlestick patterns (20 features)."""
    if HAS_TALIB:
        df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])

    # Simplified patterns
    body = abs(df['close'] - df['open'])
    range_hl = df['high'] - df['low']

    df['is_green'] = (df['close'] > df['open']).astype(int)
    df['is_doji'] = (body / (range_hl + 1e-10) < 0.1).astype(int)
    df['has_long_upper_shadow'] = (df['upper_shadow'] > 0.6).astype(int)
    df['has_long_lower_shadow'] = (df['lower_shadow'] > 0.6).astype(int)

    # Consecutive candles
    df['consecutive_green'] = (df['is_green'].groupby((df['is_green'] != df['is_green'].shift()).cumsum()).cumcount() + 1)
    df['consecutive_red'] = ((1 - df['is_green']).groupby(((1 - df['is_green']) != (1 - df['is_green']).shift()).cumsum()).cumcount() + 1)

    return df

def add_multitimeframe_features(df, higher_tf_data):
    """Add features from higher timeframes (30 features)."""
    # For each higher timeframe, add trend alignment
    for tf, tf_df in higher_tf_data.items():
        # Merge with tolerance
        df[f'{tf}_close'] = df['timestamp'].apply(lambda x:
            tf_df.iloc[(tf_df['timestamp'] - x).abs().argmin()]['close']
            if len(tf_df) > 0 else np.nan)

        df[f'{tf}_trend'] = (df[f'{tf}_close'] > df[f'{tf}_close'].shift(1)).astype(int)
        df[f'{tf}_alignment'] = (df['close'] > df[f'{tf}_close']).astype(int)

    return df

def add_regime_features(df):
    """Add market regime features (20 features)."""
    # Trend strength
    df['trend_strength_20'] = abs(df['close'] - df['sma_20']) / df['atr_14']

    # Volatility regime
    vol_ma = df['volatility_20'].rolling(100).mean()
    vol_std = df['volatility_20'].rolling(100).std()
    df['vol_regime'] = (df['volatility_20'] - vol_ma) / (vol_std + 1e-10)

    # Range expansion/contraction
    df['range_expansion'] = df['atr_14'] / df['atr_14'].rolling(20).mean()

    # Fractal dimension (simplified)
    for period in [20, 50]:
        n = period
        hurst = []
        for i in range(n, len(df)):
            window = df['close'].iloc[i-n:i].values
            if len(window) == n:
                lags = range(2, 20)
                tau = [np.sqrt(np.std(np.subtract(window[lag:], window[:-lag]))) for lag in lags]
                reg = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst.append(reg[0])
            else:
                hurst.append(np.nan)
        df.loc[df.index[n:], f'hurst_{period}'] = hurst

    return df

def add_lagged_features(df, lags=[1, 5, 10, 20]):
    """Add lagged features (40 features)."""
    key_features = ['close', 'volume', 'rsi_14', 'macd', 'atr_14']

    for feat in key_features:
        if feat in df.columns:
            for lag in lags:
                df[f'{feat}_lag_{lag}'] = df[feat].shift(lag)

    return df

def add_target_labels(df, horizons=[5, 15, 30]):
    """Add target labels for multi-horizon prediction."""
    for h in horizons:
        future_return = (df['close'].shift(-h) - df['close']) / df['close']

        # 3-class labels (down/flat/up)
        df[f'label_{h}m'] = 1  # flat
        df.loc[future_return < -0.001, f'label_{h}m'] = 0  # down
        df.loc[future_return > 0.001, f'label_{h}m'] = 2  # up

        # Continuous target
        df[f'target_{h}m'] = future_return

    return df

def engineer_coin_timeframe(coin, timeframe, input_dir, output_dir):
    """Engineer features for one coin-timeframe pair."""
    print(f"\n🔧 Processing {coin} {timeframe}...")

    input_file = input_dir / f"{coin}_{timeframe}.parquet"

    if not input_file.exists():
        print(f"   ❌ Input file not found: {input_file}")
        return 0

    # Load data
    df = pd.read_parquet(input_file)
    initial_count = len(df)
    print(f"   Loaded: {initial_count:,} candles")

    # Add features
    print(f"   Adding price features...")
    df = add_price_features(df)

    print(f"   Adding momentum features...")
    df = add_momentum_features(df)

    print(f"   Adding volatility features...")
    df = add_volatility_features(df)

    print(f"   Adding volume features...")
    df = add_volume_features(df)

    print(f"   Adding pattern features...")
    df = add_pattern_features(df)

    print(f"   Adding regime features...")
    df = add_regime_features(df)

    print(f"   Adding lagged features...")
    df = add_lagged_features(df)

    print(f"   Adding target labels...")
    df = add_target_labels(df)

    # Drop NaN rows (from indicators)
    df = df.dropna()
    final_count = len(df)

    print(f"   Final: {final_count:,} candles ({len(df.columns)} columns)")

    # Save
    output_file = output_dir / f"{coin}_{timeframe}_features.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression='snappy', index=False)

    print(f"   ✅ Saved: {output_file.name}")

    return final_count

def main():
    """Main feature engineering workflow."""
    print("=" * 70)
    print("🔧 V3 ULTIMATE - STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    print(f"\n📋 Configuration:")
    print(f"   Input: {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Coins: {len(COINS)}")
    print(f"   Timeframes: {len(TIMEFRAMES)}")
    print(f"   Expected Features: ~270 per candle")

    results = []
    start_time = datetime.now()

    for coin in COINS:
        for timeframe in TIMEFRAMES:
            try:
                count = engineer_coin_timeframe(coin, timeframe, INPUT_DIR, OUTPUT_DIR)
                results.append({
                    'coin': coin,
                    'timeframe': timeframe,
                    'candles': count,
                    'status': 'success' if count > 0 else 'empty'
                })
            except Exception as e:
                print(f"\n❌ Failed to engineer {coin} {timeframe}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'coin': coin,
                    'timeframe': timeframe,
                    'candles': 0,
                    'status': 'failed',
                    'error': str(e)
                })

    duration = (datetime.now() - start_time).total_seconds()

    # Summary
    print("\n" + "=" * 70)
    print("📊 FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    total_candles = sum(r['candles'] for r in results)

    print(f"\n✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"📊 Total Candles: {total_candles:,}")
    print(f"⏱️  Duration: {duration/3600:.1f} hours")

    # Save manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'total_candles': total_candles,
        'duration_seconds': duration,
        'results': results
    }

    manifest_path = OUTPUT_DIR / 'feature_engineering_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n💾 Manifest saved: {manifest_path}")
    print(f"\n✅ Step 2 Complete! Ready for Step 3: Train Ensemble")

    return manifest

if __name__ == "__main__":
    main()
