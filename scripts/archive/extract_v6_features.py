#!/usr/bin/env python3
"""
Extract V6 features from existing training data
Creates new training files with only runtime-compatible features
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Runtime-compatible features (31 features)
RUNTIME_FEATURES = [
    'returns', 'log_returns', 'price_change', 'price_range', 'body_size',
    'sma_5', 'sma_10', 'sma_20', 'sma_50',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_lower', 'bb_position',
    'volume_ratio', 'volatility', 'high_low_pct'
]

def generate_runtime_features(df):
    """Generate the exact features that runtime produces"""
    print(f"Generating features for {len(df)} rows...")
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_change'] = df['close'] - df['open']
    df['price_range'] = df['high'] - df['low']
    df['body_size'] = abs(df['close'] - df['open'])
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Price position
    df['high_low_pct'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    return df

def process_symbol(symbol):
    """Process one symbol and create V6 training data"""
    print(f"\n🔄 Processing {symbol}...")
    
    # Load original data
    train_path = f"data/training/{symbol}/train.parquet"
    val_path = f"data/training/{symbol}/val.parquet"
    test_path = f"data/training/{symbol}/test.parquet"
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return False
    
    # Process each split
    for split, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if not os.path.exists(path):
            print(f"⚠️  Skipping {split}: {path} not found")
            continue
            
        print(f"  Processing {split}...")
        df = pd.read_parquet(path)
        
        # Generate runtime features
        df = generate_runtime_features(df)
        
        # Create target (next candle direction)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Select only runtime features + target + timestamp
        feature_cols = ['timestamp'] + RUNTIME_FEATURES + ['target']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        v6_df = df[available_cols].copy()
        
        # Fill NaN values
        for col in RUNTIME_FEATURES:
            if col in v6_df.columns:
                v6_df[col] = v6_df[col].fillna(method='ffill').fillna(0)
        
        # Remove last row (no target)
        v6_df = v6_df[:-1]
        
        # Save V6 data
        output_dir = f"data/training_v6/{symbol}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{split}.parquet"
        
        v6_df.to_parquet(output_path, index=False)
        
        print(f"    ✅ {split}: {len(v6_df)} rows, {len([c for c in RUNTIME_FEATURES if c in v6_df.columns])} features")
    
    # Create metadata
    metadata = {
        "symbol": symbol,
        "features": len([c for c in RUNTIME_FEATURES if c in v6_df.columns]),
        "runtime_compatible": True,
        "version": "v6"
    }
    
    import json
    with open(f"data/training_v6/{symbol}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return True

def main():
    print("🚀 V6 Feature Extraction Started")
    print(f"Target features: {len(RUNTIME_FEATURES)} runtime-compatible")
    
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    
    for symbol in symbols:
        success = process_symbol(symbol)
        if success:
            print(f"✅ {symbol}: V6 data created")
        else:
            print(f"❌ {symbol}: Failed")
    
    print("\n🎯 V6 Feature Extraction Complete!")
    print("Next: Upload V6 data to GPU instance for training")

if __name__ == "__main__":
    main()
