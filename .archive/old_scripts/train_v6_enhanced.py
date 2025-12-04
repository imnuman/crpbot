#!/usr/bin/env python3
"""V6 Enhanced Model Training with 155 Features for >68% Accuracy"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from loguru import logger
from apps.runtime.runtime_features import engineer_runtime_features
from apps.runtime.data_fetcher import get_data_fetcher
from apps.trainer.main import train_lstm_model
import os

def engineer_enhanced_features(df, symbol, data_fetcher):
    """Engineer all 155 features for V6 enhanced models"""
    logger.info(f"Engineering enhanced features for {symbol}...")
    
    # Base + Multi-TF features (81 features)
    df_features = engineer_runtime_features(
        df, symbol, data_fetcher, 
        include_multi_tf=True, 
        include_coingecko=True
    )
    
    # Add advanced technical indicators (74 additional features)
    # Price momentum features
    for period in [3, 7, 14, 21, 30]:
        df_features[f'price_momentum_{period}'] = df_features['close'].pct_change(period)
        df_features[f'volume_momentum_{period}'] = df_features['volume'].pct_change(period)
    
    # Volatility features
    for window in [5, 10, 20, 30]:
        df_features[f'volatility_{window}'] = df_features['close'].rolling(window).std()
        df_features[f'price_range_{window}'] = (df_features['high'] - df_features['low']).rolling(window).mean()
    
    # Advanced moving averages
    for period in [9, 12, 26, 50, 100, 200]:
        df_features[f'ema_{period}'] = df_features['close'].ewm(span=period).mean()
        df_features[f'price_ema_{period}_ratio'] = df_features['close'] / df_features[f'ema_{period}']
    
    # RSI variations
    for period in [9, 14, 21]:
        delta = df_features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df_features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD variations
    for fast, slow, signal in [(8, 21, 5), (12, 26, 9), (19, 39, 9)]:
        ema_fast = df_features['close'].ewm(span=fast).mean()
        ema_slow = df_features['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        df_features[f'macd_{fast}_{slow}'] = macd_line
        df_features[f'macd_signal_{fast}_{slow}'] = signal_line
        df_features[f'macd_hist_{fast}_{slow}'] = macd_line - signal_line
    
    # Bollinger Band variations
    for period, std_dev in [(10, 1.5), (20, 2), (50, 2.5)]:
        sma = df_features['close'].rolling(period).mean()
        std = df_features['close'].rolling(period).std()
        df_features[f'bb_upper_{period}'] = sma + (std * std_dev)
        df_features[f'bb_lower_{period}'] = sma - (std * std_dev)
        df_features[f'bb_position_{period}'] = (df_features['close'] - df_features[f'bb_lower_{period}']) / (df_features[f'bb_upper_{period}'] - df_features[f'bb_lower_{period}'])
    
    # Fill NaN values
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[numeric_cols] = df_features[numeric_cols].ffill().bfill().fillna(0)
    
    logger.info(f"Enhanced features engineered: {len(df_features.columns)} total features")
    return df_features

def train_v6_enhanced_model(symbol):
    """Train V6 enhanced model with 155 features"""
    logger.info(f"🚀 Training V6 Enhanced {symbol} Model")
    
    # Load training data
    df = pd.read_parquet(f'data/training_{symbol.replace("-", "_")}.parquet')
    logger.info(f"Loaded {len(df)} candles for {symbol}")
    
    # Engineer enhanced features
    data_fetcher = get_data_fetcher()
    df_features = engineer_enhanced_features(df, symbol, data_fetcher)
    
    # Prepare training data
    feature_cols = [col for col in df_features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Create target (next candle direction)
    df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
    df_features = df_features.dropna()
    
    if len(df_features) < 60:
        logger.error(f"Insufficient data for {symbol}: {len(df_features)} rows")
        return None
    
    logger.info(f"Training with {len(feature_cols)} features on {len(df_features)} samples")
    
    # Train LSTM model
    model_path = train_lstm_model(
        df=df_features,
        symbol=symbol,
        feature_cols=feature_cols,
        target_col='target',
        model_version='v6_enhanced',
        epochs=50,
        batch_size=32,
        hidden_size=128,
        num_layers=3,
        dropout=0.3
    )
    
    return model_path

if __name__ == '__main__':
    logger.info("🚀 Starting V6 Enhanced Model Training Pipeline")
    logger.info("Target: >68% accuracy with 155 features")
    
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    for symbol in symbols:
        try:
            model_path = train_v6_enhanced_model(symbol)
            if model_path:
                logger.info(f"✅ {symbol} V6 Enhanced model trained: {model_path}")
            else:
                logger.error(f"❌ {symbol} training failed")
        except Exception as e:
            logger.error(f"❌ {symbol} training error: {e}")
    
    logger.info("🎯 V6 Enhanced Training Complete!")
