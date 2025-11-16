#!/usr/bin/env python3
"""
Create V6 training data with runtime-compatible features
Uses only basic Python - no ML dependencies
"""

import json
import os

def create_v6_metadata():
    """Create V6 training metadata with 31 runtime features"""
    
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    
    # Runtime features (31 total)
    runtime_features = [
        'returns', 'log_returns', 'price_change', 'price_range', 'body_size',
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_5', 'ema_10', 'ema_20', 'ema_50',
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_lower', 'bb_position',
        'volume_ratio', 'volatility', 'high_low_pct',
        'stoch_k', 'stoch_d', 'williams_r', 'cci',
        'atr', 'adx', 'momentum', 'roc'
    ]
    
    for symbol in symbols:
        # Create V6 directory
        v6_dir = f"data/training_v6/{symbol}"
        os.makedirs(v6_dir, exist_ok=True)
        
        # Create metadata
        metadata = {
            "symbol": symbol,
            "total_features": len(runtime_features),
            "runtime_compatible": True,
            "version": "v6",
            "features": runtime_features,
            "description": "Runtime-compatible training data with 31 features"
        }
        
        with open(f"{v6_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ {symbol}: V6 metadata created ({len(runtime_features)} features)")
    
    # Create V6 model specification
    v6_spec = {
        "version": "v6",
        "input_size": len(runtime_features),
        "architecture": "LSTM",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "features": runtime_features,
        "target_accuracy": 0.75,
        "runtime_compatible": True
    }
    
    with open("models/v6_model_spec.json", 'w') as f:
        json.dump(v6_spec, f, indent=2)
    
    print(f"\n🎯 V6 Specification: {len(runtime_features)} runtime features")
    print("✅ Ready for GPU training with correct feature count")

if __name__ == "__main__":
    create_v6_metadata()
