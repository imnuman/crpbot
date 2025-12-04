"""
V6 Enhanced Integration Test
Test V6 models with proper 72-feature compatibility
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from libs.features.v6_model_loader import V6ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(symbol: str) -> pd.DataFrame:
    """Load existing CSV data for testing"""
    filename_map = {
        'BTC-USD': 'btc_data.csv',
        'ETH-USD': 'eth_data.csv', 
        'SOL-USD': 'sol_data.csv'
    }
    
    filename = filename_map.get(symbol)
    if not filename:
        raise ValueError(f"No test data for {symbol}")
    
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def test_v6_integration():
    """Test V6 enhanced integration"""
    print("=== V6 Enhanced Integration Test ===")
    
    # Initialize V6 model loader
    loader = V6ModelLoader()
    
    # Load models
    print("\n1. Loading V6 Models...")
    model_results = loader.load_all_models()
    
    for symbol, loaded in model_results.items():
        status = "✅" if loaded else "❌"
        print(f"   {status} {symbol}")
    
    if not any(model_results.values()):
        print("❌ No models loaded - cannot continue test")
        return False
    
    # Test predictions
    print("\n2. Testing V6 Predictions...")
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    results = []
    
    for symbol in symbols:
        if not model_results.get(symbol):
            print(f"   ⏭️  Skipping {symbol} (model not loaded)")
            continue
            
        try:
            # Load test data
            df = load_test_data(symbol)
            print(f"   📊 {symbol}: {len(df)} data points")
            
            # Generate prediction
            prediction = loader.predict(symbol, df)
            
            if prediction:
                results.append(prediction)
                print(f"   🎯 {symbol}: {prediction['signal']} ({prediction['confidence']:.1%})")
            else:
                print(f"   ❌ {symbol}: Prediction failed")
                
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    # Display results
    print("\n3. V6 Enhanced Results Summary")
    print("="*50)
    
    if not results:
        print("❌ No predictions generated")
        return False
    
    for result in results:
        print(f"\n{result['symbol']}:")
        print(f"  Signal: {result['signal']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Model Accuracy: {result['model_accuracy']:.1%}")
        print(f"  Features Used: {result['features_used']}")
        print(f"  Version: {result['version']}")
        
        # Show probability breakdown
        probs = result['probabilities']
        print(f"  Probabilities:")
        print(f"    BUY:  {probs['buy']:.1%}")
        print(f"    HOLD: {probs['hold']:.1%}")
        print(f"    SELL: {probs['sell']:.1%}")
    
    # Feature compatibility check
    print(f"\n4. Feature Compatibility Check")
    print("="*50)
    
    from libs.features.v6_enhanced_features import V6EnhancedFeatures
    v6_features = V6EnhancedFeatures()
    
    # Test with sample data
    sample_df = load_test_data('BTC-USD').tail(300)  # Last 300 candles
    feature_matrix = v6_features.get_feature_matrix(sample_df)
    
    print(f"✅ Generated Features: {feature_matrix.shape[1]} (expected: 72)")
    print(f"✅ Valid Data Points: {feature_matrix.shape[0]}")
    print(f"✅ Feature Names: {len(v6_features.get_feature_names())}")
    
    # Model info
    print(f"\n5. Model Information")
    print("="*50)
    
    all_info = loader.get_all_model_info()
    for symbol, info in all_info.items():
        if info:
            print(f"{symbol}:")
            print(f"  Accuracy: {info['accuracy']:.1%}")
            print(f"  Training Date: {info['training_date'][:10]}")
            print(f"  Features: {info['features']}")
            print(f"  Version: {info['version']}")
    
    print(f"\n🎉 V6 Enhanced Integration Test Complete!")
    print(f"✅ Models Loaded: {sum(model_results.values())}/3")
    print(f"✅ Predictions Generated: {len(results)}")
    print(f"✅ Feature Compatibility: {feature_matrix.shape[1] == 72}")
    
    return len(results) > 0 and feature_matrix.shape[1] == 72


if __name__ == "__main__":
    success = test_v6_integration()
    exit(0 if success else 1)
