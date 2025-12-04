#!/usr/bin/env python3
"""
V8 Local Training - Minimal Implementation
Train V8 models locally with expanded data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime

def create_features(df):
    """Create basic features"""
    features = df.copy()
    
    # Basic features
    features['returns'] = df['close'].pct_change()
    features['high_low_ratio'] = df['high'] / df['low']
    features['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages
    for period in [5, 10, 20]:
        features[f'sma_{period}'] = df['close'].rolling(period).mean()
        features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
    
    # Volatility
    features['volatility_20'] = features['returns'].rolling(20).std()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    return features

def create_targets(close_prices):
    """Create balanced targets"""
    future_returns = (close_prices.shift(-1) / close_prices) - 1
    targets = np.where(
        future_returns > 0.01, 2,    # BUY
        np.where(future_returns < -0.01, 0, 1)  # SELL, HOLD
    )
    return targets

def train_v8_model(symbol):
    """Train V8 model for symbol"""
    
    print(f"\n🚀 Training V8 {symbol} Model")
    
    # Load expanded data
    filename = f"{symbol.lower().replace('-', '_')}_expanded.csv"
    df = pd.read_csv(filename)
    
    print(f"✅ Loaded {len(df)} rows")
    
    # Create features
    features_df = create_features(df)
    features_df['target'] = create_targets(features_df['close'])
    features_df = features_df.dropna()
    
    # Select features
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_columns].values
    y = features_df['target'].values
    
    print(f"✅ Features: {len(feature_columns)}")
    print(f"✅ Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (V8 fix)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Scaling: mean={X_train_scaled.mean():.6f}, std={X_train_scaled.std():.6f}")
    
    # Train model (using RandomForest as proxy for neural network)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get prediction probabilities for confidence analysis
    y_proba = model.predict_proba(X_test_scaled)
    max_proba = np.max(y_proba, axis=1)
    
    # Class distribution
    pred_counts = np.bincount(y_pred, minlength=3)
    pred_dist = pred_counts / pred_counts.sum()
    
    # Confidence stats
    confidence_stats = {
        'mean': float(max_proba.mean()),
        'overconfident_99': float(np.mean(max_proba > 0.99)),
        'overconfident_95': float(np.mean(max_proba > 0.95))
    }
    
    print(f"✅ Accuracy: {accuracy:.3f}")
    print(f"✅ Avg Confidence: {confidence_stats['mean']:.3f}")
    print(f"✅ >99% Confident: {confidence_stats['overconfident_99']:.1%}")
    print(f"✅ Class Distribution: SELL={pred_dist[0]:.2f}, HOLD={pred_dist[1]:.2f}, BUY={pred_dist[2]:.2f}")
    
    # Save model info
    model_info = {
        'symbol': symbol,
        'accuracy': accuracy,
        'confidence_stats': confidence_stats,
        'prediction_distribution': pred_dist.tolist(),
        'feature_columns': feature_columns,
        'training_date': datetime.now().isoformat(),
        'version': 'v8_local',
        'data_points': len(features_df)
    }
    
    filename = f'v8_{symbol.replace("-", "_")}_model_info.json'
    with open(filename, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Saved {filename}")
    
    return model_info

def main():
    """Train all V8 models locally"""
    
    print("🚀 V8 Local Training - V6 Issues Fixed")
    print("="*50)
    
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    results = []
    
    for symbol in symbols:
        try:
            result = train_v8_model(symbol)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to train {symbol}: {e}")
    
    # Summary
    if results:
        summary = {
            'version': 'v8_local',
            'training_date': datetime.now().isoformat(),
            'models_trained': len(results),
            'results': results,
            'average_accuracy': np.mean([r['accuracy'] for r in results]),
            'average_confidence': np.mean([r['confidence_stats']['mean'] for r in results]),
            'average_overconfident': np.mean([r['confidence_stats']['overconfident_99'] for r in results])
        }
        
        with open('v8_local_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*50)
        print("V8 LOCAL TRAINING SUMMARY")
        print("="*50)
        
        for result in results:
            print(f"{result['symbol']}: {result['accuracy']:.1%} accuracy, "
                  f"{result['confidence_stats']['mean']:.1%} avg confidence")
        
        print(f"\nOverall Average: {summary['average_accuracy']:.1%} accuracy")
        print(f"Overconfident Rate: {summary['average_overconfident']:.1%}")
        
        if summary['average_overconfident'] < 0.1:
            print("🎉 SUCCESS: V8 fixes working - low overconfidence!")
        else:
            print("⚠️  Models still need tuning")
    
    print("\n✅ V8 Local Training Complete!")

if __name__ == "__main__":
    main()
