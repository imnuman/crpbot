#!/usr/bin/env python3
"""Backtest GPU models with proper feature alignment."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class SimpleGPULSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_and_prepare_data(symbol="BTC-USD", days=30):
    """Load recent data and create simple features."""
    # Load raw data (files use hyphenated format: BTC-USD)
    try:
        df = pd.read_parquet(f"data/raw/{symbol}_1m_2023-11-10_2025-11-10.parquet")
    except:
        print(f"Raw data not found for {symbol}")
        return None
    
    # Get recent data
    df = df.tail(days * 1440).copy()  # 1440 minutes per day
    
    # Create simple 5 features (matching GPU training)
    df['returns'] = df['close'].pct_change()
    df['volume_norm'] = (df['volume'] - df['volume'].rolling(100).mean()) / df['volume'].rolling(100).std()
    df['price_norm'] = (df['close'] - df['close'].rolling(100).mean()) / df['close'].rolling(100).std()
    df['volatility'] = df['returns'].rolling(20).std()
    df['momentum'] = df['close'].pct_change(10)
    
    # Select features
    feature_cols = ['returns', 'volume_norm', 'price_norm', 'volatility', 'momentum']
    df = df[['timestamp', 'close'] + feature_cols].dropna()
    
    return df

def create_sequences(df, sequence_length=60, horizon=15):
    """Create sequences for prediction."""
    feature_cols = ['returns', 'volume_norm', 'price_norm', 'volatility', 'momentum']
    
    sequences = []
    targets = []
    timestamps = []
    prices = []
    
    for i in range(sequence_length, len(df) - horizon):
        # Features sequence
        seq = df[feature_cols].iloc[i-sequence_length:i].values
        sequences.append(seq)
        
        # Target (price direction after horizon)
        current_price = df['close'].iloc[i]
        future_price = df['close'].iloc[i + horizon]
        
        # 3-class target: 0=down, 1=flat, 2=up
        pct_change = (future_price - current_price) / current_price
        if pct_change < -0.001:  # -0.1%
            target = 0
        elif pct_change > 0.001:   # +0.1%
            target = 2
        else:
            target = 1
            
        targets.append(target)
        timestamps.append(df['timestamp'].iloc[i])
        prices.append(current_price)
    
    return np.array(sequences), np.array(targets), timestamps, prices

def backtest_model(model_path, symbol, days=30):
    """Backtest a single model."""
    print(f"\n📊 Backtesting {symbol}")
    
    # Load data
    df = load_and_prepare_data(symbol, days)
    if df is None:
        return None
    
    # Create sequences
    sequences, targets, timestamps, prices = create_sequences(df)
    
    if len(sequences) == 0:
        print(f"   ❌ No sequences created")
        return None
    
    # Load model
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model = SimpleGPULSTM()
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"   ❌ Model load failed: {e}")
        return None
    
    # Run predictions
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
            output = model(seq_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = torch.max(probs, dim=1).values.item()
            
            predictions.append(pred)
            confidences.append(conf)
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    # Calculate metrics
    accuracy = (predictions == targets).mean()
    
    # Trading simulation
    positions = []
    pnl = 0
    wins = 0
    losses = 0
    
    for i in range(len(predictions)):
        pred = predictions[i]
        actual = targets[i]
        conf = confidences[i]
        
        # Only trade high confidence signals
        if conf > 0.4:
            if pred == 2:  # BUY signal
                if actual == 2:  # Correct up prediction
                    pnl += 0.001  # 0.1% gain
                    wins += 1
                else:
                    pnl -= 0.001  # 0.1% loss
                    losses += 1
            elif pred == 0:  # SELL signal
                if actual == 0:  # Correct down prediction
                    pnl += 0.001  # 0.1% gain
                    wins += 1
                else:
                    pnl -= 0.001  # 0.1% loss
                    losses += 1
    
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # Calculate Sharpe-like ratio
    returns = []
    for i in range(1, len(predictions)):
        if confidences[i] > 0.4:
            if predictions[i] == 2 and targets[i] == 2:
                returns.append(0.001)
            elif predictions[i] == 0 and targets[i] == 0:
                returns.append(0.001)
            elif predictions[i] in [0, 2]:
                returns.append(-0.001)
    
    if len(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    results = {
        'symbol': symbol,
        'accuracy': accuracy,
        'total_predictions': len(predictions),
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': pnl,
        'sharpe_ratio': sharpe,
        'avg_confidence': np.mean(confidences),
        'high_conf_signals': np.sum(confidences > 0.4)
    }
    
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Win Rate: {win_rate:.1%} ({wins}/{total_trades} trades)")
    print(f"   Total PnL: {pnl:.2%}")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Avg Confidence: {np.mean(confidences):.1%}")
    
    return results

def main():
    """Run backtest on all GPU models."""
    print("🔍 GPU Model Performance Analysis")
    print("=" * 50)
    
    models = {
        "BTC-USD": "models/gpu_trained/BTC_lstm_model.pt",
        "ETH-USD": "models/gpu_trained/ETH_lstm_model.pt",
        "SOL-USD": "models/gpu_trained/SOL_lstm_model.pt", 
        "ADA-USD": "models/gpu_trained/ADA_lstm_model.pt"
    }
    
    results = []
    
    for symbol, model_path in models.items():
        result = backtest_model(model_path, symbol, days=30)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n📈 Performance Summary:")
    print("=" * 50)
    
    if results:
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results if r['total_trades'] > 0])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        total_trades = sum([r['total_trades'] for r in results])
        
        print(f"Average Accuracy: {avg_accuracy:.1%}")
        print(f"Average Win Rate: {avg_win_rate:.1%}")
        print(f"Average Sharpe: {avg_sharpe:.2f}")
        print(f"Total Trades: {total_trades}")
        
        # Check gates
        print(f"\n🎯 Performance Gates:")
        accuracy_pass = avg_accuracy >= 0.68
        win_rate_pass = avg_win_rate >= 0.55
        sharpe_pass = avg_sharpe >= 1.5
        
        print(f"   Accuracy ≥68%: {'✅' if accuracy_pass else '❌'} ({avg_accuracy:.1%})")
        print(f"   Win Rate ≥55%: {'✅' if win_rate_pass else '❌'} ({avg_win_rate:.1%})")
        print(f"   Sharpe ≥1.5: {'✅' if sharpe_pass else '❌'} ({avg_sharpe:.2f})")
        
        all_pass = accuracy_pass and win_rate_pass and sharpe_pass
        
        if all_pass:
            print(f"\n🎉 ALL GATES PASSED - Ready for production!")
        else:
            print(f"\n⚠️  Some gates failed - consider retraining or more data")
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_period_days': 30,
        'models_tested': len(results),
        'individual_results': results,
        'summary': {
            'avg_accuracy': avg_accuracy if results else 0,
            'avg_win_rate': avg_win_rate if results else 0,
            'avg_sharpe': avg_sharpe if results else 0,
            'total_trades': total_trades if results else 0
        } if results else {}
    }
    
    with open('gpu_backtest_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Results saved to gpu_backtest_results.json")
    
    return all_pass if results else False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
