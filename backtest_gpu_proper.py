#!/usr/bin/env python3
"""Backtest GPU models trained with proper production features."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# LSTM Model (matches training architecture)
class LSTMDirectionModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_features(symbol, data_dir='data/features'):
    """Load production feature file."""
    feature_file = Path(data_dir) / f"features_{symbol}_1m_2025-11-10.parquet"

    if not feature_file.exists():
        files = list(Path(data_dir).glob(f"features_{symbol}_1m_*.parquet"))
        if files:
            feature_file = sorted(files)[-1]
        else:
            raise FileNotFoundError(f"No feature file found for {symbol}")

    print(f"  Loading {feature_file.name}")
    df = pd.read_parquet(feature_file)

    # Get feature columns (exclude non-features)
    exclude_cols = ["timestamp", "open", "high", "low", "close", "volume", "session", "volatility_regime"]
    feature_columns = [col for col in df.columns if col not in exclude_cols]

    # Fill NaN
    df[feature_columns] = df[feature_columns].fillna(0)

    return df, feature_columns

def normalize_features(df, feature_columns):
    """Normalize features."""
    df = df.copy()
    means = df[feature_columns].mean()
    stds = df[feature_columns].std().replace(0, 1)
    df[feature_columns] = (df[feature_columns] - means) / stds
    return df

def create_sequences(df, feature_columns, sequence_length=60, horizon=15):
    """Create sequences for prediction."""
    sequences = []
    labels = []
    timestamps = []
    prices = []

    for i in range(sequence_length, len(df) - horizon):
        seq = df[feature_columns].iloc[i-sequence_length:i].values
        sequences.append(seq)

        current_price = df['close'].iloc[i]
        future_price = df['close'].iloc[i + horizon]
        pct_change = (future_price - current_price) / current_price

        # 3-class: 0=down, 1=flat, 2=up
        if pct_change < -0.001:
            label = 0
        elif pct_change > 0.001:
            label = 2
        else:
            label = 1

        labels.append(label)
        timestamps.append(df['timestamp'].iloc[i])
        prices.append(current_price)

    return np.array(sequences, dtype=np.float32), np.array(labels), timestamps, prices

def backtest_model(symbol, model_path, days=30):
    """Backtest a single model."""
    print(f"\n{'='*60}")
    print(f"📊 Backtesting {symbol}")
    print(f"{'='*60}")

    # Load features
    df, feature_columns = load_features(symbol)
    print(f"  Total rows: {len(df):,}")
    print(f"  Features: {len(feature_columns)}")

    # Use last N days
    test_rows = days * 1440  # 1440 minutes per day
    df = df.tail(test_rows).copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  Test rows: {len(df):,} ({days} days)")

    # Normalize
    df = normalize_features(df, feature_columns)

    # Create sequences
    sequences, labels, timestamps, prices = create_sequences(df, feature_columns)
    print(f"  Sequences: {len(sequences):,}")

    if len(sequences) == 0:
        print("  ❌ No sequences created")
        return None

    # Load model
    try:
        model = LSTMDirectionModel(input_size=len(feature_columns), hidden_size=64, num_layers=2)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  ✅ Model loaded")
    except Exception as e:
        print(f"  ❌ Model load failed: {e}")
        return None

    # Run predictions
    predictions = []
    confidences = []

    print(f"  Running predictions...")
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

    # Calculate accuracy
    accuracy = (predictions == labels).mean()

    # Per-class accuracy
    class_acc = {}
    for cls in [0, 1, 2]:
        mask = labels == cls
        if mask.sum() > 0:
            class_acc[cls] = (predictions[mask] == labels[mask]).mean()
        else:
            class_acc[cls] = 0

    # Trading simulation (high confidence only)
    wins = 0
    losses = 0
    total_pnl = 0
    conf_threshold = 0.4

    for i in range(len(predictions)):
        pred = predictions[i]
        actual = labels[i]
        conf = confidences[i]

        if conf > conf_threshold:
            if pred == 2:  # BUY
                if actual == 2:
                    wins += 1
                    total_pnl += 0.001
                else:
                    losses += 1
                    total_pnl -= 0.001
            elif pred == 0:  # SELL
                if actual == 0:
                    wins += 1
                    total_pnl += 0.001
                else:
                    losses += 1
                    total_pnl -= 0.001

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0

    # Sharpe ratio
    returns = []
    for i in range(len(predictions)):
        if confidences[i] > conf_threshold:
            if predictions[i] == 2 and labels[i] == 2:
                returns.append(0.001)
            elif predictions[i] == 0 and labels[i] == 0:
                returns.append(0.001)
            elif predictions[i] in [0, 2]:
                returns.append(-0.001)

    if len(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # Results
    print(f"\n  📈 Results:")
    print(f"    Accuracy: {accuracy:.1%}")
    print(f"    Class 0 (Down): {class_acc[0]:.1%}")
    print(f"    Class 1 (Flat): {class_acc[1]:.1%}")
    print(f"    Class 2 (Up): {class_acc[2]:.1%}")
    print(f"    Trades: {total_trades:,} (conf>{conf_threshold})")
    print(f"    Win Rate: {win_rate:.1%} ({wins}/{total_trades})")
    print(f"    Total PnL: {total_pnl:.2%}")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Avg Confidence: {np.mean(confidences):.1%}")

    return {
        'symbol': symbol,
        'accuracy': accuracy,
        'class_accuracy': class_acc,
        'total_predictions': len(predictions),
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'sharpe_ratio': sharpe,
        'avg_confidence': np.mean(confidences),
        'high_conf_signals': np.sum(confidences > conf_threshold)
    }

def main():
    """Run backtest on all GPU models."""
    print("🔍 GPU Model Backtest - Production Features")
    print("=" * 60)

    # Map symbols to model files
    models = {
        "BTC-USD": "models/gpu_trained_proper/BTC_lstm_model.pt",
        "ETH-USD": "models/gpu_trained_proper/ETH_lstm_model.pt",
        "SOL-USD": "models/gpu_trained_proper/SOL_lstm_model.pt",
        "ADA-USD": "models/gpu_trained_proper/ADA_lstm_model.pt"
    }

    results = []

    for symbol, model_path in models.items():
        try:
            result = backtest_model(symbol, model_path, days=30)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to backtest {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")

    if results:
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results if r['total_trades'] > 0])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        total_trades = sum([r['total_trades'] for r in results])

        print(f"\nAverage Accuracy: {avg_accuracy:.1%}")
        print(f"Average Win Rate: {avg_win_rate:.1%}")
        print(f"Average Sharpe: {avg_sharpe:.2f}")
        print(f"Total Trades: {total_trades:,}")

        # Performance gates
        print(f"\n🎯 Performance Gates:")
        accuracy_pass = avg_accuracy >= 0.68
        win_rate_pass = avg_win_rate >= 0.55
        sharpe_pass = avg_sharpe >= 1.5

        print(f"  Accuracy ≥68%: {'✅' if accuracy_pass else '❌'} ({avg_accuracy:.1%})")
        print(f"  Win Rate ≥55%: {'✅' if win_rate_pass else '❌'} ({avg_win_rate:.1%})")
        print(f"  Sharpe ≥1.5: {'✅' if sharpe_pass else '❌'} ({avg_sharpe:.2f})")

        all_pass = accuracy_pass and win_rate_pass and sharpe_pass

        if all_pass:
            print(f"\n🎉 ALL GATES PASSED - Models ready for production!")
        else:
            print(f"\n⚠️  Some gates failed - Models need improvement")

        # Save results
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_period_days': 30,
            'models_tested': len(results),
            'individual_results': results,
            'summary': {
                'avg_accuracy': float(avg_accuracy),
                'avg_win_rate': float(avg_win_rate),
                'avg_sharpe': float(avg_sharpe),
                'total_trades': int(total_trades),
                'all_gates_passed': all_pass
            }
        }

        with open('gpu_proper_backtest_results.json', 'w') as f:
            json.dump(report, f, indent=2, default=float)

        print(f"\n💾 Results saved: gpu_proper_backtest_results.json")

        return all_pass
    else:
        print("\n❌ No results to summarize")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
