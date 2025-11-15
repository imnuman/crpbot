#!/usr/bin/env python3
"""
Proper GPU training script using production feature pipeline.
Can be run locally (GPU) or on Google Colab Pro.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import os

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Production LSTM Model (matches CPU training)
class LSTMDirectionModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Dataset class
class TradingDataset(Dataset):
    def __init__(self, df, feature_columns, sequence_length=60, horizon=15):
        self.df = df
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.sequences, self.labels = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        labels = []

        for i in range(self.sequence_length, len(self.df) - self.horizon):
            seq = self.df[self.feature_columns].iloc[i-self.sequence_length:i].values
            sequences.append(seq)

            current_price = self.df['close'].iloc[i]
            future_price = self.df['close'].iloc[i + self.horizon]
            pct_change = (future_price - current_price) / current_price

            if pct_change < -0.001:
                label = 0
            elif pct_change > 0.001:
                label = 2
            else:
                label = 1

            labels.append(label)

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), self.labels[idx]

def load_and_prepare_features(symbol, data_dir='data/features'):
    """Load feature file."""
    feature_file = Path(data_dir) / f"features_{symbol}_1m_2025-11-10.parquet"

    if not feature_file.exists():
        files = list(Path(data_dir).glob(f"features_{symbol}_1m_*.parquet"))
        if files:
            feature_file = sorted(files)[-1]
        else:
            raise FileNotFoundError(f"No feature file found for {symbol}")

    print(f"📂 Loading {feature_file}")
    df = pd.read_parquet(feature_file)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    exclude_cols = ["timestamp", "open", "high", "low", "close", "volume", "session", "volatility_regime"]
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    print(f"   Using {len(feature_columns)} features")

    df[feature_columns] = df[feature_columns].fillna(0)

    return df, feature_columns

def normalize_features(df, feature_columns, train_df=None):
    """Normalize features using standard scaler."""
    df = df.copy()

    if train_df is None:
        means = df[feature_columns].mean()
        stds = df[feature_columns].std().replace(0, 1)
    else:
        means = train_df[feature_columns].mean()
        stds = train_df[feature_columns].std().replace(0, 1)

    df[feature_columns] = (df[feature_columns] - means) / stds

    return df, (means, stds)

def train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device='cuda'):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return history, best_val_acc

def train_coin(symbol, epochs=15, batch_size=32, data_dir='data/features', output_dir='models/gpu_trained_proper'):
    """Train model for one symbol."""
    print(f"\n{'='*60}")
    print(f"🔥 Training {symbol}")
    print(f"{'='*60}")

    df, feature_columns = load_and_prepare_features(symbol, data_dir)

    df = df.sort_values('timestamp').reset_index(drop=True)
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()

    print(f"📊 Data split:")
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Val: {len(val_df):,} rows")

    print(f"🔧 Normalizing features...")
    train_df, norm_params = normalize_features(train_df, feature_columns)
    val_df, _ = normalize_features(val_df, feature_columns, train_df=train_df)

    print(f"📦 Creating datasets...")
    train_dataset = TradingDataset(train_df, feature_columns, sequence_length=60, horizon=15)
    val_dataset = TradingDataset(val_df, feature_columns, sequence_length=60, horizon=15)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"   Train sequences: {len(train_dataset):,}")
    print(f"   Val sequences: {len(val_dataset):,}")

    print(f"🧠 Creating model...")
    model = LSTMDirectionModel(input_size=len(feature_columns), hidden_size=64, num_layers=2)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"🚀 Starting training...")
    start_time = datetime.now()
    history, best_val_acc = train_model(model, train_loader, val_loader, epochs=epochs, device=device)
    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.1f}%")
    print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{symbol.replace('-', '_')}_lstm_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")

    metadata = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'best_val_acc': best_val_acc,
        'duration_seconds': duration,
        'epochs': epochs,
        'batch_size': batch_size,
        'num_features': len(feature_columns),
        'feature_columns': feature_columns,
        'device': str(device),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }

    metadata_path = output_dir / f"{symbol.replace('-', '_')}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata

def main():
    """Train all symbols."""
    print("🚀 GPU Model Training - Production Features")
    print("=" * 60)

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    results = []

    start_time = datetime.now()

    for symbol in symbols:
        try:
            result = train_coin(symbol, epochs=15, batch_size=32)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to train {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_duration = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"📊 Training Summary")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"\nResults:")
    for r in results:
        print(f"  {r['symbol']}: {r['best_val_acc']:.1f}% accuracy ({r['duration_seconds']:.0f}s)")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_duration': total_duration,
        'device': str(device),
        'models': results
    }

    os.makedirs('models/gpu_trained_proper', exist_ok=True)
    with open('models/gpu_trained_proper/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Summary saved: models/gpu_trained_proper/training_summary.json")

    avg_acc = sum(r['best_val_acc'] for r in results) / len(results) if results else 0
    print(f"\n📈 Average Accuracy: {avg_acc:.1f}%")

    if avg_acc >= 68:
        print("✅ Models PASS the 68% threshold!")
    else:
        print(f"⚠️  Models below 68% threshold")

    print(f"\n✅ All done!")

if __name__ == "__main__":
    main()
