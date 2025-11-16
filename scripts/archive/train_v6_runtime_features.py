#!/usr/bin/env python3
"""
V6 Training Script - Runtime Feature Compatible
Trains models using only the 31 features that runtime can generate
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Runtime-compatible features (31 features)
RUNTIME_FEATURES = [
    'returns', 'log_returns', 'price_change', 'price_range', 'body_size',
    'sma_5', 'sma_10', 'sma_20', 'sma_50',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_lower', 'bb_position',
    'volume_ratio', 'volatility', 'high_low_pct',
    'stoch_k', 'stoch_d', 'williams_r', 'cci',
    'atr', 'adx', 'momentum', 'roc'
]

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def generate_runtime_features(df):
    """Generate the exact 31 features that runtime produces"""
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
    
    # Additional indicators
    df['high_low_pct'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
    
    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['cci'] = (tp - sma_tp) / (0.015 * mad)
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # ADX (simplified)
    df['adx'] = df['atr'].rolling(14).mean() / df['close'] * 100
    
    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10)
    
    # Rate of Change
    df['roc'] = (df['close'] / df['close'].shift(12) - 1) * 100
    
    return df

def prepare_sequences(df, seq_length=60):
    """Create sequences using only runtime features"""
    # Generate all features
    df = generate_runtime_features(df)
    
    # Select only runtime features
    feature_df = df[RUNTIME_FEATURES].copy()
    
    # Fill NaN values
    feature_df = feature_df.fillna(method='ffill').fillna(0)
    
    # Create target (next candle up/down)
    targets = (df['close'].shift(-1) > df['close']).astype(int)
    
    X, y = [], []
    for i in range(seq_length, len(feature_df) - 1):
        X.append(feature_df.iloc[i-seq_length:i].values)
        y.append(targets.iloc[i])
    
    return np.array(X), np.array(y)

def train_v6_model(symbol, epochs=10):
    """Train V6 model with runtime features"""
    logger.info(f"🔥 Training V6 {symbol} with {len(RUNTIME_FEATURES)} runtime features")
    
    # Load original training data
    train_df = pd.read_parquet(f"data/training/{symbol}/train.parquet")
    val_df = pd.read_parquet(f"data/training/{symbol}/val.parquet")
    
    # Prepare sequences with runtime features only
    X_train, y_train = prepare_sequences(train_df)
    X_val, y_val = prepare_sequences(val_df)
    
    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTM(input_size=len(RUNTIME_FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        # Simple batch processing
        batch_size = 256
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(device)
            batch_y = torch.FloatTensor(y_train[i:i+batch_size]).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = torch.FloatTensor(X_val[i:i+batch_size]).to(device)
                batch_y = torch.FloatTensor(y_val[i:i+batch_size]).to(device)
                
                outputs = model(batch_X).squeeze()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
        
        train_acc = train_correct / len(y_train)
        val_acc = val_correct / len(y_val)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            
            # Save model
            model_path = f"models/lstm_{symbol}_1m_v6_runtime.pt"
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'epoch': epoch,
                'input_size': len(RUNTIME_FEATURES),
                'features': RUNTIME_FEATURES
            }, model_path)
            
            size = os.path.getsize(model_path)
            logger.info(f"✅ Model saved: {model_path} ({size:,} bytes)")
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.3f}, Val: {val_acc:.3f}")
    
    logger.info(f"🎯 {symbol} V6 Final Accuracy: {best_accuracy:.3f}")
    return best_accuracy

def main():
    logger.info("🚀 V6 Runtime-Compatible Training Started")
    logger.info(f"Features: {len(RUNTIME_FEATURES)} runtime-compatible")
    
    results = {}
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    
    for symbol in symbols:
        accuracy = train_v6_model(symbol)
        results[symbol] = accuracy
        logger.info(f"✅ {symbol}: {accuracy:.1%}")
    
    logger.info("\n🎯 V6 TRAINING SUMMARY:")
    for symbol, acc in results.items():
        status = "✅ PASSED" if acc > 0.65 else "⚠️  LOW"
        logger.info(f"  {symbol}: {acc:.1%} {status}")
    
    return results

if __name__ == "__main__":
    main()
