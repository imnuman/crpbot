#!/usr/bin/env python3
"""
V6 Real Training - 31 Runtime Features
Simplified training script for GPU instance
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=31, hidden_size=64, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def train_v6_model(symbol):
    """Train V6 model with 31 features"""
    print(f"🔥 Training V6 {symbol} with 31 runtime features")
    
    # Load data
    train_df = pd.read_parquet(f"data/training/{symbol}/train.parquet")
    
    # Select first 31 numeric columns (excluding timestamp, target)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    feature_cols = numeric_cols[:31]  # Take first 31 numeric features
    
    print(f"Using features: {feature_cols}")
    
    # Prepare data
    X = train_df[feature_cols].fillna(0).values
    y = (train_df['close'].shift(-1) > train_df['close']).astype(int).values[:-1]
    X = X[:-1]  # Remove last row
    
    # Create sequences
    seq_len = 60
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"Training data: {X_seq.shape}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTM(input_size=31).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Simple training loop
    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        
        for i in range(0, len(X_seq), 256):
            batch_X = torch.FloatTensor(X_seq[i:i+256]).to(device)
            batch_y = torch.FloatTensor(y_seq[i:i+256]).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / len(y_seq)
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}, Accuracy {accuracy:.3f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/lstm_{symbol}_1m_v6_real.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'input_size': 31,
        'features': feature_cols
    }, model_path)
    
    print(f"✅ V6 model saved: {model_path}")
    return accuracy

def main():
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    
    for symbol in symbols:
        try:
            accuracy = train_v6_model(symbol)
            print(f"✅ {symbol}: {accuracy:.1%}")
        except Exception as e:
            print(f"❌ {symbol}: {e}")

if __name__ == "__main__":
    main()
