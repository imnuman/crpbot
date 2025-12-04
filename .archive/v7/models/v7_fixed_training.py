"""
V7 Enhanced Training - Fixed BatchNorm Issue
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== V7 Enhanced Training (Fixed) ===")
print(f"CUDA available: {torch.cuda.is_available()}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class V7TradingNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=3, dropout=0.3):
        super(V7TradingNet, self).__init__()
        
        # Layers without input batch norm to avoid single sample issues
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        # Temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * 2.5)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output with temperature scaling
        logits = self.fc4(x)
        scaled_logits = logits / self.temperature
        
        return scaled_logits

def create_enhanced_features(df):
    features = df.copy()
    close_col = 'close'
    high_col = 'high'
    low_col = 'low'
    open_col = 'open'
    volume_col = 'volume'
    
    # Basic features
    features['returns'] = df[close_col].pct_change()
    features['log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
    features['high_low_ratio'] = df[high_col] / df[low_col]
    features['close_open_ratio'] = df[close_col] / df[open_col]
    features['volume_ratio'] = df[volume_col] / df[volume_col].rolling(20).mean()
    features['volume_price_trend'] = df[volume_col] * features['returns']
    
    # Momentum indicators
    for period in [5, 10, 20, 50]:
        features[f'momentum_{period}'] = df[close_col] / df[close_col].shift(period) - 1
        features[f'roc_{period}'] = ((df[close_col] - df[close_col].shift(period)) / df[close_col].shift(period)) * 100
    
    # Moving averages
    for period in [5, 10, 20, 50, 200]:
        features[f'sma_{period}'] = df[close_col].rolling(period).mean()
        features[f'ema_{period}'] = df[close_col].ewm(span=period).mean()
        features[f'price_to_sma_{period}'] = df[close_col] / features[f'sma_{period}']
        features[f'price_to_ema_{period}'] = df[close_col] / features[f'ema_{period}']
    
    # Volatility
    features['volatility_20'] = features['returns'].rolling(20).std()
    features['volatility_50'] = features['returns'].rolling(50).std()
    features['atr_14'] = (df[high_col] - df[low_col]).rolling(14).mean()
    
    # RSI
    for period in [14, 21, 30]:
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for period in [20, 50]:
        sma = df[close_col].rolling(period).mean()
        std = df[close_col].rolling(period).std()
        features[f'bb_upper_{period}'] = sma + (2 * std)
        features[f'bb_lower_{period}'] = sma - (2 * std)
        features[f'bb_position_{period}'] = (df[close_col] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
    
    # MACD
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
        ema_fast = df[close_col].ewm(span=fast).mean()
        ema_slow = df[close_col].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        features[f'macd_{fast}_{slow}'] = macd
        features[f'macd_signal_{fast}_{slow}'] = macd_signal
        features[f'macd_histogram_{fast}_{slow}'] = macd - macd_signal
    
    # Stochastic
    for period in [14, 21]:
        low_min = df[low_col].rolling(window=period).min()
        high_max = df[high_col].rolling(window=period).max()
        features[f'stoch_k_{period}'] = 100 * ((df[close_col] - low_min) / (high_max - low_min))
        features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()
    
    # Williams %R
    for period in [14, 21]:
        high_max = df[high_col].rolling(window=period).max()
        low_min = df[low_col].rolling(window=period).min()
        features[f'williams_r_{period}'] = -100 * ((high_max - df[close_col]) / (high_max - low_min))
    
    # Price channels
    for period in [20, 50]:
        features[f'price_channel_high_{period}'] = df[high_col].rolling(period).max()
        features[f'price_channel_low_{period}'] = df[low_col].rolling(period).min()
        features[f'price_channel_position_{period}'] = (df[close_col] - features[f'price_channel_low_{period}']) / (features[f'price_channel_high_{period}'] - features[f'price_channel_low_{period}'])
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        features[f'volume_lag_{lag}'] = features['volume_ratio'].shift(lag)
    
    return features

def train_v7_model(symbol, epochs=50):  # Reduced epochs for speed
    print(f"\n=== Training V7 {symbol} ===")
    
    # Load data
    filename_map = {'BTC-USD': 'btc_data.csv', 'ETH-USD': 'eth_data.csv', 'SOL-USD': 'sol_data.csv'}
    df = pd.read_csv(filename_map[symbol])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Create features
    features_df = create_enhanced_features(df)
    features_df['target'] = np.where(
        features_df['close'].shift(-1) > features_df['close'] * 1.015, 2,  # BUY
        np.where(features_df['close'].shift(-1) < features_df['close'] * 0.985, 0, 1)  # SELL, HOLD
    )
    
    features_df = features_df.dropna()
    
    # Prepare features
    exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_columns].values
    y = features_df['target'].values
    
    print(f"Features: {len(feature_columns)}, Data: {len(X)}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # CRITICAL: Feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Scaling: mean={X_train_scaled.mean():.3f}, std={X_train_scaled.std():.3f}")
    
    # Random Forest baseline
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest: {rf_accuracy:.3f}")
    
    # Neural Network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Model
    model = V7TradingNet(X_train.shape[1], hidden_size=256, dropout=0.3).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training
    best_accuracy = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_probs = F.softmax(val_outputs, dim=1)
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy = (val_predicted == y_test_tensor).float().mean().item()
                
                max_probs = torch.max(val_probs, dim=1)[0]
                avg_confidence = max_probs.mean().item()
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model_state = model.state_dict().copy()
                
                print(f"Epoch {epoch+1:2d}: Loss={epoch_loss/len(train_loader):.4f}, "
                      f"Acc={val_accuracy:.3f}, Conf={avg_confidence:.3f}, "
                      f"Temp={model.temperature.item():.2f}")
    
    # Final evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probs = F.softmax(test_outputs, dim=1)
        _, predicted = torch.max(test_outputs, 1)
        final_accuracy = (predicted == y_test_tensor).float().mean().item()
        
        max_probs = torch.max(test_probs, dim=1)[0]
        confidence_stats = {
            'mean': max_probs.mean().item(),
            'std': max_probs.std().item(),
            'min': max_probs.min().item(),
            'max': max_probs.max().item()
        }
        
        pred_dist = torch.bincount(predicted, minlength=3).float()
        pred_dist = pred_dist / pred_dist.sum()
    
    print(f"Final: Acc={final_accuracy:.3f}, Conf={confidence_stats['mean']:.3f}")
    print(f"Pred dist: SELL={pred_dist[0]:.2f}, HOLD={pred_dist[1]:.2f}, BUY={pred_dist[2]:.2f}")
    
    # Save model
    model_data = {
        'model_state_dict': best_model_state,
        'accuracy': best_accuracy,
        'final_accuracy': final_accuracy,
        'input_size': X_train.shape[1],
        'feature_columns': feature_columns,
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'confidence_stats': confidence_stats,
        'prediction_distribution': pred_dist.cpu().numpy().tolist(),
        'version': 'v7_enhanced_fixed',
        'training_date': datetime.now().isoformat(),
        'symbol': symbol,
        'data_points': len(features_df)
    }
    
    filename = f'lstm_{symbol}_v7_enhanced.pt'
    torch.save(model_data, filename)
    print(f"✅ Saved {filename}")
    
    return {
        'symbol': symbol,
        'rf_accuracy': rf_accuracy,
        'nn_accuracy': final_accuracy,
        'confidence_stats': confidence_stats,
        'filename': filename
    }

# Main execution
if __name__ == "__main__":
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    results = []
    
    print("🚀 V7 Enhanced Training (Fixed)")
    
    for symbol in symbols:
        result = train_v7_model(symbol, epochs=50)
        results.append(result)
    
    # Summary
    summary = {
        'version': 'v7_enhanced_fixed',
        'training_date': datetime.now().isoformat(),
        'models': results,
        'average_accuracy': np.mean([r['nn_accuracy'] for r in results]),
        'average_confidence': np.mean([r['confidence_stats']['mean'] for r in results])
    }
    
    with open('v7_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("V7 ENHANCED RESULTS")
    print("="*50)
    
    for result in results:
        print(f"{result['symbol']}: {result['nn_accuracy']:.1%} "
              f"(conf: {result['confidence_stats']['mean']:.1%})")
    
    print(f"\nAverage: {summary['average_accuracy']:.1%}")
    print("🎉 V7 Training Complete!")

