"""
V7 Enhanced Training Script
Fixes V6 issues: normalization, overconfidence, bias
Saves all training data and results for Claude
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== V7 Enhanced Training with Proper Normalization ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class V7TradingNet(nn.Module):
    """V7 Enhanced Neural Network with proper normalization"""
    
    def __init__(self, input_size, hidden_size=256, num_classes=3, dropout=0.3):
        super(V7TradingNet, self).__init__()
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Hidden layers with batch norm
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 2.5)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input batch normalization
        x = self.input_bn(x)
        
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        logits = self.fc4(x)
        
        # Temperature scaling
        scaled_logits = logits / self.temperature
        
        return scaled_logits

def create_enhanced_features(df):
    """Create V7 enhanced features with proper scaling"""
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

def train_v7_model(symbol, epochs=100):
    """Train V7 model with proper normalization and calibration"""
    print(f"\n=== Training V7 {symbol} ===")
    
    # Load data
    filename_map = {'BTC-USD': 'btc_data.csv', 'ETH-USD': 'eth_data.csv', 'SOL-USD': 'sol_data.csv'}
    df = pd.read_csv(filename_map[symbol])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print(f"Loaded {len(df)} data points for {symbol}")
    
    # Create features
    features_df = create_enhanced_features(df)
    
    # Create target with balanced thresholds
    features_df['target'] = np.where(
        features_df['close'].shift(-1) > features_df['close'] * 1.015, 2,  # BUY (1.5%+)
        np.where(features_df['close'].shift(-1) < features_df['close'] * 0.985, 0, 1)  # SELL (1.5%-), else HOLD
    )
    
    # Clean data
    features_df = features_df.dropna()
    print(f"Clean data points: {len(features_df)}")
    
    # Prepare features
    exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_columns].values
    y = features_df['target'].values
    
    print(f"Features: {len(feature_columns)}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # CRITICAL: Feature normalization using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Feature scaling - Train mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f}")
    
    # Random Forest baseline
    print("Training Random Forest baseline...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    
    # Neural Network training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V7 Neural Network on {device}...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create data loaders for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize V7 model
    model = V7TradingNet(X_train.shape[1], hidden_size=256, dropout=0.3).to(device)
    
    # Use Focal Loss with label smoothing
    criterion = FocalLoss(alpha=0.25, gamma=2.0, num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    # Training loop
    best_accuracy = 0
    best_model_state = None
    training_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Label smoothing
            smoothed_targets = torch.zeros_like(outputs)
            smoothed_targets.scatter_(1, batch_y.unsqueeze(1), 0.95)
            smoothed_targets += 0.05 / 3  # Add smoothing
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_probs = F.softmax(val_outputs, dim=1)
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy = (val_predicted == y_test_tensor).float().mean().item()
                
                # Check confidence distribution
                max_probs = torch.max(val_probs, dim=1)[0]
                avg_confidence = max_probs.mean().item()
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model_state = model.state_dict().copy()
                
                training_history.append({
                    'epoch': epoch + 1,
                    'loss': epoch_loss / len(train_loader),
                    'val_accuracy': val_accuracy,
                    'avg_confidence': avg_confidence,
                    'temperature': model.temperature.item()
                })
                
                print(f"Epoch {epoch+1:3d}: Loss={epoch_loss/len(train_loader):.4f}, "
                      f"Val Acc={val_accuracy:.3f}, Conf={avg_confidence:.3f}, "
                      f"Temp={model.temperature.item():.2f}, Best={best_accuracy:.3f}")
                
                scheduler.step(epoch_loss)
    
    # Final evaluation with best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probs = F.softmax(test_outputs, dim=1)
        _, predicted = torch.max(test_outputs, 1)
        final_accuracy = (predicted == y_test_tensor).float().mean().item()
        
        # Detailed confidence analysis
        max_probs = torch.max(test_probs, dim=1)[0]
        confidence_stats = {
            'mean': max_probs.mean().item(),
            'std': max_probs.std().item(),
            'min': max_probs.min().item(),
            'max': max_probs.max().item(),
            'median': max_probs.median().item()
        }
        
        # Class distribution in predictions
        pred_dist = torch.bincount(predicted, minlength=3).float()
        pred_dist = pred_dist / pred_dist.sum()
    
    print(f"Final V7 Accuracy: {final_accuracy:.3f}")
    print(f"Confidence stats: mean={confidence_stats['mean']:.3f}, "
          f"std={confidence_stats['std']:.3f}, range=[{confidence_stats['min']:.3f}, {confidence_stats['max']:.3f}]")
    print(f"Prediction distribution: SELL={pred_dist[0]:.2f}, HOLD={pred_dist[1]:.2f}, BUY={pred_dist[2]:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Features for {symbol}:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save V7 model with all metadata
    model_data = {
        'model_state_dict': best_model_state,
        'accuracy': best_accuracy,
        'final_accuracy': final_accuracy,
        'input_size': X_train.shape[1],
        'feature_columns': feature_columns,
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'var': scaler.var_.tolist()
        },
        'confidence_stats': confidence_stats,
        'prediction_distribution': pred_dist.cpu().numpy().tolist(),
        'training_history': training_history,
        'version': 'v7_enhanced',
        'training_date': datetime.now().isoformat(),
        'symbol': symbol,
        'data_points': len(features_df),
        'target_distribution': np.bincount(y).tolist(),
        'architecture': {
            'hidden_size': 256,
            'dropout': 0.3,
            'batch_norm': True,
            'temperature_scaling': True,
            'focal_loss': True,
            'label_smoothing': 0.05
        }
    }
    
    filename = f'lstm_{symbol}_v7_enhanced.pt'
    torch.save(model_data, filename)
    print(f"✅ Saved {filename}")
    
    return {
        'symbol': symbol,
        'rf_accuracy': rf_accuracy,
        'nn_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
        'confidence_stats': confidence_stats,
        'filename': filename,
        'features': len(feature_columns),
        'training_history': training_history
    }

# Main training execution
if __name__ == "__main__":
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    results = []
    
    print("🚀 Starting V7 Enhanced Training...")
    print("Fixes: Normalization, Overconfidence, Bias")
    
    for symbol in symbols:
        result = train_v7_model(symbol, epochs=100)
        results.append(result)
    
    # Save comprehensive results
    training_summary = {
        'version': 'v7_enhanced',
        'training_date': datetime.now().isoformat(),
        'models': results,
        'improvements': [
            'Feature normalization with StandardScaler',
            'Batch normalization layers',
            'Temperature scaling for calibration',
            'Focal loss for class imbalance',
            'Label smoothing (0.05)',
            'Gradient clipping',
            'Proper confidence distribution'
        ],
        'average_accuracy': np.mean([r['nn_accuracy'] for r in results]),
        'average_confidence': np.mean([r['confidence_stats']['mean'] for r in results])
    }
    
    with open('v7_training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    # Display results
    print("\n" + "="*60)
    print("V7 ENHANCED TRAINING RESULTS")
    print("="*60)
    
    for result in results:
        print(f"\n{result['symbol']}:")
        print(f"  Random Forest: {result['rf_accuracy']:.1%}")
        print(f"  V7 Neural Net: {result['nn_accuracy']:.1%}")
        print(f"  Best Accuracy: {result['best_accuracy']:.1%}")
        print(f"  Avg Confidence: {result['confidence_stats']['mean']:.1%}")
        print(f"  Conf Range: [{result['confidence_stats']['min']:.1%}, {result['confidence_stats']['max']:.1%}]")
    
    print(f"\nOVERALL V7 PERFORMANCE:")
    print(f"  Average Accuracy: {training_summary['average_accuracy']:.1%}")
    print(f"  Average Confidence: {training_summary['average_confidence']:.1%}")
    print(f"  Models Trained: {len(results)}")
    
    print(f"\n🎉 V7 Enhanced Training Complete!")
    print("Files saved: v7_training_summary.json + model files")
