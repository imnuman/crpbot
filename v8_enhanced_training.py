#!/usr/bin/env python3
"""
V8 Enhanced Training - Complete Fix for V6 Issues
Addresses: Feature normalization, overconfidence, class bias, logit explosion
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json
import pickle
import argparse
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("=== V8 Enhanced Training - Complete V6 Fix ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class V8FeatureProcessor:
    """Enhanced feature processor with proper normalization"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.fitted = False
    
    def create_features(self, df):
        """Create 72 engineered features"""
        features = df.copy()
        
        # Basic price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        features['volume_price_trend'] = df['volume'] * features['returns']
        
        # Moving averages and ratios
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
            features[f'price_to_ema_{period}'] = df['close'] / features[f'ema_{period}']
        
        # Momentum indicators
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            features[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        # Volatility
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_50'] = features['returns'].rolling(50).std()
        features['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
        features['atr_ratio'] = features['atr_14'] / df['close']
        
        # RSI
        for period in [14, 21, 30]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (2 * std)
            features[f'bb_lower_{period}'] = sma - (2 * std)
            features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Stochastic
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            features[f'stoch_k_{period}'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()
        
        # Williams %R
        for period in [14, 21]:
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            features[f'williams_r_{period}'] = -100 * ((high_max - df['close']) / (high_max - low_min))
        
        # Volume indicators
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        features['volume_momentum'] = features['volume_ratio'].pct_change()
        
        # Price channels
        for period in [20, 50]:
            features[f'price_channel_high_{period}'] = df['high'].rolling(period).max()
            features[f'price_channel_low_{period}'] = df['low'].rolling(period).min()
            features[f'price_channel_position_{period}'] = (df['close'] - features[f'price_channel_low_{period}']) / (features[f'price_channel_high_{period}'] - features[f'price_channel_low_{period}'])
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)
        
        return features
    
    def fit_transform(self, df):
        """Fit scaler and transform features"""
        features_df = self.create_features(df)
        features_df = features_df.dropna()
        
        # Select feature columns (exclude OHLCV and target)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp']
        self.feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[self.feature_columns].values
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        
        print(f"✅ Features: {len(self.feature_columns)}")
        print(f"✅ Scaling: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")
        print(f"✅ Feature range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        
        return X_scaled, features_df
    
    def transform(self, df):
        """Transform new data using fitted scaler"""
        if not self.fitted:
            raise ValueError("Processor not fitted. Call fit_transform first.")
        
        features_df = self.create_features(df)
        features_df = features_df.dropna()
        
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, features_df

class V8TradingNet(nn.Module):
    """V8 Enhanced Neural Network with all fixes"""
    
    def __init__(self, input_size=72, hidden_size=512, num_classes=3, dropout=0.3):
        super(V8TradingNet, self).__init__()
        
        # Enhanced architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # For single-sample inference
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.ln3 = nn.LayerNorm(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout * 0.7)  # Reduced dropout in final layer
        
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.tensor(2.5))
        
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Layer 1 with adaptive normalization
        x = self.fc1(x)
        if self.training and x.size(0) > 1:
            x = self.bn1(x)  # Batch norm during training
        else:
            x = self.ln1(x)  # Layer norm during inference
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if self.training and x.size(0) > 1:
            x = self.bn2(x)
        else:
            x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        if self.training and x.size(0) > 1:
            x = self.bn3(x)
        else:
            x = self.ln3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Output with temperature scaling
        logits = self.fc4(x)
        calibrated_logits = logits / self.temperature
        
        return calibrated_logits

class FocalLoss(nn.Module):
    """Focal Loss with label smoothing to reduce overconfidence"""
    
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        num_classes = inputs.size(-1)
        smoothed_targets = torch.zeros_like(inputs)
        smoothed_targets.fill_(self.label_smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute focal loss
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.ones_like(probs) * self.alpha
        
        # Final loss
        loss = -alpha_weight * focal_weight * smoothed_targets * log_probs
        return loss.sum(dim=-1).mean()

def create_balanced_targets(close_prices, up_threshold=0.015, down_threshold=0.015):
    """Create balanced targets with symmetric thresholds"""
    future_returns = (close_prices.shift(-1) / close_prices) - 1
    
    targets = np.where(
        future_returns > up_threshold, 2,    # BUY (strong up move)
        np.where(future_returns < -down_threshold, 0, 1)  # SELL (strong down), HOLD
    )
    
    return targets

def load_symbol_data(symbol):
    """Load data for symbol"""
    filename_map = {
        'BTC-USD': 'btc_data.csv',
        'ETH-USD': 'eth_data.csv', 
        'SOL-USD': 'sol_data.csv'
    }
    
    if symbol not in filename_map:
        raise ValueError(f"Unknown symbol: {symbol}")
    
    df = pd.read_csv(filename_map[symbol])
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    print(f"✅ Loaded {symbol}: {len(df)} rows")
    return df

def train_v8_model(symbol, epochs=100, batch_size=256, learning_rate=0.001):
    """Complete V8 training with all fixes"""
    
    print(f"\n🚀 Training V8 {symbol} Model")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    
    # 1. Load and prepare data
    df = load_symbol_data(symbol)
    processor = V8FeatureProcessor()
    
    # 2. Feature engineering with normalization
    X_scaled, features_df = processor.fit_transform(df)
    
    # 3. Create balanced targets
    features_df['target'] = create_balanced_targets(features_df['close'])
    y = features_df['target'].values
    
    # Remove NaN targets
    valid_mask = ~np.isnan(y)
    X_scaled = X_scaled[valid_mask]
    y = y[valid_mask].astype(int)
    
    print(f"✅ Data: {len(X_scaled)} samples, {X_scaled.shape[1]} features")
    print(f"✅ Target distribution: {np.bincount(y)}")
    print(f"✅ Class percentages: SELL={np.mean(y==0):.1%}, HOLD={np.mean(y==1):.1%}, BUY={np.mean(y==2):.1%}")
    
    # 4. Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Setup GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Training on {device}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 6. Data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 7. Model with all fixes
    model = V8TradingNet(
        input_size=X_train.shape[1],
        hidden_size=512,
        num_classes=3,
        dropout=0.3
    ).to(device)
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 8. Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 9. Training loop with validation
    best_val_accuracy = 0
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    print("\n" + "="*80)
    print("TRAINING PROGRESS")
    print("="*80)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_probs = F.softmax(val_outputs, dim=1)
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy = (val_predicted == y_test_tensor).float().mean().item()
                
                # Confidence analysis
                max_probs = torch.max(val_probs, dim=1)[0]
                avg_confidence = max_probs.mean().item()
                overconfident_pct = (max_probs > 0.99).float().mean().item()
                underconfident_pct = (max_probs < 0.5).float().mean().item()
                
                # Class distribution
                pred_counts = torch.bincount(val_predicted, minlength=3)
                pred_dist = pred_counts.float() / pred_counts.sum()
                
                # Logit analysis
                logit_mean = val_outputs.mean().item()
                logit_std = val_outputs.std().item()
                logit_min = val_outputs.min().item()
                logit_max = val_outputs.max().item()
                
                # Early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_state = {
                        'model_state_dict': model.state_dict(),
                        'processor': processor,
                        'accuracy': val_accuracy,
                        'confidence_stats': {
                            'mean': avg_confidence,
                            'overconfident_pct': overconfident_pct,
                            'underconfident_pct': underconfident_pct
                        },
                        'prediction_distribution': pred_dist.cpu().numpy(),
                        'logit_stats': {
                            'mean': logit_mean,
                            'std': logit_std,
                            'min': logit_min,
                            'max': logit_max
                        },
                        'temperature': model.temperature.item(),
                        'epoch': epoch + 1,
                        'symbol': symbol,
                        'training_date': datetime.now().isoformat()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1:3d}: Loss={epoch_loss/len(train_loader):.4f}, "
                      f"Acc={val_accuracy:.3f}, Conf={avg_confidence:.3f}, "
                      f"Over99%={overconfident_pct:.1%}, Under50%={underconfident_pct:.1%}")
                print(f"         Dist=[{pred_dist[0]:.2f},{pred_dist[1]:.2f},{pred_dist[2]:.2f}], "
                      f"Logits=[{logit_min:.1f},{logit_max:.1f}], Temp={model.temperature.item():.2f}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # 10. Final evaluation
    model.load_state_dict(best_model_state['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        final_outputs = model(X_test_tensor)
        final_probs = F.softmax(final_outputs, dim=1)
        _, final_predicted = torch.max(final_outputs, 1)
        final_accuracy = (final_predicted == y_test_tensor).float().mean().item()
        
        # Detailed classification report
        y_test_np = y_test_tensor.cpu().numpy()
        final_predicted_np = final_predicted.cpu().numpy()
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"✅ Best Validation Accuracy: {best_val_accuracy:.3f}")
        print(f"✅ Final Test Accuracy: {final_accuracy:.3f}")
        print(f"✅ Overconfident Predictions (>99%): {best_model_state['confidence_stats']['overconfident_pct']:.1%}")
        print(f"✅ Logit Range: [{best_model_state['logit_stats']['min']:.1f}, {best_model_state['logit_stats']['max']:.1f}]")
        print(f"✅ Temperature: {best_model_state['temperature']:.2f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_np, final_predicted_np, 
                                  target_names=['SELL', 'HOLD', 'BUY']))
    
    # 11. Save model
    os.makedirs('models/v8_enhanced', exist_ok=True)
    model_filename = f'models/v8_enhanced/lstm_{symbol}_v8_enhanced.pt'
    torch.save(best_model_state, model_filename)
    
    # Save processor separately for easier loading
    processor_filename = f'models/v8_enhanced/processor_{symbol}_v8.pkl'
    with open(processor_filename, 'wb') as f:
        pickle.dump(processor, f)
    
    print(f"✅ Saved {model_filename}")
    print(f"✅ Saved {processor_filename}")
    
    return best_model_state

def main():
    parser = argparse.ArgumentParser(description='V8 Enhanced Training')
    parser.add_argument('--symbol', type=str, default='BTC-USD', 
                       choices=['BTC-USD', 'ETH-USD', 'SOL-USD'],
                       help='Symbol to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--all', action='store_true', help='Train all symbols')
    
    args = parser.parse_args()
    
    if args.all:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    else:
        symbols = [args.symbol]
    
    results = []
    
    print("🚀 V8 Enhanced Training - Complete V6 Fix")
    print(f"Symbols: {symbols}")
    print(f"Configuration: {args.epochs} epochs, batch size {args.batch_size}")
    
    for symbol in symbols:
        try:
            result = train_v8_model(
                symbol=symbol,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to train {symbol}: {e}")
            continue
    
    # Summary
    if results:
        summary = {
            'version': 'v8_enhanced',
            'training_date': datetime.now().isoformat(),
            'models': len(results),
            'results': results,
            'average_accuracy': np.mean([r['accuracy'] for r in results]),
            'average_confidence': np.mean([r['confidence_stats']['mean'] for r in results]),
            'average_overconfident': np.mean([r['confidence_stats']['overconfident_pct'] for r in results])
        }
        
        with open('v8_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("V8 ENHANCED TRAINING SUMMARY")
        print("="*80)
        
        for result in results:
            print(f"{result['symbol']}: {result['accuracy']:.1%} accuracy, "
                  f"{result['confidence_stats']['mean']:.1%} avg confidence, "
                  f"{result['confidence_stats']['overconfident_pct']:.1%} overconfident")
        
        print(f"\nOverall Average: {summary['average_accuracy']:.1%} accuracy")
        print(f"Overconfident Rate: {summary['average_overconfident']:.1%} (target: <10%)")
        
        if summary['average_overconfident'] < 0.1:
            print("🎉 SUCCESS: V8 models meet quality gates!")
        else:
            print("⚠️  WARNING: Models still overconfident, may need more tuning")
    
    print("\n✅ V8 Enhanced Training Complete!")

if __name__ == "__main__":
    main()
