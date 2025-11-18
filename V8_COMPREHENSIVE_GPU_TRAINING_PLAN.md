# V8 Comprehensive GPU Training Plan - Complete V6 Fix

**Date**: 2025-11-16 15:58 EST  
**Status**: READY TO EXECUTE  
**Target**: Fix all V6 critical issues with proper GPU training  
**Cost**: ~$15-20 for complete training  
**Timeline**: 4-6 hours total  

## 🚨 Critical Issues Analysis

### V6 Model Failures (100% Broken)
Based on diagnostic analysis, ALL V6 models suffer from:

1. **Extreme Overconfidence**: 99-100% predictions >99% confidence
2. **Severe Class Bias**: 97-100% "DOWN" predictions only  
3. **Logit Explosion**: Values 16,000-52,000 (should be ±10)
4. **No Feature Normalization**: Raw BTC prices (79,568) fed directly
5. **Architecture Issues**: No dropout, no batch norm, no regularization

### V7 Partial Fix Attempt
- ✅ Added dropout, batch norm, temperature scaling
- ❌ **FAILED**: Still no feature normalization
- ❌ **FAILED**: Scaler not saved in checkpoint
- ❌ **WORSE**: Logits now ±158,000 (worse than V6)

## 🎯 V8 Complete Solution

### 1. Feature Engineering & Normalization
```python
class V8FeatureProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def create_features(self, df):
        """Create 72 engineered features with proper scaling"""
        features = df.copy()
        
        # Price features (normalized)
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Technical indicators (all normalized)
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Volatility indicators
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
        
        # RSI (already 0-100 scale)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Bollinger Bands position (0-1 scale)
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume features
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        return features
    
    def fit_transform(self, df):
        """Fit scaler and transform features"""
        features_df = self.create_features(df)
        features_df = features_df.dropna()
        
        # Select feature columns (exclude OHLCV and target)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        self.feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Features: {len(self.feature_columns)}")
        print(f"✅ Scaling: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")
        
        return X_scaled, features_df
    
    def transform(self, df):
        """Transform new data using fitted scaler"""
        features_df = self.create_features(df)
        features_df = features_df.dropna()
        
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, features_df
```

### 2. V8 Enhanced Architecture
```python
class V8TradingNet(nn.Module):
    """V8 Enhanced Neural Network with all fixes"""
    
    def __init__(self, input_size=72, hidden_size=256, num_classes=3):
        super(V8TradingNet, self).__init__()
        
        # Architecture with proper regularization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.tensor(2.5))
        
        # Layer normalization for single-sample inference
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 4)
        
        self.relu = nn.ReLU()
        
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
    """Focal Loss to handle class imbalance and reduce overconfidence"""
    
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
```

### 3. Training Configuration
```python
def train_v8_model(symbol, epochs=100):
    """Complete V8 training with all fixes"""
    
    print(f"\n🚀 Training V8 {symbol} Model")
    
    # 1. Load and prepare data
    df = load_symbol_data(symbol)  # 2 years of 1m data
    processor = V8FeatureProcessor()
    
    # 2. Feature engineering with normalization
    X_scaled, features_df = processor.fit_transform(df)
    
    # 3. Create targets with balanced thresholds
    features_df['target'] = create_balanced_targets(features_df['close'])
    y = features_df['target'].values
    
    print(f"✅ Data: {len(X_scaled)} samples, {X_scaled.shape[1]} features")
    print(f"✅ Target distribution: {np.bincount(y)}")
    
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
    
    # 6. Data loader with proper batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 7. Model with all fixes
    model = V8TradingNet(
        input_size=X_train.shape[1],
        hidden_size=512,  # Larger network
        num_classes=3
    ).to(device)
    
    # 8. Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
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
                
                # Class distribution
                pred_counts = torch.bincount(val_predicted, minlength=3)
                pred_dist = pred_counts.float() / pred_counts.sum()
                
                # Early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_state = {
                        'model_state_dict': model.state_dict(),
                        'processor': processor,  # Include feature processor
                        'accuracy': val_accuracy,
                        'confidence_stats': {
                            'mean': avg_confidence,
                            'overconfident_pct': overconfident_pct
                        },
                        'prediction_distribution': pred_dist.cpu().numpy(),
                        'temperature': model.temperature.item(),
                        'epoch': epoch + 1
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1:3d}: Loss={epoch_loss/len(train_loader):.4f}, "
                      f"Acc={val_accuracy:.3f}, Conf={avg_confidence:.3f}, "
                      f"Over99%={overconfident_pct:.1%}, "
                      f"Dist=[{pred_dist[0]:.2f},{pred_dist[1]:.2f},{pred_dist[2]:.2f}], "
                      f"Temp={model.temperature.item():.2f}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # 10. Final evaluation and save
    model.load_state_dict(best_model_state['model_state_dict'])
    
    # Save complete model with processor
    model_filename = f'models/v8_enhanced/lstm_{symbol}_v8_enhanced.pt'
    torch.save(best_model_state, model_filename)
    
    print(f"✅ Saved {model_filename}")
    print(f"✅ Best accuracy: {best_val_accuracy:.3f}")
    print(f"✅ Overconfident predictions: {best_model_state['confidence_stats']['overconfident_pct']:.1%}")
    
    return best_model_state

def create_balanced_targets(close_prices, up_threshold=0.012, down_threshold=0.012):
    """Create balanced targets with symmetric thresholds"""
    future_returns = (close_prices.shift(-1) / close_prices) - 1
    
    targets = np.where(
        future_returns > up_threshold, 2,    # BUY (strong up move)
        np.where(future_returns < -down_threshold, 0, 1)  # SELL (strong down), HOLD
    )
    
    return targets
```

## 📋 Execution Plan

### Phase 1: AWS Setup (15 minutes)
1. **Launch GPU Instance**
   ```bash
   # Launch g5.xlarge with Deep Learning AMI
   aws ec2 run-instances \
     --image-id ami-0c02fb55956c7d316 \
     --instance-type g5.xlarge \
     --key-name crpbot-training \
     --security-group-ids sg-xxxxxxxxx \
     --subnet-id subnet-xxxxxxxxx \
     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=V8-Training}]'
   ```

2. **Setup Environment**
   ```bash
   ssh -i ~/.ssh/crpbot-training.pem ubuntu@<instance-ip>
   
   # Install dependencies
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install pandas numpy scikit-learn matplotlib seaborn
   
   # Verify GPU
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   nvidia-smi
   ```

3. **Upload Training Code**
   ```bash
   # Upload V8 training script
   scp -i ~/.ssh/crpbot-training.pem v8_enhanced_training.py ubuntu@<instance-ip>:~/
   scp -i ~/.ssh/crpbot-training.pem -r data/ ubuntu@<instance-ip>:~/
   ```

### Phase 2: Data Preparation (30 minutes)
1. **Download Fresh Data**
   ```bash
   # Get 2 years of 1-minute data for each symbol
   python fetch_training_data.py --symbols BTC-USD,ETH-USD,SOL-USD --timeframe 1m --period 2y
   ```

2. **Feature Engineering**
   ```bash
   # Create 72 engineered features with proper normalization
   python prepare_v8_features.py
   ```

3. **Data Quality Check**
   ```bash
   # Verify no NaN, proper scaling, balanced targets
   python validate_training_data.py
   ```

### Phase 3: Model Training (3-4 hours)
1. **Train BTC Model** (60-80 minutes)
   ```bash
   python v8_enhanced_training.py --symbol BTC-USD --epochs 100 --batch-size 256
   ```

2. **Train ETH Model** (60-80 minutes)
   ```bash
   python v8_enhanced_training.py --symbol ETH-USD --epochs 100 --batch-size 256
   ```

3. **Train SOL Model** (60-80 minutes)
   ```bash
   python v8_enhanced_training.py --symbol SOL-USD --epochs 100 --batch-size 256
   ```

### Phase 4: Validation (30 minutes)
1. **Run Comprehensive Diagnostic**
   ```bash
   python diagnose_v8_models.py --all-models
   ```

2. **Quality Gates Check**
   - ✅ Logit range: ±10 (not ±40,000)
   - ✅ Overconfident predictions: <10% (not 100%)
   - ✅ Class balance: Each class 25-40% (not 100% DOWN)
   - ✅ Confidence distribution: Realistic 60-85% range
   - ✅ No NaN/Inf values in predictions

3. **Backtesting**
   ```bash
   python backtest_v8_models.py --period 30d
   ```

### Phase 5: Deployment (15 minutes)
1. **Download Models**
   ```bash
   # Download trained models with processors
   scp -i ~/.ssh/crpbot-training.pem ubuntu@<instance-ip>:~/models/v8_enhanced/*.pt ./models/v8_enhanced/
   ```

2. **Update Production**
   ```bash
   # Update runtime to use V8 models
   sed -i 's/v6_enhanced/v8_enhanced/g' apps/runtime/main.py
   
   # Deploy to production server
   rsync -av models/v8_enhanced/ production-server:~/crpbot/models/v8_enhanced/
   ```

3. **Restart Services**
   ```bash
   # Restart runtime with new models
   systemctl restart trading-ai
   ```

## 🎯 Success Criteria

### Model Quality Gates
| Metric | Target | V6 Actual | V8 Expected |
|--------|--------|-----------|-------------|
| **Overconfident (>99%)** | <10% | 100% | <5% |
| **DOWN Predictions** | 25-40% | 100% | 30-35% |
| **UP Predictions** | 25-40% | 0% | 30-35% |
| **HOLD Predictions** | 25-40% | 0% | 30-35% |
| **Logit Range** | ±10 | ±40,000 | ±8 |
| **Confidence Mean** | 60-80% | 99.9% | 70-75% |
| **Test Accuracy** | >68% | N/A | 70-75% |

### Production Ready
- ✅ All 3 models pass quality gates
- ✅ Feature processors saved with models
- ✅ Realistic confidence distributions
- ✅ Balanced class predictions
- ✅ No single-sample inference crashes
- ✅ Proper temperature calibration

## 💰 Cost Estimate

| Component | Time | Cost |
|-----------|------|------|
| **g5.xlarge Instance** | 6 hours | $6.04 |
| **Data Transfer** | - | $0.50 |
| **Storage** | 100GB | $1.00 |
| **Total** | - | **~$7.54** |

## 🚨 Risk Mitigation

### Rollback Plan
1. **If V8 training fails**: Use V5 models as fallback
2. **If quality gates fail**: Debug on smaller dataset
3. **If GPU issues**: Fall back to CPU training (slower)
4. **If cost overruns**: Set billing alerts at $10

### Monitoring
- AWS CloudWatch for instance metrics
- Training logs for convergence
- Validation metrics for quality
- Cost tracking in AWS console

## 📁 Deliverables

### Models
- `models/v8_enhanced/lstm_BTC-USD_v8_enhanced.pt`
- `models/v8_enhanced/lstm_ETH-USD_v8_enhanced.pt`
- `models/v8_enhanced/lstm_SOL-USD_v8_enhanced.pt`

### Documentation
- `reports/v8_training_report.json` - Training metrics
- `reports/v8_diagnostic_report.json` - Quality validation
- `reports/v8_backtest_results.json` - Performance analysis

### Code
- `v8_enhanced_training.py` - Complete training script
- `v8_feature_processor.py` - Feature engineering
- `v8_model_architecture.py` - Neural network definition
- `diagnose_v8_models.py` - Validation script

---

## 🚀 Next Steps

1. **Create V8 training scripts** with all fixes implemented
2. **Launch AWS g5.xlarge instance** for GPU training
3. **Execute training plan** following phases 1-5
4. **Validate models** meet all quality gates
5. **Deploy to production** and monitor performance

**This plan addresses ALL V6 critical issues and provides a complete solution for reliable trading signal generation.**
