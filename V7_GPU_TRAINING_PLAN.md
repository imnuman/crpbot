# V7 GPU Training Plan - Fix V6 Critical Issues

**Date**: 2025-11-16  
**GPU Instance**: g5.xlarge (i-0f09e5bb081e72ae1) - RUNNING  
**Public IP**: 35.153.176.224  
**Estimated Time**: 45-90 minutes total  
**Cost**: ~$1.50  

## 🚨 Critical Issues to Fix

### V6 Enhanced Models Failed Due To:
1. **Severe Overconfidence**: 100% predictions >99% confidence
2. **Extreme Class Bias**: 98-100% "Down" predictions only
3. **Logit Explosion**: Values 16,946-52,717 (should be -10 to +10)
4. **Feature Scaling**: Raw price data not normalized
5. **Batch Normalization Bug**: Single sample batches causing crashes

## 🎯 V7 Training Fixes

### 1. Feature Normalization
```python
# Fix: Proper StandardScaler per feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Save scaler for inference
```

### 2. Model Architecture Changes
```python
# Fix: Add dropout + label smoothing
model = nn.Sequential(
    nn.Linear(72, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),  # Prevent overconfidence
    # ... more layers
    nn.Linear(64, 3)
)

# Fix: Label smoothing loss
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### 3. Training Parameters
```python
# Fix: Larger batch sizes for stable batch norm
batch_size = 64  # Was causing single-sample crashes
learning_rate = 0.001  # Conservative
epochs = 50  # Sufficient for convergence
```

### 4. Class Balancing
```python
# Fix: Weighted loss for balanced predictions
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

## 📋 Execution Plan

### Phase 1: Setup (5 minutes)
1. **Connect to GPU Instance**
   ```bash
   ssh -i ~/.ssh/crpbot-training.pem ubuntu@35.153.176.224
   ```

2. **Upload Fixed Training Script**
   ```bash
   scp -i ~/.ssh/crpbot-training.pem v7_fixed_training.py ubuntu@35.153.176.224:~/
   scp -i ~/.ssh/crpbot-training.pem -r data/ ubuntu@35.153.176.224:~/
   ```

3. **Verify GPU Availability**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Phase 2: Training (45-90 minutes)
1. **Train BTC Model** (15-30 min)
   ```bash
   python v7_fixed_training.py --coin BTC --epochs 50 --batch-size 64
   ```

2. **Train ETH Model** (15-30 min)
   ```bash
   python v7_fixed_training.py --coin ETH --epochs 50 --batch-size 64
   ```

3. **Train SOL Model** (15-30 min)
   ```bash
   python v7_fixed_training.py --coin SOL --epochs 50 --batch-size 64
   ```

### Phase 3: Validation (10 minutes)
1. **Run Diagnostic on Each Model**
   ```bash
   python diagnose_v7_models.py
   ```

2. **Quality Gates Check**
   - Confidence distribution: <80% predictions >90% confidence
   - Class balance: Each class 20-40% of predictions
   - Logit range: -10 to +10 (reasonable values)
   - No NaN/Inf values in weights

### Phase 4: Download & Deploy (5 minutes)
1. **Download Trained Models**
   ```bash
   scp -i ~/.ssh/crpbot-training.pem ubuntu@35.153.176.224:~/models/v7_*.pt ./models/
   ```

2. **Update Production Config**
   ```bash
   # Point runtime to V7 models
   sed -i 's/v6_enhanced/v7_fixed/g' apps/runtime/main.py
   ```

## 🔧 V7 Training Script Requirements

### Critical Fixes Needed:
```python
class V7EnhancedLSTM(nn.Module):
    def __init__(self, input_size=72, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, 3)
        
    def forward(self, x):
        # Fix: Handle single samples in eval mode
        if self.training or x.size(0) > 1:
            x = self.bn1(F.relu(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        if self.training or x.size(0) > 1:
            x = self.bn2(F.relu(self.fc2(x)))
        else:
            x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        if self.training or x.size(0) > 1:
            x = self.bn3(F.relu(self.fc3(x)))
        else:
            x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        return self.fc4(x)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Prevent overconfidence
        log_prob = F.log_softmax(pred, dim=-1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
```

### Training Loop Fixes:
```python
def train_v7_model(symbol, epochs=50):
    # Fix: Proper feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fix: Balanced class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    
    # Fix: Label smoothing + weighted loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Fix: Larger batch size for stable batch norm
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'models/v7_{symbol}_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    return model, scaler
```

## ✅ Success Criteria

### Model Quality Gates:
- **Confidence**: <80% predictions >90% confidence
- **Balance**: Each class 20-40% of predictions  
- **Stability**: Logits in range [-10, +10]
- **Accuracy**: >65% on validation set
- **No Crashes**: Handle single-sample inference

### Deployment Ready:
- All 3 models pass quality gates
- Scalers saved for inference
- Diagnostic report shows fixes worked
- Models uploaded to production

## 🚨 Rollback Plan

If V7 training fails:
1. Keep V6 models offline (they're broken)
2. Use V5 models as fallback
3. Debug V7 issues on local CPU
4. Retry GPU training with fixes

## 💰 Cost Control

- **Training Time**: 45-90 minutes max
- **Instance Cost**: ~$1.50 total
- **Auto-shutdown**: Stop instance after training
- **Monitor**: Check costs in AWS console

---

**Next Action**: Create `v7_fixed_training.py` with all fixes above, then execute Phase 1.
