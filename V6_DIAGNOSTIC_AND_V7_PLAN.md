# V6 Model Diagnostic & V7 Training Plan

## Executive Summary

**Status**: V6 and V7 models **both completely broken** due to lack of feature normalization.
**Solution**: Retrain V7 on **AWS SageMaker** with guaranteed StandardScaler integration.
**Cost**: ~$12-14 for complete training (6-9 hours)
**Update (2025-11-16)**: V7 training attempted but failed - scaler not saved in checkpoint

---

## V6 Diagnostic Results

### Critical Issues Found (ALL 3 MODELS)

✅ **Diagnostic Complete**: `/home/numan/crpbot/reports/v6_model_diagnostic.json`

| Issue | BTC-USD | ETH-USD | SOL-USD |
|-------|---------|---------|---------|
| **Overconfidence >99%** | 100% | 100% | 99% |
| **Class Bias (DOWN)** | 100% | 100% | 97% |
| **Average DOWN Logit** | 33,503 | 1,398 | 98.6 |
| **Average UP Logit** | -9,475 | -981 | -23.3 |
| **Logit Range** | 15,684 to 52,011 | 780 to 2,167 | 37 to 215 |

### Root Causes

1. ❌ **No Feature Normalization**
   - Raw BTC prices (79,568) fed directly to network
   - Raw ETH prices (2,977) fed directly to network
   - Causes extreme activations in first layer

2. ❌ **No Output Calibration**
   - Final layer produces logits in 10,000+ range (should be ±10)
   - No temperature scaling during training
   - No label smoothing

3. ❌ **No Regularization**
   - No dropout layers
   - Model overfits to training data patterns
   - 100% confidence on every prediction

4. ❌ **Architecture Issues**
   - 4-layer FNN without normalization between layers
   - No BatchNorm or LayerNorm
   - ReLU activations can amplify extreme values

### Example: BTC-USD Model Failure

```
Input features (unnormalized):
- close: 79,568.14
- open: 79,568.08
- high: 79,596.19
- sma_7: 79,567.92

First layer (fc1) weight range: ±0.16
Activation after fc1: 79,568 × 0.16 = 12,730 (!!)

This cascades through layers:
fc1 → fc2 → fc3 → fc4
Result: Logits of 40,000+ → Softmax → 100% DOWN probability
```

---

## V7 Diagnostic Results (2025-11-16 UPDATE)

### ❌ V7 Training FAILED - Same Issues as V6

**Diagnostic Run**: `/tmp/v7_diagnostic.log`

| Issue | BTC-USD | ETH-USD | SOL-USD | Target |
|-------|---------|---------|---------|--------|
| **Logit Range** | ±158,041 | ±4,759 | ±500 | ≤20 |
| **Scaler Present** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Overconfidence >99%** | (Test incomplete) | (Test incomplete) | (Test incomplete) | <10% |

### V7 Failure Analysis

**What Worked**:
- ✅ Dropout layers (0.3)
- ✅ Batch normalization
- ✅ Temperature scaling parameter (2.5)
- ✅ Model architecture improvements

**What Failed**:
- ❌ **StandardScaler not fitted**
- ❌ **Scaler not saved in checkpoint**
- ❌ **Raw features (79,568 for BTC) still fed to network**
- ❌ **Logits WORSE than V6** (±158,000 vs ±40,000)

**Conclusion**: V7 training partially implemented architecture improvements but **completely missed** the critical normalization step. Models are unusable.

### Next Step: AWS SageMaker

Move training to **AWS SageMaker** with mandatory scaler verification:
- See: `/home/numan/crpbot/docs/SAGEMAKER_TRAINING_SETUP.md`
- See: `/home/numan/crpbot/V7_SAGEMAKER_MIGRATION_PLAN.md`

---

## V7 Training Plan (UPDATED FOR SAGEMAKER)

### Architecture Improvements

```python
class V7EnhancedFNN(nn.Module):
    """V7 Enhanced FNN with proper normalization and calibration."""

    def __init__(self, input_size=72):
        super().__init__()

        # Input normalization (fitted on training data)
        self.scaler = StandardScaler()  # ← NEW

        # Architecture with dropout
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # ← NEW
        self.dropout1 = nn.Dropout(0.3)  # ← NEW

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  # ← NEW
        self.dropout2 = nn.Dropout(0.3)  # ← NEW

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)  # ← NEW
        self.dropout3 = nn.Dropout(0.3)  # ← NEW

        self.fc4 = nn.Linear(64, 3)

        # Temperature for calibration
        self.temperature = nn.Parameter(torch.tensor(2.5))  # ← NEW

        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, -1, :]

        # Normalize input
        x = torch.FloatTensor(self.scaler.transform(x.cpu().numpy())).to(x.device)  # ← NEW

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        logits = self.fc4(x)

        # Temperature scaling for calibration
        return logits / self.temperature  # ← NEW
```

### Training Configuration

| Parameter | V6 Enhanced | V7 Enhanced |
|-----------|-------------|-------------|
| **Feature Normalization** | ❌ None | ✅ StandardScaler |
| **Dropout** | ❌ None | ✅ 0.3 |
| **Batch Normalization** | ❌ None | ✅ After each layer |
| **Label Smoothing** | ❌ 0.0 | ✅ 0.05 |
| **Temperature Scaling** | ❌ None | ✅ Learnable (init=2.5) |
| **Loss Function** | CrossEntropy | Focal Loss (α=0.25, γ=2.0) |
| **Learning Rate** | 0.001 | 0.001 with cosine decay |
| **Batch Size** | 32 | 64 (GPU) |
| **Epochs** | 15 | 30 |
| **Early Stopping** | patience=5 | patience=5 |
| **Device** | CPU | CUDA (A10G GPU) |

### Expected V7 Results

After proper training, models should produce:

| Metric | Target Range |
|--------|--------------|
| **Logit Range** | ±10 (not ±40,000) |
| **Confidence >99%** | <10% (not 100%) |
| **Confidence 50-90%** | 70-80% |
| **DOWN Predictions** | 40-60% (not 100%) |
| **UP Predictions** | 40-60% (not 0%) |
| **Calibration Error** | <5% |
| **Test Accuracy** | 68-72% |

---

## AWS Training Setup

### Instance: g5.xlarge

- **GPU**: NVIDIA A10G (24GB VRAM)
- **Cost**: $1.006/hour
- **Training Time**: 6-9 hours for 3 models
- **Total Cost**: ~$12

### Training Steps for Amazon Q

1. **Launch Instance** - Deploy g5.xlarge in us-east-1
2. **Setup Environment** - CUDA 11.8, PyTorch 2.1, UV package manager
3. **Clone Repository** - Get latest code from GitHub
4. **Prepare Data** - Download 2 years of 1m candles, engineer features with normalization
5. **Train V7 Models** - Run training with proper hyperparameters
6. **Validate** - Run diagnostic to verify calibration
7. **Deploy** - Upload models to S3 or download to local
8. **Terminate** - Shut down instance when complete

**Full Instructions**: `/home/numan/crpbot/docs/AWS_GPU_TRAINING_SETUP.md`

---

## Signal Format Implementation

✅ **COMPLETED**: New signal format with entry zones, order types, SL/TP
✅ **DEPLOYED**: Running on cloud server
⚠️ **BLOCKED**: No signals generated due to V6 model issues

**Once V7 models are trained and deployed**, the signal format will work correctly with realistic confidence levels (60-85% instead of fake 100%).

---

## Next Steps

### Immediate Actions

1. ✅ **Diagnostic Complete** - Issues identified
2. ✅ **Training Plan Created** - V7 architecture defined
3. ✅ **AWS Setup Guide Ready** - Full documentation available
4. ⏸️ **Waiting on Amazon Q** - To execute AWS GPU training

### For Amazon Q

**Please execute the V7 training plan on AWS g5.xlarge:**

1. Read: `/home/numan/crpbot/docs/AWS_GPU_TRAINING_SETUP.md`
2. Launch g5.xlarge instance with Deep Learning AMI
3. Set up PyTorch + CUDA environment
4. Clone repository and prepare data
5. Train V7 Enhanced models (BTC, ETH, SOL)
6. Run diagnostic to verify calibration
7. Download trained models + scalers
8. Terminate instance

**Expected Deliverables:**
- `lstm_BTC-USD_v7_enhanced.pt` with realistic confidence (60-85%)
- `lstm_ETH-USD_v7_enhanced.pt` with realistic confidence (60-85%)
- `lstm_SOL-USD_v7_enhanced.pt` with realistic confidence (60-85%)
- `scaler_BTC-USD.pkl` (StandardScaler for normalization)
- `scaler_ETH-USD.pkl`
- `scaler_SOL-USD.pkl`
- Diagnostic report showing <10% predictions with >99% confidence

---

## Files Created

- ✅ `/home/numan/crpbot/scripts/diagnose_v6_model.py` - Diagnostic script
- ✅ `/home/numan/crpbot/reports/v6_model_diagnostic.json` - Diagnostic results
- ✅ `/home/numan/crpbot/docs/AWS_GPU_TRAINING_SETUP.md` - AWS setup guide
- ✅ `/home/numan/crpbot/V6_DIAGNOSTIC_AND_V7_PLAN.md` - This document
- ✅ `/home/numan/crpbot/apps/runtime/signal_formatter.py` - New signal format (ready to use)
- ✅ `/home/numan/crpbot/apps/runtime/ensemble.py` - Fixed confidence calculation

---

## Summary

**V6 Models**: Completely broken - 100% overconfident, always predict DOWN
**Root Cause**: No feature normalization (raw prices 79,568 → extreme activations)
**Solution**: V7 training on GPU with StandardScaler + BatchNorm + Dropout + Temperature
**Cost**: ~$12 for complete retraining
**ETA**: 10 hours from Amazon Q start

**All infrastructure is ready. Waiting for Amazon Q to execute AWS GPU training.**
