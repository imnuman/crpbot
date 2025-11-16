# 🔄 Retraining Plan - 31-Feature Original Design

**Date**: 2025-11-14
**Status**: APPROVED by QC Claude
**Reason**: 50-feature models completely failed (50% accuracy)

---

## 📊 What Went Wrong

**Evaluation Results (50-feature models):**
```
BTC-USD: 50.26% accuracy, 47.7% calibration - FAILED
ETH-USD: 49.56% accuracy, 45.3% calibration - FAILED
SOL-USD: 49.52% accuracy, 50.4% calibration - FAILED
```

**Root Cause:**
- Models never learned (50% from start of training)
- Multi-timeframe features (50 total) may have introduced complexity/bias
- Original 31-feature design was cleaner and simpler

---

## ✅ New Approach: Back to Basics

### Files to Use

**Original Feature Files (33 actual features):**
```bash
data/features/features_BTC-USD_1m_2025-11-13.parquet  (33 features)
data/features/features_ETH-USD_1m_2025-11-13.parquet  (33 features)
data/features/features_SOL-USD_1m_2025-11-13.parquet  (33 features)
```

**DO NOT USE:**
```bash
# These failed:
features_*_50feat.parquet  ❌
```

---

## 🔧 Step-by-Step Retraining

### Step 1: Verify Original Features (2 min)

```bash
cd /home/numan/crpbot

# Check original feature files exist
ls -lh data/features/features_BTC-USD_1m_2025-11-13.parquet
ls -lh data/features/features_ETH-USD_1m_2025-11-13.parquet
ls -lh data/features/features_SOL-USD_1m_2025-11-13.parquet

# Verify they have 33 features
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/features/features_BTC-USD_1m_2025-11-13.parquet')
print(f'Columns: {len(df.columns)}')
print(f'Features (excluding metadata): {len([c for c in df.columns if c not in [\"timestamp\", \"target\", \"symbol\", \"interval\"]])}')
print(f'Rows: {len(df):,}')
"
```

### Step 2: Update Training Configuration (5 min)

**File**: `apps/trainer/features.py`

Update `FEATURE_COLUMNS` to match the original 33 features:

```python
# Original 33 features (not 50)
FEATURE_COLUMNS = [
    # OHLCV (5)
    'open', 'high', 'low', 'close', 'volume',

    # Session features (5)
    'session_tokyo', 'session_london', 'session_ny',
    'day_of_week', 'is_weekend',

    # Spread features (4)
    'spread', 'spread_pct', 'atr', 'spread_atr_ratio',

    # Volume features (3)
    'volume_ma', 'volume_ratio', 'volume_trend',

    # Moving averages (8)
    'sma_7', 'sma_14', 'sma_21', 'sma_50',
    'price_sma7_ratio', 'price_sma14_ratio',
    'price_sma21_ratio', 'price_sma50_ratio',

    # Technical indicators (8)
    'rsi', 'macd', 'macd_signal', 'macd_diff',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width'
]

# Total: 33 features
```

**Verify count:**
```bash
uv run python -c "
from apps.trainer.features import FEATURE_COLUMNS
print(f'Total features defined: {len(FEATURE_COLUMNS)}')
print(f'Features: {FEATURE_COLUMNS}')
"
```

### Step 3: Update Model Architecture (5 min)

**File**: `apps/trainer/models/lstm.py`

Change input_size to 33:

```python
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=33,  # Changed from 50 to 33
        hidden_size=128,
        num_layers=3,
        dropout=0.35
    ):
        # ... rest of the code
```

### Step 4: Prepare Files for Colab (10 min)

```bash
cd /home/numan/crpbot

# Create new upload directory
mkdir -p /tmp/colab_upload_31feat/models
mkdir -p /tmp/colab_upload_31feat/features

# Copy original feature files
cp data/features/features_BTC-USD_1m_2025-11-13.parquet \
   /tmp/colab_upload_31feat/features/features_BTC-USD_1m_2025-11-13_33feat.parquet

cp data/features/features_ETH-USD_1m_2025-11-13.parquet \
   /tmp/colab_upload_31feat/features/features_ETH-USD_1m_2025-11-13_33feat.parquet

cp data/features/features_SOL-USD_1m_2025-11-13.parquet \
   /tmp/colab_upload_31feat/features/features_SOL-USD_1m_2025-11-13_33feat.parquet

# Verify
ls -lh /tmp/colab_upload_31feat/features/
```

### Step 5: Create Colab Training Notebook (Builder Claude)

**Request to Builder Claude:**
```
Create colab_train_31feat_models.ipynb that:
1. Uses 33 features (not 50)
2. Same LSTM architecture (128 hidden, 3 layers)
3. Trains for 20 epochs (not 15 - give it more time)
4. Saves models to /content/models/
5. Includes validation accuracy monitoring
```

### Step 6: Upload to Google Drive (15 min)

```bash
# From local machine
scp -r root@178.156.136.185:/tmp/colab_upload_31feat ~/Downloads/

# Then upload to Google Drive:
# My Drive/crpbot/training_31feat/
#   ├── features/ (3 files, ~600 MB)
#   └── notebook: colab_train_31feat_models.ipynb
```

### Step 7: Train on Colab GPU (60-90 min)

```
1. Open colab_train_31feat_models.ipynb in Colab
2. Runtime → Change runtime type → GPU (A100 if available)
3. Mount Drive and copy feature files
4. Run all cells
5. Monitor validation accuracy - should be >60% by epoch 10
6. Download trained models
```

### Step 8: Evaluate New Models (10 min)

Use the same evaluation script but update paths:
- Load models from new training
- Use 33-feature files for evaluation
- Check if accuracy improves beyond 50%

---

## 🎯 Success Criteria

**Training:**
- ✅ Validation accuracy >60% by epoch 10
- ✅ Training loss decreasing steadily
- ✅ No signs of overfitting (val_loss < train_loss)

**Evaluation:**
- ✅ Test accuracy ≥68% (promotion gate)
- ✅ Calibration error ≤5%
- ✅ Precision/Recall both >0 (model not collapsed)

**If This Also Fails:**
Then we need to revisit:
1. Target definition (maybe 15-min horizon is too short)
2. Prediction task (try regression instead of classification)
3. Architecture (try simpler models)
4. Data quality (check for leakage or issues)

---

## ⏱️ Timeline

```
Now:        Prepare files [20 min]
+20 min:    Upload to Drive [15 min]
+35 min:    Train on Colab A100 [60 min]
+95 min:    Evaluate models [10 min]
+105 min:   Know if 31-feature approach works

Total: ~2 hours to answer
```

---

## 📋 Checklist

**Preparation:**
- [ ] Verify original feature files exist (33 features each)
- [ ] Update FEATURE_COLUMNS in features.py to 33 features
- [ ] Update LSTMModel input_size to 33
- [ ] Create /tmp/colab_upload_31feat/ directory
- [ ] Copy feature files to upload directory

**Builder Claude Tasks:**
- [ ] Create colab_train_31feat_models.ipynb
- [ ] Update model architecture to use 33 features
- [ ] Set training to 20 epochs
- [ ] Add progress monitoring

**Execution:**
- [ ] Upload feature files to Google Drive
- [ ] Upload notebook to Colab
- [ ] Enable A100 GPU
- [ ] Run training
- [ ] Monitor validation accuracy
- [ ] Download trained models
- [ ] Evaluate on test set
- [ ] Check promotion gates

**Decision Point:**
- [ ] If models pass (≥68% accuracy): Deploy to production
- [ ] If models improve but don't pass (55-67%): Tune hyperparameters
- [ ] If models still at 50%: Rethink problem formulation

---

## 🚨 Fallback Plan

**If 31-feature models also fail at 50%:**

The problem is NOT the features, it's one of:

1. **Target Definition Issue**
   - 15-minute binary up/down is too noisy
   - Try: 30 or 60-minute horizon
   - Try: >0.5% change threshold instead of any change

2. **Data Leakage Prevention Too Aggressive**
   - Walk-forward splits may be cutting useful patterns
   - Try: Different split ratios (80/10/10)

3. **Architecture Mismatch**
   - LSTM may not be right for this task
   - Try: Simpler logistic regression baseline first
   - Try: Transformer architecture

4. **Market Reality**
   - Crypto at 1-minute resolution may be unpredictable
   - Try: 5-minute or 15-minute base timeframe
   - Try: Different symbols (less volatile)

---

**File**: `RETRAINING_PLAN_31FEAT.md`
**Status**: Ready for execution
**Next**: Builder Claude creates training notebook
