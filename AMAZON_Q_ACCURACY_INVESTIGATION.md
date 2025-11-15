# Amazon Q: V5 Training Accuracy Investigation

**Date**: 2025-11-15 18:30 EST
**Priority**: 🔴 HIGH
**Issue**: Accuracy discrepancy between two training runs
**Requestor**: Builder Claude (via user request)

---

## 🚨 THE PROBLEM

Two V5 training runs produced **different accuracy results**:

### First Training Run (Incomplete Save)
- **BTC-USD LSTM**: 74.0% accuracy ✅
- **ETH-USD LSTM**: 70.6% accuracy ✅
- **SOL-USD LSTM**: 72.1% accuracy ✅
- **Issue**: Models saved without `model_state_dict` (only metadata)
- **Status**: Models unusable for inference

### Second Training Run (FIXED - Complete Save)
- **BTC-USD LSTM**: 63.6% accuracy ⚠️ (-10.4 points)
- **ETH-USD LSTM**: 65.1% accuracy ⚠️ (-5.5 points)
- **SOL-USD LSTM**: 65.6% accuracy ⚠️ (-6.5 points)
- **Fix**: Models saved WITH `model_state_dict` (complete weights)
- **Status**: Models complete and usable

### The Questions

1. **Were the 70-74% accuracies REAL or a measurement error?**
2. **What caused the 5-10 point accuracy drop in the second run?**
3. **Can we reproduce the 70-74% results with complete model saves?**

---

## 🔍 INVESTIGATION CHECKLIST

### Phase 1: Review First Training Run Artifacts

#### 1.1 Check Training Logs (First Run)

Look for the **original training session logs** that reported 70-74% accuracy:

```bash
# If you have the training instance logs or terminal output
# Look for lines like:

# Example expected output:
# Epoch 14/15 - BTC-USD
#   Train Loss: 0.234
#   Val Loss: 0.312
#   Test Accuracy: 74.0%  ← THIS NUMBER
#
# Questions:
# - Was this test accuracy on the full test set?
# - Was this accuracy calculated correctly?
# - Was there any data leakage in the first run?
```

**Where to look**:
- GPU instance CloudWatch logs (if configured)
- Local terminal output if you saved it
- Any training history files (e.g., `training_history.json`)
- GPU instance filesystem (if snapshots exist)

#### 1.2 Check if Original Models Still Exist

```bash
# Check if the original 74%/70%/72% models exist anywhere:

# On terminated GPU instance (if EBS volume still exists):
aws ec2 describe-volumes --filters "Name=tag:Name,Values=crpbot-training*"

# If volume exists, you can:
# 1. Create snapshot
# 2. Attach to new instance
# 3. Mount and check for model files

# Files to look for:
# ~/crpbot/models/lstm_BTC-USD_*.pt (from first run)
# ~/crpbot/models/training_logs/
# ~/crpbot/models/checkpoints/
```

#### 1.3 Compare Training Configurations

Create a comparison between the two runs:

```bash
# Create investigation report
cat > V5_TRAINING_COMPARISON.md << 'EOF'
# V5 Training Comparison Report

## Training Run #1 (74%/70%/72% accuracy)
- **Date**: 2025-11-15 ~17:30 EST
- **Instance**: [INSTANCE_ID_1]
- **Training Script**: [VERSION/COMMIT]
- **Data**: data/training/ (665MB)
- **Hyperparameters**:
  - Epochs: ???
  - Batch size: ???
  - Learning rate: ???
  - Optimizer: ???
  - Loss function: ???
- **Results**:
  - BTC-USD: 74.0%
  - ETH-USD: 70.6%
  - SOL-USD: 72.1%
- **Issue**: Incomplete model save (no state_dict)

## Training Run #2 (63%/65%/65% accuracy)
- **Date**: 2025-11-15 ~18:00 EST
- **Instance**: i-058ebb1029b5512e2
- **Training Script**: FIXED version with state_dict
- **Data**: Same (data/training/, 665MB)
- **Hyperparameters**:
  - Epochs: 13-14
  - Batch size: ???
  - Learning rate: ???
  - Optimizer: ???
  - Loss function: ???
- **Results**:
  - BTC-USD: 63.6%
  - ETH-USD: 65.1%
  - SOL-USD: 65.6%
- **Status**: Complete model save (with state_dict)

## Differences Identified
[List any differences found between the two runs]

## Hypothesis
[Your theory on why accuracy dropped]

## Recommendation
[Whether to retrain or accept 63-66% models]
EOF

git add V5_TRAINING_COMPARISON.md
git commit -m "docs: V5 training comparison and investigation"
git push
```

---

### Phase 2: Verify Second Training Run Accuracy

#### 2.1 Independent Accuracy Verification

Verify that the 63-66% accuracies are correctly calculated:

```bash
# On the CURRENT GPU instance (or locally):
cd ~/crpbot

# Test the FIXED models on the test set
uv run python << 'EOF'
import torch
import pandas as pd
from pathlib import Path

# Load test data and FIXED model
test_data = pd.read_parquet('data/training/BTC-USD/test.parquet')
model_path = 'models/lstm_BTC-USD_1m_v5_FIXED.pt'

# Load model
checkpoint = torch.load(model_path, map_location='cpu')
print(f"Checkpoint accuracy: {checkpoint['accuracy']:.1%}")

# TODO: Run inference on test set and calculate accuracy manually
# This verifies the 63.6% is correct

# Expected: Should match 63.6%
EOF
```

#### 2.2 Check for Training Issues

Look for signs of training problems in the second run:

```bash
# Check if training converged properly
# Look for:
# - Loss decreasing over epochs
# - No overfitting (train >> test)
# - No underfitting (train ~= test ~= random)
# - No data issues (NaN losses, exploding gradients)

# Review training logs from second run
# Look for warnings or errors
```

---

### Phase 3: Root Cause Analysis

#### Possible Causes of Accuracy Drop

**Hypothesis 1: First Run Had Data Leakage**
- Accidentally used val/test data in training
- Features calculated incorrectly (looking ahead)
- Shuffle=True contaminated time series splits
- **Test**: Check first training script for data handling bugs

**Hypothesis 2: Second Run Had Training Issues**
- Learning rate too high/low
- Optimizer state corrupted
- Random seed different
- Early stopping triggered too early
- **Test**: Compare hyperparameters between runs

**Hypothesis 3: Measurement Error in First Run**
- Accuracy calculated on wrong dataset
- Metrics logged incorrectly
- Terminal output showed wrong metric (val instead of test)
- **Test**: Review first run's accuracy calculation code

**Hypothesis 4: Model Architecture Difference**
- First run used different architecture
- Second run loaded wrong config
- Feature engineering changed between runs
- **Test**: Compare model configs in checkpoints

**Hypothesis 5: Data Changed Between Runs**
- Data files modified between runs
- Different data uploaded to second instance
- Cache invalidated causing different preprocessing
- **Test**: Verify data checksums match

---

### Phase 4: Decisive Action Plan

Based on investigation findings, choose ONE path forward:

#### Option A: First Run Accuracies Were REAL → Retrain to Match

**If you determine**:
- 70-74% accuracies were correctly calculated
- No data leakage or bugs in first run
- Second run had a configuration issue

**Action**:
```bash
# 1. Identify what was different in first run
# 2. Replicate EXACT configuration
# 3. Retrain with FIXED model saving
# 4. Expect: 70-74% accuracy WITH complete model weights

# Launch new training run
# Cost: $0.53-0.59 additional
# Time: 28 minutes
# ETA: 19:15 EST
```

#### Option B: First Run Accuracies Were ERROR → Use 63-66% Models

**If you determine**:
- 70-74% accuracies were measurement errors
- First run had data leakage or bug
- Second run (63-66%) is correct

**Action**:
```bash
# Accept 63-66% models as correct V5 performance
# Proceed with deployment
# No retraining needed

# Deploy immediately
# Timeline: 20 minutes to go-live
```

#### Option C: Unclear → Quick Retrain with Verification

**If investigation is inconclusive**:
```bash
# Do a THIRD training run with:
# 1. Extra logging and validation
# 2. Manual accuracy verification after each epoch
# 3. Complete model saving
# 4. Different random seed

# This will give us a third data point
# If third run gets 70-74%: Option A was right
# If third run gets 63-66%: Option B was right
```

---

## 📋 DELIVERABLES

After completing investigation, provide:

1. **V5_TRAINING_COMPARISON.md** - Side-by-side comparison of both runs
2. **Root cause analysis** - What caused the discrepancy
3. **Accuracy verification** - Independent confirmation of 63-66%
4. **Training logs** - From both runs (if available)
5. **Recommendation** - Retrain or proceed with current models
6. **Timeline** - If retraining, ETA for completion

---

## 🎯 INVESTIGATION TIMELINE

**Quick Investigation** (if logs available): 15 minutes
- Review logs
- Compare configs
- Make determination

**Full Investigation** (if need to check AWS): 30-45 minutes
- Check for volume snapshots
- Review CloudWatch logs
- Verify data checksums

**With Retraining** (if needed): +28 minutes
- Total: 45-75 minutes to resolution

---

## 📞 REPORT BACK FORMAT

```
🔍 V5 ACCURACY INVESTIGATION COMPLETE

## Findings:
- First run accuracy (74%/70%/72%): [REAL / MEASUREMENT ERROR]
- Root cause: [Explanation]
- Evidence: [Logs, configs, verification results]

## Recommendation:
[RETRAIN / USE 63-66% MODELS / THIRD RUN]

## Reasoning:
[Why this recommendation]

## Next Steps:
[Specific actions to take]

## ETA to Production:
[Timeline]
```

---

**Priority**: 🔴 HIGH - Blocking production deployment
**Expected Time**: 15-75 minutes depending on findings
**Goal**: Determine true V5 model performance and deploy best models

Let's figure out what really happened! 🔍
