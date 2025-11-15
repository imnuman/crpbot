# 🔍 Next Action: Run Investigation

**Date**: 2025-11-14
**Status**: READY TO EXECUTE
**Time Required**: ~15 minutes

---

## 📋 What I've Created

✅ **Investigation Script**: `scripts/investigate_50feat_failure.py`
- Checks data quality (NaN, Inf, outliers)
- Analyzes target distribution
- Tests simple baseline model (Logistic Regression)
- Detects data leakage
- Provides specific recommendations

✅ **Quick Runner**: `scripts/run_investigation.sh`
- Runs investigation on all 3 symbols automatically
- Saves results to timestamped file
- Takes ~15 minutes total

✅ **Documentation**: `INVESTIGATION_GUIDE_50FEAT.md`
- How to interpret results
- Decision tree for next steps
- Complete guide

---

## 🚀 What You Do Now (Simple)

### Option 1: Quick Run (All 3 symbols) ⭐ Recommended

```bash
cd /home/numan/crpbot
./scripts/run_investigation.sh
```

That's it! Wait ~15 minutes, then share the output with me.

### Option 2: Manual Run (One symbol at a time)

```bash
cd /home/numan/crpbot

# BTC-USD
uv run python scripts/investigate_50feat_failure.py --symbol BTC-USD

# ETH-USD
uv run python scripts/investigate_50feat_failure.py --symbol ETH-USD

# SOL-USD
uv run python scripts/investigate_50feat_failure.py --symbol SOL-USD
```

---

## 🎯 What We're Looking For

The investigation will tell us THE ROOT CAUSE:

### Scenario 1: Baseline < 52% ❌ DATA/TARGET ISSUE
```
Test Accuracy: 0.5087 (50.87%)
❌ CRITICAL: Even simple baseline can't beat random
```

**Meaning**: The 15-minute binary prediction is essentially random
**Fix**: Change target definition (longer horizon or threshold)
**Action**: Modify target, re-engineer features

### Scenario 2: Baseline 52-55% ⚠️ VERY HARD PROBLEM
```
Test Accuracy: 0.5342 (53.42%)
⚠️ WARNING: Baseline barely beats random
```

**Meaning**: Features have weak signal
**Fix**: Better features or simpler model first
**Action**: Try 31-feature model to compare

### Scenario 3: Baseline 55-65% 🟡 TRAINABLE
```
Test Accuracy: 0.6012 (60.12%)
✅ Baseline shows signal (60% accuracy)
```

**Meaning**: Problem is learnable, LSTM should work
**Fix**: LSTM training issue (hyperparameters)
**Action**: Tune training or simplify architecture

### Scenario 4: Baseline ≥ 65% ✅ TRAINING ISSUE
```
Test Accuracy: 0.6823 (68.23%)
✅ Strong baseline (>65%)
→ LSTM failing at 50% suggests training issue
```

**Meaning**: Data is GOOD, LSTM is the problem
**Fix**: Fix LSTM training (learning rate, architecture)
**Action**: Debug training, not data

---

## 📊 What to Share With Me

After running, share:

1. **Baseline accuracies**:
   ```
   BTC-USD: XX.X%
   ETH-USD: XX.X%
   SOL-USD: XX.X%
   ```

2. **Any critical issues**:
   ```
   ❌ Found NaN in 5 features
   ⚠️ High correlation: feature_X <-> target: 0.92
   ```

3. **The recommendation**:
   ```
   🔴 DATA/TARGET ISSUE
   or
   🟡 MODEL/TRAINING ISSUE
   ```

---

## ⏱️ Timeline

```
Now:        Run investigation [15 min]
+15 min:    Share results with QC Claude
+20 min:    QC Claude analyzes and recommends fix
+30 min:    Execute the fix (varies based on issue)

Decision point: +20 minutes from now
```

---

## 💡 Why This Matters

Instead of blindly retraining with 31 features, we're:

✅ **Finding the ROOT CAUSE**
- Is it the data?
- Is it the target?
- Is it the model?
- Is it the training?

✅ **Making INFORMED decision**
- Don't waste 2 hours retraining if target is broken
- Don't change data if training is broken
- Fix the RIGHT thing

✅ **Learning for future**
- Understand what went wrong
- Prevent same mistake
- Build better models

---

## 🚀 Ready?

Just run:

```bash
cd /home/numan/crpbot
./scripts/run_investigation.sh
```

Then paste the results here. I'll tell you exactly what to fix!

---

**File**: `NEXT_ACTION_INVESTIGATION.md`
**Status**: READY TO EXECUTE
**Required**: Your local machine (has the data)
**Time**: 15 minutes
**Next**: Share results with me
