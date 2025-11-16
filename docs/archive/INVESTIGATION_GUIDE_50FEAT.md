# 🔍 Investigation Guide - 50-Feature Model Failure

**Date**: 2025-11-14
**Status**: INVESTIGATING ROOT CAUSE
**Current Results**: All 3 models at ~50% accuracy (random)

---

## 📊 What We Know

**Evaluation Results:**
```
BTC-USD: 50.26% accuracy, 47.7% calibration ❌
ETH-USD: 49.56% accuracy, 45.3% calibration ❌
SOL-USD: 49.52% accuracy, 50.4% calibration ❌
```

**Training Status:**
- Models were at 50% accuracy during training too
- Validation loss: 0.6894 (high)
- Models never learned anything

---

## 🔬 Investigation Steps

### Step 1: Run Comprehensive Investigation (15 min)

I've created a script that checks everything:

```bash
cd /home/numan/crpbot

# Investigate BTC-USD
uv run python scripts/investigate_50feat_failure.py --symbol BTC-USD

# Investigate ETH-USD
uv run python scripts/investigate_50feat_failure.py --symbol ETH-USD

# Investigate SOL-USD
uv run python scripts/investigate_50feat_failure.py --symbol SOL-USD
```

**What it checks:**

1. **Data Quality**
   - NaN values
   - Infinite values
   - Constant features (no variance)
   - Extreme outliers

2. **Target Distribution**
   - Class balance (should be ~50/50)
   - Is target even defined?
   - Majority class baseline

3. **Feature Statistics**
   - Feature scales
   - Low variance features
   - High correlations (multicollinearity)

4. **Data Leakage**
   - Features that perfectly predict target
   - Look-ahead features
   - Suspiciously high correlations

5. **Simple Baseline**
   - Trains Logistic Regression
   - If baseline works (>55%), problem is LSTM training
   - If baseline fails (<52%), problem is data/target

---

## 🎯 What to Look For

### ❌ **Critical Issues** (Stop and fix immediately)

1. **Baseline < 52% accuracy**
   ```
   ❌ CRITICAL: Even simple baseline can't beat random
   ```
   **Meaning**: The target is unpredictable from the features
   **Fix**: Change target definition or features

2. **High NaN/Inf counts**
   ```
   ❌ FOUND NaN values in 15+ features
   ```
   **Meaning**: Data quality issue
   **Fix**: Improve feature engineering

3. **Target correlation > 0.8**
   ```
   ⚠️ WARNING: feature_X correlates 0.95 with target
   ```
   **Meaning**: Data leakage
   **Fix**: Remove leaking features

### ⚠️ **Warning Issues** (May cause problems)

1. **Baseline 52-55% accuracy**
   ```
   ⚠️ WARNING: Baseline barely beats random
   ```
   **Meaning**: Very hard problem
   **Fix**: Try different architecture or features

2. **High feature correlations**
   ```
   ⚠️ Found 20 highly correlated pairs (>0.95)
   ```
   **Meaning**: Redundant features
   **Fix**: Feature selection

3. **Class imbalance**
   ```
   ⚠️ WARNING: Class 0: 70%, Class 1: 30%
   ```
   **Meaning**: Unbalanced target
   **Fix**: Use class weights or resampling

### ✅ **Good Signs**

1. **Baseline ≥ 55% accuracy**
   ```
   ✅ Baseline shows signal (60% accuracy)
   ```
   **Meaning**: Data is good, LSTM training issue
   **Fix**: Tune LSTM hyperparameters

2. **No data quality issues**
   ```
   ✅ No NaN, Inf, or constant features
   ```
   **Meaning**: Data is clean
   **Action**: Focus on model/training

---

## 📋 Quick Diagnosis Decision Tree

Run the investigation script, then:

```
Baseline Accuracy:
│
├─ < 52% ──→ 🔴 DATA/TARGET ISSUE
│             Actions:
│             1. Change target (longer horizon, threshold)
│             2. Use 5m or 15m candles instead of 1m
│             3. Try regression instead of classification
│
├─ 52-55% ─→ ⚠️  VERY HARD PROBLEM
│             Actions:
│             1. Improve features
│             2. Try simpler models first
│             3. Consider different prediction task
│
├─ 55-65% ─→ 🟡 TRAINABLE BUT HARD
│             Actions:
│             1. LSTM should work with good training
│             2. Check hyperparameters carefully
│             3. Try more epochs or different architecture
│
└─ ≥ 65% ──→ ✅ LSTM TRAINING ISSUE
              Actions:
              1. Data is good! Problem is training
              2. Check learning rate, batch size
              3. Try simpler LSTM (2 layers, 64 hidden)
              4. Add gradient clipping
```

---

## 🚀 Quick Run (Right Now)

**Run this to investigate all 3 symbols:**

```bash
cd /home/numan/crpbot

# Run investigations
echo "=== BTC-USD ===" > investigation_results.txt
uv run python scripts/investigate_50feat_failure.py --symbol BTC-USD >> investigation_results.txt 2>&1

echo -e "\n\n=== ETH-USD ===" >> investigation_results.txt
uv run python scripts/investigate_50feat_failure.py --symbol ETH-USD >> investigation_results.txt 2>&1

echo -e "\n\n=== SOL-USD ===" >> investigation_results.txt
uv run python scripts/investigate_50feat_failure.py --symbol SOL-USD >> investigation_results.txt 2>&1

echo "✅ Investigation complete! Check investigation_results.txt"
cat investigation_results.txt
```

**Time**: ~15 minutes total

---

## 📊 Expected Output

You'll see for each symbol:

```
======================================================================
INVESTIGATING 50-FEATURE MODEL FAILURE
Symbol: BTC-USD
File: data/features/features_BTC-USD_1m_2025-11-13_50feat.parquet
======================================================================

1. DATA QUALITY CHECK
   ✅ No NaN values
   ✅ No Inf values
   ✅ No constant features

2. TARGET DISTRIBUTION
   Class 0: 515,234 (50.1%)
   Class 1: 515,278 (49.9%)
   ✅ Well balanced classes

3. FEATURE STATISTICS
   ⚠️ Found 12 highly correlated pairs (>0.95)

4. DATA LEAKAGE CHECK
   ✅ No suspiciously high correlations

5. SIMPLE BASELINE MODEL
   Train Accuracy: 0.5234 (52.34%)
   Test Accuracy:  0.5187 (51.87%)

   ⚠️ WARNING: Baseline barely beats random
   → Problem is difficult, may need better features

======================================================================
RECOMMENDATION
======================================================================

🟡 Problem is very hard - baseline only 52%
Consider:
1. Change target (30-min horizon or 0.5% threshold)
2. Try 5-minute timeframe
3. Better feature engineering
```

---

## 🎯 Next Steps Based on Results

### If Baseline < 52% for ALL symbols:
→ **CRITICAL DATA ISSUE**
- Target definition is wrong
- 15-minute binary up/down is too noisy/random
- **Action**: Change to 30-60 minute or use thresholds

### If Baseline 52-55% for ALL symbols:
→ **HARD PROBLEM**
- Features have weak signal
- May need different approach
- **Action**: Try 31-feature model first to compare

### If Baseline ≥ 55% for ANY symbol:
→ **TRAINING ISSUE**
- Data is learnable
- LSTM not training properly
- **Action**: Debug LSTM training (hyperparameters, architecture)

### If Results VARY by symbol:
→ **SYMBOL-SPECIFIC ISSUE**
- Some symbols more predictable
- **Action**: Deploy only for working symbols

---

## 📝 What to Share

After running investigation, share:

1. **Baseline accuracy for each symbol**
   ```
   BTC: 51.2%
   ETH: 52.8%
   SOL: 50.9%
   ```

2. **Critical issues found**
   ```
   - NaN values in 5 features
   - Target imbalance: 65/35
   - 10 features with perfect correlation
   ```

3. **Recommendation from script**
   ```
   🔴 DATA/TARGET ISSUE
   or
   🟡 MODEL/TRAINING ISSUE
   ```

Then we'll know exactly what to fix!

---

**File**: `INVESTIGATION_GUIDE_50FEAT.md`
**Script**: `scripts/investigate_50feat_failure.py`
**Status**: Ready to run
**Time**: ~15 minutes
**Next**: Run investigation, share results
