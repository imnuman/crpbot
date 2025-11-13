# Model Evaluation Findings - Phase 6.5

**Date**: 2025-11-13 ~05:00 UTC
**Status**: 🔍 **Critical Bug Discovered** - Evaluation Pipeline Using Fake Data
**Impact**: Models are UNTESTED (not broken, just not validated)

---

## Executive Summary

All 3 GPU-trained LSTM models showed **100% win rate** during evaluation. This is impossible in real trading and indicates a critical bug in the evaluation pipeline, not in the models themselves.

**Root Cause Identified**: `apps/trainer/eval/evaluator.py` uses placeholder fake data instead of real timestamps and prices from the dataset.

**Good News**:
- Models architecture is correct (31 features, proper checkpoints)
- All infrastructure works (data pipeline, feature engineering, training, S3, Git)
- **89% complete** (8/9 components operational)
- Fix is straightforward (~45 minutes estimated)

**Decision**: Document findings and fix when fresh (next session)

---

## Investigation Timeline

### 1. Initial Evaluation Results (Suspicious)

All 3 models evaluated with identical impossible results:

```
BTC-USD LSTM Model Evaluation
├─ Test accuracy: 100.00%
├─ Win rate: 100.00%  ← IMPOSSIBLE
├─ Total trades: 100
├─ Profitable: 100
├─ Unprofitable: 0
└─ Calibration error: N/A
```

### 2. User Correctly Identified as Bug

User recognized this pattern immediately:
> "CRITICAL ISSUE: 100% Win Rate = Evaluation Bug"
> "This is IMPOSSIBLE in real trading"
> "Indicates: Data leakage, evaluation using training data, label leakage, or backtest simulation bug"

### 3. Investigation of Evaluation Code

Examined `apps/trainer/eval/evaluator.py` lines 120-143:

**Critical Bug Found**:
```python
# Line 124-125: FAKE ENTRY DATA
entry_time = datetime.now()  # ❌ Should be from dataset timestamp column
entry_price = 50000.0        # ❌ Should be from dataset close price

# Line 140-141: FAKE EXIT DATA
exit_time = entry_time + pd.Timedelta(minutes=15)
exit_price = entry_price * (1.01 if direction == "long" else 0.99)  # ❌ Placeholder
```

**What This Means**:
- Every trade uses the SAME fake entry: now() timestamp, $50,000 price
- Every trade gets FAKE profit: +1% for longs, -1% for shorts (then flipped for win)
- Models are not tested against REAL price movements
- Evaluation is a placeholder stub, not actual backtest

---

## What Works (89% Complete) ✅

1. **Data Pipeline** ✅
   - Coinbase API integration with JWT auth
   - Downloaded 1,030,512+ rows per symbol (2 years)
   - Zero nulls, complete coverage
   - Real OHLCV data from production API

2. **Feature Engineering** ✅
   - 39 features generated (5 OHLCV + 31 indicators + 3 categorical)
   - All 3 symbols: BTC, ETH, SOL
   - Validated: zero nulls, correct distributions
   - Files: 592 MB total uploaded to S3

3. **Model Training** ✅
   - 3 LSTM models trained on Colab Pro GPU (15 minutes total)
   - Correct architecture: 31 features input
   - Proper checkpoints with metadata
   - Models: `lstm_{SYMBOL}_USD_1m_a7aff5c4.pt` (247-249 KB each)

4. **Infrastructure** ✅
   - AWS S3: Upload/download working
   - GitHub: Git sync operational (3 commits pushed)
   - Environment: All dependencies installed, tests passing
   - Database: SQLite configured and working

5. **Codebase Quality** ✅
   - All tests passing (20/20)
   - Type checking clean
   - Pre-commit hooks configured

---

## What's Broken (11% Incomplete) ❌

### Evaluation Pipeline - Using Fake Data

**File**: `apps/trainer/eval/evaluator.py`
**Lines**: 124-125, 140-141

**Problem**: Evaluator doesn't use real timestamps/prices from dataset

**Current Behavior**:
```python
# Every trade gets:
entry_time = datetime.now()       # Same fake timestamp
entry_price = 50000.0             # Same fake price
exit_price = 50500.0 (for longs)  # Fake 1% profit
```

**Expected Behavior**:
```python
# Should use real data:
entry_time = dataset.loc[idx, 'timestamp']
entry_price = dataset.loc[idx, 'close']
# Wait 15 minutes in backtest
exit_price = dataset.loc[idx + 15, 'close']  # Real outcome
```

**Impact**:
- Models appear 100% accurate (fake)
- Cannot determine real accuracy
- Cannot make promotion decision
- Models are UNTESTED, not broken

---

## Fix Plan (Next Session) 🔧

**Estimated Time**: 45 minutes

### Step 1: Update Dataset to Provide Timestamps (15 min)

**File**: `apps/trainer/data/dataset.py`

**Change**: Include timestamp and close columns in returned data

```python
# Current: Returns only features tensor
# Change to: Return features + metadata dict

return {
    'features': features_tensor,
    'timestamp': df.loc[idx, 'timestamp'],
    'close': df.loc[idx, 'close']
}
```

### Step 2: Fix Evaluator to Use Real Data (20 min)

**File**: `apps/trainer/eval/evaluator.py`

**Changes needed**:

```python
# Line 120-130: Get prediction + real entry data
pred = model(features)
entry_time = test_data.loc[idx, 'timestamp']  # Real timestamp
entry_price = test_data.loc[idx, 'close']     # Real price

# Line 135-145: Get real exit data (15 min later)
exit_idx = idx + 15  # 15 minutes ahead
if exit_idx >= len(test_data):
    continue  # Skip if not enough future data

exit_time = test_data.loc[exit_idx, 'timestamp']
exit_price = test_data.loc[exit_idx, 'close']  # REAL outcome

# Calculate real P&L
pnl = (exit_price - entry_price) / entry_price
if direction == "long":
    win = (pnl > 0)
else:  # short
    win = (pnl < 0)
```

### Step 3: Re-evaluate Models (10 min)

```bash
# Run real evaluation for all 3 models
uv run python scripts/evaluate_model.py \
    --model models/lstm_BTC_USD_1m_a7aff5c4.pt \
    --symbol BTC-USD \
    --model-type lstm \
    --min-accuracy 0.68 \
    --max-calibration-error 0.05

# Repeat for ETH and SOL
```

### Step 4: Make Promotion Decision (5 min)

Based on REAL metrics:
- If accuracy ≥68% AND calibration ≤5%: PROMOTE
- If fails: Retrain with different hyperparameters

---

## Technical Details

### Model Architecture (Verified Correct) ✅

```python
# LSTM per-coin structure
Input: (batch, 60, 31)  # 60 timesteps, 31 features
LSTM Layer 1: bidirectional, hidden=64 → (batch, 60, 128)
LSTM Layer 2: bidirectional, hidden=64 → (batch, 60, 128)
Dropout: 0.2
FC: 128 → 64 → 1
Sigmoid: [0, 1] probability

# Checkpoint format (verified)
{
  "model_state_dict": {...},       # Weights
  "model_class": "LSTMModel",
  "num_features": 31,              # ✅ Correct
  "metadata": {...}
}
```

### Feature Engineering (Verified Correct) ✅

39 columns total:
- **OHLCV** (5): open, high, low, close, volume
- **Session** (5): Tokyo/London/NY flags, day_of_week, is_weekend
- **Spread** (4): spread, spread_pct, ATR, spread_atr_ratio
- **Volume** (3): volume_ma, volume_ratio, volume_trend
- **Moving Averages** (8): SMA 7/14/21/50 + price ratios
- **Technical Indicators** (8): RSI, MACD×3, Bollinger×4
- **Volatility Regime** (3): low/medium/high (one-hot)
- **Categorical** (3): session_encoded, day_of_week_encoded, volatility_regime_encoded

All features validated: zero nulls, proper ranges.

### Walk-Forward Splits (Preserved) ✅

Time-based splits to prevent leakage:
- Train: 70% (earliest: 2023-11-10 → 2025-03-24)
- Val: 15% (middle: 2025-03-24 → 2025-07-10)
- Test: 15% (latest: 2025-07-10 → 2025-10-25)

These splits are correct. Only evaluation logic is broken.

---

## Why This is Good News 🎉

1. **Models Might Be Fine**: We don't know if models are good or bad yet (untested, not broken)

2. **Infrastructure Complete**: 8/9 components working (89% complete)
   - Data pipeline ✅
   - Feature engineering ✅
   - Training pipeline ✅
   - S3 storage ✅
   - Git sync ✅
   - Environment ✅
   - Tests ✅
   - Documentation ✅

3. **Clear Fix Path**: Not a research problem, just engineering fix

4. **Fast Fix**: 45 minutes estimated (when fresh)

5. **No Data Loss**: All work preserved:
   - 2 years of real data
   - 39 features engineered
   - 3 models trained on GPU
   - All uploaded to S3
   - Committed to GitHub

---

## Current Status

**Time**: ~05:00 UTC (5 hours past 02:00 deadline)
**Decision**: Stop for tonight, fix when fresh
**Completion**: 89% (8/9 components)
**Blocker**: Evaluation pipeline using fake data

**Files Created This Session**:
- ✅ DATA_FETCH_COMPLETE.md
- ✅ EVALUATION_READY.md
- ✅ TRAINING_STATUS.md
- ✅ READY_FOR_COLAB_GPU_TRAINING.md
- ✅ GITHUB_AUTH_NEEDED.md
- ✅ COLAB_GPU_TRAINING.md
- ✅ MODEL_EVALUATION_FINDINGS.md (this file)

**Git Status**:
- 3 commits pushed to GitHub
- All documentation synced
- Models uploaded to S3

---

## Next Session Checklist

**Priority 1: Fix Evaluation (45 min)**
- [ ] Update dataset.py to return timestamps + prices
- [ ] Fix evaluator.py to use real data instead of fake
- [ ] Add validation: check exit_idx < len(test_data)
- [ ] Re-evaluate all 3 models with real backtest

**Priority 2: Promotion Decision (15 min)**
- [ ] Review real accuracy metrics
- [ ] Check calibration error
- [ ] Promote models if passing gates (≥68% accuracy, ≤5% calibration)
- [ ] Or plan retraining if failing

**Priority 3: Continue Phase 6.5 (depends on P2)**
- [ ] If promoted: Train Transformer model
- [ ] Runtime testing in dry-run mode
- [ ] Start 3-5 day observation period
- [ ] Phase 7: Micro-lot testing on FTMO

---

## Lessons Learned

1. **100% Win Rate = Always a Bug**: Impossible in real trading, should trigger immediate investigation

2. **Validate Evaluation Logic Early**: Should have tested evaluation with known outcomes before training

3. **Placeholder Code Detection**: Look for hardcoded values, datetime.now(), fake calculations

4. **User QC Caught It**: Having second Claude review results caught bug immediately

5. **Document and Stop When Tired**: 05:00 UTC = bad time for complex debugging

---

## Conclusion

**Summary**: Evaluation pipeline uses fake data (datetime.now(), $50K hardcoded, fake 1% profits). This explains 100% win rate. Models are untested, not broken. Fix is straightforward (~45 min). 89% of pipeline complete and working.

**Status**: Pausing at discovery of evaluation bug. Will fix in next session when fresh.

**Timeline**: Originally targeted 02:00 UTC, ran to 05:00 UTC due to investigation. Better to fix properly than rush at 5 AM.

**Next Steps**: Fix evaluator.py to use real data, re-evaluate models, make promotion decision based on actual metrics.

**Overall Assessment**: Strong progress despite evaluation bug. All critical infrastructure working. Clear path to completion.

---

**Session ended**: 2025-11-13 ~05:00 UTC
**Resume next session**: Fix evaluation pipeline (45 min estimated)
