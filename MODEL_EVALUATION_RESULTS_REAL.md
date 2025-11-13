# Model Evaluation Results (Real Data)

**Date**: 2025-11-13 05:28 UTC
**Evaluation**: Using REAL market data (fixed evaluation pipeline)
**Fix Status**: ✅ Complete

---

## Executive Summary

All 3 LSTM models have been evaluated with **REAL market data** after fixing the evaluation pipeline bug. Results show models perform poorly and need retraining.

**Verdict**: ❌ **ALL 3 MODELS FAIL** promotion gates (13-30% win rate vs 68% threshold)

**Root Cause**: Models were trained correctly, but need better hyperparameters/architecture/features

---

## Evaluation Pipeline Fix Summary

### What Was Broken

**Before (Fake Data)**:
```python
entry_time = datetime.now()      # Fake timestamp
entry_price = 50000.0            # Fake $50K price
exit_price = entry_price * 1.01  # Fake 1% profit
```
**Result**: 100% win rate (impossible)

### What Was Fixed

**After (Real Data)**:
```python
entry_time = sample["timestamp"]           # Real timestamp from dataset
entry_price = sample["entry_price"].item() # Real BTC/ETH/SOL price
exit_price = sample["exit_price"].item()   # Real price 15 min later
```
**Result**: 13-30% win rate (realistic)

### Files Modified

1. **apps/trainer/train/dataset.py**:
   - Added `timestamps`, `entry_prices`, `exit_prices` arrays
   - Return them in `__getitem__`

2. **apps/trainer/eval/evaluator.py**:
   - Extract real timestamps/prices from batch
   - Use real data instead of fake values

3. **apps/trainer/eval/backtest.py**:
   - Fixed timezone issue (datetime.now() → datetime.now(timezone.utc))

---

## Model Evaluation Results

### BTC-USD LSTM Model

```
Model: lstm_BTC_USD_1m_a7aff5c4.pt
Architecture: 31 features, 2-layer bidirectional LSTM
Test Set: 154,503 sequences (July-October 2025)
```

**Metrics**:
- **Win Rate**: 13.32% ❌ (threshold: 68%)
- **Total Trades**: 44,993
- **Winning Trades**: 5,992
- **Losing Trades**: 39,001
- **Total PnL**: -$5,401.21
- **Avg PnL per Trade**: -$0.12
- **Max Drawdown**: 54.01%
- **Sharpe Ratio**: -13.73
- **Calibration Error**: 56.94% ❌ (threshold: 5%)
- **Avg Latency**: 1.55ms ✅
- **P90 Latency**: 2.93ms ✅

**Status**: ❌ **FAIL** - Does not meet promotion gates

**Failure Reasons**:
- Win rate 13.32% << 68% threshold (off by 54.68%)
- Calibration error 56.94% >> 5% threshold (off by 51.94%)

---

### ETH-USD LSTM Model

```
Model: lstm_ETH_USD_1m_a7aff5c4.pt
Architecture: 31 features, 2-layer bidirectional LSTM
Test Set: 154,503 sequences (July-October 2025)
```

**Metrics**:
- **Win Rate**: 25.42% ❌ (threshold: 68%)
- **Total Trades**: 54,298
- **Winning Trades**: 13,802
- **Losing Trades**: 40,496
- **Total PnL**: -$6,002.02
- **Avg PnL per Trade**: -$0.11
- **Max Drawdown**: 60.02%
- **Sharpe Ratio**: -7.44
- **Calibration Error**: 45.27% ❌ (threshold: 5%)
- **Avg Latency**: 1.54ms ✅
- **P90 Latency**: 2.90ms ✅

**Status**: ❌ **FAIL** - Does not meet promotion gates

**Failure Reasons**:
- Win rate 25.42% << 68% threshold (off by 42.58%)
- Calibration error 45.27% >> 5% threshold (off by 40.27%)

---

### SOL-USD LSTM Model

```
Model: lstm_SOL_USD_1m_a7aff5c4.pt
Architecture: 31 features, 2-layer bidirectional LSTM
Test Set: 154,503 sequences (July-October 2025)
```

**Metrics**:
- **Win Rate**: 30.14% ❌ (threshold: 68%) *[BEST OF 3]*
- **Total Trades**: 29,952
- **Winning Trades**: 9,027
- **Losing Trades**: 20,925
- **Total PnL**: -$4,158.34
- **Avg PnL per Trade**: -$0.14
- **Max Drawdown**: 41.58%
- **Sharpe Ratio**: -6.53
- **Calibration Error**: 39.67% ❌ (threshold: 5%)
- **Avg Latency**: 1.66ms ✅
- **P90 Latency**: 3.25ms ✅

**Status**: ❌ **FAIL** - Does not meet promotion gates

**Failure Reasons**:
- Win rate 30.14% << 68% threshold (off by 37.86%)
- Calibration error 39.67% >> 5% threshold (off by 34.67%)

---

## Comparative Analysis

| Metric | BTC-USD | ETH-USD | SOL-USD | Target |
|--------|---------|---------|---------|--------|
| **Win Rate** | 13.32% | 25.42% | **30.14%** | ≥68% |
| **Calibration Error** | 56.94% | 45.27% | **39.67%** | ≤5% |
| **Total PnL** | -$5,401 | -$6,002 | **-$4,158** | Positive |
| **Sharpe Ratio** | -13.73 | -7.44 | **-6.53** | >1.0 |
| **Max Drawdown** | 54.01% | 60.02% | **41.58%** | <15% |
| **Avg Latency** | 1.55ms | 1.54ms | 1.66ms | <5ms |

**Winner**: SOL-USD performs best across all metrics, but still fails gates

**Gap to Target**: All models are 37-55% below target win rate

---

## Why Models Are Failing

### 1. Architecture Issues
- **Current**: 2-layer LSTM, 64 hidden units
- **Possible Fix**: Deeper network (3-4 layers), larger hidden size (128-256)

### 2. Feature Engineering
- **Current**: 31 features (moving averages, RSI, MACD, sessions)
- **Missing**: Cross-timeframe features, order book depth, volatility regimes

### 3. Training Strategy
- **Current**: 15 epochs, standard optimizer
- **Possible Fix**: More epochs (30-50), learning rate scheduling, gradient clipping

### 4. Prediction Horizon
- **Current**: 15-minute ahead prediction
- **Issue**: Too long for 1-minute granularity, too noisy

### 5. Class Imbalance
- Models may be biased toward one direction due to imbalanced training data

### 6. Overfitting to Training Data
- High calibration error suggests poor probability estimates
- Models not generalizing well to test period

---

## Next Steps

### Option 1: Retrain with Improvements (Recommended)

**Priority Fixes** (~2-3 hours):
1. Increase model capacity (3-4 layers, 128 hidden units)
2. Train for more epochs (30-50 with early stopping)
3. Add learning rate scheduling (warm-up + cosine decay)
4. Implement class balancing (weighted loss)
5. Add dropout (0.3-0.4) for regularization

**Expected Improvement**: 30% → 55-65% win rate

### Option 2: Feature Engineering Improvements (~3-4 hours)

1. Add multi-timeframe features (5m, 15m, 1h alignment)
2. Add volatility regime features
3. Add order flow features (if available)
4. Feature selection (remove correlated features)

**Expected Improvement**: Additional 5-10% win rate

### Option 3: Alternative Architectures (~4-5 hours)

1. Try Transformer-only (already have code)
2. Try CNN-LSTM hybrid
3. Try attention mechanisms
4. Ensemble multiple architectures

**Expected Improvement**: 10-15% win rate boost

### Option 4: Hyperparameter Tuning (~4-6 hours)

1. Grid search over learning rates (1e-4 to 1e-2)
2. Try different sequence lengths (30, 60, 120 minutes)
3. Try different prediction horizons (5m, 10m, 15m)
4. Experiment with batch sizes (16, 32, 64)

**Expected Improvement**: 5-10% win rate

---

## Recommendation

**Immediate Action**: **Option 1 (Retrain with Improvements)**

**Why**:
- Fastest path to improvement (2-3 hours)
- Highest expected ROI (30% → 60%+ win rate)
- Addresses most critical issues (capacity, training, regularization)

**Steps**:
1. Update LSTM architecture (3 layers, 128 hidden, dropout 0.3)
2. Train for 50 epochs with early stopping (patience=10)
3. Use AdamW with learning rate schedule
4. Implement weighted loss for class balance
5. Re-evaluate against gates

**Timeline**:
- Code changes: 30 min
- Training (GPU): 20-30 min for 3 models
- Evaluation: 15 min
- **Total**: ~1.5 hours

**Success Criteria**: At least 1 model passes 68% win rate gate

---

## Conclusion

**Summary**: Evaluation pipeline now works correctly with real market data. All 3 models fail promotion gates (13-30% win rate vs 68% target). Models need retraining with improved architecture, more training, and better regularization.

**What Works** ✅:
- Data pipeline (1M+ rows real Coinbase data)
- Feature engineering (39 features, zero nulls)
- Model architecture (correct 31-feature input)
- Evaluation pipeline (NOW uses real data)
- Infrastructure (S3, Git, AWS, testing)

**What Doesn't Work** ❌:
- Model performance (win rates too low)
- Calibration (confidence scores inaccurate)
- Generalization (poor test set performance)

**Path Forward**: Retrain models with Option 1 improvements, re-evaluate, and make promotion decision based on real metrics.

---

**Next Session**: Implement Option 1 retraining improvements (~1.5 hours estimated)

**Current Status**: Phase 6.5 paused pending model retraining

**Last Updated**: 2025-11-13 05:28 UTC
