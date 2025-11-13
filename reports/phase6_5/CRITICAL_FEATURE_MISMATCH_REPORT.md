# CRITICAL: Feature Mismatch Between Colab Training and Local Evaluation

**Date**: 2025-11-13
**Status**: 🔴 **BLOCKING** - Models cannot be evaluated
**Impact**: Phase 6.5 training blocked, retraining required

---

## Problem Summary

The models trained on Colab Pro (Nov 13, 2025) **cannot be loaded** for evaluation due to a critical feature dimension mismatch.

```
Error: RuntimeError: size mismatch for lstm.weight_ih_l0:
  copying a param with shape torch.Size([512, 50]) from checkpoint,
  the shape in current model is torch.Size([512, 31])
```

### Root Cause Analysis

| Environment | Input Features | Status |
|------------|----------------|---------|
| **Colab Training** | 50 features | ✅ Models trained successfully |
| **Local Evaluation** | 31 features | ❌ Cannot load models |
| **Mismatch** | **+19 unknown features** | 🔴 **BLOCKING** |

**Hypothesis**: Colab training environment used additional features (likely multi-timeframe features from Phase 3.5) that were not synchronized with local feature files.

---

## Failed Models

All 3 newly trained models are **incompatible** with local evaluation:

1. `models/new/lstm_BTC_USD_1m_7b5f0829.pt` (3.9 MB)
   - Input size: 50 features
   - Local expects: 31 features
   - Status: ❌ Cannot load

2. `models/new/lstm_ETH_USD_1m_7b5f0829.pt` (3.9 MB)
   - Input size: 50 features
   - Local expects: 31 features
   - Status: ❌ Cannot load

3. `models/new/lstm_SOL_USD_1m_7b5f0829.pt` (3.9 MB)
   - Input size: 50 features
   - Local expects: 31 features
   - Status: ❌ Cannot load

**Training Metadata** (from checkpoint):
```python
{
  'epoch': 1,
  'val_accuracy': 0.4996,  # ~50% (baseline)
  'val_loss': 0.6894,
  'learning_rate': 0.001
}
```

---

## Current Feature Set (31 Features)

Our local feature engineering produces **31 features** (excluding OHLCV + metadata):

### Session Features (5)
- `session_tokyo`, `session_london`, `session_new_york`
- `day_of_week`, `is_weekend`

### Spread Features (4)
- `spread`, `spread_pct`, `atr`, `spread_atr_ratio`

### Volume Features (3)
- `volume_ma`, `volume_ratio`, `volume_trend`

### Moving Averages (8)
- `sma_7`, `price_sma_7_ratio`
- `sma_14`, `price_sma_14_ratio`
- `sma_21`, `price_sma_21_ratio`
- `sma_50`, `price_sma_50_ratio`

### Technical Indicators (8)
- `rsi`, `macd`, `macd_signal`, `macd_diff`
- `bb_high`, `bb_low`, `bb_width`, `bb_position`

### Volatility Regime (3)
- `volatility_low`, `volatility_medium`, `volatility_high`

**Total**: 31 features

---

## Missing 19 Features

The Colab models were trained with **50 input features**, indicating 19 additional features were used. These are likely:

### Suspected Multi-Timeframe Features (Phase 3.5)
- Cross-TF alignment scores (3-5 features)
- Higher timeframe OHLCV (5m, 15m, 1h prefixes) (12-16 features)
- Multi-TF volatility indicators

**Note**: The exact 19 features are **unknown** because the Colab training environment's feature set was not documented.

---

## Solution: Retrain with 31-Feature Files

**Recommended Action**: Retrain all 3 models on Colab Pro using the **correct 31-feature parquet files**.

### Why This Solution?

1. ✅ **Known Feature Set**: We have stable 31-feature engineering pipeline
2. ✅ **Ready Files**: Feature parquets already generated and validated
3. ✅ **Consistency**: Ensures training/evaluation parity
4. ✅ **Clean State**: Avoids unknown feature dependencies
5. ⏱️ **Feasible**: ~57 minutes GPU time (already proven successful)

### Alternative (Not Recommended)

Add 19 unknown multi-TF features to local evaluation:
- ❌ Don't know which 19 features Colab used
- ❌ Requires multi-TF data fetching infrastructure
- ❌ Adds complexity without validation
- ❌ May break existing feature pipeline

---

## Next Steps

### Step 1: Prepare Feature Files for Upload ✅ COMPLETE
```bash
# Files ready in data/features/:
features_BTC-USD_1m_2025-11-13.parquet  (210 MB, 31 features)
features_ETH-USD_1m_2025-11-13.parquet  (200 MB, 31 features)
features_SOL-USD_1m_2025-11-13.parquet  (184 MB, 31 features)
```

### Step 2: Upload to Google Drive (Manual)
1. Upload 3 parquet files to Google Drive
2. Mount Drive in Colab notebook
3. Verify files loaded correctly with 31 features

### Step 3: Retrain on Colab Pro
```python
# Expected training config:
- Architecture: 128 hidden, 3 layers, bidirectional, 0.35 dropout
- Epochs: 50
- Input size: 31 features ✅ (CRITICAL FIX)
- Batch size: 32
- Learning rate: 0.001 with cosine annealing
- Duration: ~57 minutes total for 3 models
```

### Step 4: Download and Evaluate
```bash
# Download new models from S3
aws s3 sync s3://crpbot-models/ models/retrained/

# Evaluate with correct architecture
python scripts/evaluate_model.py \
  --model models/retrained/lstm_BTC_USD_1m_*.pt \
  --symbol BTC-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05
```

### Step 5: Promotion Gate Check
- ✅ Win rate ≥ 68%
- ✅ Calibration error ≤ 5%
- ✅ Backtest Sharpe > 1.0 (optional)
- ✅ Max drawdown < 15% (optional)

---

## Impact on Timeline

**Previous Timeline** (blocked):
- ✅ Data pipeline: COMPLETE
- ✅ Feature engineering: COMPLETE
- ❌ LSTM training: BLOCKED (feature mismatch)
- ⏹️ Model evaluation: BLOCKED
- ⏹️ Promotion gates: BLOCKED

**Revised Timeline** (after retraining):
- ✅ Data pipeline: COMPLETE
- ✅ Feature engineering: COMPLETE
- 🔄 **LSTM retraining: ~57 minutes** (manual Colab upload required)
- 🔄 Model evaluation: ~1-2 hours (after retraining)
- 🔄 Promotion gates: ~15 minutes
- 🔄 Phase 6.5 dry-run: 3-5 days observation

**Total Delay**: ~2 hours + manual upload time

---

## Lessons Learned

1. **Feature Versioning**: Always version feature sets and document training configuration
2. **Environment Parity**: Ensure Colab training uses identical feature files as local evaluation
3. **Pre-Flight Checks**: Verify input dimensions before starting expensive GPU training
4. **Metadata Logging**: Save feature column names in model metadata for validation

---

## Files Reference

### Local Feature Files (31 features)
```
data/features/features_BTC-USD_1m_2025-11-13.parquet
data/features/features_ETH-USD_1m_2025-11-13.parquet
data/features/features_SOL-USD_1m_2025-11-13.parquet
```

### Failed Colab Models (50 features)
```
models/new/lstm_BTC_USD_1m_7b5f0829.pt
models/new/lstm_ETH_USD_1m_7b5f0829.pt
models/new/lstm_SOL_USD_1m_7b5f0829.pt
```

### Training Code
```
apps/trainer/train/train_lstm.py      (local - 31 features ✅)
scripts/train_colab_btc.py            (Colab - used 50 features ❌)
```

---

**Prepared by**: Claude Code
**Next Action**: Upload feature parquets to Colab and retrain with 31-feature configuration
