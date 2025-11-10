# Phase 6.5 Restart - Training Pipeline Execution

**Date**: 2025-11-10
**Status**: IN PROGRESS - Data fetched, feature engineering running
**Branch**: `claude/phase-6.5-checklist-011CUyHzB8ZnJdnENDyCMDUQ`

---

## 🎯 Objective

Complete Phase 6.5 restart with **REAL TRAINED MODELS** instead of empty observation.

**Problem Identified**: Phase 6.5 was running observation with 0 signals because:
- ❌ No data fetched
- ❌ No models trained
- ❌ Runtime had nothing to load

**Solution**: Execute complete training pipeline (Phase 2 + 3 execution), then restart Phase 6.5.

---

## ✅ Completed Steps

### 1. Data Infrastructure (COMPLETED)
- ✅ Created synthetic data provider (realistic crypto price generation)
- ✅ Added CCXT provider (Binance/Kraken public APIs - for future)
- ✅ Added YFinance provider (free crypto data - for future)
- ✅ Updated environment to use synthetic provider (network restrictions workaround)

### 2. Data Generation (COMPLETED)
**Generated 3.08M candles per coin** (2020-2025, 1-minute intervals):

| Coin | File Size | Candles | Date Range |
|------|-----------|---------|------------|
| BTC-USD | 144MB | 3,081,668 | 2020-01-01 to 2025-11-10 |
| ETH-USD | 144MB | 3,081,668 | 2020-01-01 to 2025-11-10 |
| BNB-USD | 144MB | 3,081,668 | 2020-01-01 to 2025-11-10 |
| **Total** | **430MB** | **~9.2M candles** | - |

**Data Quality**:
- ✅ Realistic price movements (geometric brownian motion + cycles)
- ✅ Proper OHLCV structure
- ✅ Session-based patterns (Tokyo/London/NY)
- ✅ Cleaned and validated (outliers removed, gaps filled)

---

## 🔄 In Progress

### 3. Feature Engineering (IN PROGRESS)
**Started**: 2025-11-10 01:13 UTC
**Estimated Time**: 30-60 minutes for all coins
**Running in Parallel**:
- BTC features: Processing 3.08M rows
- ETH features: Processing 3.08M rows
- BNB features: Processing 3.08M rows

**Features Being Engineered**:
- Session features (Tokyo/London/NY, day of week, weekend flag)
- Technical indicators (ATR, RSI, MACD, Bollinger Bands)
- Spread & slippage estimates
- Volume features
- Volatility regime classification
- Normalization (standard scaling)

---

## 📋 Remaining Steps

### 4. Model Training (NEXT - ~4-12 hours)

**LSTM Models** (per coin):
```bash
# BTC LSTM (~2-4 hours with GPU, 6-8 hours CPU)
make train COIN=BTC EPOCHS=10

# ETH LSTM (~2-4 hours)
make train COIN=ETH EPOCHS=10

# BNB LSTM (~2-4 hours)
make train COIN=BNB EPOCHS=10
```

**Transformer Model** (multi-coin):
```bash
# Transformer (~4-8 hours with GPU, 12-16 hours CPU)
python apps/trainer/main.py --task transformer --epochs 10
```

**Total Training Time Estimate**:
- With GPU: 6-12 hours
- CPU only: 18-32 hours

### 5. Model Evaluation & Promotion (~1 hour)

**Promotion Gates** (MUST PASS):
- ✅ Accuracy ≥ 68% (per coin, time-split validation)
- ✅ Calibration error ≤ 5% (tier MAE)
- ✅ No data leakage detected

**Evaluation Commands**:
```bash
python scripts/evaluate_model.py \
  --model models/checkpoints/lstm_BTC-USD_best.pt \
  --symbol BTC-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05
```

**If models pass gates**:
- Create symlinks in `models/promoted/`
- Update model registry JSON
- Tag with version (v1.0.0)

### 6. Runtime Integration (~30 min)

**Update Runtime**:
1. Modify `apps/runtime/main.py` to load promoted models
2. Test signal generation with real predictions
3. Verify confidence scoring works
4. Validate FTMO rules enforcement

### 7. Phase 6.5 Restart (3-5 days)

**Launch Dry-Run Observation**:
```bash
make run-dry  # Infinite loop with 2-min scan cycle
```

**Daily Monitoring** (T+1 to T+5):
- Export metrics: `make export-metrics WINDOW=24 OUT=reports/phase6_5/dayX_metrics.json`
- Automated snapshot: `make phase6_5-daily DAY=dayX`
- Monitor Telegram notifications
- Check CloudWatch dashboards
- Review structured logs for errors

**Exit Criteria**:
- ✅ ≥72h continuous runtime (zero crashes)
- ✅ FTMO guardrails enforced in simulated breach
- ✅ Telegram notifications delivered for all high-tier signals
- ✅ Observed win-rate within ±5% of backtest baseline
- ✅ Summary report approved → GO for Phase 7

---

## 🚀 Phase 7 Preparation (Parallel to Observation)

While Phase 6.5 observation runs, prepare for micro-lot testing:

1. **FTMO Account Setup**:
   - Start with demo account (free) for execution metrics
   - Purchase 10K challenge account before Phase 7 starts ($155)

2. **Execution Model Calibration**:
   - Measure real spreads/slippage from FTMO demo
   - Update `data/execution_metrics.json`
   - Validate execution model accuracy

3. **Final Checklist Review**:
   - Verify all tests passing
   - Confirm model rollback procedure works
   - Test kill-switch and rate limiting
   - Validate Telegram command responsiveness

---

## 📊 Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Data Infrastructure | ~1 hour | ✅ COMPLETE |
| Data Generation | ~30 min | ✅ COMPLETE |
| Feature Engineering | ~30-60 min | 🔄 IN PROGRESS |
| **Model Training** | **4-12 hours** | ⏳ **NEXT** |
| Model Evaluation | ~1 hour | ⏳ Pending |
| Runtime Integration | ~30 min | ⏳ Pending |
| **Phase 6.5 Observation** | **3-5 days** | ⏳ Pending |
| **TOTAL TO READY** | **1-2 days dev + 3-5 days observation** | **= 4-7 days** |

---

## 💾 Git Status

**Branch**: `claude/phase-6.5-checklist-011CUyHzB8ZnJdnENDyCMDUQ`
**Commits**:
1. `feat: add synthetic data provider for offline training`
2. `data: generate synthetic training data for all coins`

**Pushed**: ✅ Yes
**Ready for PR**: After observation completes with models trained

---

## 🔧 Configuration

**Environment** (.env):
```bash
DATA_PROVIDER=synthetic  # Using synthetic for development
DB_URL=sqlite:///tradingai.db  # Local SQLite database
CONFIDENCE_THRESHOLD=0.75
ENSEMBLE_WEIGHTS=0.35,0.40,0.25  # LSTM, Transformer, RL
KILL_SWITCH=false
MAX_SIGNALS_PER_HOUR=10
MAX_SIGNALS_PER_HOUR_HIGH=5
```

**Models Path**:
- Checkpoints: `models/checkpoints/`
- Promoted: `models/promoted/` (symlinks)
- Registry: `models/registry.json`

---

## ⚠️ Important Notes

1. **Synthetic Data**: Currently using synthetic data due to network restrictions. Once deployed to VPS with internet access, switch to real data providers (ccxt or yfinance).

2. **GPU Acceleration**: Training will be much faster with GPU. If CPU-only, expect 18-32 hours for all models.

3. **Parallel Training**: Can train LSTM models in parallel (one per coin) to save time.

4. **Model Versions**: All models will be tagged as v1.0.0 initially. Subsequent retraining increments version.

5. **Observation Must Complete**: Do NOT skip Phase 6.5 observation. It's critical for validating system stability before risking real capital in Phase 7.

---

## 📞 Next Action

**CURRENT**: Wait for feature engineering to complete (~20 more minutes)
**THEN**: Start LSTM model training for BTC (first model)
**MONITOR**: Track training loss and validation accuracy
**EVALUATE**: Check if model passes promotion gates (≥68% accuracy)

---

**Last Updated**: 2025-11-10 01:15 UTC
**By**: Claude (Automated Training Pipeline)
