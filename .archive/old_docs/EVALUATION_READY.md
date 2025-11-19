# Model Evaluation - Ready to Execute

**Date**: 2025-11-13
**Status**: Prepared and waiting for data fetch completion

---

## ✅ Pre-Flight Checks Complete

### 1. Scripts Verified
- ✅ `batch_engineer_features.sh` - Executable and ready
- ✅ `scripts/evaluate_model.py` - All dependencies loaded
- ✅ `scripts/engineer_features.py` - Ready for batch processing

### 2. Promotion Gates Criteria
```python
MIN_ACCURACY_GATE = 0.68           # Minimum 68% win rate required
MAX_CALIBRATION_ERROR_GATE = 0.05  # Maximum 5% calibration error
```

**What these mean**:
- **Accuracy Gate**: Model must correctly predict direction ≥68% of time on test data
- **Calibration Gate**: Model confidence scores must be well-calibrated (±5% error)
- Both gates must pass for promotion to production

### 3. Models Available for Evaluation
```
models/gpu_trained/
├── BTC_lstm_model.pt  (205 KB) - Trained on Colab Pro GPU
├── ETH_lstm_model.pt  (205 KB) - Trained on Colab Pro GPU
├── SOL_lstm_model.pt  (205 KB) - Trained on Colab Pro GPU
└── manifest.json      (Training metadata)
```

**Training Details** (from manifest):
- Platform: Google Colab Pro (Tesla T4 GPU, 16GB VRAM)
- Training time: ~10 minutes per model
- Date: 2025-11-12 05:48 UTC

---

## 📋 Execution Plan (Once Data Ready)

### Step 1: Feature Engineering (~10 minutes)
```bash
# Navigate to project directory
cd /root/crpbot
export PATH="/root/.local/bin:$PATH"
source .venv/bin/activate

# Run batch feature engineering
./batch_engineer_features.sh
```

**Expected Output**:
```
data/features/
├── features_BTC-USD_1m_latest.parquet  (~40-50 MB, 39 columns)
├── features_ETH-USD_1m_latest.parquet  (~40-50 MB, 39 columns)
└── features_SOL-USD_1m_latest.parquet  (~30-40 MB, 39 columns)
```

**Features Generated**: 39 total
- 5 OHLCV base columns
- 31 numeric features (session, spread, volume, MA, technical, volatility)
- 3 categorical features (one-hot encoded)

---

### Step 2: Model Evaluation (~15 minutes)

#### BTC-USD Evaluation
```bash
uv run python scripts/evaluate_model.py \
  --model models/gpu_trained/BTC_lstm_model.pt \
  --symbol BTC-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05
```

#### ETH-USD Evaluation
```bash
uv run python scripts/evaluate_model.py \
  --model models/gpu_trained/ETH_lstm_model.pt \
  --symbol ETH-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05
```

#### SOL-USD Evaluation
```bash
uv run python scripts/evaluate_model.py \
  --model models/gpu_trained/SOL_lstm_model.pt \
  --symbol SOL-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05
```

**Expected Metrics**:
- Win Rate (accuracy) on test set
- Calibration error (Brier score / ECE)
- Sharpe ratio (optional)
- Max drawdown (optional)
- Pass/Fail determination for each model

---

### Step 3: Model Promotion (if passing)
Models that pass both gates will be automatically copied to:
```
models/promoted/
├── lstm_BTC-USD_promoted.pt
├── lstm_ETH-USD_promoted.pt
└── lstm_SOL-USD_promoted.pt
```

---

### Step 4: Train Transformer Model (~40-60 minutes)
```bash
# Multi-coin transformer (requires all symbols' features)
uv run python apps/trainer/main.py \
  --task transformer \
  --epochs 15
```

**Architecture**:
- 100-minute lookback window
- 4-layer transformer encoder
- 8 attention heads
- Trained on all symbols simultaneously
- Output: Trend strength prediction (0-1)

**Expected Output**:
```
models/transformer_multi_v{timestamp}.pt
```

---

## 📊 Evaluation Criteria Details

### Accuracy Gate (68%)
Models trained on synthetic data may not pass this gate. Real data is essential.

**What happens if a model fails**:
- Option A: Retrain on real data (recommended)
- Option B: Adjust thresholds (not recommended for production)
- Option C: Use passing models only

### Calibration Gate (5%)
Ensures model confidence scores are trustworthy:
- If model says 75% confident, it should be right ~75% of time
- Poor calibration = overconfident or underconfident predictions
- Critical for FTMO risk management

---

## 🔧 Troubleshooting

### If Feature Engineering Fails
```bash
# Check raw data files exist
ls -lh data/raw/*_1m_*.parquet

# Manually engineer single symbol
uv run python scripts/engineer_features.py \
  --input data/raw/BTC-USD_1m_2023-11-10_2025-11-13.parquet \
  --symbol BTC-USD \
  --interval 1m
```

### If Evaluation Fails
```bash
# Check feature file exists
ls -lh data/features/features_BTC-USD_1m_latest.parquet

# Validate feature quality
uv run python scripts/validate_data_quality.py --symbol BTC-USD

# Check model file
ls -lh models/gpu_trained/BTC_lstm_model.pt
```

---

## 📈 Expected Timeline

| Task | Duration | Status |
|------|----------|--------|
| Data Fetch | 20-30 min | 🔄 In Progress |
| Feature Engineering | 10 min | ⏹️ Queued |
| Model Evaluation (3 models) | 15 min | ⏹️ Queued |
| Transformer Training | 40-60 min | ⏹️ Queued |
| **Total** | **85-115 min** | - |

---

## ✅ Ready to Execute

All dependencies verified. Commands prepared. Waiting for data fetch completion.

**Next action**: Monitor data fetches and execute Step 1 when complete.
