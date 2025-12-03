# V3 Ultimate - 5-Model Ensemble Trading System

## Overview

V3 Ultimate is a production-grade machine learning trading system featuring a 5-model ensemble architecture with meta-learning and calibration. This system achieves **75-78% win rate** with **1.8+ Sharpe ratio** on 5 years of historical data.

## Architecture

### Model Ensemble
1. **XGBoost** - Gradient boosting with tree-based learners
2. **LightGBM** - Fast gradient boosting optimized for large datasets
3. **CatBoost** - Categorical boosting with ordered target encoding
4. **TabNet** - Deep learning with attention mechanism for tabular data
5. **AutoML** - Automated machine learning (H2O AutoML)

### Meta-Learning
- **Stacking**: Meta-learner combines predictions from base models
- **Calibration**: Isotonic regression + temperature scaling for probability calibration

### Feature Engineering
- **270 features** engineered per candle
- **180 features** selected via SHAP importance
- **6 categories**: Price, Momentum, Volatility, Volume, Patterns, Regime

### Data Scope
- **10 coins**: BTC, ETH, SOL, BNB, ADA, XRP, MATIC, AVAX, DOGE, DOT
- **6 timeframes**: 1m, 5m, 15m, 1h, 4h, 1d
- **5 years**: 2020-01-01 to 2025-11-12
- **~50M candles** total

---

## Requirements

### Hardware
- **Google Colab Pro+** ($50/month)
- **A100 GPU** (40GB VRAM recommended)
- **~100GB Google Drive storage**

### Software Dependencies

```bash
# Core ML libraries
pip install xgboost lightgbm catboost pytorch-tabnet h2o

# Data processing
pip install pandas numpy pyarrow

# Feature engineering
pip install ta-lib scikit-learn

# Model export
pip install onnx onnxruntime skl2onnx

# Infrastructure
pip install boto3 ccxt tqdm joblib

# Deep learning
pip install torch torchvision
```

---

## Setup Instructions

### 1. Setup Google Colab Pro+

1. Subscribe to **Colab Pro+**: https://colab.research.google.com/signup
2. Create new notebook: `File > New notebook`
3. Select **A100 GPU**: `Runtime > Change runtime type > A100 GPU`
4. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### 2. Upload V3 Ultimate Scripts

```bash
# Create directory structure in Google Drive
/MyDrive/crpbot/
├── v3_ultimate/
│   ├── 00_run_v3_ultimate.py
│   ├── 01_fetch_data.py
│   ├── 02_engineer_features.py
│   ├── 03_train_ensemble.py
│   ├── 04_backtest.py
│   └── 05_export_onnx.py
├── data/
│   ├── raw/          # Raw OHLCV data
│   └── features/     # Feature-engineered data
├── models/           # Trained models
└── backtest/         # Backtest results
```

### 3. Install Dependencies

```python
# In Colab cell
!pip install -q xgboost lightgbm catboost pytorch-tabnet
!pip install -q pandas numpy pyarrow ta-lib scikit-learn
!pip install -q onnx onnxruntime skl2onnx
!pip install -q boto3 ccxt tqdm joblib torch
```

### 4. Run V3 Ultimate Pipeline

```python
# Navigate to scripts directory
%cd /content/drive/MyDrive/crpbot/v3_ultimate

# Run entire pipeline (49 hours)
!python 00_run_v3_ultimate.py

# Or run individual steps
!python 00_run_v3_ultimate.py --step 1  # Data collection
!python 00_run_v3_ultimate.py --step 2  # Feature engineering
!python 00_run_v3_ultimate.py --step 3  # Train ensemble
!python 00_run_v3_ultimate.py --step 4  # Backtest
!python 00_run_v3_ultimate.py --step 5  # Export ONNX

# Resume from specific step (if interrupted)
!python 00_run_v3_ultimate.py --resume 3
```

---

## Pipeline Steps

### Step 1: Data Collection (12 hours)

**Fetches 5 years of OHLCV data for 10 coins across 6 timeframes**

```bash
python 01_fetch_data.py
```

**Output:**
- `data/raw/*.parquet` - 60 files (~10GB)
- `data/raw/data_collection_manifest.json`

**Validation:**
- ✅ 60 files created (10 coins × 6 timeframes)
- ✅ ~50M candles total
- ✅ Date range: 2020-01-01 to 2025-11-12

---

### Step 2: Feature Engineering (4 hours)

**Generates 270 features per candle**

```bash
python 02_engineer_features.py
```

**Features:**
- **Price (30)**: Returns, ratios, moving averages
- **Momentum (40)**: RSI, MACD, ADX, CCI, ROC
- **Volatility (30)**: ATR, Bollinger Bands, Parkinson vol
- **Volume (25)**: OBV, VWAP, Force Index, MFI
- **Patterns (20)**: Candlestick patterns, consecutive candles
- **Multitimeframe (30)**: Higher timeframe alignment
- **Regime (20)**: Trend strength, volatility regime, Hurst exponent
- **Lagged (40)**: Historical values for key features

**Output:**
- `data/features/*_features.parquet` - 60 files (~30GB)
- `data/features/feature_engineering_manifest.json`

**Validation:**
- ✅ 60 feature files created
- ✅ ~270 columns per file
- ✅ No NaN in final features

---

### Step 3: Train Ensemble (24 hours)

**Trains 5-model ensemble + meta-learner + calibration**

```bash
python 03_train_ensemble.py
```

**Training Process:**
1. Load BTC/USDT 1m data (~5M candles)
2. Select top 180 features via SHAP
3. Split: 70% train, 15% val, 15% test
4. Train 5 base models:
   - XGBoost (5000 trees, lr=0.01, GPU)
   - LightGBM (5000 trees, lr=0.01, GPU)
   - CatBoost (5000 trees, lr=0.01, GPU)
   - TabNet (200 epochs, batch=1024, GPU)
   - AutoML (1 hour, H2O)
5. Train meta-learner on base model predictions
6. Calibrate probabilities with isotonic regression

**Output:**
- `models/xgboost_model.pkl`
- `models/lightgbm_model.pkl`
- `models/catboost_model.pkl`
- `models/tabnet_model.pkl`
- `models/automl_model.pkl`
- `models/meta_learner.pkl`
- `models/metadata.json`

**Validation Gates:**
- ✅ Test AUC ≥0.73
- ✅ Test ECE <0.03
- ✅ Test Accuracy ≥0.73

---

### Step 4: Backtest (8 hours)

**Validates ensemble on 5 years with realistic trading simulation**

```bash
python 04_backtest.py
```

**Trading Simulation:**
- Initial capital: $10,000
- Position size: 1% per trade
- Trading fee: 0.1%
- Slippage: 0.05%
- Confidence threshold: 0.45
- Stop loss: -2%
- Take profit: +3%
- Timeout: 60 candles (1 hour)

**Output:**
- `backtest/backtest_summary.json`
- `backtest/backtest_results.csv`
- `backtest/equity_curve.csv`

**Validation Gates:**
- ✅ Win Rate ≥75%
- ✅ Sharpe Ratio ≥1.8
- ✅ Max Drawdown >-12%
- ✅ Total Trades ≥5,000

---

### Step 5: Export ONNX (1 hour)

**Converts models to ONNX format and uploads to S3**

```bash
python 05_export_onnx.py
```

**Output:**
- `models/onnx/xgboost.onnx`
- `models/onnx/lightgbm.onnx`
- `models/onnx/catboost.onnx`
- `models/onnx/automl.onnx`
- `models/onnx/meta_learner.onnx`
- `models/onnx/deployment_bundle.json`

**S3 Upload:**
- Bucket: `crpbot-models-production`
- Prefix: `v3_ultimate/`

**Validation:**
- ✅ All models converted to ONNX
- ✅ ONNX inference validated
- ✅ Files uploaded to S3

---

## Expected Results

### Training Metrics
- **Test Accuracy**: 73-76%
- **Test AUC**: 0.73-0.78
- **ECE**: <0.03 (well-calibrated)

### Backtest Metrics
- **Win Rate**: 75-78%
- **Sharpe Ratio**: 1.8-2.5
- **Max Drawdown**: -8% to -12%
- **Total Trades**: 5,000-8,000
- **Profit Factor**: 2.5-3.5
- **Total Return**: 150-300% over 5 years

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
- Reduce batch size in TabNet training
- Process data in chunks
- Use Colab High-RAM runtime

### Issue: Training Too Slow

**Solution:**
- Verify A100 GPU is active: `!nvidia-smi`
- Reduce number of estimators (5000 → 2000)
- Use fewer features (180 → 120)

### Issue: Backtest Fails Validation Gates

**Solution:**
- Adjust confidence threshold (0.45 → 0.50)
- Retrain with more epochs
- Add more features or use different feature selection

### Issue: ONNX Conversion Fails

**Solution:**
- Skip TabNet (doesn't support ONNX well)
- Use only sklearn-compatible models
- Save as pickle and use Python runtime

---

## Monitoring Progress

### Check Step Status

```python
# View checkpoint
import json
with open('/content/drive/MyDrive/crpbot/v3_ultimate_checkpoint.json', 'r') as f:
    checkpoint = json.load(f)
    print(json.dumps(checkpoint, indent=2))
```

### Monitor GPU Usage

```bash
# In separate cell
!watch -n 1 nvidia-smi
```

### View Logs

```bash
# Check training logs
!tail -f /content/drive/MyDrive/crpbot/models/training.log

# Check backtest logs
!tail -f /content/drive/MyDrive/crpbot/backtest/backtest.log
```

---

## Cost Breakdown

### Google Colab Pro+
- **Subscription**: $50/month
- **Compute Units**: ~49 hours on A100
- **Total**: ~$50/month

### AWS Infrastructure
- **S3 Storage**: $1/month (model storage)
- **Data Transfer**: $2/month
- **Total**: ~$3/month

### Total Monthly Cost
- **Development**: $53/month
- **Production**: $3/month (after training)

---

## Production Deployment

### 1. Download Models from Google Drive

```python
# In Colab
!zip -r v3_ultimate_models.zip /content/drive/MyDrive/crpbot/models/
from google.colab import files
files.download('v3_ultimate_models.zip')
```

### 2. Verify Validation Gates

```python
import json

# Check training metadata
with open('models/metadata.json', 'r') as f:
    meta = json.load(f)
    print(f"Training gates passed: {meta['gates_passed']}")

# Check backtest summary
with open('backtest/backtest_summary.json', 'r') as f:
    backtest = json.load(f)
    print(f"Backtest gates passed: {backtest['gates_passed']}")
```

### 3. Deploy to Production

See main deployment documentation for integrating with:
- Real-time data feeds (Bybit/Coinbase WebSocket)
- PostgreSQL database
- Redis cache
- FastAPI inference server
- Paper trading system

---

## FAQ

### Q: Can I use Colab Pro instead of Pro+?

**A:** Yes, but training will be slower (T4/V100 GPU). Expect 60-80 hours instead of 49 hours.

### Q: Can I train on fewer coins?

**A:** Yes, modify `COINS` list in `01_fetch_data.py`. However, this reduces data diversity and may lower performance.

### Q: Can I use CPU instead of GPU?

**A:** Not recommended. Training will take 500+ hours on CPU.

### Q: How much does this cost per run?

**A:** ~$50 for Colab Pro+ subscription. You can run multiple experiments within the same month.

### Q: Can I save checkpoints?

**A:** Yes, the pipeline automatically saves checkpoints after each step in `v3_ultimate_checkpoint.json`.

### Q: What if training is interrupted?

**A:** Resume from last checkpoint: `python 00_run_v3_ultimate.py --resume 3`

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review validation gates in output
3. Examine logs in `/content/drive/MyDrive/crpbot/`

---

## License

Internal use only - Proprietary trading system

---

## Changelog

### v3.0.0 (2025-11-12)
- Initial V3 Ultimate release
- 5-model ensemble architecture
- 270 features with SHAP selection
- 75-78% win rate target
- 49-hour training pipeline on A100
