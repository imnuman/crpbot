# V3 Ultimate - Quick Start Checklist

## Prerequisites

- [ ] Subscribe to **Google Colab Pro+** ($50/month)
  - Link: https://colab.research.google.com/signup

- [ ] Ensure you have **~100GB free** in Google Drive

## Step-by-Step Setup (15 minutes)

### 1. Upload Files to Google Drive

- [ ] Navigate to: `My Drive > colab notebooks`
- [ ] Create folder: `crpbot`
- [ ] Inside `crpbot`, create folder: `v3_ultimate`
- [ ] Upload these 6 Python files to `crpbot/v3_ultimate/`:
  - [ ] `00_run_v3_ultimate.py`
  - [ ] `01_fetch_data.py`
  - [ ] `02_engineer_features.py`
  - [ ] `03_train_ensemble.py`
  - [ ] `04_backtest.py`
  - [ ] `05_export_onnx.py`

**OR** upload the Colab notebook:
- [ ] Upload `V3_Ultimate_Colab.ipynb` to `My Drive`

### 2. Open Google Colab

- [ ] Go to: https://colab.research.google.com/
- [ ] Upload `V3_Ultimate_Colab.ipynb` **OR** create new notebook

### 3. Select A100 GPU

- [ ] Click: `Runtime > Change runtime type`
- [ ] Select: `A100 GPU`
- [ ] Click: `Save`

### 4. Run Setup (in notebook cells)

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

- [ ] Run Cell 1, authorize Google Drive access

```python
# Cell 2: Check GPU
!nvidia-smi
```

- [ ] Run Cell 2, verify A100 GPU is active

```python
# Cell 3: Install dependencies
!pip install -q xgboost lightgbm catboost pytorch-tabnet
!pip install -q pandas numpy pyarrow scikit-learn
!pip install -q onnx onnxruntime skl2onnx
!pip install -q boto3 ccxt tqdm joblib shap torch
```

- [ ] Run Cell 3, wait for installations (~5 minutes)

### 5. Run V3 Ultimate Pipeline

**Option A: Run all steps at once (49 hours)**

```python
%cd /content/drive/MyDrive/crpbot/v3_ultimate
!python 00_run_v3_ultimate.py
```

- [ ] Run command above
- [ ] Keep Colab tab open (or use Colab Pro+ background execution)
- [ ] Check back in ~49 hours

**Option B: Run steps individually (recommended)**

```python
# Step 1: Data Collection (12 hours)
%cd /content/drive/MyDrive/crpbot/v3_ultimate
!python 01_fetch_data.py
```

- [ ] Run Step 1, wait 12 hours
- [ ] Verify: Check `data/raw/` has 60 .parquet files

```python
# Step 2: Feature Engineering (4 hours)
!python 02_engineer_features.py
```

- [ ] Run Step 2, wait 4 hours
- [ ] Verify: Check `data/features/` has 60 .parquet files

```python
# Step 3: Train Ensemble (24 hours)
!python 03_train_ensemble.py
```

- [ ] Run Step 3, wait 24 hours
- [ ] Verify: Check `models/` has 6 .pkl files + metadata.json

```python
# Step 4: Backtest (8 hours)
!python 04_backtest.py
```

- [ ] Run Step 4, wait 8 hours
- [ ] Verify: Check `backtest/` has backtest_summary.json

```python
# Step 5: Export ONNX (1 hour)
!python 05_export_onnx.py
```

- [ ] Run Step 5, wait 1 hour
- [ ] Verify: Check `models/onnx/` has 5 .onnx files

## Validation Checklist

### After Step 3 (Training)

```python
# Check training results
import json
with open('/content/drive/MyDrive/crpbot/models/metadata.json', 'r') as f:
    meta = json.load(f)
    print(json.dumps(meta['metrics'], indent=2))
```

- [ ] **Test Accuracy ≥0.73**: ______%
- [ ] **Test AUC ≥0.73**: ______
- [ ] **ECE <0.03**: ______
- [ ] **Training gates passed**: Yes / No

### After Step 4 (Backtest)

```python
# Check backtest results
import json
with open('/content/drive/MyDrive/crpbot/backtest/backtest_summary.json', 'r') as f:
    backtest = json.load(f)
    print(json.dumps(backtest['metrics'], indent=2))
```

- [ ] **Win Rate ≥75%**: ______%
- [ ] **Sharpe Ratio ≥1.8**: ______
- [ ] **Max Drawdown >-12%**: ______%
- [ ] **Total Trades ≥5,000**: ______
- [ ] **Backtest gates passed**: Yes / No

## Download Results

```python
# Zip and download
!zip -r v3_ultimate_complete.zip /content/drive/MyDrive/crpbot/models/ /content/drive/MyDrive/crpbot/backtest/

from google.colab import files
files.download('v3_ultimate_complete.zip')
```

- [ ] Download complete package
- [ ] Extract locally
- [ ] Verify all files present:
  - [ ] `models/*.pkl` (6 files)
  - [ ] `models/metadata.json`
  - [ ] `models/onnx/*.onnx` (5 files)
  - [ ] `backtest/backtest_summary.json`
  - [ ] `backtest/backtest_results.csv`

## Expected Deliverables

After successful completion, you should have:

### Models (6 files)
- [ ] `xgboost_model.pkl` (~100MB)
- [ ] `lightgbm_model.pkl` (~50MB)
- [ ] `catboost_model.pkl` (~80MB)
- [ ] `tabnet_model.pkl` (~20MB)
- [ ] `automl_model.pkl` (~60MB)
- [ ] `meta_learner.pkl` (~5MB)

### ONNX Models (5 files)
- [ ] `xgboost.onnx`
- [ ] `lightgbm.onnx`
- [ ] `catboost.onnx`
- [ ] `automl.onnx`
- [ ] `meta_learner.onnx`

### Metadata & Results
- [ ] `models/metadata.json` - Training metrics
- [ ] `backtest/backtest_summary.json` - Backtest metrics
- [ ] `backtest/backtest_results.csv` - All trades
- [ ] `backtest/equity_curve.csv` - Capital over time

### Data Files (for reference)
- [ ] `data/raw/*.parquet` (60 files, ~10GB)
- [ ] `data/features/*.parquet` (60 files, ~30GB)

## Troubleshooting

### Colab disconnects during training

**Solution:**
- Colab Pro+ allows background execution
- Or: Use resume feature: `!python 00_run_v3_ultimate.py --resume 3`

### Out of Memory error

**Solution:**
- Restart runtime: `Runtime > Factory reset runtime`
- Clear outputs: `Edit > Clear all outputs`
- Reduce batch size in `03_train_ensemble.py` (line 240)

### GPU not A100

**Solution:**
- Disconnect: `Runtime > Disconnect and delete runtime`
- Wait 5 minutes
- Reconnect and select A100 again

### Dependencies installation fails

**Solution:**
```python
# Reinstall all dependencies
!pip install --upgrade pip
!pip install --force-reinstall xgboost lightgbm catboost pytorch-tabnet
```

### "No module named 'talib'" error

**Solution:**
```python
# Install TA-Lib from source
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install
%cd ..
!pip install ta-lib
```

## Next Steps After Completion

1. [ ] Verify validation gates passed
2. [ ] Review backtest metrics
3. [ ] Download all models and results
4. [ ] Deploy to production infrastructure
5. [ ] Configure real-time data feeds
6. [ ] Start paper trading
7. [ ] Monitor live performance
8. [ ] Retrain monthly with new data

## Timeline

| Step | Name | Duration | Cumulative |
|------|------|----------|------------|
| 1 | Data Collection | 12h | 12h |
| 2 | Feature Engineering | 4h | 16h |
| 3 | Train Ensemble | 24h | 40h |
| 4 | Backtest | 8h | 48h |
| 5 | Export ONNX | 1h | 49h |

**Total: ~49 hours on Colab Pro+ A100**

## Cost Summary

- **Colab Pro+ subscription**: $50/month (includes compute)
- **Google Drive storage**: Free (first 15GB)
- **Total**: $50/month

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs in Google Drive
3. Check checkpoint status
4. Verify GPU is A100
5. Ensure sufficient Drive storage

---

**Ready to start?** Follow the checklist step-by-step and you'll have production-ready models in ~49 hours!
