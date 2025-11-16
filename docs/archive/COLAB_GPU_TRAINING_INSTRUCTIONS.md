# GPU Training on Google Colab Pro - Proper Features

## Quick Summary

The previous GPU training on Colab used only 5 simple features and resulted in 24% accuracy. This guide shows how to retrain with your production 58-column multi-timeframe features.

## Files You Need

1. **Feature files** (from `data/features/`):
   - `features_BTC-USD_1m_2025-11-10.parquet` (239 MB)
   - `features_ETH-USD_1m_2025-11-10.parquet` (228 MB)
   - `features_SOL-USD_1m_2025-11-10.parquet` (207 MB)
   - Optional: `features_ADA-USD_1m_2025-11-10.parquet` if available

2. **Training script**: `train_gpu_proper.py`

## Step-by-Step Instructions

### 1. Upload Files to Google Drive

```bash
# Create folder in Google Drive
mkdir -p ~/GoogleDrive/crpbot_training/data/features

# Copy feature files
cp data/features/features_*-USD_1m_2025-11-10.parquet ~/GoogleDrive/crpbot_training/data/features/

# Copy training script
cp train_gpu_proper.py ~/GoogleDrive/crpbot_training/
```

**OR** use Google Drive web interface to upload these files.

### 2. Open Google Colab Pro

1. Go to https://colab.research.google.com/
2. Create new notebook
3. Go to **Runtime > Change runtime type**
4. Select **GPU** (T4, V100, or A100)
5. Click **Save**

### 3. Mount Google Drive in Colab

Run this in first cell:
```python
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your training folder
import os
os.chdir('/content/drive/MyDrive/crpbot_training')
```

### 4. Install Dependencies

```python
!pip install torch pandas pyarrow loguru
```

### 5. Run Training

```python
!python train_gpu_proper.py
```

### 6. Expected Output

```
🚀 GPU Model Training - Production Features
============================================================
🖥️  Using device: cuda
   GPU: Tesla T4
   Memory: 15.36 GB

============================================================
🔥 Training BTC-USD
============================================================
📂 Loading data/features/features_BTC-USD_1m_latest.parquet
   Loaded 1,030,512 rows, 58 columns
   Using 50 features
📊 Data split:
   Train: 721,358 rows
   Val: 154,576 rows
🔧 Normalizing features...
📦 Creating datasets...
   Train sequences: 721,298 rows
   Val sequences: 154,516 rows
🧠 Creating model...
   Parameters: 57,091
🚀 Starting training...
Epoch 1/15: Train Loss=0.9876, Train Acc=45.2%, Val Loss=0.9234, Val Acc=48.1%
Epoch 2/15: Train Loss=0.9123, Train Acc=52.3%, Val Loss=0.8765, Val Acc=54.2%
...
Epoch 15/15: Train Loss=0.7234, Train Acc=68.5%, Val Loss=0.7456, Val Acc=67.2%

✅ Training complete!
   Best validation accuracy: 69.1%
   Duration: 621.3s (10.4 min)
💾 Model saved: models/gpu_trained_proper/BTC_USD_lstm_model.pt
```

**Expected time**:
- T4 GPU: ~10-12 minutes per model
- V100 GPU: ~6-8 minutes per model
- A100 GPU: ~4-6 minutes per model

**Total for 4 models**: 40-50 minutes

### 7. Download Trained Models

After training completes:

```python
# Zip the models
!zip -r gpu_trained_proper.zip models/gpu_trained_proper/

# Download using Colab
from google.colab import files
files.download('gpu_trained_proper.zip')
```

### 8. Upload to Your Server

On your server:
```bash
# Extract models
unzip gpu_trained_proper.zip

# Test models
python backtest_gpu_models.py  # Update to use new models folder
```

## Alternative: Use Colab Notebook Interface

Create `colab_train.ipynb` with these cells:

**Cell 1: Setup**
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/crpbot_training')

!pip install -q torch pandas pyarrow loguru
```

**Cell 2: Run Training**
```python
!python train_gpu_proper.py
```

**Cell 3: Download Models**
```python
!zip -r gpu_trained_proper.zip models/gpu_trained_proper/
from google.colab import files
files.download('gpu_trained_proper.zip')
```

## Troubleshooting

### Issue: "Feature file not found"
- Make sure you uploaded the feature files to the correct folder
- Check the symlinks work or specify exact filename

### Issue: "CUDA out of memory"
- Reduce batch_size from 32 to 16 or 8
- Edit in `train_gpu_proper.py` line 276

### Issue: "Colab disconnects"
- Colab Pro has longer runtimes
- Add keep-alive code:
  ```python
  from IPython.display import display, Javascript
  display(Javascript('''
   function ClickConnect(){
     console.log("Clicked connect");
     document.querySelector("colab-connect-button").click()
   }
   setInterval(ClickConnect, 60000)
  '''))
  ```

## What's Different from Previous GPU Training?

| Aspect | Previous (Colab) | New (Proper) |
|--------|------------------|--------------|
| **Features** | 5 simple (returns, volume, etc) | 58 multi-timeframe |
| **Data** | Raw OHLCV only | Full feature engineering |
| **Accuracy** | 24% (worse than random) | Expected: 65-70% |
| **Model Size** | 204KB | ~250KB |
| **Training Source** | Hand-crafted features | Production pipeline |

## Success Criteria

After retraining, you should see:
- ✅ Validation accuracy ≥68%
- ✅ Models match production features
- ✅ Backtest win rate >55%
- ✅ Sharpe ratio >1.5

## Next Steps After Training

1. Extract models: `unzip gpu_trained_proper.zip`
2. Update backtest script to use new folder
3. Run backtest: `python backtest_gpu_models.py`
4. If models pass (≥68%): Deploy to production
5. If models fail: Need more data or different architecture

---

**Questions?** Check the training script output for detailed progress. All metrics are logged during training.
