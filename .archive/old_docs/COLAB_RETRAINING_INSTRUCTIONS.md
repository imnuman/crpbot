# Colab Retraining Instructions (31 Features Fix)

**Date**: 2025-11-13
**Issue**: Feature mismatch (50 vs 31) preventing model evaluation
**Solution**: Retrain with correct 31-feature parquet files

---

## Step 1: Download Feature Files from Server

Download the 3 feature parquet files from your server to your local machine:

```bash
# From your local machine (not server):
scp root@your-server:/root/crpbot/data/features/features_BTC-USD_1m_2025-11-13.parquet ~/Downloads/
scp root@your-server:/root/crpbot/data/features/features_ETH-USD_1m_2025-11-13.parquet ~/Downloads/
scp root@your-server:/root/crpbot/data/features/features_SOL-USD_1m_2025-11-13.parquet ~/Downloads/
```

**Files to download**:
- `features_BTC-USD_1m_2025-11-13.parquet` (210 MB)
- `features_ETH-USD_1m_2025-11-13.parquet` (200 MB)
- `features_SOL-USD_1m_2025-11-13.parquet` (184 MB)

---

## Step 2: Upload to Google Drive

1. Open Google Drive in your browser
2. Create a folder: `CRPBot/features/`
3. Upload all 3 parquet files to this folder
4. Wait for upload to complete (~5-10 minutes depending on connection)

---

## Step 3: Open Colab Pro Notebook

1. Go to https://colab.research.google.com/
2. Create a new notebook: **File → New notebook**
3. Enable GPU: **Runtime → Change runtime type → GPU → T4**
4. Verify GPU: Run this cell:
   ```python
   !nvidia-smi
   ```

---

## Step 4: Mount Google Drive

Run this cell in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the link, authorize, and paste the code.

---

## Step 5: Install Dependencies

Run this cell:

```python
!pip install torch pandas numpy loguru pyarrow ta scikit-learn tqdm
```

---

## Step 6: Clone Repository and Setup

Run these cells:

```python
# Clone repository
!git clone https://github.com/imnuman/crpbot.git
%cd crpbot

# Pull latest changes (includes fixes)
!git pull origin main

# Verify we're on latest commit
!git log -1 --oneline
# Should show: "docs: add critical feature mismatch report (50 vs 31 features)"
```

---

## Step 7: Copy Feature Files from Drive

Run this cell:

```python
import shutil
from pathlib import Path

# Create feature directory
Path("data/features").mkdir(parents=True, exist_ok=True)

# Copy files from Google Drive
drive_path = "/content/drive/MyDrive/CRPBot/features/"
local_path = "data/features/"

files = [
    "features_BTC-USD_1m_2025-11-13.parquet",
    "features_ETH-USD_1m_2025-11-13.parquet",
    "features_SOL-USD_1m_2025-11-13.parquet"
]

for file in files:
    src = drive_path + file
    dst = local_path + file
    print(f"Copying {file}...")
    shutil.copy(src, dst)
    print(f"✅ Copied to {dst}")

# Create symlinks for "latest" version
for file in files:
    symbol = file.split("_")[1]  # Extract BTC-USD, ETH-USD, SOL-USD
    src = file
    dst = f"features_{symbol}_1m_latest.parquet"
    dst_path = Path(local_path) / dst
    if dst_path.exists():
        dst_path.unlink()
    (Path(local_path) / dst).symlink_to(src)
    print(f"✅ Created symlink: {dst} -> {src}")

print("\n✅ All feature files ready!")
```

---

## Step 8: Verify Feature Count (CRITICAL!)

**THIS IS THE MOST IMPORTANT STEP**

Run this cell to verify we have exactly **31 features**:

```python
import pandas as pd

# Load one file to check feature count
df = pd.read_parquet("data/features/features_BTC-USD_1m_latest.parquet")

print(f"Total columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}\n")

# Excluded columns (non-features)
exclude_cols = ["timestamp", "open", "high", "low", "close", "volume", "session", "volatility_regime"]
feature_columns = [col for col in df.columns if col not in exclude_cols]

print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

# CRITICAL CHECK
assert len(feature_columns) == 31, f"❌ ERROR: Expected 31 features, got {len(feature_columns)}!"
print(f"\n✅ VERIFIED: {len(feature_columns)} features (correct!)")
```

**Expected output**:
```
Total columns: 39
Feature columns (31): ['session_tokyo', 'session_london', 'session_new_york', ...]

✅ VERIFIED: 31 features (correct!)
```

**If you get any other number, STOP and debug before proceeding!**

---

## Step 9: Train BTC-USD Model

Run this cell (~19 minutes):

```python
!python apps/trainer/main.py --task lstm --coin BTC --epochs 50
```

**Expected output**:
```
Creating LSTM model...
Model parameters: 1,XXX,XXX  (should be ~1M+ params, much larger than old 62K)
Input size: 31  ← CRITICAL: Must be 31!
...
Training for 50 epochs with improved configuration...
✅ Training complete! Best validation accuracy: 0.XXXX
```

**Watch for**:
- Input size: **31** (not 50!)
- Model parameters: **~1M+** (not 62K)
- Training completes without errors

---

## Step 10: Train ETH-USD Model

Run this cell (~19 minutes):

```python
!python apps/trainer/main.py --task lstm --coin ETH --epochs 50
```

---

## Step 11: Train SOL-USD Model

Run this cell (~19 minutes):

```python
!python apps/trainer/main.py --task lstm --coin SOL --epochs 50
```

**Total training time**: ~57 minutes for all 3 models

---

## Step 12: Verify Model Input Size

**CRITICAL CHECK** before uploading models:

```python
import torch

# Check BTC model
checkpoint = torch.load("models/lstm_BTC_USD_1m_*.pt", map_location="cpu", weights_only=False)
weight_ih = checkpoint['model_state_dict']['lstm.weight_ih_l0']
input_size = weight_ih.shape[1]

print(f"Model input size: {input_size}")
assert input_size == 31, f"❌ ERROR: Model has {input_size} features, expected 31!"
print("✅ Model has correct input size (31 features)")
```

**Expected output**:
```
Model input size: 31
✅ Model has correct input size (31 features)
```

**If you see 50 or any other number, DO NOT UPLOAD! Debug first.**

---

## Step 13: Upload Models to S3

Run this cell:

```python
# Install AWS CLI
!pip install awscli boto3

# Configure AWS credentials (use your credentials)
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Upload to S3
!aws s3 sync models/ s3://crpbot-models/ --exclude "*.gitkeep" --include "lstm_*_USD_1m_*.pt"

print("✅ Models uploaded to S3!")
```

---

## Step 14: Download Models on Server

Back on your server, run:

```bash
cd /root/crpbot

# Create retrained directory
mkdir -p models/retrained

# Download from S3
aws s3 sync s3://crpbot-models/ models/retrained/ --exclude "*" --include "lstm_*_USD_1m_*.pt"

# List downloaded models
ls -lh models/retrained/*.pt
```

---

## Step 15: Evaluate Retrained Models

Run evaluations on your server:

```bash
# BTC-USD
python scripts/evaluate_model.py \
  --model models/retrained/lstm_BTC_USD_1m_*.pt \
  --symbol BTC-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05

# ETH-USD
python scripts/evaluate_model.py \
  --model models/retrained/lstm_ETH_USD_1m_*.pt \
  --symbol ETH-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05

# SOL-USD
python scripts/evaluate_model.py \
  --model models/retrained/lstm_SOL_USD_1m_*.pt \
  --symbol SOL-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05
```

**Expected**: Models should load successfully and evaluate (no shape mismatch errors!)

---

## Troubleshooting

### Problem: "FileNotFoundError: Feature file not found"
**Solution**: Check symlinks in Step 7. The evaluation script expects `features_*_1m_latest.parquet`.

### Problem: "RuntimeError: size mismatch ... [512, 50] vs [512, 31]"
**Solution**: You trained with wrong feature files! The Colab environment still has 50-feature files. Delete them and re-run Step 7.

### Problem: "AssertionError: Expected 31 features, got XX"
**Solution**: Feature files are wrong. Re-download from server (Step 1) and verify locally before uploading.

### Problem: Training runs but validation accuracy is ~50% (random)
**Solution**: This is normal for early epochs. Monitor final epochs - should improve to 52-55% or higher.

---

## Success Criteria

✅ **Feature verification**: 31 features confirmed in Step 8
✅ **Training**: All 3 models train without errors
✅ **Model verification**: Input size = 31 confirmed in Step 12
✅ **Upload**: Models successfully uploaded to S3
✅ **Evaluation**: Models load and evaluate without shape mismatch errors
✅ **Promotion gates**: At least 1 model achieves ≥68% win rate

---

## Next Steps After Successful Retraining

1. **Evaluate promotion gates**: Check if models pass 68% win rate threshold
2. **Promote models**: Copy passing models to `models/promoted/`
3. **Update runtime**: Integrate promoted models into runtime ensemble
4. **Phase 6.5 observation**: Start 3-5 day dry-run observation period
5. **Phase 7**: Micro-lot testing on FTMO account

---

**Reference Documents**:
- Problem report: `reports/phase6_5/CRITICAL_FEATURE_MISMATCH_REPORT.md`
- Feature engineering: `scripts/engineer_features.py`
- Training code: `apps/trainer/train/train_lstm.py`

**Commit**: e25b970 (docs: add critical feature mismatch report)
