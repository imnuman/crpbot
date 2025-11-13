# Colab Pro Training - Complete Instructions

**Time**: ~15-20 minutes
**What**: Train BTC, ETH, SOL LSTM models on GPU

---

## 🚀 Quick Start (5 Steps)

### Step 1: Open Colab Pro (30 seconds)

1. Go to: **https://colab.research.google.com/**
2. Click **File** → **New notebook**
3. Click **Runtime** → **Change runtime type**
4. Select **GPU** (NOT CPU or TPU)
5. Click **Save**

✅ You should see "GPU" in the top right corner

---

### Step 2: Mount Google Drive (30 seconds)

**Create a new cell, paste this, and run:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click **"Connect to Google Drive"** when prompted.

✅ You should see: "Mounted at /content/drive"

---

### Step 3: Get Your AWS Credentials (1 minute)

**On your local machine terminal:**

```bash
cat ~/.aws/credentials
```

You'll see:
```
[default]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
```

**Copy both values** - you'll need them in Step 4.

---

### Step 4: Run Training Script (15 minutes)

**Create a new cell and paste the ENTIRE script below:**

👉 **Copy from `COLAB_TRAINING_SCRIPT.py`** or paste this:

```python
# ============================================================================
# CRPBot LSTM Training on Colab Pro GPU
# ============================================================================

print("📦 Installing dependencies...")
!pip install -q torch pandas pyarrow scikit-learn boto3 pyyaml tqdm

import os

# 🔑 REPLACE THESE WITH YOUR AWS CREDENTIALS FROM STEP 3
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA...'  # ← PASTE YOUR KEY HERE
os.environ['AWS_SECRET_ACCESS_KEY'] = '...'   # ← PASTE YOUR SECRET HERE
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

print("✅ AWS credentials configured")

# Clone repository
print("\n📂 Cloning repository...")
!git clone https://github.com/imnuman/crpbot.git
%cd crpbot
!pip install -q -e .

# Setup data from Google Drive
print("\n📊 Setting up training data...")
from pathlib import Path
import shutil

Path('data/features').mkdir(parents=True, exist_ok=True)

# Check Google Drive for data
drive_data = '/content/drive/MyDrive/crpbot/data/features'

if Path(drive_data).exists():
    print(f"✅ Found data in Google Drive")
    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        src = f'{drive_data}/features_{symbol}_1m_latest.parquet'
        dst = f'data/features/features_{symbol}_1m_latest.parquet'
        if Path(src).exists():
            shutil.copy(src, dst)
            print(f"  ✅ {symbol} copied from Drive")
else:
    # Download from S3 if not in Drive
    print("📥 Downloading from S3...")
    import boto3
    s3 = boto3.client('s3')
    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        s3.download_file(
            'crpbot-market-data-dev',
            f'data/features/features_{symbol}_1m_latest.parquet',
            f'data/features/features_{symbol}_1m_latest.parquet'
        )
        print(f"  ✅ {symbol} downloaded")

# Check GPU
print("\n🔍 Checking GPU...")
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("❌ No GPU detected!")
    device = "cpu"

# Train all 3 models
print("\n" + "="*70)
print("🚀 Training 3 LSTM Models on GPU")
print("="*70)

import time
start = time.time()

for symbol in ['BTC', 'ETH', 'SOL']:
    print(f"\n{'='*70}")
    print(f"Training {symbol}-USD")
    print(f"{'='*70}\n")

    !python apps/trainer/main.py --task lstm --coin {symbol} --epochs 15 --device {device}

total = time.time() - start
print(f"\n✅ All models trained in {total/60:.1f} minutes!")

# Upload to S3
print("\n📤 Uploading to S3...")
!aws s3 sync models/ s3://crpbot-market-data-dev/models/ --exclude "*" --include "*.pt"

# Backup to Google Drive
print("\n💾 Backing up to Google Drive...")
Path('/content/drive/MyDrive/crpbot/models').mkdir(parents=True, exist_ok=True)
!cp models/*.pt /content/drive/MyDrive/crpbot/models/

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"Time: {total/60:.1f} minutes")
print("Models in: S3 + Google Drive")
print("="*70)
```

**Before running:**
1. Replace `AWS_ACCESS_KEY_ID` with your key from Step 3
2. Replace `AWS_SECRET_ACCESS_KEY` with your secret from Step 3

**Then click the ▶️ play button** (or press Shift+Enter)

---

### Step 5: Monitor Progress (15 minutes)

You'll see output like:

```
📦 Installing dependencies...
✅ AWS credentials configured

📂 Cloning repository...
✅ Repository cloned

📊 Setting up training data...
  ✅ BTC-USD copied from Drive
  ✅ ETH-USD copied from Drive
  ✅ SOL-USD copied from Drive

🔍 Checking GPU...
✅ GPU: Tesla T4

======================================================================
🚀 Training 3 LSTM Models on GPU
======================================================================

======================================================================
Training BTC-USD
======================================================================

Epoch 1/15: 100%|██████████| 1000/1000 [00:15<00:00]
Train Loss: 0.6543, Val Loss: 0.6789, Val Acc: 52.3%

Epoch 2/15: 100%|██████████| 1000/1000 [00:14<00:00]
...

✅ All models trained in 12.3 minutes!

📤 Uploading to S3...
✅ Models uploaded

💾 Backing up to Google Drive...
✅ Backed up

🎉 TRAINING COMPLETE!
```

**Wait for "🎉 TRAINING COMPLETE!"**

---

## ✅ After Training Completes

### Tell Cloud Claude:

```
✅ Colab training complete!

Models uploaded to:
- S3: s3://crpbot-market-data-dev/models/
- Google Drive: MyDrive/crpbot/models/

Download and evaluate:
aws s3 sync s3://crpbot-market-data-dev/models/ models/
python scripts/evaluate_model.py --model models/lstm_BTC_USD_1m_*.pt --symbol BTC-USD
```

---

## 🚨 Troubleshooting

### "No GPU detected"
- **Fix**: Runtime → Change runtime type → GPU → Save → Runtime → Restart runtime

### "AWS credentials error"
- **Fix**: Check you pasted the correct keys (no quotes, no spaces)

### "Data not found in Google Drive"
- **Fix**: Script will auto-download from S3 (takes 2-3 min)

### "Repository clone failed"
- **Fix**: Run `!git clone https://github.com/imnuman/crpbot.git --depth 1`

---

## ⏱️ Expected Timeline

| Time | Task |
|------|------|
| 0:00 | Start script |
| 0:30 | Dependencies installed |
| 1:00 | Data loaded from Drive |
| 1:30 | BTC training starts |
| 5:30 | BTC done, ETH starts |
| 9:30 | ETH done, SOL starts |
| 13:30 | SOL done, uploading |
| 15:00 | **COMPLETE** ✅ |

---

## 📋 Quick Reference

| Step | Action | Time |
|------|--------|------|
| 1 | Open Colab, enable GPU | 30 sec |
| 2 | Mount Google Drive | 30 sec |
| 3 | Get AWS credentials | 1 min |
| 4 | Paste script, add credentials, run | 2 min |
| 5 | Wait for completion | 15 min |

**Total**: ~19 minutes

---

## 🎯 Success Indicators

✅ "GPU available: Tesla T4" (or similar)
✅ "All models trained in X minutes"
✅ "Models uploaded to S3"
✅ "Backed up to Google Drive"
✅ "🎉 TRAINING COMPLETE!"

---

**Ready? Open Colab and let's train!** 🚀

Current time: ~01:15 UTC
Deadline: 02:00 UTC
Time after training: ~45 minutes (plenty of buffer)
