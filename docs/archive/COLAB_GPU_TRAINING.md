# Colab Pro GPU Training - Ready to Execute

**Estimated Time**: 15 minutes
**Current Time**: ~01:05 UTC
**Completion**: ~01:20 UTC

---

## 🎯 Quick Start

1. Open Colab Pro: https://colab.research.google.com/
2. Create new notebook
3. Change runtime to **GPU**: Runtime → Change runtime type → GPU → Save
4. Copy the complete script below into a single cell
5. Run the cell
6. Wait 15 minutes
7. Models will be uploaded to S3 automatically

---

## 📝 Complete Colab Pro Training Script

Copy this entire block into a Colab cell:

```python
# ============================================================================
# CRPBot LSTM Training on Colab Pro GPU
# Estimated time: 15 minutes for 3 models (BTC, ETH, SOL)
# ============================================================================

# 1. Install dependencies
print("📦 Installing dependencies...")
!pip install -q torch pandas pyarrow scikit-learn boto3 pyyaml tqdm

# 2. Setup AWS credentials
import os

# ⚠️ REPLACE THESE WITH YOUR AWS CREDENTIALS
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_ACCESS_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

print("✅ AWS credentials configured")

# 3. Download features from S3
print("\n📥 Downloading features from S3...")
import boto3
from pathlib import Path

s3 = boto3.client('s3')
bucket = 'crpbot-market-data-dev'

# Create directories
Path('data/features').mkdir(parents=True, exist_ok=True)

# Download features for all 3 symbols
symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
for symbol in symbols:
    file_key = f'data/features/features_{symbol}_1m_latest.parquet'
    local_path = f'data/features/features_{symbol}_1m_latest.parquet'

    print(f"  Downloading {symbol}...")
    s3.download_file(bucket, file_key, local_path)

    # Get file size
    size_mb = Path(local_path).stat().st_size / (1024 * 1024)
    print(f"  ✅ {symbol}: {size_mb:.1f} MB")

print("✅ All features downloaded from S3")

# 4. Clone repository for training code
print("\n📂 Cloning repository...")
!git clone https://github.com/imnuman/crpbot.git
%cd crpbot

# Install project dependencies
!pip install -q -e .

print("✅ Repository cloned and installed")

# 5. Verify GPU availability
print("\n🔍 Checking GPU availability...")
import torch

if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ WARNING: No GPU detected! Training will be slow.")

# 6. Train all 3 LSTM models on GPU
print("\n" + "="*70)
print("🚀 Starting GPU Training for 3 LSTM Models")
print("="*70)

import time
start_time = time.time()

for symbol in ['BTC', 'ETH', 'SOL']:
    print(f"\n{'='*70}")
    print(f"Training {symbol}-USD LSTM Model on GPU")
    print(f"{'='*70}\n")

    model_start = time.time()

    # Train model
    !python apps/trainer/main.py \
        --task lstm \
        --coin {symbol} \
        --epochs 15 \
        --device cuda

    model_time = time.time() - model_start
    print(f"\n✅ {symbol}-USD training complete in {model_time/60:.1f} minutes!")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"✅ All 3 models trained in {total_time/60:.1f} minutes!")
print(f"{'='*70}\n")

# 7. List trained models
print("📋 Trained models:")
!ls -lh models/*.pt

# 8. Upload models to S3
print("\n📤 Uploading models to S3...")

# Create S3 sync command
import subprocess

# Sync models directory to S3
result = subprocess.run(
    ['aws', 's3', 'sync', 'models/', f's3://{bucket}/models/', '--exclude', '*', '--include', '*.pt'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ Models uploaded to S3 successfully!")
    print(result.stdout)
else:
    print("❌ Upload failed:")
    print(result.stderr)

# 9. Verify upload
print("\n🔍 Verifying S3 upload...")
result = subprocess.run(
    ['aws', 's3', 'ls', f's3://{bucket}/models/', '--recursive', '--human-readable'],
    capture_output=True,
    text=True
)
print(result.stdout)

# 10. Summary
print("\n" + "="*70)
print("🎉 COLAB PRO GPU TRAINING COMPLETE!")
print("="*70)
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Models trained: BTC-USD, ETH-USD, SOL-USD")
print(f"Models uploaded to: s3://{bucket}/models/")
print("\nNext steps:")
print("1. Download models on cloud server")
print("2. Evaluate against 68% promotion gate")
print("3. Train Transformer model")
print("="*70)
```

---

## 🔑 Before Running

**You need to add your AWS credentials** to the script above.

### Get AWS Credentials

On your local machine:

```bash
# Display AWS credentials
cat ~/.aws/credentials
```

You'll see something like:
```
[default]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
```

**Copy these values and replace in the Colab script:**
```python
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA...'  # ← Your key here
os.environ['AWS_SECRET_ACCESS_KEY'] = '...'  # ← Your secret here
```

---

## ⚡ Quick Execution Steps

1. **Open Colab Pro**: https://colab.research.google.com/
2. **New notebook**: File → New notebook
3. **Enable GPU**: Runtime → Change runtime type → GPU → Save
4. **Paste script**: Copy the entire Python block above into a cell
5. **Add credentials**: Replace AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
6. **Run**: Click the play button or Shift+Enter
7. **Monitor**: Watch the output (takes ~15 minutes)

---

## 📊 Expected Output

```
📦 Installing dependencies...
✅ AWS credentials configured

📥 Downloading features from S3...
  Downloading BTC-USD...
  ✅ BTC-USD: 209.0 MB
  Downloading ETH-USD...
  ✅ ETH-USD: 199.0 MB
  Downloading SOL-USD...
  ✅ SOL-USD: 184.0 MB
✅ All features downloaded from S3

📂 Cloning repository...
✅ Repository cloned and installed

🔍 Checking GPU availability...
✅ GPU available: Tesla T4
   Memory: 15.00 GB

======================================================================
🚀 Starting GPU Training for 3 LSTM Models
======================================================================

======================================================================
Training BTC-USD LSTM Model on GPU
======================================================================

Epoch 1/15: 100%|██████████| 1000/1000 [00:15<00:00, 64.5it/s]
...
✅ BTC-USD training complete in 4.2 minutes!

======================================================================
Training ETH-USD LSTM Model on GPU
======================================================================
...
✅ ETH-USD training complete in 4.3 minutes!

======================================================================
Training SOL-USD LSTM Model on GPU
======================================================================
...
✅ SOL-USD training complete in 4.1 minutes!

======================================================================
✅ All 3 models trained in 12.6 minutes!
======================================================================

📤 Uploading models to S3...
✅ Models uploaded to S3 successfully!

🎉 COLAB PRO GPU TRAINING COMPLETE!
```

---

## 🚨 Troubleshooting

### If GPU not available:
- Runtime → Change runtime type → GPU → Save
- Runtime → Restart runtime
- Try again

### If AWS credentials fail:
- Verify credentials are correct (no quotes, spaces)
- Check IAM permissions for S3 access

### If repository clone fails:
- Repository is public, should work
- Try: `!git clone https://github.com/imnuman/crpbot.git --depth 1`

### If training fails:
- Check error message
- Verify features downloaded correctly
- Check GPU memory usage

---

## ⏱️ Timeline

| Time | Task |
|------|------|
| 01:05 UTC | Start Colab notebook |
| 01:07 UTC | Dependencies installed, features downloaded |
| 01:08 UTC | BTC training starts |
| 01:12 UTC | BTC complete, ETH starts |
| 01:16 UTC | ETH complete, SOL starts |
| 01:20 UTC | SOL complete, uploading to S3 |
| 01:21 UTC | Upload complete, DONE ✅ |

**Total**: ~16 minutes

---

## 🎯 After Colab Completes

Tell Cloud Claude:
```
✅ Colab training complete!
Models uploaded to S3.

Next steps:
1. Download models from S3
2. Evaluate against 68% gate
3. Train Transformer
```

---

**Start the Colab notebook now!** ⏰

The script is complete and ready to run. Just add your AWS credentials and execute.
