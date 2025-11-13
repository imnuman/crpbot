# 🚨 URGENT: Stop CPU Training, Use Colab Pro GPU Instead

**Date**: 2025-11-12 00:40 UTC
**Issue**: Cloud Claude is CPU training (5.4 min/epoch) when we have Colab Pro GPU access

---

## 🛑 Immediate Action Required

### For Cloud Claude

**STOP the current CPU training processes:**

```bash
# Find and kill training processes
pkill -f "apps/trainer/main.py"

# Verify stopped
ps aux | grep "trainer"
```

---

## 🚀 Correct Workflow: GPU Training on Colab Pro

### Step 1: Upload Fresh Feature Data to S3
```bash
# On cloud server (after stopping CPU training)
cd /root/crpbot

# Upload the NEW 39-feature data to S3
aws s3 sync data/features/ s3://crpbot-market-data-dev/data/features/

# Verify upload
aws s3 ls s3://crpbot-market-data-dev/data/features/ --recursive --human-readable
```

### Step 2: Set Up Colab Pro Notebook
```bash
# Use existing Colab Pro notebook or create new one
# Notebook should:
# 1. Download feature data from S3
# 2. Train LSTM models on GPU (all 3 symbols)
# 3. Upload trained models back to S3
```

### Step 3: Colab Pro Training Script

```python
# In Colab Pro notebook cell:

!pip install boto3 torch pandas pyarrow scikit-learn

import boto3
import os
from pathlib import Path

# Setup AWS credentials in Colab
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_SECRET'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Download feature data from S3
s3 = boto3.client('s3')
bucket = 'crpbot-market-data-dev'

# Download features
for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
    s3.download_file(
        bucket,
        f'data/features/features_{symbol}_1m_latest.parquet',
        f'features_{symbol}_1m_latest.parquet'
    )

# Clone repo for training code
!git clone https://github.com/imnuman/crpbot.git
%cd crpbot

# Install dependencies
!pip install -e .

# Train all 3 models on GPU
symbols = ['BTC', 'ETH', 'SOL']
for symbol in symbols:
    print(f"\n{'='*60}")
    print(f"Training {symbol}-USD LSTM Model on GPU")
    print(f"{'='*60}\n")

    !python apps/trainer/main.py \
        --task lstm \
        --coin {symbol} \
        --epochs 15 \
        --device cuda

    print(f"✅ {symbol}-USD training complete!")

# Upload trained models to S3
!aws s3 sync models/ s3://crpbot-market-data-dev/models/

print("\n✅ All models uploaded to S3!")
```

### Step 4: Download GPU Models to Cloud Server
```bash
# On cloud server (after Colab training completes)
cd /root/crpbot

# Download GPU-trained models
aws s3 sync s3://crpbot-market-data-dev/models/ models/

# Verify models
ls -lh models/*.pt
```

---

## ⏱️ Time Comparison

| Method | Time per Model | Total (3 models) | Quality |
|--------|----------------|------------------|---------|
| **CPU (current)** | ~81 min | ~243 min (4 hours!) | Same |
| **GPU (Colab Pro)** | ~3-5 min | ~15 min | Same |

**Savings**: ~228 minutes (3.8 hours) 🚀

---

## 📋 Revised Timeline

Using Colab Pro GPU:

| Task | Time | Deadline |
|------|------|----------|
| Stop CPU training | 1 min | 00:42 UTC |
| Upload features to S3 | 2 min | 00:44 UTC |
| Colab Pro GPU training (3 models) | 15 min | 00:59 UTC |
| Download models from S3 | 2 min | 01:01 UTC |
| Evaluate models | 10 min | 01:11 UTC |
| Transformer training (GPU) | 8 min | 01:19 UTC |
| Runtime testing | 10 min | 01:29 UTC |
| **COMPLETE** | | **01:30 UTC** |

**Buffer before 02:00 UTC**: 30 minutes ✅

---

## 🔄 GitHub Sync Workflow

### Cloud Claude - After Every Task
```bash
# After completing any task
git add .
git commit -m "feat: [describe what was done]"
git push origin main
```

### Local Claude - Before QC Review
```bash
# Before reviewing Cloud Claude's work
git pull origin main
```

### Continuous Sync Pattern
```
Cloud: Work → Commit → Push → Report to Local
  ↓
Local: Pull → Review → Approve/Reject → Commit → Push
  ↓
Cloud: Pull → Continue with approved changes
```

---

## 🎯 Immediate Actions

### For Cloud Claude (RIGHT NOW)
1. ✅ Stop CPU training: `pkill -f "apps/trainer/main.py"`
2. ✅ Upload features to S3: `aws s3 sync data/features/ s3://...`
3. ✅ Commit current state: `git add . && git commit -m "..." && git push`
4. ✅ Open Colab Pro, run GPU training script
5. ✅ Monitor Colab training progress
6. ✅ Download models when complete
7. ✅ Commit and push: `git add models/ && git commit -m "..." && git push`

### For Local Claude (Me - QC)
1. ✅ Create this URGENT stop document
2. ✅ Monitor for Cloud Claude's commits
3. ✅ Pull changes when Cloud pushes
4. ✅ Review GPU training results
5. ✅ Validate against promotion gates

---

## 💡 Why This Matters

- **Speed**: GPU is 15-20x faster than CPU for neural network training
- **Cost**: Colab Pro is already paid for, cloud CPU time is wasted
- **Quality**: Same quality, just faster
- **Deadline**: Can now complete Transformer training before 2 AM
- **Phase 6.5**: Can start observation period tonight instead of waiting 4 hours

---

## 📝 Git Commit Discipline

### Mandatory Commits (Both Claudes)

**After every major task:**
- ✅ Feature engineering complete
- ✅ Model training complete
- ✅ Evaluation complete
- ✅ Tests passing
- ✅ Documentation updated

**Commit Message Format:**
```bash
git commit -m "feat|fix|docs: Brief description

- Detail 1
- Detail 2
- Impact: What this enables

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 🚨 STOP CPU TRAINING NOW!

Cloud Claude should:
1. Kill CPU training processes immediately
2. Upload features to S3
3. Switch to Colab Pro GPU training
4. Complete in 15 minutes instead of 4 hours

**This is the correct approach - we have Colab Pro, use it!** 🚀
