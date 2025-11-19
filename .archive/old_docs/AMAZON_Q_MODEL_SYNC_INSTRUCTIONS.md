# Amazon Q: V5 Model Sync Instructions

**Date**: 2025-11-15 18:05 EST
**Priority**: 🔴 CRITICAL
**Requestor**: Builder Claude
**For**: QC Claude and Builder Claude model validation

---

## 🚨 CRITICAL ISSUE

The V5 model files currently in `models/v5/` are **incomplete**. They only contain training metadata (accuracy, epoch) but are **missing the actual model weights** (`model_state_dict`).

**Current V5 files** (908-916 bytes):
```
models/v5/lstm_BTC-USD_1m_v5.pt     ❌ Only metadata
models/v5/lstm_ETH-USD_1m_v5.pt     ❌ Only metadata
models/v5/lstm_SOL-USD_1m_v5.pt     ❌ Only metadata
models/v5/transformer_multi_v5.pt   ❌ Only metadata
```

**What's missing**: `model_state_dict` with trained weights (~200-500 KB per file)

---

## 🎯 MISSION: Download Full V5 Models

### Step 1: Check GPU Instance Status

```bash
# Check if GPU instance is still running
aws ec2 describe-instances \
  --instance-ids i-XXXXXXXXX \
  --query 'Reservations[0].Instances[0].State.Name'
```

**Expected outputs**:
- `"running"` → ✅ Instance alive, proceed to Step 2
- `"stopped"` → Start instance first, then proceed to Step 2
- `"terminated"` → ⚠️ Models lost, skip to Step 4 (Alternative Plan)

---

### Step 2: Download Full V5 Models from GPU Instance

**IF INSTANCE IS RUNNING OR STOPPED**:

```bash
# 1. Start instance if stopped
aws ec2 start-instances --instance-ids i-XXXXXXXXX

# 2. Wait for instance to be running (if you started it)
aws ec2 wait instance-running --instance-ids i-XXXXXXXXX

# 3. Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids i-XXXXXXXXX \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance IP: $INSTANCE_IP"

# 4. Check what model files exist on the instance
ssh -i ~/.ssh/crpbot-training.pem ubuntu@$INSTANCE_IP \
  "ls -lh ~/crpbot/models/v5/"

# 5. Download FULL model files (with state_dict)
# The training script should have saved full checkpoints somewhere
# Check these locations:
ssh -i ~/.ssh/crpbot-training.pem ubuntu@$INSTANCE_IP \
  "find ~/crpbot -name '*.pt' -size +100k -ls"

# 6. Download the FULL model files
# Replace the source path based on what you found in step 5
rsync -avz --progress \
  -e "ssh -i ~/.ssh/crpbot-training.pem" \
  ubuntu@$INSTANCE_IP:~/crpbot/models/v5/*.pt \
  models/v5_full/

# 7. Verify downloaded files have state_dict
uv run python -c "
import torch
import sys
checkpoint = torch.load('models/v5_full/lstm_BTC-USD_1m_v5.pt', map_location='cpu')
print('Keys in checkpoint:', list(checkpoint.keys()))
if 'model_state_dict' not in checkpoint:
    print('❌ ERROR: model_state_dict not found!')
    sys.exit(1)
else:
    print('✅ SUCCESS: Full model checkpoint!')
"

# 8. Terminate instance (cost control)
aws ec2 terminate-instances --instance-ids i-XXXXXXXXX
```

---

### Step 3: Replace Incomplete V5 Models

```bash
# Backup incomplete models
mkdir -p models/v5_metadata_only
mv models/v5/*.pt models/v5_metadata_only/

# Move full models to v5 directory
mv models/v5_full/*.pt models/v5/

# Verify file sizes (should be ~200-500 KB each)
ls -lh models/v5/

# Expected output:
# lstm_BTC-USD_1m_v5.pt     (~200-500 KB) ✅
# lstm_ETH-USD_1m_v5.pt     (~200-500 KB) ✅
# lstm_SOL-USD_1m_v5.pt     (~200-500 KB) ✅
# transformer_multi_v5.pt   (~200-500 KB) ✅
```

---

### Step 4: Alternative Plan (If Instance Terminated)

**IF GPU INSTANCE IS TERMINATED** (models lost):

You have **TWO OPTIONS**:

#### Option A: Use Existing Complete Models (FASTEST)
```bash
# We have complete models from Nov 12 in gpu_trained_proper/
# Copy these to promoted/ for immediate use

cp models/gpu_trained_proper/BTC_lstm_model.pt models/promoted/lstm_BTC-USD_1m_backup.pt
cp models/gpu_trained_proper/ETH_lstm_model.pt models/promoted/lstm_ETH-USD_1m_backup.pt
cp models/gpu_trained_proper/SOL_lstm_model.pt models/promoted/lstm_SOL-USD_1m_backup.pt

# Builder Claude will evaluate these and go live tonight
```

#### Option B: Retrain V5 Models with Full Checkpoint Saving
```bash
# 1. Spin up new GPU instance (use existing script)
# 2. Modify training script to save FULL checkpoints:

# In the training script, change:
torch.save({
    'accuracy': test_accuracy,
    'epoch': epoch
}, model_path)

# To:
torch.save({
    'model_state_dict': model.state_dict(),  # ← ADD THIS!
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': test_accuracy,
    'epoch': epoch,
    'loss': train_loss
}, model_path)

# 3. Re-run training (28 minutes, $0.53)
# 4. Download FULL models
# 5. Sync to GitHub
```

---

### Step 5: Sync to GitHub

```bash
# Add full V5 models to git
git add models/v5/*.pt

# Commit with clear message
git commit -m "feat: add FULL V5 model checkpoints with state_dict

- BTC-USD LSTM: 74.0% accuracy (with weights)
- ETH-USD LSTM: 70.6% accuracy (with weights)
- SOL-USD LSTM: 72.1% accuracy (with weights)
- Transformer: 63.4% accuracy (with weights)

Previous commit had metadata-only files (908 bytes)
These files now include full model_state_dict (~200-500 KB each)

Ready for Builder Claude and QC Claude evaluation and deployment!"

# Push to GitHub
git push origin main

# Confirm sync
echo "✅ Full V5 models synced to GitHub!"
echo "✅ QC Claude and Builder Claude can now access complete models"
```

---

### Step 6: Notify Claude Instances

Create a status file for visibility:

```bash
cat > V5_MODELS_READY.md << 'EOF'
# ✅ V5 Models Ready for Deployment

**Date**: 2025-11-15 18:XX EST
**Status**: COMPLETE - Full models synced

## 📁 Model Files

All V5 models now include complete `model_state_dict`:

```
models/v5/
├── lstm_BTC-USD_1m_v5.pt     (74.0% acc, ~XXX KB) ✅
├── lstm_ETH-USD_1m_v5.pt     (70.6% acc, ~XXX KB) ✅
├── lstm_SOL-USD_1m_v5.pt     (72.1% acc, ~XXX KB) ✅
└── transformer_multi_v5.pt   (63.4% acc, ~XXX KB) ✅
```

## ✅ Verification

```bash
# Verify models have full state_dict
uv run python -c "
import torch
for model in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
    checkpoint = torch.load(f'models/v5/lstm_{model}_1m_v5.pt', map_location='cpu')
    print(f'{model}: {list(checkpoint.keys())}')
    assert 'model_state_dict' in checkpoint, f'{model} missing state_dict!'
print('✅ All models verified!')
"
```

## 🚀 Next Steps

**Builder Claude** can now:
1. Evaluate models locally
2. Promote to `models/promoted/`
3. Test dry-run mode
4. GO LIVE tonight! 🚀

**QC Claude** can now:
1. Review model architecture
2. Validate training results
3. Approve for production use
EOF

git add V5_MODELS_READY.md
git commit -m "docs: V5 models ready for deployment"
git push origin main
```

---

## 📋 Verification Checklist

After completing the sync, verify:

- [ ] V5 model files are > 100 KB each (not 908 bytes)
- [ ] Each model contains `model_state_dict` key
- [ ] Model state dict contains weight tensors
- [ ] Files committed and pushed to GitHub
- [ ] Both Claude instances can access the files

**Verification command**:
```bash
uv run python << 'EOF'
import torch
from pathlib import Path

models = [
    'models/v5/lstm_BTC-USD_1m_v5.pt',
    'models/v5/lstm_ETH-USD_1m_v5.pt',
    'models/v5/lstm_SOL-USD_1m_v5.pt',
    'models/v5/transformer_multi_v5.pt'
]

print("="*70)
print("V5 Model Verification")
print("="*70)

all_valid = True
for model_path in models:
    path = Path(model_path)
    if not path.exists():
        print(f"❌ {path.name}: NOT FOUND")
        all_valid = False
        continue

    size_kb = path.stat().st_size / 1024
    checkpoint = torch.load(model_path, map_location='cpu')
    has_state = 'model_state_dict' in checkpoint

    status = "✅" if (size_kb > 100 and has_state) else "❌"
    print(f"{status} {path.name}:")
    print(f"   Size: {size_kb:.1f} KB")
    print(f"   Keys: {list(checkpoint.keys())}")
    print(f"   Has weights: {'YES' if has_state else 'NO'}")

    if size_kb < 100 or not has_state:
        all_valid = False

print("="*70)
if all_valid:
    print("✅ ALL MODELS VERIFIED - Ready for production!")
else:
    print("❌ VERIFICATION FAILED - Models incomplete")
    print("   Re-run sync or use Alternative Plan")
print("="*70)
EOF
```

---

## 🎯 Success Criteria

**Task complete when**:
1. ✅ All V5 model files contain `model_state_dict`
2. ✅ File sizes are 200-500 KB (not 908 bytes)
3. ✅ Files pushed to GitHub
4. ✅ Verification script passes
5. ✅ Builder Claude and QC Claude can load models

---

## 📞 Communication

**Report back with**:
1. GPU instance status (running/stopped/terminated)
2. Model file sizes after download
3. Verification script results
4. GitHub sync confirmation

**Example report**:
```
✅ MISSION COMPLETE

Instance Status: running (then terminated)
Models Downloaded: 4 files
File Sizes:
  - lstm_BTC-USD: 245 KB ✅
  - lstm_ETH-USD: 245 KB ✅
  - lstm_SOL-USD: 245 KB ✅
  - transformer: 312 KB ✅
Verification: PASSED ✅
GitHub Sync: COMPLETE ✅

QC Claude and Builder Claude can now evaluate and deploy V5 models!
```

---

**Priority**: 🔴 CRITICAL - Blocking tonight's go-live
**ETA**: 10-15 minutes (if instance still running)
**Fallback**: Use existing complete models from gpu_trained_proper/

Let's get those V5 models synced! 🚀
