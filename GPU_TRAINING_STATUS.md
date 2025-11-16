# GPU Training Status - V6 Models

**Started**: 2025-11-15 ~20:50 EST
**Instance**: 98.91.192.206 (AWS g5.xlarge, NVIDIA A10G)
**Handler**: Amazon Q
**Status**: 🔄 **TRAINING IN PROGRESS**

---

## Training Details

### Models Being Trained
- ✅ BTC-USD LSTM (31 features, 15 epochs)
- ✅ ETH-USD LSTM (31 features, 15 epochs)
- ✅ SOL-USD LSTM (31 features, 15 epochs)

### Feature Count: 31 (Runtime-Aligned)
This fixes the V5 FIXED feature mismatch (80 → 31)

### Expected Training Time
- **Per model**: ~10-15 minutes on GPU
- **Total (3 models)**: ~30-45 minutes
- **ETA**: 21:30 EST

---

## Root Cause Fixed

### Problem
```
V5 FIXED models: 80 features
Runtime pipeline: 31 features
Result: 50% predictions (random)
```

### Solution
```
V6 models: 31 features (matches runtime)
Training data: Using current 31-feature pipeline
Result: Real predictions (expected 60-70%)
```

---

## When Training Completes

### 1. Download Models
```bash
./scripts/download_v6_models.sh
```

### 2. Test Predictions
```bash
# Kill current bot
pkill -f "runtime/main.py"

# Test with V6 models
./run_runtime_with_env.sh --mode dryrun --iterations 5
```

**Expected Output**:
```
BTC-USD: long @ 67.3% (LSTM: 0.673)  # >50% ✅
ETH-USD: short @ 62.1% (LSTM: 0.379)
SOL-USD: long @ 71.8% (LSTM: 0.718)
```

### 3. Restart Live Bot
```bash
# If predictions look good
./run_runtime_with_env.sh --mode live --iterations -1
```

---

## Monitoring

### Check Training Progress
```bash
ssh -i ~/.ssh/crpbot-training.pem ec2-user@98.91.192.206 "tail -f ~/training.log"
```

### Check GPU Utilization
```bash
ssh -i ~/.ssh/crpbot-training.pem ec2-user@98.91.192.206 "nvidia-smi"
```

---

## Cost Tracking

**Training Time**: ~45 minutes
**Instance**: g5.xlarge spot (~$0.30/hour)
**Total Cost**: ~$0.22

---

## Next Session Tasks

- [ ] Monitor training completion
- [ ] Download V6 models
- [ ] Test predictions (verify >50%)
- [ ] Restart live bot with V6
- [ ] Commit V6 models to git
- [ ] Update documentation

---

**Last Updated**: 2025-11-15 20:50 EST
**Update Frequency**: Check every 15 minutes
