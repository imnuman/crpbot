# Retraining Execution Plan - Start Here

**Time**: ~1.5 hours total
**Status**: Ready to execute
**Current**: 2025-11-13 ~05:30 UTC

---

## 🎯 Three-Part Execution

### Part 1: Cloud Claude Updates Code (25 min)
### Part 2: Human Runs Colab Training (30 min)
### Part 3: Cloud Claude Evaluates Results (15 min)

---

## 📋 Part 1: Cloud Claude Updates Code (25 min)

**Location**: Cloud server `/root/crpbot`

**Task**: Update model architecture and training configuration

**Instructions**: Send this message to Cloud Claude:

```
I need you to implement the improved LSTM architecture for retraining.

Follow the complete guide in: RETRAINING_IMPLEMENTATION.md

Execute Steps 1-3:
1. Update apps/trainer/models/lstm.py (hidden_size=128, num_layers=3, dropout=0.35)
2. Update apps/trainer/train/train_lstm.py (epochs=50, LR scheduler, weighted loss, early stopping patience=7)
3. Verify changes with grep commands

After completing code changes:
- Commit with descriptive message
- Push to GitHub
- Confirm you're done and ready for Colab training

DO NOT start training on cloud server - we're using Colab Pro GPU.

Estimated time: 25 minutes
```

**Expected Response from Cloud Claude**:
```
✅ Model architecture updated (128/3/0.35)
✅ Training config updated (50 epochs, scheduler, weighted loss)
✅ Changes committed and pushed to GitHub
✅ Ready for Colab Pro GPU training

Git commit: abc1234 "feat: implement improved LSTM architecture"
```

---

## 📋 Part 2: Human Runs Colab Training (30 min)

**Location**: Google Colab Pro

**Prerequisites**:
- ✅ Google Colab Pro access
- ✅ GPU runtime enabled
- ✅ AWS credentials
- ✅ Data in Google Drive: `/MyDrive/crpbot/data/features/`

**Steps**:

### 1. Open Colab Pro (1 min)
- Go to: https://colab.research.google.com/
- File → New notebook
- Runtime → Change runtime type → **GPU** → Save
- Verify "GPU" shows in top right corner

### 2. Setup Colab Secrets (2 min)
- Click 🔑 (Keys icon) in left sidebar
- Add secret: `AWS_ACCESS_KEY_ID` = `<your key>`
- Add secret: `AWS_SECRET_ACCESS_KEY` = `<your secret>`
- Enable "Notebook access" toggle for both
- ✅ Secrets should show green checkmark when enabled

### 3. Paste and Run Script (1 min)
- Create new code cell
- Copy **entire contents** of `COLAB_TRAINING_SCRIPT_V2.py`
- Paste into cell
- Click ▶️ (or press Shift+Enter)

### 4. Monitor Progress (30 min)
Watch the output for:
```
🚀 Starting IMPROVED GPU Training for 3 LSTM Models
Architecture: 128 hidden units, 3 layers, 0.35 dropout, 50 epochs

Training BTC-USD LSTM Model (IMPROVED) on GPU
Epoch 1/50: Train Loss=0.6543, Val Loss=0.6789, Val Acc=0.523, LR=0.001000
Epoch 2/50: Train Loss=0.6432, Val Loss=0.6654, Val Acc=0.547, LR=0.000951
...
Early stopping triggered at epoch 34
✅ BTC-USD training complete in 11.2 minutes!

Training ETH-USD LSTM Model (IMPROVED) on GPU
...
✅ ETH-USD training complete in 10.8 minutes!

Training SOL-USD LSTM Model (IMPROVED) on GPU
...
✅ SOL-USD training complete in 10.5 minutes!

Training Summary (IMPROVED MODELS)
BTC-USD: ✅ SUCCESS (11.2 min)
ETH-USD: ✅ SUCCESS (10.8 min)
SOL-USD: ✅ SUCCESS (10.5 min)
Total time: 32.5 minutes

📤 Uploading models to S3...
✅ Models uploaded to S3 successfully!

💾 Backing up models to Google Drive...
✅ Models backed up to Google Drive

🎉 IMPROVED MODEL TRAINING COMPLETE!
```

### 5. Notify Cloud Claude (1 min)
When you see "🎉 IMPROVED MODEL TRAINING COMPLETE!", send this message to Cloud Claude:

```
✅ Colab Pro training complete!

Total time: [X] minutes
Models uploaded to: s3://crpbot-market-data-dev/models/

Architecture: 128 hidden units, 3 layers, 0.35 dropout, 50 epochs

Trained models:
- lstm_BTC_USD_1m_[hash].pt
- lstm_ETH_USD_1m_[hash].pt
- lstm_SOL_USD_1m_[hash].pt

Please download from S3 and evaluate against 68% promotion gate.
Follow Part 3 of RETRAINING_EXECUTION_PLAN.md
```

---

## 📋 Part 3: Cloud Claude Evaluates Results (15 min)

**Location**: Cloud server `/root/crpbot`

**Task**: Download, evaluate, and document results

**Instructions**: Send this message to Cloud Claude:

```
Download and evaluate the improved models from S3.

Follow Step 5 from RETRAINING_IMPLEMENTATION.md:

1. Download models from S3
2. Evaluate each model (BTC, ETH, SOL) with real data
3. Document results in MODEL_EVALUATION_RESULTS_IMPROVED.md
4. Compare old vs new results
5. Make promotion decision based on 68% accuracy, 5% calibration gates
6. Commit and push results

Expected gates:
- Test Accuracy: ≥68%
- Calibration Error: ≤5%
- Need at least 2/3 models to pass

Report back with:
- Evaluation results for each model
- Pass/Fail status
- Promotion decision
- Next recommended steps
```

**Expected Response from Cloud Claude**:
```
✅ Models downloaded from S3
✅ Evaluated all 3 models with real data

Results (Improved Architecture):
- BTC-USD: 71.2% accuracy, 3.8% calibration ✅ PASS
- ETH-USD: 69.4% accuracy, 4.2% calibration ✅ PASS
- SOL-USD: 66.8% accuracy, 4.9% calibration ❌ FAIL (close!)

Comparison:
Old (64/2/15): 13-30% win rates → All FAILED
New (128/3/50): 67-71% accuracy → 2/3 PASSED ✅

Decision: PROMOTE BTC and ETH models to production
Status: Ready for Phase 6.5 observation period

Git commit: def5678 "docs: add improved model evaluation results"
```

---

## ⏱️ Complete Timeline

| Time | Task | Who | Duration |
|------|------|-----|----------|
| 05:30 | Part 1: Update code | Cloud Claude | 25 min |
| 05:55 | Part 2: Colab training | Human | 30 min |
| 06:25 | Part 3: Evaluate models | Cloud Claude | 15 min |
| **06:40** | **COMPLETE** | | **1 hr 10 min** |

---

## 🚨 Troubleshooting

### Part 1 Issues:

**"Can't find files to edit"**
- Cloud Claude: Pull latest from GitHub first: `git pull origin main`

**"Syntax errors after editing"**
- Cloud Claude: Check Python indentation carefully
- Run: `python -m py_compile apps/trainer/models/lstm.py`

### Part 2 Issues:

**"No GPU detected"**
- Runtime → Change runtime type → GPU → Save
- Runtime → Restart runtime
- Try again

**"Colab Secrets not found"**
- Make sure "Notebook access" toggle is enabled for both secrets
- Click 🔑 icon, verify green checkmarks

**"AWS credentials error"**
- Check secrets have correct names: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- No quotes, no spaces in the values

**"Data not found in Google Drive"**
- Verify path: `/content/drive/MyDrive/crpbot/data/features/`
- Should have 3 files: `features_BTC-USD_1m_latest.parquet`, etc.

### Part 3 Issues:

**"Models not found in S3"**
- Check Colab output: confirm upload completed successfully
- Run: `aws s3 ls s3://crpbot-market-data-dev/models/ --recursive`

**"Evaluation still shows 100% win rate"**
- This should NOT happen (we fixed evaluation pipeline)
- If it does: evaluation pipeline wasn't properly committed
- Cloud Claude: Re-read and re-apply FIX_EVALUATION.md

---

## ✅ Success Indicators

### Part 1 (Code Update):
- ✅ Files modified: `lstm.py`, `train_lstm.py`
- ✅ Git commit created with changes
- ✅ Pushed to GitHub successfully

### Part 2 (Colab Training):
- ✅ "GPU available: Tesla T4" (or similar)
- ✅ All 3 models trained successfully
- ✅ Models uploaded to S3
- ✅ Backed up to Google Drive
- ✅ "🎉 IMPROVED MODEL TRAINING COMPLETE!"

### Part 3 (Evaluation):
- ✅ Models downloaded from S3
- ✅ Evaluation shows realistic win rates (not 100%)
- ✅ Results documented in markdown file
- ✅ Promotion decision made
- ✅ Committed and pushed to GitHub

---

## 🎯 What Success Looks Like

**Ideal Outcome**:
```
Improved Models (128/3/50):
- BTC: 70-75% accuracy ✅ PASS
- ETH: 68-72% accuracy ✅ PASS
- SOL: 65-70% accuracy ✅ PASS

All 3 models PASS promotion gates!
→ Promote to production
→ Continue to Transformer training
→ Start Phase 6.5 observation
```

**Acceptable Outcome**:
```
Improved Models (128/3/50):
- BTC: 68-70% accuracy ✅ PASS
- ETH: 67-69% accuracy ✅ PASS
- SOL: 62-66% accuracy ❌ FAIL

2/3 models PASS promotion gates
→ Promote BTC and ETH
→ Consider further improvements for SOL
→ Start Phase 6.5 with 2 models
```

**Needs More Work**:
```
Improved Models (128/3/50):
- BTC: 60-65% accuracy ❌ FAIL
- ETH: 58-63% accuracy ❌ FAIL
- SOL: 55-60% accuracy ❌ FAIL

Models improved but didn't reach gate
→ See "Further Improvements" in RETRAINING_IMPLEMENTATION.md
→ Consider 4 layers, attention mechanism, more features
```

---

**Ready to start!** Begin with Part 1 - send instructions to Cloud Claude.
