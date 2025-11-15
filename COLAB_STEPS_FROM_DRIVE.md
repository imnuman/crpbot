# 🚀 Run Colab Evaluation from Google Drive

**Status**: Files uploaded to Google Drive ✅
**Location**: `/My Drive/crpbot/`
**Next**: Run evaluation on Colab GPU (15 minutes)

---

## 📋 Quick Steps (15 Minutes Total)

### Step 1: Open Notebook in Colab (2 min)

**Option A: From Google Drive**
```
1. Go to Google Drive: https://drive.google.com/
2. Navigate to: My Drive/crpbot/
3. Find: colab_evaluate_50feat_models.ipynb
4. Right-click → Open with → Google Colaboratory
```

**Option B: Upload to Colab**
```
1. Go to: https://colab.research.google.com/
2. File → Upload notebook
3. Choose file from: My Drive/crpbot/colab_evaluate_50feat_models.ipynb
```

---

### Step 2: Enable GPU (1 min)

```
1. In Colab, click: Runtime → Change runtime type
2. Hardware accelerator: GPU
3. GPU type: T4 (free) or V100/A100 (Colab Pro)
4. Click: Save
```

**Verify GPU**:
Run this in a cell:
```python
!nvidia-smi
```
Should show: Tesla T4 (or V100/A100)

---

### Step 3: Mount Google Drive (1 min)

**Run this cell** (Colab will ask for permission):
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Steps**:
1. Click the link that appears
2. Choose your Google account
3. Click "Allow"
4. Copy the authorization code
5. Paste it back in Colab
6. Press Enter

**Verify mounted**:
```python
!ls /content/drive/MyDrive/crpbot/
```
Should show your uploaded files.

---

### Step 4: Copy Files from Drive to Colab Workspace (3 min)

**Check what files you have in Drive**:
```python
!ls -lh /content/drive/MyDrive/crpbot/
```

**If you uploaded everything to Drive, run**:
```python
# Create directories
!mkdir -p models/new data/features

# Copy model files
!cp /content/drive/MyDrive/crpbot/models/*.pt models/new/ 2>/dev/null || \
 cp /content/drive/MyDrive/crpbot/*.pt models/new/ 2>/dev/null || \
 echo "Model files not found in expected location"

# Copy feature files
!cp /content/drive/MyDrive/crpbot/features/*.parquet data/features/ 2>/dev/null || \
 cp /content/drive/MyDrive/crpbot/*.parquet data/features/ 2>/dev/null || \
 echo "Feature files not found in expected location"

# Verify
!ls -lh models/new/
!ls -lh data/features/
```

**Expected output**:
```
models/new/
- lstm_BTC_USD_1m_7b5f0829.pt (3.9M)
- lstm_ETH_USD_1m_7b5f0829.pt (3.9M)
- lstm_SOL_USD_1m_7b5f0829.pt (3.9M)

data/features/
- features_BTC-USD_1m_2025-11-13_50feat.parquet (228M)
- features_ETH-USD_1m_2025-11-13_50feat.parquet (218M)
- features_SOL-USD_1m_2025-11-13_50feat.parquet (198M)
```

---

### Step 5: Run Evaluation (5-10 min on GPU)

**Method 1: Run All Cells**
```
Click: Runtime → Run all

Wait for completion (5-10 minutes on GPU)
```

**Method 2: Run Cell by Cell** (to watch progress)
```
1. Click first cell, press Shift+Enter
2. Watch output
3. Repeat for each cell
```

**You'll see progress**:
```
Setting up environment... ✅
Loading model definitions... ✅
Loading BTC model... ✅
Loading ETH model... ✅
Loading SOL model... ✅
Loading BTC features... ✅
Loading ETH features... ✅
Loading SOL features... ✅
Evaluating BTC model... ✅ (takes ~2-3 min on GPU)
Evaluating ETH model... ✅ (takes ~2-3 min on GPU)
Evaluating SOL model... ✅ (takes ~2-3 min on GPU)
Generating report... ✅
```

---

### Step 6: Check Results (2 min)

**Look for this output**:
```
═══════════════════════════════════════════════
    MODEL EVALUATION RESULTS
═══════════════════════════════════════════════

BTC-USD Model:
  Test Accuracy:      0.XX (XX%)
  Calibration Error:  0.XX (XX%)
  Promotion Gate:     [✅ PASS / ❌ FAIL]

ETH-USD Model:
  Test Accuracy:      0.XX (XX%)
  Calibration Error:  0.XX (XX%)
  Promotion Gate:     [✅ PASS / ❌ FAIL]

SOL-USD Model:
  Test Accuracy:      0.XX (XX%)
  Calibration Error:  0.XX (XX%)
  Promotion Gate:     [✅ PASS / ❌ FAIL]

═══════════════════════════════════════════════
OVERALL: X/3 models passed promotion gates
═══════════════════════════════════════════════

Promotion Criteria:
- Accuracy: ≥68% (0.68)
- Calibration Error: ≤5% (0.05)
```

---

### Step 7: Download Results (2 min)

**Find and download the CSV**:
```python
# The notebook should create this file
!ls -lh evaluation_results.csv
```

**Download it**:
1. In Colab, click folder icon (left sidebar)
2. Find: `evaluation_results.csv`
3. Right-click → Download

**Or copy to Drive**:
```python
!cp evaluation_results.csv /content/drive/MyDrive/crpbot/
print("✅ Results saved to Google Drive")
```

---

### Step 8: Share Results with Local Claude (2 min)

**Tell me one of these**:

**Option A: Quick Summary**
```
"Evaluation complete!
BTC: 72% accuracy, 3% calibration - PASS ✅
ETH: 69% accuracy, 4% calibration - PASS ✅
SOL: 70% accuracy, 4% calibration - PASS ✅

All 3 models passed! Ready for deployment."
```

**Option B: Detailed Results**
```
Paste the full output from the evaluation
```

**Option C: Upload CSV**
```
Download evaluation_results.csv and I'll analyze it
```

---

## 🚨 Troubleshooting

### "Files not found in Drive"
**Check your Drive structure**:
```python
!ls -R /content/drive/MyDrive/crpbot/
```

**If files are in different folders, adjust paths**:
```python
# Example if files are directly in crpbot folder
!cp /content/drive/MyDrive/crpbot/lstm_*.pt models/new/
!cp /content/drive/MyDrive/crpbot/features_*.parquet data/features/
```

### "Out of memory error"
**This shouldn't happen on GPU, but if it does**:
```
- Runtime → Factory reset runtime
- Re-run from Step 3
- Or upgrade to Colab Pro
```

### "Module not found"
**Run the setup cells first**:
The notebook should have installation cells at the top. Make sure they run first.

### "Model loading error"
**Check model files are copied correctly**:
```python
!ls -lh models/new/*.pt
# Should show 3 files, each ~3.9MB
```

### "Feature loading error"
**Check feature files**:
```python
!ls -lh data/features/*.parquet
# Should show 3 files, total ~644MB
```

---

## ⏱️ Timeline

```
Now:       Open notebook, enable GPU [3 min]
+3 min:    Mount Drive, copy files [4 min]
+7 min:    Run evaluation on GPU [5-10 min]
+17 min:   Download results [2 min]
+19 min:   Share with Local Claude [1 min]

Total: ~20 minutes
```

---

## 🎯 What Happens After Results

### If ALL 3 Models PASS:
```
✅ You tell me: "All passed!"
✅ Cloud Claude: Validates models
✅ Amazon Q: Uploads to S3
   q "Upload models to s3://crpbot-models/production/"
✅ Amazon Q: Deploys to EC2
   q "Deploy to production EC2"
✅ Production dry-run starts: TODAY
✅ Live trading: 5 days from now

Timeline: +2 hours to production
```

### If SOME Models PASS:
```
⚠️ You tell me: "BTC and ETH passed, SOL failed"
⚠️ We decide: Deploy 2 models OR retrain SOL
⚠️ Quick retrain if needed: ~60 min on Colab

Timeline: +1-3 hours
```

### If ALL Models FAIL:
```
❌ You tell me: "All failed"
❌ Cloud Claude: Analyzes why (hyperparameters)
❌ Cloud Claude: Prepares retraining notebook
❌ You: Rerun training on Colab GPU: ~60 min
❌ Re-evaluate: another 20 min

Timeline: +2-3 hours total
```

---

## 📊 Expected Results

**Good Results** (Models Pass):
```
BTC: 68-75% accuracy, 3-5% calibration ✅
ETH: 68-75% accuracy, 3-5% calibration ✅
SOL: 68-75% accuracy, 3-5% calibration ✅
```

**Borderline Results** (May Pass):
```
BTC: 65-68% accuracy, 4-6% calibration ⚠️
(Might pass one gate, fail another)
```

**Poor Results** (Will Fail):
```
BTC: <65% accuracy, >6% calibration ❌
(Need retraining with adjusted params)
```

---

## 💡 Pro Tips

1. **Keep Colab tab open** - Don't close it during evaluation
2. **Watch the progress** - Each step shows completion status
3. **GPU is fast** - 10x faster than CPU (5-10 min vs 60+ min)
4. **Save to Drive** - Results automatically saved if you use the copy command
5. **Don't panic on errors** - Most are easy to fix (check troubleshooting)

---

## ✅ Checklist

Before you start:
- [ ] Files uploaded to Google Drive (/My Drive/crpbot/)
- [ ] Colab notebook ready
- [ ] Have 20 minutes uninterrupted time

During execution:
- [ ] Notebook opened in Colab
- [ ] GPU enabled (Runtime → Change runtime type)
- [ ] Drive mounted (authorized)
- [ ] Files copied to Colab workspace
- [ ] Evaluation running
- [ ] Results showing

After completion:
- [ ] Results downloaded or copied to Drive
- [ ] Share results with Local Claude
- [ ] Wait for next steps

---

## 🚀 START NOW!

Everything is ready. Follow Steps 1-8 above.

**You have**: Notebook + files in Google Drive ✅
**You need**: 20 minutes of focused time ✅
**You'll get**: Know if deploying or retraining ✅

**Let's get those results!** 🎯

---

**File**: `COLAB_STEPS_FROM_DRIVE.md`
**Status**: Ready to execute
**Next**: You run evaluation, share results
