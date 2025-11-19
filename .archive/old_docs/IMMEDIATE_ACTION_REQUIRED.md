# ⚡ IMMEDIATE ACTION REQUIRED

**Date**: 2025-11-13
**Priority**: 🔴 HIGH
**Time Required**: 30 minutes

---

## 🎯 What You Need to Do RIGHT NOW

Cloud Claude has prepared everything for **fast GPU evaluation**. You just need to run it on Google Colab.

---

## 📋 Quick Checklist (30 minutes total)

### ✅ Step 1: Get the Colab Notebook (5 min)

Cloud Claude should have created this file. Check the cloud server:

```bash
# SSH to cloud server
ssh root@your-cloud-server

# Look for Colab files
cd ~/crpbot
ls -la colab_evaluate_50feat_models.ipynb
ls -la COLAB_EVALUATION.md
ls -lh /tmp/colab_upload/
```

**Expected Files**:
- `colab_evaluate_50feat_models.ipynb` - Main notebook
- `COLAB_EVALUATION.md` - Instructions
- `/tmp/colab_upload/models/` - 3 model files (~12 MB)
- `/tmp/colab_upload/features/` - 3 feature files (~644 MB)

**Download to your local machine**:
```bash
scp root@cloud-server:~/crpbot/colab_evaluate_50feat_models.ipynb ~/Downloads/
```

---

### ✅ Step 2: Open Google Colab (2 min)

1. Go to: https://colab.research.google.com/
2. Click: **File → Upload notebook**
3. Upload: `colab_evaluate_50feat_models.ipynb`
4. Click: **Runtime → Change runtime type**
5. Select: **GPU** (Tesla T4)
6. Click: **Save**

---

### ✅ Step 3: Upload Files to Colab (10 min)

**Option A: Direct Upload** (If files are on your local machine)
```
1. In Colab, click folder icon (left sidebar)
2. Create folder: models/new/
3. Upload 3 model files from cloud server /tmp/colab_upload/models/
4. Create folder: data/features/
5. Upload 3 feature files from cloud server /tmp/colab_upload/features/
```

**Option B: Mount Google Drive** (If files uploaded to Drive)
```python
# Run this cell in Colab:
from google.colab import drive
drive.mount('/content/drive')

# Then copy files from Drive to Colab workspace
!cp /content/drive/MyDrive/crpbot/models/*.pt models/new/
!cp /content/drive/MyDrive/crpbot/features/*.parquet data/features/
```

---

### ✅ Step 4: Run Evaluation (10 min)

1. Click: **Runtime → Run all**
2. Wait for completion (5-10 minutes on GPU)
3. Watch for output:
   ```
   Evaluating BTC model... ✅
   Evaluating ETH model... ✅
   Evaluating SOL model... ✅

   === PROMOTION GATE RESULTS ===
   BTC: Accuracy 0.XX, Calibration 0.XX - [PASS/FAIL]
   ETH: Accuracy 0.XX, Calibration 0.XX - [PASS/FAIL]
   SOL: Accuracy 0.XX, Calibration 0.XX - [PASS/FAIL]
   ```

---

### ✅ Step 5: Download Results (3 min)

1. In Colab, find: `evaluation_results.csv`
2. Right-click → Download
3. Upload to cloud server:
   ```bash
   scp ~/Downloads/evaluation_results.csv root@cloud-server:~/crpbot/reports/phase6_5/
   ```

4. Or paste results in a GitHub issue for Cloud Claude

---

## 🎯 What Happens Next

### If Models PASS (≥68% accuracy, ≤5% calibration):
✅ Cloud Claude deploys to production (2 hours)
✅ Start dry-run observation
✅ Live trading in 3-5 days

### If Models FAIL:
🔄 Cloud Claude adjusts hyperparameters
🔄 Quick retrain on Colab GPU (~60 min)
🔄 Re-evaluate
🔄 Deploy when passing

---

## 📊 Expected Results

**Performance Comparison**:
```
CPU Evaluation:  60-90 minutes  ❌ TOO SLOW
GPU Evaluation:  5-10 minutes   ✅ FAST!

Speedup: 10-12x faster
```

**Promotion Gates**:
- Accuracy: ≥68% (win rate)
- Calibration Error: ≤5%
- Both must pass for deployment

---

## 🚨 Troubleshooting

### "I can't find the Colab notebook"
→ Cloud Claude needs to create it. Tell them to push it to GitHub.

### "Upload is slow"
→ Use Google Drive. Upload files to Drive first, then mount in Colab.

### "GPU runtime not available"
→ Use Colab Pro (paid) or wait for free GPU availability.

### "Evaluation fails with error"
→ Copy full error message, share with Cloud Claude for debugging.

---

## 📞 Communication

After running evaluation:

**If Success**:
```
"Evaluation complete! Results:
- BTC: 72% accuracy, 3% calibration - PASS ✅
- ETH: 69% accuracy, 4% calibration - PASS ✅
- SOL: 70% accuracy, 4.5% calibration - PASS ✅

Ready for deployment!"
```

**If Failure**:
```
"Evaluation complete, but models need retraining:
- BTC: 62% accuracy - FAIL (need 68%)
- ETH: 65% accuracy - FAIL
- SOL: 63% accuracy - FAIL

Need to retrain with adjusted parameters."
```

---

## ⏱️ Timeline

```
Now:      You run Colab evaluation (30 min)
+30 min:  Share results with Cloud Claude
+45 min:  Cloud Claude processes results
+2 hours: Models deployed OR retraining starts
+4 hours: Production dry-run active
+5 days:  Live trading begins
```

---

## 🎯 Your Goal Today

```
☐ Download Colab notebook from cloud server
☐ Upload to Google Colab
☐ Enable GPU runtime
☐ Upload model + feature files
☐ Run evaluation (5-10 min)
☐ Download results
☐ Share with Cloud Claude

Total Time: 30 minutes
Result: Know if models are production-ready
```

---

**This is the FASTEST path to production. Let's execute NOW!** ⚡

---

## 📚 Full Details

For complete context, see:
- `MASTER_FAST_EXECUTION_PLAN.md` - Complete strategy
- `PROJECT_MEMORY.md` - Agent roles and context
- `COLAB_EVALUATION.md` - Detailed Colab instructions (on cloud server)
