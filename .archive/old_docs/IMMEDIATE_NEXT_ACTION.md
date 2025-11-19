# ⚡ IMMEDIATE NEXT ACTION - Run GPU Evaluation

**Status**: 🟢 READY TO EXECUTE
**Time Required**: 30 minutes
**Your Role**: Run Colab GPU evaluation

---

## 🎯 WHAT YOU'RE DOING

Running GPU evaluation of the 3 trained models (BTC, ETH, SOL) on Google Colab to:
1. Check if they pass promotion gates (68% accuracy, 5% calibration)
2. Decide: Deploy to production OR Quick retrain

**Expected Result**: Know in 30 minutes if models are production-ready

---

## 📋 STEP-BY-STEP (30 Minutes Total)

### Step 1: Verify Files on Cloud Server (5 min)

```bash
# SSH to cloud server
ssh root@your-cloud-server

# Navigate to project
cd ~/crpbot

# Check for Colab files (Cloud Claude prepared these)
ls -la colab_evaluate_50feat_models.ipynb
ls -la COLAB_EVALUATION.md
ls -lh /tmp/colab_upload/

# You should see:
# - colab_evaluate_50feat_models.ipynb (notebook)
# - /tmp/colab_upload/models/*.pt (3 files, ~12 MB total)
# - /tmp/colab_upload/features/*.parquet (3 files, ~644 MB total)
```

**If files are missing**: Tell Cloud Claude to prepare them

**If files exist**: Continue to Step 2 ✅

---

### Step 2: Download Files to Local Machine (5 min)

```bash
# On your local machine (/home/numan/crpbot)

# Create download directory
mkdir -p ~/Downloads/colab_eval

# Download notebook
scp root@cloud-server:~/crpbot/colab_evaluate_50feat_models.ipynb ~/Downloads/colab_eval/

# Download instructions (optional)
scp root@cloud-server:~/crpbot/COLAB_EVALUATION.md ~/Downloads/colab_eval/

# Note: We'll upload model/feature files directly from cloud server
# OR use Google Drive if you prefer
```

---

### Step 3: Set Up Google Colab (5 min)

```
1. Open browser: https://colab.research.google.com/

2. Upload notebook:
   - Click "File → Upload notebook"
   - Select: ~/Downloads/colab_eval/colab_evaluate_50feat_models.ipynb
   - Click "Upload"

3. Enable GPU:
   - Click "Runtime → Change runtime type"
   - Hardware accelerator: GPU
   - GPU type: T4 (free) or V100 (Colab Pro)
   - Click "Save"

4. Verify GPU:
   - Run first cell (should show GPU info)
   - Should see: Tesla T4 or V100
```

---

### Step 4: Upload Files to Colab (10 min)

**Option A: Direct Upload** (Recommended if files are on local machine)

```
1. In Colab, click folder icon (left sidebar)
2. Create directories:
   - Right-click → New folder → "models"
   - Right-click models → New folder → "new"
   - Right-click → New folder → "data"
   - Right-click data → New folder → "features"

3. Upload model files to models/new/:
   - Download from cloud server: /tmp/colab_upload/models/*.pt
   - Upload to Colab: models/new/

4. Upload feature files to data/features/:
   - Download from cloud server: /tmp/colab_upload/features/*.parquet
   - Upload to Colab: data/features/
```

**Option B: Google Drive** (If files already in Drive)

```python
# Run this cell in Colab:
from google.colab import drive
drive.mount('/content/drive')

# Then copy files
!mkdir -p models/new data/features
!cp /content/drive/MyDrive/crpbot/models/*.pt models/new/
!cp /content/drive/MyDrive/crpbot/features/*.parquet data/features/
```

---

### Step 5: Run Evaluation (5-10 min GPU time)

```
1. Click: "Runtime → Run all"

2. Watch for progress:
   - Setting up environment... ✅
   - Loading models... ✅
   - Loading features... ✅
   - Evaluating BTC model... ✅
   - Evaluating ETH model... ✅
   - Evaluating SOL model... ✅

3. Wait for final output:
   === PROMOTION GATE RESULTS ===
   BTC: Accuracy X.XX, Calibration X.XX - [PASS/FAIL]
   ETH: Accuracy X.XX, Calibration X.XX - [PASS/FAIL]
   SOL: Accuracy X.XX, Calibration X.XX - [PASS/FAIL]

   Overall: X/3 models passed gates
```

**GPU Time**: 5-10 minutes (vs 60-90 minutes on CPU!)

---

### Step 6: Download & Share Results (5 min)

```
1. In Colab, find: evaluation_results.csv
2. Right-click → Download
3. Save to: ~/Downloads/colab_eval/

4. Share with Cloud Claude:
   Option A: Upload to cloud server
   scp ~/Downloads/colab_eval/evaluation_results.csv root@cloud-server:~/crpbot/reports/phase6_5/

   Option B: Create GitHub issue with results
   - Paste the final output
   - Attach evaluation_results.csv

   Option C: Tell Cloud Claude directly
   "Evaluation complete! Results: BTC X%, ETH X%, SOL X%"
```

---

## 🎯 DECISION POINT (Based on Results)

### If ALL 3 Models PASS (≥68% accuracy, ≤5% calibration):
```
✅ PROCEED TO DEPLOYMENT

Next steps (Cloud Claude + Amazon Q):
1. Cloud Claude: Validate models
2. Amazon Q: Upload to S3
   q "Upload models/promoted/*.pt to s3://crpbot-models/production/"
3. Amazon Q: Deploy to EC2
   q "Deploy to production EC2 with new models"
4. Production dry-run starts (TODAY)
5. Live trading in 3-5 days

Timeline: Production in ~2 hours from now
```

### If 1-2 Models PASS:
```
⚠️ PARTIAL SUCCESS

Options:
1. Deploy passing models only (conservative approach)
2. Quick retrain failing models (~60 min on Colab GPU)
3. User decides based on which models passed

Timeline: +1-2 hours
```

### If ALL 3 Models FAIL:
```
🔄 RETRAIN NEEDED

Cloud Claude will:
1. Analyze why models failed
2. Adjust hyperparameters
3. Prepare retraining notebook
4. User reruns on Colab GPU (~60 min)

Timeline: +2-3 hours
```

---

## ⏱️ TIMELINE FROM NOW

```
Now:        You run Steps 1-6 [30 min]
+30 min:    Share results with Cloud Claude
+45 min:    Cloud Claude analyzes results
+2 hours:   Models deployed (if passing)
            OR Retraining starts (if failing)
+4 hours:   Production dry-run active
+5 days:    Live trading begins
```

---

## 🚨 TROUBLESHOOTING

### "Can't find Colab files on cloud server"
→ Cloud Claude needs to prepare them. Tell Cloud Claude: "Prepare Colab evaluation files"

### "Upload is too slow"
→ Option 1: Upload to Google Drive first, then mount in Colab
→ Option 2: Use smaller feature files (Cloud Claude can prepare)

### "GPU not available in Colab"
→ Use Colab Pro (guaranteed GPU)
→ Or wait for free GPU availability

### "Evaluation fails with error"
→ Copy full error message
→ Share with Cloud Claude for debugging

### "Results are unclear"
→ Look for: "Accuracy: X.XX" and "Calibration Error: X.XX"
→ Pass criteria: Accuracy ≥0.68, Calibration ≤0.05

---

## 📊 EXPECTED OUTPUT

### Success Example:
```
=== EVALUATION RESULTS ===

BTC-USD Model:
- Test Accuracy: 0.72 (72%) ✅ PASS (≥68%)
- Calibration Error: 0.03 (3%) ✅ PASS (≤5%)
- Verdict: ✅ READY FOR PRODUCTION

ETH-USD Model:
- Test Accuracy: 0.69 (69%) ✅ PASS
- Calibration Error: 0.04 (4%) ✅ PASS
- Verdict: ✅ READY FOR PRODUCTION

SOL-USD Model:
- Test Accuracy: 0.71 (71%) ✅ PASS
- Calibration Error: 0.04 (4%) ✅ PASS
- Verdict: ✅ READY FOR PRODUCTION

=== OVERALL: 3/3 MODELS PASSED ===
✅ CLEARED FOR DEPLOYMENT
```

### Failure Example:
```
=== EVALUATION RESULTS ===

BTC-USD Model:
- Test Accuracy: 0.62 (62%) ❌ FAIL (need 68%)
- Calibration Error: 0.07 (7%) ❌ FAIL (need ≤5%)
- Verdict: ❌ NEEDS RETRAINING

ETH-USD Model:
- Test Accuracy: 0.65 (65%) ❌ FAIL
- Calibration Error: 0.06 (6%) ❌ FAIL
- Verdict: ❌ NEEDS RETRAINING

SOL-USD Model:
- Test Accuracy: 0.63 (63%) ❌ FAIL
- Calibration Error: 0.06 (6%) ❌ FAIL
- Verdict: ❌ NEEDS RETRAINING

=== OVERALL: 0/3 MODELS PASSED ===
🔄 RETRAINING REQUIRED
```

---

## 📞 AFTER COMPLETION

**Tell me (Local Claude)**:
"Evaluation complete! Results: [paste output or summary]"

I will:
1. Document results
2. Coordinate next steps
3. Update master plan

**Cloud Claude will**:
- Analyze detailed metrics
- Prepare deployment (if passing)
- Prepare retraining (if failing)

**Amazon Q will** (if deploying):
- Upload models to S3
- Deploy to production EC2
- Setup monitoring

---

## 🎯 SUCCESS CRITERIA

By the end of this step, we will know:
- ✅ Exact accuracy of each model
- ✅ Which models pass promotion gates
- ✅ Whether to deploy or retrain
- ✅ Timeline to production

**This 30-minute evaluation unlocks the entire deployment pipeline!**

---

## 🚀 READY TO START?

1. ✅ Amazon Q connected on both machines
2. ✅ Master plan created
3. ✅ All agents aligned
4. ✅ Cloud Claude prepared files
5. ✅ Instructions clear

**EXECUTE NOW!** ⚡

Follow Steps 1-6 above. Report results when done.

---

**File**: `IMMEDIATE_NEXT_ACTION.md`
**Created**: 2025-11-13
**Status**: READY FOR EXECUTION
