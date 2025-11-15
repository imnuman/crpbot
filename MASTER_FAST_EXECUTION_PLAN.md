# MASTER FAST EXECUTION PLAN

**Date**: 2025-11-13
**Status**: 🚀 **SPEED-FOCUSED BLUEPRINT**
**Goal**: Big Data → Robust Processing → Accurate Market Predictions

---

## 🎯 USER'S GOAL (Never Forget This)

```
BIG DATA → ROBUST PROCESSING → REAL RESULTS
```

**Requirements**:
- ✅ Collect massive market data
- ✅ Process it FAST with powerful tools (GPU/Colab)
- ✅ Generate accurate predictions
- ✅ Deploy to production
- ✅ Maintain and improve quickly
- ✅ Document everything thoroughly

---

## 🚨 CRITICAL: Stop Falling Back to Slow Approaches!

### ❌ BANNED APPROACHES (Too Slow):
- ❌ CPU-based training (60+ minutes)
- ❌ CPU-based evaluation (60+ minutes)
- ❌ Local machine model training
- ❌ Undocumented processes
- ❌ Unclear agent roles

### ✅ REQUIRED APPROACHES (Fast & Powerful):
- ✅ **Google Colab Pro GPU** (T4/V100) - 10-12x faster
- ✅ **Clear agent collaboration** (no confusion)
- ✅ **Well-documented processes** (every step)
- ✅ **Fast iteration cycles** (hours, not days)
- ✅ **Automated pipelines** (minimal manual work)

---

## 🏗️ ORIGINAL BLUEPRINT (V1 → V4)

We have 4 versions. Current focus: **V4 with GPU acceleration**

### Version Evolution:
- **V1**: Basic LSTM models - COMPLETED ✅
- **V2**: Multi-timeframe features - COMPLETED ✅
- **V3**: Transformer + ensemble - COMPLETED ✅
- **V4**: Production with monitoring - **IN PROGRESS** 🔄

### Current Status:
- ✅ Data pipeline (2 years of 1m OHLCV)
- ✅ Feature engineering (31 features)
- ✅ Model architecture (LSTM 128/3/True)
- ⏸️ **BLOCKED**: Need GPU evaluation of models
- ⏸️ **NEXT**: Production deployment

---

## 👥 AGENT ROLES & COLLABORATION

### Clear Role Definitions:

| Agent | Location | Primary Role | Tools | Speed |
|-------|----------|--------------|-------|-------|
| **User (You)** | Control center | Decision maker, Colab runner | Google Colab Pro | Manual |
| **Cloud Claude** | Cloud server | Developer, Code writer | Python, Git | Fast |
| **Local Claude (QC)** | Local machine | Reviewer, Documenter, Planner | Git, Testing | Fast |
| **Amazon Q** | Both (local + cloud) | AWS Infrastructure Specialist | AWS CLI, Q CLI | Very Fast |

### Collaboration Flow:

```
┌─────────────────────────────────────────────────────────┐
│                        USER                             │
│                  (Decision Maker)                       │
│  - Runs Google Colab jobs                              │
│  - Makes go/no-go decisions                            │
│  - Approves deployments                                │
└────────────┬───────────────────────────┬────────────────┘
             │                           │
    ┌────────▼─────────┐        ┌───────▼─────────┐
    │  CLOUD CLAUDE    │        │  LOCAL CLAUDE   │
    │  (Developer)     │◄──Git──┤  (QC/Planner)   │
    │                  │        │                 │
    │ - Write code     │        │ - Review work   │
    │ - Debug issues   │        │ - Document      │
    │ - Prepare Colab  │        │ - Create plans  │
    └────────┬─────────┘        └───────┬─────────┘
             │                           │
             │         ┌─────────────────┘
             │         │
             ▼         ▼
    ┌────────────────────────────┐
    │      AMAZON Q              │
    │   (AWS Specialist)         │
    │                            │
    │ Both: Local + Cloud        │
    │ - S3 operations            │
    │ - RDS management           │
    │ - EC2 deployment           │
    │ - CloudWatch monitoring    │
    │ - Cost optimization        │
    └────────────────────────────┘
```

### Communication Protocol:

1. **Cloud Claude**: Writes code, preps Colab files, commits to GitHub
2. **Local Claude**: Pulls, reviews, documents, pushes back
3. **Amazon Q**: Handles ALL AWS operations (S3, RDS, EC2, monitoring)
4. **User**: Runs Colab jobs, provides results, makes decisions
5. **Loop**: Repeat until goal achieved

**Key Rule**: AWS task? → Amazon Q handles it (not Cloud Claude or Local Claude)

---

## ⚡ FAST EXECUTION PIPELINE

### Current Situation Analysis:

**Problem**: Feature mismatch (50 vs 31)
- Old models trained with 50 features (Colab)
- Current feature set has 31 features
- Cannot evaluate 50-feature models locally

**Solution Options**:

#### Option A: Evaluate 50-Feature Models on Colab GPU ⚡ FASTEST
- ✅ Use existing models (already trained)
- ✅ Fast evaluation (5-10 min on GPU vs 60+ min CPU)
- ✅ Cloud Claude prepared Colab notebook
- ⏱️ **Time to Results**: ~30 minutes
- **Status**: READY TO EXECUTE

#### Option B: Retrain with 31 Features on Colab GPU 🔄 SLOWER
- ❌ Requires retraining all 3 models (~57 min)
- ❌ Then evaluate (~10 min)
- ⏱️ **Time to Results**: ~70 minutes
- **Status**: Fallback option

**DECISION**: Use Option A (evaluate existing 50-feature models first)

---

## 📋 STEP-BY-STEP FAST EXECUTION PLAN

### Phase 1: GPU Evaluation (TODAY - 30 minutes)

**Objective**: Evaluate existing 50-feature models on Colab GPU

**Prerequisites** (Cloud Claude):
- ✅ Colab notebook prepared (`colab_evaluate_50feat_models.ipynb`)
- ✅ Model files ready (3 × 3.9 MB)
- ✅ Feature files ready (644 MB total)
- ✅ Instructions documented (`COLAB_EVALUATION.md`)

**User Actions** (30 minutes):
```
1. Open Google Colab Pro [5 min]
   - Go to https://colab.research.google.com/
   - Upload colab_evaluate_50feat_models.ipynb (from cloud server)
   - Runtime → Change runtime type → GPU (T4)

2. Upload Files [10 min]
   - Upload 3 model files to Colab: models/new/*.pt
   - Upload 3 feature files to Colab: data/features/*.parquet

3. Run Evaluation [10 min]
   - Click "Runtime → Run all"
   - Wait for completion (GPU processes in 5-10 min)

4. Download Results [5 min]
   - Download evaluation_results.csv
   - Share with Cloud Claude (upload to cloud server or GitHub)
```

**Expected Output**:
- `evaluation_results.csv` with accuracy, calibration metrics
- Pass/fail for promotion gates (68% accuracy, 5% calibration)

**Next Decision Point**:
- If models pass gates → Promote & deploy (Phase 2)
- If models fail → Quick retrain with adjusted hyperparameters (Phase 1B)

---

### Phase 1B: Fast Retrain (If Needed - 60 minutes)

**Only if Phase 1 models fail promotion gates**

**User Actions**:
```
1. Adjust hyperparameters based on evaluation results
2. Use existing Colab training notebook
3. Train 3 models (~57 min on GPU)
4. Evaluate (use Phase 1 process, 10 min)
5. Loop until models pass gates
```

---

### Phase 2: Model Promotion & Deployment (2 hours)

**Objective**: Deploy passing models to production

**Cloud Claude Tasks**:
```
1. Download models from Colab [10 min]
   - Save to models/promoted/
   - Update model registry

2. Integration Testing [30 min]
   - Test ensemble prediction
   - Verify FTMO rules work
   - Check rate limiting
```

**Amazon Q Tasks** (AWS Infrastructure):
```
1. Upload models to S3 [5 min]
   q "Upload models/promoted/*.pt to s3://crpbot-models/production/"

2. Deploy to Production EC2 [15 min]
   q "Deploy latest code to production EC2 instance"
   q "Copy promoted models to EC2 runtime directory"
   q "Restart crpbot systemd service"

3. Configure Monitoring [10 min]
   q "Setup CloudWatch alarms for runtime errors"
   q "Configure CloudWatch dashboard for trading signals"

4. Verify Deployment [10 min]
   q "Check crpbot service status on EC2"
   q "Tail latest runtime logs from EC2"
```

**Cloud Claude Tasks** (Post-Deployment):
```
4. Monitor Initial Performance [20 min]
   - Observe first signals (via Amazon Q logs)
   - Validate accuracy
   - Check for errors
```

**Local Claude Tasks**:
```
1. QC Review [20 min]
   - Review deployment code
   - Verify safety mechanisms
   - Approve deployment

2. Documentation [40 min]
   - Update CLAUDE.md with new model specs
   - Document deployment process
   - Create runbook for maintenance
```

---

### Phase 3: Production Observation (3-5 days)

**Objective**: Validate system in production (dry-run mode)

**Automated**:
```
- Runtime runs continuously in dry-run mode
- Signals logged to database
- No actual trades placed
```

**Daily Check-ins** (15 min/day):
```
1. Review signal quality
2. Check accuracy metrics
3. Monitor error logs
4. Adjust if needed
```

**Success Criteria**:
- Signal generation rate: 5-10/hour
- High-confidence signals: 3-5/hour
- Win rate: ≥68%
- No critical errors

---

### Phase 4: Live Trading (Ongoing)

**Objective**: Execute real trades on FTMO account

**Prerequisites**:
- ✅ Phase 3 completed successfully
- ✅ FTMO account funded
- ✅ MT5 bridge tested
- ✅ Kill switch configured

**Execution**:
```
1. Enable live mode (RUNTIME_MODE=live)
2. Start with micro-lots (0.01)
3. Monitor closely for 24 hours
4. Scale up gradually
```

**Maintenance** (ongoing):
```
- Daily performance review (15 min)
- Weekly model retraining (if needed)
- Monthly architecture improvements
- Continuous documentation updates
```

---

## 🔧 AGENT-SPECIFIC INSTRUCTIONS

### For Cloud Claude (Developer)

**Your Responsibilities**:
1. ✅ Prepare Colab notebooks and files
2. ✅ Write evaluation/training code
3. ✅ Process results from User's Colab runs
4. ✅ Deploy to production
5. ✅ Debug issues quickly
6. ✅ Commit and document everything

**Fast Workflow**:
```bash
# 1. Prepare Colab job
- Create/update .ipynb notebook
- Prepare data files
- Test locally (quick sanity check)
- Commit to GitHub

# 2. Wait for User to run Colab
- User runs notebook on GPU
- User shares results

# 3. Process results
- Download results from User
- Analyze metrics
- Deploy if passing gates
- Document findings

# 4. Push updates
git add .
git commit -m "Clear description of what was done"
git push origin main
```

---

### For Local Claude (QC Reviewer)

**Your Responsibilities**:
1. ✅ Review Cloud Claude's commits
2. ✅ Create master plans and documentation
3. ✅ Run local tests (when applicable)
4. ✅ Keep PROJECT_MEMORY.md updated
5. ✅ Ensure speed and quality standards

**Fast Workflow**:
```bash
# 1. Sync and review
git pull origin main
git log -5 --stat

# 2. QC review
- Check code quality
- Verify documentation
- Ensure GPU usage (not CPU!)
- Confirm fast execution

# 3. Create/update plans
- Master plans (like this one)
- Documentation updates
- Process improvements

# 4. Push updates
git add .
git commit -m "docs: QC review and planning"
git push origin main
```

---

### For User (Decision Maker)

**Your Responsibilities**:
1. ✅ Run GPU jobs on Google Colab
2. ✅ Make go/no-go decisions
3. ✅ Provide feedback to agents
4. ✅ Monitor production performance

**Fast Workflow**:
```
# When Cloud Claude prepares Colab job:

1. Check GitHub for latest commit
2. Download .ipynb notebook
3. Upload to Colab Pro
4. Enable GPU runtime
5. Run notebook
6. Download results
7. Share with Cloud Claude (upload to server or GitHub issue)
8. Give feedback/decision
```

---

## 📊 SPEED METRICS

### Current Performance:

| Task | CPU (Slow ❌) | GPU (Fast ✅) | Speedup |
|------|--------------|--------------|---------|
| Model Training (3 models) | 180+ min | 57 min | 3.2x |
| Model Evaluation (3 models) | 60-90 min | 5-10 min | 10x |
| Feature Engineering | 30 min | 30 min | 1x |
| Data Fetching | 15 min | 15 min | 1x |

**Total Time to Production** (from now):
- ❌ CPU approach: ~6 hours
- ✅ GPU approach: ~2.5 hours (65% faster!)

---

## 📝 DOCUMENTATION REQUIREMENTS

Every step must be documented:

### Code Changes:
```bash
# Good commit message:
feat: add Colab GPU evaluation notebook

- Created colab_evaluate_50feat_models.ipynb
- Supports Tesla T4 GPU (10x faster than CPU)
- Evaluates all 3 models in 5-10 minutes
- Auto-checks promotion gates
- Outputs evaluation_results.csv

Performance: 60 min (CPU) → 10 min (GPU)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

### Process Documentation:
- Update CLAUDE.md with current status
- Create runbooks for maintenance
- Document error handling procedures
- Keep PROJECT_MEMORY.md current

### Results Documentation:
- Save all Colab outputs
- Log evaluation metrics
- Track model performance over time
- Document production incidents

---

## 🚀 IMMEDIATE NEXT STEPS (RIGHT NOW)

### Step 1: Local Claude (Me) - NOW
```
✅ Create this master plan
✅ Commit to GitHub
✅ Notify User of ready status
```

### Step 2: User - NEXT (30 min)
```
⏸️ Check cloud server for Colab files
⏸️ Download colab_evaluate_50feat_models.ipynb
⏸️ Run evaluation on Colab GPU
⏸️ Share results
```

### Step 3: Cloud Claude - AFTER USER (2 hours)
```
⏸️ Wait for evaluation results
⏸️ Process results
⏸️ Deploy if models pass gates
⏸️ Or retrain if models fail gates
```

### Step 4: Local Claude (Me) - AFTER CLOUD (30 min)
```
⏸️ QC review deployment
⏸️ Update documentation
⏸️ Mark phase complete
```

---

## 🎯 SUCCESS CRITERIA

### Phase 1 Complete When:
- ✅ 3 models evaluated on GPU
- ✅ Results documented
- ✅ Decision made (promote or retrain)
- ⏱️ Time: <1 hour

### Phase 2 Complete When:
- ✅ Passing models deployed to production
- ✅ Dry-run mode active
- ✅ Initial signals observed
- ⏱️ Time: <3 hours

### Phase 3 Complete When:
- ✅ 3-5 days of clean dry-run data
- ✅ Win rate ≥68%
- ✅ No critical errors
- ⏱️ Time: 3-5 days

### Phase 4 Complete When:
- ✅ Live trading active
- ✅ Profitable performance
- ✅ FTMO challenge passing
- ⏱️ Time: Ongoing

---

## 🔥 SPEED ENFORCEMENT

### Rules to Maintain Speed:

1. **Always use GPU** (Colab/Cloud)
   - If anyone suggests CPU training/evaluation → REJECT
   - Exception: Quick local tests (<5 min)

2. **Clear decision points** (no ambiguity)
   - Every phase has clear success criteria
   - Go/no-go decisions made immediately
   - No waiting for "perfect" solutions

3. **Parallel work** (when possible)
   - Cloud Claude preps while User runs Colab
   - Local Claude documents while Cloud deploys
   - Don't wait unnecessarily

4. **Fast iterations** (fail fast)
   - If something doesn't work, try next approach
   - Document why it failed
   - Move on quickly

5. **Document as you go** (not after)
   - Write docs while doing work
   - Commit frequently
   - Don't batch documentation

---

## 📞 BLOCKERS & ESCALATION

### If Stuck:

1. **Check this plan** - Answer probably here
2. **Check PROJECT_MEMORY.md** - Context might help
3. **Ask in GitHub issue** - Get help from other agent
4. **User decides** - Ultimate decision maker

### Common Blockers:

| Blocker | Solution | Time |
|---------|----------|------|
| "CPU is slow" | Use Colab GPU | Switch now |
| "Missing files" | Check cloud server /tmp/colab_upload/ | 5 min |
| "Unclear next step" | Read this plan | 2 min |
| "Models failing gates" | Retrain with adjusted params | 60 min |
| "Agent confusion" | Check PROJECT_MEMORY.md roles | 2 min |

---

## 📈 PROGRESS TRACKING

### Current Status: Phase 1 - GPU Evaluation

```
[█████░░░░░] 50% Complete

✅ Data collected (2 years OHLCV)
✅ Features engineered (31 features)
✅ Models trained (3 × 50-feature LSTM)
✅ Colab evaluation prepared
⏸️ Waiting: User to run Colab GPU evaluation
⬜ Model promotion
⬜ Production deployment
⬜ Live trading
```

### Timeline:

| Milestone | Target | Status |
|-----------|--------|--------|
| GPU Evaluation | Today | ⏸️ Ready |
| Model Deployment | Today + 4 hours | ⏸️ Pending |
| Dry-run Start | Today EOD | ⏸️ Pending |
| Live Trading | +5 days | ⏸️ Pending |

---

## 💡 REMEMBER

**User's Goal**: Big Data → Robust Processing → Real Results

**Our Approach**:
- ✅ Use most powerful tools (Colab GPU)
- ✅ Clear agent roles
- ✅ Fast execution
- ✅ Thorough documentation
- ✅ Quick maintenance

**Never Forget**:
- Speed is critical
- GPU beats CPU
- Document everything
- Clear communication
- Fast iterations

---

**This is the master plan. Follow it. Stay fast. Document everything. Get to production.**

🚀 Let's execute!
