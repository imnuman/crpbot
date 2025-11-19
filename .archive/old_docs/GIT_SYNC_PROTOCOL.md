# Git Sync Protocol - Dual Claude Workflow

**Principle**: Both Claudes ALWAYS commit and push after completing tasks
**Goal**: Stay synchronized via GitHub, no manual file transfers

---

## 🔄 Commit Frequency Rules

### Cloud Claude (Builder) - Commit After:
- ✅ Data fetched from Coinbase
- ✅ Features engineered
- ✅ Models trained
- ✅ Evaluation completed
- ✅ Tests run
- ✅ Any code changes
- ✅ Configuration updates
- ✅ Documentation created

### Local Claude (QC) - Commit After:
- ✅ QC review completed
- ✅ Model promotion approved
- ✅ Documentation updated
- ✅ Test validation
- ✅ Approval/rejection decisions
- ✅ Issue reports created

---

## 📝 Commit Message Format

### Template
```bash
git commit -m "<type>: <short description>

<detailed description>
- Bullet point 1
- Bullet point 2

Impact: <what this enables>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Types
- `feat:` New feature or functionality
- `fix:` Bug fix or correction
- `docs:` Documentation only
- `test:` Test-related changes
- `refactor:` Code refactoring
- `perf:` Performance improvement
- `chore:` Maintenance tasks
- `data:` Data fetching or engineering
- `model:` Model training or evaluation
- `deploy:` Deployment or infrastructure

### Examples

**Good Commit Messages:**
```bash
feat: train BTC LSTM model with 39 features on GPU

- Trained on 2 years of Coinbase data (2023-11-10 to 2025-11-10)
- Model architecture: 2-layer bidirectional LSTM, 64 hidden units
- Training time: 4.2 minutes on Colab Pro GPU
- Validation accuracy: 68.3%
- Model file: models/lstm_BTC_USD_1m_a7aff5c4.pt (249 KB)

Impact: Ready for QC evaluation against 68% promotion gate

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

```bash
data: fetch and engineer features for BTC-USD

- Fetched 1,030,512 rows from Coinbase (2023-11-10 to 2025-10-25)
- Engineered 39 features (5 OHLCV + 31 indicators + 3 categorical)
- Data quality: Zero nulls, complete OHLCV data
- File: data/features/features_BTC-USD_1m_latest.parquet (35 MB)
- Uploaded to S3: s3://crpbot-market-data-dev/data/features/

Impact: Ready for LSTM training on GPU

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

```bash
docs: approve BTC LSTM model for Phase 6.5

QC Review Results:
- Accuracy: 68.3% ✅ (gate: ≥68%)
- Calibration error: 4.2% ✅ (gate: ≤5%)
- Data quality: No issues ✅
- Tests: All passing ✅

Decision: APPROVED for promotion
- Promoted to: models/promoted/lstm_BTC_USD_promoted.pt
- Ready for Phase 6.5 observation period

Impact: 1/3 models approved, awaiting ETH and SOL evaluations

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

**Bad Commit Messages:**
```bash
# Too vague
git commit -m "update"
git commit -m "fix stuff"
git commit -m "wip"

# No context
git commit -m "train model"
git commit -m "add features"

# Missing co-author
git commit -m "feat: add new feature"  # ← Missing Claude co-author
```

---

## 🔄 Synchronization Workflow

### Cloud Claude Workflow
```bash
# 1. Start work
cd /root/crpbot
source .venv/bin/activate
git pull origin main  # ← Always pull first!

# 2. Do work (train, test, develop)
# ... work happens ...

# 3. Add all changes
git add .

# 4. Check what's being committed
git status
git diff --cached

# 5. Commit with detailed message
git commit -m "feat: train BTC LSTM model

- Training details
- Results
- Impact

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>"

# 6. Push immediately
git push origin main

# 7. Report to Local Claude
echo "✅ Pushed to GitHub, ready for QC review"
```

### Local Claude Workflow
```bash
# 1. Receive notification from Cloud Claude
# "Training complete, pushed to GitHub"

# 2. Pull latest changes
cd /home/numan/crpbot
source .venv/bin/activate
git pull origin main  # ← Get Cloud Claude's work

# 3. Review changes
git log --oneline -5
git show HEAD

# 4. Download artifacts from S3
aws s3 sync s3://crpbot-market-data-dev/models/ models/

# 5. Run QC validation
# ... validation happens ...

# 6. Document decision
# ... update docs ...

# 7. Commit QC results
git add .
git commit -m "docs: approve BTC LSTM model

QC Review: APPROVED
- Accuracy: 68.3% ✅
- Promoted to production

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>"

# 8. Push back to GitHub
git push origin main

# 9. Notify Cloud Claude
echo "✅ QC approved, pushed to GitHub"
```

---

## 📊 Sync Patterns

### Pattern 1: Sequential Work
```
Cloud: Work → Commit → Push
  ↓ (GitHub)
Local: Pull → Review → Commit → Push
  ↓ (GitHub)
Cloud: Pull → Continue
```

### Pattern 2: Parallel Work (Different Files)
```
Cloud: Work on training → Commit → Push
Local: Work on docs → Commit → Push

Both: Pull to sync
  ↓
Auto-merge (no conflicts if different files)
```

### Pattern 3: Conflict Resolution
```
Cloud: Edit file A → Commit → Push
Local: Edit file A → Commit → Push FAILS ❌

Local: git pull → Resolve conflict → Commit → Push ✅
```

---

## 🚨 Conflict Prevention

### Rule 1: Always Pull Before Push
```bash
# WRONG ❌
git add . && git commit -m "..." && git push

# CORRECT ✅
git pull origin main && git add . && git commit -m "..." && git push
```

### Rule 2: Communicate Before Editing
```
Cloud: "I'm editing apps/runtime/main.py"
Local: "OK, I'll work on docs only"
```

### Rule 3: Use Separate Branches for Experimental Work
```bash
# Cloud: Experimental feature
git checkout -b experiment/new-feature
# ... work ...
git commit -m "..." && git push origin experiment/new-feature

# Local: Review branch
git fetch origin
git checkout experiment/new-feature
# ... review ...
# Merge if approved
git checkout main && git merge experiment/new-feature
```

---

## 📋 Mandatory Sync Points

### After Every Training Run
```bash
# Cloud Claude MUST commit after training
git add models/*.pt data/features/*.parquet
git commit -m "model: train BTC LSTM on GPU (68.3% accuracy)"
git push origin main
```

### After Every QC Review
```bash
# Local Claude MUST commit after review
git add docs/*.md models/promoted/*.pt
git commit -m "docs: approve BTC LSTM for Phase 6.5"
git push origin main
```

### After Feature Engineering
```bash
# Cloud Claude MUST commit after features
git add data/features/*.parquet scripts/*.py
git commit -m "data: engineer 39 features for BTC/ETH/SOL"
git push origin main
```

### After Evaluation
```bash
# Cloud Claude MUST commit evaluation results
git add reports/*.json docs/*.md
git commit -m "test: evaluate LSTM models (BTC: 68.3%, ETH: 67.8%, SOL: 69.1%)"
git push origin main
```

---

## 🎯 Quick Reference

### Cloud Claude Daily Workflow
```bash
# Morning
git pull origin main

# Throughout the day (after each task)
git add . && git commit -m "..." && git push origin main

# Evening
git add . && git commit -m "..." && git push origin main
```

### Local Claude Daily Workflow
```bash
# Morning
git pull origin main

# After Cloud pushes
git pull origin main
# ... QC review ...
git add . && git commit -m "..." && git push origin main

# Evening
git add . && git commit -m "..." && git push origin main
```

---

## 🔍 Verification Commands

### Check Sync Status
```bash
# Are we behind origin?
git status

# See what's different from origin
git fetch origin
git log HEAD..origin/main --oneline

# See unpushed commits
git log origin/main..HEAD --oneline
```

### View Recent Activity
```bash
# Last 10 commits
git log --oneline -10

# Commits from last 24 hours
git log --since="24 hours ago" --oneline

# Show who committed what
git log --pretty=format:"%h - %an: %s" -10
```

---

## 📈 Success Metrics

### Good Sync Health
- ✅ Both environments within 5 commits of each other
- ✅ No unpushed commits older than 1 hour
- ✅ All major tasks have corresponding commits
- ✅ Commit messages are descriptive
- ✅ No merge conflicts

### Poor Sync Health
- ❌ Environments 20+ commits apart
- ❌ Unpushed changes from hours/days ago
- ❌ "WIP" or "update" commit messages
- ❌ Frequent merge conflicts
- ❌ Manual file transfers via SCP

---

## 🎓 Best Practices

1. **Commit Often** - Small, frequent commits > Large, rare commits
2. **Pull Before Push** - Always sync before pushing
3. **Descriptive Messages** - Future you will thank you
4. **Test Before Commit** - Run `make test` before committing
5. **Review Before Push** - `git diff --cached` to see what's being committed
6. **Use Branches** - For experimental/breaking changes
7. **Co-Author Credit** - Always include Claude co-author line
8. **No Secrets** - Never commit `.env`, `.db_password`, AWS keys

---

## 🚀 Automation Ideas (Future)

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash

# Run tests before allowing commit
make test || {
    echo "❌ Tests failed, commit aborted"
    exit 1
}

# Check for secrets
if grep -r "aws_secret_access_key\|password\|token" --exclude-dir=.git .; then
    echo "❌ Potential secret detected, commit aborted"
    exit 1
fi

echo "✅ Pre-commit checks passed"
```

### Post-commit Hook
```bash
# .git/hooks/post-commit
#!/bin/bash

# Auto-push after commit (optional, use with caution)
git push origin main
```

---

**Remember: Frequent, descriptive commits = Happy collaboration!** 🎉
