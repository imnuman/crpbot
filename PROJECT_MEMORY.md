# Persistent Memory for Claude Instances

**Purpose**: This file ensures continuity between Claude sessions by documenting our dual-environment development setup and current project state.

**Last Updated**: 2025-11-13

---

## 🎯 Critical: Dual Environment Setup

### Our Development Architecture

```
┌─────────────────────────┐
│  LOCAL MACHINE          │
│  (You are here)         │
│                         │
│  Path: /home/numan/crpbot
│  User: numan            │
│  Role: QC Review        │
│  Tools: Local Claude    │
└──────────┬──────────────┘
           │
           │ Git push/pull
           │ (GitHub: imnuman/crpbot)
           ▼
┌─────────────────────────┐
│  CLOUD SERVER           │
│  (Production/Training)  │
│                         │
│  Path: ~/crpbot         │
│  User: root             │
│  Role: Main Development │
│  Tools: Cloud Claude    │
└─────────────────────────┘
```

### Key Understanding

1. **TWO ACTIVE CLAUDE INSTANCES**:
   - **Local Claude (YOU)**: Works on `/home/numan/crpbot` (local machine)
   - **Cloud Claude**: Works on `~/crpbot` (cloud server via SSH)

2. **SYNC MECHANISM**:
   - Both environments sync via **Git** (GitHub repo: `imnuman/crpbot`)
   - Code changes pushed to GitHub by either instance
   - Other instance pulls to stay in sync

3. **YOUR ROLE (Local Claude)**:
   - **QC Reviewer**: Review cloud Claude's work
   - **Documentation**: Maintain CLAUDE.md and guides
   - **Verification**: Test changes locally
   - **Coordination**: Help sync between environments

4. **Cloud Claude's Role**:
   - **Main Development**: Primary code development
   - **Training**: Run model training on cloud server
   - **Production**: Deploy and run runtime
   - **Problem Solving**: Debug and fix issues

---

## 📝 Session Continuity Protocol

### When Starting a New Chat:

**1. READ THIS FILE FIRST** (you're doing it now!)

**2. Check Latest Sync Status**:
```bash
git fetch origin
git log origin/main -5 --oneline
git status
```

**3. Read Recent Commit Messages**:
```bash
git log -10 --pretty=format:"%h - %s (%ar)"
```

**4. Check for QC Reviews**:
```bash
ls -lt QC_REVIEW_*.md | head -3
```

**5. Read Latest Status Files**:
- `CLAUDE.md` - Project architecture (always up-to-date)
- `PHASE6_5_RESTART_PLAN.md` - Current training status
- `reports/phase6_5/CRITICAL_*.md` - Any blocking issues

---

## 🔄 Current Project Status

### Phase: 6.5 - Model Training & Evaluation

**Last Major Update**: 2025-11-13

**Current Situation**:
- ✅ Feature mismatch discovered (50 vs 31 features)
- ✅ Cloud Claude identified and documented the issue
- ✅ Solution created: Retrain with 31-feature files
- ✅ Local Claude (QC) reviewed and APPROVED
- ⏸️ **BLOCKED**: Waiting for manual Colab retraining

**Next Action**:
- User must download feature files from cloud server
- Upload to Google Drive
- Run Colab training (~57 minutes for 3 models)
- After training: Evaluate and promote models

**Key Files to Check**:
1. `reports/phase6_5/CRITICAL_FEATURE_MISMATCH_REPORT.md`
2. `COLAB_RETRAINING_INSTRUCTIONS.md`
3. `QC_REVIEW_CLOUD_CLAUDE_2025-11-13.md`

---

## 🎭 Role Definitions

### Local Claude (YOU)

**Primary Responsibilities**:
1. **Quality Control**: Review cloud Claude's commits
2. **Testing**: Run tests locally to verify changes
3. **Documentation**: Keep CLAUDE.md and guides updated
4. **Sync Coordination**: Help manage git sync between environments

**When to Act**:
- After cloud Claude pushes commits: Review and QC
- User asks for local testing
- Documentation needs updates
- Sync issues need resolution

**Common Tasks**:
```bash
# Pull latest from cloud
git pull origin main

# Review recent commits
git log -5 --stat

# Run local tests
make test

# Create QC report
# (see QC_REVIEW_*.md for template)

# Push updates
git push origin main
```

### Cloud Claude

**Primary Responsibilities**:
1. **Development**: Write and modify code
2. **Training**: Run model training on GPU/cloud resources
3. **Debugging**: Fix bugs and issues
4. **Production**: Deploy and monitor runtime

**Their Workflow**:
- Work directly on cloud server
- Commit and push to GitHub
- Wait for local Claude QC review
- Continue based on feedback

---

## 📋 Communication Protocol

### How We Communicate Across Sessions

**1. Git Commit Messages**:
```bash
# Good commit message format:
"feat: add new feature X

- Details about what changed
- Why it was changed
- Any issues blocked/unblocked

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>"
```

**2. Status Files**:
- `QC_REVIEW_*.md`: Local Claude's reviews
- `reports/phase6_5/*.md`: Progress reports
- `*_STATUS.md`: Current status documents

**3. CLAUDE.md Updates**:
- Keep "Current Project Status" section up-to-date
- Update "Training Progress" section
- Document any architecture changes

---

## 🚨 Important Conventions

### File Naming

**QC Reviews**: `QC_REVIEW_CLOUD_CLAUDE_YYYY-MM-DD.md`
**Status Reports**: `PROJECT_STATUS_YYYY-MM-DD.md`
**Critical Issues**: `reports/phase6_5/CRITICAL_*.md`

### Branch Strategy

- **main**: Production-ready code (we work here)
- **dev**: Development work (if needed)
- Feature branches: For major features

### Testing Requirements

Before pushing:
1. ✅ Code passes `make lint`
2. ✅ Code passes `make fmt`
3. ✅ Tests pass `make test` (if applicable locally)
4. ✅ No sensitive data in commits

---

## 🔍 Quick Context Check

### When User Says: "Sync with cloud Claude"

**Your Actions**:
1. Pull latest: `git pull origin main`
2. Review commits: `git log origin/main -5`
3. Check for issues: Look for CRITICAL_*.md files
4. Perform QC review if needed
5. Report status to user

### When User Says: "What's the current status?"

**Your Response**:
1. Check git log: Last 5 commits
2. Read PHASE6_5_RESTART_PLAN.md
3. Check reports/phase6_5/ for latest status
4. Summarize: What's done, what's blocked, what's next

### When User Says: "Review cloud Claude's work"

**Your Process**:
1. Pull latest changes
2. Review commit messages and diffs
3. Check code quality (architecture, tests, docs)
4. Verify safety (no secrets, no breaking changes)
5. Create QC_REVIEW_*.md
6. Commit and push review

---

## 📊 Current Technical Context

### Models

**Architecture**:
- **Old**: LSTM 64/2/False (62K params) - OBSOLETE
- **New**: LSTM 128/3/True (1M+ params) - TARGET

**Current Models**:
- 3 old models trained (incompatible - 50 features)
- Need retraining with 31-feature files

**Feature Set**: 31 features (see CLAUDE.md for details)

### Data

**Location**:
- Raw: `data/raw/*.parquet` (2 years of 1m OHLCV)
- Features: `data/features/features_*_1m_*.parquet`
- Models: `models/` (old), `models/new/` (needs retraining)

**Symbols**: BTC-USD, ETH-USD, SOL-USD

### Infrastructure

**Local Machine**:
- Path: `/home/numan/crpbot`
- Python: 3.10+ with uv
- Database: SQLite (dev)

**Cloud Server**:
- Path: `~/crpbot`
- Python: 3.10+ with uv
- Database: PostgreSQL RDS
- AWS: S3 for data/models

---

## 🎯 Session Initialization Checklist

When starting a new chat, verify:

- [x] Read this file (PERSISTENT_MEMORY.md)
- [ ] Check git status: `git fetch && git status`
- [ ] Review recent commits: `git log -5 --oneline`
- [ ] Check for blockers: `ls reports/phase6_5/CRITICAL_*.md`
- [ ] Read CLAUDE.md "Current Project Status" section
- [ ] Understand current phase and next actions
- [ ] Ready to assist based on latest context

---

## 💡 Pro Tips for Future Sessions

1. **Always start with git sync**:
   ```bash
   git fetch origin && git log origin/main -5 --oneline
   ```

2. **Check for emergency files**:
   ```bash
   ls -lt *CRITICAL* *EMERGENCY* *URGENT* 2>/dev/null | head -5
   ```

3. **Verify your role**:
   - If on local machine → You're QC Claude
   - If on cloud server → You're Development Claude

4. **When in doubt, read**:
   - CLAUDE.md for architecture
   - PHASE6_5_RESTART_PLAN.md for current status
   - Latest QC_REVIEW_*.md for recent decisions

5. **Maintain continuity**:
   - Update this file when major changes occur
   - Leave clear commit messages for next session
   - Document decisions in appropriate .md files

---

## 📞 Emergency Recovery

### If Context is Lost

1. **Read in order**:
   - This file (PERSISTENT_MEMORY.md)
   - CLAUDE.md
   - `git log -20 --oneline`
   - Latest QC_REVIEW_*.md

2. **Reconstruct state**:
   ```bash
   # Where are we?
   git log -1

   # What's changed recently?
   git log -10 --stat

   # Any blockers?
   find . -name "*CRITICAL*" -o -name "*BLOCKED*" | head

   # Current phase?
   grep -A 5 "Current Project Status" CLAUDE.md
   ```

3. **Ask user for clarification**:
   "I've reviewed the git history and documentation. We're currently at [phase/status]. Is this correct? What would you like me to focus on?"

---

## 🔒 Security Reminder

**Never commit**:
- ❌ `.env` files
- ❌ `.db_password`
- ❌ `*.pem` (private keys)
- ❌ AWS credentials
- ❌ API keys in code

**Always check**:
```bash
git diff --cached | grep -i "api_key\|password\|secret\|token"
```

---

## 📝 Update Log

| Date | Update | Author |
|------|--------|--------|
| 2025-11-13 | Initial creation | Local Claude (QC) |
| 2025-11-13 | Added dual environment documentation | Local Claude (QC) |
| 2025-11-13 | Documented current feature mismatch status | Local Claude (QC) |

---

**Remember**: This file is your anchor between sessions. Keep it updated, and future you will thank present you! 🚀
