# Persistent Memory for Claude Instances

**Purpose**: This file ensures continuity between Claude sessions by documenting our dual-environment development setup and current project state.

**Last Updated**: 2025-11-15

---

## 🎯 Critical: Dual Environment Setup

### Our Development Architecture

```
┌─────────────────────────┐
│  LOCAL MACHINE          │
│  (Local Claude)         │
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
│  Path: /root/crpbot     │
│  User: root             │
│  Role: Main Development │
│  Tools: Cloud Claude    │
└─────────────────────────┘
```

### Key Understanding

1. **TWO ACTIVE CLAUDE INSTANCES**:
   - **Local Claude**: Works on `/home/numan/crpbot` (local machine)
   - **Cloud Claude (YOU)**: Works on `/root/crpbot` (cloud server via SSH)

2. **SYNC MECHANISM**:
   - Both environments sync via **Git** (GitHub repo: `imnuman/crpbot`)
   - Code changes pushed to GitHub by either instance
   - Other instance pulls to stay in sync

3. **LOCAL CLAUDE's ROLE**:
   - **QC Reviewer**: Review cloud Claude's work
   - **Documentation**: Maintain CLAUDE.md and guides
   - **Verification**: Test changes locally
   - **Coordination**: Help sync between environments

4. **YOUR ROLE (Cloud Claude)**:
   - **Main Development**: Primary code development
   - **Training**: Run model training on GPU/cloud resources
   - **Debugging**: Fix bugs and issues
   - **Production**: Deploy and monitor runtime

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

### Phase: V5 - Data Upgrade & Microstructure Features

**Last Major Update**: 2025-11-15 (CRITICAL PIVOT)

**🚨 MAJOR STRATEGIC DECISION**:
- ❌ **V4 OBSOLETE**: 50% accuracy ceiling due to noisy free Coinbase data
- ✅ **V5 PIVOT**: Upgrade to Tardis.dev professional market data
- ✅ **UPGRADE STRATEGY**: 10% change (data layer), 90% reuse (architecture, runtime, FTMO rules)

**Current Situation**:
- 🔴 **V4 Models**: Stuck at 50% accuracy - ROOT CAUSE: Free data too noisy
- 🟢 **V5 Solution**: Professional tick data + order book from Tardis.dev
- 🟡 **Budget Approved**: $197/month Phase 1 (validation), $549/month Phase 2 (live trading)
- ⏸️ **BLOCKED**: Waiting for Tardis.dev subscription

**V5 Phase 1 Timeline (4 weeks)**:
- Week 1: Download Tardis historical data (2+ years, tick-level)
- Week 2: Build 53 features (33 existing + 20 microstructure)
- Week 3: Train models with professional data
- Week 4: Validate (target: 65-75% accuracy vs V4's 50%)

**Next Action**:
1. 🚀 Subscribe to Tardis.dev Historical - $147/month
   - URL: https://tardis.dev/pricing
   - Plan: Historical (3 exchanges: Binance, Coinbase, Kraken)
   - Get: Tick data, order book, 2+ years historical
2. Create V5 data pipeline integration
3. Engineer 20 new microstructure features (bid-ask spread dynamics, order flow imbalance, etc.)
4. Retrain models with professional data

**Key Files to Check**:
1. `V5_PHASE1_PLAN.md` - V5 roadmap (to be created)
2. `CLAUDE.md` - Updated with V5 strategy
3. `PROJECT_MEMORY.md` - This file

---

## 🎭 Role Definitions

### Local Claude

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

### Cloud Claude (YOU)

**Primary Responsibilities**:
1. **Development**: Write and modify code
2. **Training**: Run model training on GPU/cloud resources
3. **Debugging**: Fix bugs and issues
4. **Production**: Deploy and monitor runtime

**Your Workflow**:
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

### When User Says: "Sync with local Claude"

**Your Actions**:
1. Pull latest: `git pull origin main`
2. Review commits: `git log origin/main -5`
3. Check for issues: Look for CRITICAL_*.md files
4. Check for QC reviews: `ls -lt QC_REVIEW_*.md`
5. Report status to user

### When User Says: "What's the current status?"

**Your Response**:
1. Check git log: Last 5 commits
2. Read PHASE6_5_RESTART_PLAN.md
3. Check reports/phase6_5/ for latest status
4. Summarize: What's done, what's blocked, what's next

### When Local Claude Reviews Your Work

**Your Process**:
1. Read the QC_REVIEW_*.md file
2. Address any issues or concerns raised
3. Make necessary fixes
4. Commit and push updates
5. Update status files as needed

---

## 📊 Current Technical Context

### V5 Data Strategy

**Data Sources**:
- ✅ **Phase 1 (NOW)**: Tardis.dev Historical - $147/month
  - Tick-level data (all trades)
  - Full order book snapshots
  - 2+ years historical (Binance, Coinbase, Kraken)
- ✅ **Real-time**: Coinbase API - Free (already integrated)
- 🟡 **Phase 3-5 (Later)**: On-chain (Glassnode), News (CryptoPanic), Sentiment (LunarCrush)

**Feature Set Evolution**:
- ❌ **V4**: 31-50 features (free OHLCV data) → 50% accuracy ceiling
- ✅ **V5**: 53 features (professional tick data) → Target 65-75% accuracy
  - 33 existing features (OHLCV, technicals, sessions)
  - 20 NEW microstructure features:
    - Bid-ask spread dynamics
    - Order flow imbalance
    - Volume-weighted metrics
    - Order book pressure
    - Tick-level volatility
    - Market microstructure indicators

### Models

**Architecture** (90% UNCHANGED):
- **LSTM**: 128/3/bidirectional (1M+ params) - REUSABLE
- **Transformer**: Multi-coin, 4-layer - REUSABLE
- **RL Agent**: PPO stub - REUSABLE
- **Ensemble**: 35/40/25 weights - REUSABLE

**What Changes in V5**:
- ✅ Input features: 31-50 → 53 features (microstructure additions)
- ✅ Data source: Free Coinbase → Tardis.dev professional
- ❌ Architecture: NO CHANGE (models stay the same)
- ❌ Runtime/FTMO: NO CHANGE (90% code reuse)

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

**Cloud Server (YOU ARE HERE)**:
- Path: `/root/crpbot`
- Python: 3.10+ with uv
- Database: PostgreSQL RDS
- AWS: S3 for data/models

---

## 🎯 Session Initialization Checklist

When starting a new chat, verify:

- [x] Read this file (PROJECT_MEMORY.md)
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
   - This file (PROJECT_MEMORY.md)
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
| 2025-11-13 | Resolved merge conflict, updated cloud path | Cloud Claude |
| 2025-11-15 | **CRITICAL PIVOT**: V4→V5 data upgrade strategy | QC Claude → Cloud Claude |
| 2025-11-15 | Updated status: Tardis.dev integration, 53 features, $197/mo budget | Cloud Claude |

---

**Remember**: This file is your anchor between sessions. Keep it updated, and future you will thank present you! 🚀
