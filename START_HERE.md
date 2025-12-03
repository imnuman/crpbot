# START HERE - New CLI Chat Session Instructions

**Last Updated**: 2025-12-02
**System**: HYDRA 3.0 Multi-Agent Trading System
**Progress**: 11/38 Steps Complete (28.9%) - Phase 1 Week 1 DONE

---

## 🎯 CRITICAL: Read This First

### Dual-Environment Architecture

This project operates across **TWO environments**:

```
┌─────────────────────────────────────┐
│  LOCAL MACHINE (Build Environment) │
│  - Path: /home/numan/crpbot        │
│  - RAM: 32 GB                      │
│  - Purpose: Development & Building │
│  - QC Claude runs here             │
└─────────────────────────────────────┘
              ↕ SSH
┌─────────────────────────────────────┐
│  CLOUD MACHINE (Deploy Environment)│
│  - Path: /root/crpbot              │
│  - IP: 178.156.136.185             │
│  - RAM: 16 GB                      │
│  - Purpose: Production Runtime     │
│  - Builder Claude runs here        │
└─────────────────────────────────────┘
```

### Identify Your Environment FIRST

```bash
pwd
# /home/numan/crpbot  → You are QC Claude (Local)
# /root/crpbot        → You are Builder Claude (Cloud)
```

---

## 📋 Quick Status Check

Run these commands to understand the current state:

```bash
# 1. Confirm your location
pwd

# 2. Check git branch and status
git branch
git status

# 3. Check HYDRA 3.0 implementation progress
ls -la libs/hydra/safety/guardian.py  # Should exist (Step 11 complete)
ls -la libs/hydra/mother_ai.py        # Should exist (Step 1 complete)
ls -la libs/hydra/engines/            # Should have 4 engines

# 4. Check running processes (Cloud only)
ps aux | grep -E "hydra|mother_ai|v7_runtime" | grep -v grep

# 5. Check latest commit
git log --oneline -5
```

---

## 🏗️ Current System State (2025-12-02)

### Branch
- **Active**: `feature/v7-ultimate`
- **Latest Commit**: `98f4ee2` - "Complete HYDRA 3.0 Phase 1 Week 1"

### Systems Running

**V7 Ultimate** (Legacy - Still operational):
- Status: Running on cloud
- Database: SQLite at `/root/crpbot/tradingai.db`
- Paper trades: 13 completed, 53.8% win rate
- Check: `ps aux | grep v7_runtime`

**HYDRA 3.0** (New - In Development):
- Status: Phase 1 Week 1 COMPLETE
- Components: 11/38 steps implemented
- Next: Steps 12-38 (Kill cycle, breeding, validators, etc.)

### Completed Components (Steps 1-11)

✅ **Step 1**: Mother AI orchestrator (`libs/hydra/mother_ai.py`)
✅ **Step 2**: Terminal dashboard (`apps/dashboard_terminal/`)
✅ **Step 3**: Engine A - DeepSeek (`libs/hydra/engines/engine_a_deepseek.py`)
✅ **Step 4**: Engine B - Claude (`libs/hydra/engines/engine_b_claude.py`)
✅ **Step 5**: Engine C - Grok (`libs/hydra/engines/engine_c_grok.py`)
✅ **Step 6**: Engine D - Gemini (`libs/hydra/engines/engine_d_gemini.py`)
✅ **Step 7**: Emotion prompts in all engines
✅ **Step 8**: Portfolio tracking (`libs/hydra/engine_portfolio.py`)
✅ **Step 9**: Parallel execution system
✅ **Step 10**: Tournament manager (`libs/hydra/tournament/tournament_manager.py`)
✅ **Step 11**: Guardian safety system (`libs/hydra/safety/guardian.py`)

### Pending Components (Steps 12-38)

**High Priority (Steps 12-16)**:
- Step 12: Kill cycle (24hr weak engine elimination)
- Step 13: Breeding cycle (4 days - winner teaches losers)
- Step 14: Winner teaches losers mechanism
- Step 15: Stats injection ({rank}, {wr}, {gap})
- Step 16: Weight adjustment system

**Validators (Steps 17-18)**:
- Step 17: Walk-forward validator
- Step 18: Monte-Carlo validator (< 1 sec)

**Strategy Management (Steps 19-21)**:
- Step 19: Strategy counter
- Step 20: "No edge today" mechanism
- Step 21: Edge graveyard system

**Data Feeds (Steps 22-27)**:
- Step 22: Internet Search API (Serper)
- Step 23: Order-book data feed
- Step 24: Funding rates feed
- Step 25: Liquidations feed (Coinglass)
- Step 26: 72-hour historical storage
- Step 27: API caching (cost reduction)

**Dashboard & Database (Steps 28-36)**:
- Steps 28-32: Dashboard enhancements
- Steps 33-36: Database tables

**Final (Steps 37-38)**:
- Step 37: End-to-end testing
- Step 38: Production deployment

---

## 🚀 Workflow for New Session

### If You Are QC Claude (Local Machine)

```bash
# 1. Sync latest from GitHub
cd /home/numan/crpbot
git pull origin feature/v7-ultimate

# 2. Check what needs to be built
cat HYDRA_3.0_IMPLEMENTATION_PLAN.md  # Review steps 12-38

# 3. Work on development
# - Build new features (Steps 12+)
# - Test locally if possible
# - DO NOT run production processes

# 4. When ready, commit and push
git add .
git commit -m "feat: [describe your work]"
git push origin feature/v7-ultimate

# 5. Notify Builder Claude to pull and deploy
```

### If You Are Builder Claude (Cloud Machine)

```bash
# 1. Pull latest changes
cd /root/crpbot
git pull origin feature/v7-ultimate

# 2. Check running processes
ps aux | grep -E "v7_runtime|mother_ai|hydra" | grep -v grep

# 3. Monitor logs
tail -f /tmp/v7_runtime_*.log        # V7 Ultimate
tail -f /tmp/mother_ai_*.log         # Mother AI (if running)
tail -f /tmp/hydra_*.log             # HYDRA (if running)

# 4. Deploy new features when ready
# (Follow deployment instructions from QC Claude)
```

---

## 📁 Key Files Reference

### Implementation Plan
- `HYDRA_3.0_IMPLEMENTATION_PLAN.md` - Complete 38-step blueprint

### Core HYDRA Files
```
libs/hydra/
├── mother_ai.py              # Supreme orchestrator
├── engines/
│   ├── engine_a_deepseek.py  # DeepSeek engine
│   ├── engine_b_claude.py    # Claude engine
│   ├── engine_c_grok.py      # Grok (X.AI) engine
│   └── engine_d_gemini.py    # Gemini engine
├── safety/
│   └── guardian.py           # FTMO-style safety system
├── tournament/
│   └── tournament_manager.py # P&L ranking & weighting
└── engine_portfolio.py       # Independent P&L tracking
```

### Runtimes
```
apps/runtime/
├── v7_runtime.py           # V7 Ultimate (production)
├── mother_ai_runtime.py    # HYDRA 3.0 runtime (development)
└── hydra_runtime.py        # Legacy HYDRA runtime
```

### Documentation
```
CLAUDE.md                   # Main project instructions
START_HERE.md               # This file
HYDRA_3.0_COMPLETE_2025-12-01.md
HYDRA_QUICK_REFERENCE.md
validation/INDEX.md         # Validation documentation
```

---

## 🔧 Common Tasks

### Check HYDRA 3.0 Progress

```bash
# See which steps are complete
grep -E "Step [0-9]+" HYDRA_3.0_IMPLEMENTATION_PLAN.md | head -20

# Check if key files exist
ls -la libs/hydra/safety/guardian.py
ls -la libs/hydra/tournament/tournament_manager.py
ls -la libs/hydra/mother_ai.py
```

### View Guardian Safety Rules

```bash
# View Guardian implementation
cat libs/hydra/safety/guardian.py | grep -A 10 "class Guardian"

# Check safety limits
grep -E "MAX_|MIN_" libs/hydra/safety/guardian.py
```

### Check Database

```bash
# SQLite database (cloud only)
sqlite3 tradingai.db "SELECT COUNT(*) FROM signals;"
sqlite3 tradingai.db "SELECT COUNT(*) FROM signal_results;"

# Check Guardian state
cat /root/crpbot/data/hydra/guardian_state.json
```

### Test Components

```bash
# Run tests (if available)
pytest tests/unit/
pytest tests/integration/

# Manual component test
python -c "from libs.hydra.safety import Guardian; g = Guardian(); print('Guardian OK')"
python -c "from libs.hydra.tournament import TournamentManager; print('Tournament OK')"
```

---

## 🎯 Next Steps (Steps 12+)

### Immediate Priority: Step 12 - Kill Cycle

**What**: Eliminate weakest engine every 24 hours

**Location**: Create `libs/hydra/cycles/kill_cycle.py`

**Requirements**:
- Monitor engine P&L every 24 hours
- Identify worst performer (rank #4)
- Replace with fresh instance
- Log to database

**Integration**: Called by Mother AI runtime

### After Kill Cycle: Step 13 - Breeding

**What**: Winner teaches losers every 4 days

**Location**: Create `libs/hydra/cycles/breeding_cycle.py`

**Requirements**:
- Every 4 days, extract winner's best strategies
- Inject into other 3 engines as "lessons"
- Track breeding events in database
- Mother AI manages breeding schedule

---

## 🚨 Important Notes

### DO NOT

❌ Modify `.env` files without backing up
❌ Train models locally (ALWAYS use AWS GPU)
❌ Commit API keys to GitHub
❌ Stop V7 runtime without checking with user
❌ Make breaking changes to production code
❌ Delete database files

### ALWAYS

✅ Pull latest from GitHub before starting work
✅ Check which environment you're in (local vs cloud)
✅ Read CLAUDE.md for detailed instructions
✅ Test before deploying
✅ Use TodoWrite tool to track progress
✅ Commit with clear messages
✅ Sync between local and cloud

---

## 💡 Tips for Claude

### Managing TODOs

Use the TodoWrite tool to track steps 12-38:

```
Mark step as in_progress when starting
Mark step as completed when done
Keep todo list updated throughout session
```

### Code Quality

- Follow existing patterns in `libs/hydra/`
- Use dataclasses for data structures
- Add logging with `logger.info()`, `logger.warning()`, etc.
- Include docstrings
- Type hints preferred

### Testing Strategy

1. Unit test new components
2. Integration test with existing systems
3. Manual test on cloud (Builder Claude)
4. Monitor logs for errors

---

## 📞 Getting Help

### Check Documentation

1. `CLAUDE.md` - Main project instructions
2. `HYDRA_3.0_IMPLEMENTATION_PLAN.md` - Complete blueprint
3. `HYDRA_QUICK_REFERENCE.md` - Quick commands
4. `validation/INDEX.md` - Validation docs

### Check Code Examples

Look at existing implementations:
- Guardian: `libs/hydra/safety/guardian.py`
- Tournament: `libs/hydra/tournament/tournament_manager.py`
- Engine: `libs/hydra/engines/engine_a_deepseek.py`

### Ask User

When in doubt about:
- Architecture decisions
- Breaking changes
- Production deployment
- Budget/cost concerns

---

## 🎬 Ready to Start?

### Checklist

- [ ] Identified environment (local vs cloud)
- [ ] Pulled latest from GitHub
- [ ] Checked git status and branch
- [ ] Reviewed current progress (11/38 steps)
- [ ] Understood next steps (12+)
- [ ] Read key documentation files

### First Action

```bash
# Confirm everything is ready
pwd
git status
git log --oneline -1

# You're ready to start working on Steps 12-38!
```

---

**Welcome to HYDRA 3.0 Development!**

Current mission: Implement Steps 12-38 to complete the Pre-AGI Multi-Agent Trading System.

Phase 1, Week 1 is DONE. Let's build Week 2 and beyond! 🚀
