# Quick Start for Claude Instances

**When starting a new chat, do this**:

## 1. Identify Your Environment (5 seconds)

```bash
pwd
```

- If `/home/numan/crpbot` → You're **Local Claude** (QC role)
- If `~/crpbot` (or `/root/crpbot`) → You're **Cloud Claude** (Development role)

## 2. Read Context Files (1 minute)

**Required reading**:
1. `PROJECT_MEMORY.md` - Session continuity, dual environment setup
2. `CLAUDE.md` - Full project architecture and current status

**Quick status check**:
```bash
git fetch origin
git log origin/main -5 --oneline
ls -lt reports/phase6_5/*.md | head -3
```

## 3. Understand Your Role

### If You're Local Claude (QC):
- ✅ Review cloud Claude's commits
- ✅ Run local tests
- ✅ Create QC reviews (QC_REVIEW_*.md)
- ✅ Update documentation
- ❌ Don't make major code changes (that's cloud Claude's job)

### If You're Cloud Claude (Development):
- ✅ Write and modify code
- ✅ Run training and evaluation
- ✅ Debug and fix issues
- ✅ Deploy to production
- ❌ Don't forget to push commits for QC review

## 4. Check Current Status

```bash
# What's the latest work?
git log -3 --stat

# Any blocking issues?
find . -name "*CRITICAL*" -type f 2>/dev/null

# Current phase?
grep -A 10 "Current Project Status" CLAUDE.md
```

## 5. Ready to Work!

**Common first questions from user**:
- "Sync with cloud Claude" → Pull latest, review commits, report status
- "What's the current status?" → Read PHASE6_5_RESTART_PLAN.md, summarize
- "Review cloud Claude's work" → QC process (see PROJECT_MEMORY.md)

---

## Emergency Context Recovery

If you have no context:

```bash
# 1. Read these files in order:
cat PROJECT_MEMORY.md
cat CLAUDE.md | head -200
git log -10 --oneline

# 2. Check for recent QC reviews:
ls -lt QC_REVIEW_*.md | head -1
cat $(ls -t QC_REVIEW_*.md | head -1)

# 3. Read latest status:
cat PHASE6_5_RESTART_PLAN.md

# 4. Now you have context!
```

---

## Key Files Reference

| File | Purpose | When to Read |
|------|---------|--------------|
| `PROJECT_MEMORY.md` | Session continuity | Every new chat |
| `CLAUDE.md` | Full architecture | Every new chat |
| `PHASE6_5_RESTART_PLAN.md` | Current training status | For status updates |
| `QC_REVIEW_*.md` | Recent reviews | Before new work |
| `reports/phase6_5/CRITICAL_*.md` | Blocking issues | When checking blockers |

---

## Remember

1. ✅ **Always read PROJECT_MEMORY.md first**
2. ✅ **Check git log for recent work**
3. ✅ **Know your role** (Local QC vs Cloud Dev)
4. ✅ **Sync before starting** (`git pull origin main`)
5. ✅ **Leave clear commits** for next session

**That's it! You're ready to continue the work.** 🚀
