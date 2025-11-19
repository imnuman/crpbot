# Sync Protocol: QC Claude ↔ Builder Claude ↔ GitHub

**Date**: 2025-11-18
**Purpose**: Ensure continuous sync between Local Claude (QC), Cloud Claude (Builder), and GitHub repo
**Critical**: This protocol prevents conflicts and ensures all environments stay in sync

---

## 🎯 Three-Way Sync Architecture

```
┌────────────────────────┐                    ┌────────────────────────┐
│   QC CLAUDE            │                    │   BUILDER CLAUDE       │
│   (Local Machine)      │                    │   (Cloud Server)       │
│                        │                    │                        │
│  /home/numan/crpbot    │                    │  /root/crpbot          │
│  Role: Review, Train   │                    │  Role: Prod, Deploy    │
└───────────┬────────────┘                    └───────────┬────────────┘
            │                                             │
            │ git pull (before work)                      │ git pull (before work)
            │ git push (after work)                       │ git push (after work)
            │                                             │
            └──────────────┐               ┌─────────────┘
                           │               │
                           ▼               ▼
                    ┌──────────────────────────┐
                    │   GITHUB REPO            │
                    │   (CENTRAL HUB)          │
                    │                          │
                    │  github.com/imnuman/crpbot
                    │  Branches: main, feature/*
                    │  ★ SOURCE OF TRUTH ★    │
                    └──────────────────────────┘
```

**Golden Rule**: **GitHub is the CENTRAL HUB**. Both Claudes:
1. **PULL from GitHub BEFORE starting work** (understand current state)
2. **PUSH to GitHub AFTER finishing task** (share changes)

---

## 📋 Pre-Session Checklist

### For QC Claude (Local Machine)

**BEFORE starting any work**:

```bash
# 1. Check current status
pwd  # Should be: /home/numan/crpbot
git status

# 2. Fetch latest from GitHub
git fetch origin

# 3. Check for remote changes
git log HEAD..origin/main --oneline

# 4. Pull if behind
git pull origin main

# 5. Check for Builder Claude's recent work
git log -5 --pretty=format:"%h - %s (%ar) <%an>"
```

**AFTER completing work**:

```bash
# 1. Stage changes
git add <files>

# 2. Commit with clear message
git commit -m "type(scope): description

- Detailed changes
- Why this change was made

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 3. Pull again (in case Builder Claude pushed)
git pull --rebase origin main

# 4. Push to GitHub
git push origin main

# 5. Verify push
git status
```

---

### For Builder Claude (Cloud Server)

**BEFORE starting any work**:

```bash
# 1. Verify environment
pwd  # Should be: /root/crpbot
hostname  # Should be cloud server

# 2. Check running processes (don't interrupt production)
ps aux | grep -E "v7_runtime|dashboard" | grep -v grep

# 3. Fetch latest from GitHub
git fetch origin

# 4. Check current branch
git branch --show-current

# 5. Check for QC Claude's changes
git log HEAD..origin/$(git branch --show-current) --oneline

# 6. Pull if behind
git pull origin $(git branch --show-current)
```

**AFTER completing work**:

```bash
# 1. Stage changes
git add <files>

# 2. Commit with clear message
git commit -m "type(scope): description

- What changed
- Why it changed
- Impact on production

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 3. Pull before push (critical!)
git pull --rebase origin $(git branch --show-current)

# 4. Push to GitHub
git push origin $(git branch --show-current)

# 5. Verify
git status
```

---

## 🔄 Daily Sync Routine

### Morning Sync (Both Environments)

```bash
#!/bin/bash
# scripts/morning_sync.sh

echo "🌅 Morning Sync Starting..."
echo ""

# Fetch latest
echo "📥 Fetching from GitHub..."
git fetch origin

# Check status
BRANCH=$(git branch --show-current)
echo "📍 Current branch: $BRANCH"

# Check if behind
BEHIND=$(git rev-list --count HEAD..origin/$BRANCH)
if [ "$BEHIND" -gt 0 ]; then
    echo "⚠️  You are $BEHIND commits behind origin/$BRANCH"
    echo "📥 Pulling latest changes..."
    git pull origin $BRANCH
else
    echo "✅ You are up to date with origin/$BRANCH"
fi

# Check for uncommitted changes
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -gt 0 ]; then
    echo "⚠️  You have $UNCOMMITTED uncommitted changes"
    git status --short
else
    echo "✅ No uncommitted changes"
fi

# Show recent commits
echo ""
echo "📝 Recent commits:"
git log -5 --pretty=format:"%C(yellow)%h%Creset %s %C(cyan)(%ar)%Creset %C(green)<%an>%Creset" --abbrev-commit

echo ""
echo "✅ Morning sync complete!"
```

---

### End-of-Session Sync (Both Environments)

```bash
#!/bin/bash
# scripts/end_session_sync.sh

echo "🌙 End of Session Sync..."
echo ""

# Check for uncommitted changes
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -gt 0 ]; then
    echo "⚠️  You have $UNCOMMITTED uncommitted changes"
    git status --short
    echo ""
    read -p "Do you want to commit these changes? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Please commit manually with a descriptive message"
        exit 1
    fi
fi

# Check if ahead of remote
BRANCH=$(git branch --show-current)
AHEAD=$(git rev-list --count origin/$BRANCH..HEAD)
if [ "$AHEAD" -gt 0 ]; then
    echo "📤 You are $AHEAD commits ahead of origin/$BRANCH"
    echo "🚀 Pushing to GitHub..."
    git push origin $BRANCH
    echo "✅ Pushed successfully"
else
    echo "✅ Already synced with GitHub"
fi

echo ""
echo "✅ End of session sync complete!"
```

---

## ⚠️ Conflict Resolution Protocol

### If Local and Remote Have Diverged

**Option 1: Rebase (Preferred)**

```bash
# Stash local changes
git stash save "WIP: $(date +%Y-%m-%d-%H%M%S)"

# Pull with rebase
git pull --rebase origin main

# Apply stashed changes
git stash pop

# If conflicts:
# 1. Resolve in editor
# 2. git add <resolved-files>
# 3. git rebase --continue

# Verify and push
git push origin main
```

**Option 2: Merge (If Rebase Fails)**

```bash
# Pull with merge
git pull origin main

# Resolve conflicts in editor
# Look for <<<<<<< HEAD markers

# After resolving:
git add <resolved-files>
git commit -m "fix: resolve merge conflicts"
git push origin main
```

---

## 🚨 Emergency Procedures

### Builder Claude: Production is Running, Need to Sync

```bash
# 1. Check what's running
ps aux | grep -E "v7_runtime|dashboard"

# 2. Stash any local changes
git stash save "WIP: pre-sync backup $(date +%Y-%m-%d-%H%M%S)"

# 3. Pull latest
git pull origin feature/v7-ultimate

# 4. Verify critical files didn't change
git diff stash@{0} apps/runtime/v7_runtime.py

# 5. If safe, pop stash
git stash pop

# 6. If conflicts, resolve carefully (production is running!)
# Consider restarting services after sync
```

### QC Claude: Builder Claude Pushed, Need to Review

```bash
# 1. Fetch and check what changed
git fetch origin
git log HEAD..origin/main --oneline

# 2. Review changes before pulling
git diff HEAD..origin/main

# 3. If approved, pull
git pull origin main

# 4. Test locally if needed
make test
```

---

## 📊 Sync Status Monitoring

### Create Sync Status Script

```bash
#!/bin/bash
# scripts/check_sync_status.sh

echo "🔍 Sync Status Check"
echo "===================="
echo ""

# Environment
HOSTNAME=$(hostname)
PWD=$(pwd)
echo "🖥️  Environment: $HOSTNAME"
echo "📂 Directory: $PWD"
echo ""

# Current branch
BRANCH=$(git branch --show-current)
echo "🌿 Branch: $BRANCH"
echo ""

# Fetch latest
git fetch origin -q

# Local vs Remote
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/$BRANCH)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "✅ SYNCED: Local and remote are identical"
else
    echo "⚠️  OUT OF SYNC: Local and remote differ"

    AHEAD=$(git rev-list --count origin/$BRANCH..HEAD)
    BEHIND=$(git rev-list --count HEAD..origin/$BRANCH)

    if [ "$AHEAD" -gt 0 ]; then
        echo "   📤 Local is $AHEAD commits AHEAD (need to push)"
        echo ""
        echo "   Unpushed commits:"
        git log origin/$BRANCH..HEAD --pretty=format:"   %C(yellow)%h%Creset %s %C(cyan)(%ar)%Creset" --abbrev-commit
        echo ""
    fi

    if [ "$BEHIND" -gt 0 ]; then
        echo "   📥 Local is $BEHIND commits BEHIND (need to pull)"
        echo ""
        echo "   Commits not in local:"
        git log HEAD..origin/$BRANCH --pretty=format:"   %C(yellow)%h%Creset %s %C(cyan)(%ar)%Creset %C(green)<%an>%Creset" --abbrev-commit
        echo ""
    fi
fi

echo ""

# Uncommitted changes
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -eq 0 ]; then
    echo "✅ No uncommitted changes"
else
    echo "⚠️  $UNCOMMITTED uncommitted file(s):"
    git status --short
fi

echo ""
echo "===================="
```

**Usage**:

```bash
# Make executable
chmod +x scripts/check_sync_status.sh scripts/morning_sync.sh scripts/end_session_sync.sh

# Run sync check anytime
./scripts/check_sync_status.sh

# Run at start of session
./scripts/morning_sync.sh

# Run at end of session
./scripts/end_session_sync.sh
```

---

## 🎯 Coordination Protocol

### When QC Claude Makes Changes

1. ✅ **Document changes** in commit message
2. ✅ **Update CLAUDE.md** if architecture changes
3. ✅ **Create handoff doc** if Builder Claude needs to deploy
4. ✅ **Push to GitHub**
5. ✅ **Optional**: Create GitHub issue for Builder Claude to track

**Example Handoff**:

```bash
cat > HANDOFF_TO_BUILDER_$(date +%Y%m%d).md <<EOF
# Handoff to Builder Claude - $(date +%Y-%m-%d)

## Changes Made
- [List specific changes]
- [Files modified]

## Testing Done
- [Tests run locally]
- [Verification steps]

## Deployment Steps
1. Pull latest: \`git pull origin main\`
2. [Specific commands to run]
3. [Services to restart if needed]

## Expected Impact
- [What will change in production]
- [Any downtime expected]

## Rollback Plan
- [How to revert if issues occur]
EOF

git add HANDOFF_TO_BUILDER_$(date +%Y%m%d).md
git commit -m "docs: handoff for $(date +%Y-%m-%d)"
git push origin main
```

### When Builder Claude Makes Changes

1. ✅ **Test on cloud** before committing
2. ✅ **Commit with detailed message**
3. ✅ **Push to GitHub immediately**
4. ✅ **Update CLAUDE.md** if production changes
5. ✅ **Optional**: Notify QC Claude via commit message or issue

---

## 🔒 Protected Files (Never Commit)

```gitignore
# Secrets
.env
.env.local
*.pem
*.key
*_secrets.json

# Production Data
tradingai.db
*.db-journal
/data/raw/*
/data/features/*

# Logs
*.log
/tmp/*.log

# Models (use DVC or S3)
/models/*.pt
/models/*.pth
```

---

## ✅ Quick Reference

### Pre-Work (ALWAYS)

```bash
git pull origin $(git branch --show-current)
```

### Post-Work (ALWAYS)

```bash
git add .
git commit -m "type: description"
git push origin $(git branch --show-current)
```

### Check Sync

```bash
./scripts/check_sync_status.sh
```

### Emergency: Discard Local Changes

```bash
# ⚠️ DESTRUCTIVE - Only if you're sure
git fetch origin
git reset --hard origin/$(git branch --show-current)
```

### Emergency: Force Push (USE WITH CAUTION)

```bash
# ⚠️ DANGEROUS - Coordinate with other Claude first!
# Only use if you're absolutely sure
git push --force-with-lease origin $(git branch --show-current)
```

---

## 📅 Sync Schedule

### Daily
- **Morning**: Run `morning_sync.sh`
- **End of session**: Run `end_session_sync.sh`

### Before Any Work
- **Pull latest**: `git pull`

### After Any Commit
- **Push immediately**: `git push`

### Weekly
- **Review sync status**: Check for any drift
- **Clean up branches**: Delete merged feature branches
- **Update documentation**: Keep CLAUDE.md current

---

## 🎓 Training: Sync Scenarios

### Scenario 1: Both Claudes Work Simultaneously

**QC Claude**: Working on documentation
**Builder Claude**: Fixing production bug

**Solution**:
1. Both pull before starting
2. Work on different files (documentation vs code)
3. Both commit and push separately
4. No conflicts expected (different files)

### Scenario 2: Both Claudes Modify Same File

**QC Claude**: Updates CLAUDE.md
**Builder Claude**: Also updates CLAUDE.md

**Solution**:
1. First to push wins (no conflict)
2. Second to push: `git pull --rebase`
3. Resolve any conflicts manually
4. `git push` after resolving

### Scenario 3: Builder Claude Has Production Changes, QC Claude Needs to Pull

**Builder Claude**: Pushed V7 bug fixes
**QC Claude**: About to start work

**Solution**:
1. QC Claude runs `morning_sync.sh`
2. Sees Builder Claude's commits
3. Reviews changes with `git log` and `git diff`
4. Pulls and tests locally before making own changes

---

**Last Updated**: 2025-11-18
**Maintained By**: Both QC Claude & Builder Claude
**Status**: Active Protocol - Follow Strictly
