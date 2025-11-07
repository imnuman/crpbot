# Workflow Sync Setup: Cursor IDE, Claude AI, and GitHub

This guide ensures smooth synchronization between **Cursor IDE** (local development), **Claude AI** (code review & assistance), and **GitHub** (repository & collaboration).

---

## 🎯 Overview

### The Three-Way Sync Architecture

```
┌─────────────────┐
│   Cursor IDE    │ ← Local development & code editing
│  (Local Machine)│
└────────┬────────┘
         │ git push/pull
         ▼
┌─────────────────┐
│   GitHub Repo   │ ← Central repository & version control
│  (Remote Sync)  │
└────────┬────────┘
         │ API access / repo connection
         ▼
┌─────────────────┐
│   Claude AI     │ ← Code review, suggestions, fixes
│  (AI Assistant) │
└─────────────────┘
```

**Key Principle**: GitHub is the **single source of truth**. Both Cursor and Claude read from and write to GitHub.

---

## 📋 Prerequisites

### 1. GitHub Repository
- ✅ Repository created: `https://github.com/imnuman/crpbot`
- ✅ Branch: `main`
- ✅ Remote configured: `origin/main`

### 2. Cursor IDE Setup
- ✅ Cursor installed on local machine
- ✅ Git configured with user credentials
- ✅ Repository cloned locally

### 3. Claude AI Access
- ✅ Claude account (Claude.ai or Anthropic API)
- ✅ GitHub integration enabled (if using Claude via GitHub integration)

---

## 🔧 Setup Instructions

### Part 1: Cursor IDE Configuration

#### 1.1 Verify Git Configuration

```bash
# Check git config
git config --global user.name "imnuman"
git config --global user.email "imnuman@users.noreply.github.com"

# Verify remote
git remote -v
# Should show:
# origin  https://github.com/imnuman/crpbot.git (fetch)
# origin  https://github.com/imnuman/crpbot.git (push)
```

#### 1.2 Cursor IDE Git Integration

Cursor IDE automatically detects Git repositories. Verify:

1. **Open Cursor IDE** in your project directory (`/home/numan/crpbot`)
2. **Check Source Control** (Ctrl+Shift+G or Cmd+Shift+G):
   - Should show branch: `main`
   - Should show sync status: "Up to date with origin/main"
3. **Enable Auto-Fetch**:
   - Go to Settings → Git
   - Enable "Auto Fetch"
   - Set fetch interval to 5 minutes (recommended)

#### 1.3 Cursor Git Settings

```json
// In Cursor settings (settings.json)
{
  "git.enableSmartCommit": true,
  "git.confirmSync": false,
  "git.autofetch": true,
  "git.autofetchPeriod": 5,
  "git.postCommitCommand": "sync",
  "git.showPushSuccessNotification": true
}
```

#### 1.4 Pre-Commit Hooks (Already Configured)

```bash
# Verify pre-commit hooks are installed
pre-commit --version

# If not installed:
make setup
# or
pre-commit install
```

---

### Part 2: Claude AI Integration

#### 2.1 Option A: Claude via GitHub Integration (Recommended)

**If using Claude.ai with GitHub integration:**

1. **Connect Claude to GitHub**:
   - Go to Claude.ai → Settings → Integrations
   - Connect your GitHub account
   - Grant access to `crpbot` repository
   - Enable "Read and Write" permissions (for code review)

2. **Using Claude in GitHub**:
   - Claude can access the repository directly
   - Can review code, suggest changes, and provide feedback
   - Changes should be committed via Cursor IDE

#### 2.2 Option B: Claude via Repository URL

**If using Claude with manual repo access:**

1. **Provide Repository URL**:
   ```
   https://github.com/imnuman/crpbot
   ```

2. **Claude can**:
   - Read repository files
   - Review code
   - Suggest changes
   - Provide implementation guidance

3. **Important**: Claude cannot directly push to GitHub. All changes must be:
   - Applied in Cursor IDE
   - Committed via Git
   - Pushed to GitHub

#### 2.3 Option C: Claude via API (Advanced)

**If using Anthropic API:**

```python
# Example: Using Claude API to review code
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

# Fetch code from GitHub
# Review with Claude
# Apply suggestions in Cursor
```

---

### Part 3: Synchronization Workflow

#### 3.1 Daily Workflow

**Morning Routine**:
```bash
# 1. Pull latest changes from GitHub
git pull origin main

# 2. Check status
git status

# 3. Verify you're up to date
git log --oneline -5
```

**During Development**:
1. Make changes in Cursor IDE
2. Pre-commit hooks run automatically (format, lint)
3. Test changes locally
4. Commit frequently with descriptive messages

**Before Pushing**:
```bash
# 1. Check status
git status

# 2. Pull latest changes (in case someone else pushed)
git pull --rebase origin main

# 3. Verify no conflicts
git status

# 4. Push changes
git push origin main
```

#### 3.2 Code Review Workflow

**When Claude reviews code**:

1. **Claude analyzes code** (from GitHub or provided context)
2. **Claude provides suggestions** (in chat or via comments)
3. **You apply changes in Cursor IDE**
4. **Commit and push**:
   ```bash
   git add .
   git commit -m "fix: Apply Claude's suggestions for [issue]"
   git push origin main
   ```
5. **Claude can verify** the changes were applied correctly

#### 3.3 Conflict Resolution

**If conflicts occur**:

```bash
# 1. Pull with rebase
git pull --rebase origin main

# 2. If conflicts:
# - Resolve in Cursor IDE
# - Stage resolved files: git add <file>
# - Continue rebase: git rebase --continue

# 3. Push resolved changes
git push origin main
```

---

## 🔄 Sync Best Practices

### 1. **Always Pull Before Starting Work**
```bash
git pull origin main
```

### 2. **Commit Frequently**
- Small, focused commits
- Descriptive commit messages
- Follow conventional commits: `feat:`, `fix:`, `docs:`, etc.

### 3. **Push After Each Session**
```bash
# At end of work session
git status
git add .
git commit -m "feat: [your changes]"
git push origin main
```

### 4. **Verify Sync Status**
```bash
# Check if local is ahead/behind
git status

# Compare local vs remote
git log HEAD..origin/main  # Commits on remote not in local
git log origin/main..HEAD  # Commits in local not on remote
```

### 5. **Use Branches for Features**
```bash
# Create feature branch
git checkout -b feat/new-feature

# Work on feature
# ...

# Push feature branch
git push origin feat/new-feature

# Merge to main (via PR or direct)
git checkout main
git merge feat/new-feature
git push origin main
```

---

## 🛠️ Troubleshooting

### Issue 1: Local Changes Conflict with Remote

**Solution**:
```bash
# Stash local changes
git stash

# Pull latest
git pull origin main

# Apply stashed changes
git stash pop

# Resolve conflicts if any
# Then commit and push
```

### Issue 2: Cursor Shows Out-of-Sync Warning

**Solution**:
```bash
# Refresh git status
git fetch origin

# Pull latest
git pull origin main

# Verify
git status
```

### Issue 3: Claude Can't Access Latest Code

**Solution**:
1. Ensure code is pushed to GitHub
2. If using GitHub integration, refresh Claude's access
3. Provide direct file content to Claude if needed

### Issue 4: Pre-commit Hooks Fail

**Solution**:
```bash
# Fix formatting
make fmt

# Fix linting
make lint

# Try commit again
git commit -m "your message"
```

### Issue 5: Large Files or Binary Data

**Solution**:
- Use `.gitignore` for large files
- Use DVC for data/models (already configured)
- Never commit `.env` files (already in `.gitignore`)

---

## 📊 Sync Status Check Script

Create a script to verify sync status:

```bash
#!/bin/bash
# scripts/check_sync.sh

echo "🔍 Checking Sync Status"
echo "======================"

# Check git status
echo ""
echo "📊 Git Status:"
git status --short

# Check if up to date
echo ""
echo "🔄 Sync Status:"
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "✅ Local and remote are in sync"
else
    echo "⚠️  Local and remote differ"
    echo "   Local:  $LOCAL"
    echo "   Remote: $REMOTE"
    
    # Check commits ahead/behind
    AHEAD=$(git rev-list --count HEAD ^origin/main)
    BEHIND=$(git rev-list --count origin/main ^HEAD)
    
    if [ "$AHEAD" -gt 0 ]; then
        echo "   📤 Local is $AHEAD commits ahead (need to push)"
    fi
    
    if [ "$BEHIND" -gt 0 ]; then
        echo "   📥 Local is $BEHIND commits behind (need to pull)"
    fi
fi

# Check uncommitted changes
echo ""
echo "📝 Uncommitted Changes:"
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -eq 0 ]; then
    echo "✅ No uncommitted changes"
else
    echo "⚠️  $UNCOMMITTED uncommitted file(s)"
    git status --short
fi

echo ""
echo "======================"
```

**Usage**:
```bash
chmod +x scripts/check_sync.sh
./scripts/check_sync.sh
```

---

## ✅ Verification Checklist

### Initial Setup Verification

- [ ] Git configured with correct user name/email
- [ ] Remote repository configured (`origin/main`)
- [ ] Cursor IDE detects Git repository
- [ ] Pre-commit hooks installed and working
- [ ] Claude AI has access to repository (if using integration)
- [ ] Can push/pull without issues

### Daily Sync Verification

- [ ] `git status` shows clean working tree (or intentional changes)
- [ ] `git log --oneline -5` shows recent commits
- [ ] Local branch is up to date with `origin/main`
- [ ] No uncommitted sensitive files (`.env`, secrets)
- [ ] All changes pushed to GitHub

### After Each Session

- [ ] All changes committed
- [ ] Changes pushed to `origin/main`
- [ ] GitHub shows latest commits
- [ ] Claude can access latest code (if needed)

---

## 🚀 Quick Reference Commands

### Daily Commands

```bash
# Start of day
git pull origin main                    # Pull latest
git status                              # Check status

# During development
make fmt                                # Format code
make lint                               # Check linting
make test                               # Run tests

# End of day
git add .                               # Stage changes
git commit -m "feat: [description]"     # Commit
git push origin main                    # Push to GitHub
```

### Sync Commands

```bash
# Check sync status
git status
git log --oneline -5
git fetch origin
git log HEAD..origin/main               # See remote changes
git log origin/main..HEAD               # See local changes

# Sync with remote
git pull origin main                    # Pull and merge
git pull --rebase origin main           # Pull with rebase
git push origin main                    # Push to remote
```

### Emergency Commands

```bash
# Discard local changes
git restore <file>                      # Restore single file
git restore .                           # Restore all files

# Stash changes
git stash                               # Stash changes
git stash pop                           # Restore stashed changes

# Reset to remote
git fetch origin
git reset --hard origin/main            # ⚠️ DANGEROUS: Discards all local changes
```

---

## 📚 Additional Resources

- **Git Documentation**: https://git-scm.com/doc
- **Cursor IDE Git**: https://cursor.sh/docs/git
- **Claude GitHub Integration**: https://claude.ai/integrations
- **Conventional Commits**: https://www.conventionalcommits.org/

---

## 🎯 Summary

### The Golden Rules

1. **GitHub is the source of truth** - Always sync with GitHub
2. **Pull before you push** - Avoid conflicts
3. **Commit frequently** - Small, focused commits
4. **Push regularly** - Keep remote up to date
5. **Use branches for features** - Keep main stable
6. **Claude reviews, you implement** - Apply changes in Cursor

### Workflow Summary

```
Morning → git pull → Work in Cursor → Commit → Push → End of Day
                ↓                           ↓
            Claude reviews              GitHub updated
                ↓                           ↓
        Suggestions applied          All synced ✅
```

---

**Last Updated**: 2025-11-06  
**Repository**: https://github.com/imnuman/crpbot  
**Branch**: main

