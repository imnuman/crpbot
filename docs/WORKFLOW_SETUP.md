# Workflow Setup: Cursor + Claude Code + GitHub

This document explains how the development workflow is set up and how to keep everything synchronized between Cursor (local), Claude Code (remote AI), and GitHub.

## 🏗️ Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│     Cursor      │  push   │   GitHub Repo    │  fetch  │   Claude Code   │
│  (Your Local)   │────────▶│  imnuman/crpbot  │◀────────│  (Remote AI)    │
│                 │         │                  │         │                 │
│  - Edit code    │  pull   │  - Source of     │  push   │  - Code review  │
│  - Run tests    │◀────────│    truth         │────────▶│  - Fix issues   │
│  - Commit       │         │  - CI/CD runs    │         │  - Run tests    │
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │                            │                            │
        │                            ▼                            │
        │                   ┌─────────────────┐                  │
        │                   │   CI Pipeline   │                  │
        │                   │                 │                  │
        │                   │  - Lint (ruff)  │                  │
        └──────────────────▶│  - Type check   │◀─────────────────┘
                            │  - Security     │
                            │  - Tests        │
                            └─────────────────┘
```

## 📋 Current Session Summary

### Branch Information
- **Working Branch**: `claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih`
- **Remote**: `origin` (GitHub: imnuman/crpbot)
- **Status**: ✅ All changes committed and pushed
- **Latest Commit**: `133fb58 - test: Fix smoke tests and add unit tests for data pipeline`

### Changes Made in This Session

#### Part 1: Security Fixes 🔐
1. **Fixed exposed credentials** in `.env.example`
   - Removed real Coinbase API key and private key
   - Replaced with placeholder values
   - Added helpful setup comments

2. **Removed sensitive logging** in `libs/data/coinbase.py`
   - Changed from logging actual private key to metadata only
   - Now logs: `starts_with_header` and `length` instead of key content

3. **Verified .env protection**
   - Confirmed `.env` is in `.gitignore`

**Commit**: `a2d670a - security: Remove exposed credentials and sensitive logging`

#### Part 2: CI/CD Enforcement ⚙️
1. **Enabled mypy type checking**
   - Removed `|| true` from `.github/workflows/ci.yml:37`
   - Type errors now fail the build

2. **Enabled bandit security scanning**
   - Removed `|| true` from `.github/workflows/ci.yml:40`
   - Security issues now fail the build

3. **Added bandit configuration** to `pyproject.toml`
   - Excluded test directories
   - Skipped B101 (assert_used) for tests

**Commits**:
- `ce7edff - ci: Enable mypy and bandit enforcement`
- `52d3618 - chore: Add uv.lock for reproducible dependency resolution`

#### Part 3: Testing Infrastructure ✅
1. **Fixed smoke tests** in `tests/smoke/test_backtest_smoke.py`
   - `test_smoke_backtest`: Now simulates 20 real trades
   - `test_backtest_winrate_floor`: Validates ≥65% win rate requirement
   - Both use actual BacktestEngine and ExecutionModel
   - Complete in ~8 seconds

2. **Added unit tests** in `tests/test_data_pipeline.py`
   - `test_clean_and_validate_data_basic`: Data cleaning validation
   - `test_create_walk_forward_splits_basic`: Train/val/test splitting
   - `test_interval_mapping`: Time interval constants
   - All 3 tests passing

**Commit**: `133fb58 - test: Fix smoke tests and add unit tests for data pipeline`

### Files Modified
```
Modified:
  .env.example                           - Removed real credentials
  .github/workflows/ci.yml               - Enabled mypy/bandit enforcement
  libs/data/coinbase.py                  - Removed sensitive logging
  pyproject.toml                         - Added bandit config
  tests/smoke/test_backtest_smoke.py     - Fixed non-functional tests

Created:
  tests/test_data_pipeline.py            - New unit tests
  uv.lock                                - Dependency lock file
```

---

## 🔄 Synchronization Workflow

### 1. **Before Starting Work** (Cursor - Local)

```bash
# Fetch latest changes from GitHub
git fetch origin

# Check current branch
git status

# Pull latest changes (if working on existing branch)
git pull origin <branch-name>

# Or create new branch
git checkout -b feature/your-feature-name
```

### 2. **Making Changes** (Cursor - Local)

```bash
# Make your code changes in Cursor
# Run tests locally
uv run pytest tests/ -v

# Check code quality
uv run ruff check .
uv run ruff format .
uv run mypy .

# Stage changes
git add <files>

# Commit with clear message
git commit -m "type: description"

# Push to GitHub
git push -u origin <branch-name>
```

### 3. **Claude Code Review** (Remote AI)

When Claude Code is working on your repo:
- Claude works on a dedicated branch (e.g., `claude/review-repo-issues-*`)
- Claude commits and pushes changes automatically
- All changes are pushed to GitHub immediately
- You can see commits in real-time on GitHub

### 4. **Syncing Claude's Changes to Local** (Cursor - Local)

```bash
# Fetch all branches from GitHub
git fetch origin

# List all remote branches
git branch -r

# Switch to Claude's branch to review
git checkout claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih

# Or merge Claude's changes into your branch
git checkout your-branch
git merge claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih

# Or cherry-pick specific commits
git cherry-pick <commit-hash>
```

### 5. **Keeping Everything in Sync**

#### Daily Workflow:
```bash
# Morning: Start fresh
git fetch origin
git status
git pull origin <your-branch>

# During work: Commit frequently
git add .
git commit -m "descriptive message"
git push origin <your-branch>

# Before ending: Ensure pushed
git status  # Should show "nothing to commit, working tree clean"
```

---

## ⚠️ Important Rules to Avoid Conflicts

### 1. **Never Work on Same Branch Simultaneously**
- ❌ Bad: Both you and Claude editing `main` at same time
- ✅ Good: You work on `feature/x`, Claude on `claude/review-y`

### 2. **Always Fetch Before Starting**
```bash
# ALWAYS do this first
git fetch origin
git pull origin <your-branch>
```

### 3. **Commit and Push Frequently**
- Don't let uncommitted changes pile up
- Push at least every hour when actively working
- This keeps GitHub as source of truth

### 4. **Use Descriptive Branch Names**
- Your branches: `feature/add-telegram-bot`, `fix/api-credentials`
- Claude's branches: `claude/review-repo-issues-*` (auto-generated)

### 5. **Check GitHub Before Pulling**
- Visit: https://github.com/imnuman/crpbot/branches
- See what branches exist
- See what commits were made

---

## 🔧 Common Scenarios

### Scenario 1: Claude Made Changes, You Want to Review
```bash
# Fetch Claude's branch
git fetch origin

# List Claude's branches
git branch -r | grep claude

# Switch to Claude's branch
git checkout claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih

# Review changes
git log --oneline -5
git diff main

# If you like the changes, merge to main
git checkout main
git merge claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih
git push origin main
```

### Scenario 2: You Made Local Changes, Need to Sync
```bash
# Check status
git status

# Stage your changes
git add .

# Commit
git commit -m "feat: your changes"

# Pull latest (in case remote changed)
git pull origin <your-branch>

# Resolve conflicts if any, then
git push origin <your-branch>
```

### Scenario 3: Conflict Detected
```bash
# Pull failed due to conflicts
git pull origin <your-branch>

# Fix conflicts in files (Cursor will show them)
# Look for <<<<<<< HEAD markers

# After fixing, stage resolved files
git add <resolved-files>

# Complete the merge
git commit -m "merge: resolve conflicts with remote"

# Push
git push origin <your-branch>
```

### Scenario 4: You Want to Start Fresh from GitHub
```bash
# Discard ALL local changes (⚠️ DESTRUCTIVE)
git fetch origin
git reset --hard origin/<branch-name>

# Safer: Stash changes first
git stash
git pull origin <branch-name>
git stash pop  # Re-apply your changes
```

---

## 📊 Monitoring and Verification

### Check Sync Status (Cursor - Local)
```bash
# Are you in sync with remote?
git status

# Expected output when in sync:
# "Your branch is up to date with 'origin/<branch-name>'"
# "nothing to commit, working tree clean"

# See what's different from remote
git fetch origin
git diff origin/<branch-name>

# See commits on remote not in local
git log origin/<branch-name>..HEAD

# See commits in local not on remote
git log HEAD..origin/<branch-name>
```

### Check CI Pipeline Status
1. Visit: https://github.com/imnuman/crpbot/actions
2. See if latest commit passed all checks:
   - ✅ Lint with ruff
   - ✅ Format check
   - ✅ Type check with mypy
   - ✅ Security scan with bandit
   - ✅ Unit tests
   - ✅ Smoke tests

---

## 🎯 Best Practices

### For You (Local Development in Cursor):
1. **Pull before you start** each work session
2. **Commit small, logical changes** (not giant commits)
3. **Write clear commit messages** following conventional commits
4. **Run tests before pushing**: `uv run pytest tests/ -v`
5. **Push frequently** (every 30-60 minutes when working)

### For Claude Code (Remote AI):
- Claude automatically:
  - Creates dedicated branches
  - Commits with descriptive messages
  - Pushes immediately after changes
  - Runs tests before committing
  - You just need to fetch and review

### For Both:
- **GitHub is the source of truth** - always sync there
- **Never force push** unless absolutely necessary
- **Communicate via commit messages** - they're your async conversation
- **Review changes before merging** to main/production branches

---

## 🚀 Quick Reference Commands

```bash
# Daily startup
git fetch origin && git pull origin <branch>

# Check status
git status

# See recent commits
git log --oneline -10

# See branches
git branch -a

# Switch branch
git checkout <branch-name>

# Create and switch to new branch
git checkout -b <new-branch>

# Stage all changes
git add .

# Commit
git commit -m "type: message"

# Push
git push origin <branch-name>

# Pull Claude's latest work
git fetch origin && git checkout claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih

# Run tests
uv run pytest tests/ -v

# Run CI checks locally
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run bandit -r . -c pyproject.toml
```

---

## 📞 Troubleshooting

### Problem: "Your branch has diverged"
```bash
# Option 1: Rebase (cleaner history)
git fetch origin
git rebase origin/<branch-name>

# Option 2: Merge (safer)
git fetch origin
git merge origin/<branch-name>
```

### Problem: "Permission denied" when pushing
```bash
# Check remote URL
git remote -v

# Should be: http://127.0.0.1:44903/git/imnuman/crpbot (for Claude)
# Or: git@github.com:imnuman/crpbot.git (for you with SSH)

# Update if needed
git remote set-url origin <correct-url>
```

### Problem: "Uncommitted changes"
```bash
# Option 1: Commit them
git add .
git commit -m "wip: work in progress"

# Option 2: Stash them temporarily
git stash
# ... do other git operations ...
git stash pop

# Option 3: Discard them (⚠️ DESTRUCTIVE)
git reset --hard HEAD
```

### Problem: Lost track of what changed
```bash
# See all changes since last commit
git diff

# See staged changes
git diff --cached

# See changes in a specific file
git diff <filename>

# See commit history
git log --oneline --graph --all
```

---

## ✅ Sync Checklist

Use this before ending your work session:

- [ ] All changes committed: `git status` shows clean
- [ ] All commits pushed: `git push origin <branch>`
- [ ] CI pipeline passed: Check GitHub Actions
- [ ] Branch documented: Updated relevant docs if needed
- [ ] Ready for review: Create PR if feature complete

---

## 📚 Additional Resources

- Git Documentation: https://git-scm.com/doc
- GitHub Flow: https://docs.github.com/en/get-started/quickstart/github-flow
- Conventional Commits: https://www.conventionalcommits.org/
- Project README: `../README.md`
- Work Plan: `../WORK_PLAN.md`

---

**Last Updated**: 2025-11-07
**Session**: Claude Code review session for repository issue fixes
**Branch**: `claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih`
