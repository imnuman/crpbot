# Workflow Setup: Cursor + Claude Code + Amazon Q + GitHub

This document explains how the development workflow is set up and how to keep everything synchronized between Cursor (local), Claude Code (remote AI), Amazon Q (AWS AI), and GitHub.

## 🏗️ Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│     Cursor      │  push   │   GitHub Repo    │  fetch  │   Claude Code   │
│  (Your Local)   │────────▶│  imnuman/crpbot  │◀────────│  (Remote AI)    │
│                 │         │                  │         │                 │
│  - Edit code    │  pull   │  - Source of     │  push   │  - Code review  │
│  - Run tests    │◀────────│    truth         │────────▶│  - Fix issues   │
│  - Development  │         │  - CI/CD runs    │         │  - Refactoring  │
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │                            │                            │
        │                            │                            │
┌─────────────────┐                 │                            │
│    Amazon Q     │                 │                            │
│  (Your Local)   │                 │                            │
│                 │  push           │                            │
│  - AWS infra    │────────────────▶│                            │
│  - Lambda       │         pull    │                            │
│  - Deployment   │◀────────────────│                            │
│  - Cloud svcs   │                 │                            │
└─────────────────┘                 │                            │
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

## 🤖 AI Tool Selection Guide

### When to Use Each Tool

**🎯 Cursor (Local Development)**
- General Python development and coding
- Local testing and debugging
- Quick edits and refactoring
- Real-time code completion
- File navigation and exploration

**🔍 Claude Code (Remote AI - Advanced Analysis)**
- Code reviews and issue detection
- Complex refactoring across multiple files
- Architecture analysis and improvements
- Documentation generation
- Test creation and validation
- Breaking down large tasks into parts

**☁️ Amazon Q (AWS Specialist)**
- AWS infrastructure setup (EC2, S3, RDS, etc.)
- Lambda function development and deployment
- AWS SDK integration (boto3)
- CloudFormation/CDK templates
- AWS best practices and security
- Cost optimization for AWS resources
- Debugging AWS-specific errors
- Setting up CI/CD with AWS services

**💡 Best Practice**: Use the right tool for the job:
- Start with **Cursor** for day-to-day coding
- Bring in **Claude Code** for code quality and complex issues
- Use **Amazon Q** for anything AWS-related

### Example Task Routing

| Task | Tool | Reason |
|------|------|--------|
| Fix bug in trading algorithm | Cursor or Claude Code | Core Python logic |
| Add database models | Cursor or Claude Code | SQLAlchemy/Python |
| Deploy to AWS Lambda | **Amazon Q** | AWS deployment |
| Set up S3 bucket for data | **Amazon Q** | AWS infrastructure |
| Code review entire repo | **Claude Code** | Complex analysis |
| Add new trading feature | Cursor | Local development |
| Optimize AWS costs | **Amazon Q** | AWS expertise |
| Create CloudWatch alarms | **Amazon Q** | AWS monitoring |
| Refactor constants | Claude Code | Multi-file changes |
| Debug boto3 connection | **Amazon Q** | AWS SDK issue |

## 🔀 Working with Multiple AI Assistants

### Branch Strategy for Multiple AIs

To avoid conflicts when multiple AI assistants work on the same repository:

**1. Dedicated Branches per Tool**
```bash
# Claude Code uses auto-generated branches
claude/review-repo-issues-<session-id>

# Amazon Q can use feature branches for AWS work
aws/lambda-deployment
aws/s3-setup
aws/cloudwatch-monitoring

# Your development branches
feature/your-feature-name
fix/bug-description
```

**2. Never Have Two AIs on Same Branch**
- ❌ BAD: Claude Code and Amazon Q both editing `main`
- ✅ GOOD: Claude on `claude/review-*`, Amazon Q on `aws/lambda-*`, you on `feature/trading-bot`

**3. Sync Strategy**
```bash
# Before starting with any tool, always fetch latest
git fetch origin

# Check what branches exist
git branch -r

# Pull your working branch
git pull origin <your-branch>
```

### Coordination Workflow

**Scenario: AWS Infrastructure + Code Changes**

1. **Plan the work**: Decide what goes where
   - Core trading logic → Cursor/Claude Code
   - AWS deployment → Amazon Q
   - Code review → Claude Code

2. **Sequential approach** (safer):
   ```bash
   # Step 1: Amazon Q sets up AWS infrastructure
   git checkout -b aws/lambda-setup
   # ... Amazon Q creates Lambda, IAM roles, etc.
   git push origin aws/lambda-setup

   # Step 2: You/Cursor integrates AWS SDK
   git checkout -b feature/aws-integration
   git merge aws/lambda-setup
   # ... Add boto3 code, environment variables
   git push origin feature/aws-integration

   # Step 3: Claude Code reviews integration
   # Claude works on claude/review-aws-integration
   # ... Reviews code, adds tests, fixes issues
   ```

3. **Parallel approach** (when tasks are independent):
   ```bash
   # Amazon Q: Sets up S3 buckets (aws/s3-setup)
   # Claude Code: Refactors trading logic (claude/refactor-signals)
   # You in Cursor: Adds new features (feature/telegram-bot)

   # All three can work simultaneously, then merge in order
   ```

### Merge Order Best Practice

When merging multiple AI contributions:

```bash
# 1. Merge infrastructure first (foundation)
git checkout main
git merge aws/lambda-setup

# 2. Merge code that uses infrastructure
git merge feature/aws-integration

# 3. Merge code quality improvements last (cleanup)
git merge claude/review-aws-integration

# 4. Push merged main
git push origin main
```

### Communication via Commit Messages

Since AIs can't directly communicate, use commit messages:

```bash
# Amazon Q creates infrastructure
git commit -m "aws: Create Lambda function for signal processing

Lambda ARN: arn:aws:lambda:us-east-1:123456789:function:crpbot-signals
Environment: production
Memory: 512MB
Timeout: 30s

TODO: Integration code should use environment variable LAMBDA_ARN"

# You see this and add integration
git commit -m "feat: Integrate Lambda for signal processing

Uses LAMBDA_ARN from environment (set by Amazon Q)
Refs: aws/lambda-setup commit abc123"
```

### Automated Review (Bugbot)

- Install the [Bugbot GitHub App](https://github.com/apps/bugbot) on the repository (admin required).
- When you open a PR, comment `@Bugbot review` to trigger an automated defect scan.
- Claude follows up on Bugbot findings, Cursor applies fixes, and Amazon Q handles any AWS concerns.
- Full setup details live in `docs/BUGBOT_SETUP.md`.

### Avoiding Conflicts Checklist

Before starting work with ANY tool:

- [ ] `git fetch origin` - Get latest changes
- [ ] `git status` - Ensure clean working tree
- [ ] `git branch -r` - Check what branches exist
- [ ] Choose a unique branch name for your work
- [ ] Pull any branches you need to base your work on

During work:

- [ ] Commit frequently with clear messages
- [ ] Push regularly to GitHub (every 30-60 min)
- [ ] Check GitHub web UI to see what others are doing
- [ ] Don't edit files another AI is actively working on

After work:

- [ ] Push final changes: `git push origin <branch>`
- [ ] Document what was done in commit message
- [ ] Create PR if ready for review/merge
- [ ] Notify team (via docs/commits) about new branches

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
6. **Coordinate AI assistants** - ensure they work on different branches

### For Claude Code (Remote AI):
- Claude automatically:
  - Creates dedicated branches (claude/*)
  - Commits with descriptive messages
  - Pushes immediately after changes
  - Runs tests before committing
  - You just need to fetch and review
- Best for: Code review, refactoring, testing, documentation

### For Amazon Q (AWS Specialist):
- Amazon Q should work on dedicated AWS branches (aws/*)
- Use for all AWS infrastructure and deployment tasks
- Document AWS resource ARNs and configs in commit messages
- Test AWS changes in a dev environment first
- Include AWS cost estimates in commit messages when relevant
- Best for: Lambda, S3, RDS, CloudFormation, boto3, AWS SDK

### For All Tools:
- **GitHub is the source of truth** - always sync there
- **Never force push** unless absolutely necessary
- **Communicate via commit messages** - they're your async conversation
- **Review changes before merging** to main/production branches
- **Use dedicated branches** - one branch per tool/feature
- **Tag AWS-related commits** with `aws:` prefix for easy filtering

---

## 🚀 Quick Reference Commands

### General Git Commands
```bash
# Daily startup
git fetch origin && git pull origin <branch>

# Check status
git status

# See recent commits
git log --oneline -10

# See branches (including remote)
git branch -a

# See only AWS branches
git branch -r | grep aws/

# See only Claude branches
git branch -r | grep claude/

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

# Run tests
uv run pytest tests/ -v

# Run CI checks locally
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run bandit -r . -c pyproject.toml
```

### Working with Claude Code
```bash
# Pull Claude's latest work
git fetch origin
git checkout claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih

# Review what Claude changed
git log --oneline -5
git diff main

# Merge Claude's changes to your branch
git checkout your-branch
git merge claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih
```

### Working with Amazon Q
```bash
# Start AWS infrastructure work
git checkout -b aws/lambda-deployment

# After Amazon Q makes changes in VS Code, review them
git status
git diff

# Commit AWS changes (Amazon Q may do this automatically)
git add .
git commit -m "aws: Setup Lambda function for signal processing

Lambda ARN: arn:aws:lambda:region:account:function:name
Runtime: Python 3.11
Memory: 512MB"

# Push AWS branch
git push origin aws/lambda-deployment

# Pull AWS changes from another machine
git fetch origin
git checkout aws/lambda-deployment

# Merge AWS infrastructure to main (after testing)
git checkout main
git merge aws/lambda-deployment
git push origin main
```

### Multi-AI Workflow
```bash
# See all active work
git fetch origin
git branch -r | grep -E "(aws/|claude/|feature/)"

# Pull changes from all AI branches
git fetch origin

# Merge in order (infrastructure → features → cleanup)
git checkout main
git merge aws/s3-setup          # Infrastructure first
git merge feature/data-pipeline  # Features that use infrastructure
git merge claude/code-review     # Code quality improvements last
git push origin main
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

**Last Updated**: 2025-11-08
**Session**: Multi-AI workflow integration (Cursor + Claude Code + Amazon Q)
**Branch**: `claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih`
**Status**: ✅ Amazon Q integration documented and ready for AWS tasks
