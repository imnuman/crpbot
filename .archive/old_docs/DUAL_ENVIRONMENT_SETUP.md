# Dual Environment Setup Guide

Setting up synchronized local and cloud environments for CRPBot development.

## Overview

This guide helps you maintain **two active environments**:
1. **Local Machine** (`/home/numan/crpbot`) - Development & testing
2. **Cloud Server** (`~/crpbot`) - Production & training

Both environments stay synchronized via Git and can be accessed by Claude Code.

## Architecture

```
┌─────────────────────┐
│   Local Machine     │
│  (Development)      │
│                     │
│  - Code editing     │
│  - Quick tests      │
│  - Claude Code      │
└──────────┬──────────┘
           │
           │ git push/pull
           ▼
┌─────────────────────┐
│   GitHub/GitLab     │
│  (Source of Truth)  │
└──────────┬──────────┘
           │
           │ git push/pull
           ▼
┌─────────────────────┐
│   Cloud Server      │
│   (Production)      │
│                     │
│  - Long training    │
│  - Production runs  │
│  - Claude Code      │
└─────────────────────┘
```

## Prerequisites

### 1. SSH Key Setup for Cloud Server

Since your server uses SSH key authentication:

```bash
# On local machine, check if you have SSH keys
ls -la ~/.ssh/id_*

# If no keys exist, generate new SSH key pair
ssh-keygen -t ed25519 -C "your-email@example.com"
# Press Enter to accept default location (~/.ssh/id_ed25519)
# Set a passphrase (recommended) or press Enter for no passphrase

# Copy public key to cloud server
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@your-cloud-server-ip

# Or manually copy:
cat ~/.ssh/id_ed25519.pub
# Then SSH to server and add to ~/.ssh/authorized_keys

# Test connection (should work without password)
ssh -i ~/.ssh/id_ed25519 user@your-cloud-server-ip
```

### 2. Git Repository Setup

```bash
# Ensure you have a Git remote (GitHub/GitLab)
cd /home/numan/crpbot
git remote -v

# If no remote exists, create one:
# 1. Create repo on GitHub/GitLab
# 2. Add remote:
git remote add origin git@github.com:YOUR_USERNAME/crpbot.git
# or for HTTPS:
git remote add origin https://github.com/YOUR_USERNAME/crpbot.git

# Push current code
git add .
git commit -m "Initial commit for dual environment setup"
git push -u origin main
```

### 3. SSH Config for Easy Connection

Create/edit `~/.ssh/config` on your local machine:

```bash
# Edit SSH config
nano ~/.ssh/config

# Add this configuration:
Host crpbot-cloud
    HostName your-cloud-server-ip
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3

# Save and test
ssh crpbot-cloud  # Should connect without password
```

Now you can use `ssh crpbot-cloud` instead of the full command!

## Initial Sync Setup

### Step 1: Prepare Local Environment

```bash
cd /home/numan/crpbot

# Ensure all changes are committed
git status
git add .
git commit -m "Sync checkpoint: $(date +%Y-%m-%d)"
git push origin main

# Tag current state (optional but recommended)
git tag -a v1.0-pre-cloud -m "State before cloud deployment"
git push origin v1.0-pre-cloud

# Verify everything works locally
source .venv/bin/activate
make test
python test_runtime_connection.py
```

### Step 2: Deploy to Cloud with SSH Key

I'll update the deployment script to support SSH keys:

```bash
cd /home/numan/crpbot

# Deploy using SSH key
./scripts/deploy_to_cloud.sh -k ~/.ssh/id_ed25519 user@your-cloud-server-ip

# Or using SSH config alias
./scripts/deploy_to_cloud.sh crpbot-cloud
```

### Step 3: Verify Both Environments

**Local:**
```bash
cd /home/numan/crpbot
source .venv/bin/activate
python --version
git log --oneline -1
make test
```

**Cloud:**
```bash
ssh crpbot-cloud
cd ~/crpbot
source .venv/bin/activate
python --version
git log --oneline -1
make test
```

Both should show identical git commits and pass tests.

## Daily Workflow

### Workflow 1: Develop Locally, Deploy to Cloud

```bash
# 1. Work on local machine
cd /home/numan/crpbot
source .venv/bin/activate

# 2. Make changes, test locally
# ... edit files ...
make test

# 3. Commit and push
git add .
git commit -m "Add new feature X"
git push origin main

# 4. Sync to cloud
ssh crpbot-cloud "cd ~/crpbot && git pull origin main && source .venv/bin/activate && uv pip install -e ."

# Or use sync script (I'll create this)
./scripts/sync_to_cloud.sh
```

### Workflow 2: Train on Cloud, Download Results

```bash
# 1. Start training on cloud
ssh crpbot-cloud
cd ~/crpbot
source .venv/bin/activate
make train COIN=BTC EPOCHS=20

# 2. While training, continue working locally
# ... on local machine ...

# 3. After training completes, download models
# On local machine:
rsync -avz --progress crpbot-cloud:~/crpbot/models/ /home/numan/crpbot/models/

# Or upload to S3 from cloud, then download locally
```

### Workflow 3: Hotfix on Cloud, Sync Back

```bash
# 1. Make urgent fix on cloud
ssh crpbot-cloud
cd ~/crpbot
# ... make changes ...
git add .
git commit -m "Hotfix: issue X"
git push origin main

# 2. Pull changes to local
# On local machine:
cd /home/numan/crpbot
git pull origin main
```

## Keeping Environments Synced

### Automated Sync Scripts

I'll create these scripts for you:

1. **sync_to_cloud.sh** - Push local changes to cloud
2. **sync_from_cloud.sh** - Pull cloud changes to local
3. **sync_models.sh** - Sync models between environments
4. **sync_data.sh** - Sync data between environments

### Manual Sync Commands

**Sync code:**
```bash
# Local → Cloud
cd /home/numan/crpbot
git push origin main
ssh crpbot-cloud "cd ~/crpbot && git pull origin main"

# Cloud → Local
ssh crpbot-cloud "cd ~/crpbot && git push origin main"
cd /home/numan/crpbot
git pull origin main
```

**Sync models:**
```bash
# Local → Cloud
rsync -avz --progress /home/numan/crpbot/models/ crpbot-cloud:~/crpbot/models/

# Cloud → Local
rsync -avz --progress crpbot-cloud:~/crpbot/models/ /home/numan/crpbot/models/

# Via S3 (recommended for large files)
# Upload from either machine:
aws s3 sync models/ s3://crpbot-market-data-dev/models/
# Download on other machine:
aws s3 sync s3://crpbot-market-data-dev/models/ models/
```

**Sync data:**
```bash
# Always use S3 for data (764MB is too large for direct transfer)
# Upload:
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/
# Download:
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/
```

## Claude Code Integration

### Using Claude Code on Local Machine

Claude Code can already access your local environment at `/home/numan/crpbot`.

**Benefits:**
- Fast local development
- Quick tests and iterations
- No network latency

**Configuration:**
```bash
# Ensure Claude Code has access to local project
cd /home/numan/crpbot
code .  # If using VS Code with Claude Code extension

# Or use Claude Code CLI
claude-code /home/numan/crpbot
```

### Using Claude Code on Cloud Server

You have two options:

**Option A: SSH + Remote Development (Recommended)**

Use VS Code Remote-SSH or similar:

1. Install VS Code with Remote-SSH extension
2. Configure SSH connection:
   ```
   Host crpbot-cloud
       HostName your-server-ip
       User ubuntu
       IdentityFile ~/.ssh/id_ed25519
   ```
3. Connect to cloud server via Remote-SSH
4. Open folder: `~/crpbot`
5. Claude Code will work on the remote environment

**Option B: Terminus + Claude Code CLI**

If Claude Code has a CLI that works over SSH:
```bash
# Connect with Terminus
ssh crpbot-cloud

# Use Claude Code CLI (if available)
cd ~/crpbot
claude-code .
```

**Option C: Web-Based IDE**

Set up a web-based IDE on the cloud server:
- code-server (VS Code in browser)
- Jupyter Lab
- Cloud9

### Switching Between Environments

**To work locally:**
```bash
cd /home/numan/crpbot
source .venv/bin/activate
# Use Claude Code here
```

**To work on cloud:**
```bash
ssh crpbot-cloud
cd ~/crpbot
source .venv/bin/activate
# Use Claude Code here
```

## Git Workflow Best Practices

### Branch Strategy

```bash
# Main branch: production-ready code
# Dev branch: development work
# Feature branches: specific features

# On local machine:
git checkout -b dev
git push -u origin dev

# Work on features:
git checkout -b feature/new-model
# ... make changes ...
git add .
git commit -m "Add new model architecture"
git push origin feature/new-model

# Merge to dev:
git checkout dev
git merge feature/new-model
git push origin dev

# When ready for production:
git checkout main
git merge dev
git push origin main

# Pull on cloud:
ssh crpbot-cloud "cd ~/crpbot && git pull origin main"
```

### Avoiding Conflicts

**Rule 1: Never edit same files on both machines simultaneously**

**Rule 2: Always pull before starting work**
```bash
git pull origin main
```

**Rule 3: Use branches for major work**
```bash
# On local: work on feature branches
# On cloud: use main branch for stable runs
```

**Rule 4: Commit frequently**
```bash
# Commit small, logical changes
git add .
git commit -m "Descriptive message"
git push origin main
```

### Handling Conflicts

If you get merge conflicts:
```bash
# Pull latest changes
git pull origin main

# If conflicts occur:
git status  # See conflicted files

# Edit files to resolve conflicts
nano conflicted_file.py

# Mark as resolved
git add conflicted_file.py
git commit -m "Resolve merge conflict"
git push origin main
```

## Environment-Specific Configuration

### Using Environment Variables

Create separate `.env` files for each environment:

**Local** (`.env`):
```bash
RUNTIME_MODE=dryrun
DB_URL=sqlite:///tradingai.db
LOG_LEVEL=DEBUG
ENVIRONMENT=local
```

**Cloud** (`.env`):
```bash
RUNTIME_MODE=live
DB_URL=postgresql+psycopg://crpbot_admin:PASSWORD@rds-endpoint:5432/crpbot
LOG_LEVEL=INFO
ENVIRONMENT=production
```

Add `.env` to `.gitignore` (already done) and manage separately.

### Environment Detection

Add to your code:
```python
# libs/config/settings.py
import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

def is_production():
    return ENVIRONMENT == "production"

def is_local():
    return ENVIRONMENT == "local"
```

## Backup Strategy for Both Environments

### Daily Automated Backups

**Local:**
```bash
# Add to crontab: crontab -e
0 2 * * * cd /home/numan/crpbot && ./scripts/backup_local.sh
```

**Cloud:**
```bash
# Add to crontab on cloud server
0 3 * * * cd ~/crpbot && ./scripts/backup_cloud.sh
```

### Manual Backup Before Major Changes

```bash
# Local
cd /home/numan/crpbot
tar -czf ~/backups/crpbot-local-$(date +%Y%m%d).tar.gz \
  .env .db_password models/

# Cloud
ssh crpbot-cloud
cd ~/crpbot
tar -czf ~/backups/crpbot-cloud-$(date +%Y%m%d).tar.gz \
  .env .db_password models/
```

## Monitoring Both Environments

### Check Status Script

Create a script to check both environments:

```bash
#!/bin/bash
# scripts/check_both_environments.sh

echo "=== LOCAL ENVIRONMENT ==="
cd /home/numan/crpbot
git log --oneline -1
git status -s
source .venv/bin/activate
python --version

echo ""
echo "=== CLOUD ENVIRONMENT ==="
ssh crpbot-cloud "cd ~/crpbot && git log --oneline -1 && git status -s && source .venv/bin/activate && python --version"
```

### Dashboard View

```bash
# Quick status check
./scripts/check_both_environments.sh

# Expected output shows:
# - Current git commit on both
# - Any uncommitted changes
# - Python version
# - Running processes
```

## Troubleshooting

### Issue: Git Commits Out of Sync

```bash
# Check status on both
git log --oneline -5  # Local
ssh crpbot-cloud "cd ~/crpbot && git log --oneline -5"  # Cloud

# Force sync (careful!)
git fetch origin
git reset --hard origin/main  # On machine that's behind
```

### Issue: SSH Key Not Working

```bash
# Test SSH connection
ssh -v crpbot-cloud

# Check key permissions (must be 600)
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# Verify key is added to cloud server
ssh crpbot-cloud "cat ~/.ssh/authorized_keys"
```

### Issue: Dependencies Out of Sync

```bash
# Sync dependencies
uv lock  # Local
git add uv.lock
git commit -m "Update dependencies"
git push origin main

# Update cloud
ssh crpbot-cloud "cd ~/crpbot && git pull && source .venv/bin/activate && uv pip sync"
```

## Quick Reference

### Essential Commands

| Task | Command |
|------|---------|
| Connect to cloud | `ssh crpbot-cloud` |
| Push code to cloud | `git push origin main && ssh crpbot-cloud "cd ~/crpbot && git pull"` |
| Sync models to cloud | `rsync -avz models/ crpbot-cloud:~/crpbot/models/` |
| Pull models from cloud | `rsync -avz crpbot-cloud:~/crpbot/models/ models/` |
| Check both environments | `./scripts/check_both_environments.sh` |
| Run tests locally | `make test` |
| Run tests on cloud | `ssh crpbot-cloud "cd ~/crpbot && make test"` |

### Decision Tree: Where to Work?

**Work Locally when:**
- Quick code changes
- Rapid testing
- Small experiments
- Documentation updates

**Work on Cloud when:**
- Long training runs (hours)
- Production runtime
- GPU training (if cloud has GPU)
- High memory tasks

**Use Both when:**
- Develop locally, test locally
- Push to git
- Deploy to cloud for production
- Monitor cloud, iterate locally

## Next Steps

1. ✅ Set up SSH key authentication
2. ✅ Configure SSH config for easy connection
3. ✅ Deploy to cloud using updated script
4. ✅ Test both environments
5. ✅ Set up sync scripts
6. ✅ Configure Claude Code for both environments
7. ✅ Establish git workflow
8. ✅ Set up monitoring

See the updated deployment script for SSH key support!
