# Environment Conflict Audit - Local vs Cloud

**Date**: 2025-11-12
**Audit Type**: Comprehensive conflict detection for dual-environment setup
**Environments**:
- Local: `/home/numan/crpbot` (user: numan)
- Cloud: `/root/crpbot` (user: root, IP: 178.156.136.185)

---

## ✅ No Conflicts Found (Safe)

### 1. **Path References** ✅
- **Status**: FIXED (commit b4f5ceb)
- **Previous Issue**: `apps/runtime/aws_runtime.py` had hardcoded `/home/numan/crpbot` paths
- **Fix**: Now uses dynamic `Path(__file__).resolve().parent.parent.parent`
- **Verification**: All tests passing (16/16)

### 2. **Git Configuration** ✅
- **Status**: Clean
- **Remote**: `https://github.com/imnuman/crpbot.git` (same on both)
- **No user.name/email** configured locally (uses global config)
- **Branches**: All remote-tracking branches point to origin

### 3. **Python Dependencies** ✅
- **Status**: Locked with `uv.lock`
- **Dependencies**: All version-pinned in `pyproject.toml`
- **Package Manager**: Using `uv` (deterministic installs)
- **Verification**: Same packages will install on both environments

### 4. **AWS Configuration** ✅
- **Status**: Minimal, environment-agnostic
- **Config**: `~/.aws/config` only has `output = json`
- **Credentials**: Stored in `~/.aws/credentials` (synced via script)
- **Region**: Not hardcoded, will use AWS SDK defaults

### 5. **IDE Files** ✅
- **Status**: Properly ignored
- **Gitignore**: Ignores `.vscode/`, `.idea/`, `.cursor/`
- **No Cursor files**: Project no longer uses Cursor
- **Amazon Q**: No config files yet (will be environment-specific)

### 6. **Scripts** ✅
- **Status**: All use relative paths or dynamic detection
- **Bash scripts**: Use `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`
- **Python scripts**: Use `Path(__file__)` or relative imports
- **No hardcoded paths** found in operational scripts

---

## ⚠️ Potential Conflicts (Action Required)

### 1. **SQLite Database Path** ⚠️
**Issue**: `.env` contains `DB_URL=sqlite:///tradingai.db` (relative path)
- **Local**: Creates `/home/numan/crpbot/tradingai.db`
- **Cloud**: Creates `/root/crpbot/tradingai.db`
- **Impact**: Two separate databases (not synced)

**Solution Options**:
- **Option A** (Recommended): Use PostgreSQL RDS for both environments
  ```bash
  # Local .env
  DB_URL=postgresql+psycopg://crpbot_admin:PASSWORD@crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com:5432/crpbot

  # Cloud .env (same)
  DB_URL=postgresql+psycopg://crpbot_admin:PASSWORD@crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com:5432/crpbot
  ```
- **Option B**: Keep SQLite for local dev, PostgreSQL for cloud production
  ```bash
  # Local .env
  DB_URL=sqlite:///tradingai.db

  # Cloud .env
  DB_URL=postgresql+psycopg://crpbot_admin:PASSWORD@crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com:5432/crpbot
  ```

**Recommendation**: **Option A** - Use shared PostgreSQL RDS on both for consistency

---

### 2. **Model Files Location** ⚠️
**Issue**: Models stored locally, not synced automatically
- **Local**: `/home/numan/crpbot/models/` (gitignored)
- **Cloud**: `/root/crpbot/models/` (gitignored)
- **Impact**: Training on one machine doesn't update the other

**Solution**: Use S3 as single source of truth
```bash
# After training (on either machine)
aws s3 sync models/ s3://crpbot-market-data-dev/models/

# Before runtime (on either machine)
aws s3 sync s3://crpbot-market-data-dev/models/ models/
```

**Workflow**:
1. Cloud Claude trains models → uploads to S3
2. Local Claude downloads from S3 for QC review
3. Approved models stay in S3 `models/promoted/`

---

### 3. **Data Files Location** ⚠️
**Issue**: Data stored locally, not synced automatically
- **Local**: `/home/numan/crpbot/data/` (gitignored)
- **Cloud**: `/root/crpbot/data/` (gitignored)
- **Impact**: Data fetched on one machine isn't available on other

**Solution**: Use S3 as single source of truth
```bash
# After data fetch (typically on cloud)
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/

# For local QC (download specific files)
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/ --exclude "*" --include "BTC-USD*"
```

**Workflow**:
1. Cloud Claude fetches/engineers data → uploads to S3
2. Local Claude downloads subsets for QC validation
3. S3 is canonical storage, local copies are cache

---

### 4. **Log Files** ⚠️
**Issue**: Logs stored locally, not shared
- **Local**: Logs in `/home/numan/crpbot/logs/` (gitignored)
- **Cloud**: Logs in `/root/crpbot/logs/` (gitignored)
- **Impact**: Can't review cloud logs from local machine

**Solution**: Use CloudWatch Logs (future enhancement)
```bash
# Configure logging in apps/runtime/main.py
# Send logs to CloudWatch Logs for centralized access
```

**Current Workaround**:
```bash
# View cloud logs while SSH'd
ssh root@178.156.136.185 "tail -f ~/crpbot/logs/runtime.log"
```

---

### 5. **Amazon Q Configuration** ⚠️
**Issue**: Amazon Q will need separate configs per environment
- **Local**: Amazon Q for QC (review, validation)
- **Cloud**: Amazon Q for AWS infrastructure building

**Solution**: Environment-specific Amazon Q configs
```bash
# Local .amazon-q/config.json
{
  "role": "qc",
  "tasks": ["review", "validate", "document"],
  "aws_access": false
}

# Cloud .amazon-q/config.json
{
  "role": "builder",
  "tasks": ["infrastructure", "deployment", "aws"],
  "aws_access": true
}
```

**Action**: Create `.amazon-q/` directory and add to `.gitignore`

---

### 6. **RUNTIME_MODE Configuration** ⚠️
**Issue**: Should differ between environments
- **Local**: Should always use `RUNTIME_MODE=dryrun` (never trade)
- **Cloud**: Can use `RUNTIME_MODE=live` (production)

**Solution**: Different `.env` values
```bash
# Local .env
RUNTIME_MODE=dryrun

# Cloud .env
RUNTIME_MODE=dryrun  # Change to 'live' only after Phase 6.5 validation
```

**Safeguard**: Add check in `apps/runtime/main.py`
```python
import os
if os.geteuid() != 0 and runtime_mode == "live":
    raise ValueError("Live mode only allowed on cloud (root user)")
```

---

## 🔄 Sync Strategy

### What Should Be Synced via Git
✅ Code (`apps/`, `libs/`, `scripts/`)
✅ Tests (`tests/`)
✅ Documentation (`*.md`)
✅ Configuration templates (`pyproject.toml`, `Makefile`)
✅ Infrastructure code (`infra/`)

### What Should NOT Be Synced via Git
❌ `.env` (environment-specific)
❌ `models/` (use S3)
❌ `data/` (use S3)
❌ `logs/` (environment-specific)
❌ `.venv/` (rebuild on each machine)
❌ `tradingai.db` (environment-specific)
❌ `.db_password` (synced separately via `scripts/sync_credentials.sh`)

### Sync Workflows

#### Code Sync (via Git)
```bash
# Local → Cloud
git add . && git commit -m "..." && git push origin main
# Then on cloud: git pull origin main

# Cloud → Local
# On cloud: git add . && git commit -m "..." && git push origin main
git pull origin main
```

#### Models Sync (via S3)
```bash
# Upload after training
aws s3 sync models/ s3://crpbot-market-data-dev/models/

# Download for use
aws s3 sync s3://crpbot-market-data-dev/models/ models/
```

#### Data Sync (via S3)
```bash
# Upload after fetching
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/

# Download for QC
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/ --exclude "*" --include "BTC-USD*"
```

#### Credentials Sync (via script)
```bash
# Local → Cloud (one-time or after updates)
./scripts/sync_credentials.sh -i ~/.ssh/id_ed25519 root@178.156.136.185
```

---

## 🎯 Role-Based Development Setup

### Claude Local (QC)
- **Role**: Quality Control, Review, Documentation
- **Tools**: Claude Code CLI
- **Environment**: `/home/numan/crpbot`
- **Mode**: Always `RUNTIME_MODE=dryrun`
- **Workflow**:
  1. Review outputs from Cloud Claude
  2. Validate metrics against gates
  3. Approve/reject model promotions
  4. Update documentation
  5. Commit approved changes to Git

### Claude Cloud (Builder)
- **Role**: Development, Training, Testing
- **Tools**: Claude Code CLI
- **Environment**: `/root/crpbot`
- **Mode**: `RUNTIME_MODE=dryrun` (live after validation)
- **Workflow**:
  1. Fetch data from Coinbase
  2. Engineer features
  3. Train models
  4. Run evaluations
  5. Upload results to S3
  6. Report metrics to Local Claude for QC

### Amazon Q Local (QC)
- **Role**: Code Review, Security Audits
- **Tools**: Amazon Q CLI/IDE
- **Environment**: `/home/numan/crpbot`
- **Config**: Review-only mode
- **Workflow**:
  1. Review code changes
  2. Security vulnerability scanning
  3. Best practices enforcement
  4. Documentation generation

### Amazon Q Cloud (AWS Builder)
- **Role**: AWS Infrastructure Management
- **Tools**: Amazon Q CLI/IDE
- **Environment**: `/root/crpbot`
- **Config**: AWS admin mode
- **Workflow**:
  1. Deploy AWS resources (RDS, S3, Lambda)
  2. Manage IAM roles and policies
  3. Configure CloudWatch monitoring
  4. Set up CI/CD pipelines
  5. Infrastructure as Code (Terraform/CDK)

---

## 📋 Pre-Flight Checklist

Before starting work on either environment:

### Local (QC Environment)
- [ ] `git pull origin main` - Get latest code
- [ ] `source .venv/bin/activate` - Activate Python env
- [ ] `RUNTIME_MODE=dryrun` in `.env` - Safety check
- [ ] AWS credentials present - `aws sts get-caller-identity`
- [ ] Can access S3 - `aws s3 ls s3://crpbot-market-data-dev/`

### Cloud (Builder Environment)
- [ ] `git pull origin main` - Get latest code
- [ ] `source .venv/bin/activate` - Activate Python env
- [ ] Credentials synced - `.env`, `.db_password`, AWS
- [ ] S3 access verified - `aws s3 ls s3://crpbot-market-data-dev/`
- [ ] RDS access verified - `psql -h crpbot-rds-postgres-db... -U crpbot_admin -d crpbot -c "SELECT 1"`

---

## 🚨 Critical Safety Rules

### Never Do on Local
❌ Set `RUNTIME_MODE=live`
❌ Execute real trades
❌ Modify production database directly
❌ Delete S3 production data

### Never Do on Cloud
❌ Commit AWS credentials to Git
❌ Run untested code in live mode
❌ Delete models without backup
❌ Skip QC approval before promotion

### Always Do
✅ Test on local before deploying to cloud
✅ Use S3 for model/data sharing
✅ Commit code changes to Git
✅ Run tests before pushing
✅ Document major changes
✅ Get QC approval before Phase 7 (live trading)

---

## 🔧 Fixing Existing Conflicts

### Action 1: Switch to PostgreSQL (Both Environments)
```bash
# Local .env
DB_URL=postgresql+psycopg://crpbot_admin:$(cat .db_password)@crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com:5432/crpbot

# Cloud .env (same)
DB_URL=postgresql+psycopg://crpbot_admin:$(cat .db_password)@crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com:5432/crpbot
```

### Action 2: Add Amazon Q to .gitignore
```bash
echo ".amazon-q/" >> .gitignore
echo ".amazonq/" >> .gitignore
```

### Action 3: Document S3 Sync in Workflow
Update `DUAL_ENVIRONMENT_SETUP.md` with S3 sync procedures

### Action 4: Add Runtime Mode Safeguard
```python
# In apps/runtime/main.py
import os
import sys

def check_runtime_mode(mode: str):
    if mode == "live":
        if os.geteuid() != 0:
            print("❌ ERROR: Live mode only allowed on cloud server (root user)")
            sys.exit(1)
        if input("⚠️  Live mode enabled. Type 'CONFIRM' to proceed: ") != "CONFIRM":
            print("❌ Live mode cancelled")
            sys.exit(1)
```

---

## 📊 Summary

| Conflict Area | Status | Action Required |
|---------------|--------|-----------------|
| **Path References** | ✅ Fixed | None - already resolved |
| **Git Configuration** | ✅ Clean | None |
| **Python Dependencies** | ✅ Clean | None |
| **AWS Configuration** | ✅ Clean | None |
| **IDE Files** | ✅ Clean | None |
| **Scripts** | ✅ Clean | None |
| **Database Path** | ⚠️ Needs Fix | Switch both to PostgreSQL RDS |
| **Model Files** | ⚠️ Needs Strategy | Use S3 as single source of truth |
| **Data Files** | ⚠️ Needs Strategy | Use S3 as single source of truth |
| **Log Files** | ⚠️ Minor | Use CloudWatch (future) |
| **Amazon Q Config** | ⚠️ Pending | Create configs, add to .gitignore |
| **Runtime Mode** | ⚠️ Needs Safeguard | Add user check in code |

**Overall Status**: 🟡 **Mostly Clean** - 6 areas need attention, but all are manageable

**Next Steps**:
1. Switch to PostgreSQL on both environments
2. Establish S3 sync workflows
3. Add Amazon Q configs to .gitignore
4. Add runtime mode safeguards
5. Document sync procedures

**Estimated Time**: 15-20 minutes to resolve all conflicts
