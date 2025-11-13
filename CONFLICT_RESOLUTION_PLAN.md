# Conflict Resolution Plan - 15 Minute Fix

**Goal**: Resolve all environment conflicts before Cloud Claude starts building

---

## 🎯 Quick Actions (15 minutes)

### Action 1: Update .gitignore (1 min)
```bash
# Add Amazon Q configs
echo "" >> .gitignore
echo "# Amazon Q configurations (environment-specific)" >> .gitignore
echo ".amazon-q/" >> .gitignore
echo ".amazonq/" >> .gitignore
echo "amazonq/" >> .gitignore
```

### Action 2: Add Runtime Mode Safeguard (3 min)
Add to `apps/runtime/main.py`:
```python
def check_runtime_mode_safety(mode: str):
    """Prevent accidental live trading on local machine"""
    import os
    import sys

    if mode.lower() == "live":
        # Check if running as root (cloud server indicator)
        if os.name == 'posix' and os.geteuid() != 0:
            print("❌ ERROR: Live mode only allowed on cloud server (root user)")
            print("   Local machines should always use RUNTIME_MODE=dryrun")
            sys.exit(1)

        # Double confirmation for live mode
        print("⚠️  WARNING: Live mode enabled - real trades will be executed!")
        confirm = input("Type 'CONFIRM LIVE MODE' to proceed: ")
        if confirm != "CONFIRM LIVE MODE":
            print("❌ Live mode cancelled")
            sys.exit(1)

        print("✅ Live mode confirmed - starting runtime...")
```

### Action 3: Create Amazon Q Config Template (2 min)
```bash
# Create directory structure
mkdir -p .amazon-q-templates

# Local QC config template
cat > .amazon-q-templates/local-qc-config.json <<'EOF'
{
  "role": "quality-control",
  "environment": "local",
  "tasks": [
    "code-review",
    "security-audit",
    "documentation-validation",
    "test-verification"
  ],
  "permissions": {
    "aws_access": false,
    "read_only": true,
    "approve_deployments": false
  },
  "git": {
    "auto_commit": false,
    "require_approval": true
  }
}
EOF

# Cloud Builder config template
cat > .amazon-q-templates/cloud-builder-config.json <<'EOF'
{
  "role": "aws-infrastructure-builder",
  "environment": "cloud",
  "tasks": [
    "aws-resource-management",
    "infrastructure-as-code",
    "deployment-automation",
    "cloudwatch-monitoring"
  ],
  "permissions": {
    "aws_access": true,
    "aws_admin": true,
    "deploy_resources": true
  },
  "aws": {
    "allowed_services": [
      "rds",
      "s3",
      "lambda",
      "cloudwatch",
      "iam",
      "secrets-manager",
      "elasticache"
    ],
    "region": "us-east-1"
  }
}
EOF
```

### Action 4: Document S3 Sync Workflow (3 min)
Create quick reference card:
```bash
cat > S3_SYNC_QUICKREF.md <<'EOF'
# S3 Sync Quick Reference

## After Training Models (Cloud)
```bash
aws s3 sync models/ s3://crpbot-market-data-dev/models/ --exclude "*" --include "*.pt"
```

## Download Models for QC (Local)
```bash
aws s3 sync s3://crpbot-market-data-dev/models/ models/
```

## After Fetching Data (Cloud)
```bash
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/
```

## Download Data for QC (Local)
```bash
# Download specific symbol
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/ --exclude "*" --include "BTC-USD*"

# Download all
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/
```

## Check What's in S3
```bash
aws s3 ls s3://crpbot-market-data-dev/models/ --recursive --human-readable
aws s3 ls s3://crpbot-market-data-dev/data/raw/ --recursive --human-readable
```
EOF
```

### Action 5: Switch to PostgreSQL (5 min)
```bash
# Get RDS password
DB_PASS=$(cat .db_password)

# Test connection first
PGPASSWORD="$DB_PASS" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "SELECT version();"

# If connection works, update .env
# Backup current .env
cp .env .env.backup

# Update DB_URL
sed -i 's|DB_URL=sqlite:///tradingai.db|DB_URL=postgresql+psycopg://crpbot_admin:'"$DB_PASS"'@crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com:5432/crpbot|g' .env

# Verify change
grep "DB_URL" .env
```

### Action 6: Create Conflict-Free Workflow Doc (1 min)
```bash
cat > DAILY_WORKFLOW.md <<'EOF'
# Daily Workflow - Conflict-Free Operations

## Cloud Claude (Builder)
1. `ssh root@178.156.136.185`
2. `cd ~/crpbot && source .venv/bin/activate`
3. `git pull origin main` - Get latest code
4. Work on tasks (train, test, develop)
5. `aws s3 sync models/ s3://crpbot-market-data-dev/models/` - Upload results
6. `git add . && git commit -m "..." && git push` - Share code
7. Report metrics/outputs to Local Claude for QC

## Local Claude (QC)
1. `cd /home/numan/crpbot && source .venv/bin/activate`
2. `git pull origin main` - Get latest code from Cloud
3. `aws s3 sync s3://crpbot-market-data-dev/models/ models/` - Download models
4. Review outputs, validate metrics
5. Approve/reject changes
6. Update documentation if needed
7. `git push` - Commit approved docs

## Amazon Q Cloud (AWS Builder)
1. SSH to cloud server
2. Work on AWS infrastructure (RDS, S3, Lambda, etc.)
3. Use Infrastructure as Code (Terraform/CDK)
4. Deploy resources
5. Configure monitoring
6. Report infrastructure status

## Amazon Q Local (QC)
1. Review infrastructure changes
2. Security audits
3. Cost optimization reviews
4. Compliance checks
5. Approve AWS deployments
EOF
```

---

## 🚀 Execute All Actions

Run this single command to fix everything:

```bash
cd /home/numan/crpbot

# Action 1: Update .gitignore
cat >> .gitignore <<'EOF'

# Amazon Q configurations (environment-specific)
.amazon-q/
.amazonq/
amazonq/
EOF

# Action 2: Create Amazon Q templates
mkdir -p .amazon-q-templates
cat > .amazon-q-templates/local-qc-config.json <<'EOF'
{
  "role": "quality-control",
  "environment": "local",
  "tasks": ["code-review", "security-audit", "documentation-validation", "test-verification"],
  "permissions": {
    "aws_access": false,
    "read_only": true,
    "approve_deployments": false
  },
  "git": {
    "auto_commit": false,
    "require_approval": true
  }
}
EOF

cat > .amazon-q-templates/cloud-builder-config.json <<'EOF'
{
  "role": "aws-infrastructure-builder",
  "environment": "cloud",
  "tasks": ["aws-resource-management", "infrastructure-as-code", "deployment-automation", "cloudwatch-monitoring"],
  "permissions": {
    "aws_access": true,
    "aws_admin": true,
    "deploy_resources": true
  },
  "aws": {
    "allowed_services": ["rds", "s3", "lambda", "cloudwatch", "iam", "secrets-manager", "elasticache"],
    "region": "us-east-1"
  }
}
EOF

# Action 3: Create S3 sync reference
cat > S3_SYNC_QUICKREF.md <<'EOF'
# S3 Sync Quick Reference

## Models
# Upload: aws s3 sync models/ s3://crpbot-market-data-dev/models/
# Download: aws s3 sync s3://crpbot-market-data-dev/models/ models/

## Data
# Upload: aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/
# Download: aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/
EOF

# Action 4: Create workflow doc
cat > DAILY_WORKFLOW.md <<'EOF'
# Daily Workflow - Conflict-Free Operations

## Cloud Claude: Build, train, test → Push code + Upload to S3
## Local Claude: Pull code, Download from S3 → QC review
## Amazon Q Cloud: Deploy AWS infrastructure
## Amazon Q Local: Review infrastructure, security audits
EOF

# Action 5: Test PostgreSQL connection
echo "Testing PostgreSQL connection..."
PGPASSWORD="$(cat .db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "SELECT version();" && echo "✅ PostgreSQL connection works!"

# Done
echo ""
echo "✅ All conflict resolutions applied!"
echo ""
echo "Next steps:"
echo "1. git add . && git commit -m 'fix: resolve environment conflicts'"
echo "2. git push origin main"
echo "3. On cloud: git pull origin main"
echo ""
```

---

## 📊 Verification Checklist

After running actions, verify:

### Local Environment
- [ ] `.gitignore` includes Amazon Q directories
- [ ] Amazon Q templates created in `.amazon-q-templates/`
- [ ] PostgreSQL connection works
- [ ] S3 access works: `aws s3 ls s3://crpbot-market-data-dev/`
- [ ] Git status clean: `git status`

### Cloud Environment
- [ ] Pull latest changes: `git pull origin main`
- [ ] Amazon Q templates present
- [ ] PostgreSQL connection works
- [ ] S3 access works
- [ ] Runtime safeguard present in code

---

## ⏱️ Estimated Time

- Action 1 (gitignore): 1 min ✅
- Action 2 (safeguard): 3 min ⏳ (optional - can add later)
- Action 3 (Amazon Q templates): 2 min ✅
- Action 4 (S3 reference): 1 min ✅
- Action 5 (PostgreSQL): 5 min ⏳ (test only, no .env change yet)
- Action 6 (workflow doc): 1 min ✅

**Total**: ~13 minutes (safeguard and PostgreSQL migration can wait)

---

## 🎯 Priority Actions NOW

### Must Do Immediately (5 min)
1. ✅ Update .gitignore for Amazon Q
2. ✅ Create Amazon Q config templates
3. ✅ Create S3 sync reference
4. ✅ Commit and push to Git

### Can Do Later (During Phase 6.5)
- Add runtime mode safeguard to code
- Migrate from SQLite to PostgreSQL (both environments)
- Set up CloudWatch logging

---

Ready to execute! Run the bash command block above to fix all conflicts in ~5 minutes.
