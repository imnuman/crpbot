# Daily Workflow - Multi-Agent Development

Team: **Claude Local (QC)** | **Claude Cloud (Builder)** | **Amazon Q Local (QC)** | **Amazon Q Cloud (AWS)**

---

## 🏗️ Claude Cloud (Builder)

**Role**: Development, Training, Testing, Building
**Environment**: `/root/crpbot` on 178.156.136.185
**Mode**: `RUNTIME_MODE=dryrun` (until Phase 7)

### Morning Start
```bash
# Connect and sync
ssh root@178.156.136.185
cd ~/crpbot && source .venv/bin/activate

# Pull latest code from Local Claude QC
git pull origin main

# Download latest models if needed
aws s3 sync s3://crpbot-market-data-dev/models/ models/
```

### During Development
```bash
# Tasks: Feature engineering, model training, testing, debugging
# Examples:
./batch_engineer_features.sh
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15
uv run python scripts/evaluate_model.py --model models/lstm_*.pt
make test
```

### Evening Wrap-Up
```bash
# Upload results to S3
aws s3 sync models/ s3://crpbot-market-data-dev/models/
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/
aws s3 sync data/features/ s3://crpbot-market-data-dev/data/features/

# Commit code changes
git add .
git commit -m "feat: train BTC LSTM model, 68.5% accuracy"
git push origin main

# Report to Local Claude for QC
# Provide: metrics, logs, file paths, screenshots
```

---

## 🔍 Claude Local (QC - Quality Control)

**Role**: Review, Validation, Approval, Documentation
**Environment**: `/home/numan/crpbot`
**Mode**: `RUNTIME_MODE=dryrun` (always)

### Morning Review
```bash
# Sync with Cloud Claude
cd /home/numan/crpbot && source .venv/bin/activate
git pull origin main

# Download artifacts for review
aws s3 sync s3://crpbot-market-data-dev/models/ models/
aws s3 sync s3://crpbot-market-data-dev/data/features/ data/features/
```

### QC Review Process
```bash
# 1. Review code changes from Cloud Claude
git log --oneline -5
git diff HEAD~1

# 2. Validate model metrics
uv run python scripts/evaluate_model.py --model models/lstm_BTC_USD_1m_*.pt

# 3. Check data quality
uv run python scripts/validate_data_quality.py --symbol BTC-USD

# 4. Run tests
make test

# 5. Review against promotion gates
# - Accuracy ≥68%?
# - Calibration error ≤5%?
# - No data quality issues?
# - Tests passing?
```

### Approval/Rejection
```bash
# If APPROVED:
# 1. Promote models (if applicable)
cp models/lstm_BTC_USD_1m_*.pt models/promoted/

# 2. Update documentation
# Edit PHASE1_COMPLETE_NEXT_STEPS.md, CLAUDE.md, etc.

# 3. Commit approvals
git add .
git commit -m "docs: approve BTC LSTM model for Phase 6.5"
git push origin main

# If REJECTED:
# 1. Document issues found
# 2. Create issue report
# 3. Communicate back to Cloud Claude for fixes
```

---

## ☁️ Amazon Q Cloud (AWS Infrastructure Builder)

**Role**: AWS Resource Management, Infrastructure as Code, Deployment
**Environment**: `/root/crpbot` on 178.156.136.185
**Access**: Full AWS admin (us-east-1, account 980104576869)

### AWS Infrastructure Tasks
```bash
# Connect to cloud
ssh root@178.156.136.185
cd ~/crpbot

# Infrastructure Operations:
# 1. RDS Management
aws rds describe-db-instances --region us-east-1
# Modify, backup, restore RDS instances

# 2. S3 Management
aws s3api list-buckets
aws s3api get-bucket-versioning --bucket crpbot-market-data-dev
# Configure lifecycle policies, versioning, replication

# 3. Lambda Deployment
# Deploy serverless functions for alerts, monitoring, automation

# 4. CloudWatch Monitoring
aws cloudwatch put-metric-alarm ...
# Set up alarms, dashboards, log groups

# 5. Secrets Manager
aws secretsmanager list-secrets --region us-east-1
# Rotate secrets, update credentials

# 6. IAM Management
aws iam list-roles
# Create roles, policies, service accounts

# 7. Cost Optimization
aws ce get-cost-and-usage ...
# Monitor costs, set budgets, optimize resources
```

### Infrastructure as Code
```bash
# Using Terraform or CDK
cd infra/terraform
terraform plan
terraform apply

# Or using AWS CDK
cd infra/cdk
cdk synth
cdk deploy
```

### Monitoring & Alerts
```bash
# CloudWatch Logs
aws logs create-log-group --log-group-name /crpbot/runtime
aws logs put-retention-policy --log-group-name /crpbot/runtime --retention-in-days 30

# SNS Topics for alerts
aws sns create-topic --name crpbot-alerts
aws sns subscribe --topic-arn ... --protocol email --notification-endpoint user@example.com
```

---

## 🛡️ Amazon Q Local (QC - Security & Compliance)

**Role**: Security Audits, Code Review, Cost Analysis, Compliance
**Environment**: `/home/numan/crpbot`
**Access**: AWS read-only

### Security Audits
```bash
cd /home/numan/crpbot && source .venv/bin/activate
git pull origin main

# 1. Code Security Review
# - Check for hardcoded credentials
# - SQL injection vulnerabilities
# - OWASP Top 10 compliance

# 2. AWS Security Posture
aws iam get-account-authorization-details --region us-east-1
# Review IAM policies, S3 bucket policies, security groups

# 3. Secrets Management
# Verify no secrets in code, all in AWS Secrets Manager

# 4. Dependency Vulnerabilities
pip-audit
bandit -r apps/ libs/

# 5. Infrastructure Review
# - RDS encryption at rest?
# - S3 bucket encryption?
# - VPC security groups configured correctly?
# - CloudWatch logging enabled?
```

### Cost Analysis
```bash
# Review AWS costs
aws ce get-cost-and-usage \
  --time-period Start=2025-11-01,End=2025-11-12 \
  --granularity MONTHLY \
  --metrics "UnblendedCost" \
  --group-by Type=SERVICE

# Identify cost optimization opportunities
# - Unused resources
# - Over-provisioned instances
# - S3 storage classes
# - Data transfer costs
```

### Compliance Checks
```bash
# FTMO Rules Compliance
# - Daily loss limit enforced?
# - Rate limiting working?
# - Kill switch tested?
# - Position sizing correct?

# Data Privacy
# - No PII in logs
# - Credentials encrypted
# - API keys secured
```

---

## 🔄 Daily Sync Workflow

### Morning (8 AM)
1. **Claude Local**: Pull code, download models/data, review overnight work
2. **Amazon Q Local**: Security scan, cost review, compliance check

### Midday (12 PM)
1. **Claude Cloud**: Push morning's work, upload to S3
2. **Claude Local**: QC review, approve/reject changes
3. **Amazon Q Cloud**: Deploy any infrastructure changes

### Evening (6 PM)
1. **Claude Cloud**: Final push, upload results
2. **Claude Local**: Final QC review, update docs
3. **Amazon Q Local**: Final security/cost check

### Before Bed (10 PM - 2 AM)
1. **Claude Cloud**: Long-running tasks (transformer training, data fetch)
2. **Monitor progress**, report status

---

## 📊 Communication Flow

```
Claude Cloud (Build) → Upload to S3 + Push Git → Claude Local (QC Review)
                                                         ↓
                                                    Approve/Reject
                                                         ↓
                                                   Update Docs
                                                         ↓
                                                    Push to Git
                                                         ↓
                                                   Cloud Pulls

Amazon Q Cloud (AWS) → Deploy Infra → Amazon Q Local (Audit)
                                              ↓
                                         Approve/Flag
                                              ↓
                                      Review Security
                                              ↓
                                        Cost Check
```

---

## 🎯 Role Responsibilities Summary

| Agent | Primary Role | Key Activities | Environment | Git Access | AWS Access | S3 Access |
|-------|--------------|----------------|-------------|------------|------------|-----------|
| **Claude Cloud** | Builder | Train, test, develop | Cloud (`/root/crpbot`) | Read/Write | Read/Write | Read/Write |
| **Claude Local** | QC | Review, validate, approve | Local (`/home/numan/crpbot`) | Read/Write | Read-only | Read-only |
| **Amazon Q Cloud** | AWS Builder | Infrastructure, deployment | Cloud (`/root/crpbot`) | Read | Admin | Read/Write |
| **Amazon Q Local** | Security QC | Audit, compliance, cost | Local (`/home/numan/crpbot`) | Read | Read-only | Read-only |

---

## 🚨 Conflict Prevention Rules

1. **Never edit the same file simultaneously** - Use Git to coordinate
2. **Always pull before push** - `git pull origin main` first
3. **Use S3 for artifacts** - Never commit models/data to Git
4. **Document changes** - Clear commit messages
5. **Report before wrapping up** - Let other agents know what you did
6. **Test before pushing** - `make test` must pass
7. **Follow promotion gates** - No shortcuts in QC

---

**This workflow ensures conflict-free, efficient multi-agent development!** 🚀
