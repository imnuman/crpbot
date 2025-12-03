# AWS Cost Cleanup - November 22, 2025

## Problem
- Bill increased from $100 → $140 (40% increase)
- Multiple unused resources running 24/7

## Actions Taken

### ✅ STOPPED/DELETED Resources

#### 1. RDS Databases (STOPPED)
| Resource | Type | Status | Monthly Cost |
|----------|------|--------|--------------|
| `crpbot-rds-postgres-db` | db.t4g.small (100GB) | STOPPING | ~$35/month |
| `crpbot-dev` | db.t3.micro (20GB) | STOPPING | ~$14/month |
| **Compute savings** | | | **~$37/month** |
| **Storage still charged** | | | **~$12/month** |

**Note**: Stopped RDS still charges for storage. To save additional $12/month, consider:
```bash
# Delete if not needed (BACKUP FIRST if any important data)
aws rds delete-db-instance --db-instance-identifier crpbot-rds-postgres-db --skip-final-snapshot
aws rds delete-db-instance --db-instance-identifier crpbot-dev --skip-final-snapshot
```

#### 2. ElastiCache Redis (DELETED)
| Resource | Type | Status | Monthly Cost |
|----------|------|--------|--------------|
| `crpbot-redis-dev` | cache.t3.micro | DELETING | ~$12/month |
| `crp-re-wymqmkzvh0gm` | cache.t4g.micro | DELETING | ~$12/month |
| **Total savings** | | | **~$24/month** |

#### 3. Already Stopped (No Action Needed)
| Resource | Type | Status | Cost |
|----------|------|--------|------|
| EC2 `i-0f1af51b632fb9954` | t2.medium | STOPPED | $0 (stopped) |
| SageMaker `amazon-braket-test` | ml.t3.medium | STOPPED | $0 (stopped) |

---

## Cost Savings Summary

| Category | Before | After | Savings |
|----------|--------|-------|---------|
| RDS (compute) | ~$49/month | ~$12/month (storage only) | **~$37/month** |
| Redis | ~$24/month | $0 | **~$24/month** |
| **TOTAL** | **~$73/month** | **~$12/month** | **~$61/month** |

**Expected new monthly bill**: $140 - $61 = **~$79/month**

(After RDS storage charges settle, could drop to ~$67/month if stopped RDS is acceptable)

---

## What's Still Running (Production)

### Cloud Server (178.156.136.185)
- **V7 Runtime**: Running (PID 2620770)
- **Database**: SQLite (local, no AWS cost)
- **Dashboard**: Running (port 3000)
- **Status**: ✅ OPERATIONAL

### S3 Storage
```
amazon-braket-us-east-1-980104576869
bengali-ai-models-20251026
bengali-ai-monitoring-20251026
bengali-ai-training-data-20251026
cdk-hnb659fds-assets-980104576869-us-east-2
crpbot-backups-dev
crpbot-logs-dev
crpbot-market-data-dev
crpbot-ml-data-20251110
crpbot-sagemaker-training
sagemaker-ap-southeast-1-980104576869
sagemaker-us-east-1-980104576869
```
**Estimated cost**: ~$1-5/month (depends on data size)

---

## Recommendations

### 1. Delete Unused S3 Buckets (Optional)
If these buckets are not needed, delete to save $1-5/month:
```bash
# Check bucket sizes first
aws s3 ls s3://bengali-ai-models-20251026 --recursive --summarize
aws s3 ls s3://bengali-ai-monitoring-20251026 --recursive --summarize
aws s3 ls s3://bengali-ai-training-data-20251026 --recursive --summarize

# Delete if not needed (IRREVERSIBLE)
aws s3 rb s3://bengali-ai-models-20251026 --force
aws s3 rb s3://bengali-ai-monitoring-20251026 --force
aws s3 rb s3://bengali-ai-training-data-20251026 --force
```

### 2. Delete RDS Storage (Optional)
If RDS databases have no important data, delete completely to save $12/month:
```bash
# Check if any data exists
PGPASSWORD="<password>" psql -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com -U crpbot_admin -d crpbot -c "\dt"

# If empty or not needed, delete (WAIT FOR STATUS TO BE "stopped" FIRST)
aws rds delete-db-instance --db-instance-identifier crpbot-rds-postgres-db --skip-final-snapshot
aws rds delete-db-instance --db-instance-identifier crpbot-dev --skip-final-snapshot
```

### 3. Delete CloudFormation Stacks (Cleanup)
The infrastructure stacks are no longer needed since V7 uses SQLite:
```bash
# Delete stacks (this will clean up associated resources)
aws cloudformation delete-stack --stack-name crpbot-redis
aws cloudformation delete-stack --stack-name crpbot-rds-postgres
aws cloudformation delete-stack --stack-name crpbot-rds-dev
aws cloudformation delete-stack --stack-name crpbot-alarms-dev
aws cloudformation delete-stack --stack-name crpbot-dashboards-dev
aws cloudformation delete-stack --stack-name crpbot-telegram-bot-dev
aws cloudformation delete-stack --stack-name crpbot-risk-monitor-dev
aws cloudformation delete-stack --stack-name crpbot-lambda-signal-dev
aws cloudformation delete-stack --stack-name crpbot-secrets-dev
aws cloudformation delete-stack --stack-name crpbot-s3-dev
```

### 4. Set Up Billing Alerts
```bash
# Create budget alert for $100/month
aws budgets create-budget --account-id 980104576869 \
  --budget file://budget-config.json
```

---

## Verification

After 24-48 hours, verify cost reduction:

```bash
# Check resource status
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name]'
aws rds describe-db-instances --query 'DBInstances[*].[DBInstanceIdentifier,DBInstanceStatus]'
aws elasticache describe-cache-clusters --query 'CacheClusters[*].[CacheClusterId,CacheClusterStatus]'

# Monitor billing (if access granted)
aws ce get-cost-and-usage --time-period Start=2025-11-22,End=2025-11-23 --granularity DAILY --metrics BlendedCost
```

---

## Current Production Setup (V7)

**IMPORTANT**: V7 does NOT use any of the stopped/deleted AWS resources:

- ❌ Not using RDS (uses local SQLite on cloud server)
- ❌ Not using Redis (no caching needed)
- ❌ Not using Lambda (runs as Python process on cloud server)
- ✅ Uses cloud server directly (178.156.136.185)
- ✅ Uses Coinbase API, CoinGecko API, DeepSeek API
- ✅ Local SQLite database (`tradingai.db`)

**Impact on V7**: NONE - all stopped/deleted resources were unused.

---

**Date**: 2025-11-22
**Performed by**: QC Claude
**Monthly Savings**: ~$61/month (83% reduction in unnecessary costs)
**Next Bill Estimate**: ~$79/month (down from $140)
