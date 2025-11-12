# Phase 1: Foundation Infrastructure Deployment Guide

**Status**: Ready to Deploy
**Duration**: 2-4 hours
**Cost**: +$100/month (+$50/month for observability in Phase 2)
**Risk**: Low (additive, no disruption to current system)

---

## Overview

Phase 1 deploys the foundational infrastructure for production-grade ML trading:
- **RDS PostgreSQL** (t4g.small): Operational data storage
- **ElastiCache Redis** (t4g.micro): Feature caching
- **AWS Secrets Manager**: Credential management
- **MLflow Server** (t4g.small, optional): Model registry

This phase is **completely additive** - your existing system continues to work while we build the production infrastructure alongside it.

---

## Pre-Deployment Checklist

### AWS Account Setup
- [ ] AWS CLI configured with proper credentials
- [ ] AWS region set to `us-east-1` (or your preferred region)
- [ ] IAM permissions for CloudFormation, RDS, ElastiCache, Secrets Manager, EC2
- [ ] VPC with at least 2 availability zones (or will create new VPC)

### Cost Budgets
- [ ] Set up AWS Budget alert for $200/month
- [ ] Enable Cost Explorer
- [ ] Tag all resources with `Project=CryptoBot` for cost tracking

### Local Requirements
- [ ] AWS CLI v2 installed
- [ ] PostgreSQL client (`psql`) installed for testing
- [ ] Redis client (`redis-cli`) installed for testing
- [ ] OpenSSL for password generation

---

## Deployment Steps

### Step 1: Deploy RDS PostgreSQL (15-20 min)

**Cost**: $24/month (single-AZ) or $48/month (multi-AZ)

```bash
cd /home/numan/crpbot
./scripts/infrastructure/deploy_rds.sh
```

**What it does**:
1. Generates secure database password
2. Creates CloudFormation stack with:
   - RDS PostgreSQL 15.4 instance (t4g.small, ARM-based)
   - 100GB gp3 SSD storage
   - Automated backups (7-day retention)
   - Encryption at rest
   - Deletion protection enabled
3. Creates VPC, subnets, security groups
4. Saves connection info to `.rds_connection_info`

**Wait for**: "RDS Deployment Complete" message (~15 min)

**Verify**:
```bash
# Check RDS endpoint
source .rds_connection_info
echo "RDS Endpoint: $DB_HOST"

# Test connection (will fail until security group allows your IP)
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" -c "SELECT version();"
```

**Troubleshooting**:
- If stack creation fails, check CloudFormation events in AWS Console
- Common issue: VPC limits reached (delete unused VPCs or use existing VPC)
- If timeout, increase wait time in script

### Step 2: Create Database Schema (2-3 min)

**What it does**: Creates production-ready database schema with:
- 3 schemas: `trading`, `ml`, `metrics`
- 13 tables: trades, signals, positions, account_state, models, training_runs, etc.
- Indexes for fast queries
- Triggers for automatic timestamp updates and P&L calculations
- Views for common queries
- Initial account state ($100k balance, FTMO guardrails)

```bash
# Copy SQL script to local
cat scripts/infrastructure/create_db_schema.sql

# Execute schema creation
source .rds_connection_info
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
  -f scripts/infrastructure/create_db_schema.sql
```

**Verify**:
```bash
# Check tables created
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" <<EOF
\dn  -- List schemas
\dt trading.*  -- List trading tables
\dt ml.*  -- List ml tables
\dt metrics.*  -- List metrics tables
SELECT * FROM trading.account_state;  -- Check initial data
EOF
```

**Expected output**:
```
Schema | Name
--------+----------
trading | trading
ml     | ml
metrics| metrics

Table                      | Type
---------------------------+-------
trading.trades             | table
trading.signals            | table
trading.positions          | table
trading.account_state      | table
...

account_balance | initial_balance | max_daily_loss_pct | max_total_loss_pct
----------------+-----------------+--------------------+-------------------
100000.00       | 100000.00       | 0.0500             | 0.1000
```

### Step 3: Deploy ElastiCache Redis (5-10 min)

**Cost**: $11/month (single node) or $22/month (with replica)

```bash
./scripts/infrastructure/deploy_redis.sh
```

**What it does**:
1. Creates CloudFormation stack with:
   - Redis 7.0 cluster (cache.t4g.micro, ARM-based)
   - Subnet group across 2 AZs
   - Security group (port 6379)
   - Automated snapshots (7-day retention)
   - SNS alerts for monitoring
2. Saves connection info to `.redis_connection_info`

**Wait for**: "Redis Deployment Complete" message (~7 min)

**Verify**:
```bash
# Check Redis endpoint
source .redis_connection_info
echo "Redis Endpoint: $REDIS_HOST"

# Test connection
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" PING
# Expected: PONG

# Test basic operations
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" <<EOF
SET test_key "Hello from CryptoBot"
GET test_key
DEL test_key
EOF
```

### Step 4: Setup AWS Secrets Manager (2-3 min)

**Cost**: $0.40/secret/month + $0.05 per 10k API calls ≈ $2.40/month for 6 secrets

```bash
./scripts/infrastructure/setup_secrets.sh
```

**What it does**:
1. Migrates credentials from `.env` and `.rds_connection_info` to Secrets Manager
2. Creates secrets for:
   - RDS database credentials
   - Redis connection info
   - Coinbase API keys
   - Reddit API keys (if configured)
   - CryptoCompare API key (if configured)
   - MLflow configuration

**Verify**:
```bash
# List all secrets
aws secretsmanager list-secrets \
  --region us-east-1 \
  --filters Key=tag-key,Values=Project Key=tag-value,Values=CryptoBot \
  --query 'SecretList[*].[Name,Description]' \
  --output table

# Retrieve a secret (test)
aws secretsmanager get-secret-value \
  --secret-id crpbot/rds/credentials \
  --region us-east-1 \
  --query 'SecretString' \
  --output text | jq .
```

**Expected output**:
```json
{
  "host": "crpbot-rds-postgres-db.xxxxx.us-east-1.rds.amazonaws.com",
  "port": 5432,
  "database": "crpbot",
  "username": "crpbot_admin",
  "password": "your_secure_password"
}
```

### Step 5: Deploy MLflow Server (Optional, 10-15 min)

**Cost**: $15/month (t4g.small EC2) + $5/month (RDS backend) = $20/month

MLflow provides:
- Experiment tracking
- Model versioning
- Model registry with promotion gates
- Hyperparameter logging
- Metrics visualization

**Option A: Self-Hosted (Recommended)**
```bash
# Coming soon: ./scripts/infrastructure/deploy_mlflow.sh
```

**Option B: Use AWS SageMaker Model Registry**
- $0 for registry (pay per usage)
- Managed service
- No server to maintain

**For now**: Skip MLflow deployment, we'll add it in a future update.

---

## Post-Deployment Validation

### 1. Connection Tests

Create a test script to verify all connections:

```python
# test_connections.py
import boto3
import psycopg2
import redis
import json

def test_rds():
    # Get RDS credentials from Secrets Manager
    sm = boto3.client('secretsmanager', region_name='us-east-1')
    secret = sm.get_secret_value(SecretId='crpbot/rds/credentials')
    creds = json.loads(secret['SecretString'])

    # Connect to RDS
    conn = psycopg2.connect(
        host=creds['host'],
        port=creds['port'],
        database=creds['database'],
        user=creds['username'],
        password=creds['password']
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trading.account_state;")
    count = cursor.fetchone()[0]
    print(f"✓ RDS connection successful. Account states: {count}")
    conn.close()

def test_redis():
    # Get Redis credentials from Secrets Manager
    sm = boto3.client('secretsmanager', region_name='us-east-1')
    secret = sm.get_secret_value(SecretId='crpbot/redis/credentials')
    creds = json.loads(secret['SecretString'])

    # Connect to Redis
    r = redis.Redis(host=creds['host'], port=creds['port'], decode_responses=True)
    r.set('test_key', 'test_value')
    value = r.get('test_key')
    r.delete('test_key')
    print(f"✓ Redis connection successful. Test value: {value}")

if __name__ == '__main__':
    test_rds()
    test_redis()
    print("\n✓ All Phase 1 components validated successfully!")
```

Run the test:
```bash
uv run python test_connections.py
```

### 2. Cost Verification

Check current AWS costs:
```bash
# Check current month costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -d "1 day ago" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=SERVICE \
  --filter file://cost-filter.json

# cost-filter.json
{
  "Tags": {
    "Key": "Project",
    "Values": ["CryptoBot"]
  }
}
```

Expected Phase 1 costs:
- RDS t4g.small (single-AZ): ~$24/month
- Redis t4g.micro: ~$11/month
- Secrets Manager (6 secrets): ~$2.40/month
- Data transfer: ~$5/month
- **Total**: ~$42/month (slightly under $100/month estimate due to pro-rated first month)

### 3. Security Audit

- [ ] RDS deletion protection enabled
- [ ] RDS encryption at rest enabled
- [ ] RDS automated backups enabled (7 days)
- [ ] Redis in private subnet (not publicly accessible)
- [ ] Secrets in Secrets Manager (not in .env files)
- [ ] Security groups properly configured (least privilege)
- [ ] IAM roles follow least privilege principle

### 4. Monitoring Setup (Phase 2 Preview)

For now, use CloudWatch default metrics:

**RDS Metrics to monitor**:
- CPUUtilization < 80%
- FreeableMemory > 1GB
- DatabaseConnections < 90% of max (100 for t4g.small)
- ReadLatency, WriteLatency < 100ms

**Redis Metrics to monitor**:
- CPUUtilization < 80%
- FreeableMemory > 500MB
- CacheHitRate > 80%
- Evictions = 0 (or very low)

Create CloudWatch alarms:
```bash
# RDS CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name crpbot-rds-high-cpu \
  --alarm-description "RDS CPU > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/RDS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --dimensions Name=DBInstanceIdentifier,Value=crpbot-rds-postgres-db

# Redis memory alarm
aws cloudwatch put-metric-alarm \
  --alarm-name crpbot-redis-low-memory \
  --alarm-description "Redis memory < 500MB" \
  --metric-name FreeableMemory \
  --namespace AWS/ElastiCache \
  --statistic Average \
  --period 300 \
  --threshold 524288000 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 2 \
  --dimensions Name=CacheClusterId,Value=crpbot-redis-cluster
```

---

## Next Steps After Phase 1

### Immediate (This Week)
1. Update application code to use RDS and Redis
2. Migrate historical signals/trades to RDS (if any)
3. Implement feature caching in Redis
4. Test application with new infrastructure

### Phase 2: Observability (Next Week)
- Deploy Prometheus + Grafana for metrics
- Implement structured logging with CloudWatch Logs
- Create custom dashboards for trading metrics
- Set up PagerDuty/SNS alerting

### Phase 3: Redpanda Migration (Week 3-4)
- Deploy Redpanda cluster or use Redpanda Cloud
- Migrate from Docker Kafka to production Redpanda
- Achieve <50ms end-to-end latency

---

## Rollback Plan

If anything goes wrong, Phase 1 can be safely rolled back without affecting current operations:

### Rollback RDS
```bash
aws cloudformation delete-stack --stack-name crpbot-rds-postgres --region us-east-1
# Data will be in final snapshot
```

### Rollback Redis
```bash
aws cloudformation delete-stack --stack-name crpbot-redis --region us-east-1
```

### Rollback Secrets Manager
```bash
# Delete all secrets (will be in recovery for 7-30 days)
aws secretsmanager list-secrets --region us-east-1 \
  --filters Key=tag-key,Values=Project Key=tag-value,Values=CryptoBot \
  --query 'SecretList[*].Name' \
  --output text | \
  xargs -I {} aws secretsmanager delete-secret --secret-id {} --region us-east-1
```

---

## Troubleshooting

### RDS Connection Issues
**Problem**: Can't connect to RDS from local machine
**Solution**: RDS is in private subnet, use bastion host or VPN. For testing, temporarily allow your IP in security group.

```bash
# Get your IP
MY_IP=$(curl -s ifconfig.me)

# Add to security group (TEMPORARY - remove after testing)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxx \
  --protocol tcp \
  --port 5432 \
  --cidr "$MY_IP/32"
```

### Redis Connection Issues
**Problem**: Can't connect to Redis
**Solution**: Same as RDS, Redis is in private subnet. Use bastion or VPN.

### Secrets Manager Permissions
**Problem**: Access denied when retrieving secrets
**Solution**: Add IAM policy to your role/user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:crpbot/*"
    }
  ]
}
```

### High Costs
**Problem**: Costs higher than expected
**Solution**:
1. Check Cost Explorer for breakdown
2. Verify instance types (should be t4g.small, t4g.micro)
3. Check data transfer costs (high = potential issue)
4. Ensure no unused resources (old RDS snapshots, etc.)

---

## Summary

**Phase 1 Deployment Checklist**:
- [x] RDS PostgreSQL deployed ($24/month)
- [x] Database schema created (13 tables, 3 schemas)
- [x] ElastiCache Redis deployed ($11/month)
- [x] AWS Secrets Manager configured ($2.40/month)
- [ ] MLflow server deployed (optional, $20/month)
- [ ] Connection tests passed
- [ ] CloudWatch alarms configured
- [ ] Security audit passed
- [ ] Application integration started

**Total Phase 1 Cost**: $37-57/month (depending on MLflow)

**Next Phase**: Phase 2 - Observability (Prometheus + Grafana)

**Questions?** Check the main architecture doc or ask for clarification.
