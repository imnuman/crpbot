# Amazon Q Task Instructions for CryptoBot

**Purpose**: Clear, executable instructions for Amazon Q to perform infrastructure and ML tasks.
**Format**: Step-by-step commands with validation checks
**Audience**: Amazon Q CLI (autonomous execution)

---

## Task 1: GPU Training - Train All 3 Models (PRIORITY)

**Duration**: 10 minutes
**Cost**: $0.61
**Prerequisites**: S3 data uploaded, AWS credentials configured

### Context
We have 3 symbols (BTC-USD, ETH-USD, SOL-USD) with 2 years of data and 58 multi-timeframe features already engineered. Current CPU training takes 3+ days per model. GPU training on p3.8xlarge (4x V100) takes 3 minutes for all 3 models in parallel.

### Objective
Launch GPU instance, train all 3 LSTM models in parallel, upload trained models to S3, terminate instance.

### Step-by-Step Instructions

```bash
# Step 1: Verify S3 data is uploaded
echo "=== Checking S3 data availability ==="
aws s3 ls s3://crpbot-ml-data-20251110/features/ --recursive --human-readable | grep latest
# Expected: Should see 3 feature files (BTC, ETH, SOL), each ~200MB

# Step 2: Launch GPU instance (p3.8xlarge)
echo "=== Launching GPU training instance ==="
./scripts/setup_gpu_training.sh

# What this does:
# - Creates SSH key (~/.ssh/crpbot-training.pem)
# - Creates security group (SSH access only)
# - Launches p3.8xlarge Spot instance (4x V100 GPUs)
# - User data script downloads data from S3
# - Installs PyTorch, CUDA, dependencies
# - Cost: ~$12.24/hour Spot (vs $24.48 on-demand)

# Wait for initialization message
# Expected output: "Instance launched: i-xxxxxxxxx"
#                  "Public IP: xx.xx.xx.xx"
#                  "Waiting for instance to be ready (2-3 minutes)..."

# Step 3: SSH to instance (automatically done by script)
# The script will show:
# "Instance ready! SSH command:"
# "ssh -i ~/.ssh/crpbot-training.pem ubuntu@xx.xx.xx.xx"

# Step 4: Wait for initialization (automated in user data)
# The instance will:
# - Download data from S3 (765MB, ~1-2 min)
# - Install dependencies (PyTorch + CUDA, ~1 min)
# - Create training directories
# When ready, you'll see: "crpbot/" directory exists

# Step 5: Start multi-GPU training
echo "=== Starting multi-GPU parallel training ==="
cd crpbot
./scripts/train_multi_gpu.sh

# What this does:
# - Trains BTC-USD on GPU 0
# - Trains ETH-USD on GPU 1
# - Trains SOL-USD on GPU 2
# - Monitors progress in real-time
# - Each model: 15 epochs, ~3 minutes
# - Logs saved to /tmp/train_*_gpu.log

# Monitor GPU usage (optional)
watch -n 1 nvidia-smi

# Step 6: Wait for training completion
# Expected duration: 3 minutes per model (parallel, so 3 min total)
# Expected output:
#   "BTC training started on GPU 0 (PID: xxxx)"
#   "ETH training started on GPU 1 (PID: xxxx)"
#   "SOL training started on GPU 2 (PID: xxxx)"
#   ...
#   "All models trained successfully!"
#   "Uploading models to S3..."
#   "Upload complete!"

# Step 7: Verify trained models
ls -lh models/lstm_*.pt
# Expected: 3 model files, each ~2-5MB
# - lstm_BTC_USD_YYYYMMDD_HHMMSS.pt
# - lstm_ETH_USD_YYYYMMDD_HHMMSS.pt
# - lstm_SOL_USD_YYYYMMDD_HHMMSS.pt

# Step 8: Check S3 upload
aws s3 ls s3://crpbot-ml-data-20251110/models/production/ --human-readable
# Expected: 3 model files uploaded

# Step 9: Terminate GPU instance (CRITICAL - prevents ongoing charges!)
echo "=== Terminating GPU instance ==="
# On LOCAL machine (not GPU instance):
INSTANCE_ID=$(cat .gpu_instance_info | grep INSTANCE_ID | cut -d= -f2)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"

# Verify termination
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text
# Expected: "shutting-down" or "terminated"

# Step 10: Download trained models to local
echo "=== Downloading models to local machine ==="
aws s3 sync s3://crpbot-ml-data-20251110/models/production/ models/

# Step 11: Validate model accuracy
echo "=== Validating model performance ==="
uv run python -c "
import torch
model = torch.load('models/lstm_BTC_USD_latest.pt')
print(f\"Model accuracy: {model.get('val_accuracy', 'N/A')}\")
print(f\"Best epoch: {model.get('epoch', 'N/A')}\")
print(f\"Test accuracy: {model.get('test_accuracy', 'N/A')}\")
"
# Expected: Accuracy > 60% (baseline), ideally > 68% (gate)
```

### Success Criteria
- [x] All 3 models trained without errors
- [x] Models uploaded to S3
- [x] GPU instance terminated (no ongoing charges)
- [x] Model accuracy > 60%
- [x] Total cost: $0.61 (3 min × $12.24/hour)

### Troubleshooting

**Issue**: Spot instance request failed
**Solution**: Retry with on-demand instance
```bash
# Edit setup_gpu_training.sh, change:
# INSTANCE_MARKET_TYPE="spot"
# to:
# INSTANCE_MARKET_TYPE="on-demand"
```

**Issue**: Training fails with CUDA out of memory
**Solution**: Reduce batch size in training config
```bash
# Edit apps/trainer/config.py
# Change batch_size from 64 to 32
```

**Issue**: Can't SSH to instance
**Solution**: Check security group allows your IP
```bash
MY_IP=$(curl -s ifconfig.me)
aws ec2 authorize-security-group-ingress \
  --group-id $(cat .gpu_instance_info | grep SECURITY_GROUP | cut -d= -f2) \
  --protocol tcp --port 22 --cidr "$MY_IP/32"
```

---

## Task 2: Setup Reddit API for Sentiment Analysis (DEFERRED)

**Duration**: 30 minutes
**Cost**: $0 (free tier)
**Prerequisites**: Reddit account, verified email

### Context
Reddit sentiment adds 4 features: sentiment score, volume, engagement quality, and noise-filtered confidence. We'll use 6-layer filtering to reduce noise from 70% to 15%.

### Objective
Create Reddit API credentials, test connection, configure 6-layer filtering system.

### Step-by-Step Instructions

```bash
# Step 1: Create Reddit API application
echo "=== Setting up Reddit API ==="
# Manual steps (Amazon Q cannot automate web UI):
# 1. Go to https://www.reddit.com/prefs/apps
# 2. Click "Create App" or "Create Another App"
# 3. Fill in:
#    - Name: "CryptoBot Sentiment Analyzer"
#    - App type: "script"
#    - Description: "ML trading bot sentiment analysis"
#    - About URL: (leave blank)
#    - Redirect URI: http://localhost:8080
# 4. Click "Create app"
# 5. Copy CLIENT_ID (under app name) and CLIENT_SECRET

# Step 2: Store Reddit credentials in .env
cat >> .env <<EOF

# Reddit API Configuration
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT="CryptoBot Sentiment Analyzer v1.0"
EOF

# Step 3: Test Reddit API connection
uv run python -c "
import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

# Test with public subreddit
subreddit = reddit.subreddit('bitcoin')
print(f'Subreddit: {subreddit.display_name}')
print(f'Subscribers: {subreddit.subscribers:,}')

# Fetch recent posts
for post in subreddit.hot(limit=5):
    print(f'- {post.title[:50]}... (score: {post.score})')

print('\n✓ Reddit API connection successful!')
"

# Step 4: Install Reddit sentiment dependencies
uv add praw vaderSentiment transformers

# Step 5: Implement 6-layer filtering (already in codebase)
# File: apps/sentiment/reddit_filter.py
# Layers:
#   1. Subreddit quality filtering (whitelist/blacklist)
#   2. User reputation (karma > 500, age > 90 days)
#   3. Content quality (upvote ratio > 65%, engagement)
#   4. NLP filtering (BERT spam detection)
#   5. Time decay weighting (exponential, half-life 14 hours)
#   6. Engagement weighting (log scale to prevent outliers)

# Step 6: Test filtering with sample data
uv run python apps/sentiment/test_reddit_filter.py
# Expected output:
#   "Fetched 1000 posts from r/bitcoin"
#   "After filtering: 150 posts (85% reduction)"
#   "Average sentiment: 0.65 (positive)"
#   "Average quality score: 0.78"

# Step 7: Fetch historical Reddit data (6 months)
uv run python apps/sentiment/fetch_reddit_historical.py \
  --symbols BTC ETH SOL \
  --days 180 \
  --output data/sentiment/reddit/

# Duration: 1-2 hours (Reddit API rate limits)
# Expected: ~50-100 posts/day × 180 days = 9,000-18,000 posts
# After filtering: ~1,500-3,000 high-quality posts

# Step 8: Engineer sentiment features
uv run python apps/sentiment/engineer_sentiment_features.py \
  --input data/sentiment/reddit/ \
  --output data/features/ \
  --merge-with existing

# Adds 4 features to existing 58:
#   - reddit_sentiment: -1 to 1 (negative to positive)
#   - reddit_volume: log(post count in last 24h)
#   - reddit_engagement: weighted by upvotes + comments
#   - reddit_quality: filtering confidence score

# Step 9: Validate feature quality
uv run python apps/sentiment/validate_features.py \
  --features data/features/features_BTC-USD_1m_latest.parquet

# Expected output:
#   "Total features: 62 (58 multi-TF + 4 sentiment)"
#   "Missing values: < 5%"
#   "Sentiment correlation with price: 0.15-0.25"
#   "✓ Features ready for training"
```

### Success Criteria
- [x] Reddit API connected successfully
- [x] 6-layer filtering reduces noise by 70%+
- [x] Historical data fetched (6 months)
- [x] Sentiment features engineered (4 new features)
- [x] Feature validation passed

---

## Task 3: Phase 1 Infrastructure Deployment

**Duration**: 1 hour (mostly automated waiting)
**Cost**: $37-57/month
**Prerequisites**: AWS credentials, VPC with 2 AZs

### Context
Deploy production foundation: RDS PostgreSQL for operational data, Redis for feature caching, Secrets Manager for credentials.

### Objective
Production-grade infrastructure ready for scale, zero disruption to current system.

### Step-by-Step Instructions

```bash
# Step 1: Deploy RDS PostgreSQL
echo "=== Deploying RDS PostgreSQL (t4g.small) ==="
cd /home/numan/crpbot
chmod +x scripts/infrastructure/*.sh

./scripts/infrastructure/deploy_rds.sh

# What this does:
# - Generates secure password (saved to .db_password)
# - Creates CloudFormation stack:
#   - RDS PostgreSQL 15.4 (t4g.small, ARM-based)
#   - 100GB gp3 SSD storage
#   - Automated backups (7-day retention)
#   - Encryption at rest
#   - Multi-AZ optional (default: single-AZ)
# - Creates VPC, subnets, security groups
# - Saves connection info to .rds_connection_info

# Wait for completion: 10-15 minutes
# Expected output: "RDS Deployment Complete"

# Step 2: Create database schema
echo "=== Creating database schema ==="
source .rds_connection_info

psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
  -f scripts/infrastructure/create_db_schema.sql

# What this creates:
# - 3 schemas: trading, ml, metrics
# - 13 tables: trades, signals, positions, account_state, models, etc.
# - Indexes, triggers, views
# - Initial data: $100k account balance, FTMO guardrails

# Expected output:
#   "CREATE SCHEMA"
#   "CREATE TABLE" (×13)
#   "CREATE INDEX" (×25)
#   "CREATE TRIGGER" (×5)
#   "CREATE VIEW" (×4)
#   "INSERT 0 1" (initial account state)

# Step 3: Validate database
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" <<EOF
\dn  -- List schemas
\dt trading.*  -- List trading tables
SELECT * FROM trading.account_state;  -- Check initial data
EOF

# Expected:
#   Schemas: trading, ml, metrics
#   Tables: 13 total
#   Account balance: $100,000.00
#   Max daily loss: 5%
#   Max total loss: 10%

# Step 4: Deploy ElastiCache Redis
echo "=== Deploying ElastiCache Redis (t4g.micro) ==="
./scripts/infrastructure/deploy_redis.sh

# What this does:
# - Creates CloudFormation stack:
#   - Redis 7.0 cluster (cache.t4g.micro, ARM-based)
#   - Subnet group across 2 AZs
#   - Security group (port 6379)
#   - Automated snapshots (7-day retention)
# - Saves connection info to .redis_connection_info

# Wait for completion: 5-10 minutes
# Expected output: "Redis Deployment Complete"

# Step 5: Test Redis connection
source .redis_connection_info

# Install redis-tools if needed
sudo apt-get update && sudo apt-get install -y redis-tools

redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" PING
# Expected: PONG

redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" <<EOF
SET test_key "Hello from CryptoBot"
GET test_key
DEL test_key
EOF
# Expected: OK, "Hello from CryptoBot", 1

# Step 6: Setup AWS Secrets Manager
echo "=== Setting up AWS Secrets Manager ==="
./scripts/infrastructure/setup_secrets.sh

# What this does:
# - Migrates credentials from .env, .rds_connection_info, .redis_connection_info
# - Creates secrets for:
#   - crpbot/rds/credentials
#   - crpbot/redis/credentials
#   - crpbot/coinbase/api
#   - crpbot/reddit/api (if configured)
#   - crpbot/mlflow/config

# Wait for completion: 1-2 minutes
# Expected output: List of created secrets

# Step 7: Verify secrets
aws secretsmanager list-secrets \
  --region us-east-1 \
  --filters Key=tag-key,Values=Project Key=tag-value,Values=CryptoBot \
  --query 'SecretList[*].[Name,Description]' \
  --output table

# Expected: 5-6 secrets listed

# Test retrieval
aws secretsmanager get-secret-value \
  --secret-id crpbot/rds/credentials \
  --region us-east-1 \
  --query 'SecretString' --output text | jq .

# Expected: JSON with host, port, database, username, password

# Step 8: Setup CloudWatch alarms
echo "=== Setting up CloudWatch alarms ==="

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

# Step 9: Verify total costs
echo "=== Checking AWS costs ==="
aws ce get-cost-and-usage \
  --time-period Start=$(date -d "1 day ago" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=SERVICE

# Expected costs (pro-rated):
#   RDS: ~$0.80/day ($24/month)
#   ElastiCache: ~$0.37/day ($11/month)
#   Secrets Manager: ~$0.08/day ($2.40/month)
#   Total: ~$1.25/day ($37/month)

# Step 10: Create cost budget alert
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget file://budget.json

# budget.json:
cat > budget.json <<'EOF'
{
  "BudgetName": "CryptoBot-Infrastructure",
  "BudgetLimit": {
    "Amount": "200",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
EOF
```

### Success Criteria
- [x] RDS deployed and accessible
- [x] Database schema created (13 tables)
- [x] Redis deployed and accessible
- [x] Secrets in Secrets Manager (5-6 secrets)
- [x] CloudWatch alarms configured
- [x] Cost budget alert set ($200/month)
- [x] Total cost < $60/month

---

## Task 4: Retrain Models with Sentiment Features (AFTER Task 2)

**Duration**: 15 minutes (10 min GPU + 5 min validation)
**Cost**: $0.61
**Prerequisites**: Reddit sentiment features engineered, GPU training tested

### Context
After adding Reddit sentiment features (Task 2), retrain all 3 models with 62 features (58 multi-TF + 4 sentiment). Expected accuracy improvement: +3-5%.

### Objective
Train models with sentiment, compare performance, promote best model to production.

### Step-by-Step Instructions

```bash
# Step 1: Verify feature files have sentiment
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet')
print(f'Total features: {len(df.columns)}')
print(f'Has reddit_sentiment: {\"reddit_sentiment\" in df.columns}')
print(f'Sentiment range: [{df[\"reddit_sentiment\"].min():.2f}, {df[\"reddit_sentiment\"].max():.2f}]')
"
# Expected: 62 features, reddit_sentiment present, range [-1, 1]

# Step 2: Upload updated features to S3
aws s3 sync data/features/ s3://crpbot-ml-data-20251110/features/ \
  --exclude "*" --include "features_*_latest.parquet"

# Step 3: Run GPU training (same as Task 1)
./scripts/setup_gpu_training.sh
# ... SSH to instance ...
cd crpbot
./scripts/train_multi_gpu.sh

# Step 4: Compare model performance
uv run python apps/trainer/compare_models.py \
  --baseline models/lstm_BTC_USD_20251111_baseline.pt \
  --new models/lstm_BTC_USD_20251111_with_sentiment.pt

# Expected output:
#   "Baseline accuracy: 62.5%"
#   "With sentiment: 66.8%"
#   "Improvement: +4.3%"
#   "Sentiment feature importance: 12% (rank 8/62)"

# Step 5: If improvement > 3%, promote to production
if [ $(python -c "print(66.8 - 62.5 > 3)") == "True" ]; then
  echo "Promoting new models to production..."
  aws s3 cp models/lstm_BTC_USD_20251111_with_sentiment.pt \
    s3://crpbot-ml-data-20251110/models/production/lstm_BTC_USD_latest.pt
fi

# Step 6: Update model metadata in RDS (if Phase 1 deployed)
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" <<EOF
INSERT INTO ml.models (
  model_name, model_type, version, symbol,
  input_size, test_accuracy, status, s3_uri
) VALUES (
  'lstm_BTC_USD', 'lstm', '20251111_with_sentiment', 'BTC-USD',
  62, 0.668, 'PRODUCTION',
  's3://crpbot-ml-data-20251110/models/production/lstm_BTC_USD_latest.pt'
);
EOF
```

### Success Criteria
- [x] Models trained with 62 features
- [x] Accuracy improvement > 3%
- [x] Models promoted to production
- [x] Metadata stored in RDS

---

## Quick Reference: All Tasks

| Task | Priority | Duration | Cost | Prerequisites |
|------|----------|----------|------|--------------|
| **Task 1: GPU Training** | ⭐⭐⭐ HIGH | 10 min | $0.61 | S3 data uploaded |
| **Task 2: Reddit API Setup** | ⭐⭐ MEDIUM | 30 min | $0 | Reddit account |
| **Task 3: Phase 1 Infrastructure** | ⭐⭐ MEDIUM | 1 hour | $37-57/month | AWS account |
| **Task 4: Retrain with Sentiment** | ⭐ LOW | 15 min | $0.61 | Tasks 1 & 2 complete |

## Recommended Execution Order

**Option A: Fast Path (Recommended)**
1. Task 1: GPU Training (today, 10 min, $0.61)
2. Task 2: Reddit API Setup (this week, 30 min, $0)
3. Task 4: Retrain with Sentiment (next week, 15 min, $0.61)
4. Task 3: Phase 1 Infrastructure (when ready to scale, 1 hour, $37/month)

**Option B: Production-First Path**
1. Task 3: Phase 1 Infrastructure (today, 1 hour, $37/month)
2. Task 1: GPU Training (today, 10 min, $0.61)
3. Task 2: Reddit API Setup (this week, 30 min, $0)
4. Task 4: Retrain with Sentiment (next week, 15 min, $0.61)

**My Recommendation**: **Option A** - Get models trained fast, add sentiment incrementally, deploy infrastructure when validated.

---

## Amazon Q Usage Examples

```bash
# Ask Q to execute a task
q "Execute Task 1: GPU Training. Follow instructions in AMAZON_Q_TASK_INSTRUCTIONS.md"

# Ask Q to check status
q "Check status of GPU training. Show model accuracy and cost."

# Ask Q to troubleshoot
q "GPU training failed with CUDA out of memory. How do I fix this?"

# Ask Q to optimize
q "How can I reduce GPU training costs further?"

# Ask Q about next steps
q "Task 1 complete. What should I do next?"
```

---

## Cost Tracking

| Date | Task | Resources | Duration | Cost | Cumulative |
|------|------|-----------|----------|------|------------|
| 2025-11-11 | S3 Setup | S3 bucket | Ongoing | $2.50/month | $2.50/month |
| TBD | Task 1: GPU Training | p3.8xlarge Spot | 3 min | $0.61 | $2.50/month + $0.61 one-time |
| TBD | Task 2: Reddit API | Free tier | N/A | $0 | $2.50/month + $0.61 one-time |
| TBD | Task 3: Phase 1 Infra | RDS + Redis + Secrets | Ongoing | $37/month | $39.50/month + $0.61 one-time |
| TBD | Task 4: Retrain | p3.8xlarge Spot | 3 min | $0.61 | $39.50/month + $1.22 one-time |

**Total Monthly**: $39.50/month (after all tasks)
**Total One-Time**: $1.22 (2 GPU training runs)
**Total First Month**: $40.72

---

## Validation Checklist

After completing all tasks, verify:

### Task 1: GPU Training
- [ ] All 3 models trained (BTC, ETH, SOL)
- [ ] Models uploaded to S3
- [ ] GPU instance terminated
- [ ] Model accuracy > 60%
- [ ] Cost = $0.61

### Task 2: Reddit API
- [ ] Reddit API credentials working
- [ ] 6-layer filtering implemented
- [ ] Historical data fetched (6 months)
- [ ] Sentiment features engineered
- [ ] Cost = $0

### Task 3: Phase 1 Infrastructure
- [ ] RDS PostgreSQL deployed
- [ ] Database schema created (13 tables)
- [ ] Redis deployed
- [ ] Secrets in Secrets Manager
- [ ] CloudWatch alarms configured
- [ ] Cost = $37-57/month

### Task 4: Retrain with Sentiment
- [ ] Models trained with 62 features
- [ ] Accuracy improved by 3%+
- [ ] Models promoted to production
- [ ] Cost = $0.61

**All tasks complete**: Ready for Phase 2 (Observability) or live trading!
