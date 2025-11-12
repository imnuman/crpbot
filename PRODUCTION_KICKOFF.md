# Production Kickoff - Full Deployment Sequence

**Started**: 2025-11-11
**Target**: Full production in 4 weeks
**Budget**: $422/month operational + $0.61 one-time

---

## ✅ STEP 1: GPU Training (EXECUTING NOW)

**Time**: 10 minutes
**Cost**: $0.61 one-time
**Status**: ⏳ IN PROGRESS

### Commands to Execute
```bash
# Verify S3 data is ready
aws s3 ls s3://crpbot-ml-data-20251110/features/ --human-readable | grep latest

# Launch GPU training
./scripts/setup_gpu_training.sh
```

### What This Does
1. Launches p3.8xlarge Spot instance (4x V100 GPUs)
2. Downloads 765MB data from S3
3. Trains BTC, ETH, SOL models in parallel
4. Each model: 15 epochs, ~3 minutes
5. Uploads trained models to S3
6. **YOU MUST TERMINATE INSTANCE** after training completes

### Expected Output
```
Instance launched: i-xxxxxxxxx
Public IP: xx.xx.xx.xx
Waiting for instance to be ready (2-3 minutes)...
Instance ready! SSH command:
ssh -i ~/.ssh/crpbot-training.pem ubuntu@xx.xx.xx.xx

[On instance, training auto-starts]
BTC training started on GPU 0 (PID: xxxx)
ETH training started on GPU 1 (PID: xxxx)
SOL training started on GPU 2 (PID: xxxx)
...
All models trained successfully!
Uploading models to S3...
Upload complete!
```

### Termination (CRITICAL!)
```bash
# On LOCAL machine after training completes:
INSTANCE_ID=$(cat .gpu_instance_info | grep INSTANCE_ID | cut -d= -f2)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"

# Verify termination
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text
# Should show: "shutting-down" or "terminated"
```

**⚠️ SET PHONE ALARM FOR 10 MINUTES** - Don't forget to terminate!

---

## STEP 2: Phase 1 Infrastructure (NEXT)

**Time**: 1 hour
**Cost**: $37/month
**Status**: ⏳ READY TO DEPLOY

### 2.1 Deploy RDS PostgreSQL (15 min)
```bash
./scripts/infrastructure/deploy_rds.sh
```

**What it does**:
- Creates t4g.small PostgreSQL 15.4 instance
- 100GB gp3 SSD storage, encryption enabled
- Automated backups (7-day retention)
- Saves connection info to `.rds_connection_info`

**Wait for**: "RDS Deployment Complete" (~15 min)

### 2.2 Create Database Schema (2 min)
```bash
source .rds_connection_info
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
  -f scripts/infrastructure/create_db_schema.sql
```

**What it creates**:
- 3 schemas: trading, ml, metrics
- 13 tables: trades, signals, positions, account_state, models, etc.
- Indexes, triggers, views
- Initial $100k account with FTMO guardrails

### 2.3 Deploy ElastiCache Redis (7 min)
```bash
./scripts/infrastructure/deploy_redis.sh
```

**What it does**:
- Creates cache.t4g.micro Redis 7.0 cluster
- Automated snapshots (7-day retention)
- Saves connection info to `.redis_connection_info`

**Wait for**: "Redis Deployment Complete" (~7 min)

### 2.4 Setup AWS Secrets Manager (2 min)
```bash
./scripts/infrastructure/setup_secrets.sh
```

**What it does**:
- Migrates credentials from .env to Secrets Manager
- Creates 6 secrets (RDS, Redis, Coinbase, etc.)
- Cost: $0.40/secret/month = $2.40/month

### 2.5 Verify Deployment
```bash
# Test RDS connection
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" -c "SELECT COUNT(*) FROM trading.account_state;"

# Test Redis connection
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" PING

# List secrets
aws secretsmanager list-secrets --filters Key=tag-key,Values=Project Key=tag-value,Values=CryptoBot
```

**Success Criteria**:
- [x] RDS returns 1 account record
- [x] Redis returns PONG
- [x] 6 secrets listed

---

## STEP 3: Redpanda Cloud Setup (NEXT - 4-8 hours)

**Time**: 4-8 hours
**Cost**: $299/month
**Status**: ⏳ PENDING

### 3.1 Sign Up for Redpanda Cloud
```bash
# Go to: https://redpanda.com/try-redpanda
# Create account (free trial available)
# Verify email
```

### 3.2 Create Cluster
**Configuration**:
- Name: crpbot-production
- Cloud: AWS
- Region: us-east-1
- Tier: Tier 1 (3 nodes, 300 MB/s ingress)
- Cost: $299/month

**Create via Web UI**:
1. Click "Create Cluster"
2. Select AWS → us-east-1
3. Choose Tier 1 (production-grade)
4. Name: crpbot-production
5. Click "Create" (~10-15 min provisioning)

### 3.3 Get Connection Details
After cluster is ready:
1. Go to cluster dashboard
2. Copy bootstrap servers (e.g., `crpbot-production-xxxx.redpanda.cloud:9092`)
3. Create SASL credentials
4. Download certificate (if using TLS)

Save to `.redpanda_connection_info`:
```bash
cat > .redpanda_connection_info <<EOF
REDPANDA_BROKERS=crpbot-production-xxxx.redpanda.cloud:9092
REDPANDA_SASL_USERNAME=your_username
REDPANDA_SASL_PASSWORD=your_password
REDPANDA_SASL_MECHANISM=SCRAM-SHA-256
EOF
chmod 600 .redpanda_connection_info
```

### 3.4 Create Topics
```bash
# Install rpk (Redpanda CLI)
curl -LO https://github.com/redpanda-data/redpanda/releases/latest/download/rpk-linux-amd64.zip
unzip rpk-linux-amd64.zip
sudo mv rpk /usr/local/bin/
rm rpk-linux-amd64.zip

# Configure rpk
source .redpanda_connection_info
rpk cluster info \
  --brokers "$REDPANDA_BROKERS" \
  --user "$REDPANDA_SASL_USERNAME" \
  --password "$REDPANDA_SASL_PASSWORD" \
  --sasl-mechanism "$REDPANDA_SASL_MECHANISM"

# Create topics (from KAFKA_IMPLEMENTATION.md)
topics=(
  "market.data.raw.BTC-USD"
  "market.data.raw.ETH-USD"
  "market.data.raw.SOL-USD"
  "features.multi_tf.BTC-USD"
  "features.multi_tf.ETH-USD"
  "features.multi_tf.SOL-USD"
  "predictions.lstm.BTC-USD"
  "predictions.lstm.ETH-USD"
  "predictions.lstm.SOL-USD"
  "signals.aggregated.BTC-USD"
  "signals.aggregated.ETH-USD"
  "signals.aggregated.SOL-USD"
  "signals.calibrated.all"
  "trades.orders.all"
  "trades.executions.all"
)

for topic in "${topics[@]}"; do
  rpk topic create "$topic" \
    --brokers "$REDPANDA_BROKERS" \
    --user "$REDPANDA_SASL_USERNAME" \
    --password "$REDPANDA_SASL_PASSWORD" \
    --sasl-mechanism "$REDPANDA_SASL_MECHANISM" \
    --partitions 3 \
    --replicas 3
done
```

### 3.5 Update Application Config
```bash
# Update apps/kafka/config.py with Redpanda connection
# Replace KAFKA_BOOTSTRAP_SERVERS with Redpanda brokers
# Add SASL authentication config
```

**Success Criteria**:
- [x] Cluster provisioned and healthy
- [x] 15 topics created with 3 partitions each
- [x] Test message sent and received successfully

---

## STEP 4: Build & Deploy Kafka Consumers (NEXT - 1-2 days)

**Time**: 1-2 days
**Cost**: $50/month
**Status**: ⏳ PENDING

### 4.1 Create Dockerfiles
```bash
# Create Dockerfile for each consumer
# apps/kafka/streams/Dockerfile
# apps/kafka/consumers/Dockerfile
```

### 4.2 Build Docker Images
```bash
# Build feature engineering stream
docker build -t crpbot-feature-engineer:latest \
  -f apps/kafka/streams/Dockerfile .

# Build model inference consumer
docker build -t crpbot-model-inference:latest \
  -f apps/kafka/consumers/Dockerfile .

# Build signal aggregator
docker build -t crpbot-signal-aggregator:latest \
  -f apps/kafka/consumers/Dockerfile .

# Build execution engine
docker build -t crpbot-execution-engine:latest \
  -f apps/kafka/consumers/Dockerfile .
```

### 4.3 Push to ECR
```bash
# Create ECR repositories
aws ecr create-repository --repository-name crpbot-feature-engineer
aws ecr create-repository --repository-name crpbot-model-inference
aws ecr create-repository --repository-name crpbot-signal-aggregator
aws ecr create-repository --repository-name crpbot-execution-engine

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag crpbot-feature-engineer:latest \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com/crpbot-feature-engineer:latest
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com/crpbot-feature-engineer:latest

# Repeat for other images...
```

### 4.4 Deploy to ECS Fargate
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name crpbot-production

# Create task definitions (JSON files)
# Deploy using AWS CLI or Terraform
```

**Coming Soon**: `./scripts/deploy_consumers.sh` (automates all of this)

**Success Criteria**:
- [x] All 4 consumers deployed to ECS
- [x] Consumers connected to Redpanda
- [x] Health checks passing
- [x] Logs visible in CloudWatch

---

## STEP 5: Alpaca Broker Integration (NEXT - 2-3 days)

**Time**: 2-3 days
**Cost**: $0 (paper trading)
**Status**: ⏳ PENDING

### 5.1 Create Alpaca Account
```bash
# Go to: https://alpaca.markets/
# Sign up for free account
# Navigate to API Keys (Paper Trading)
# Generate API key and secret
```

### 5.2 Store Credentials in Secrets Manager
```bash
aws secretsmanager create-secret \
  --name crpbot/alpaca/api \
  --description "Alpaca API credentials for paper trading" \
  --secret-string '{
    "api_key": "your_api_key_here",
    "api_secret": "your_api_secret_here",
    "base_url": "https://paper-api.alpaca.markets"
  }' \
  --region us-east-1
```

### 5.3 Implement Alpaca Integration
**File**: `apps/kafka/consumers/brokers/alpaca_broker.py`

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
import boto3
import json

class AlpacaBroker:
    def __init__(self, paper=True):
        # Get credentials from Secrets Manager
        sm = boto3.client('secretsmanager', region_name='us-east-1')
        secret = sm.get_secret_value(SecretId='crpbot/alpaca/api')
        creds = json.loads(secret['SecretString'])

        self.client = TradingClient(
            api_key=creds['api_key'],
            secret_key=creds['api_secret'],
            paper=paper
        )

    def place_market_order(self, symbol, quantity, side):
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )

        order = self.client.submit_order(order_request)
        return order

    def get_account(self):
        return self.client.get_account()

    def get_positions(self):
        return self.client.get_all_positions()
```

### 5.4 Update Execution Engine
**File**: `apps/kafka/consumers/execution_engine.py`

Add Alpaca broker integration:
```python
from brokers.alpaca_broker import AlpacaBroker

# In __init__:
self.broker = AlpacaBroker(paper=True)

# In _execute_order:
def _execute_order(self, order: dict) -> dict:
    try:
        alpaca_order = self.broker.place_market_order(
            symbol=order['symbol'],
            quantity=order['quantity'],
            side=order['side']
        )

        # Store in RDS
        self._store_execution(order, alpaca_order)

        return {
            "status": "executed",
            "order_id": alpaca_order.id,
            "filled_qty": alpaca_order.filled_qty,
            "filled_avg_price": alpaca_order.filled_avg_price
        }
    except Exception as e:
        logger.error(f"Order execution failed: {e}")
        return {"status": "failed", "error": str(e)}
```

### 5.5 Test Paper Trading
```bash
# Deploy updated execution engine
# Send test signal
# Verify order placed in Alpaca paper account
```

**Success Criteria**:
- [x] Alpaca account connected
- [x] Test order placed successfully (paper)
- [x] Order status retrieved
- [x] Execution logged to RDS

---

## STEP 6: Monitoring & Observability (NEXT - 1-2 days)

**Time**: 1-2 days
**Cost**: $25/month
**Status**: ⏳ PENDING

### 6.1 Deploy Prometheus + Grafana
```bash
# Coming soon: ./scripts/infrastructure/deploy_monitoring.sh
```

### 6.2 Configure Dashboards
- Trading metrics (P&L, win rate, Sharpe ratio)
- System health (latency, throughput, errors)
- Model performance (accuracy, confidence distributions)

### 6.3 Setup Alerts
- High error rate (>5%)
- High latency (>150ms)
- FTMO guardrails hit
- Model accuracy degradation

**Success Criteria**:
- [x] Prometheus collecting metrics
- [x] Grafana dashboards accessible
- [x] Alerts configured
- [x] Test alert fires successfully

---

## STEP 7: Paper Trading Validation (2 weeks)

**Time**: 2 weeks
**Cost**: $0
**Status**: ⏳ PENDING

### 7.1 Start Paper Trading
```bash
# Enable execution engine (paper mode)
# Monitor for 2 weeks
# Track all metrics
```

### 7.2 Success Gates
- [ ] Sharpe ratio > 1.5
- [ ] Win rate > 55%
- [ ] Max drawdown < 15%
- [ ] No system errors
- [ ] FTMO guardrails never hit
- [ ] Average latency < 100ms

### 7.3 Daily Monitoring
- Check Grafana dashboards
- Review CloudWatch logs
- Verify RDS data integrity
- Check Alpaca positions

---

## STEP 8: Go Live (Week 4-5)

**Time**: 1 day
**Cost**: $0
**Status**: ⏳ PENDING (if paper trading successful)

### 8.1 Pre-Live Checklist
- [ ] Paper trading validated (2 weeks, all gates passed)
- [ ] Live Alpaca account funded
- [ ] Production API keys configured
- [ ] All monitoring operational
- [ ] Runbooks documented
- [ ] Team trained

### 8.2 Go Live Steps
```bash
# Update Secrets Manager with live API keys
# Set execution engine to live mode (paper=False)
# Start with 10% position sizes
# Monitor closely for 1 week
```

### 8.3 Post-Live Monitoring
- Monitor every trade manually for first week
- Increase position sizes gradually
- Watch for any anomalies

---

## Cost Summary

| Component | Monthly Cost | One-Time |
|-----------|--------------|----------|
| **GPU Training** | - | $0.61 |
| **RDS PostgreSQL** | $24 | - |
| **ElastiCache Redis** | $11 | - |
| **Secrets Manager** | $2.40 | - |
| **Redpanda Cloud** | $299 | - |
| **ECS Fargate (4 consumers)** | $50 | - |
| **Prometheus + Grafana** | $25 | - |
| **Data Transfer** | ~$10 | - |
| **TOTAL** | **$421.40/month** | **$0.61** |

---

## Timeline

| Week | Tasks | Hours | Status |
|------|-------|-------|--------|
| **Week 1** | GPU training, Phase 1 infra, Redpanda setup | 16 hours | ⏳ Starting |
| **Week 2** | Deploy consumers, Alpaca integration | 24 hours | ⏳ Pending |
| **Week 3** | Monitoring, testing, fixes | 16 hours | ⏳ Pending |
| **Week 4-5** | Paper trading validation | 8 hours | ⏳ Pending |
| **Week 6** | Go live (if validated) | 4 hours | ⏳ Pending |

**Total**: 6 weeks, 68 hours work

---

## Next Action: GPU Training

**Execute now**:
```bash
aws s3 ls s3://crpbot-ml-data-20251110/features/ --human-readable | grep latest
./scripts/setup_gpu_training.sh
```

**Set phone alarm for 10 minutes** to terminate instance!

Let's begin! 🚀
