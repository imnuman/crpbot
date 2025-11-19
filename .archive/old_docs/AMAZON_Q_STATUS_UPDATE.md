# CryptoBot: Amazon Q Status Update

**Date**: 2025-11-11
**Time**: 23:40 UTC

---

## Current Production Kickoff Status

### ❌ BLOCKED: GPU Training
**Issue**: AWS account vCPU limit = 0 for GPU instances
**Attempts Made**:
- Tried p3.8xlarge (4x V100): InsufficientInstanceCapacity
- Tried p3.2xlarge (1x V100): Unsupported configuration
- Tried g4dn.xlarge (1x T4): VcpuLimitExceeded

**Required Action** (NEEDS ACCOUNT ADMIN):
```
1. Visit: https://console.aws.amazon.com/servicequotas/
2. Request limit increase for:
   - "Running On-Demand G and VT instances" → 32 vCPUs
   - "Running On-Demand P instances" → 32 vCPUs
3. Expected approval: 1-2 business days
```

**IAM Permission Issue**:
User `ncldev` lacks `servicequotas:RequestServiceQuotaIncrease` permission

**Alternative**: Continue with CPU training (50+ hours remaining)

---

### ⏳ IN PROGRESS: RDS PostgreSQL Deployment

**Status**: Deploying (2nd attempt)
**Issue Fixed**: PostgreSQL version 15.4 → 16.10 (15.4 not available)
**Stack**: crpbot-rds-postgres
**Configuration**:
- Instance: db.t4g.small (ARM Graviton, 2vCPU, 2GB RAM)
- Engine: PostgreSQL 16.10
- Storage: 100GB GP3 SSD
- Region: us-east-1
- Multi-AZ: false (single-AZ for cost savings)
- Encryption: enabled
- Backups: 7-day retention

**Cost**: $24/month

**ETA**: 10-15 minutes
**Log**: /tmp/rds_deployment_v2.log

---

## Completed Tasks

### ✅ S3 Storage (DONE)
- Bucket: s3://crpbot-ml-data-20251110
- Data uploaded: 765MB
  - Raw data: BTC, ETH, SOL (2 years, 1m interval)
  - Features: 58 columns multi-TF (1m, 5m, 15m, 1h)
  - Models: Untrained architectures
- Cost: $2.50/month

### ✅ Infrastructure Scripts (DONE)
All scripts created and ready:
```
scripts/infrastructure/
├── deploy_rds.sh              # PostgreSQL (fixed to v16.10)
├── create_db_schema.sql       # 13 tables, 3 schemas
├── deploy_redis.sh            # ElastiCache Redis
└── setup_secrets.sh           # AWS Secrets Manager

scripts/
├── setup_s3_storage.sh        # DONE
├── upload_to_s3.sh            # DONE
├── setup_gpu_training.sh      # Ready (blocked by limits)
└── train_multi_gpu.sh         # Ready (blocked by limits)
```

### ✅ Documentation (DONE)
All guides created:
```
PRODUCTION_KICKOFF.md              # Master deployment plan
QUICKSTART_GPU_TRAINING.md         # GPU training guide
README_NEXT_STEPS.md               # Navigation hub
docs/AMAZON_Q_TASK_INSTRUCTIONS.md # Task-by-task instructions
docs/PHASE1_DEPLOYMENT_GUIDE.md    # Infrastructure deployment
docs/PRODUCTION_READINESS_CHECKLIST.md  # Gap analysis
```

### ✅ Kafka Phase 2 (DONE)
5 components implemented:
- Market data ingester
- Feature engineering stream
- Model inference consumer
- Signal aggregator
- Execution engine (dry-run mode)

---

## Pending Tasks

### Phase 1 Infrastructure (IN PROGRESS)

**Task 1**: ⏳ RDS PostgreSQL (deploying now)
- ETA: 10-15 min
- Next: Create database schema (2 min)

**Task 2**: ❌ ElastiCache Redis
- Command: `./scripts/infrastructure/deploy_redis.sh`
- Time: 7 minutes
- Cost: $11/month

**Task 3**: ❌ AWS Secrets Manager
- Command: `./scripts/infrastructure/setup_secrets.sh`
- Time: 2 minutes
- Cost: $2.40/month (6 secrets)

**Total Phase 1 Cost**: $37.40/month

---

### GPU Training (BLOCKED - Waiting for Limits)

Once vCPU limits approved:
```bash
# Stop CPU training
pkill -f "train.*lstm"

# Launch GPU training (g4dn.xlarge recommended)
./scripts/setup_gpu_training.sh
# Will prompt for instance type selection

# Expected:
# - Duration: 10-15 minutes (T4 GPU)
# - Cost: ~$0.09 for 15 minutes
# - Output: 3 trained models (BTC, ETH, SOL)
```

**Alternative**: Use CPU-trained models (ready in 50+ hours)

---

### Phase 2: Redpanda Cloud (NOT STARTED)

**Status**: Waiting for Phase 1 completion
**Action**:
1. Sign up: https://redpanda.com/try-redpanda
2. Create cluster: 3 nodes, AWS us-east-1
3. Create 15 Kafka topics (5 categories)
4. Update consumer configs with Redpanda broker URLs

**Cost**: $299/month
**Time**: 4-8 hours

---

### Phase 3: Deploy Kafka Consumers (NOT STARTED)

**Status**: Waiting for Redpanda
**Components to Deploy**:
1. Feature engineering stream (ECS Fargate)
2. Model inference consumers (ECS Fargate)
3. Signal aggregator (ECS Fargate)
4. Execution engine (ECS Fargate)

**Requirements**:
- Docker images built
- ECS task definitions created
- Load balancer (if needed)
- Health checks configured

**Cost**: ~$50/month
**Time**: 1-2 days

---

### Phase 4: Alpaca Broker Integration (NOT STARTED)

**Status**: Code skeleton exists, needs implementation
**File**: apps/kafka/consumers/execution_engine.py
**Action**: Implement Alpaca API integration

```python
# Current: Dry-run mode (logs signals, no execution)
# Needed: Alpaca paper trading integration

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest

client = TradingClient(api_key, secret_key, paper=True)
# ... implement order execution ...
```

**Cost**: $0 (paper trading)
**Time**: 2-3 days

---

### Phase 5: Monitoring Stack (NOT STARTED)

**Status**: Optional for initial testing
**Components**:
- Prometheus (metrics)
- Grafana (dashboards)
- CloudWatch Logs (centralized logging)

**Cost**: $25/month (self-hosted)
**Time**: 1-2 days

---

### Phase 6: Paper Trading Validation (NOT STARTED)

**Duration**: 2 weeks minimum
**Goals**:
- Sharpe ratio > 1.5
- Win rate > 55%
- Max drawdown < 10%
- FTMO guardrails working

**Cost**: $0 (paper trading)

---

## Updated Timeline

### Conservative (1 person, part-time)
```
Week 1: Phase 1 infrastructure + GPU training    (8 hours) - IN PROGRESS
Week 2: Redpanda + Consumers deployment          (16 hours)
Week 3: Broker integration + Observability       (16 hours)
Week 4-5: Paper trading + debugging              (20 hours)
Week 6: Go live (if validated)                   (4 hours)
```

**Total**: 6 weeks, 64 hours

### Critical Path Dependencies
```
1. GPU vCPU limits approved (1-2 business days) ← BLOCKER
2. Phase 1 infrastructure complete (today)       ← IN PROGRESS
3. Redpanda cluster deployed (4-8 hours)
4. Consumers deployed (1-2 days)
5. Broker integrated (2-3 days)
6. Paper trading validated (2 weeks)
7. Go live
```

---

## Cost Summary

### Current Monthly Costs
```
S3 storage (765MB):              $2.50/month
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL CURRENT:                   $2.50/month
```

### After Phase 1 (Today)
```
S3 storage:                      $2.50/month
RDS PostgreSQL (t4g.small):      $24.00/month
ElastiCache Redis (t4g.micro):   $11.00/month
Secrets Manager (6 secrets):     $2.40/month
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL PHASE 1:                   $39.90/month
```

### Full MVP Production
```
Phase 1 (above):                 $39.90/month
Redpanda Cloud (managed):        $299.00/month
ECS Fargate (4 consumers):       $50.00/month
Prometheus + Grafana:            $25.00/month
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL MVP:                       $413.90/month
```

### One-Time Costs
```
GPU training (per run):          ~$0.10-1.00
```

---

## Amazon Q Action Items

### IMMEDIATE (Today)
1. **Monitor RDS deployment**
   ```bash
   # Check status:
   tail -f /tmp/rds_deployment_v2.log

   # Verify completion:
   aws cloudformation describe-stacks \
     --stack-name crpbot-rds-postgres \
     --query 'Stacks[0].StackStatus'
   ```

2. **Create database schema** (once RDS ready)
   ```bash
   source .rds_connection_info
   psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
     -f scripts/infrastructure/create_db_schema.sql
   ```

3. **Deploy Redis**
   ```bash
   ./scripts/infrastructure/deploy_redis.sh
   ```

4. **Setup Secrets Manager**
   ```bash
   ./scripts/infrastructure/setup_secrets.sh
   ```

### HIGH PRIORITY (Needs Admin)
5. **Request GPU vCPU limits**
   - Requires AWS Console access
   - Account admin must submit request
   - 1-2 business days for approval

### NEXT WEEK (After Phase 1)
6. Setup Redpanda Cloud
7. Deploy Kafka consumers to ECS
8. Integrate Alpaca broker
9. Deploy monitoring stack

---

## Known Issues & Blockers

### Issue 1: GPU vCPU Limit (BLOCKER)
- **Impact**: Cannot train models quickly
- **Workaround**: CPU training (50+ hours)
- **Resolution**: Admin request limit increase

### Issue 2: PostgreSQL Version Mismatch (FIXED)
- **Error**: Version 15.4 not available
- **Fix**: Updated to 16.10
- **Status**: Redeploying now

### Issue 3: IAM Permissions
- **User**: ncldev lacks ServiceQuotas permissions
- **Impact**: Cannot request limit increases via CLI
- **Workaround**: Use AWS Console

---

## Success Criteria

### Phase 1 Complete When:
- ✅ RDS PostgreSQL deployed and accessible
- ✅ Database schema created (13 tables)
- ✅ Redis cache deployed
- ✅ Secrets Manager configured
- ✅ All services verified with test connections

### Ready for Production When:
- ✅ Trained models with ≥68% accuracy
- ✅ Redpanda cluster operational
- ✅ All consumers deployed and processing
- ✅ Broker integration tested (paper trading)
- ✅ 2 weeks successful paper trading
- ✅ Sharpe > 1.5, win rate > 55%

---

## Questions for User

1. **GPU Limits**: Can you request the vCPU limit increase via AWS Console?
   - Need 32 vCPUs for G/VT instances (g4dn)
   - Need 32 vCPUs for P instances (p3)

2. **Production Timeline**: Are we proceeding with full deployment while waiting for GPU access?
   - Phase 1 infrastructure (today): $40/month
   - Redpanda (next week): +$299/month
   - Consumers (next week): +$50/month

3. **Alternative Training**: Should we continue CPU training as backup?
   - CPU: 50+ hours remaining, $0 cost
   - GPU: 10-15 minutes, $0.10 cost (when available)

---

## Next Steps (In Order)

1. ⏳ Wait for RDS deployment (10-15 min)
2. ✅ Create database schema
3. ✅ Deploy Redis
4. ✅ Setup Secrets Manager
5. ⏸️ Request GPU limits (admin task)
6. ⏸️ GPU training (when limits approved)
7. ➡️ Redpanda Cloud setup
8. ➡️ Deploy consumers
9. ➡️ Integrate Alpaca
10. ➡️ Paper trading validation

---

**Current Bottleneck**: GPU vCPU limits (needs admin intervention)
**Parallel Work**: Phase 1 infrastructure deployment (in progress)
**ETA to Unblocked**: 1-2 business days (GPU limits) + 30 minutes (Phase 1)
