# Production Readiness Checklist

**Last Updated**: 2025-11-11
**Current Status**: 60% Ready (Infrastructure scripts ready, need deployment + model training)
**Estimated Time to Production**: 2-4 weeks

---

## ✅ What We Have (COMPLETE)

### Infrastructure Scripts
- [x] S3 storage setup and configured ($2.50/month)
- [x] S3 data uploaded (765MB: raw + features + models)
- [x] RDS deployment script (t4g.small PostgreSQL)
- [x] Redis deployment script (t4g.micro)
- [x] Secrets Manager setup script
- [x] GPU training scripts (p3.8xlarge)
- [x] Multi-GPU parallel training support

### Application Code
- [x] Kafka Phase 2 implementation (5 components):
  - [x] Market data ingester
  - [x] Feature engineering stream
  - [x] Model inference consumer
  - [x] Signal aggregator
  - [x] Execution engine
- [x] LSTM model architecture
- [x] Transformer model architecture
- [x] Multi-TF feature engineering (58 features)
- [x] FTMO guardrail logic
- [x] Position sizing with Kelly criterion

### Documentation
- [x] Complete architecture docs (12 files)
- [x] Amazon Q task instructions
- [x] Deployment guides
- [x] Cost analysis
- [x] Migration plan (7 phases)

### Current Data
- [x] 2 years historical OHLCV (BTC, ETH, SOL)
- [x] Multi-timeframe features (1m, 5m, 15m, 1h)
- [x] 58 engineered features

---

## ❌ What's Missing (GAPS)

### CRITICAL (P0 - Cannot Go Live Without)

#### 1. Trained Production Models ⚠️ HIGH PRIORITY
**Status**: ⏳ Training in progress (CPU, 50+ hours remaining)
**What's Needed**:
- [ ] Train all 3 models to completion
- [ ] Models meet accuracy gate (≥68%)
- [ ] Models meet calibration gate (<5% error)
- [ ] Backtesting shows Sharpe > 1.5
- [ ] Models uploaded to S3
- [ ] Model metadata stored

**Action**: Run GPU training NOW
```bash
pkill -f "train.*lstm"  # Stop slow CPU training
./scripts/setup_gpu_training.sh  # 10 min, $0.61
```

**Time**: 10 minutes
**Cost**: $0.61
**Blocker**: Yes - no inference without trained models

---

#### 2. Production Kafka/Redpanda Cluster ⚠️ HIGH PRIORITY
**Status**: ❌ Using Docker Compose (single node, dev only)
**What's Needed**:
- [ ] Deploy Redpanda Cloud (managed) OR
- [ ] Deploy Redpanda cluster on AWS (3 nodes)
- [ ] Configure replication (factor=3)
- [ ] Setup monitoring (Prometheus exporter)
- [ ] Migrate topics from dev Kafka
- [ ] Test failover scenarios

**Action**: Deploy Redpanda Cloud (easiest)
```bash
# Option A: Managed Redpanda Cloud
# Sign up at https://redpanda.com/try-redpanda
# Create cluster: 3 nodes, AWS us-east-1
# Cost: $299/month

# Option B: Self-hosted (use Terraform)
# Coming soon: ./scripts/infrastructure/deploy_redpanda.sh
```

**Time**: 4-8 hours (cloud) or 2 weeks (self-hosted)
**Cost**: $299/month (cloud) or $350/month (self-hosted)
**Blocker**: Yes - cannot process real-time data without production Kafka

---

#### 3. Deployed Kafka Consumers ⚠️ HIGH PRIORITY
**Status**: ❌ Code exists, not deployed
**What's Needed**:
- [ ] Deploy feature engineering stream (Docker/ECS)
- [ ] Deploy model inference consumers (Docker/ECS)
- [ ] Deploy signal aggregator (Docker/ECS)
- [ ] Deploy execution engine (Docker/ECS)
- [ ] Configure auto-restart on failure
- [ ] Setup health checks

**Action**: Deploy on ECS Fargate
```bash
# Coming soon: ./scripts/deploy_consumers.sh
# Uses Docker images, deploys to ECS Fargate
# ~$50/month for 4 consumers (CPU)
```

**Time**: 1-2 days
**Cost**: ~$50/month (ECS Fargate)
**Blocker**: Yes - no real-time trading without consumers

---

#### 4. Broker Integration (Alpaca/MT5) ⚠️ CRITICAL
**Status**: ❌ Execution engine has dry-run mode only
**What's Needed**:
- [ ] Choose broker (Alpaca recommended for crypto)
- [ ] Setup Alpaca account + API keys
- [ ] Implement Alpaca order execution
- [ ] Test order placement (paper trading)
- [ ] Implement order status polling
- [ ] Handle order errors/rejections
- [ ] Store execution results in RDS

**Action**: Integrate Alpaca API
```python
# apps/kafka/consumers/execution_engine.py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest

client = TradingClient(api_key, secret_key, paper=True)

order = MarketOrderRequest(
    symbol=symbol,
    qty=quantity,
    side=OrderSide.BUY,
    type=OrderType.MARKET,
    time_in_force=TimeInForce.GTC
)
client.submit_order(order)
```

**Time**: 2-3 days
**Cost**: $0 (paper trading)
**Blocker**: YES - cannot execute trades without broker

---

#### 5. Phase 1 Infrastructure Deployment
**Status**: ✅ Scripts ready, ❌ not deployed
**What's Needed**:
- [ ] Deploy RDS PostgreSQL (15 min)
- [ ] Create database schema (2 min)
- [ ] Deploy Redis cache (7 min)
- [ ] Setup Secrets Manager (2 min)
- [ ] Test all connections
- [ ] Update app configs to use RDS/Redis

**Action**: Run deployment scripts
```bash
./scripts/infrastructure/deploy_rds.sh
source .rds_connection_info
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
  -f scripts/infrastructure/create_db_schema.sql

./scripts/infrastructure/deploy_redis.sh
./scripts/infrastructure/setup_secrets.sh
```

**Time**: 1 hour
**Cost**: $37/month (RDS + Redis + Secrets)
**Blocker**: Medium - can work without, but limits observability

---

### IMPORTANT (P1 - Needed for Reliable Production)

#### 6. Model Serving Infrastructure
**Status**: ❌ Using in-process inference
**What's Needed**:
- [ ] Convert models to ONNX/TorchScript
- [ ] Deploy Triton Inference Server (ECS)
- [ ] Create model repository (S3)
- [ ] Update inference consumer to use Triton
- [ ] Implement batching for throughput
- [ ] Monitor inference latency

**Action**: Deploy Triton on ECS
**Time**: 3-5 days
**Cost**: $50/month (CPU) or $350/month (GPU)
**Blocker**: No - can use in-process inference initially

---

#### 7. Observability Stack (Prometheus + Grafana)
**Status**: ❌ Using basic logging only
**What's Needed**:
- [ ] Deploy Prometheus (EC2 t4g.small)
- [ ] Deploy Grafana (same instance)
- [ ] Configure metrics exporters
- [ ] Create dashboards:
  - Trading metrics (P&L, win rate, Sharpe)
  - System health (latency, throughput, errors)
  - Model performance (accuracy, confidence)
- [ ] Setup CloudWatch Logs integration
- [ ] Configure alerts (PagerDuty)

**Action**: Deploy monitoring stack
**Time**: 1-2 days
**Cost**: $25/month (self-hosted) or $50/month (managed)
**Blocker**: No - but critical for debugging issues

---

#### 8. Real-Time Data Ingestion at Scale
**Status**: ❌ Using manual fetching only
**What's Needed**:
- [ ] Deploy market data ingester (ECS)
- [ ] Connect to Coinbase WebSocket (real-time)
- [ ] Implement reconnection logic
- [ ] Handle WebSocket errors/timeouts
- [ ] Backfill missed data
- [ ] Monitor data quality

**Action**: Deploy ingester
**Time**: 1 day
**Cost**: $10/month (ECS Fargate)
**Blocker**: Medium - need real-time data for live trading

---

### NICE TO HAVE (P2 - Optimize Later)

#### 9. Workflow Orchestration (Prefect)
**Status**: ❌ Manual training/deployment
**What's Needed**:
- [ ] Deploy Prefect server
- [ ] Create training workflow
- [ ] Create model promotion workflow
- [ ] Schedule weekly retraining
- [ ] Implement model gates (68% accuracy)
- [ ] Automatic deployment to Triton

**Time**: 2-3 days
**Cost**: $0 (self-hosted)
**Priority**: Low - can train manually initially

---

#### 10. Additional Data Sources
**Status**: ❌ Only 2 years OHLCV
**What's Needed**:
- [ ] Fetch 7 years historical data (3-4 hours)
- [ ] Setup Reddit API for sentiment
- [ ] Fetch Reddit historical (6 months)
- [ ] Setup CryptoCompare API
- [ ] Setup CoinGecko API
- [ ] Engineer sentiment features (+15 features)

**Time**: 1-2 days (setup) + 2-4 hours (data fetch)
**Cost**: $0 (all free tier)
**Priority**: Medium - improves model accuracy

---

#### 11. CI/CD Pipeline
**Status**: ❌ Manual deployments
**What's Needed**:
- [ ] GitHub Actions workflows
- [ ] Automated testing
- [ ] Docker image building
- [ ] Blue/green deployments
- [ ] Rollback procedures

**Time**: 3-5 days
**Cost**: $0 (GitHub Actions free tier)
**Priority**: Low - can deploy manually initially

---

#### 12. Multi-AZ Redundancy
**Status**: ❌ Single-AZ infrastructure
**What's Needed**:
- [ ] RDS: Single-AZ → Multi-AZ
- [ ] Redis: Single-node → With replica
- [ ] Consumers: 1 instance → 2+ instances
- [ ] Load balancer (if needed)

**Time**: 1 day
**Cost**: +$50/month (2x for RDS + Redis)
**Priority**: Low - not needed for paper trading

---

## 📊 Gap Summary by Priority

### Must Have Before Going Live (P0)
```
1. ⏳ Trained models (10 min, $0.61)           - IN PROGRESS
2. ❌ Production Kafka cluster (4-8 hours, $299/month) - NOT STARTED
3. ❌ Deployed Kafka consumers (1-2 days, $50/month) - NOT STARTED
4. ❌ Broker integration (2-3 days, $0)       - NOT STARTED
5. ❌ Phase 1 infrastructure (1 hour, $37/month) - SCRIPTS READY
```

**Total P0**: ~1 week work, $387/month

### Important for Reliability (P1)
```
6. ❌ Model serving (Triton) (3-5 days, $50-350/month)
7. ❌ Observability (Prometheus + Grafana) (1-2 days, $25/month)
8. ❌ Real-time data ingestion (1 day, $10/month)
```

**Total P1**: ~1 week work, +$85-385/month

### Optimize Later (P2)
```
9. ❌ Workflow orchestration (Prefect) (2-3 days, $0)
10. ❌ Additional data (7yr + sentiment) (1-2 days, $0)
11. ❌ CI/CD pipeline (3-5 days, $0)
12. ❌ Multi-AZ redundancy (1 day, +$50/month)
```

**Total P2**: ~1-2 weeks work, +$50/month

---

## 🚀 Fastest Path to Production

### Phase 0: Immediate (TODAY - 10 minutes, $0.61)
```bash
# Get trained models ASAP
pkill -f "train.*lstm"
./scripts/setup_gpu_training.sh
# Wait 10 min, terminate instance
```

### Phase 1: Foundation (WEEK 1 - 8 hours, $387/month)
```bash
# Day 1: Infrastructure (1 hour)
./scripts/infrastructure/deploy_rds.sh
./scripts/infrastructure/deploy_redis.sh
./scripts/infrastructure/setup_secrets.sh

# Day 2-3: Redpanda Cloud setup (4 hours)
# Sign up, create cluster, migrate topics

# Day 4-5: Deploy consumers to ECS (1-2 days)
# Build Docker images, deploy to Fargate
```

### Phase 2: Broker Integration (WEEK 2 - 3 days, $0)
```bash
# Day 1: Setup Alpaca account
# Day 2: Implement order execution
# Day 3: Paper trading tests
```

### Phase 3: Observability (WEEK 2 - 2 days, $25/month)
```bash
# Day 4-5: Deploy Prometheus + Grafana
# Create dashboards, configure alerts
```

### Phase 4: Paper Trading (WEEK 3-4 - 2 weeks, $0)
```bash
# Run system with paper trading
# Monitor performance, fix bugs
# Validate Sharpe > 1.5, win rate > 55%
```

### Phase 5: Go Live (WEEK 5, $0)
```bash
# If paper trading successful:
# - Switch from paper to live trading
# - Start with small position sizes
# - Monitor closely for 1 week
```

---

## 💰 Total Cost to Production

### Minimum Viable Production (P0 only)
```
RDS PostgreSQL (t4g.small):        $24/month
Redis (t4g.micro):                 $11/month
Secrets Manager (6 secrets):       $2.40/month
Redpanda Cloud (managed):          $299/month
ECS Fargate (4 consumers):         $50/month
GPU training (one-time):           $0.61
────────────────────────────────────────────
TOTAL:                             $387/month + $0.61 one-time
```

### Recommended Production (P0 + P1)
```
Minimum viable (above):            $387/month
Prometheus + Grafana:              $25/month
Real-time data ingester:           $10/month
Triton Inference (CPU):            $50/month
────────────────────────────────────────────
TOTAL:                             $472/month
```

### Full Production (P0 + P1 + P2)
```
Recommended (above):               $472/month
Multi-AZ redundancy:               $50/month
Triton GPU (if needed):            +$300/month
────────────────────────────────────────────
TOTAL:                             $522-822/month
```

---

## ⏱️ Timeline to Production

### Conservative Estimate (1 person, part-time)
```
Week 1: Infrastructure + GPU training       (8 hours)
Week 2: Redpanda + Consumers deployment     (16 hours)
Week 3: Broker integration + Observability  (16 hours)
Week 4-5: Paper trading + debugging         (20 hours)
Week 6: Go live (if validated)              (4 hours)
────────────────────────────────────────────
TOTAL:                                      6 weeks, 64 hours
```

### Aggressive Estimate (1 person, full-time)
```
Days 1-2: Infrastructure + GPU + Redpanda   (16 hours)
Days 3-4: Deploy consumers                  (16 hours)
Days 5-6: Broker integration                (16 hours)
Days 7-8: Observability                     (16 hours)
Week 2-3: Paper trading                     (2 weeks)
Week 4: Go live                             (1 week)
────────────────────────────────────────────
TOTAL:                                      4 weeks, 64 hours
```

---

## ✅ Recommended Action Plan

### Start TODAY
1. **GPU Training** (10 min, $0.61)
   ```bash
   ./scripts/setup_gpu_training.sh
   ```

2. **Evaluate Models** (1 hour, $0)
   - Check accuracy > 68%
   - Check calibration < 5%
   - Backtest Sharpe > 1.5
   - If fails, need more data/features

### This Week
3. **Deploy Phase 1 Infrastructure** (1 hour, $37/month)
   ```bash
   ./scripts/infrastructure/deploy_rds.sh
   ./scripts/infrastructure/deploy_redis.sh
   ./scripts/infrastructure/setup_secrets.sh
   ```

4. **Setup Redpanda Cloud** (4 hours, $299/month)
   - Create account
   - Deploy cluster
   - Migrate topics

### Next Week
5. **Deploy Consumers** (2 days, $50/month)
6. **Integrate Broker** (3 days, $0)
7. **Setup Monitoring** (2 days, $25/month)

### Week 3-4
8. **Paper Trading** (2 weeks, $0)
9. **Validate Performance**
10. **Go Live** (if successful)

---

## 🎯 Bottom Line: What's Missing?

**To run MINIMAL production (paper trading)**:
1. ⏳ Trained models (10 min) - IN PROGRESS
2. ❌ Redpanda cluster (4-8 hours)
3. ❌ Deployed consumers (1-2 days)
4. ❌ Broker integration (2-3 days)
5. ❌ Phase 1 infra (1 hour)

**Time**: ~1 week
**Cost**: $387/month + $0.61 one-time
**Effort**: ~40 hours work

**To run RELIABLE production (live trading)**:
- Add observability (+$25/month, +2 days)
- Add real-time ingestion (+$10/month, +1 day)
- 2 weeks paper trading validation

**Total Time**: 4 weeks
**Total Cost**: $422/month
**Total Effort**: ~64 hours

**Current blockers**: Trained models (resolving now with GPU), Redpanda cluster, broker integration.

**Can start today**: GPU training → Phase 1 infra → Redpanda → Consumers → Broker → Paper trade → Live!
