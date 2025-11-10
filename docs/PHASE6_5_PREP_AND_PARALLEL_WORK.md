# Phase 6.5: Pre-Flight Checklist & Parallel Work Plan

**Status**: 🟡 NOT READY - Critical Items Missing
**Created**: 2025-11-09
**Target Start**: After pre-requisites complete

---

## ⚠️ CRITICAL: Phase 6.5 Cannot Start Yet

Based on review of current branch, the following **BLOCKING** items are missing:

### 🔴 Critical Missing Items

| Item | Status | Blocker Level | Estimated Time |
|------|--------|---------------|----------------|
| **Trained Models** | ❌ Missing | CRITICAL | 6-12 hours |
| **Training Data** | ❌ Missing | CRITICAL | 2-4 hours |
| **Feature Store** | ❌ Missing | CRITICAL | 1-2 hours |
| **Model Inference Module** | ❌ Missing | CRITICAL | 2-3 hours |
| **AWS CloudWatch** | ❌ Missing | HIGH | 1 hour (merge) |
| **Lambda EventBridge/SNS** | ❌ Missing | HIGH | 1 hour (merge) |
| **FTMO Demo Account** | ❌ Missing | MEDIUM | 30 mins |
| **Phase 6.5 Runbook** | ❌ Missing | MEDIUM | 10 mins (merge) |
| **Reports Directory** | ❌ Missing | LOW | 5 mins |

**Total Prep Time**: 13-23 hours of work

---

## 📋 Phase 6.5 Readiness Status

### Current State Assessment

#### ✅ What We Have (Ready)
- [x] Core trading system complete (Phases 1-6)
- [x] 24/24 tests passing
- [x] Runtime loop with mock predictions
- [x] FTMO rules enforced
- [x] Rate limiter working
- [x] Telegram bot implemented
- [x] Database schema ready
- [x] AWS Phase 1 (S3, RDS, Secrets Manager)
- [x] Enhanced confidence scoring
- [x] Auto-learning system

#### 🟡 What We're Missing (Partial)
- [ ] Real model predictions (using mocks)
- [ ] Lambda EventBridge schedule (partial - missing triggers)
- [ ] CloudWatch monitoring (not on current branch)
- [ ] Observation tooling (not on current branch)

#### ❌ What We Don't Have (Blocking)
- [ ] Trained LSTM models
- [ ] Trained Transformer models
- [ ] Historical training data (30 days OHLCV)
- [ ] Feature engineering pipeline executed
- [ ] Model inference module
- [ ] Promoted model checkpoints
- [ ] Backtest baseline snapshot
- [ ] Phase 6.5 directory structure

---

## 🎯 Strategy: Two-Track Approach

### Track 1: Quick Start (1-2 days) - Merge Infrastructure
**Goal**: Get monitoring and infrastructure in place

**Action**: Merge `aws/rds-setup` branch to get:
- ✅ Complete Lambda infrastructure
- ✅ CloudWatch dashboards and alarms
- ✅ Phase 6.5 runbook and tooling
- ✅ EventBridge schedules
- ✅ SNS notification topics

**Steps**:
```bash
git checkout claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih
git merge origin/aws/rds-setup --no-ff
# Resolve conflicts, test, push
```

**Time**: 2-4 hours (review + merge + test)

---

### Track 2: Model Training (Parallel) - Build Real Models
**Goal**: Replace mock predictions with real trained models

This work happens IN PARALLEL with infrastructure merge and observation period.

---

## 📊 Detailed Pre-Flight Checklist

### Part A: Infrastructure (Merge aws/rds-setup) ⏱️ 2-4 hours

- [ ] **Review aws/rds-setup changes**
  ```bash
  git diff claude/review-repo-issues-011CUshBtYfVHjA4Q6nBNaih origin/aws/rds-setup --stat
  ```

- [ ] **Merge branch**
  ```bash
  git merge origin/aws/rds-setup --no-ff
  ```

- [ ] **Resolve conflicts** (if any)
  - Prioritize aws/rds-setup for AWS infrastructure files
  - Keep current branch for core trading logic

- [ ] **Deploy missing CloudFormation stacks**
  ```bash
  # If not already deployed on AWS
  make deploy-cloudwatch-dashboards
  make deploy-cloudwatch-alarms
  make deploy-lambda-risk-monitor
  make deploy-lambda-telegram-bot
  ```

- [ ] **Verify CloudWatch dashboards**
  - Trading Metrics: `CRPBot-Trading-dev`
  - System Health: `CRPBot-System-dev`

- [ ] **Verify CloudWatch alarms**
  - 7 alarms should be in OK or INSUFFICIENT_DATA state

- [ ] **Test Lambda functions**
  ```bash
  aws lambda invoke --function-name crpbot-signal-processor-dev /tmp/response.json
  ```

**Status After Part A**: Infrastructure ready for observation ✅

---

### Part B: Data Collection ⏱️ 2-4 hours

- [ ] **Create data directories**
  ```bash
  mkdir -p data/raw data/features data/processed
  ```

- [ ] **Verify Coinbase API credentials**
  ```bash
  python scripts/check_credentials.py coinbase
  ```

- [ ] **Collect historical OHLCV data (30 days)**
  ```bash
  python apps/trainer/data_pipeline.py \
    --symbols BTC-USD,ETH-USD \
    --days 30 \
    --interval 5min \
    --output data/raw/
  ```
  **Time**: 1-2 hours (API rate limits)

- [ ] **Engineer features**
  ```bash
  python apps/trainer/features.py \
    --input data/raw/ \
    --output data/features/ \
    --save-parquet
  ```
  **Time**: 30 minutes

- [ ] **Validate feature quality**
  ```bash
  python scripts/validate_features.py \
    --features data/features/ \
    --check-leakage \
    --check-completeness
  ```
  **Time**: 15 minutes

**Status After Part B**: Training data ready ✅

---

### Part C: Model Training ⏱️ 6-12 hours

This is the MOST TIME-CONSUMING step and can run in parallel with observation.

#### C.1: Train LSTM Models (per coin)

- [ ] **Train LSTM for BTC-USD**
  ```bash
  python apps/trainer/train/train_lstm.py \
    --symbol BTC-USD \
    --features data/features/BTC-USD_features.parquet \
    --epochs 100 \
    --batch-size 64 \
    --hidden-size 128 \
    --num-layers 3 \
    --checkpoint models/checkpoints/lstm_BTC-USD_best.pt
  ```
  **Time**: 2-4 hours (GPU) or 6-8 hours (CPU)

- [ ] **Train LSTM for ETH-USD**
  ```bash
  python apps/trainer/train/train_lstm.py \
    --symbol ETH-USD \
    --features data/features/ETH-USD_features.parquet \
    --epochs 100 \
    --batch-size 64 \
    --checkpoint models/checkpoints/lstm_ETH-USD_best.pt
  ```
  **Time**: 2-4 hours (GPU) or 6-8 hours (CPU)

#### C.2: Train Transformer Models

- [ ] **Train Transformer (multi-coin)**
  ```bash
  python apps/trainer/train/train_transformer.py \
    --features data/features/ \
    --epochs 50 \
    --batch-size 32 \
    --model-dim 256 \
    --num-heads 8 \
    --checkpoint models/checkpoints/transformer_best.pt
  ```
  **Time**: 3-6 hours (GPU) or 10-15 hours (CPU)

#### C.3: Evaluate Models

- [ ] **Evaluate LSTM models**
  ```bash
  python scripts/evaluate_model.py \
    --model models/checkpoints/lstm_BTC-USD_best.pt \
    --symbol BTC-USD \
    --model-type lstm \
    --min-accuracy 0.68 \
    --max-calibration-error 0.05
  ```

- [ ] **Evaluate Transformer model**
  ```bash
  python scripts/evaluate_model.py \
    --model models/checkpoints/transformer_best.pt \
    --symbol BTC-USD,ETH-USD \
    --model-type transformer
  ```

#### C.4: Promote Models

- [ ] **Promote passing models**
  ```bash
  # Only if accuracy ≥68% and calibration error ≤5%
  python scripts/promote_model.py \
    --model models/checkpoints/lstm_BTC-USD_best.pt \
    --dest models/promoted/lstm_BTC-USD.pt

  python scripts/promote_model.py \
    --model models/checkpoints/transformer_best.pt \
    --dest models/promoted/transformer.pt
  ```

**Status After Part C**: Models trained and promoted ✅

---

### Part D: Model Inference Integration ⏱️ 2-3 hours

- [ ] **Create inference module**
  ```bash
  # Create libs/inference/predictor.py
  # Load promoted models
  # Implement predict() method
  # Return ensemble predictions
  ```

- [ ] **Update runtime to use real models**
  ```python
  # In apps/runtime/main.py
  # Replace generate_mock_signal() with:
  from libs.inference.predictor import ModelPredictor

  predictor = ModelPredictor(
      lstm_path="models/promoted/lstm_{symbol}.pt",
      transformer_path="models/promoted/transformer.pt"
  )

  predictions = predictor.predict(symbol, current_features)
  ```

- [ ] **Test inference locally**
  ```bash
  python scripts/test_inference.py \
    --symbol BTC-USD \
    --verify-latency \
    --max-latency-ms 500
  ```

- [ ] **Run integration test**
  ```bash
  pytest tests/integration/test_model_inference.py -v
  ```

**Status After Part D**: Real models integrated ✅

---

### Part E: Configuration & Setup ⏱️ 30 minutes

- [ ] **Update .env file**
  ```bash
  # Verify all required variables
  RUNTIME_MODE=dryrun
  KILL_SWITCH=false
  CONFIDENCE_THRESHOLD=0.75

  # FTMO demo credentials (get from FTMO website)
  FTMO_LOGIN=demo_account
  FTMO_PASS=demo_password
  FTMO_SERVER=FTMO-Demo

  # Telegram (if not already set)
  TELEGRAM_TOKEN=your_bot_token
  TELEGRAM_CHAT_ID=your_chat_id

  # AWS
  AWS_REGION=us-east-1
  # ... (rest from .env.aws)
  ```

- [ ] **Create reports directory**
  ```bash
  mkdir -p reports/phase6_5
  touch reports/phase6_5/day0.md
  touch reports/phase6_5/day1.md
  touch reports/phase6_5/day2.md
  touch reports/phase6_5/day3.md
  touch reports/phase6_5/day4.md
  touch reports/phase6_5/summary.md
  ```

- [ ] **Verify database**
  ```bash
  python scripts/init_database.py
  python scripts/test_database.py
  ```

- [ ] **Test Telegram bot**
  ```bash
  python scripts/test_telegram_bot.py
  ```

**Status After Part E**: Configuration complete ✅

---

### Part F: Final Validation ⏱️ 1 hour

- [ ] **Run smoke test with real models**
  ```bash
  pytest tests/smoke/test_backtest_smoke.py -v
  ```

- [ ] **Run runtime locally (5 minutes)**
  ```bash
  python apps/runtime/main.py --mode dryrun --iterations 3
  ```

- [ ] **Verify logs**
  ```bash
  tail -f logs/runtime_*.log
  # Should see:
  # - Real model predictions (not random mocks)
  # - Signal generation
  # - FTMO rule checks
  # - Telegram notifications
  ```

- [ ] **Check CloudWatch metrics**
  - Open `CRPBot-Trading-dev` dashboard
  - Verify data is flowing

- [ ] **Test alarm triggering**
  ```bash
  # Manually trigger error to test alarms
  python scripts/test_cloudwatch_alarm.py
  ```

**Status After Part F**: System validated, ready for observation ✅

---

## 🔄 Parallel Work During Phase 6.5 Observation

**While observation is running** (3-5 days), you can work on these in parallel:

### Track A: Production Infrastructure (2-3 days)

- [ ] **Set up VPS**
  - Provider: Hetzner or DigitalOcean
  - Specs: 4 vCPU, 8GB RAM, 160GB SSD
  - OS: Ubuntu 22.04 LTS
  - Cost: ~$25-40/month

- [ ] **Configure VPS**
  ```bash
  # SSH to VPS
  ssh root@your-vps-ip

  # Install dependencies
  apt update && apt upgrade -y
  apt install python3.11 python3-pip git uv -y

  # Clone repository
  git clone https://github.com/imnuman/crpbot.git
  cd crpbot

  # Install Python dependencies
  uv sync

  # Set up systemd service
  cp infra/systemd/crpbot-runtime.service /etc/systemd/system/
  systemctl enable crpbot-runtime
  ```

- [ ] **Configure firewall**
  ```bash
  ufw allow 22/tcp  # SSH
  ufw allow 8080/tcp  # Health check
  ufw enable
  ```

- [ ] **Set up log rotation**
  ```bash
  cp infra/logrotate/crpbot /etc/logrotate.d/
  ```

- [ ] **Test VPS deployment**
  ```bash
  systemctl start crpbot-runtime
  systemctl status crpbot-runtime
  curl http://localhost:8080/healthz
  ```

---

### Track B: FTMO Account Setup (1 day)

- [ ] **Purchase FTMO Challenge**
  - Go to: https://ftmo.com/
  - Choose: FTMO Challenge (not Express)
  - Account size: $10,000 (recommended for testing)
  - Trading platform: MetaTrader 5
  - Cost: ~$155 USD

- [ ] **Configure MT5**
  - Download MT5 from FTMO email
  - Install on Windows or VPS
  - Login with FTMO credentials
  - Verify connection

- [ ] **Update credentials**
  ```bash
  # In .env (production)
  FTMO_LOGIN=your_ftmo_account
  FTMO_PASS=your_ftmo_password
  FTMO_SERVER=FTMO-Server
  ```

- [ ] **Test FTMO connection**
  ```bash
  python scripts/test_ftmo_connection.py
  ```

---

### Track C: Model Refinement (Ongoing)

- [ ] **Analyze observation results**
  - Compare predicted vs actual win rates
  - Check calibration drift
  - Identify pattern failures

- [ ] **Retrain if needed**
  ```bash
  # If calibration error > 10%
  python apps/trainer/train/train_lstm.py --retrain
  python apps/trainer/train/train_transformer.py --retrain
  ```

- [ ] **A/B test models**
  - Deploy model v1.0 to Lambda
  - Train model v1.1 locally
  - Compare backtest results
  - Promote better model

---

### Track D: Lambda Deployment Prep (1-2 days)

- [ ] **Create Lambda deployment package**
  ```bash
  # Package models and dependencies
  bash infra/scripts/package_lambda.sh

  # Creates: lambda_deployment.zip (< 250MB)
  ```

- [ ] **Upload models to S3**
  ```bash
  aws s3 cp models/promoted/ s3://crpbot-backups-dev/models/promoted/ --recursive
  ```

- [ ] **Update Lambda function code**
  ```bash
  aws lambda update-function-code \
    --function-name crpbot-signal-processor-dev \
    --zip-file fileb://lambda_deployment.zip
  ```

- [ ] **Test Lambda with real models**
  ```bash
  aws lambda invoke \
    --function-name crpbot-signal-processor-dev \
    --payload '{"symbol":"BTC-USD"}' \
    /tmp/response.json

  cat /tmp/response.json | jq
  ```

---

### Track E: Documentation & Runbooks (1 day)

- [ ] **Production deployment runbook**
  - VPS setup steps
  - Environment configuration
  - Systemd service management
  - Rollback procedures

- [ ] **Incident response playbook**
  - FTMO limit breach response
  - Lambda timeout handling
  - Database connection failures
  - Coinbase API rate limits

- [ ] **Operational procedures**
  - Daily health checks
  - Weekly performance review
  - Monthly model retraining
  - Quarterly infrastructure review

---

## 🎯 Final Integration After Observation

**After successful 3-5 day observation**, here's the final integration:

### Step 1: Review Observation Results (2 hours)

- [ ] **Compile metrics**
  ```bash
  python scripts/compile_observation_report.py \
    --period "2025-11-10 to 2025-11-15" \
    --output reports/phase6_5/summary.md
  ```

- [ ] **Verify exit criteria**
  - [ ] ≥72h continuous runtime ✅
  - [ ] Zero crash loops ✅
  - [ ] FTMO guardrails enforced ✅
  - [ ] Telegram notifications delivered ✅
  - [ ] Win rate within ±5% of backtest ✅

- [ ] **Go/No-Go decision**
  - **GO**: Proceed to Phase 7 (Micro-Lot Testing)
  - **NO-GO**: Fix issues, repeat observation

---

### Step 2: Deploy to Production (if GO) (4 hours)

- [ ] **Update production configuration**
  ```bash
  # On VPS
  cd crpbot
  git pull origin main

  # Update .env
  RUNTIME_MODE=live  # CRITICAL: Switch to live mode
  KILL_SWITCH=false
  DB_URL=postgresql://...  # Use RDS, not SQLite
  ```

- [ ] **Migrate to production database**
  ```bash
  # Backup SQLite data
  python scripts/backup_sqlite.py --output backups/

  # Migrate to PostgreSQL
  python scripts/migrate_to_postgres.py \
    --from sqlite:///tradingai.db \
    --to $DB_URL
  ```

- [ ] **Deploy real models to Lambda**
  ```bash
  aws lambda update-function-code \
    --function-name crpbot-signal-processor-prod \
    --zip-file fileb://lambda_deployment.zip
  ```

- [ ] **Start production runtime**
  ```bash
  # On VPS
  systemctl restart crpbot-runtime
  systemctl status crpbot-runtime

  # Verify
  tail -f /var/log/crpbot/runtime.log
  ```

---

### Step 3: Phase 7 - Micro-Lot Testing (2-3 weeks)

- [ ] **Configure micro-lot sizes**
  ```python
  # In libs/constants.py
  POSITION_SIZE_BTC = 0.001  # ~$40 at $40k BTC
  POSITION_SIZE_ETH = 0.01   # ~$20 at $2k ETH
  ```

- [ ] **Monitor closely**
  - Check CloudWatch every 4 hours
  - Review Telegram notifications immediately
  - Track FTMO limits daily
  - Verify every trade execution

- [ ] **Collect 100+ trades**
  - Target: 100-200 trades over 2-3 weeks
  - Mix of BTC and ETH
  - Mix of high/medium/low tiers

- [ ] **Validate performance**
  ```bash
  python scripts/analyze_live_performance.py \
    --period "last 30 days" \
    --min-trades 100 \
    --min-winrate 0.68
  ```

---

### Step 4: Production Go-Live (if validated) (1 day)

- [ ] **Increase position sizes**
  ```python
  # Gradually increase to full size
  POSITION_SIZE_BTC = 0.01  # ~$400
  POSITION_SIZE_ETH = 0.1   # ~$200
  ```

- [ ] **Enable full automation**
  - Remove manual approval requirements
  - Enable auto-execution to MT5
  - Activate all Telegram alerts

- [ ] **Set up monitoring**
  - 24/7 CloudWatch alarms
  - SMS alerts for critical issues
  - Daily performance emails

- [ ] **Document handoff**
  - Operational runbook
  - Emergency contacts
  - Rollback procedures

---

## 📊 Summary Timeline

| Phase | Duration | Can Run in Parallel? | Blocking? |
|-------|----------|---------------------|-----------|
| **Merge aws/rds-setup** | 2-4 hours | No | YES - Needed for monitoring |
| **Collect training data** | 2-4 hours | No | YES - Needed for models |
| **Train models** | 6-12 hours | Yes (can run overnight) | YES - Critical for real predictions |
| **Create inference module** | 2-3 hours | No | YES - Needed to use models |
| **Configuration** | 30 mins | No | YES - Needed to run |
| **Final validation** | 1 hour | No | YES - Verify before observation |
| **Total Prep Time** | **13-23 hours** | - | - |
| | | | |
| **Phase 6.5 Observation** | 3-5 days | - | - |
| **VPS setup** (parallel) | 2-3 days | Yes | No - for Phase 7 |
| **FTMO account** (parallel) | 1 day | Yes | No - for Phase 7 |
| **Lambda deployment** (parallel) | 1-2 days | Yes | No - optimization |
| | | | |
| **Observation review** | 2 hours | No | YES - Go/No-Go |
| **Production deployment** | 4 hours | No | YES - If GO |
| **Phase 7 micro-lot** | 2-3 weeks | No | YES - Final validation |
| **Production go-live** | 1 day | No | YES - Final step |

---

## ✅ Recommendation

### Option 1: Full Preparation (Recommended)
**Timeline**: 2-3 days prep + 3-5 days observation + 2-3 weeks testing = **4-5 weeks total**

1. Day 1-2: Merge infrastructure + collect data
2. Day 2-3: Train models (can run overnight)
3. Day 3: Create inference module + validate
4. Day 4-8: Phase 6.5 observation (with parallel VPS/FTMO setup)
5. Week 2-4: Phase 7 micro-lot testing
6. Week 5: Production go-live

### Option 2: Partial Start (Faster but riskier)
**Timeline**: 1 day prep + 3-5 days observation = **4-6 days to start**

1. Merge aws/rds-setup (monitoring in place)
2. Start observation with mock predictions
3. Train models DURING observation (parallel)
4. Integrate models AFTER observation
5. Phase 7 with real models

**Risk**: Observation data less meaningful with mock predictions

---

## 🎯 Next Immediate Actions

1. **Decide on approach** (Option 1 recommended)
2. **Merge aws/rds-setup branch** (required for both options)
3. **Start data collection** (if Option 1)
4. **Start model training** (if Option 1, can run overnight)
5. **Set up observation environment**

---

**Document Created By**: Claude Code
**Date**: 2025-11-09
**Status**: Ready for execution
**Next Update**: After infrastructure merge
