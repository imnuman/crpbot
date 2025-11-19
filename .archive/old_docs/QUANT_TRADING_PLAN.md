# 🎯 Quant Trading Software - Production Plan

**Goal**: Build professional quant trading software with quality data pipeline
**Budget**: $500/month data budget
**Timeline**: 8 weeks to production
**Approach**: Start small, scale step by step

---

## 🎯 Core Objectives

1. ✅ **Fix current model** to be "usable" (≥65% accuracy)
2. ✅ **Connect paid data pipeline** (Tardis.dev Premium - $499/month)
3. ✅ **Build solid software** that processes all data accurately
4. ✅ **Scale step by step** as we prove success

---

## 📋 4-Phase Development Plan

### **PHASE 1: Foundation & Data Pipeline** (Weeks 1-2)

**Goal**: Subscribe to quality data + build robust ingestion pipeline

#### Week 1: Data Infrastructure

**Day 1-2: Subscribe & Setup**
```bash
Tasks:
1. Subscribe to Tardis.dev Premium ($499/month)
   - Sign up at https://tardis.dev/pricing
   - Get API credentials
   - Setup billing

2. Install Tardis.dev SDK
   cd /home/numan/crpbot
   uv add tardis-dev

3. Test connection
   uv run python scripts/test_tardis_connection.py
```

**Day 3-5: Data Ingestion Pipeline**
```
Create: libs/data/tardis_client.py
- Connect to Tardis.dev API
- Download historical tick data (2 years)
- Handle rate limits, retries
- Validate data quality
- Store in parquet format

Create: scripts/download_tardis_data.py
- Download BTC-USD, ETH-USD, SOL-USD
- Tick data + order book depth
- Progress tracking
- Error handling
```

**Day 6-7: Data Validation**
```
Create: scripts/validate_tardis_data.py
- Check completeness (no gaps)
- Verify tick counts
- Compare to free Coinbase data
- Generate quality report
```

**Deliverables**:
- ✅ Tardis.dev subscription active
- ✅ 2 years tick data downloaded for BTC/ETH/SOL
- ✅ Data quality validation report
- ✅ Ingestion pipeline tested

---

#### Week 2: Feature Engineering V2

**Day 8-10: Microstructure Features**
```
Create: libs/features/microstructure.py
- Order book imbalance
- Trade direction (buy vs sell pressure)
- Volume-weighted average price (VWAP)
- Bid-ask spread dynamics
- Price impact
- Order flow imbalance

New feature set:
- Original 33 features (from free data)
- +20 microstructure features (from tick data)
- Total: 53 features
```

**Day 11-12: Feature Pipeline**
```
Create: scripts/engineer_features_v2.py
- Process tick data → 1-minute aggregates
- Compute all 53 features
- Validate feature quality
- Save to parquet

Run for all symbols:
./scripts/batch_engineer_features_v2.sh
```

**Day 13-14: Feature Analysis**
```
Create: scripts/analyze_features_v2.py
- Compare old features vs new features
- Check correlations with target
- Feature importance baseline
- Expected: Higher correlation than free data
```

**Deliverables**:
- ✅ 53-feature dataset for BTC/ETH/SOL
- ✅ Microstructure features validated
- ✅ Feature analysis report
- ✅ Ready for model training

---

### **PHASE 2: Model Training & Validation** (Weeks 3-4)

**Goal**: Train models on quality data, achieve ≥65% accuracy

#### Week 3: Model Architecture Refinement

**Day 15-17: Fix LSTM Architecture**
```
Update: apps/trainer/models/lstm.py
Changes:
1. Input size: 53 features (was 50)
2. Hidden size: 128 (keep)
3. Layers: 3 → 2 (simplify)
4. Dropout: 0.35 → 0.3
5. Add gradient clipping
6. Add learning rate scheduler

Why:
- Simpler architecture with better data
- Better generalization
- Prevent overfitting
```

**Day 18-19: Training Pipeline V2**
```
Create: scripts/train_v2.py
- Load 53-feature data
- Updated LSTM architecture
- Better hyperparameters:
  - Learning rate: 0.001 → 0.0005
  - Batch size: 32 → 64
  - Epochs: 20 (early stopping patience=5)
- Track metrics properly
- Save checkpoints
```

**Day 20-21: Train All Models**
```bash
# Train on Colab A100 (faster) or local
uv run python scripts/train_v2.py --symbol BTC-USD --epochs 20
uv run python scripts/train_v2.py --symbol ETH-USD --epochs 20
uv run python scripts/train_v2.py --symbol SOL-USD --epochs 20

Expected results:
- Validation accuracy: 62-68% (vs 50% before)
- Training should converge by epoch 10-15
```

**Deliverables**:
- ✅ 3 trained models (BTC/ETH/SOL)
- ✅ Training curves showing convergence
- ✅ Validation accuracy ≥60%
- ✅ Models saved in models/v2/

---

#### Week 4: Evaluation & Validation

**Day 22-24: Comprehensive Evaluation**
```
Create: scripts/evaluate_v2.py
- Load trained models
- Evaluate on test set (walk-forward split)
- Metrics:
  - Accuracy (target: ≥65%)
  - Precision, Recall, F1
  - Calibration error (target: ≤5%)
  - Sharpe ratio on backtest
  - Max drawdown
- Generate detailed report
```

**Day 25-26: Backtesting**
```
Create: scripts/backtest_v2.py
- Full backtest on 2-year data
- Walk-forward validation
- FTMO rules applied:
  - 5% daily loss limit
  - 10% total loss limit
  - Position sizing (1% risk)
- Generate equity curve
- Calculate metrics:
  - Total return
  - Sharpe ratio
  - Win rate
  - Max drawdown
```

**Day 27-28: Decision Point**
```
Review results:
- If accuracy ≥65%: PROCEED to Phase 3
- If accuracy 55-65%: TUNE hyperparameters, retry
- If accuracy <55%: INVESTIGATE (should not happen with quality data)
```

**Deliverables**:
- ✅ Evaluation report (accuracy, calibration)
- ✅ Backtest report (Sharpe, drawdown)
- ✅ Decision: PASS/TUNE/INVESTIGATE
- ✅ Models promoted if passing gates

---

### **PHASE 3: Production Pipeline** (Weeks 5-6)

**Goal**: Build production-grade runtime system

#### Week 5: Real-time Data Pipeline

**Day 29-31: Tardis WebSocket Integration**
```
Create: libs/data/tardis_realtime.py
- Connect to Tardis.dev WebSocket
- Subscribe to BTC/ETH/SOL trades + order book
- Process tick data in real-time
- Aggregate to 1-minute candles
- Compute features on-the-fly
- Store in buffer for inference
```

**Day 32-33: Feature Computation Service**
```
Create: apps/runtime/feature_service.py
- Real-time feature computation
- 1-minute aggregation window
- Compute all 53 features
- Cache for LSTM input
- Handle missing data gracefully
```

**Day 34-35: Model Serving**
```
Update: apps/runtime/inference.py
- Load promoted models
- Real-time inference on 1-min intervals
- Ensemble predictions (if multiple models)
- Confidence calibration
- Rate limiting (max 10 signals/hour)
```

**Deliverables**:
- ✅ Real-time data pipeline working
- ✅ Feature computation validated
- ✅ Model serving tested
- ✅ End-to-end latency <1 second

---

#### Week 6: Signal Generation & Risk Management

**Day 36-38: Signal Generation**
```
Update: apps/runtime/ensemble.py
- Load all 3 models (BTC/ETH/SOL)
- Generate signals on 1-min intervals
- Confidence thresholds:
  - High: ≥75% (max 5/hour)
  - Medium: ≥65% (max 10/hour)
  - Low: ≥55% (log only, don't trade)
- Position sizing: 1% risk per trade
```

**Day 39-40: FTMO Risk Management**
```
Update: libs/risk/ftmo_rules.py
- Real-time P&L tracking
- Daily loss limit: 5%
- Total loss limit: 10%
- Emergency stop loss
- Kill switch (env var)
- Alert system (Telegram/email)
```

**Day 41-42: Dry-run Testing**
```
Run: apps/runtime/main.py --mode dryrun
- 48-hour dry-run
- No real trades
- Log all signals
- Track hypothetical P&L
- Validate risk management
- Check for bugs
```

**Deliverables**:
- ✅ Signal generation working
- ✅ FTMO rules enforced
- ✅ Dry-run completed successfully
- ✅ Ready for paper trading

---

### **PHASE 4: Production Deployment** (Weeks 7-8)

**Goal**: Deploy to production with monitoring

#### Week 7: Infrastructure & Deployment

**Day 43-45: AWS Deployment (Amazon Q)**
```
Tasks for Amazon Q:
1. Setup EC2 instance (t3.medium)
2. Deploy code to production
3. Setup RDS PostgreSQL for trade logs
4. Configure S3 for model storage
5. Setup CloudWatch monitoring
6. Configure auto-restart on failure
```

**Day 46-47: Monitoring & Alerts**
```
Create: apps/monitoring/dashboard.py
- Real-time P&L display
- Daily/total drawdown tracking
- Signal count monitoring
- Model performance metrics
- Alert on:
  - Daily loss >4%
  - Total loss >8%
  - No signals for 1 hour (stale data)
  - Model errors
```

**Day 48-49: Database & Logging**
```
Setup:
- PostgreSQL for signal logs
- Store every prediction with:
  - Timestamp
  - Symbol
  - Prediction
  - Confidence
  - Features snapshot
  - Outcome (for retraining)
- Retention: 2 years
```

**Deliverables**:
- ✅ Production deployment on AWS
- ✅ Monitoring dashboard live
- ✅ Database logging all signals
- ✅ Alerts configured

---

#### Week 8: Paper Trading & Validation

**Day 50-52: Paper Trading (FTMO Demo)**
```
Start paper trading:
- Use FTMO demo account
- Run for 3-5 days minimum
- Track metrics:
  - Win rate
  - Sharpe ratio
  - Max drawdown
  - Daily P&L
- Validate system stability
```

**Day 53-54: Performance Analysis**
```
Analyze paper trading results:
- Compare to backtest expectations
- Check if accuracy holds (≥65%)
- Verify risk management working
- Look for edge cases/bugs
```

**Day 55-56: Go/No-Go Decision**
```
Review:
- If paper trading successful: GO LIVE
- If issues found: DEBUG and retry
- If accuracy drops: INVESTIGATE data/model drift

Success criteria:
✅ Win rate ≥60%
✅ Max drawdown <10%
✅ Daily loss never >5%
✅ System runs 24/7 without crashes
```

**Deliverables**:
- ✅ 5 days paper trading complete
- ✅ Performance analysis report
- ✅ System validated for live trading
- ✅ Ready for FTMO challenge OR Go live

---

## 💰 Budget Breakdown

### Month 1-2 (Foundation)
```
Tardis.dev Premium:     $499/month
AWS EC2 (t3.medium):    ~$30/month
AWS RDS (db.t3.micro):  ~$15/month
AWS S3 (storage):       ~$5/month
Misc (domains, etc):    ~$10/month
───────────────────────────────
Total:                  ~$559/month
```

### Cost Optimization
```
Start:
- Tardis.dev Premium: $499 (essential)
- AWS minimal setup: $50

Scale if successful:
- Upgrade EC2 for lower latency
- Add more data sources
- Expand to more symbols
```

---

## 🎯 Success Metrics

### Phase 1 Success (Week 2)
```
✅ Tardis.dev data downloaded
✅ 53-feature datasets created
✅ Data quality >10x better than free data
```

### Phase 2 Success (Week 4)
```
✅ Model accuracy ≥65% (vs 50% before)
✅ Calibration error ≤5%
✅ Backtest Sharpe ratio >1.0
✅ Max drawdown <15%
```

### Phase 3 Success (Week 6)
```
✅ Real-time pipeline <1s latency
✅ Dry-run 48hrs without errors
✅ Signal generation working
✅ Risk management enforced
```

### Phase 4 Success (Week 8)
```
✅ Paper trading 5 days successful
✅ Win rate ≥60%
✅ Ready for FTMO challenge
✅ System runs 24/7 stable
```

---

## 🚀 Immediate Next Steps (This Week)

### Today (Day 1)
```
1. Subscribe to Tardis.dev Premium
   https://tardis.dev/pricing

2. Get API credentials

3. Install SDK:
   cd /home/numan/crpbot
   uv add tardis-dev
```

### Tomorrow (Day 2)
```
1. Builder Claude creates:
   - libs/data/tardis_client.py
   - scripts/download_tardis_data.py
   - scripts/test_tardis_connection.py

2. Test connection and download sample data

3. Validate data quality
```

### Day 3-4
```
1. Download full 2-year tick data
   - BTC-USD
   - ETH-USD
   - SOL-USD

2. Create data quality report

3. Compare to free Coinbase data
   (Should see huge improvement)
```

### Day 5-7
```
1. Build microstructure features

2. Create 53-feature datasets

3. Run baseline analysis
   (Expect logistic regression >55% vs 50% before)
```

---

## 📊 What Changes From Current Setup

### Architecture (STAYS THE SAME ✅)
```
✅ Feature engineering pipeline
✅ LSTM model architecture (minor tweaks)
✅ Walk-forward validation
✅ Ensemble approach
✅ FTMO risk management
✅ Runtime signal generation

All solid - no major rewrites needed
```

### What Changes (DATA QUALITY 🩸)
```
OLD: Free Coinbase 1-min OHLCV
NEW: Tardis.dev tick data + order book

OLD: 33 basic features
NEW: 53 features (33 basic + 20 microstructure)

OLD: 50% accuracy (random)
NEW: 65-75% accuracy (tradeable edge)

OLD: Can't pass FTMO
NEW: Can pass FTMO challenge
```

---

## 🎯 Key Principles

1. **Start Small, Scale Fast**
   - Week 1: Just get data working
   - Week 2: Just get features working
   - Week 3-4: Just get models working
   - Week 5-6: Just get runtime working
   - Week 7-8: Just validate it works

2. **Validate at Every Step**
   - Don't move to next phase until current phase succeeds
   - Use metrics to make go/no-go decisions
   - If something fails, debug before proceeding

3. **Keep It Simple**
   - Don't over-engineer
   - Use existing architecture where possible
   - Add complexity only when proven necessary

4. **Data Quality First**
   - $500/month for data is non-negotiable
   - Everything else builds on this foundation
   - Poor data = poor models, always

---

## 📋 Agent Responsibilities

### Builder Claude (Cloud Server)
- Write all new code
- Create data pipeline
- Update feature engineering
- Model training scripts
- Create monitoring tools

### QC Claude (Local - Me)
- Review all code
- Validate architecture decisions
- Check data quality
- Review model results
- Approve phase transitions

### Amazon Q (Both Machines)
- All AWS operations
- EC2/RDS/S3 setup
- CloudWatch monitoring
- Infrastructure scaling
- Deployment automation

### You (User)
- Make budget decisions
- Subscribe to services
- Run training on Colab (if needed)
- Final go/no-go on live trading
- Monitor production

---

## 🎯 End Goal (Week 8)

```
✅ Professional quant trading software
✅ High-quality tick data pipeline
✅ Models with ≥65% accuracy
✅ Real-time signal generation
✅ FTMO-compliant risk management
✅ Production-ready deployment
✅ Validated on paper trading
✅ Ready for FTMO challenge

Budget: $500/month (data + infrastructure)
Timeline: 8 weeks from start
Success Rate: HIGH (with quality data)
```

---

**File**: `QUANT_TRADING_PLAN.md`
**Status**: READY TO EXECUTE
**First Action**: Subscribe to Tardis.dev Premium ($499/month)
**Timeline**: 8 weeks to production
**Next**: Your approval to start Phase 1
