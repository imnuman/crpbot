# 🚀 Version 5 Upgrade Plan - Premium Data Integration

**Current Version**: V4 (Free Coinbase data, 50% accuracy)
**Target Version**: V5 (Tardis.dev premium data, 65-75% accuracy)
**Approach**: UPGRADE existing system, not rebuild
**Timeline**: 2-3 weeks
**Budget**: $500/month

---

## ✅ What We Already Have (V1-V4)

### V1: Core Architecture ✅
```
✅ Project structure
✅ Configuration management (.env, settings.py)
✅ Database models (SQLite/PostgreSQL)
✅ Logging and monitoring
✅ Test framework
```

### V2: Data Pipeline ✅
```
✅ Coinbase API client (libs/data/coinbase_client.py)
✅ Data fetching scripts (scripts/fetch_data.py)
✅ Data validation (scripts/validate_data_quality.py)
✅ Parquet storage format
```

### V3: Feature Engineering ✅
```
✅ Feature pipeline (apps/trainer/features.py)
✅ 33 features implemented:
   - OHLCV (5)
   - Session features (5)
   - Spread features (4)
   - Volume features (3)
   - Moving averages (8)
   - Technical indicators (8)
✅ Batch processing (batch_engineer_features.sh)
✅ Walk-forward splits
```

### V4: Model Training ✅
```
✅ LSTM model (apps/trainer/models/lstm.py)
✅ Transformer model (apps/trainer/models/transformer.py)
✅ Training pipeline (apps/trainer/main.py)
✅ Evaluation scripts (scripts/evaluate_model.py)
✅ Promotion gates (68% accuracy, 5% calibration)
✅ Model checkpointing
```

### V4: Runtime System ✅
```
✅ Ensemble predictions (apps/runtime/ensemble.py)
✅ Signal generation (apps/runtime/main.py)
✅ FTMO risk management (libs/risk/ftmo_rules.py)
✅ Rate limiting (10 signals/hour)
✅ Dry-run mode
✅ Database logging
```

### V4: Infrastructure (Partial) ✅
```
✅ Docker setup (infra/docker/)
✅ AWS configs (infra/aws/)
✅ Makefile for common tasks
✅ CI/CD ready
```

---

## 🆕 What Changes in V5

### Only 3 Things Change:

#### 1. Data Source: Coinbase → Tardis.dev
```
OLD: libs/data/coinbase_client.py (keep for backup)
NEW: libs/data/tardis_client.py (add)

Changes:
- Add Tardis.dev SDK
- Create Tardis API client
- Download tick data + order book
- Convert to same parquet format (compatible with existing pipeline)
```

#### 2. Features: 33 → 53
```
OLD: 33 features (keep all)
NEW: +20 microstructure features

Add to apps/trainer/features.py:
- Order book imbalance (5 features)
- Trade flow (buy/sell pressure) (5 features)
- Volume-weighted metrics (5 features)
- Spread dynamics (3 features)
- Price impact (2 features)

Total: 53 features
```

#### 3. Retrain Models
```
Same architecture, same code, just:
- Point to new 53-feature datasets
- Retrain LSTM (input_size: 33 → 53)
- Retrain Transformer (same change)
- Re-run evaluation
- Promote if passing gates (expect 65-75%)
```

**That's it!** Everything else stays the same.

---

## 📋 V5 Upgrade Timeline (2-3 Weeks)

### Week 1: Data Integration

**Day 1: Subscribe & Setup**
```bash
Tasks:
1. Subscribe to Tardis.dev Premium ($499/month)
2. Get API credentials
3. Install SDK:
   uv add tardis-dev
4. Add to .env:
   TARDIS_API_KEY=tard_xxxxx
   TARDIS_API_SECRET=xxxxx
```

**Day 2-3: Create Tardis Client**
```python
# Builder Claude creates:
libs/data/tardis_client.py

Similar to coinbase_client.py but for Tardis:
- Connect to Tardis API
- Download tick data
- Download order book snapshots
- Save to parquet (same format as Coinbase)
- Stored in: data/tardis/raw/
```

**Day 4-6: Download Historical Data**
```bash
# New script (similar to fetch_data.py):
scripts/fetch_tardis_data.py --symbol BTC-USD --start 2023-11-10
scripts/fetch_tardis_data.py --symbol ETH-USD --start 2023-11-10
scripts/fetch_tardis_data.py --symbol SOL-USD --start 2023-11-10

Output:
data/tardis/raw/
  ├── BTC-USD_trades_2023-2025.parquet
  ├── BTC-USD_orderbook_2023-2025.parquet
  ├── ETH-USD_trades_2023-2025.parquet
  └── ...

Time: 4-6 hours per symbol (background download)
```

**Day 7: Validate Data Quality**
```bash
# Reuse existing validation script:
scripts/validate_data_quality.py --source tardis --symbol BTC-USD

# Compare to Coinbase:
scripts/compare_data_sources.py
  -> Shows Tardis has 100x more data points
  -> No gaps
  -> Complete order book
```

**Week 1 Deliverable**: ✅ 2 years of Tardis tick data downloaded and validated

---

### Week 2: Feature Engineering V5

**Day 8-9: Add Microstructure Features**
```python
# Update existing file:
apps/trainer/features.py

Add new functions:
- compute_order_book_imbalance(df_orderbook)
- compute_trade_flow(df_trades)
- compute_vwap(df_trades)
- compute_spread_dynamics(df_orderbook)
- compute_price_impact(df_trades, df_orderbook)

Update FEATURE_COLUMNS:
FEATURE_COLUMNS = [
    # Existing 33 features (keep all)
    'open', 'high', 'low', 'close', 'volume',
    'session_tokyo', 'session_london', ...

    # NEW: 20 microstructure features
    'ob_imbalance_1', 'ob_imbalance_5', ...
    'trade_flow_buy', 'trade_flow_sell', ...
    'vwap_1m', 'vwap_5m', ...
]
# Total: 53 features
```

**Day 10-11: Update Feature Engineering Script**
```bash
# Update existing script:
scripts/engineer_features.py

Changes:
- Add --source tardis flag
- Load tick data + order book
- Aggregate to 1-minute
- Compute all 53 features
- Save to: data/features/features_{SYMBOL}_1m_v5.parquet

Run:
./batch_engineer_features.sh --version v5
```

**Day 12-13: Validate Features**
```bash
# Reuse existing validation:
scripts/validate_data_quality.py --symbol BTC-USD

# New baseline check:
scripts/investigate_50feat_failure.py --symbol BTC-USD

Expected:
- Logistic regression baseline: 55-60% (vs 50% before)
- Features have higher correlation with target
- No NaN/Inf issues
```

**Day 14: Feature Analysis**
```python
# New quick script:
scripts/analyze_v5_features.py

Output:
- Top 10 features by importance
- Correlation with target
- V4 vs V5 comparison
- Expected: Microstructure features in top 10
```

**Week 2 Deliverable**: ✅ 53-feature datasets for BTC/ETH/SOL ready

---

### Week 3: Model Retraining & Validation

**Day 15-16: Update Model Architecture**
```python
# Minimal change to existing model:
apps/trainer/models/lstm.py

Change:
- input_size=33 → input_size=53

That's it! Everything else stays the same.
```

**Day 17-18: Retrain Models**
```bash
# Use existing training pipeline:
make train COIN=BTC EPOCHS=20
make train COIN=ETH EPOCHS=20
make train COIN=SOL EPOCHS=20

# Or on Colab A100 (faster):
# Builder Claude updates colab notebook for 53 features
# You run in Colab

Expected:
- Validation accuracy: 62-68% by epoch 10
- Should converge much better than V4 (50%)
```

**Day 19: Evaluate Models**
```bash
# Use existing evaluation script:
scripts/evaluate_model.py \
  --model models/lstm_BTC_USD_1m_v5.pt \
  --symbol BTC-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05

Expected results:
BTC-USD: 68-72% accuracy ✅
ETH-USD: 65-70% accuracy ✅
SOL-USD: 64-68% accuracy ✅
```

**Day 20: Backtest**
```bash
# Use existing backtest:
make smoke  # 5-minute backtest

# Full backtest:
scripts/backtest_v5.py --days 90

Expected:
- Win rate: 60-65%
- Sharpe ratio: >1.0
- Max drawdown: <15%
```

**Day 21: Decision Point**
```
Review:
✅ If accuracy ≥68%: PROMOTE TO PRODUCTION
⚠️ If accuracy 60-67%: TUNE hyperparameters
❌ If accuracy <60%: INVESTIGATE (unlikely with Tardis data)

Promote:
cp models/lstm_*_v5.pt models/promoted/
```

**Week 3 Deliverable**: ✅ V5 models trained, evaluated, promoted (if passing)

---

## 🔄 Migration Strategy

### Phase 1: Parallel Running (Week 4-5)
```
Keep V4 running (Coinbase data)
Start V5 in dry-run (Tardis data)

Compare:
- V4 signals vs V5 signals
- V4 accuracy vs V5 accuracy
- Look for issues

Expected: V5 should perform much better
```

### Phase 2: Gradual Cutover (Week 6)
```
Day 1-2: V5 50% weight, V4 50% weight
Day 3-4: V5 75% weight, V4 25% weight
Day 5-7: V5 100% weight, V4 deprecated

Monitor closely for any issues
```

### Phase 3: Full V5 (Week 7+)
```
V4 code archived (keep as reference)
V5 is production
Coinbase API kept as backup data source
```

---

## 📊 What Stays Exactly The Same

### Runtime System (No Changes) ✅
```
apps/runtime/main.py          # Same
apps/runtime/ensemble.py      # Same (just load V5 models)
apps/runtime/inference.py     # Same
libs/risk/ftmo_rules.py       # Same
```

### Database (No Changes) ✅
```
Schema: Same
Logging: Same
Queries: Same
```

### Infrastructure (No Changes) ✅
```
AWS deployment: Same
Docker: Same
Makefile: Same
```

### FTMO Rules (No Changes) ✅
```
5% daily loss limit: Same
10% total loss limit: Same
Position sizing: Same
Rate limiting: Same
```

---

## 💰 Costs (V4 vs V5)

### V4 Costs (Current):
```
Data: $0 (Coinbase free)
AWS: ~$50/month
Total: $50/month
```

### V5 Costs (Upgrade):
```
Data: $499/month (Tardis.dev Premium)
AWS: ~$50/month (same)
Total: ~$550/month
```

**Increase**: +$499/month
**ROI**: One FTMO 10% profit ($1,000-20,000) pays for 2-40 months of data

---

## 🎯 Success Criteria

### Week 1 Success:
```
✅ Tardis.dev data downloaded
✅ Data quality validated
✅ 10x more data points than Coinbase
```

### Week 2 Success:
```
✅ 53-feature datasets created
✅ Baseline accuracy >55% (vs 50% in V4)
✅ Features validated
```

### Week 3 Success:
```
✅ Models trained
✅ Test accuracy ≥68% (promotion gate)
✅ Calibration error ≤5%
✅ Backtest Sharpe >1.0
```

### V5 Production Success:
```
✅ V5 outperforms V4 in dry-run
✅ 60%+ win rate in paper trading
✅ System stable 24/7
✅ Ready for FTMO challenge
```

---

## 📋 File Changes Summary

### New Files (V5):
```
libs/data/tardis_client.py              # Tardis API client
scripts/fetch_tardis_data.py            # Download tick data
scripts/compare_data_sources.py         # V4 vs V5 comparison
scripts/analyze_v5_features.py          # Feature analysis
```

### Modified Files (V5):
```
apps/trainer/features.py                # +20 microstructure features
apps/trainer/models/lstm.py             # input_size: 33→53
scripts/engineer_features.py            # Support Tardis data
.env                                     # Add TARDIS_API_KEY/SECRET
pyproject.toml                          # Add tardis-dev dependency
```

### Unchanged Files (Keep As-Is):
```
apps/runtime/*                          # All runtime code
libs/risk/*                             # All risk management
libs/constants/*                        # All constants
tests/*                                 # All tests (may need updates)
infra/*                                 # All infrastructure
```

**Total new code**: ~500-800 lines
**Total modified code**: ~200 lines
**Total unchanged code**: ~5,000+ lines

**Effort**: ~90% reuse, ~10% new

---

## 🚀 Immediate Next Steps

### Today (Day 1):
```
1. Subscribe to Tardis.dev Premium ($499/month)
   https://tardis.dev/pricing

2. Get API credentials

3. Add to .env:
   echo "TARDIS_API_KEY=tard_xxxxx" >> .env
   echo "TARDIS_API_SECRET=xxxxx" >> .env

4. Install SDK:
   uv add tardis-dev
```

### Tomorrow (Day 2):
```
Builder Claude creates:
- libs/data/tardis_client.py
- scripts/fetch_tardis_data.py
- Test connection
```

### This Week (Week 1):
```
Download all historical data
Validate quality
Prepare for feature engineering
```

---

## 🎯 V5 vs V4 Comparison

| Aspect | V4 (Current) | V5 (Upgrade) |
|--------|--------------|--------------|
| **Data Source** | Coinbase Free | Tardis.dev Premium |
| **Data Type** | 1-min OHLCV | Tick + Order Book |
| **Features** | 33 | 53 |
| **Model Accuracy** | 50% (random) | 65-75% (expected) |
| **Code Changes** | - | ~10% new, 90% reuse |
| **Timeline** | - | 2-3 weeks |
| **Cost** | $50/month | $550/month |
| **FTMO Ready** | ❌ No | ✅ Yes |
| **Architecture** | Same | Same |
| **Runtime** | Same | Same |
| **Risk Mgmt** | Same | Same |

---

## ✅ Version History

```
V1: Core architecture, database, logging
V2: Coinbase data pipeline, basic features
V3: Feature engineering (33 features), validation
V4: LSTM + Transformer models, runtime system
V5: Tardis.dev data, microstructure features (53), retrained models

Current: V4
Target: V5
Approach: Upgrade, not rebuild
```

---

**This is a SIMPLE UPGRADE:**
1. Add Tardis.dev data source (Week 1)
2. Add 20 microstructure features (Week 2)
3. Retrain models (Week 3)
4. Deploy V5 (Week 4+)

**Everything else you built stays exactly the same!** ✅

---

**File**: `V5_UPGRADE_PLAN.md`
**Type**: Upgrade path (not rebuild)
**Timeline**: 2-3 weeks
**Effort**: ~10% new code, 90% reuse
**Next**: Subscribe to Tardis.dev, start Week 1
