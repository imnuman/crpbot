# V5 Phase 1 Plan - Data Upgrade Strategy

**Created**: November 15, 2025
**Duration**: 4 weeks (validation period)
**Budget**: $148/month (Canada-compliant)
**Goal**: Achieve 65-75% model accuracy (vs V4's 50% ceiling)

---

## 🎯 Strategic Context

### The V4 Problem

**V4 Results**:
- ❌ Models stuck at 50% accuracy
- ❌ Root cause: Free Coinbase OHLCV data too noisy
- ❌ No amount of hyperparameter tuning could break 50% ceiling
- ❌ Conclusion: Need professional-grade data

### The V5 Solution

**Strategy**: UPGRADE, not rebuild
- ✅ **10% change**: Data layer only (Tardis.dev professional data)
- ✅ **90% reuse**: Architecture, training pipeline, runtime, FTMO rules
- ✅ **Budget-conscious**: $148/month validation, scale to $549/month only if successful
- ✅ **Clear metrics**: 65-75% accuracy target in 4 weeks

---

## 📊 V5 Data Strategy

### Phase 1 Data Sources (NOW)

**Primary**: Tardis.dev Historical - **$98/month**
- **What**: Professional tick-level market data
- **Coverage**: 2 exchanges (Coinbase, Kraken) - Canada-compliant
- **Depth**:
  - All trades (tick-by-tick)
  - Full order book snapshots
  - 2+ years historical data
- **Symbols**: BTC-USD, ETH-USD, SOL-USD
- **Subscribe**: https://tardis.dev/pricing
- **Note**: Binance excluded (banned in Canada)

**Secondary**: Coinbase Advanced Trade API - **Free**
- **What**: Real-time OHLCV + orderbook
- **Status**: Already integrated in V4
- **Purpose**: Real-time inference during runtime

### Phase 3-5 Additions (LATER)

If Phase 1 succeeds, add:
1. **On-chain data** (Glassnode) - ~$100/month
   - Network metrics, whale movements, supply distribution
2. **News sentiment** (CryptoPanic) - ~$50/month
   - Real-time crypto news, sentiment scoring
3. **Social sentiment** (LunarCrush) - ~$100/month
   - Twitter/Reddit volume, engagement metrics

**Phase 1 Total**: $148/month ($98 Tardis + $50 AWS - Canada-compliant)

---

## 🔢 V5 Feature Engineering

### 53 Total Features

**33 Existing Features** (from V4 - REUSE):
1. **OHLCV** (5): open, high, low, close, volume
2. **Session Features** (5): Tokyo/London/NY sessions, day_of_week, is_weekend
3. **Spread Features** (4): spread, spread_pct, ATR, spread_atr_ratio
4. **Volume Features** (3): volume_ma, volume_ratio, volume_trend
5. **Moving Averages** (8): SMA 7/14/21/50 + price ratios
6. **Technical Indicators** (8): RSI, MACD×3, Bollinger Bands×4

**20 NEW Microstructure Features** (V5 additions):

**Order Book Features** (8):
1. `bid_ask_spread` - Immediate spread (basis points)
2. `spread_volatility` - Rolling std of spread (10-min window)
3. `order_book_imbalance` - (bid_volume - ask_volume) / (bid + ask)
4. `book_pressure_5` - Volume imbalance at 5 levels deep
5. `book_pressure_10` - Volume imbalance at 10 levels deep
6. `weighted_mid_price` - Volume-weighted mid price
7. `microprice` - Order flow microprice estimator
8. `effective_spread` - Realized spread from trades

**Order Flow Features** (6):
9. `trade_intensity` - Trades per minute
10. `buy_volume_ratio` - Buy trades / total trades
11. `sell_volume_ratio` - Sell trades / total trades
12. `trade_size_mean` - Average trade size
13. `trade_size_std` - Trade size volatility
14. `large_trade_ratio` - Trades >2σ / total trades

**Tick-Level Volatility** (4):
15. `tick_volatility` - Std of tick prices (1-min)
16. `tick_volatility_5m` - Std of tick prices (5-min)
17. `realized_volatility` - Sum of squared returns
18. `jump_indicator` - Large price jumps (>3σ)

**Execution Quality** (2):
19. `vwap_distance` - Price distance from VWAP (basis points)
20. `arrival_price_impact` - Market impact of recent trades

### Feature Engineering Script

**New file**: `scripts/engineer_v5_features.py`

```python
def engineer_v5_features(tardis_data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 53 features from Tardis tick data.

    Args:
        tardis_data: Raw tick data with trades + order book snapshots

    Returns:
        DataFrame with 53 features, 1-minute aggregation
    """
    # Step 1: Aggregate ticks to 1-minute OHLCV (5 features)
    ohlcv = aggregate_ohlcv(tardis_data)

    # Step 2: Compute V4 features (28 features - reuse existing code)
    v4_features = engineer_v4_features(ohlcv)

    # Step 3: Compute microstructure features (20 features - NEW)
    microstructure = engineer_microstructure_features(tardis_data)

    # Step 4: Combine all 53 features
    return pd.concat([ohlcv, v4_features, microstructure], axis=1)
```

---

## 🗓️ 4-Week Timeline

### Week 1: Data Download & Validation (Nov 15-22)

**Objective**: Download and validate Tardis historical data

**Tasks**:
1. ✅ Subscribe to Tardis.dev Historical ($98/month)
2. 📥 Download tick data for BTC-USD, ETH-USD, SOL-USD
   - Timeframe: 2 years (Nov 2023 - Nov 2025)
   - Exchanges: Coinbase, Kraken (Canada-compliant only)
   - Data types: Trades + Order book snapshots
   - Note: Binance excluded (banned in Canada)
3. 📊 Validate data quality:
   - Check for gaps, missing timestamps
   - Verify tick counts, order book depth
   - Compare against Coinbase free data (sanity check)
4. 💾 Store in S3 + local parquet
   - Raw format: `data/tardis/raw/{exchange}_{symbol}_trades_{date}.parquet`
   - Order book: `data/tardis/orderbook/{exchange}_{symbol}_book_{date}.parquet`

**Expected Output**:
- ~50-100 GB of raw tick data
- Data quality report
- Storage setup complete

**Scripts to Create**:
- `scripts/download_tardis_data.py`
- `scripts/validate_tardis_data.py`

---

### Week 2: Feature Engineering (Nov 22-29)

**Objective**: Engineer 53 features from tick data

**Tasks**:
1. 🔧 Implement microstructure feature calculations
   - Order book features (8)
   - Order flow features (6)
   - Tick volatility (4)
   - Execution quality (2)
2. 🔄 Integrate with V4 feature pipeline
   - Reuse 33 existing features
   - Combine into 53-feature dataset
3. ✅ Validate feature distributions
   - Check for NaNs, infinities
   - Verify feature correlations
   - Plot feature importance heatmap
4. 💾 Generate final feature files
   - Format: `data/features/features_{symbol}_v5_1m.parquet`
   - Size: ~53 columns × 1M rows × 3 symbols = ~1-2 GB

**Expected Output**:
- 3 parquet files (BTC, ETH, SOL) with 53 features each
- Feature validation report
- Feature engineering notebook

**Scripts to Create**:
- `scripts/engineer_v5_features.py`
- `scripts/validate_v5_features.py`
- `notebooks/v5_feature_analysis.ipynb`

---

### Week 3: Model Training (Nov 29 - Dec 6)

**Objective**: Train models with V5 professional data

**Tasks**:
1. 🎯 Update model input dimensions
   - Change: 31-50 features → 53 features
   - Models affected: LSTM input layer, Transformer input projection
   - Code location: `apps/trainer/models/lstm.py`, `apps/trainer/models/transformer.py`
2. 🏋️ Train LSTM models (per-coin)
   - BTC-USD: 60-min lookback, 53 features
   - ETH-USD: 60-min lookback, 53 features
   - SOL-USD: 60-min lookback, 53 features
   - Architecture: 128/3/bidirectional (1M+ params) - UNCHANGED
   - Epochs: 15-20 (with early stopping)
3. 🏋️ Train Transformer (multi-coin)
   - 100-min lookback, 53 features
   - Architecture: 4-layer, 8-head attention - UNCHANGED
   - Epochs: 15-20
4. 💾 Save trained models
   - Format: `models/v5/lstm_{symbol}_v5.pt`
   - Format: `models/v5/transformer_multi_v5.pt`

**Expected Output**:
- 3 LSTM models + 1 Transformer model
- Training logs and loss curves
- Model checkpoints

**Training Options**:
- **Local/Cloud CPU**: 60-90 min per model (slow)
- **Google Colab GPU**: 5-10 min per model (fast, recommended)

---

### Week 4: Validation & Promotion (Dec 6-13)

**Objective**: Validate models against promotion gates

**Tasks**:
1. 📊 Evaluate all models on test set
   - Metrics: Accuracy, Precision, Recall, F1, Calibration Error
   - Test set: 15% holdout (most recent data)
2. ✅ Check promotion gates
   - **Accuracy**: ≥65% (relaxed from V4's 68% for first validation)
   - **Calibration**: ≤5% (ECE or Brier score)
   - **Target**: 65-75% accuracy range
3. 📈 Compare V4 vs V5 results
   - V4: 50% accuracy (baseline)
   - V5: Target 65-75% accuracy
   - Delta: +15-25 percentage points improvement
4. 🎯 Decision gate
   - **If ≥65% accuracy**: ✅ Promote to production, proceed to Phase 2
   - **If <65% accuracy**: ❌ Investigate, iterate features, or pivot

**Expected Output**:
- Evaluation report with accuracy comparison
- Promotion decision (GO / NO-GO for Phase 2)
- Models promoted to `models/v5/promoted/` if successful

**Scripts to Create**:
- `scripts/evaluate_v5_models.py`
- `scripts/compare_v4_v5.py`

---

## 💰 Budget Breakdown

### Phase 1 (Validation) - 4 Weeks

| Item | Cost | Notes |
|------|------|-------|
| Tardis Historical | $98/month | Tick data + order book, 2 exchanges (Coinbase + Kraken) |
| Coinbase API | $0/month | Already have, real-time data |
| AWS EC2 (training) | ~$20/month | Or use Colab GPU (free) |
| AWS S3 (storage) | ~$10/month | ~100 GB storage |
| AWS RDS (database) | ~$20/month | PostgreSQL for signals |
| **Phase 1 Total** | **~$148/month** | ✅ Under $200 budget (Canada-compliant) |

### Phase 2 (Live Trading) - Only if Phase 1 Succeeds

| Item | Cost | Notes |
|------|------|-------|
| Tardis Premium | $499/month | Real-time tick data + order book |
| AWS (scaled up) | ~$50/month | Production infrastructure |
| **Phase 2 Total** | **~$549/month** | Only if models hit 65-75% accuracy |

**Cost Control**:
- Phase 1 is **validation only** - abort if <65% accuracy
- Phase 2 only starts if Phase 1 proves ROI
- Can cancel Tardis anytime (no long-term contract)

---

## 📋 Implementation Checklist

### Pre-Phase 1 (NOW)
- [ ] Subscribe to Tardis.dev Historical ($98/month)
- [ ] Set up Tardis API credentials in `.env`
- [ ] Create data storage directories
- [ ] Update PROJECT_MEMORY.md (✅ Done)
- [ ] Update CLAUDE.md (✅ Done)

### Week 1: Data
- [ ] Download Tardis tick data (2 years, 3 symbols)
- [ ] Download Tardis order book snapshots
- [ ] Validate data quality (gaps, timestamps, counts)
- [ ] Store in S3 + local parquet
- [ ] Create data quality report

### Week 2: Features
- [ ] Implement 20 microstructure features
- [ ] Integrate with 33 existing V4 features
- [ ] Validate 53-feature dataset (NaNs, distributions)
- [ ] Generate final parquet files (3 symbols)
- [ ] Create feature analysis notebook

### Week 3: Training
- [ ] Update model input dimensions (31-50 → 53)
- [ ] Train 3 LSTM models (BTC, ETH, SOL)
- [ ] Train 1 Transformer model (multi-coin)
- [ ] Save model checkpoints
- [ ] Review training logs

### Week 4: Validation
- [ ] Evaluate models on test set
- [ ] Check promotion gates (≥65% accuracy, ≤5% calibration)
- [ ] Compare V4 vs V5 performance
- [ ] **DECISION**: GO / NO-GO for Phase 2
- [ ] Promote models if successful

---

## 🎯 Success Criteria

### Phase 1 Success = GO to Phase 2

**Requirements**:
1. ✅ **Accuracy**: ≥65% on test set (at least 1 model)
2. ✅ **Calibration**: ≤5% calibration error
3. ✅ **Improvement**: +15 percentage points vs V4 (50% baseline)
4. ✅ **Stability**: Models converge during training (no divergence)

**If all criteria met**:
- Promote V5 models to production
- Subscribe to Tardis Premium ($499/month)
- Start Phase 2 (live trading observation)

### Phase 1 Failure = Iterate or Pivot

**If <65% accuracy**:
- Analyze failure modes (which features didn't help?)
- Consider:
  - Different feature combinations
  - Alternative data sources (e.g., add on-chain data)
  - Different model architectures
  - Longer training, different hyperparameters
- **Pivot option**: Try ensemble of multiple data sources

---

## 🔧 Technical Implementation

### Data Pipeline Changes

**New modules**:
```
libs/
├── tardis/                # NEW
│   ├── client.py         # Tardis API client
│   ├── download.py       # Bulk download utilities
│   └── transform.py      # Tick → OHLCV aggregation
```

**Updated modules**:
```
scripts/
├── engineer_v5_features.py   # NEW - 53 features
├── validate_v5_features.py   # NEW - Feature validation
└── download_tardis_data.py   # NEW - Data download

apps/trainer/
├── features.py           # UPDATED - Add microstructure features
└── models/
    ├── lstm.py           # UPDATED - 53 input features
    └── transformer.py    # UPDATED - 53 input features
```

### Architecture Changes (Minimal)

**LSTM Model**:
```python
# OLD (V4)
Input: (batch_size, 60, 31-50)  # Variable features

# NEW (V5)
Input: (batch_size, 60, 53)     # Fixed 53 features
```

**Transformer Model**:
```python
# OLD (V4)
Input: (batch_size, 100, 31-50)  # Variable features

# NEW (V5)
Input: (batch_size, 100, 53)     # Fixed 53 features
```

**Everything else UNCHANGED**:
- LSTM architecture: 128/3/bidirectional
- Transformer: 4-layer, 8-head attention
- Training loop, optimizer, loss functions
- Runtime, ensemble, FTMO rules
- Confidence calibration, rate limiting

---

## 📊 Expected Outcomes

### Conservative Estimate
- **Accuracy**: 65-68% (V4: 50%)
- **Improvement**: +15-18 percentage points
- **Confidence**: Medium (data quality should help significantly)

### Optimistic Estimate
- **Accuracy**: 70-75% (V4: 50%)
- **Improvement**: +20-25 percentage points
- **Confidence**: High (professional data + microstructure features)

### Risk Factors
- ⚠️ Microstructure features may not generalize (overfitting risk)
- ⚠️ Tardis data quality issues (gaps, errors)
- ⚠️ Model may need architecture changes (currently assuming no changes)
- ⚠️ 4 weeks may not be enough (could need 6-8 weeks)

---

## 🔄 Rollback Plan

**If V5 fails (<65% accuracy)**:
1. Cancel Tardis subscription (no long-term commitment)
2. Total sunk cost: ~$148 (4 weeks validation)
3. Keep V5 code (may be useful later)
4. Consider alternatives:
   - Try different data provider (e.g., CryptoCompare, Kaiko)
   - Focus on alternative strategies (arbitrage, market making)
   - Pivot to different asset classes

**V4 code is preserved** in git history - can always revert if needed.

---

## 📞 Communication Protocol

### Weekly Updates

**Every Friday**:
- Update `V5_PHASE1_WEEKLY_REPORT.md` with:
  - Week N summary (what was done)
  - Current blockers
  - Next week plan
  - Budget actuals vs planned

**QC Claude Review**:
- Local Claude reviews cloud Claude's V5 work
- QC reviews posted in `QC_REVIEW_V5_YYYY-MM-DD.md`

### Decision Points

**End of Week 4**:
- **GO / NO-GO meeting** for Phase 2
- If GO: Subscribe to Tardis Premium, start Phase 2
- If NO-GO: Document lessons learned, plan next steps

---

## 🚀 Next Immediate Actions

1. **NOW**: Subscribe to Tardis.dev Historical ($98/month)
   - URL: https://tardis.dev/pricing
   - Plan: Historical
   - Exchanges: Coinbase, Kraken (Canada-compliant, no Binance)

2. **Week 1**: Download data
   - Create `scripts/download_tardis_data.py`
   - Download 2 years tick data (BTC, ETH, SOL)

3. **Week 2**: Engineer features
   - Create `scripts/engineer_v5_features.py`
   - Implement 20 microstructure features

4. **Week 3**: Train models
   - Update model input dimensions
   - Train on Google Colab GPU

5. **Week 4**: Validate & decide
   - Evaluate against 65% accuracy gate
   - GO / NO-GO for Phase 2

---

## 📚 Resources

**Tardis.dev**:
- Website: https://tardis.dev/
- Pricing: https://tardis.dev/pricing
- Docs: https://docs.tardis.dev/
- API: https://docs.tardis.dev/api

**Microstructure References**:
- "Market Microstructure in Practice" (Lehalle & Laruelle)
- "Algorithmic and High-Frequency Trading" (Cartea et al.)
- Order book imbalance: Kyle (1985), Glosten-Milgrom (1985)

**Git References**:
- V4 code: `git log --before="2025-11-15"` (archived)
- V5 code: `git log --since="2025-11-15"` (new)

---

**Last Updated**: 2025-11-15
**Next Review**: 2025-11-22 (End of Week 1)
**Owner**: Cloud Claude (Builder)
**Reviewer**: QC Claude (Local)
