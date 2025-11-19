# ✅ V5 Simple Plan - REVISED

**Created**: 2025-11-14 10:00 EST (Toronto)
**Last Updated**: 2025-11-15 14:45 EST (Toronto)
**Author**: QC Claude
**Status**: REVISED - Corrected Data Provider Pricing
**Budget**: <$200/month Phase 1
**Timeline**: 3-4 weeks to validation

---

## ⚠️ PRICING CORRECTION

**Previous Error**: Stated Tardis.dev Historical at $98/month
**Actual Pricing**: Tardis.dev minimum is **$300-350+/month** ($6000+ for enterprise)
**New Recommendation**: CoinGecko Analyst at **$129/month**

---

## 🎯 The Plan (Simple):

### What We Have:
```
✅ Coinbase API (FREE) - Real-time data, already working
✅ Software architecture - All built and working
✅ Runtime system - Ready to test
✅ Canada compliant - Using Coinbase (allowed)
```

### What We Add:
```
🆕 CoinGecko Analyst API ($129/month)
   - High-quality OHLCV historical data for TRAINING
   - 2+ years historical data
   - Multiple exchanges (Coinbase, Kraken, etc.)
   - Use to train models to 65-75% accuracy
   - Canada-compliant
```

### What We Keep:
```
✅ Coinbase real-time (FREE)
   - For runtime testing
   - Dry-run mode
   - Already configured
   - API key obtained
```

---

## 📊 Data Setup:

```
TRAINING:
└── CoinGecko Analyst ($129/month)
    ├── OHLCV data (1m, 5m, 15m, 1h, 1d intervals)
    ├── 2+ years historical
    ├── Multiple exchanges aggregated
    └── Canada-compliant

RUNTIME TESTING:
└── Coinbase API (FREE, already have)
    ├── Real-time WebSocket
    └── Good for testing

LIVE TRADING (Phase 2 - if Phase 1 proves profitable):
└── Option A: Continue with CoinGecko + Coinbase (same $129)
└── Option B: Upgrade to Tardis Premium ($499/month for tick data)
    └── Only if ROI justifies the 4x cost increase
```

---

## 💰 Budget:

### Phase 1 (Validation):
```
CoinGecko Analyst:  $129/month
Coinbase real-time: $0/month (already have)
AWS:                ~$25/month (S3 + RDS optimized)
────────────────────────────────────
Total:              $154/month ✅ (Under $200 budget)
```

### Phase 2 (Live Trading - if validation succeeds):
```
Option A - Conservative (recommended):
CoinGecko Analyst:  $129/month
AWS:                ~$50/month (GPU + production scale)
────────────────────────────────────
Total:              $179/month ✅

Option B - Premium (if ROI proven >$500/month):
Tardis Premium:     $499/month (tick data + order book)
AWS:                ~$50/month
────────────────────────────────────
Total:              $549/month
(Only upgrade if profit justifies 3x cost increase)
```

---

## 📋 4-Week Timeline:

### Week 1: Download Data
```
Day 1: Use CoinGecko API ($129 - already obtained key!)
Day 2-3: Download OHLCV data (BTC/ETH/SOL, 2 years, multiple intervals)
Day 4-7: Validate data quality and engineer features
```

### Week 2: Build Features
```
Day 8-10: Add price action features from OHLCV
Day 11-14: Engineer 40-50 feature datasets
Expected: Baseline >55% (vs current 50%)
```

### Week 3: Train Models
```
Day 15-17: Update architecture for new features
Day 18-21: Train on CoinGecko data
Expected: 65-75% validation accuracy
```

### Week 4: Validate
```
Day 22-25: Backtest on CoinGecko data
Day 26-28: Test runtime with Coinbase real-time
Decision: Continue if ≥68% accuracy
```

---

## 🚀 Immediate Next Steps:

### Step 1: Configure CoinGecko API ✅
**Status**: DONE - API key obtained and configured in `.env`

**Location**: `/home/numan/crpbot/.env` line 24
```bash
COINGECKO_API_KEY=your-key-here
```

### Step 2: Create Data Fetcher Script
**Task**: Write script to download OHLCV data from CoinGecko
**Timeline**: 1-2 hours
**Output**: 2 years of 1m candles for BTC/ETH/SOL

### Step 3: Feature Engineering
**Task**: Engineer features from OHLCV data
**Timeline**: 2-3 hours
**Output**: Feature files ready for training

### Step 4: Start Training
**Task**: Train models on CoinGecko data
**Timeline**: 3-4 hours (using AWS GPU)
**Output**: Promoted models in `models/promoted/`

---

## ✅ Success Criteria:

**Week 4 Decision Point**:
```
IF models achieve ≥68% accuracy on test set:
✅ Validation successful
✅ Continue with CoinGecko ($129/month)
✅ Start live trading with Coinbase real-time

IF models achieve 60-67%:
⚠️  Tune hyperparameters
⚠️  Retry training
⚠️  Then decide on continuation

IF models <60%:
❌ Investigate data quality
❌ Consider alternative approaches
```

---

## 🎯 Bottom Line:

**Phase 1**:
- Add: CoinGecko Analyst ($129)
- Keep: Coinbase real-time (free)
- Validate: Models can achieve 65-75%
- Cost: $154/month ✅
- Timeline: 4 weeks

**Phase 2** (only if Phase 1 succeeds):
- Continue: CoinGecko ($129) OR
- Upgrade: Tardis Premium ($499) if ROI justifies
- Deploy: Live trading
- Start: FTMO challenge
- Cost: $179-549/month (depending on choice)

**Total at-risk**: $129 (just CoinGecko subscription)

---

## 📁 Quick Reference:

**Current Status**: V4 (50% accuracy, free Coinbase data)

**Target**: V5 (65-75% accuracy, CoinGecko training data)

**What changes**: 10% (data source + features)

**What stays**: 90% (all architecture, runtime, FTMO rules)

**Budget**: <$200/month Phase 1 ✅

**Canada compliant**: ✅ (Coinbase + CoinGecko both work)

**Real-time data**: ✅ (Coinbase free, already have)

---

## 📊 Data Provider Comparison (for reference):

| Provider | Phase 1 Cost | Data Type | Canada OK | Notes |
|----------|--------------|-----------|-----------|-------|
| **CoinGecko** | **$129/mo** | OHLCV historical | ✅ | **RECOMMENDED** - Good balance |
| Coinbase | $0 | Real-time OHLCV | ✅ | Already have, keep for runtime |
| CryptoCompare | $0-49 | OHLCV | ✅ | Free tier available, limited |
| Tardis.dev | $300-350+ | Tick + order book | ✅ | Too expensive for Phase 1 |
| Binance | N/A | N/A | ❌ | Banned in Canada |

---

## 🔄 What Changed From Previous Version:

1. **Corrected Tardis pricing**: $98 → $300-350+ minimum
2. **New data provider**: Tardis → CoinGecko Analyst
3. **Updated Phase 1 budget**: $148 → $154/month
4. **Simplified approach**: OHLCV data instead of tick data
5. **Faster timeline**: CoinGecko API easier to integrate
6. **Lower risk**: $129/month vs $300-350+/month

---

**Ready to start? API key is already configured!**

Next step: Create CoinGecko data fetcher script

---

**File**: `V5_SIMPLE_PLAN.md`
**Status**: REVISED with correct pricing
**Next**: Build CoinGecko integration
**Timeline**: 4 weeks to validation
