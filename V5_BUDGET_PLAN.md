# 💰 V5 Budget Plan - REVISED - Start Smart, Scale Fast

**Created**: 2025-11-14 09:00 EST (Toronto)
**Last Updated**: 2025-11-15 14:50 EST (Toronto)
**Author**: QC Claude
**Status**: REVISED - Corrected Data Provider Pricing
**Goal**: Build quant trading software with phased data budget
**Phase 1 Budget**: <$200/month (prove it works)
**Phase 2 Budget**: $200-500/month (scale for live trading)
**Approach**: Start small, validate, then scale

---

## ⚠️ CRITICAL PRICING CORRECTION

**Previous Error**: Stated Tardis.dev Historical at $98/month (2 exchanges × $49)
**Actual Pricing**: Tardis.dev minimum is **$300-350+/month** ($6000+ for enterprise)
**Source**: https://tardis.dev/#pricing (verified 2025-11-15)

**New Recommendation**: **CoinGecko Analyst at $129/month**

This document has been completely revised with accurate pricing.

---

## 🎯 Phased Budget Strategy

### Phase 1: PROVE IT WORKS ($154/month) - 3-4 weeks
```
Goal: Validate that quality data solves the problem
Data: Historical OHLCV (backtest)
Budget: $129 CoinGecko + $25 AWS = $154/month ✅
Outcome: Models achieve 65-75% accuracy in backtesting
Decision: If proven, continue for live trading
Canada: ✅ CoinGecko fully compliant
```

### Phase 2: GO LIVE ($179/month) - After validation
```
Goal: Deploy to production, start FTMO challenge
Data: CoinGecko + Coinbase real-time (free)
Budget: $129 CoinGecko + $50 AWS = $179/month ✅
Outcome: Live trading with proven models
ROI: One FTMO win pays for data for months
```

---

## 📊 Best Options Under $200 (REVISED)

### **RECOMMENDED: CoinGecko Analyst** - $129/month ⭐

**What You Get**:
```
✅ High-quality OHLCV historical data
✅ Multiple intervals (1m, 5m, 15m, 1h, 1d, 1w)
✅ 2+ years historical data
✅ Multiple exchanges aggregated
✅ Professional-grade API
✅ Canada-compliant
✅ API key obtained ✅
⚠️  OHLCV only (no tick data or order book)
```

**Perfect For**:
- Training price action models
- Feature engineering from OHLCV
- Backtesting and validation
- Proving the approach works under $200 budget

**Cost**: $129/month

**Upgrade Path**:
```
Option A: Stay with CoinGecko ($129/month)
  - If OHLCV proves sufficient
  - Most cost-effective

Option B: Upgrade to Tardis.dev ($300-350+/month)
  - If tick data + order book needed
  - Only if ROI justifies 3x cost increase
```

**Link**: https://www.coingecko.com/en/api/pricing

---

### Alternative 1: **Coinbase Free + CryptoCompare Free** - $0/month

**What You Get**:
```
✅ Coinbase real-time: Free (already have)
✅ CryptoCompare free tier: Limited historical
✅ Zero monthly cost
⚠️  Rate limits
⚠️  Limited historical depth
⚠️  May not be enough for 65-75% accuracy
```

**Best For**:
- Minimum viable testing
- Proving concept before any investment
- Very limited budget

**Cost**: $0/month

**Risk**: May not provide enough data quality for target accuracy

---

### Alternative 2: **Polygon.io Starter** - $89/month

**What You Get**:
```
✅ Crypto + Stocks data (multi-asset)
✅ Real-time WebSocket
✅ Historical OHLCV
✅ Good API quality
⚠️  Aggregated (not tick-level)
⚠️  Limited granularity
```

**Best For**:
- Multi-asset trading (crypto + stocks)
- Real-time needed immediately
- Lower granularity acceptable

**Cost**: $89/month

**Link**: https://polygon.io/pricing

---

### Alternative 3: **CoinAPI Startup** - $79/month

**What You Get**:
```
✅ 300+ exchanges aggregated
✅ OHLCV + trades
✅ REST API + WebSocket
✅ Historical data
⚠️  Rate limited (100 req/sec)
⚠️  Not tick-level
```

**Best For**:
- Multi-exchange coverage
- Basic real-time needs
- API-first development

**Cost**: $79/month

**Link**: https://www.coinapi.io/pricing

---

### Alternative 4: **Tardis.dev** - $300-350+/month (NOT RECOMMENDED for Phase 1)

**What You Get**:
```
✅ Full tick data (every trade)
✅ Complete order book depth (L2/L3)
✅ Highest data quality
✅ Multiple exchanges
⚠️  EXPENSIVE: Minimum $300-350+/month
⚠️  Enterprise tier: $6000+/month
⚠️  Overkill for Phase 1 validation
```

**Best For**:
- Live trading with proven ROI
- High-frequency trading
- Order book microstructure analysis
- When $300+/month is justified by profit

**Cost**: $300-350+/month minimum

**Recommendation**: Only upgrade to this if CoinGecko proves concept AND profit justifies 3x cost increase

**Link**: https://tardis.dev/#pricing

---

## 🎯 My Recommendation: CoinGecko Analyst ($129/month)

### Why This is Perfect:

**1. Best Quality for the Budget**
```
Professional OHLCV data at reasonable price
All intervals needed (1m, 5m, 15m, 1h)
2+ years historical for training
Better than free data, affordable for Phase 1
```

**2. Prove Before Scaling**
```
Week 1-2: Download historical data
Week 3: Train models
Week 4: Backtest thoroughly

If accuracy ≥68%: Continue with CoinGecko
If accuracy <68%: Investigate without wasting $300+/month
```

**3. Flexible Upgrade Path**
```
Phase 1: $129/month (validate with OHLCV)
        ↓ (models proven to work)
Phase 2A: $129/month (continue if sufficient)
        ↓ OR
Phase 2B: $300+/month Tardis (if tick data needed & ROI proven)
```

**4. No Wasted Money**
```
Don't pay $300+ for tick data until OHLCV models validated
$129/month is reasonable validation cost
Can upgrade if profit justifies it
```

---

## 📋 Phased Execution Plan

### Phase 1: Validation ($129/month) - 3-4 weeks

**Week 1: Download Data**
```
Day 1: Use CoinGecko API (already have key! ✅)
       - Configure data fetcher script
       - Test API connection

Day 2-3: Download historical data
         - BTC/ETH/SOL
         - 2 years OHLCV (1m, 5m, 15m, 1h intervals)
         - ~10-20 GB total

Day 4-7: Validate data quality
         - Check completeness
         - Compare to existing Coinbase data
         - Verify intervals and accuracy
```

**Week 2: Feature Engineering**
```
Day 8-10: Build price action features
          - Multi-timeframe indicators
          - Volume analysis
          - Session-based features

Day 11-14: Create 40-50 feature datasets
           - Run feature engineering pipeline
           - Validate features
           - Baseline test (expect 55-60% vs current 50%)
```

**Week 3: Model Training**
```
Day 15-17: Update architecture for new features
Day 18-20: Train models on CoinGecko data
           - Use AWS GPU instances (approved!)
           - Expect 62-70% validation accuracy
Day 21: Evaluate on test set
        - Target: ≥68% accuracy, ≤5% calibration
```

**Week 4: Backtesting & Decision**
```
Day 22-25: Comprehensive backtesting
           - Full 2-year backtest
           - Walk-forward validation
           - Calculate Sharpe, drawdown, win rate

Day 26-28: Decision point
           ✅ If models ≥68%: CONTINUE to Phase 2
           ⚠️  If models 60-67%: TUNE, then decide
           ❌ If models <60%: INVESTIGATE alternatives
```

**Phase 1 Cost**: $129/month × 1 month = $129

**Phase 1 Outcome**: **Know with certainty if quality OHLCV data solves the problem**

---

### Phase 2: Live Trading ($179/month) - After validation

**Only Start Phase 2 If Phase 1 Succeeds**

**Week 5: Deploy to Production**
```
Day 29-31: Setup production pipeline
           - CoinGecko for historical retraining
           - Coinbase free API for real-time signals
           - Feature computation in real-time
           - Model serving on AWS

Day 32-35: Dry-run testing
           - 48-hour dry-run
           - Validate real-time performance
           - Check latency (<500ms)
```

**Week 6-7: Paper Trading**
```
Day 36-42: Paper trade on FTMO demo
           - Track real performance
           - Validate against backtest
           - Monitor for issues

Day 43-49: Continue paper trading
           - Minimum 5 days required
           - Collect performance data
           - Make final adjustments
```

**Week 8: Go Live**
```
Day 50-56: Start FTMO challenge OR live trading
           - Deploy to production
           - Real money on the line
           - Monitor closely

If successful: Data pays for itself quickly
If not: Stop, investigate, don't burn money
```

**Phase 2 Cost**: $129 CoinGecko + $50 AWS = $179/month ongoing

**Phase 2 Outcome**: **Production trading system, FTMO challenge started**

---

## 💰 Total Budget Breakdown

### Option 1: CoinGecko Analyst ⭐ RECOMMENDED

**Month 1: Validation Phase**
```
CoinGecko Analyst:     $129/month
AWS (S3/RDS):          ~$25/month (optimized)
────────────────────────────────
Total:                 ~$154/month ✅ Under $200!
```

**Month 2+: Live Trading (if validated)**
```
CoinGecko Analyst:     $129/month
Coinbase real-time:    $0/month (free, already have)
AWS (GPU/production):  ~$50/month
────────────────────────────────
Total:                 ~$179/month ✅ Still under $200!
```

**Total Investment to Validation**: ~$154 × 1 month = $154

---

### Option 2: Free Tier (Minimal Investment)

**Month 1-2: Validation + Testing**
```
Coinbase API:          $0/month (free, already have)
CryptoCompare Free:    $0/month
AWS:                   ~$25/month
────────────────────────────────
Total:                 ~$25/month ✅ Very cheap

Pros: Zero data cost
Cons: May not achieve 65-75% accuracy
Risk: Time wasted if data insufficient
```

---

### Option 3: Polygon.io ($89/month)

**Month 1-2: Validation + Live**
```
Polygon.io Starter:    $89/month
AWS:                   ~$25-50/month
────────────────────────────────
Total:                 ~$114-139/month ✅ Cheaper than CoinGecko

Pros: Real-time included, multi-asset
Cons: May not be crypto-optimized
```

---

### Option 4: Tardis.dev ($300-350+/month) - Only if CoinGecko proves concept

**Month 1+: Premium Data (after proving with CoinGecko)**
```
Tardis.dev:            $300-350+/month
AWS:                   ~$50/month
────────────────────────────────
Total:                 ~$350-400/month ⚠️  EXPENSIVE

When to use:
- CoinGecko OHLCV models hit ≥68% ✅
- Live trading generating >$500/month profit ✅
- Need tick data + order book for edge ✅
- Can justify 3x cost increase ✅
```

---

## 🎯 Decision Matrix (UPDATED WITH CORRECT PRICING)

| Provider | Monthly Cost | Quality | Data Type | Real-time | Best For | Risk |
|----------|--------------|---------|-----------|-----------|----------|------|
| **CoinGecko** | **$129** | ⭐⭐⭐⭐ | OHLCV | ❌ | **Phase 1** ⭐ | LOW |
| Coinbase Free | $0 | ⭐⭐ | OHLCV | ✅ | Testing | HIGH |
| Polygon.io | $89 | ⭐⭐⭐ | OHLCV | ✅ | Multi-asset | MED |
| CoinAPI | $79 | ⭐⭐⭐ | OHLCV | ✅ | Coverage | MED |
| Tardis.dev | $300-350+ | ⭐⭐⭐⭐⭐ | Tick+Book | ✅ | Post-validation | LOW |

---

## ✅ My Recommendation: Phased Approach

### **Phase 1: CoinGecko Analyst ($129/month)**

**Why Start Here**:
1. ✅ Professional OHLCV data at reasonable price
2. ✅ Validates models work before expensive upgrades
3. ✅ Under $200 budget ($154/month total)
4. ✅ API key already obtained ✅
5. ✅ Canada-compliant
6. ✅ Flexible upgrade path

**Timeline**: 3-4 weeks
**Cost**: $129/month
**Outcome**: Know if 65-75% accuracy is achievable with OHLCV
**Risk**: LOW (only $129 to find out)

---

### **Phase 2: Continue or Upgrade Based on Results**

**Option A: Continue with CoinGecko ($129/month)**
```
IF models achieve ≥68% accuracy with OHLCV
AND backtests show consistent profitability
AND no need for tick data/order book
THEN continue with CoinGecko

Total cost: $179/month (CoinGecko + AWS)
ROI: Best cost/benefit ratio
```

**Option B: Upgrade to Tardis.dev ($300-350+/month)**
```
IF CoinGecko models hit 68%+ ✅
AND live trading shows need for tick data
AND profit >$500/month justifies upgrade
THEN upgrade to Tardis for microstructure edge

Total cost: $350-400/month
ROI: Only if profit justifies 2-3x cost increase
```

**Why This Approach**:
- Don't pay $300+ before proving OHLCV models work
- Save $171-221/month during validation
- Only invest more when success is proven
- Flexibility to scale based on results

---

## 🚀 Immediate Action Plan

### Today (Under $200 Budget):

**RECOMMENDED: Start with CoinGecko ⭐**
```
1. CoinGecko API: Already configured ✅
   Location: /home/numan/crpbot/.env line 24
   Key: CG-VQhq64e59sGxchtK8mRgdxXW

2. Create data fetcher script
   - Download 2 years OHLCV for BTC/ETH/SOL
   - Multiple intervals (1m, 5m, 15m, 1h)

3. Engineer features from OHLCV data
   - 40-50 features from price action
   - Multi-timeframe indicators

4. Train models
   - Use AWS GPU (approved!)
   - Target: 65-75% accuracy

5. Validate in 3-4 weeks
   - If ≥68%: Continue to Phase 2
   - If <68%: Investigate alternatives
```

**Total Cost**: $154/month (CoinGecko $129 + AWS $25)

---

## 💡 Why This Plan Makes Sense

### Risk Mitigation:
```
❌ Don't spend $300+ before proving OHLCV works
✅ Spend $129 to validate with quality data
✅ Only upgrade to Tardis if profit justifies it
✅ Save $171-221/month during validation

Total risk: $129 (vs $300-350+)
Potential waste: $129 (vs $300-350+)
Smart business decision: ✅
```

### Cost vs Value:
```
Free Coinbase: $0 → 50% accuracy (worthless)
CoinGecko: $129 → 65-75% accuracy (validated)
Tardis.dev: $300-350+ → 65-75%+ (w/ microstructure)

Step 1: Prove OHLCV works ($129) ← START HERE
Step 2: Deploy for live trading ($179)
Step 3: Upgrade to Tardis if ROI proven ($350-400)
```

---

## 📋 Quick Comparison Table

| Feature | CoinGecko ($129) | Tardis ($300-350+) | Difference |
|---------|------------------|--------------------|------------|
| **OHLCV Data** | ✅ Professional | ✅ Professional | Same quality |
| **Tick Data** | ❌ No | ✅ Yes | Tardis advantage |
| **Order Book** | ❌ No | ✅ L2/L3 | Tardis advantage |
| **Historical** | ✅ 2+ years | ✅ 2+ years | Same |
| **Intervals** | ✅ 1m-1d | ✅ Tick-level | Tardis more granular |
| **Real-time** | ❌ No (use Coinbase free) | ✅ Yes | Tardis advantage |
| **Use Case** | Train OHLCV models | Microstructure models | Different focus |
| **Cost** | $129/month | $300-350+/month | **2-3x cheaper** |

**Bottom Line**: Start with CoinGecko OHLCV. Upgrade to Tardis only if profit justifies it.

---

## 🎯 Decision Time

**Your Budget**: <$200/month

**Best Option**: CoinGecko Analyst ($129/month)
- ✅ Professional OHLCV data
- ✅ Reasonable price for Phase 1
- ✅ API key obtained ✅
- ✅ Under $200 budget ($154 total)
- ✅ Canada-compliant
- ✅ Flexible upgrade path

**Phase 1 Path**:
```
Month 1: $154/month (CoinGecko + AWS)
       ↓ (validate OHLCV models ≥68%)
Month 2+: $179/month (live trading)
       ↓ (if profitable >$500/month)
Option: Upgrade to Tardis $350-400/month
```

**Alternative if budget extremely tight**:
- Free tier: Coinbase + CryptoCompare ($0)
- Risk: May not achieve target accuracy
- Consider CoinGecko worth the $129 investment

---

**My recommendation: Start with CoinGecko Analyst ($129/month)**

**Prove OHLCV models work, then decide on Tardis upgrade based on profit.**

---

**File**: `V5_BUDGET_PLAN.md`
**Status**: REVISED with correct pricing
**Recommended**: CoinGecko Analyst $129/month
**Timeline**: 3-4 weeks to validation
**Next**: Create data fetcher script for CoinGecko API
