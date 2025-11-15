# 💰 V5 Budget Plan - Start Smart, Scale Fast

**Goal**: Build quant trading software with phased data budget
**Phase 1 Budget**: <$200/month (prove it works)
**Phase 2 Budget**: $200-500/month (scale for live trading)
**Approach**: Start small, validate, then scale

---

## 🎯 Phased Budget Strategy

### Phase 1: PROVE IT WORKS ($148/month) - 3-4 weeks
```
Goal: Validate that quality data solves the problem
Data: Historical only (backtest)
Budget: $98 Tardis + $50 AWS = $148/month (Canada-compliant)
Outcome: Models achieve 65-75% accuracy in backtesting
Decision: If proven, upgrade to real-time for live trading
Note: 2 exchanges only (Binance excluded - banned in Canada)
```

### Phase 2: GO LIVE ($500/month) - After validation
```
Goal: Deploy to production, start FTMO challenge
Data: Historical + Real-time
Budget: $499/month
Outcome: Live trading with proven models
ROI: One FTMO win pays for data for months
```

---

## 📊 Best Options Under $200

### **RECOMMENDED: Tardis.dev Historical** - $98/month ⭐ (Canada-compliant)

**What You Get**:
```
✅ Full tick data (every single trade)
✅ Complete order book depth (L2/L3)
✅ 2 exchanges: $49/month × 2 = $98/month (Canada-compliant)
   - ❌ ~~Binance~~ (BANNED IN CANADA - excluded)
   - Coinbase Pro (BTC/ETH/SOL)
   - Kraken (BTC/ETH/SOL)
✅ 2+ years historical data
✅ Same quality as Premium ($499)
❌ No real-time (historical only)
```

**Perfect For**:
- Backtesting and validation
- Training models
- Proving the approach works
- Building features offline

**Cost**: $98/month (2 exchanges, Canada-compliant, best quality)

**Upgrade Path**:
```
When ready for live trading:
$98/month → $499/month (adds real-time)
Total increase: +$401/month
```

**Link**: https://tardis.dev/pricing

---

### Alternative 1: **Polygon.io Starter** - $89/month

**What You Get**:
```
✅ Crypto + Stocks data (multi-asset)
✅ Real-time WebSocket
✅ Historical data
✅ Good API quality
⚠️  Aggregated (not tick-level)
⚠️  Not as granular as Tardis
```

**Best For**:
- Multi-asset trading (crypto + stocks)
- Real-time needed immediately
- Lower granularity acceptable

**Cost**: $89/month

**Link**: https://polygon.io/pricing

---

### Alternative 2: **CoinAPI Startup** - $79/month

**What You Get**:
```
✅ 300+ exchanges aggregated
✅ OHLCV + trades + order book
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

### Alternative 3: **CryptoCompare** - $150/month

**What You Get**:
```
✅ Aggregated crypto data
✅ Multiple exchanges
✅ Historical OHLCV
✅ Real-time WebSocket
⚠️  Aggregated, not raw ticks
⚠️  No order book depth
```

**Best For**:
- Retail-level quality
- Multiple coins
- Real-time included

**Cost**: $150/month (starter tier)

**Link**: https://www.cryptocompare.com/pricing

---

## 🎯 My Recommendation: Tardis.dev Historical ($98/month - Canada-compliant)

### Why This is Perfect:

**1. Best Quality for the Price**
```
Same data quality as $499/month Premium
Only difference: No real-time (we don't need it yet!)
Perfect for backtesting and training
```

**2. Prove Before Scaling**
```
Week 1-2: Download historical data
Week 3: Train models
Week 4: Backtest thoroughly

If accuracy ≥65%: Proven → Upgrade to real-time
If accuracy <65%: Investigate without wasting $500/month
```

**3. Clean Upgrade Path**
```
Phase 1: $98/month (historical, 2 exchanges)
        ↓ (models proven to work)
Phase 2: $499/month (+ real-time)
        ↓ (FTMO challenge started)
Phase 3: ROI positive (data pays for itself)
```

**4. No Wasted Money**
```
Don't pay for real-time until models are validated
$98/month buys same quality as $499 (just historical)
Save $401/month during validation phase
```

---

## 📋 Phased Execution Plan

### Phase 1: Validation ($98/month) - 3-4 weeks

**Week 1: Subscribe & Download**
```
Day 1: Subscribe to Tardis.dev Historical
       - 2 exchanges × $49 = $98/month (Canada-compliant)
       - ❌ ~~Binance~~ (BANNED IN CANADA - excluded)
       - Select: Coinbase (BTC/ETH/SOL)
       - Select: Kraken (BTC/ETH/SOL)

Day 2-3: Download historical data
         - 2 years tick data
         - Order book snapshots
         - ~50-100 GB per symbol

Day 4-7: Validate data quality
         - Check completeness
         - Compare to Coinbase (expect huge improvement)
```

**Week 2: Feature Engineering**
```
Day 8-10: Build microstructure features
          - Order book imbalance
          - Trade flow (buy/sell pressure)
          - VWAP, spread dynamics

Day 11-14: Create 53-feature datasets
           - Run feature engineering
           - Validate features
           - Baseline test (expect 55-60% vs 50%)
```

**Week 3: Model Training**
```
Day 15-17: Update architecture (input_size: 33→53)
Day 18-20: Train models on Tardis data
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
           ✅ If models ≥68%: UPGRADE to real-time
           ⚠️  If models 60-67%: TUNE, then upgrade
           ❌ If models <60%: INVESTIGATE
```

**Phase 1 Cost**: $98/month × 1 month = $98 (Canada-compliant)

**Phase 1 Outcome**: **Know with certainty if quality data solves the problem**

---

### Phase 2: Live Trading ($499/month) - After validation

**Only Start Phase 2 If Phase 1 Succeeds**

**Week 5: Upgrade to Real-Time**
```
Day 29: Upgrade Tardis.dev: Historical → Premium
        Cost increase: +$401/month
        New total: $499/month

Day 30-31: Setup real-time pipeline
           - WebSocket integration
           - Feature computation in real-time
           - Model serving

Day 32-35: Dry-run testing
           - 48-hour dry-run
           - Validate real-time performance
           - Check latency (<1 second)
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
```

**Phase 2 Cost**: $499/month ongoing

**Phase 2 Outcome**: **Production trading system, FTMO challenge started**

---

## 💰 Total Budget Breakdown

### Option 1: Start with Historical ($98/month) ⭐ RECOMMENDED (Canada-compliant)

**Month 1: Validation Phase**
```
Tardis.dev Historical: $98/month (Coinbase + Kraken, no Binance)
AWS (EC2/RDS/S3):      ~$50/month
────────────────────────────────
Total:                 ~$148/month ✅ Under $200!
```

**Month 2+: If Validated, Upgrade to Live**
```
Tardis.dev Premium:    $499/month
AWS:                   ~$50/month
────────────────────────────────
Total:                 ~$549/month
```

**Total Investment to Validation**: ~$148 × 1 month = $148

---

### Option 2: Start with Polygon.io ($89/month)

**Month 1-2: Validation + Live**
```
Polygon.io Starter:    $89/month
AWS:                   ~$50/month
────────────────────────────────
Total:                 ~$139/month ✅ Cheapest

Pros: Real-time included, multi-asset
Cons: Not tick-level, may not be enough quality
```

---

### Option 3: Start with CoinAPI ($79/month)

**Month 1-2: Validation + Live**
```
CoinAPI Startup:       $79/month
AWS:                   ~$50/month
────────────────────────────────
Total:                 ~$129/month ✅ Very cheap

Pros: 300+ exchanges, real-time
Cons: Rate limited, aggregated data
```

---

## 🎯 Decision Matrix

| Provider | Monthly Cost | Quality | Real-time | Best For | Risk |
|----------|-------------|---------|-----------|----------|------|
| **Tardis Historical** | **$98** | ⭐⭐⭐⭐⭐ | ❌ | Validation | LOW ⭐ |
| Tardis Premium | $499 | ⭐⭐⭐⭐⭐ | ✅ | Live trading | LOW |
| Polygon.io | $89 | ⭐⭐⭐ | ✅ | Multi-asset | MEDIUM |
| CoinAPI | $79 | ⭐⭐⭐ | ✅ | Coverage | MEDIUM |
| CryptoCompare | $150 | ⭐⭐⭐⭐ | ✅ | Retail trading | MEDIUM |

---

## ✅ My Recommendation: 2-Phase Approach

### **Phase 1: Tardis.dev Historical ($98/month)** (Canada-compliant)

**Why Start Here**:
1. ✅ Same quality as $499 Premium
2. ✅ Validates models work before live trading
3. ✅ Under $200 budget
4. ✅ No wasted money on real-time you can't use yet
5. ✅ Canada-compliant (Coinbase + Kraken, no Binance)
5. ✅ Clean upgrade path when ready

**Timeline**: 3-4 weeks
**Cost**: $98/month
**Outcome**: Know if 65-75% accuracy is achievable
**Risk**: LOW (only $98 to find out)

---

### **Phase 2: Upgrade to Premium ($499/month)**

**When to Upgrade**:
```
IF models achieve ≥68% accuracy in backtesting
AND backtest Sharpe ratio >1.0
AND ready to start paper trading
THEN upgrade to real-time
```

**Why Wait**:
- Don't pay for real-time before models proven
- Save $352/month during validation
- Only invest more when success is likely

**Timeline**: After Phase 1 validation
**Cost**: $499/month
**Outcome**: Live trading system
**Risk**: LOW (models already validated)

---

## 🚀 Immediate Action Plan

### Today (Under $200 Budget):

**Option A: Start Smart** ⭐ RECOMMENDED (Canada-compliant)
```
1. Subscribe: Tardis.dev Historical
   - ❌ ~~Binance~~ (BANNED IN CANADA - excluded)
   - Coinbase (BTC/ETH/SOL): $49/month
   - Kraken (BTC/ETH/SOL): $49/month
   Total: $98/month

2. Get API credentials

3. Start downloading historical data

4. Validate models in 3-4 weeks

5. Upgrade to real-time if validated
```

**Option B: Cheaper with Caveats**
```
1. Subscribe: Polygon.io Starter ($89/month)
   OR CoinAPI Startup ($79/month)

2. Test with aggregated data

3. May still not be enough quality

4. Might need to upgrade to Tardis anyway
```

---

## 💡 Why This Plan Makes Sense

### Risk Mitigation:
```
❌ Don't spend $500/month before proving it works
✅ Spend $98/month to validate
✅ Only upgrade when success is proven
✅ Save $401/month during validation

Total risk: $98 (vs $500)
Potential waste: $98 (vs $500)
Smart business decision: ✅
```

### Cost vs Value:
```
Free Coinbase: $0 → 50% accuracy (worthless)
Tardis Historical: $98 → 65-75% accuracy (validated)
Tardis Premium: $499 → 65-75% + live trading

Step 1: Prove quality data fixes problem ($98)
Step 2: Deploy for real ($499)
Step 3: ROI positive (FTMO wins)
```

---

## 📋 Quick Comparison: Tardis Historical vs Premium

| Feature | Historical ($98) | Premium ($499) | Difference |
|---------|------------------|----------------|------------|
| **Tick Data** | ✅ Full | ✅ Full | Same |
| **Order Book** | ✅ L2/L3 | ✅ L2/L3 | Same |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Same |
| **Exchanges** | ✅ 2 (CA-compliant) | ✅ 30+ | Limited |
| **Historical** | ✅ 2+ years | ✅ 2+ years | Same |
| **Real-time** | ❌ No | ✅ Yes | **Main difference** |
| **Use Case** | Backtest, train | Live trading | - |
| **Cost** | $98/month | $499/month | +$401 |

**Bottom Line**: Same quality data, only missing real-time. Perfect for validation phase.

---

## 🎯 Decision Time

**Your Budget**: <$200/month

**Best Option**: Tardis.dev Historical ($98/month) - Canada-compliant
- ✅ Best quality under $200
- ✅ Validates approach before live trading
- ✅ Clean upgrade path to real-time
- ✅ No wasted money
- ✅ Canada-compliant (no Binance)

**Upgrade Path**:
```
Month 1: $98/month (validate)
       ↓ (models proven ≥68%)
Month 2+: $499/month (go live)
       ↓ (FTMO challenge)
ROI: Positive (data pays for itself)
```

**Alternative if $98 too high**:
- Polygon.io ($89) or CoinAPI ($79)
- Lower quality, but real-time included
- May still not solve 50% accuracy problem
- Riskier bet

---

**My vote: Start with Tardis.dev Historical ($98/month)**

**Prove it works, then scale to Premium when ready to trade.**

---

**File**: `V5_BUDGET_PLAN.md`
**Status**: Ready for decision
**Recommended**: Tardis.dev Historical $98/month (Canada-compliant)
**Timeline**: 3-4 weeks to validation
**Next**: Your approval to subscribe
