# ⚠️ CRITICAL PRICING CORRECTION - November 15, 2025

**Created**: 2025-11-15 14:55 EST (Toronto)
**Author**: QC Claude
**Status**: URGENT - All V5 Documentation Requires Update

---

## 🚨 CRITICAL ERROR DISCOVERED

**Error**: All V5 documentation stated Tardis.dev Historical at $98/month
**Reality**: Tardis.dev minimum is **$300-350+/month** ($6000+ for enterprise)
**Source**: https://tardis.dev/#pricing (verified 2025-11-15)

### Impact

This pricing error affects ALL V5 documentation created on 2025-11-15:
- V5_PHASE1_PLAN.md
- V5_FEATURES_REFERENCE.md
- V5_SIMPLE_PLAN.md
- V5_BUDGET_PLAN.md
- DATA_STRATEGY_COMPLETE.md
- BUILDER_CLAUDE_SUMMARY_2025-11-15.md
- PROJECT_MEMORY.md (if updated with $98 pricing)
- CLAUDE.md (if updated with $98 pricing)

---

## ✅ CORRECTED STRATEGY

### Phase 1: CoinGecko Analyst ($129/month)

**What You Get**:
- High-quality OHLCV historical data
- Multiple intervals (1m, 5m, 15m, 1h, 1d)
- 2+ years historical
- Multiple exchanges aggregated
- Canada-compliant
- API key obtained: CG-VQhq64e59sGxchtK8mRgdxXW ✅

**Budget Phase 1**:
```
CoinGecko Analyst:  $129/month
AWS (S3/RDS):       ~$25/month
─────────────────────────────
Total:              $154/month ✅ (Under $200 budget)
```

### Phase 2: Two Options

**Option A - Conservative (Recommended)**:
```
CoinGecko Analyst:  $129/month
AWS (production):   ~$50/month
─────────────────────────────
Total:              $179/month ✅
```

**Option B - Premium (Only if ROI proven)**:
```
Tardis.dev:         $300-350+/month
AWS (production):   ~$50/month
─────────────────────────────
Total:              $350-400+/month
Only upgrade if:
- CoinGecko models hit ≥68% accuracy ✅
- Live trading profit >$500/month ✅
- Tick data + order book needed for edge ✅
```

---

## 📊 CORRECTED Budget Summary

| Phase | OLD (Error) | NEW (Correct) | Difference |
|-------|-------------|---------------|------------|
| Phase 1 | $148/mo | $154/mo | +$6 |
| Phase 2 Option A | $549/mo | $179/mo | -$370 ✅ |
| Phase 2 Option B | $549/mo | $350-400/mo | -$149 to -$199 |

---

## 🔄 Documents Updated (2025-11-15 14:45-14:55 EST)

### ✅ CORRECTED:
1. V5_SIMPLE_PLAN.md - Complete rewrite with CoinGecko
2. V5_BUDGET_PLAN.md - Complete rewrite with CoinGecko
3. DATA_STRATEGY_COMPLETE.md - All Tardis $98 → CoinGecko $129
4. .env - Added COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW
5. libs/config/config.py - Added coingecko_api_key field

### ⏸️ NEEDS CORRECTION:
6. BUILDER_CLAUDE_SUMMARY_2025-11-15.md - Contains $98 Tardis references
7. V5_PHASE1_PLAN.md - Contains $98 Tardis references
8. V5_FEATURES_REFERENCE.md - May reference 53 features (should be 40-50 for OHLCV)
9. PROJECT_MEMORY.md - If updated with $98 pricing
10. CLAUDE.md - If updated with $98 pricing

---

## 🎯 Corrected V5 Strategy Summary

### Data Provider: CoinGecko Analyst
- **Cost**: $129/month
- **Data**: OHLCV historical (professional grade)
- **Intervals**: 1m, 5m, 15m, 1h, 1d, 1w
- **History**: 2+ years
- **Status**: API key configured ✅

### Features: 40-50 (not 53)
- **Existing**: 31 features from V4 (OHLCV-based)
- **New**: 9-19 additional OHLCV-derived features
- **NOT INCLUDED**: Microstructure features (tick data/order book)
  - These require Tardis.dev ($300-350+/month)
  - Only add if Phase 1 proves concept AND ROI justifies cost

### Timeline: 4 weeks
- Week 1: Download CoinGecko OHLCV data
- Week 2: Engineer 40-50 features
- Week 3: Train models on CoinGecko data
- Week 4: Validate (target: ≥68% accuracy)

### Budget:
- **Phase 1**: $154/month (validation)
- **Phase 2A**: $179/month (if OHLCV sufficient)
- **Phase 2B**: $350-400/month (if upgrade to Tardis justified)

---

## 🚀 Current Status

✅ **CoinGecko API configured** - Ready to start Week 1
✅ **Budget approved** - $154/month Phase 1
✅ **Documentation corrected** - V5_SIMPLE_PLAN.md, V5_BUDGET_PLAN.md, DATA_STRATEGY_COMPLETE.md

⏸️ **Remaining work**: Update Builder Claude documents with corrected pricing

---

## 📝 Next Actions

1. Update BUILDER_CLAUDE_SUMMARY_2025-11-15.md with pricing correction
2. Review V5_PHASE1_PLAN.md and V5_FEATURES_REFERENCE.md
3. Update PROJECT_MEMORY.md if it contains $98 Tardis references
4. Update CLAUDE.md if it contains $98 Tardis references
5. Create CoinGecko data fetcher script
6. Commit all corrected documentation to Git

---

**File**: `PRICING_CORRECTION_2025-11-15.md`
**Purpose**: Document the critical pricing error and correction strategy
**Status**: Reference for updating remaining documents
