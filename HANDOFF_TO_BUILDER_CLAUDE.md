# 🔄 Handoff to Builder Claude - V5 Ready to Start

**Created**: 2025-11-15 15:05 EST (Toronto)
**Author**: QC Claude (Local Machine)
**Status**: COMPLETE - Ready for Builder Claude
**Priority**: HIGH

---

## ✅ What's Complete (QC Claude)

### 1. CoinGecko API Setup ✅
- API Key obtained: `CG-VQhq64e59sGxchtK8mRgdxXW`
- Configured in local `.env`
- Config system updated (`libs/config/config.py`)
- `.env.example` documented

### 2. Pricing Error Corrected ✅
- **ERROR**: All V5 docs stated Tardis.dev at $98/month
- **REALITY**: Tardis.dev is $300-350+/month minimum
- **SOLUTION**: Switched to CoinGecko Analyst at $129/month
- **IMPACT**: Budget improved! Phase 1: $154/mo (vs old $148)

### 3. Documentation Updated ✅
All V5 plans corrected with accurate pricing:
- `V5_SIMPLE_PLAN.md` - Complete rewrite
- `V5_BUDGET_PLAN.md` - Complete rewrite
- `DATA_STRATEGY_COMPLETE.md` - All references updated
- `PRICING_CORRECTION_2025-11-15.md` - Error documentation

### 4. Builder Instructions Created ✅
- `BUILDER_CLAUDE_INSTRUCTIONS_2025-11-15.md` - **READ THIS FIRST**
- Complete 4-week timeline
- Step-by-step tasks for each week
- Technical requirements and code examples
- Budget tracking and success criteria

### 5. Git Commits ✅
```
Commit 1: 8187b84 - fix: correct V5 data provider pricing
Commit 2: 7c5e71e - docs: add Builder Claude V5 execution instructions
```

---

## 📋 What Builder Claude Needs to Do

### Immediate (Week 1 - Next 7 Days)

**Priority 1: Sync & Configure** (30 min)
```bash
cd ~/crpbot
git pull origin main  # Get latest changes
echo 'COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW' >> .env
```

**Priority 2: Create Data Fetcher** (2-3 hours)
- File: `scripts/fetch_coingecko_data.py`
- Download OHLCV data from CoinGecko API
- Handle rate limiting (50 calls/min)
- Save to parquet format

**Priority 3: Download Data** (1-2 hours)
- BTC-USD: 2 years of 1-minute OHLCV
- ETH-USD: 2 years of 1-minute OHLCV
- SOL-USD: 2 years of 1-minute OHLCV
- Total: ~100-150 MB compressed

**Priority 4: Validate Quality** (1 hour)
- Check for gaps
- Compare to existing Coinbase data
- Document improvements

---

## 🎯 4-Week Overview

| Week | Focus | Key Deliverable | Time Est. |
|------|-------|-----------------|-----------|
| **1** | Data download | 3 parquet files (BTC/ETH/SOL) | 6-8 hrs |
| **2** | Feature engineering | 40-50 features per symbol | 8-10 hrs |
| **3** | Model training | 4 trained models (AWS GPU) | 6-8 hrs |
| **4** | Validation | GO/NO-GO decision report | 6-8 hrs |

**Total Estimated Effort**: 26-34 hours over 4 weeks

---

## 💰 Budget Status

### Phase 1 (Current)
```
CoinGecko Analyst:     $129/month  ✅ Subscribed
AWS S3/RDS:            ~$25/month  ✅ Running
AWS GPU (training):    ~$5-10      ⏸️  Week 3
───────────────────────────────────────────────
Total Phase 1:         ~$159-164   ✅ Under $200
```

### Phase 2 (If Phase 1 succeeds)
```
Option A - Conservative:
  CoinGecko + AWS:     $179/month  ✅ Recommended

Option B - Premium:
  Tardis + AWS:        $350-400/month
  (Only if ROI >$500/month proven)
```

---

## 📊 Success Criteria (Week 4 Decision)

### ✅ GO TO PHASE 2 IF:
- Test accuracy ≥68%
- Calibration error ≤5%
- Sharpe ratio >1.0
- Max drawdown <15%
- Win rate >60%

### ⚠️ TUNE & RETRY IF:
- Test accuracy 60-67%
- Minor issues with calibration

### ❌ INVESTIGATE IF:
- Test accuracy <60%
- Severe overfitting
- Poor data quality

---

## 🔗 Key Documents for Builder Claude

### Must Read (Priority Order)
1. **BUILDER_CLAUDE_INSTRUCTIONS_2025-11-15.md** ← START HERE
2. **V5_SIMPLE_PLAN.md** - High-level strategy
3. **PRICING_CORRECTION_2025-11-15.md** - What changed and why
4. **V5_BUDGET_PLAN.md** - Detailed budget breakdown

### Reference Documents
- `DATA_STRATEGY_COMPLETE.md` - Complete data strategy
- `CLAUDE.md` - Codebase overview (if updated)
- `PROJECT_MEMORY.md` - Project history (if updated)

### CoinGecko Resources
- API Docs: https://docs.coingecko.com/reference/introduction
- Pricing: https://www.coingecko.com/en/api/pricing
- API Key Dashboard: https://www.coingecko.com/account/api

---

## 🚀 Quick Start Commands for Builder Claude

```bash
# 1. Sync with latest changes
cd ~/crpbot
git pull origin main
git log --oneline -5  # Verify you have latest commits

# 2. Read the instructions
cat BUILDER_CLAUDE_INSTRUCTIONS_2025-11-15.md

# 3. Configure API key
echo 'COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW' >> .env

# 4. Verify configuration
grep COINGECKO .env
python -c "from libs.config.config import load_settings; s = load_settings(); print(f'CoinGecko: {s.coingecko_api_key[:10]}...')"

# 5. Start Week 1!
# Create scripts/fetch_coingecko_data.py (see instructions for template)
```

---

## 📝 Communication Protocol

### Daily Updates
Builder Claude should create progress files:
- `WEEK1_DAY1_PROGRESS.md` - What was done, what's next, any blockers
- `WEEK1_DAY2_PROGRESS.md` - etc.

### Questions
If Builder Claude has questions:
1. Create `QUESTION_<topic>_2025-11-XX.md`
2. Tag QC Claude for review
3. Continue with other tasks while waiting

### Blockers
If blocked on something critical:
1. Create `BLOCKER_<issue>_2025-11-XX.md`
2. Document the issue and attempted solutions
3. Flag for immediate QC Claude review

---

## ✅ Checklist Before Starting

Builder Claude should verify:
- [ ] Git is synced (commit `7c5e71e` present)
- [ ] `BUILDER_CLAUDE_INSTRUCTIONS_2025-11-15.md` exists
- [ ] CoinGecko API key added to `.env`
- [ ] Can import and load settings successfully
- [ ] Understands the 4-week timeline
- [ ] Knows success criteria for Week 4

---

## 🎯 Expected Week 1 Output

By end of Week 1, Builder Claude should have:

### Files Created
```
scripts/
  fetch_coingecko_data.py         # CoinGecko API fetcher
  validate_coingecko_data.py      # Data quality validator

data/raw/coingecko/
  BTC-USD_1m_20251115.parquet     # ~30-50 MB
  ETH-USD_1m_20251115.parquet     # ~30-50 MB
  SOL-USD_1m_20251115.parquet     # ~20-40 MB

WEEK1_PROGRESS_SUMMARY.md         # Week 1 results
```

### Deliverables
- ✅ 2 years OHLCV data for BTC/ETH/SOL
- ✅ Data quality validation report
- ✅ Comparison: CoinGecko vs Coinbase quality
- ✅ Week 1 summary with findings

---

## 💡 Tips for Builder Claude

### CoinGecko API Tips
- Rate limit: 50 calls/min (Analyst tier)
- Add `time.sleep(1.2)` between calls
- Use exponential backoff for errors
- Cache responses to avoid re-fetching

### Data Quality Tips
- CoinGecko aggregates multiple exchanges
- Expect better quality than free Coinbase
- Some gaps normal for low-liquidity periods
- Forward-fill small gaps (<5 min)

### AWS GPU Tips
- Wait until Week 3 for GPU training
- Use `g4dn.xlarge` (cheapest, sufficient)
- Cost: ~$0.526/hour = ~$2-3 for all training
- Spot instances can save 70%

---

## 🔄 Next Handoff (After Week 4)

After Week 4 validation, Builder Claude should:
1. Create `V5_PHASE1_RESULTS_2025-11-XX.md`
2. Document all metrics and decision
3. Hand off to QC Claude for review
4. Wait for user approval before Phase 2

---

## 📞 QC Claude Availability

QC Claude (local machine) is available for:
- Questions about requirements
- Code reviews
- Decision approval
- Blocker resolution

**How to reach**: Create markdown file with question/issue and commit to Git

---

## 🎉 Summary

**Status**: Everything is ready for Builder Claude to start V5 Week 1

**What's Different**:
- Using CoinGecko ($129) instead of Tardis ($300+)
- 40-50 features (OHLCV-based) instead of 53 (microstructure)
- Budget improved: $154/mo Phase 1, $179/mo Phase 2

**Next Step**: Builder Claude reads `BUILDER_CLAUDE_INSTRUCTIONS_2025-11-15.md` and starts Week 1

**Timeline**: 4 weeks to validation decision

**Confidence**: HIGH - Clear plan, API configured, budget approved ✅

---

**File**: `HANDOFF_TO_BUILDER_CLAUDE.md`
**Created**: 2025-11-15 15:05 EST
**Status**: COMPLETE
**Next**: Builder Claude starts Week 1

**Let's build V5! 🚀**
